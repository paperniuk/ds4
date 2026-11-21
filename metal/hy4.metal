// Native Hy4 Flash text kernels. The GGUF weights use the same block formats
// as Qwen; reuse its SIMD-group dot products and the DS4 projection dispatch.
struct hy4_args {
    uint rows, pos, width, heads, capacity, groups, type, row_bytes;
    float theta;
};

static inline float hy4_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static inline float2 hy4_rotate(float2 v, uint pair, uint pos, float theta) {
    const float angle = float(pos) * pow(theta, -float(pair) / 32.0f);
    const float c = cos(angle), s = sin(angle);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

kernel void kernel_hy4_repeat(constant hy4_args &a,
                             device const float *in, device float *streams,
                             uint2 id [[thread_position_in_grid]]) {
    if (id.x < 12288) streams[id.y * 12288 + id.x] = in[id.y * 3072 + id.x % 3072];
}

kernel void kernel_hy4_hc_mix(constant hy4_args &a,
                             device const float *streams, device const float *mix,
                             device const float *scale, device const float *base,
                             device float *out, device float *post,
                             uint2 id [[thread_position_in_grid]]) {
    const uint d = id.x, row = id.y;
    if (d >= 3072) return;
    float sum = 0;
    for (uint h = 0; h < 4; h++) {
        const float pre = hy4_sigmoid(mix[row * a.width + h] * scale[0] + base[h]) + 1e-6f;
        sum += pre * streams[row * 12288 + h * 3072 + d];
        if (a.width == 8 && d == 0)
            post[row * 4 + h] = 2.0f * hy4_sigmoid(mix[row * 8 + 4 + h] * scale[1] + base[4 + h]) + 1e-6f;
    }
    out[row * 3072 + d] = sum;
}

kernel void kernel_hy4_hc_add(constant hy4_args &a,
                             device float *streams, device const float *branch,
                             device const float *post, uint2 id [[thread_position_in_grid]]) {
    if (id.x >= 12288) return;
    streams[id.y * 12288 + id.x] += post[id.y * 4 + id.x / 3072] * branch[id.y * 3072 + id.x % 3072];
}

// Split the flattened iHC projection across the GPU while accumulating its
// RMS denominator. The next kernel applies one denominator to every dot.
kernel void kernel_hy4_hc_partial(constant hy4_args &a,
                                  device const float4 *streams, device const float4 *weights,
                                  device float *partial, uint2 id [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint output [[simdgroup_index_in_threadgroup]]) {
    const uint block = 3072 / a.groups;
    float4 dot4 = 0, sq4 = 0;
    for (uint i = lane; i < block; i += 32) {
        const uint offset = id.x * block + i;
        const float4 x = streams[id.y * 3072 + offset];
        dot4 = fma(x, weights[output * 3072 + offset], dot4);
        if (output == 0) sq4 = fma(x, x, sq4);
    }
    const float result = simd_sum(dot(dot4, float4(1)));
    const float sq = simd_sum(dot(sq4, float4(1)));
    if (lane == 0) {
        device float *dst = partial + (id.y * a.groups + id.x) * 9;
        dst[output] = result;
        if (output == 0) dst[8] = sq;
    }
}

kernel void kernel_hy4_hc_finish(constant hy4_args &a,
                                 device const float *streams, device const float *partial,
                                 device const float *scale, device const float *base,
                                 device float *out, device float *post,
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float pre[4];
    if (lane < 4) {
        float sq = 0, p = 0, q = 0;
        for (uint b = 0; b < a.groups; b++) {
            device const float *part = partial + (group.y * a.groups + b) * 9;
            sq += part[8];
            p += part[lane];
            if (a.width == 8) q += part[lane + 4];
        }
        const float inv = rsqrt(sq / 12288.0f + 1e-5f);
        pre[lane] = hy4_sigmoid(p * inv * scale[0] + base[lane]) + 1e-6f;
        if (a.width == 8 && group.x == 0)
            post[group.y * 4 + lane] = 2.0f * hy4_sigmoid(q * inv * scale[1] + base[lane + 4]) + 1e-6f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = group.x * 256 + lane;
    float sum = 0;
    for (uint h = 0; h < 4; h++) sum += pre[h] * streams[group.y * 12288 + h * 3072 + d];
    out[group.y * 3072 + d] = sum;
}

kernel void kernel_hy4_engram(constant hy4_args &a,
                             device float *streams, device const float *kv,
                             device const float *bias, device const float *gate,
                             device const float *gamma, device const float *hc_scale,
                             device const int *valid, uint2 id [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    if (!valid[id.y]) return;
    device float *x = streams + id.y * 12288 + id.x * 3072;
    device const float *k = kv + id.y * 15360 + id.x * 3072;
    device const float *v = kv + id.y * 15360 + 12288;
    float x2 = 0, k2 = 0, v2 = 0, dot = 0;
    for (uint d = lane; d < 3072; d += 32) {
        const float kd = k[d] + bias[id.x * 3072 + d], vd = v[d] + bias[12288 + d];
        x2 += x[d] * x[d];
        k2 += kd * kd;
        v2 += vd * vd;
        dot += x[d] * kd * gate[id.x * 3072 + d];
    }
    const float eps = 1.1920929e-7f;
    const float s = simd_sum(dot) * rsqrt(3072.0f)
                  * rsqrt(simd_sum(x2) / 3072.0f + eps)
                  * rsqrt(simd_sum(k2) / 3072.0f + eps);
    const float g = hy4_sigmoid(copysign(sqrt(max(abs(s), 1e-6f)), s));
    const float norm = rsqrt(simd_sum(v2) / 3072.0f + eps);
    for (uint d = lane; d < 3072; d += 32)
        x[d] += (v[d] + bias[12288 + d]) * norm * gamma[d] * (g + hc_scale[id.x * 3072 + d]);
}

kernel void kernel_hy4_rope(constant hy4_args &a, device float *x,
                           uint2 id [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    device float2 *p = (device float2 *)(x + (id.y * a.heads + id.x) * a.width + a.width - 64);
    p[lane] = hy4_rotate(p[lane], lane, a.pos + id.y, a.theta);
}

kernel void kernel_hy4_store(constant hy4_args &a,
                            device const float *in, device const float *weight,
                            device half *cache, uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    device const float *x = in + row * 576;
    device half *out = cache + ulong((row + a.pos) % a.capacity) * 576;
    float sum = 0;
    for (uint d = lane; d < 512; d += 32) sum += x[d] * x[d];
    const float inv = rsqrt(simd_sum(sum) / 512.0f + 1e-5f);
    for (uint d = lane; d < 512; d += 32) out[d] = half(x[d] * weight[d] * inv);
    const float2 v = hy4_rotate(float2(x[512 + 2*lane], x[513 + 2*lane]), lane, row + a.pos, a.theta);
    out[512 + 2*lane] = half(v.x);
    out[513 + 2*lane] = half(v.y);
}

// Reuse the DS4 quant row dot for MLA's two per-head projections.
kernel void kernel_hy4_head_mm(constant hy4_args &a,
                              device const float *in, device const char *weight, device float *out,
                              uint3 id [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint out_dim = a.width == 192 ? 512 : 256;
    const uint stride = a.width == 192 ? 256 : 512;
    const uint d = id.x * 4 + sg;
    const uint head = id.y, row = id.z;
    const ulong offset = (ulong(head) * (a.groups ? a.groups : out_dim) + d) * a.row_bytes;
    const float value = qwen4_row_dot(weight + offset, in + (row * a.heads + head) * stride,
                                      a.type, a.width, lane);
    if (lane == 0) out[(row * a.heads + head) * out_dim + d] = value;
}

kernel void kernel_hy4_pool_stage(constant hy4_args &a,
                                 device const float *value, device const float *gate,
                                 device float *ring, uint2 id [[thread_position_in_grid]]) {
    if (id.x < 128) {
        device float *slot = ring + ((a.pos + id.y) % a.capacity) * 256;
        slot[id.x] = value[id.y * 128 + id.x];
        slot[128 + id.x] = gate[id.y * 128 + id.x];
    }
}

kernel void kernel_hy4_pool(constant hy4_args &a,
                           device const float *ring, device const float *ape,
                           device const float *norm, device half *pool,
                           uint block [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint group = a.pos / 4 + block;
    float p[4];
    float sum2 = 0;
    for (uint i = 0; i < 4; i++) {
        const uint d = lane + 32 * i;
        float v[4], g[4];
        float best = -INFINITY;
        for (uint t = 0; t < 4; t++) {
            device const float *slot = ring + ((group * 4 + t) % a.capacity) * 256;
            v[t] = slot[d];
            g[t] = slot[128 + d] + ape[t * 128 + d];
            best = max(best, g[t]);
        }
        float den = 0, num = 0;
        for (uint t = 0; t < 4; t++) {
            const float w = exp(g[t] - best);
            den += w;
            num += w * v[t];
        }
        p[i] = num / den;
        sum2 += p[i] * p[i];
    }
    const float inv = rsqrt(simd_sum(sum2) / 128.0f + 1e-5f);
    device half *dst = pool + ulong(group) * 128;
    dst[lane] = half(p[0] * norm[lane] * inv);
    dst[lane + 32] = half(p[1] * norm[lane + 32] * inv);
    const uint d = 64 + 2*lane;
    // Shuffle the appropriate half separately, since the source value must
    // be selected by the receiving lane.
    const float x2 = simd_shuffle(p[2], (2*lane) % 32), y2 = simd_shuffle(p[2], (2*lane+1) % 32);
    const float x3 = simd_shuffle(p[3], (2*lane) % 32), y3 = simd_shuffle(p[3], (2*lane+1) % 32);
    const float2 rot = hy4_rotate(float2((lane < 16 ? x2 : x3) * norm[d],
                                        (lane < 16 ? y2 : y3) * norm[d+1]) * inv,
                                   lane, group * 4, a.theta);
    dst[d] = half(rot.x);
    dst[d+1] = half(rot.y);
}

kernel void kernel_hy4_scores(constant hy4_args &a,
                             device const half *pool, device const float *q,
                             device const float *w, device float *scores,
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const uint index = group.x, row = group.y;
    const uint pos = a.pos + row;
    if (index >= pos / 4) {
        if (!lane) scores[row * a.groups + index] = -INFINITY;
        return;
    }
    // One SIMD group scores all 16 index heads against its shared pool key.
    const half k0 = pool[index * 128 + lane], k1 = pool[index * 128 + lane + 32];
    const half k2 = pool[index * 128 + lane + 64], k3 = pool[index * 128 + lane + 96];
    float sum = 0;
    for (uint h = 0; h < 16; h++) {
        device const float *x = q + (row * 16 + h) * 128 + lane;
        const float s = simd_sum(k0*x[0] + k1*x[32] + k2*x[64] + k3*x[96]);
        sum += max(s, 0.0f) * w[row * 16 + h];
    }
    if (!lane) scores[row * a.groups + index] = sum;
}

// Each SIMD group accumulates an independent softmax partition in registers.
// Only the latent values and rotary key are stored per token, shared by all heads.
static inline void hy4_attention_partition(constant hy4_args &a, device const float *absorbed,
                                           device const float *q, device const half *cache,
                                           device const int *selected, device const float *sink,
                                           uint head, uint row, uint part, uint parts, uint lane,
                                           thread float &m, thread float &den, thread float (&v)[16]) {
    const uint pos = a.pos + row;
    const bool sparse = a.type != 0;
    const bool draft = a.type == 2;
    const uint history = draft ? a.pos - 1 : pos;
    const uint start = sparse || pos < 511 ? 0 : pos - 511;
    const bool indexed = sparse && history / 4 >= 512;
    const uint tail = history / 4 * 4;
    const uint count = indexed ? 511 * 4 + pos - tail + 1 : pos - start + 1;
    device const float *z = absorbed + (row * a.heads + head) * 512;
    device const float *r = q + (row * a.heads + head) * 256 + 192;
    float ql[16];
    for (uint i = 0; i < 16; i++) ql[i] = z[lane + 32*i];
    m = part == 0 ? sink[head] : -INFINITY;
    den = part == 0 ? 1.0f : 0.0f;
    for (uint i = part; i < count; i += parts) {
        uint token = start + i;
        if (indexed) token = i >= 2044 ? tail + i - 2044 :
                             uint(selected[(draft ? a.groups : row) * 511 + i / 4]) * 4 + i % 4;
        device const half *key = cache + ulong(token % a.capacity) * 576;
        float dot = r[lane] * key[512 + lane] + r[32+lane] * key[544 + lane];
        float k[16];
        for (uint j = 0; j < 16; j++) {
            k[j] = float(key[lane + 32*j]);
            dot += k[j] * ql[j];
        }
        const float s = simd_sum(dot) * (1.0f / 16.0f);
        const float next = max(m, s), keep = exp(m - next), add = exp(s - next);
        den = den * keep + add;
        for (uint j = 0; j < 16; j++) v[j] = v[j] * keep + add * k[j];
        m = next;
    }
}

kernel void kernel_hy4_attention(constant hy4_args &a,
                                device const float *absorbed, device const float *q,
                                device const half *cache, device const int *selected,
                                device const float *sink, device float *out,
                                uint2 id [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = id.x, row = id.y;
    float v[16] = {}, m, den;
    hy4_attention_partition(a, absorbed, q, cache, selected, sink, head, row, sg, 4, lane, m, den, v);
    threadgroup float partial[4][512];
    threadgroup float maxima[4], sums[4];
    if (!lane) { maxima[sg] = m; sums[sg] = den; }
    for (uint j = 0; j < 16; j++) partial[sg][lane + 32*j] = v[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float best = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    float normalizer = 0, factor[4];
    for (uint j = 0; j < 4; j++) {
        factor[j] = exp(maxima[j] - best);
        normalizer += factor[j] * sums[j];
    }
    for (uint j = 0; j < 16; j++) {
        const uint d = lane + 32*j;
        float total = 0;
        for (uint b = 0; b < 4; b++) total += factor[b] * partial[b][d];
        out[(row * a.heads + head) * 512 + d] = total / normalizer;
    }
}

// For longer single-token queries each workgroup processes one key partition
// for four heads. The sink belongs only to partition zero.
kernel void kernel_hy4_attention_partial(constant hy4_args &a,
                                         device const float *absorbed, device const float *q,
                                         device const half *cache, device const int *selected,
                                         device const float *sink, device float *out,
                                         uint2 id [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]],
                                         uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint part = id.x, head = id.y * 4 + sg;
    float v[16] = {}, m, den;
    hy4_attention_partition(a, absorbed, q, cache, selected, sink, head, 0, part, a.groups, lane, m, den, v);
    device float *dst = out + (head * a.groups + part) * 514;
    for (uint j = 0; j < 16; j++) dst[lane + 32*j] = v[j];
    if (!lane) { dst[512] = m; dst[513] = den; }
}

kernel void kernel_hy4_attention_merge(constant hy4_args &a, device const float *partial,
                                       device float *out, uint head [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float weights[64], maxima[64], sums[64], normalizer;
    device const float *p = partial + head * a.groups * 514;
    if (tid < a.groups) {
        maxima[tid] = p[tid * 514 + 512]; sums[tid] = p[tid * 514 + 513];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = maxima[0], den = 0;
        for (uint i = 1; i < a.groups; i++) m = max(m, maxima[i]);
        for (uint i = 0; i < a.groups; i++) {
            weights[i] = exp(maxima[i] - m);
            den += weights[i] * sums[i];
        }
        normalizer = den;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 value = 0;
    for (uint i = 0; i < a.groups; i++) {
        device const float *src = p + i * 514 + 4 * tid;
        value += weights[i] * float4(src[0], src[1], src[2], src[3]);
    }
    ((device float4 *)out)[head * 128 + tid] = value / normalizer;
}

kernel void kernel_hy4_gate(constant hy4_args &a,
                           device float *x, device const float *gate,
                           uint i [[thread_position_in_grid]]) {
    if (i < a.width * a.rows) x[i] *= hy4_sigmoid(gate[i]);
}

kernel void kernel_hy4_draft_fusion(constant hy4_args &a,
                                  device float *seed, device const float *mix,
                                  device const float *gate, uint i [[thread_position_in_grid]]) {
    if (i < 3072) seed[i] += mix[i] * gate[i];
}

kernel void kernel_hy4_router(constant hy4_args &a,
                             device const float *logits, device const float *bias,
                             device int *ids, device float *weights,
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    float rank[16], prob[16];
    for (uint i = 0; i < 16; i++) {
        const uint id = lane + 32*i;
        const float value = logits[row * 512 + id];
        prob[i] = isfinite(value) ? hy4_sigmoid(value) : 0.0f;
        rank[i] = prob[i] + bias[id];
    }
    float sum = 0;
    for (uint slot = 0; slot < 16; slot++) {
        float best = -INFINITY;
        uint id = 512;
        for (uint i = 0; i < 16; i++)
            if (rank[i] > best) { best = rank[i]; id = lane + 32*i; }
        const float top = simd_max(best);
        id = simd_min(best == top ? id : 512u);
        const float p = simd_shuffle(prob[id / 32], id % 32);
        sum += p;
        if (lane == id % 32) rank[id / 32] = -INFINITY;
        if (lane == 0) { ids[row * 16 + slot] = int(id); weights[row * 16 + slot] = p; }
    }
    if (lane == 0)
        for (uint i = 0; i < 16; i++) weights[row * 16 + i] *= 3.998f / max(sum, 1e-20f);
}

// Hy4 clamps routed SwiGLU before the down projection. The shared dense
// branch has no clamp. Keep this separate from the Qwen expert kernels.
kernel void kernel_hy4_expert_mid(constant hy4_args &a,
                                 device const float *x, device const int *ids,
                                 device const char *gate, device const char *up, device float *mid,
                                 uint3 id [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint d = id.x * 8 + sg, pair = id.z * 16 + id.y;
    const ulong off = (ulong(ids[pair]) * 1536 + d) * a.row_bytes;
    const float g = min(qwen4_row_dot(gate + off, x + id.z * 1536, a.type, 1536, lane), 10.0f);
    const float u = clamp(qwen4_row_dot(up + off, x + id.z * 1536, a.type, 1536, lane), -10.0f, 10.0f);
    if (lane == 0) mid[pair * 1536 + d] = g * hy4_sigmoid(g) * u;
}

kernel void kernel_hy4_expert_down_iq2_xxs(constant hy4_args &a,
                                            device const float *mid, device const int *ids,
                                            device const uchar *weight, device float *down,
                                            uint3 id [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint sg [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong table[128 + 256];
    qwen4_cb_load_tables<16>(table, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (id.x * 4 + sg) * 2, pair = id.z * 16 + id.y;
    const ulong offset = (ulong(ids[pair]) * 1536 + row) * a.row_bytes;
    device const float4 *x = (device const float4 *)(mid + pair * 1536);
    float sum0 = 0, sum1 = 0;
    for (uint block = lane / 16; block < 6; block += 2) {
        const uint col = lane % 16;
        device const float4 *p = x + block * 64 + col * 4;
        const float4 y[4] = {p[0], p[1], p[2], p[3]};
        device const uchar *w = weight + offset + block * 66;
        sum0 += qwen4_cb_dot16<16>(w, col, y, table);
        sum1 += qwen4_cb_dot16<16>(w + a.row_bytes, col, y, table);
    }
    const float y0 = simd_sum(sum0), y1 = simd_sum(sum1);
    if (lane == 0) {
        down[pair * 1536 + row] = y0;
        down[pair * 1536 + row + 1] = y1;
    }
}

// Reuse a decoded activation tile across several Q6_K weight rows. Each
// output retains the lane mapping and reduction order of the dense matvec.
template <ushort NR>
kernel void hy4_q6_rows(constant hy4_args &a, device const float4 *x,
                         device const block_q6_K *weight, device float *out,
                         uint group [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]],
                         ushort nsg [[simdgroups_per_threadgroup]]) {
    const uint row = (group * nsg + sg) * NR, nb = a.width / 256;
    const short h = lane % 8 / 4, part = lane % 4;
    float sums[NR] = {};
    for (uint ib = lane / 8; ib < nb; ib += 4) {
        device const float4 *src = x + ib * 64 + h * 32 + part * 2;
        float4 y[1][8];
        for (short k = 0; k < 4; k++) {
            y[0][2*k] = src[8*k].xzyw;
            y[0][2*k+1] = src[8*k+1].xzyw;
        }
        for (ushort r = 0; r < NR; r++) {
            float partial[1];
            ds4q_q6_K_dotn<1>(weight + (row + r) * nb + ib, y, h, part, partial);
            sums[r] += partial[0];
        }
    }
    for (ushort r = 0; r < NR; r++) {
        const float value = simd_sum(sums[r]);
        if (lane == 0) out[row + r] = value;
    }
}
typedef decltype(hy4_q6_rows<1>) hy4_q6_rows_t;
template [[host_name("kernel_hy4_q6_r4")]] kernel hy4_q6_rows_t hy4_q6_rows<4>;
template [[host_name("kernel_hy4_q6_r8")]] kernel hy4_q6_rows_t hy4_q6_rows<8>;
