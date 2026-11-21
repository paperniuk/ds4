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

// Eight lanes own one output row. Four independent rows share each SIMD
// group, including for the short 192-element key projection.
kernel void kernel_hy4_head_mv(constant hy4_args &a,
                              device const float4 *in, device const char *weight, device float *out,
                              uint2 id [[threadgroup_position_in_grid]],
                              ushort lane [[thread_index_in_simdgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint col = lane % 8;
    const uint row = id.x * 16 + sg * 4 + lane / 8;
    const bool key = a.width == 192;
    const uint out_dim = key ? 512 : 256;
    device const float4 *x = in + id.y * (key ? 64 : 128);
    device const char *w = weight + (ulong(id.y) * out_dim + row) * a.row_bytes;
    float sum = 0.0f;
    if (a.type == 8) {
        for (uint block = 0; block < a.width / 32; block++) {
            device const char *b = w + block * 34;
            const char4 q = *(device const packed_char4 *)(b + 2 + 4 * col);
            sum += float(*(device const half *)b) * dot(float4(q), x[block * 8 + col]);
        }
    } else {
        const short h = col / 4, part = col % 4;
        for (uint block = 0; block < a.width / 256; block++) {
            device const float4 *src = x + block * 64 + h * 32 + part * 2;
            float4 y[1][8];
#pragma unroll
            for (uint k = 0; k < 4; k++) {
                y[0][2 * k] = src[8 * k].xzyw;
                y[0][2 * k + 1] = src[8 * k + 1].xzyw;
            }
            float partial[1];
            ds4q_q6_K_dotn<1>((device const block_q6_K *)w + block, y, h, part, partial);
            sum += partial[0];
        }
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    if (col == 0) out[id.y * out_dim + row] = sum;
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

// Prefill MLA head projections. A threadgroup covers 32 tokens and 32 outputs
// of one head. Each SIMD group reuses one 8x8 weight tile across four token tiles.
template <typename Block, short Chunks, void (*deq)(device const Block *, short, thread float4x4 &)>
kernel void kernel_hy4_head_mm_batch(constant hy4_args &a,
                                    device const float *in, device const char *weight, device float *out,
                                    uint3 id [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint out_dim = a.width == 192 ? 512 : 256;
    const uint in_dim = a.width == 192 ? 256 : 512;
    const uint in_stride = a.heads * in_dim;
    const uint out_stride = a.heads * out_dim;
    const uint col0 = id.x * 32, row0 = id.z * 32;
    const uint tokens = min(a.rows - row0, 32u);
    const uint row = tid / 4, col = tid % 4 * 8;
    device const float *src = in + row0 * in_stride + id.y * in_dim;
    device float *dst = out + row0 * out_stride + id.y * out_dim + col0;
    threadgroup float4 weights[32][8], inputs[32][8];
    simdgroup_float8x8 accum[4];
#pragma unroll
    for (uint t = 0; t < 4; t++) accum[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k = 0; k < a.width; k += 32) {
        // Use the same Q6_K chunks as the row dot and the same signed Q8_0 scales.
        if (tid < 64) {
            const uint d = tid / 2, c = k / 16 + tid % 2;
            const ulong offset = (ulong(id.y) * out_dim + col0 + d) * a.row_bytes;
            device const Block *blocks = (device const Block *)(weight + offset);
            float4x4 decoded;
            deq(blocks + c / Chunks, short(c % Chunks), decoded);
#pragma unroll
            for (uint i = 0; i < 4; i++)
                weights[d][(tid % 2) * 4 + i] = decoded[i];
        }
        threadgroup float4 *staged = inputs[row] + col / 4;
        if (row < tokens) {
            device const float4 *x = (device const float4 *)(src + row * in_stride + k + col);
            staged[0] = x[0];
            staged[1] = x[1];
        } else {
            staged[0] = staged[1] = float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 w;
            simdgroup_load(w, (threadgroup float *)weights[sg * 8] + kk, 32, 0, true);
#pragma unroll
            for (uint t = 0; t < 4; t++) {
                if (t * 8 < tokens) {
                    simdgroup_float8x8 x_tile;
                    simdgroup_load(x_tile, (threadgroup float *)inputs[t * 8] + kk, 32, 0, false);
                    simdgroup_multiply_accumulate(accum[t], x_tile, w, accum[t]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#pragma unroll
    for (uint t = 0; t < 4; t++) {
        if ((t + 1) * 8 <= tokens) {
            simdgroup_store(accum[t], dst + t * 8 * out_stride + sg * 8, out_stride, 0, false);
        } else if (t * 8 < tokens) {
            // Reuse the input tile as scratch for the last partial SIMD tile.
            simdgroup_store(accum[t], (threadgroup float *)inputs[t * 8] + sg * 8, 32, 0, false);
        }
    }
    const uint tail = tokens & ~7u;
    if (tail != tokens) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row >= tail && row < tokens) {
            device float4 *y = (device float4 *)(dst + row * out_stride + col);
            const threadgroup float4 *staged = inputs[row] + col / 4;
            y[0] = staged[0];
            y[1] = staged[1];
        }
    }
}

template [[host_name("kernel_hy4_head_mm_batch_q8_0")]]
kernel void kernel_hy4_head_mm_batch<block_q8_0, 2, dequantize_q8_0>(
    constant hy4_args &, device const float *, device const char *, device float *, uint3, uint, uint);
template [[host_name("kernel_hy4_head_mm_batch_q6_k")]]
kernel void kernel_hy4_head_mm_batch<block_q6_K, 16, dequantize_q6_K>(
    constant hy4_args &, device const float *, device const char *, device float *, uint3, uint, uint);

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

// Process contiguous partitions of 32 keys, sharing an eight-key tile
// across four heads. One lane per key computes its tile softmax weight.
kernel void kernel_hy4_attention_decode_partial(constant hy4_args &a,
                                                device const float4 *absorbed, device const float *q,
                                                device const half4 *cache, device const int *selected,
                                                device const float *sink, device float *out,
                                                uint2 id [[threadgroup_position_in_grid]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]],
                                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint head = id.y * 4 + sg;
    const uint start = a.type || a.pos < 511 ? 0 : a.pos - 511;
    const bool indexed = a.type && a.pos >= 2048;
    const uint tail = a.pos / 4 * 4;
    const uint count = indexed ? 2045 + a.pos % 4 : a.pos - start + 1;
    const uint end = min((id.x + 1) * 32, count);
    device const float4 *z = absorbed + head * 128;
    device const float *r = q + head * 256 + 192;
    float4 query[4], value[4] = {};
#pragma unroll
    for (uint d = 0; d < 4; d++) query[d] = z[d * 32 + lane];
    const float r0 = r[lane], r1 = r[lane + 32];
    float best = id.x == 0 ? sink[head] : -INFINITY;
    float den = id.x == 0 ? 1.0f : 0.0f;
    threadgroup half4 keys[8][144];
    threadgroup uint offsets[8];

    for (uint first = id.x * 32; first < end; first += 8) {
        const uint live = min(8u, end - first);
        if (tid < live) {
            const uint i = first + tid;
            const uint token = !indexed ? start + i :
                               i < 2044 ? uint(selected[i / 4]) * 4 + i % 4 : tail + i - 2044;
            offsets[tid] = token % a.capacity;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < live * 144; e += 128) {
            const uint k = e / 144, d = e % 144;
            keys[k][d] = cache[ulong(offsets[k]) * 144 + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = -INFINITY;
        for (uint k = 0; k < live; k++) {
            const threadgroup half *rope = (const threadgroup half *)(keys[k] + 128);
            float dot_product = r0 * float(rope[lane]) + r1 * float(rope[lane + 32]);
#pragma unroll
            for (uint d = 0; d < 4; d++)
                dot_product += dot(query[d], float4(keys[k][d * 32 + lane]));
            const float s = simd_sum(dot_product) * (1.0f / 16.0f);
            if (lane == k) score = s;
        }
        const float next = max(best, simd_max(score));
        const float keep = exp(best - next);
        const float prob = lane < live ? exp(score - next) : 0.0f;
        den = den * keep + simd_sum(prob);
        best = next;
#pragma unroll
        for (uint d = 0; d < 4; d++) value[d] *= keep;
        for (uint k = 0; k < live; k++) {
            const float weight = simd_shuffle(prob, k);
#pragma unroll
            for (uint d = 0; d < 4; d++) value[d] += weight * float4(keys[k][d * 32 + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The 514-float partition stride is only half a float4 aligned.
    device float *dst = out + (head * a.groups + id.x) * 514;
#pragma unroll
    for (uint d = 0; d < 4; d++)
#pragma unroll
        for (uint j = 0; j < 4; j++) dst[128 * d + 4 * lane + j] = value[d][j];
    if (lane == 0) { dst[512] = best; dst[513] = den; }
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

// Split the sign operation into two 32-bit byte vectors. No byte can
// overflow into the next one: the IQ2 grid magnitudes are nonzero.
static inline float hy4_iq2_dot8(uint grid, uint signs, float4 x) {
    const uint packed = (grid ^ signs) + (signs & 0x01010101u);
    return dot(float4(as_type<char4>(packed)), x);
}

static inline float hy4_iq2_mid_dot16(device const uchar *block, uint col,
                                     thread const float4 (&x)[4], threadgroup const ulong *table) {
    // The IQ2 block stride is 66 bytes, so the packed load is only 2-byte aligned.
    const ushort4 words = *(device const packed_ushort4 *)(block + 2 + (col / 2) * 8);
    const uint2 codes = as_type<uint2>(words);
    const uint half_index = col % 2;
    const uint grids = codes.x >> (half_index * 16);
    const uint signs = codes.y >> (half_index * 14);
    float sum = 0.0f;
#pragma unroll
    for (uint i = 0; i < 2; i++) {
        const uint2 grid = as_type<uint2>(qwen4_cb_grid<16>((grids >> (8 * i)) & 255u, table));
        const uint2 mask = as_type<uint2>(qwen4_cb_mask<16>((signs >> (7 * i)) & 127u, table));
        sum += hy4_iq2_dot8(grid.x, mask.x, x[2 * i]) +
               hy4_iq2_dot8(grid.y, mask.y, x[2 * i + 1]);
    }
    const float scale = float(*(device const half *)block) * (0.5f + float(codes.y >> 28)) * 0.25f;
    return scale * sum;
}

// Four gate/up output rows reuse one vector of activations in each SIMD
// group. Keep the grid and sign masks in threadgroup memory as in Qwen.
kernel void kernel_hy4_expert_mid_iq2_xxs_r4(constant hy4_args &a,
                                           device const float4 *x, device const int *ids,
                                           device const uchar *gate, device const uchar *up,
                                           device float *mid,
                                           uint2 id [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint sg [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong table[128 + 256];
    qwen4_cb_load_tables<16>(table, tid, 128);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = (id.x * 4 + sg) * 4;
    const ulong offset = (ulong(ids[id.y]) * 1536 + row) * a.row_bytes;
    float g[4] = {}, u[4] = {};
    const uint col = lane % 16;
#pragma unroll
    for (uint step = 0; step < 3; step++) {
        const uint block = lane / 16 + 2 * step;
        device const float4 *p = x + block * 64 + col * 4;
        const float4 y[4] = {p[0], p[1], p[2], p[3]};
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            const ulong off = offset + r * a.row_bytes + block * 66;
            g[r] += hy4_iq2_mid_dot16(gate + off, col, y, table);
            u[r] += hy4_iq2_mid_dot16(up + off, col, y, table);
        }
    }
#pragma unroll
    for (uint r = 0; r < 4; r++) {
        const float gv = min(simd_sum(g[r]), 10.0f);
        const float uv = clamp(simd_sum(u[r]), -10.0f, 10.0f);
        if (lane == 0) mid[id.y * 1536 + row + r] = qwen4_silu(gv) * uv;
    }
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

// Eight SIMD groups cover the 16 routed slots for four output rows.
// Reduce the slots in the original router order after all dots are ready.
kernel void kernel_hy4_expert_down_reduce_iq2_xxs(constant hy4_args &a,
                                                device const float4 *mid, device const int *ids,
                                                device const uchar *weight, device const float *probs,
                                                device float *out,
                                                uint group [[threadgroup_position_in_grid]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint sg [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong table[128 + 256];
    threadgroup float partial[16][4];
    qwen4_cb_load_tables<16>(table, tid, 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row = group * 4;
    const uint col = lane % 16;
    for (uint slot = sg; slot < 16; slot += 8) {
        const ulong offset = (ulong(ids[slot]) * 1536 + row) * a.row_bytes;
        device const float4 *x = mid + slot * 384;
        float sum[4] = {};
#pragma unroll
        for (uint step = 0; step < 3; step++) {
            const uint block = lane / 16 + 2 * step;
            device const float4 *p = x + block * 64 + col * 4;
            const float4 y[4] = {p[0], p[1], p[2], p[3]};
            device const uchar *w = weight + offset + block * 66;
#pragma unroll
            for (uint r = 0; r < 4; r++) sum[r] += qwen4_cb_dot16<16>(w + r * a.row_bytes, col, y, table);
        }
#pragma unroll
        for (uint r = 0; r < 4; r++) {
            const float value = simd_sum(sum[r]);
            if (lane == 0) partial[slot][r] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        float value = 0.0f;
        for (uint slot = 0; slot < 16; slot++) value += probs[slot] * partial[slot][tid];
        out[row + tid] = value;
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

#define HY4_VERIFY_INSTANCE(NAME, N) \
    template [[host_name(#NAME "_" #N)]] kernel decltype(NAME<N>) NAME<N>;
#define HY4_VERIFY_INSTANCES(NAME) \
    HY4_VERIFY_INSTANCE(NAME, 1) \
    HY4_VERIFY_INSTANCE(NAME, 2) \
    HY4_VERIFY_INSTANCE(NAME, 3) \
    HY4_VERIFY_INSTANCE(NAME, 4) \
    HY4_VERIFY_INSTANCE(NAME, 8)

// Decode four weights at a time and use them for each token before moving
// on. Keeping the activation loads here avoids a full block per token in
// registers. The eight lanes retain the decode Q6_K block mapping.
template <ushort NT>
static inline void hy4_q6_dot_rows(device const block_q6_K *b, device const float4 *x,
                                  uint stride, uint rows, short h, short part,
                                  thread float (&sums)[NT]) {
    const short l = 8 * part;
    device const ushort *qa = (device const ushort *)(b->ql + 64 * h + l);
    device const ushort *qb = (device const ushort *)(b->ql + 64 * h + 32 + l);
    device const ushort *qh = (device const ushort *)(b->qh + 32 * h + l);
    device const ushort *sc = (device const ushort *)b->scales + 4 * h;
    const uint lo[2] = {ds4q_u32(qa), ds4q_u32(qa + 2)};
    const uint hi[2] = {ds4q_u32(qb), ds4q_u32(qb + 2)};
    const uint upper[2] = {ds4q_u32(qh), ds4q_u32(qh + 2)};
    float partial[NT] = {};
#pragma unroll
    for (short k = 0; k < 4; k++) {
        const float scale = float(char(sc[k] >> (8 * (part / 2))));
#pragma unroll
        for (short i = 0; i < 2; i++) {
            const uint q = k == 0 ? (lo[i] & 0x0F0F0F0Fu) | ((upper[i] & 0x03030303u) << 4) :
                           k == 1 ? (hi[i] & 0x0F0F0F0Fu) | ((upper[i] & 0x0C0C0C0Cu) << 2) :
                           k == 2 ? ((lo[i] >> 4) & 0x0F0F0F0Fu) | (upper[i] & 0x30303030u) :
                                    ((hi[i] >> 4) & 0x0F0F0F0Fu) | ((upper[i] >> 2) & 0x30303030u);
            const float2 e = ds4q_h2_bits(q & 0x00FF00FFu, 1056.h);
            const float2 o = ds4q_h2_bits((q >> 8) & 0x00FF00FFu, 1056.h);
            const float4 w = scale * float4(e.x, o.x, e.y, o.y);
#pragma unroll
            for (ushort t = 0; t < NT; t++) {
                if (NT < 8 || t < rows) partial[t] += dot(w, x[t * stride + 8 * k + i]);
            }
        }
    }
    const float d = float(b->d);
#pragma unroll
    for (ushort t = 0; t < NT; t++) sums[t] += d * partial[t];
}


// Keep the decode head/output mapping and walk the same weights for all tokens.
template <ushort NT>
kernel void kernel_hy4_head_mv_rows(constant hy4_args &a, device const float4 *in,
                                     device const char *weight, device float *out,
                                     uint2 id [[threadgroup_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]],
                                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint col = lane % 8;
    const uint row = id.x * 16 + sg * 4 + lane / 8;
    const bool key = a.width == 192;
    const uint out_dim = key ? 512 : 256, head_stride = key ? 64 : 128;
    const uint token_stride = a.heads * head_stride;
    device const float4 *x = in + id.y * head_stride;
    device const char *w = weight + (ulong(id.y) * out_dim + row) * a.row_bytes;
    float sums[NT] = {};
    if (a.type == 8) {
        for (uint block = 0; block < a.width / 32; block++) {
            device const char *b = w + block * 34;
            const float4 q = float4(*(device const packed_char4 *)(b + 2 + 4 * col));
            const float scale = float(*(device const half *)b);
#pragma unroll
            for (ushort t = 0; t < NT; t++) {
                if (NT < 8 || t < a.rows)
                    sums[t] += scale * dot(q, x[t * token_stride + block * 8 + col]);
            }
        }
    } else {
        const short h = col / 4, part = col % 4;
        for (uint block = 0; block < a.width / 256; block++) {
            device const float4 *src = x + block * 64 + h * 32 + part * 2;
            hy4_q6_dot_rows((device const block_q6_K *)w + block, src,
                            token_stride, a.rows, h, part, sums);
        }
    }
#pragma unroll
    for (ushort t = 0; t < NT; t++) {
        float sum = sums[t];
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 4);
        if (col == 0 && (NT < 8 || t < a.rows)) out[(t * a.heads + id.y) * out_dim + row] = sum;
    }
}
HY4_VERIFY_INSTANCES(kernel_hy4_head_mv_rows)

template <typename Weight, ushort NT>
static inline void hy4_float_dot_rows(device const Weight *w, device const float4 *x,
                                     uint width, uint rows, uint first, uint step,
                                     thread float (&sums)[NT]) {
    for (uint i = first; i < width; i += step) {
        const float4 v = float4(w[i]);
#pragma unroll
        for (ushort t = 0; t < NT; t++) {
            if (NT < 8 || t < rows) sums[t] += dot(v, x[t * width + i]);
        }
    }
}

// A narrow output needs more SIMD groups across its input dimension.
// Both layouts write final outputs without a device-memory partial plane.
template <ushort NT>
kernel void kernel_hy4_float_mv_rows(constant hy4_args &a, device const float4 *x,
                                      device const char *weight, device float *out,
                                      uint group [[threadgroup_position_in_grid]],
                                      ushort tid [[thread_index_in_threadgroup]],
                                      ushort sg [[simdgroup_index_in_threadgroup]],
                                      ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NT][8];
    const bool narrow = a.groups < 64;
    const uint row = narrow ? group : group * 4 + sg;
    if (row >= a.groups) return;
    const uint width = a.width / 4;
    const uint first = narrow ? tid : lane, step = narrow ? 256 : 32;
    device const char *w = weight + ulong(row) * a.row_bytes;
    float sums[NT] = {};
    if (a.type == 0) hy4_float_dot_rows((device const float4 *)w, x, width, a.rows, first, step, sums);
    else hy4_float_dot_rows((device const half4 *)w, x, width, a.rows, first, step, sums);
#pragma unroll
    for (ushort t = 0; t < NT; t++) {
        const float value = simd_sum(sums[t]);
        if (lane == 0) {
            if (narrow) partial[t][sg] = value;
            else if (NT < 8 || t < a.rows) out[t * a.groups + row] = value;
        }
    }
    if (narrow) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < a.rows) {
            float value = 0.0f;
#pragma unroll
            for (ushort i = 0; i < 8; i++) value += partial[tid][i];
            out[tid * a.groups + row] = value;
        }
    }
}
HY4_VERIFY_INSTANCES(kernel_hy4_float_mv_rows)

// The iHC projection shares the weights and accumulates each token's RMS
// denominator alongside the dots. hc_finish consumes the same partial layout.
template <ushort NT>
kernel void kernel_hy4_hc_partial_rows(constant hy4_args &a,
                                        device const float4 *streams, device const float4 *weights,
                                        device float *partial, uint group [[threadgroup_position_in_grid]],
                                        ushort lane [[thread_index_in_simdgroup]],
                                        ushort output [[simdgroup_index_in_threadgroup]]) {
    const uint block = 3072 / a.groups;
    float4 dots[NT] = {}, squares[NT] = {};
    for (uint i = lane; i < block; i += 32) {
        const uint offset = group * block + i;
        const float4 w = weights[output * 3072 + offset];
#pragma unroll
        for (ushort t = 0; t < NT; t++) {
            if (NT < 8 || t < a.rows) {
                const float4 x = streams[t * 3072 + offset];
                dots[t] = fma(x, w, dots[t]);
                if (output == 0) squares[t] = fma(x, x, squares[t]);
            }
        }
    }
#pragma unroll
    for (ushort t = 0; t < NT; t++) {
        const float result = simd_sum(dot(dots[t], float4(1)));
        const float sq = simd_sum(dot(squares[t], float4(1)));
        if (lane == 0 && (NT < 8 || t < a.rows)) {
            device float *dst = partial + (t * a.groups + group) * 9;
            dst[output] = result;
            if (output == 0) dst[8] = sq;
        }
    }
}
HY4_VERIFY_INSTANCES(kernel_hy4_hc_partial_rows)
