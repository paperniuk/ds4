// Weight types of mixed-precision GGUFs (ISTA-DASLab's Qwen3.8 release):
// Q2_0, Q5_0, IQ4_NL, Q3_K, Q5_K, Q6_K, IQ4_XS and BF16. Block layouts,
// dequantizers and matvec kernels follow llama.cpp's Metal backend (MIT).
// Every dequantizer returns elements in natural order: il selects 16
// consecutive weights (the _t4 variants 4), so kernel_mul_mm and the
// mul_mv_ext template in dense.metal can use them unchanged. Q5_K and Q6_K
// reuse the blocks and 4x4 dequantizers of moe.metal.

#ifndef QK_K
#define QK_K 256
#endif
#define QK2_0 64

struct block_q2_0 {
    half d;
    uchar qs[QK2_0/4];
};

struct block_q5_0 {
    half d;
    uchar qh[4];
    uchar qs[16];
};

struct block_iq4_nl {
    half d;
    uchar qs[16];
};

struct block_q3_K {
    uchar hmask[QK_K/8];
    uchar qs[QK_K/4];
    uchar scales[12];
    half d;
};

struct block_iq4_xs {
    half d;
    ushort scales_h;
    uchar scales_l[QK_K/64];
    uchar qs[QK_K/2];
};

// 16 BF16 weights, the kernel_mul_mm block for nl = 1.
struct ds4q_bf16x16 {
    ushort v[16];
};

constant float ds4q_kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f,
};

static inline float ds4q_bf16(ushort u) {
    return as_type<float>((uint)u << 16);
}

/* Q5_0 keeps its 32 high bits at byte 2 of a 22-byte block: two aligned
 * u16 loads instead of an unaligned u32. */
static inline uint ds4q_q5_0_qh(device const block_q5_0 *xb) {
    device const ushort *q16 = (device const ushort *)xb;
    return (uint)q16[1] | ((uint)q16[2] << 16);
}

/* --- dequantizers -------------------------------------------------------- */

template <typename type4x4>
void ds4q_dequantize_q2_0(device const block_q2_0 *xb, short il, thread type4x4 &reg) {
    const float d = xb->d;
    device const uchar *qs = xb->qs + 4*il;
    float4x4 r;
    for (int i = 0; i < 4; i++) {
        const uchar b = qs[i];
        r[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
        r[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
        r[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
        r[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
    }
    reg = (type4x4)r;
}

template <typename type4>
void ds4q_dequantize_q2_0_t4(device const block_q2_0 *xb, short il, thread type4 &reg) {
    const float d = xb->d;
    const uchar b = xb->qs[il];
    reg = (type4)(float4((float)((b >> 0) & 3), (float)((b >> 2) & 3),
                         (float)((b >> 4) & 3), (float)((b >> 6) & 3)) - 1.0f) * d;
}

template <typename type4x4>
void ds4q_dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 &reg) {
    const float d = xb->d;
    const uint qh = ds4q_q5_0_qh(xb);
    float4x4 r;
    for (int i = 0; i < 16; i++) {
        const uint q = xb->qs[i];
        const uint v = il ? ((q >> 4) | (((qh >> (i + 16)) & 1u) << 4))
                          : ((q & 15u) | (((qh >> i) & 1u) << 4));
        r[i/4][i%4] = d * ((float)v - 16.0f);
    }
    reg = (type4x4)r;
}

template <typename type4>
void ds4q_dequantize_q5_0_t4(device const block_q5_0 *xb, short il, thread type4 &reg) {
    const float d = xb->d;
    const uint qh = ds4q_q5_0_qh(xb);
    const short hi = il / 4;
    const short j0 = 4*(il % 4);
    float4 r;
    for (int i = 0; i < 4; i++) {
        const uint q = xb->qs[j0 + i];
        const uint v = hi ? ((q >> 4) | (((qh >> (j0 + i + 16)) & 1u) << 4))
                          : ((q & 15u) | (((qh >> (j0 + i)) & 1u) << 4));
        r[i] = d * ((float)v - 16.0f);
    }
    reg = (type4)r;
}

template <typename type4x4>
void ds4q_dequantize_iq4_nl(device const block_iq4_nl *xb, short il, thread type4x4 &reg) {
    const float d = xb->d;
    float4x4 r;
    for (int i = 0; i < 16; i++) {
        const uchar q = xb->qs[i];
        r[i/4][i%4] = d * ds4q_kvalues_iq4nl[il ? (q >> 4) : (q & 15)];
    }
    reg = (type4x4)r;
}

template <typename type4>
void ds4q_dequantize_iq4_nl_t4(device const block_iq4_nl *xb, short il, thread type4 &reg) {
    const float d = xb->d;
    const short hi = il / 4;
    const short j0 = 4*(il % 4);
    float4 r;
    for (int i = 0; i < 4; i++) {
        const uchar q = xb->qs[j0 + i];
        r[i] = d * ds4q_kvalues_iq4nl[hi ? (q >> 4) : (q & 15)];
    }
    reg = (type4)r;
}

template <typename type4x4>
void ds4q_dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 &reg) {
    const float d_all = xb->d;
    device const uchar *q = xb->qs + 32*(il/8) + 16*(il&1);
    device const uchar *h = xb->hmask + 16*(il&1);
    device const char *scales = (device const char *)xb->scales;
    const uchar m = 1 << (il/2);
    const ushort kmask1 = (il/4) > 1 ? ((il/4) > 2 ? 192 : 48) : ((il/4) > 0 ? 12 : 3);
    const ushort kmask2 = il/8 ? 0xF0 : 0x0F;
    const ushort scale_2 = (uchar)scales[il%8], scale_1 = (uchar)scales[8 + il%4];
    const short dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                                  : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il < 8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;
    const short sh = (il/2) & 3;
    const float coef = sh > 1 ? (sh > 2 ? 1.f/64.f : 1.f/16.f) : (sh > 0 ? 1.f/4.f : 1.f);
    const uchar mask = sh > 1 ? (sh > 2 ? 192 : 48) : (sh > 0 ? 12 : 3);
    dl *= coef;
    float4x4 r;
    for (int i = 0; i < 16; i++) {
        r[i/4][i%4] = dl * (float)(q[i] & mask) - ((h[i] & m) ? 0.f : ml);
    }
    reg = (type4x4)r;
}

template <typename type4x4>
void ds4q_dequantize_iq4_xs(device const block_iq4_xs *xb, short il, thread type4x4 &reg) {
    const int ib32 = il/2;
    const short hi = il%2;
    device const uchar *qs = xb->qs + 16*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    float4x4 r;
    for (int i = 0; i < 16; i++) {
        r[i/4][i%4] = d * ds4q_kvalues_iq4nl[hi ? (qs[i] >> 4) : (qs[i] & 15)];
    }
    reg = (type4x4)r;
}

template <typename type4x4>
void ds4q_dequantize_bf16(device const ds4q_bf16x16 *xb, short il, thread type4x4 &reg) {
    float4x4 r;
    for (int i = 0; i < 16; i++) r[i/4][i%4] = ds4q_bf16(xb->v[i]);
    reg = (type4x4)r;
}

template <typename type4>
void ds4q_dequantize_bf16_t4(device const ushort4 *xb, short il, thread type4 &reg) {
    const ushort4 u = *xb;
    reg = (type4)float4(ds4q_bf16(u.x), ds4q_bf16(u.y), ds4q_bf16(u.z), ds4q_bf16(u.w));
}

/* 4-weight chunks of the 256-blocks, cut from the 16-weight dequantizer. */
#define DS4Q_T4_FROM_4X4(name, block_t, deq)                                   \
template <typename type4>                                                     \
void name(device const block_t *xb, short il, thread type4 &reg) {            \
    float4x4 tmp;                                                             \
    deq(xb, il / 4, tmp);                                                     \
    reg = (type4)tmp[il & 3];                                                 \
}

DS4Q_T4_FROM_4X4(ds4q_dequantize_q3_K_t4, block_q3_K, ds4q_dequantize_q3_K)
DS4Q_T4_FROM_4X4(ds4q_dequantize_q5_K_t4, block_q5_K, dequantize_q5_K)
DS4Q_T4_FROM_4X4(ds4q_dequantize_q6_K_t4, block_q6_K, dequantize_q6_K)

/* --- one row dot for the Qwen generic row kernels ------------------------ */

/* Each lane takes 16-weight chunks strided by the simdgroup; the caller does
 * the simd_sum. */
template <typename block_t, short nl, void (*deq)(device const block_t *, short, thread float4x4 &)>
static inline float ds4q_row_dot_chunks(device const char *row, device const float *x,
                                        uint in_dim, ushort tiisg) {
    device const block_t *xb = (device const block_t *)row;
    const uint nchunks = in_dim / 16;
    float acc = 0.0f;
    for (uint c = tiisg; c < nchunks; c += 32) {
        float4x4 w;
        deq(xb + c / nl, (short)(c % nl), w);
        device const float4 *y = (device const float4 *)(x + (uint64_t)c * 16);
        acc += dot(w[0], y[0]) + dot(w[1], y[1]) + dot(w[2], y[2]) + dot(w[3], y[3]);
    }
    return acc;
}

static inline bool ds4q_row_dot_has(uint type) {
    return type == 42 || type == 6 || type == 20 || type == 11 ||
           type == 13 || type == 14 || type == 23;
}

static inline float ds4q_row_dot_part(device const char *row, device const float *x,
                                      uint type, uint in_dim, ushort tiisg) {
    switch (type) {
    case 42: return ds4q_row_dot_chunks<block_q2_0,   4,  ds4q_dequantize_q2_0>(row, x, in_dim, tiisg);
    case 6:  return ds4q_row_dot_chunks<block_q5_0,   2,  ds4q_dequantize_q5_0>(row, x, in_dim, tiisg);
    case 20: return ds4q_row_dot_chunks<block_iq4_nl, 2,  ds4q_dequantize_iq4_nl>(row, x, in_dim, tiisg);
    case 11: return ds4q_row_dot_chunks<block_q3_K,   16, ds4q_dequantize_q3_K>(row, x, in_dim, tiisg);
    case 13: return ds4q_row_dot_chunks<block_q5_K,   16, dequantize_q5_K>(row, x, in_dim, tiisg);
    case 14: return ds4q_row_dot_chunks<block_q6_K,   16, dequantize_q6_K>(row, x, in_dim, tiisg);
    default: return ds4q_row_dot_chunks<block_iq4_xs, 16, ds4q_dequantize_iq4_xs>(row, x, in_dim, tiisg);
    }
}

/* --- matvec for 1..4 tokens ---------------------------------------------- */

/* Each simdgroup owns NR0 consecutive rows; rows past ne01 are clamped to the
 * last one so the tail group never reads beyond the tensor, and are not
 * written.  A threadgroup serves NT consecutive tokens (grid y steps by NT):
 * lanes decode each weight word once and dot it with every token's inputs,
 * so MTP verification reads the weights once.  Tokens past ne11 re-read the
 * last one and are not written.  NR0 must match the nr0 column of
 * ds4_gpu_quant_kinds in ds4_metal.m.
 *
 * Weights reach float without a per-byte convert: two small fields at a time
 * are ORed into the mantissa of 1024.0h and the bias subtracted (the fields'
 * own offset, e.g. -4 for Q3_K, rides on the bias).  Inputs are loaded as
 * .xzyw so each half2 of even/odd bytes meets its pair of inputs. */
static inline device const char *ds4q_mv_row(constant ds4_metal_args_mul_mv &args,
                                             device const char *src0, int row) {
    return src0 + (uint64_t)min(row, args.ne01 - 1) * args.nb01;
}

static inline device const float *ds4q_mv_y(constant ds4_metal_args_mul_mv &args,
                                            device const char *src1, uint3 tgpig, uint tok) {
    return (device const float *)(src1 + (uint64_t)min(tok, (uint)args.ne11 - 1u)*args.nb11 +
                                  (uint64_t)tgpig.z*args.nb12);
}

template <short NR0, short NT>
static inline void ds4q_mv_write(constant ds4_metal_args_mul_mv &args, device char *dst, uint3 tgpig,
                                 uint t0, thread float (*sumf)[NR0], int first_row, ushort tiisg) {
    device float *d = (device float *)dst + (uint64_t)tgpig.z*args.ne0*args.ne1;
    FOR_UNROLL (short t = 0; t < NT; t++) {
        for (short row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[t][row]);
            if (tiisg == 0 && t0 + t < (uint)args.ne11 && first_row + row < args.ne01) {
                d[(uint64_t)(t0 + t)*args.ne0 + first_row + row] = tot;
            }
        }
    }
}

static inline float2 ds4q_h2_bits(uint w, half bias) {
    return float2(as_type<half2>(w | 0x64006400u) - half2(bias));
}

/* Byte i of x as a float, through the 2^23 mantissa instead of a convert. */
static inline float ds4q_byte_f(uint x, short i) {
    return as_type<float>(0x4B000000u | ((x >> (8*i)) & 0xFFu)) - 8388608.f;
}

static inline uint ds4q_u32(device const ushort *p) {
    return (uint)p[0] | ((uint)p[1] << 16);
}

/* Q2_0: four lanes per 64-weight block, 16 weights each.  One u32 of quants
 * holds weights 4b..4b+3 in byte b, so the 2-bit field k of every byte is
 * weights k, 4+k, 8+k, 12+k: yl[k] holds inputs k, 8+k, 4+k, 12+k. */
static inline void ds4q_q2_0_load_y(device const float *yb, thread float4 *yl) {
    device const float4 *y4 = (device const float4 *)yb;
    const float4 a = y4[0], b = y4[1], c = y4[2], d = y4[3];
    for (short k = 0; k < 4; k++) yl[k] = float4(a[k], c[k], b[k], d[k]);
}

static inline float ds4q_q2_0_dot(device const block_q2_0 *xb, thread const float4 *yl, short il) {
    device const ushort *q16 = (device const ushort *)(xb->qs + il/4);
    const uint w = (uint)q16[0] | ((uint)q16[1] << 16);
    float acc = 0.0f;
    for (short k = 0; k < 4; k++) {
        acc += dot(ds4q_h2_bits((w >> (2*k)) & 0x00030003u, 1025.h), yl[k].xy) +
               dot(ds4q_h2_bits((w >> (2*k + 8)) & 0x00030003u, 1025.h), yl[k].zw);
    }
    return (float)xb->d * acc;
}

/* NT tokens: the compiler shares the decode across the per-token calls. */
template <short NT>
static inline void ds4q_q2_0_dotn(device const block_q2_0 *xb, thread const float4 (*yl)[4], short il,
                                  thread float *acc) {
    FOR_UNROLL (short t = 0; t < NT; t++) acc[t] = ds4q_q2_0_dot(xb, yl[t], il);
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_q2_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/QK2_0;
    const uint t0 = tgpig.y*NT;
    device const block_q2_0 *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q2_0 *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/4, il = (tiisg%4)*16;
    device const float *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) yb[t] = ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*QK2_0 + il;
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][4];

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            ds4q_q2_0_load_y(yb[t], yl[t]);
            yb[t] += QK2_0*(N_SIMDWIDTH/4);
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            float acc[NT];
            ds4q_q2_0_dotn<NT>(ax[row] + ib, yl, il, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* Q5_0: two lanes per 32-weight block, 16 weights each: qs bytes il..il+7
 * (low nibbles: weights il.., high: il+16..) with their qh bits spread into
 * bit 4 of each byte. */
static inline uint ds4q_spread4(uint x) {
    return (x * 0x00204081u) & 0x01010101u;
}

template <short NT>
static inline void ds4q_q5_0_dotn(device const block_q5_0 *xb, thread const float4 (*yl)[4], short il,
                                  thread float *acc) {
    device const ushort *q16 = (device const ushort *)xb->qs + il/2;
    const uint w0 = ds4q_u32(q16), w1 = ds4q_u32(q16 + 2);
    const uint qh = ds4q_q5_0_qh(xb);
    const uint v[4] = {
        (w0 & 0x0F0F0F0Fu)        | (ds4q_spread4((qh >> il) & 0xFu) << 4),
        (w1 & 0x0F0F0F0Fu)        | (ds4q_spread4((qh >> (il + 4)) & 0xFu) << 4),
        ((w0 >> 4) & 0x0F0F0F0Fu) | (ds4q_spread4((qh >> (il + 16)) & 0xFu) << 4),
        ((w1 >> 4) & 0x0F0F0F0Fu) | (ds4q_spread4((qh >> (il + 20)) & 0xFu) << 4),
    };
    float a[NT] = {0.f};
    for (short i = 0; i < 4; i++) {
        const float2 e = ds4q_h2_bits(v[i] & 0x00FF00FFu, 1040.h);
        const float2 o = ds4q_h2_bits((v[i] >> 8) & 0x00FF00FFu, 1040.h);
        FOR_UNROLL (short t = 0; t < NT; t++) a[t] += dot(e, yl[t][i].xy) + dot(o, yl[t][i].zw);
    }
    const float d = (float)xb->d;
    FOR_UNROLL (short t = 0; t < NT; t++) acc[t] = d*a[t];
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_q5_0(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/32;
    const uint t0 = tgpig.y*NT;
    device const block_q5_0 *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q5_0 *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/2, il = (tiisg%2)*8;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*32 + il);
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][4];

    for (int ib = ix; ib < nb; ib += 16) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            yl[t][0] = yb[t][0].xzyw; yl[t][1] = yb[t][1].xzyw;
            yl[t][2] = yb[t][4].xzyw; yl[t][3] = yb[t][5].xzyw;
            yb[t] += 16*32/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            float acc[NT];
            ds4q_q5_0_dotn<NT>(ax[row] + ib, yl, il, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* Q3_K: eight lanes per 256-weight block, four blocks per simdgroup step.
 * Lane k covers weights 8*(k%4)..+7 of each 32-weight group j of half k/4:
 * two u32 of qs and two of hmask give eight 3-bit weights per group,
 * w = q + 4*h - 4. */
static inline float ds4q_q3_K_dot(device const block_q3_K *xb, thread const float4 *yl,
                                  short h, short lg) {
    device const ushort *q16 = (device const ushort *)(xb->qs + 32*h + 8*lg);
    device const ushort *h16 = (device const ushort *)(xb->hmask + 8*lg);
    device const ushort *a = (device const ushort *)xb->scales;
    const uint q0 = (uint)q16[0] | ((uint)q16[1] << 16);
    const uint q1 = (uint)q16[2] | ((uint)q16[3] << 16);
    const uint h0 = ((uint)h16[0] | ((uint)h16[1] << 16)) >> (4*h);
    const uint h1 = ((uint)h16[2] | ((uint)h16[3] << 16)) >> (4*h);
    const uint sl = ((uint)a[0] | ((uint)a[1] << 16)) >> (8*(lg/2) + 4*h);
    const uint sm = ((uint)a[2] | ((uint)a[3] << 16)) >> (8*(lg/2) + 4*h);
    const uint sh = ((uint)a[4] | ((uint)a[5] << 16)) >> (8*(lg/2) + 4*h);
    const uint lo[4] = { sl, sl >> 16, sm, sm >> 16 };
    float acc = 0.f;
    FOR_UNROLL (short j = 0; j < 4; j++) {
        const uint w0 = ((q0 >> (2*j)) & 0x03030303u) | (((h0 >> j) & 0x01010101u) << 2);
        const uint w1 = ((q1 >> (2*j)) & 0x03030303u) | (((h1 >> j) & 0x01010101u) << 2);
        const float sc = as_type<float>(0x4B000000u | (lo[j] & 0xF) |
                                        (((sh >> (16*(j & 1) + 2*(j >> 1))) & 3) << 4)) - 8388640.f;
        acc += sc * (dot(ds4q_h2_bits(w0 & 0x00FF00FFu, 1028.h), yl[2*j].xy) +
                     dot(ds4q_h2_bits((w0 >> 8) & 0x00FF00FFu, 1028.h), yl[2*j].zw) +
                     dot(ds4q_h2_bits(w1 & 0x00FF00FFu, 1028.h), yl[2*j + 1].xy) +
                     dot(ds4q_h2_bits((w1 >> 8) & 0x00FF00FFu, 1028.h), yl[2*j + 1].zw));
    }
    return (float)xb->d * acc;
}

/* NT tokens: the decode is the same pure function of the block, so the
 * compiler shares it across the per-token calls. */
template <short NT>
static inline void ds4q_q3_K_dotn(device const block_q3_K *xb, thread const float4 (*yl)[8],
                                  short h, short lg, thread float *acc) {
    FOR_UNROLL (short t = 0; t < NT; t++) acc[t] = ds4q_q3_K_dot(xb, yl[t], h, lg);
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_q3_K(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/QK_K;
    const uint t0 = tgpig.y*NT;
    device const block_q3_K *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q3_K *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/8, h = (tiisg%8)/4, lg = tiisg%4;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) {
        yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*QK_K + 128*h + 8*lg);
    }
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][8];

    for (int ib = ix; ib < nb; ib += 4) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            FOR_UNROLL (short j = 0; j < 4; j++) {
                yl[t][2*j] = yb[t][8*j].xzyw;
                yl[t][2*j + 1] = yb[t][8*j + 1].xzyw;
            }
            yb[t] += 4*QK_K/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            float acc[NT];
            ds4q_q3_K_dotn<NT>(ax[row] + ib, yl, h, lg, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* Q4_K and Q5_K: eight lanes per 256-weight block, four blocks per step.
 * Lane k covers weights 8*(k%4)..+7 of the four 32-weight sub-blocks
 * 4*(k/4)..+3, i.e. both nibbles of two qs groups (plus, for Q5_K, the
 * matching qh bits, moved to bit 4 of each byte).  Both block sizes keep
 * u32 alignment. */
static inline void ds4q_k_qh(device const block_q4_K *xb, short jp, short lr, thread uint &h0, thread uint &h1) {
    h0 = h1 = 0;
}

static inline void ds4q_k_qh(device const block_q5_K *xb, short jp, short lr, thread uint &h0, thread uint &h1) {
    device const uint *qh = (device const uint *)(xb->qh + 8*lr);
    h0 = qh[0] >> (4*jp);
    h1 = qh[1] >> (4*jp);
}

template <short NT, typename block_t>
static inline void ds4q_q45_K_dotn(device const block_t *xb, thread const float4 (*yl)[8],
                                   thread const float (*sumy)[4], short jp, short lr, thread float *acc) {
    device const uint *s32 = (device const uint *)xb->scales;
    const uint s0 = s32[0], s1 = s32[1], s2 = s32[2];
    const uint sc = jp == 0 ? s0 & 0x3F3F3F3Fu : (s2 & 0x0F0F0F0Fu) | ((s0 >> 2) & 0x30303030u);
    const uint mn = jp == 0 ? s1 & 0x3F3F3F3Fu : ((s2 >> 4) & 0x0F0F0F0Fu) | ((s1 >> 2) & 0x30303030u);
    uint h0, h1;
    ds4q_k_qh(xb, jp, lr, h0, h1);
    float s[NT] = {0.f}, m[NT] = {0.f};
    FOR_UNROLL (short g = 0; g < 2; g++) {
        device const uint *q = (device const uint *)(xb->qs + 32*(2*jp + g) + 8*lr);
        const uint q0 = q[0], q1 = q[1];
        FOR_UNROLL (short n = 0; n < 2; n++) {
            const short sb = 2*g + n;
            const uint w0 = ((q0 >> (4*n)) & 0x0F0F0F0Fu) | (((h0 >> sb) & 0x01010101u) << 4);
            const uint w1 = ((q1 >> (4*n)) & 0x0F0F0F0Fu) | (((h1 >> sb) & 0x01010101u) << 4);
            const float2 e0 = ds4q_h2_bits(w0 & 0x00FF00FFu, 1024.h), o0 = ds4q_h2_bits((w0 >> 8) & 0x00FF00FFu, 1024.h);
            const float2 e1 = ds4q_h2_bits(w1 & 0x00FF00FFu, 1024.h), o1 = ds4q_h2_bits((w1 >> 8) & 0x00FF00FFu, 1024.h);
            FOR_UNROLL (short t = 0; t < NT; t++) {
                const float dq = dot(e0, yl[t][2*sb].xy) + dot(o0, yl[t][2*sb].zw) +
                                 dot(e1, yl[t][2*sb + 1].xy) + dot(o1, yl[t][2*sb + 1].zw);
                s[t] += ds4q_byte_f(sc, sb) * dq;
                m[t] += ds4q_byte_f(mn, sb) * sumy[t][sb];
            }
        }
    }
    FOR_UNROLL (short t = 0; t < NT; t++) acc[t] = (float)xb->d * s[t] - (float)xb->dmin * m[t];
}

template <short NR0, short NT, typename block_t>
kernel void ds4q_mul_mv_q45_K(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/QK_K;
    const uint t0 = tgpig.y*NT;
    device const block_t *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_t *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/8, jp = (tiisg%8)/4, lr = tiisg%4;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) {
        yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*QK_K + 128*jp + 8*lr);
    }
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][8];
    float sumy[NT][4];

    for (int ib = ix; ib < nb; ib += 4) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            FOR_UNROLL (short sb = 0; sb < 4; sb++) {
                const float4 a = yb[t][8*sb], b = yb[t][8*sb + 1];
                yl[t][2*sb] = a.xzyw;
                yl[t][2*sb + 1] = b.xzyw;
                const float4 u = a + b;
                sumy[t][sb] = u[0] + u[1] + u[2] + u[3];
            }
            yb[t] += 4*QK_K/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            float acc[NT];
            ds4q_q45_K_dotn<NT>(ax[row] + ib, yl, sumy, jp, lr, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* Q6_K: eight lanes per 256-weight block, four blocks per step.  Lane k
 * covers weights 8*(k%4)..+7 of the four 32-weight groups of half k/4: the
 * low and high nibbles of ql bytes l and l+32 with two qh bits each. */
template <short NT>
static inline void ds4q_q6_K_dotn(device const block_q6_K *xb, thread const float4 (*yl)[8],
                                  short h, short lr, thread float *acc) {
    const short l = 8*lr;
    device const ushort *qa = (device const ushort *)(xb->ql + 64*h + l);
    device const ushort *qb = (device const ushort *)(xb->ql + 64*h + 32 + l);
    device const ushort *qh = (device const ushort *)(xb->qh + 32*h + l);
    device const ushort *sc = (device const ushort *)xb->scales + 4*h;
    const uint A[2] = { ds4q_u32(qa), ds4q_u32(qa + 2) };
    const uint B[2] = { ds4q_u32(qb), ds4q_u32(qb + 2) };
    const uint H[2] = { ds4q_u32(qh), ds4q_u32(qh + 2) };
    float s[NT] = {0.f};
    FOR_UNROLL (short k = 0; k < 4; k++) {
        const float fs = (float)(char)(sc[k] >> (8*(lr/2)));
        FOR_UNROLL (short i = 0; i < 2; i++) {
            const uint q = k == 0 ? (A[i] & 0x0F0F0F0Fu) | ((H[i] & 0x03030303u) << 4) :
                           k == 1 ? (B[i] & 0x0F0F0F0Fu) | ((H[i] & 0x0C0C0C0Cu) << 2) :
                           k == 2 ? ((A[i] >> 4) & 0x0F0F0F0Fu) | (H[i] & 0x30303030u) :
                                    ((B[i] >> 4) & 0x0F0F0F0Fu) | ((H[i] >> 2) & 0x30303030u);
            const float2 e = ds4q_h2_bits(q & 0x00FF00FFu, 1056.h), o = ds4q_h2_bits((q >> 8) & 0x00FF00FFu, 1056.h);
            FOR_UNROLL (short t = 0; t < NT; t++) s[t] += fs * (dot(e, yl[t][2*k + i].xy) + dot(o, yl[t][2*k + i].zw));
        }
    }
    const float d = (float)xb->d;
    FOR_UNROLL (short t = 0; t < NT; t++) acc[t] = d*s[t];
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_q6_K(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/QK_K;
    const uint t0 = tgpig.y*NT;
    device const block_q6_K *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_q6_K *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/8, h = (tiisg%8)/4, lr = tiisg%4;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) {
        yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*QK_K + 128*h + 8*lr);
    }
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][8];

    for (int ib = ix; ib < nb; ib += 4) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            FOR_UNROLL (short k = 0; k < 4; k++) {
                yl[t][2*k] = yb[t][8*k].xzyw;
                yl[t][2*k + 1] = yb[t][8*k + 1].xzyw;
            }
            yb[t] += 4*QK_K/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            float acc[NT];
            ds4q_q6_K_dotn<NT>(ax[row] + ib, yl, h, lr, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* IQ4 lookups go through a threadgroup copy of the table (128 bytes). */
static inline void ds4q_iq4_table(threadgroup float *tab, ushort tiisg, ushort sgitg) {
    if (sgitg == 0 && tiisg < 16) tab[tiisg] = ds4q_kvalues_iq4nl[tiisg];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* the eight weights of a u32 of IQ4 nibbles: bytes' low nibbles, then high */
static inline void ds4q_iq4_deq8(threadgroup const float *tab, uint q, thread float4 &lo, thread float4 &hi) {
    const uint l = q & 0x0f0f0f0fu, u = (q >> 4) & 0x0f0f0f0fu;
    lo = float4(tab[l & 0xff], tab[(l >> 8) & 0xff], tab[(l >> 16) & 0xff], tab[l >> 24]);
    hi = float4(tab[u & 0xff], tab[(u >> 8) & 0xff], tab[(u >> 16) & 0xff], tab[u >> 24]);
}

/* sixteen IQ4 weights (two u32) against inputs yl = {lo0, hi0, lo1, hi1} */
template <short NT>
static inline void ds4q_iq4_dot16(threadgroup const float *tab, uint q0, uint q1, float scale,
                                  thread const float4 (*yl)[4], thread float *acc) {
    float4 l0, h0, l1, h1;
    ds4q_iq4_deq8(tab, q0, l0, h0);
    ds4q_iq4_deq8(tab, q1, l1, h1);
    FOR_UNROLL (short t = 0; t < NT; t++) {
        acc[t] = scale * (dot(l0, yl[t][0]) + dot(h0, yl[t][1]) + dot(l1, yl[t][2]) + dot(h1, yl[t][3]));
    }
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_iq4_nl(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float *tab = (threadgroup float *)shmem;
    ds4q_iq4_table(tab, tiisg, sgitg);

    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/32;
    const uint t0 = tgpig.y*NT;
    device const block_iq4_nl *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq4_nl *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/2, it = tiisg%2;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*32 + it*8);
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][4];

    for (int ib = ix; ib < nb; ib += 16) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            yl[t][0] = yb[t][0]; yl[t][1] = yb[t][4]; yl[t][2] = yb[t][1]; yl[t][3] = yb[t][5];
            yb[t] += 16*32/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const ushort *q16 = (device const ushort *)(ax[row][ib].qs + 8*it);
            float acc[NT];
            ds4q_iq4_dot16<NT>(tab, ds4q_u32(q16), ds4q_u32(q16 + 2), (float)ax[row][ib].d, yl, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

template <short NR0, short NT>
kernel void ds4q_mul_mv_iq4_xs(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup float *tab = (threadgroup float *)shmem;
    ds4q_iq4_table(tab, tiisg, sgitg);

    const int first_row = (tgpig.x*FC_mul_mv_nsg + sgitg)*NR0;
    const int nb = args.ne00/QK_K;
    const uint t0 = tgpig.y*NT;
    device const block_iq4_xs *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const block_iq4_xs *)ds4q_mv_row(args, src0, first_row + row);
    }
    const short ix = tiisg/16, it = tiisg%16, ib = it/2, il = it%2;
    device const float4 *yb[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) {
        yb[t] = (device const float4 *)(ds4q_mv_y(args, src1, tgpig, t0 + t) + ix*QK_K + ib*32 + il*8);
    }
    float sumf[NT][NR0] = {{0.f}};
    float4 yl[NT][4];

    for (int ibl = ix; ibl < nb; ibl += 2) {
        FOR_UNROLL (short t = 0; t < NT; t++) {
            yl[t][0] = yb[t][0]; yl[t][1] = yb[t][4]; yl[t][2] = yb[t][1]; yl[t][3] = yb[t][5];
            yb[t] += 2*QK_K/4;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs &xb = ax[row][ibl];
            device const uint *q4 = (device const uint *)(xb.qs + 16*ib + 8*il);
            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            float acc[NT];
            ds4q_iq4_dot16<NT>(tab, q4[0], q4[1], (float)xb.d * ls, yl, acc);
            FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += acc[t];
        }
    }
    ds4q_mv_write<NR0, NT>(args, dst, tgpig, t0, sumf, first_row, tiisg);
}

/* BF16: the Qwen BF16 projections are wide and short (hc down is 320 rows of
 * 10240), so a threadgroup owns NR0 rows and its simdgroups split K,
 * interleaved by 4-weight words, then reduce through threadgroup memory
 * (NT * NR0 * 32 floats).  The host dispatches one threadgroup per NR0 rows
 * and NT tokens. */
template <short NR0, short NT>
kernel void ds4q_mul_mv_bf16(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const int first_row = tgpig.x*NR0;
    const uint t0 = tgpig.y*NT;
    device const float4 *y4[NT];
    FOR_UNROLL (short t = 0; t < NT; t++) y4[t] = (device const float4 *)ds4q_mv_y(args, src1, tgpig, t0 + t);

    device const ushort4 *ax[NR0];
    for (short row = 0; row < NR0; ++row) {
        ax[row] = (device const ushort4 *)ds4q_mv_row(args, src0, first_row + row);
    }

    float sumf[NT][NR0] = {{0.f}};
    if ((args.ne00 & 7) == 0) {
        /* 16-byte weight loads: eight bf16 per lane and row, the rows'
         * loads issued before any of them is used */
        const int n8 = args.ne00/8;
        for (int i = sgitg*N_SIMDWIDTH + tiisg; i < n8; i += NSG*N_SIMDWIDTH) {
            uint4 w[NR0];
            FOR_UNROLL (short row = 0; row < NR0; row++) w[row] = ((device const uint4 *)ax[row])[i];
            FOR_UNROLL (short t = 0; t < NT; t++) {
                const float4 y0 = y4[t][2*i], y1 = y4[t][2*i + 1];
                FOR_UNROLL (short row = 0; row < NR0; row++) {
                    const uint4 q = w[row];
                    const float4 w0 = as_type<float4>(uint4(q.x << 16, q.x & 0xFFFF0000u, q.y << 16, q.y & 0xFFFF0000u));
                    const float4 w1 = as_type<float4>(uint4(q.z << 16, q.z & 0xFFFF0000u, q.w << 16, q.w & 0xFFFF0000u));
                    sumf[t][row] += dot(w0, y0) + dot(w1, y1);
                }
            }
        }
    } else {
        const int n4 = args.ne00/4;
        for (int i = sgitg*N_SIMDWIDTH + tiisg; i < n4; i += NSG*N_SIMDWIDTH) {
            float4 y[NT];
            FOR_UNROLL (short t = 0; t < NT; t++) y[t] = y4[t][i];
            FOR_UNROLL (short row = 0; row < NR0; row++) {
                const float4 w = as_type<float4>(uint4(ax[row][i]) << 16);
                FOR_UNROLL (short t = 0; t < NT; t++) sumf[t][row] += dot(w, y[t]);
            }
        }
    }

    threadgroup float *red = (threadgroup float *)shmem;
    FOR_UNROLL (short t = 0; t < NT; t++) {
        for (short row = 0; row < NR0; ++row) {
            const float v = simd_sum(sumf[t][row]);
            if (tiisg == 0) red[(t*NR0 + row)*N_SIMDWIDTH + sgitg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0) return;
    device float *d = (device float *)dst + (uint64_t)tgpig.z*args.ne0*args.ne1;
    FOR_UNROLL (short t = 0; t < NT; t++) {
        for (short row = 0; row < NR0; ++row) {
            const float tot = simd_sum(tiisg < NSG ? red[(t*NR0 + row)*N_SIMDWIDTH + tiisg] : 0.f);
            if (tiisg == 0 && t0 + t < (uint)args.ne11 && first_row + row < args.ne01) {
                d[(uint64_t)(t0 + t)*args.ne0 + first_row + row] = tot;
            }
        }
    }
}

typedef decltype(ds4q_mul_mv_bf16<2, 1>) ds4q_mul_mv_t;

#define DS4Q_STR_(x) #x
#define DS4Q_STR(x) DS4Q_STR_(x)

/* one, two and three tokens; the row counts must match the host (nr0, and
 * half of it at three tokens when nr0 > 4) */
#define DS4Q_MV(name, kern, r1, r3) \
template [[host_name(DS4Q_STR(kernel_mul_mv_##name##_dense_f32))]]     kernel ds4q_mul_mv_t kern<r1, 1>; \
template [[host_name(DS4Q_STR(kernel_mul_mv_##name##_dense_f32_nt2))]] kernel ds4q_mul_mv_t kern<r1, 2>; \
template [[host_name(DS4Q_STR(kernel_mul_mv_##name##_dense_f32_nt3))]] kernel ds4q_mul_mv_t kern<r3, 3>;

DS4Q_MV(q2_0, ds4q_mul_mv_q2_0, 8, 4)
DS4Q_MV(q5_0, ds4q_mul_mv_q5_0, 4, 4)
DS4Q_MV(iq4_nl, ds4q_mul_mv_iq4_nl, 2, 2)
DS4Q_MV(q3_K, ds4q_mul_mv_q3_K, 2, 2)
DS4Q_MV(q6_K, ds4q_mul_mv_q6_K, 2, 2)
DS4Q_MV(iq4_xs, ds4q_mul_mv_iq4_xs, 2, 2)
template [[host_name("kernel_mul_mv_q4_K_mixed_f32")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 1, block_q4_K>;
template [[host_name("kernel_mul_mv_q4_K_mixed_f32_nt2")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 2, block_q4_K>;
template [[host_name("kernel_mul_mv_q4_K_mixed_f32_nt3")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 3, block_q4_K>;
template [[host_name("kernel_mul_mv_q5_K_dense_f32")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 1, block_q5_K>;
template [[host_name("kernel_mul_mv_q5_K_dense_f32_nt2")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 2, block_q5_K>;
template [[host_name("kernel_mul_mv_q5_K_dense_f32_nt3")]] kernel ds4q_mul_mv_t ds4q_mul_mv_q45_K<2, 3, block_q5_K>;
template [[host_name("kernel_mul_mv_bf16_dense_f32")]]     kernel ds4q_mul_mv_t ds4q_mul_mv_bf16<2, 1>;
template [[host_name("kernel_mul_mv_bf16_dense_f32_nt2")]] kernel ds4q_mul_mv_t ds4q_mul_mv_bf16<2, 2>;
template [[host_name("kernel_mul_mv_bf16_dense_f32_nt3")]] kernel ds4q_mul_mv_t ds4q_mul_mv_bf16<2, 3>;

/* --- 2..5 token matvec (mul_mv_ext template of dense.metal) -------------- */


#define DS4Q_MV_EXT(name, block_t, epb, deq)                                                                                               \
template [[host_name(DS4Q_STR(kernel_mul_mv_ext_##name##_f32_r1_2))]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_t, epb, deq>; \
template [[host_name(DS4Q_STR(kernel_mul_mv_ext_##name##_f32_r1_3))]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_t, epb, deq>; \
template [[host_name(DS4Q_STR(kernel_mul_mv_ext_##name##_f32_r1_4))]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_t, epb, deq>; \
template [[host_name(DS4Q_STR(kernel_mul_mv_ext_##name##_f32_r1_5))]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_t, epb, deq>;

DS4Q_MV_EXT(q2_0,   block_q2_0,   64,  ds4q_dequantize_q2_0_t4)
DS4Q_MV_EXT(q5_0,   block_q5_0,   32,  ds4q_dequantize_q5_0_t4)
DS4Q_MV_EXT(iq4_nl, block_iq4_nl, 32,  ds4q_dequantize_iq4_nl_t4)
DS4Q_MV_EXT(q3_K,   block_q3_K,   256, ds4q_dequantize_q3_K_t4)
DS4Q_MV_EXT(q5_K,   block_q5_K,   256, ds4q_dequantize_q5_K_t4)
DS4Q_MV_EXT(q6_K,   block_q6_K,   256, ds4q_dequantize_q6_K_t4)
DS4Q_MV_EXT(bf16,   ushort4,      4,   ds4q_dequantize_bf16_t4)

/* --- prompt matmul (kernel_mul_mm of dense.metal) ------------------------ */

#define DS4Q_MM(name, block_t, nl, deq)                                                        \
template [[host_name(DS4Q_STR(kernel_mul_mm_##name##_f32))]] kernel mul_mm_t kernel_mul_mm<half, half4x4, simdgroup_half8x8, \
    half, half2x4, simdgroup_half8x8, block_t, nl, deq, float, float4x4, float, float2x4>;

DS4Q_MM(q2_0,   block_q2_0,   4,  ds4q_dequantize_q2_0)
DS4Q_MM(q5_0,   block_q5_0,   2,  ds4q_dequantize_q5_0)
DS4Q_MM(iq4_nl, block_iq4_nl, 2,  ds4q_dequantize_iq4_nl)
DS4Q_MM(q3_K,   block_q3_K,   16, ds4q_dequantize_q3_K)
DS4Q_MM(q5_K,   block_q5_K,   16, dequantize_q5_K)
DS4Q_MM(q6_K,   block_q6_K,   16, dequantize_q6_K)
DS4Q_MM(iq4_xs, block_iq4_xs, 16, ds4q_dequantize_iq4_xs)
DS4Q_MM(bf16,   ds4q_bf16x16, 1,  ds4q_dequantize_bf16)

#undef QK2_0
#undef QK_K
