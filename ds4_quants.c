/* CPU dequantization of GGUF rows, ported from llama.cpp ggml-quants.c (MIT).
 * Blocks are read by byte offset, so no struct packing is assumed. */

#include "ds4_quants.h"

#include <string.h>

enum {
    Q_F32 = 0, Q_F16 = 1, Q_Q4_0 = 2, Q_Q5_0 = 6, Q_Q8_0 = 8, Q_Q3_K = 11,
    Q_Q4_K = 12, Q_Q5_K = 13, Q_Q6_K = 14, Q_IQ2_XS = 17, Q_IQ3_XXS = 18,
    Q_IQ4_NL = 20, Q_IQ3_S = 21, Q_IQ2_S = 22, Q_IQ4_XS = 23,
    Q_BF16 = 30, Q_Q2_0 = 42,
};

#include "ds4_iq_grids.inc"

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float half_at(const uint8_t *p) {
    return f16_to_f32((uint16_t)(p[0] | (p[1] << 8)));
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* Q2_0: f16 d, 16 bytes; element j sits in byte j/4 at bits (j%4)*2. */
static void deq_q2_0(const uint8_t *b, float *y) {
    const float d = half_at(b);
    for (int j = 0; j < 64; j++) y[j] = (float)((int)((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
}

static void deq_q4_0(const uint8_t *b, float *y) {
    const float d = half_at(b);
    for (int j = 0; j < 16; j++) {
        y[j] = (float)((b[2 + j] & 0x0F) - 8) * d;
        y[j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
    }
}

static void deq_q5_0(const uint8_t *b, float *y) {
    const float d = half_at(b);
    uint32_t qh;
    memcpy(&qh, b + 2, sizeof(qh));
    const uint8_t *qs = b + 6;
    for (int j = 0; j < 16; j++) {
        const uint8_t xh_0 = ((qh >> j) << 4) & 0x10;
        const uint8_t xh_1 = (qh >> (j + 12)) & 0x10;
        y[j] = (float)(((qs[j] & 0x0F) | xh_0) - 16) * d;
        y[j + 16] = (float)(((qs[j] >> 4) | xh_1) - 16) * d;
    }
}

static void deq_q8_0(const uint8_t *b, float *y) {
    const float d = half_at(b);
    for (int j = 0; j < 32; j++) y[j] = (float)(int8_t)b[2 + j] * d;
}

static void deq_iq4_nl(const uint8_t *b, float *y) {
    const float d = half_at(b);
    for (int j = 0; j < 16; j++) {
        y[j] = d * kvalues_iq4nl[b[2 + j] & 0xf];
        y[j + 16] = d * kvalues_iq4nl[b[2 + j] >> 4];
    }
}

/* Q3_K: hmask[32], qs[64], scales[12], f16 d. */
static void deq_q3_K(const uint8_t *b, float *y) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const uint8_t *hm = b, *q = b + 32;
    const float d_all = half_at(b + 108);
    uint32_t aux[4];
    memcpy(aux, b + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int8_t scales[16];
    memcpy(scales, aux, sizeof(scales));
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

/* Q4_K: f16 d, f16 dmin, scales[12], qs[128]. */
static void deq_q4_K(const uint8_t *b, float *y) {
    const float d = half_at(b), min = half_at(b + 2);
    const uint8_t *sc = b + 4, *q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m);
        const float d1 = d * s, m1 = min * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

/* Q5_K: f16 d, f16 dmin, scales[12], qh[32], qs[128]. */
static void deq_q5_K(const uint8_t *b, float *y) {
    const float d = half_at(b), min = half_at(b + 2);
    const uint8_t *sc = b + 4, *qh = b + 16, *ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        get_scale_min_k4(is, sc, &s, &m);
        const float d1 = d * s, m1 = min * m;
        get_scale_min_k4(is + 1, sc, &s, &m);
        const float d2 = d * s, m2 = min * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (float)((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/* Q6_K: ql[128], qh[64], int8 scales[16], f16 d. */
static void deq_q6_K(const uint8_t *b, float *y) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const float d = half_at(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

/* IQ4_XS: f16 d, u16 scales_h, scales_l[4], qs[128]. */
static void deq_iq4_xs(const uint8_t *b, float *y) {
    const float d = half_at(b);
    const uint16_t scales_h = (uint16_t)(b[2] | (b[3] << 8));
    const uint8_t *scales_l = b + 4, *qs = b + 8;
    for (int ib = 0; ib < 8; ib++) {
        const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; j++) {
            y[j] = dl * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

/* Eight weights of a codebook quant: grid bytes scaled by dl, bit j of
 * signs negates weight j. */
static void grid8(const uint8_t *grid, uint8_t signs, float dl, float *y) {
    for (int j = 0; j < 8; j++) y[j] = dl * (float)grid[j] * ((signs >> j) & 1 ? -1.0f : 1.0f);
}

/* The IQ3 grids hold four weights per entry: two entries make eight. */
static void grid4x2(const uint32_t *g1, const uint32_t *g2, uint8_t signs, float dl, float *y) {
    uint8_t grid[8];
    memcpy(grid, g1, 4);
    memcpy(grid + 4, g2, 4);
    grid8(grid, signs, dl, y);
}

/* IQ2_XS: f16 d, u16 qs[32] (9-bit grid index, 7-bit sign index), scales[8]. */
static void deq_iq2_xs(const uint8_t *b, float *y) {
    const float d = half_at(b);
    const uint8_t *qs = b + 2, *scales = b + 66;
    for (int ib = 0; ib < 8; ib++) {
        for (int l = 0; l < 4; l++) {
            const uint16_t q = (uint16_t)(qs[8 * ib + 2 * l] | (qs[8 * ib + 2 * l + 1] << 8));
            const float dl = d * (0.5f + (float)((scales[ib] >> 4 * (l / 2)) & 0xf)) * 0.25f;
            grid8((const uint8_t *)(iq2xs_grid + (q & 511)), ksigns_iq2xs[q >> 9], dl, y);
            y += 8;
        }
    }
}

/* IQ2_S: f16 d, qs[32], signs[32], qh[8], scales[8]. */
static void deq_iq2_s(const uint8_t *b, float *y) {
    const float d = half_at(b);
    const uint8_t *qs = b + 2, *signs = b + 34, *qh = b + 66, *scales = b + 74;
    for (int ib = 0; ib < 8; ib++) {
        for (int l = 0; l < 4; l++) {
            const float dl = d * (0.5f + (float)((scales[ib] >> 4 * (l / 2)) & 0xf)) * 0.25f;
            const uint32_t idx = qs[4 * ib + l] | ((uint32_t)(qh[ib] << (8 - 2 * l)) & 0x300);
            grid8((const uint8_t *)(iq2s_grid + idx), signs[4 * ib + l], dl, y);
            y += 8;
        }
    }
}

/* IQ3_XXS: f16 d, qs[64], then per 32 weights a u32 of 4 x 7 sign bits and
 * a 4-bit scale. */
static void deq_iq3_xxs(const uint8_t *b, float *y) {
    const float d = half_at(b);
    const uint8_t *qs = b + 2, *gas = b + 66;
    for (int ib = 0; ib < 8; ib++) {
        uint32_t aux;
        memcpy(&aux, gas + 4 * ib, sizeof(aux));
        const float dl = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
            grid4x2(iq3xxs_grid + qs[8 * ib + 2 * l], iq3xxs_grid + qs[8 * ib + 2 * l + 1],
                    ksigns_iq2xs[(aux >> 7 * l) & 127], dl, y);
            y += 8;
        }
    }
}

/* IQ3_S: f16 d, qs[64], qh[8], signs[32], scales[4]. */
static void deq_iq3_s(const uint8_t *b, float *y) {
    const float d = half_at(b);
    const uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
    for (int ib = 0; ib < 8; ib++) {
        const float dl = d * (float)(1 + 2 * ((scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
        for (int l = 0; l < 4; l++) {
            const uint32_t i1 = qs[8 * ib + 2 * l] | ((uint32_t)(qh[ib] << (8 - 2 * l)) & 256);
            const uint32_t i2 = qs[8 * ib + 2 * l + 1] | ((uint32_t)(qh[ib] << (7 - 2 * l)) & 256);
            grid4x2(iq3s_grid + i1, iq3s_grid + i2, signs[4 * ib + l], dl, y);
            y += 8;
        }
    }
}

typedef struct {
    uint32_t type;
    uint32_t block;
    uint32_t bytes;
    void (*deq)(const uint8_t *, float *);
} quant_info;

static const quant_info quant_infos[] = {
    { Q_Q2_0,    64,  18, deq_q2_0 },
    { Q_Q4_0,    32,  18, deq_q4_0 },
    { Q_Q5_0,    32,  22, deq_q5_0 },
    { Q_Q8_0,    32,  34, deq_q8_0 },
    { Q_IQ4_NL,  32,  18, deq_iq4_nl },
    { Q_Q3_K,   256, 110, deq_q3_K },
    { Q_Q4_K,   256, 144, deq_q4_K },
    { Q_Q5_K,   256, 176, deq_q5_K },
    { Q_Q6_K,   256, 210, deq_q6_K },
    { Q_IQ4_XS, 256, 136, deq_iq4_xs },
    { Q_IQ2_XS, 256,  74, deq_iq2_xs },
    { Q_IQ2_S,  256,  82, deq_iq2_s },
    { Q_IQ3_XXS,256,  98, deq_iq3_xxs },
    { Q_IQ3_S,  256, 110, deq_iq3_s },
};

static const quant_info *quant_find(uint32_t type) {
    for (size_t i = 0; i < sizeof(quant_infos) / sizeof(quant_infos[0]); i++) {
        if (quant_infos[i].type == type) return &quant_infos[i];
    }
    return NULL;
}

bool ds4_quant_row_bytes(uint32_t type, uint64_t n, uint64_t *bytes) {
    if (type == Q_F32 || type == Q_F16 || type == Q_BF16) {
        *bytes = n * (type == Q_F32 ? 4u : 2u);
        return true;
    }
    const quant_info *q = quant_find(type);
    if (!q || n % q->block) return false;
    *bytes = n / q->block * q->bytes;
    return true;
}

bool ds4_dequant_row(uint32_t type, const void *src, float *dst, uint64_t n) {
    const uint8_t *p = src;
    if (type == Q_F32) {
        memcpy(dst, src, n * sizeof(float));
        return true;
    }
    if (type == Q_F16 || type == Q_BF16) {
        for (uint64_t i = 0; i < n; i++) {
            const uint16_t h = (uint16_t)(p[2 * i] | (p[2 * i + 1] << 8));
            if (type == Q_F16) {
                dst[i] = f16_to_f32(h);
            } else {
                const uint32_t bits = (uint32_t)h << 16;
                memcpy(&dst[i], &bits, sizeof(bits));
            }
        }
        return true;
    }
    const quant_info *q = quant_find(type);
    if (!q || n % q->block) return false;
    for (uint64_t i = 0; i < n / q->block; i++) q->deq(p + i * q->bytes, dst + i * q->block);
    return true;
}
