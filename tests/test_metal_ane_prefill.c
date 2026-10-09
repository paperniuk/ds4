#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The Neural Engine prefill path of ds4_metal.m against a CPU reference:
 * int8 projections staged by the GPU from Q8_0 rows (two layers, a full and
 * a partial row bucket), a split projection joined with GPU columns, the
 * SwiGLU FFN with fp16 weights, and a fault that must surface in settle. */

enum { K = 256, N = 512, F = 128, R = 1024, SLOTS = 2, GPU_COLS = 192, Q8_BLOCK = 34 };
static uint32_t rng = 7;
static float random_unit(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)(rng % 2001u) / 1000.0f - 1.0f;
}

/* Q8_0 [rows][cols] at map + off: random weights, one fp16 scale per 32. */
static void fill_q8(uint8_t *p, uint32_t rows, uint32_t cols, float *ref) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t b = 0; b < cols / 32u; b++, p += Q8_BLOCK) {
            const _Float16 d = (_Float16)(0.004f + 0.002f * (random_unit() + 1.0f));
            memcpy(p, &d, sizeof d);
            for (uint32_t i = 0; i < 32; i++) {
                const int8_t q = (int8_t)lrintf(127.0f * random_unit());
                p[2 + i] = (uint8_t)q;
                ref[(uint64_t)r * cols + b * 32u + i] = (float)d * q;
            }
        }
    }
}

static void matmul(float *y, const float *x, const float *w, uint32_t T, uint32_t k, uint32_t n) {
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t j = 0; j < n; j++) {
            double s = 0.0;
            for (uint32_t i = 0; i < k; i++) s += (double)x[(uint64_t)t * k + i] * w[(uint64_t)j * k + i];
            y[(uint64_t)t * n + j] = (float)s;
        }
}

static int close_enough(const char *what, const float *a, const float *ref, uint64_t n) {
    double d = 0.0, r = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i])) {
            fprintf(stderr, "%s: element %llu is not finite\n", what, (unsigned long long)i);
            return 0;
        }
        d += ((double)a[i] - ref[i]) * ((double)a[i] - ref[i]);
        r += (double)ref[i] * ref[i];
    }
    const double err = sqrt(d / r);
    if (err > 0.02) {
        fprintf(stderr, "%s: relative error %.4f\n", what, err);
        return 0;
    }
    printf("PASS %s, relative error %.4f\n", what, err);
    return 1;
}

static ds4_gpu_ane_weight weight(const void *map, uint64_t size, uint64_t off, uint32_t cols, uint32_t r0) {
    return (ds4_gpu_ane_weight){ map, size, off, (uint64_t)cols / 32u * Q8_BLOCK, 8u, r0 };
}

int main(void) {
    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint64_t proj = (uint64_t)N * K / 32u * Q8_BLOCK, ffn = (uint64_t)F * K / 32u * Q8_BLOCK;
    const uint64_t bytes = (SLOTS * proj + SLOTS * 3u * ffn + page - 1u) / page * page;
    float *wp = malloc((uint64_t)SLOTS * N * K * sizeof(float));
    float *wf = malloc((uint64_t)SLOTS * 3u * F * K * sizeof(float));
    float *x = malloc((uint64_t)R * K * sizeof(float));
    float *y = malloc((uint64_t)R * N * sizeof(float)), *ref = malloc((uint64_t)R * N * sizeof(float));
    float *g = malloc((uint64_t)R * F * sizeof(float)), *u = malloc((uint64_t)R * F * sizeof(float));
    void *map = NULL;
    if (!wp || !wf || !x || !y || !ref || !g || !u || posix_memalign(&map, page, bytes) != 0) return 1;
    uint8_t *m = map;
    for (uint32_t s = 0; s < SLOTS; s++) fill_q8(m + s * proj, N, K, wp + (uint64_t)s * N * K);
    for (uint32_t s = 0; s < SLOTS; s++) {
        const uint64_t base = SLOTS * proj + (uint64_t)s * 3u * ffn;
        float *ws = wf + (uint64_t)s * 3u * F * K;
        fill_q8(m + base, F, K, ws);                         /* gate [F][K] */
        fill_q8(m + base + ffn, F, K, ws + (uint64_t)F * K); /* up [F][K] */
        fill_q8(m + base + 2u * ffn, K, F, ws + 2u * (uint64_t)F * K);  /* down [K][F] */
    }
    for (uint64_t i = 0; i < (uint64_t)R * K; i++) x[i] = random_unit();
    /* seven good evaluations, then a non-finite one */
    setenv("DS4_ANE_TEST_FAIL", "8:nan", 1);

    int ok = ds4_gpu_init() && ds4_gpu_set_model_map(map, bytes);
    const int hp = ok ? ds4_gpu_ane_create(K, N, R, SLOTS) : -1;
    const int hj = ok ? ds4_gpu_ane_create(K, N - GPU_COLS, R, 1) : -1;
    const int hf = ok ? ds4_gpu_ane_create_ffn(K, F, R, SLOTS) : -1;
    if (hp < 0 || hj < 0 || hf < 0) {
        printf("SKIP Neural Engine unavailable\n");
        return 0;
    }
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)R * K * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)R * N * sizeof(float));
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc((uint64_t)R * GPU_COLS * sizeof(float));
    ok = xt && out && part && ds4_gpu_tensor_write(xt, 0, x, (uint64_t)R * K * sizeof(float));
    for (uint32_t s = 0; s < SLOTS && ok; s++) {
        const ds4_gpu_ane_weight w = weight(map, bytes, s * proj, K, 0);
        ok = ds4_gpu_ane_measure(hp, s, &w);
    }
    const ds4_gpu_ane_weight wj = weight(map, bytes, 0, K, GPU_COLS);
    ok = ok && ds4_gpu_ane_measure(hj, 0, &wj) && ds4_gpu_ane_measure_done();

    /* projections: both layers through one surface, 1024 and 700 rows */
    const uint32_t rows[2] = { R, 700 };
    for (uint32_t s = 0; s < SLOTS && ok; s++) {
        for (int c = 0; c < 2 && ok; c++) {
            const uint32_t T = rows[c];
            const ds4_gpu_ane_weight w = weight(map, bytes, s * proj, K, 0);
            ok = ds4_gpu_begin_commands() && ds4_gpu_ane_stage(hp, s, 0, &w) && ds4_gpu_ane_pack(hp, xt, T) &&
                 ds4_gpu_ane_eval(hp, s, hp, T) && ds4_gpu_ane_unpack(hp, s, out, T, N, 0, NULL, 0);
            ok = ds4_gpu_end_commands() && ok && ds4_gpu_ane_settle() &&
                 ds4_gpu_tensor_read(out, 0, y, (uint64_t)T * N * sizeof(float));
            matmul(ref, x, wp + (uint64_t)s * N * K, T, K, N);
            char what[96];
            snprintf(what, sizeof what, "ANE projection %ux%u, layer %u, %u rows", K, N, s, T);
            ok = ok && close_enough(what, y, ref, (uint64_t)T * N);
        }
    }

    /* split: the GPU's leading columns from part, the ANE's from row 192 on */
    if (ok) {
        const uint32_t T = 512;
        float *gp = malloc((uint64_t)T * GPU_COLS * sizeof(float));
        ok = gp != NULL;
        for (uint64_t i = 0; ok && i < (uint64_t)T * GPU_COLS; i++) gp[i] = random_unit();
        ok = ok && ds4_gpu_tensor_write(part, 0, gp, (uint64_t)T * GPU_COLS * sizeof(float)) &&
             ds4_gpu_begin_commands() && ds4_gpu_ane_stage(hj, 0, 0, &wj) && ds4_gpu_ane_pack(hj, xt, T) &&
             ds4_gpu_ane_eval(hj, 0, hj, T) && ds4_gpu_ane_unpack(hj, 0, out, T, N, GPU_COLS, part, GPU_COLS);
        ok = ds4_gpu_end_commands() && ok && ds4_gpu_ane_settle() &&
             ds4_gpu_tensor_read(out, 0, y, (uint64_t)T * N * sizeof(float));
        matmul(ref, x, wp, T, K, N);
        for (uint32_t t = 0; ok && t < T; t++) {
            if (memcmp(y + (uint64_t)t * N, gp + (uint64_t)t * GPU_COLS, GPU_COLS * sizeof(float)) != 0) {
                fprintf(stderr, "split: GPU columns of row %u changed\n", t);
                ok = 0;
            }
            memcpy(ref + (uint64_t)t * N, gp + (uint64_t)t * GPU_COLS, GPU_COLS * sizeof(float));
        }
        ok = ok && close_enough("ANE split projection joined after 192 GPU columns", y, ref, (uint64_t)T * N);
        free(gp);
    }

    /* SwiGLU FFN, both layers */
    for (uint32_t s = 0; s < SLOTS && ok; s++) {
        const uint32_t T = 600;
        const uint64_t base = SLOTS * proj + (uint64_t)s * 3u * ffn;
        const ds4_gpu_ane_weight wg = weight(map, bytes, base, K, 0), wu = weight(map, bytes, base + ffn, K, 0);
        const ds4_gpu_ane_weight wd = weight(map, bytes, base + 2u * ffn, F, 0);
        ok = ds4_gpu_begin_commands() && ds4_gpu_ane_stage(hf, s, 0, &wg) && ds4_gpu_ane_stage(hf, s, 1, &wu) &&
             ds4_gpu_ane_stage(hf, s, 2, &wd) && ds4_gpu_ane_pack(hf, xt, T) && ds4_gpu_ane_eval(hf, s, hf, T) &&
             ds4_gpu_ane_unpack(hf, s, out, T, K, 0, NULL, 0);
        ok = ds4_gpu_end_commands() && ok && ds4_gpu_ane_settle() &&
             ds4_gpu_tensor_read(out, 0, y, (uint64_t)T * K * sizeof(float));
        const float *ws = wf + (uint64_t)s * 3u * F * K;
        matmul(g, x, ws, T, K, F);
        matmul(u, x, ws + (uint64_t)F * K, T, K, F);
        for (uint64_t i = 0; i < (uint64_t)T * F; i++) g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        matmul(ref, g, ws + 2u * (uint64_t)F * K, T, F, K);
        char what[64];
        snprintf(what, sizeof what, "ANE SwiGLU FFN %u -> %u, layer %u", K, F, s);
        ok = ok && close_enough(what, y, ref, (uint64_t)T * K);
    }

    /* the eighth evaluation returns non-finite rows: settle reports it */
    if (ok) {
        const ds4_gpu_ane_weight w = weight(map, bytes, 0, K, 0);
        ok = ds4_gpu_begin_commands() && ds4_gpu_ane_stage(hp, 0, 0, &w) && ds4_gpu_ane_pack(hp, xt, 512) &&
             ds4_gpu_ane_eval(hp, 0, hp, 512) && ds4_gpu_ane_unpack(hp, 0, out, 512, N, 0, NULL, 0);
        ok = ds4_gpu_end_commands() && ok;
        if (ok && (ds4_gpu_ane_settle() || !ds4_gpu_ane_failed() || ds4_gpu_ane_pack(hp, xt, 512))) {
            fprintf(stderr, "a non-finite evaluation did not stop the Neural Engine\n");
            ok = 0;
        }
        if (ok) printf("PASS non-finite ANE rows stop the Neural Engine\n");
    }
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(part);
    ds4_gpu_cleanup();
    free(map); free(wp); free(wf); free(x); free(y); free(ref); free(g); free(u);
    return ok ? 0 : 1;
}
