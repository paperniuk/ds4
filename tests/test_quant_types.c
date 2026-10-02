/* GPU checks for the mixed-precision weight types (Q2_0, Q5_0, IQ4_NL, Q3_K,
 * Q5_K, Q6_K, IQ4_XS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, BF16): the generic quant matmul at 1..64 tokens and the
 * Qwen row dot (multi GEMV), against a CPU reference built on ds4_quants.c.
 * Build: make test-quant-types */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "ds4_gpu.h"
#include "ds4_quants.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

static uint32_t g_rng = 0x9e3779b9u;

static uint32_t urand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static float frand(void) {
    return ((float)(urand() & 0xffffffu) / 8388608.0f) - 1.0f;
}

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } v = { f };
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void put_f16(uint8_t *p, float f) {
    const uint16_t h = f32_to_f16(f);
    p[0] = (uint8_t)h;
    p[1] = (uint8_t)(h >> 8);
}

typedef struct {
    const char *name;
    uint32_t type;
    uint32_t block;
    uint32_t bytes;
    int d_off;     /* f16 scale offset in the block */
    int dmin_off;  /* f16 min offset, or -1 */
} quant_type;

static const quant_type types[] = {
    { "q2_0",   42, 64,  18,  0,   -1 },
    { "q5_0",   6,  32,  22,  0,   -1 },
    { "iq4_nl", 20, 32,  18,  0,   -1 },
    { "q3_K",   11, 256, 110, 108, -1 },
    { "q5_K",   13, 256, 176, 0,   2  },
    { "q6_K",   14, 256, 210, 208, -1 },
    { "iq4_xs", 23, 256, 136, 0,   -1 },
    { "iq2_xs", 17, 256, 74,  0,   -1 },
    { "iq2_s",  22, 256, 82,  0,   -1 },
    { "iq3_xxs",18, 256, 98,  0,   -1 },
    { "iq3_s",  21, 256, 110, 0,   -1 },
    { "bf16",   30, 1,   2,   -1,  -1 },
};

/* Bench-only references: the older DS4 kernels. */
static const quant_type ref_types[] = {
    { "q8_0",   8,  32,  34,  0,   -1 },
    { "q4_K",   12, 256, 144, 0,   2  },
};

/* Random quant bits with sane scales, so every code path of a block is hit
 * while the dot products stay O(1). */
static void fill_rows(const quant_type *q, uint8_t *w, uint64_t rows, uint64_t in_dim) {
    const uint64_t nblocks = rows * in_dim / q->block;
    if (q->type == 30) {
        for (uint64_t i = 0; i < nblocks; i++) {
            union { float f; uint32_t u; } v = { 0.05f * frand() };
            w[2 * i] = (uint8_t)(v.u >> 16);
            w[2 * i + 1] = (uint8_t)(v.u >> 24);
        }
        return;
    }
    const float d_scale = q->type == 14 ? 0.002f / 32.0f : q->block == 256 ? 0.002f : 0.02f;
    for (uint64_t b = 0; b < nblocks; b++) {
        uint8_t *blk = w + b * q->bytes;
        for (uint32_t i = 0; i < q->bytes; i++) blk[i] = (uint8_t)urand();
        put_f16(blk + q->d_off, d_scale * (0.5f + 0.5f * frand()));
        if (q->dmin_off >= 0) put_f16(blk + q->dmin_off, d_scale * (0.5f + 0.5f * frand()));
    }
}

static void reference(const quant_type *q, const uint8_t *w, const float *x, double *ref,
                      uint64_t rows, uint64_t in_dim, uint64_t n_tok) {
    uint64_t row_bytes = 0;
    require_ok(ds4_quant_row_bytes(q->type, in_dim, &row_bytes), "row bytes");
    float *deq = malloc(in_dim * sizeof(float));
    for (uint64_t r = 0; r < rows; r++) {
        require_ok(ds4_dequant_row(q->type, w + r * row_bytes, deq, in_dim), "cpu dequant");
        for (uint64_t t = 0; t < n_tok; t++) {
            double acc = 0.0;
            for (uint64_t k = 0; k < in_dim; k++) acc += (double)deq[k] * x[t * in_dim + k];
            ref[t * rows + r] = acc;
        }
    }
    free(deq);
}

static int check_close(const char *what, const float *got, const double *ref, uint64_t n, double tol) {
    double worst = 0.0, scale = 1e-6;
    uint64_t worst_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(got[i])) {
            fprintf(stderr, "%s: non-finite value at %llu\n", what, (unsigned long long)i);
            return 0;
        }
        const double d = fabs((double)got[i] - ref[i]);
        if (d > worst) { worst = d; worst_i = i; }
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    if (worst > tol * scale) {
        fprintf(stderr, "  %-40s FAIL rel %.3e at %llu: got %.6f ref %.6f\n",
                what, worst / scale, (unsigned long long)worst_i, got[worst_i], ref[worst_i]);
        return 0;
    }
    printf("  %-40s ok  rel %.2e\n", what, worst / scale);
    return 1;
}

static ds4_gpu_tensor *upload(const float *data, uint64_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n * sizeof(float));
    require_ok(t != NULL, "tensor alloc");
    if (data) require_ok(ds4_gpu_tensor_write(t, 0, data, n * sizeof(float)), "tensor write");
    else require_ok(ds4_gpu_tensor_fill_f32(t, 0.0f, n), "tensor fill");
    return t;
}

typedef struct {
    uint8_t *base;
    uint64_t size;
    uint64_t used;
} arena_t;

/* Weights are placed with a page of slack after them, as in a real GGUF
 * where other tensors follow. */
static uint64_t arena_alloc(arena_t *a, uint64_t bytes) {
    const uint64_t off = (a->used + 4095u) & ~4095ull;
    require_ok(off + bytes + 4096u <= a->size, "arena space");
    a->used = off + bytes + 4096u;
    return off;
}

static int test_shape(arena_t *a, const quant_type *q, uint64_t in_dim, uint64_t rows) {
    static const uint64_t n_toks[] = { 1, 2, 3, 5, 8, 9, 32, 33, 64 };
    const uint64_t max_tok = 64;
    uint64_t row_bytes = 0;
    require_ok(ds4_quant_row_bytes(q->type, in_dim, &row_bytes), "row bytes");
    const uint64_t off = arena_alloc(a, rows * row_bytes);
    uint8_t *w = a->base + off;
    fill_rows(q, w, rows, in_dim);

    float *x = malloc(max_tok * in_dim * sizeof(float));
    for (uint64_t i = 0; i < max_tok * in_dim; i++) x[i] = frand();
    double *ref = malloc(max_tok * rows * sizeof(double));
    reference(q, w, x, ref, rows, in_dim, max_tok);
    float *got = malloc(max_tok * rows * sizeof(float));

    int ok = 1;
    for (size_t i = 0; i < sizeof(n_toks) / sizeof(n_toks[0]); i++) {
        const uint64_t T = n_toks[i];
        ds4_gpu_tensor *gx = upload(x, T * in_dim);
        ds4_gpu_tensor *gout = upload(NULL, T * rows);
        char what[96];
        snprintf(what, sizeof(what), "%s %llux%llu T=%llu", q->name,
                 (unsigned long long)in_dim, (unsigned long long)rows, (unsigned long long)T);
        if (!ds4_gpu_matmul_quant_tensor(gout, a->base, a->size, off, q->type, in_dim, rows, gx, T)) {
            fprintf(stderr, "  %-40s FAIL dispatch\n", what);
            ok = 0;
        } else {
            require_ok(ds4_gpu_tensor_read(gout, 0, got, T * rows * sizeof(float)), "tensor read");
            ok &= check_close(what, got, ref, T * rows, T <= 8 ? 2e-3 : 1e-2);
        }
        ds4_gpu_tensor_free(gx);
        ds4_gpu_tensor_free(gout);
    }

    for (uint32_t T = 1; T <= 2; T++) {
        if (q->type == 30) break; /* BF16 row dot predates this file */
        ds4_gpu_tensor *gx = upload(x, T * in_dim);
        ds4_gpu_tensor *gout = upload(NULL, T * rows);
        const uint32_t type = q->type, out_rows = (uint32_t)rows;
        char what[96];
        snprintf(what, sizeof(what), "%s row dot %llux%llu T=%u", q->name,
                 (unsigned long long)in_dim, (unsigned long long)rows, T);
        if (!ds4_gpu_qwen4_multi_gemv_tensor(gx, T, (uint32_t)in_dim, 1, &gout, a->base, a->size,
                                             &off, &type, &out_rows)) {
            fprintf(stderr, "  %-40s FAIL dispatch\n", what);
            ok = 0;
        } else {
            require_ok(ds4_gpu_tensor_read(gout, 0, got, T * rows * sizeof(float)), "tensor read");
            ok &= check_close(what, got, ref, T * rows, 2e-3);
        }
        ds4_gpu_tensor_free(gx);
        ds4_gpu_tensor_free(gout);
    }

    free(x);
    free(ref);
    free(got);
    return ok;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* DS4_QUANT_BENCH=1: single-token matvec bandwidth. Several weight copies
 * are cycled so the working set does not sit in the system level cache. */
static void bench_type(arena_t *a, const quant_type *q, uint64_t in_dim, uint64_t rows) {
    enum { MAX_COPIES = 8, REPS = 256 };
    uint64_t row_bytes = 0;
    if (!ds4_quant_row_bytes(q->type, in_dim, &row_bytes)) return;
    if (rows * row_bytes > a->size / 2) return;
    const int COPIES = rows * row_bytes > ((uint64_t)96 << 20) ? 1 : MAX_COPIES;
    uint64_t off[MAX_COPIES];
    for (int c = 0; c < COPIES; c++) {
        off[c] = arena_alloc(a, rows * row_bytes);
        fill_rows(q, a->base + off[c], rows, in_dim);
    }
    const char *tenv = getenv("DS4_QUANT_BENCH_T");
    const uint64_t T = tenv ? strtoull(tenv, NULL, 10) : 1;
    float *x = malloc(T * in_dim * sizeof(float));
    for (uint64_t i = 0; i < T * in_dim; i++) x[i] = frand();
    ds4_gpu_tensor *gx = upload(x, T * in_dim);
    ds4_gpu_tensor *gout = upload(NULL, T * rows);
    double best = 1e30;
    for (int pass = 0; pass < 4; pass++) {
        const double t0 = now_s();
        require_ok(ds4_gpu_begin_commands(), "begin");
        for (int r = 0; r < REPS; r++) {
            require_ok(ds4_gpu_matmul_quant_tensor(gout, a->base, a->size, off[r % COPIES], q->type,
                                                   in_dim, rows, gx, T), "bench matvec");
        }
        require_ok(ds4_gpu_end_commands(), "end");
        require_ok(ds4_gpu_synchronize(), "sync");
        const double us = 1e6 * (now_s() - t0) / REPS;
        if (pass && us < best) best = us;
    }
    printf("  %-8s T=%llu %5llux%-6llu %8.1f us %7.1f GB/s\n", q->name, (unsigned long long)T, (unsigned long long)in_dim,
           (unsigned long long)rows, best, rows * row_bytes / best * 1e-3);
    ds4_gpu_tensor_free(gx);
    ds4_gpu_tensor_free(gout);
    free(x);
    a->used = 0;
}

/* DS4_QUANT_BENCH=moe: Q2_0 routed-expert decode (10 slots + a Q2_0 shared
 * expert, Qwen3.8 Flash shapes) over 64 experts, cycling expert sets. */
static void bench_moe(arena_t *a, const quant_type *shq) {
    enum { E = 64, SLOTS = 10, SETS = 6, REPS = 240, IN = 2560, FF = 640 };
    const quant_type *q = &types[0];
    const uint64_t gu_bytes = (uint64_t)E * FF * (IN / 64) * 18, d_bytes = (uint64_t)E * IN * (FF / 64) * 18;
    const uint64_t goff = arena_alloc(a, gu_bytes), uoff = arena_alloc(a, gu_bytes), doff = arena_alloc(a, d_bytes);
    fill_rows(q, a->base + goff, (uint64_t)E * FF, IN);
    fill_rows(q, a->base + uoff, (uint64_t)E * FF, IN);
    fill_rows(q, a->base + doff, (uint64_t)E * IN, FF);
    float *x = malloc(IN * sizeof(float));
    for (int i = 0; i < IN; i++) x[i] = frand();
    ds4_gpu_tensor *gx = upload(x, IN);
    ds4_gpu_tensor *mid = upload(NULL, (SLOTS + 1) * FF);
    ds4_gpu_tensor *part = upload(NULL, (SLOTS + 1) * IN);
    ds4_gpu_tensor *sel[SETS];
    for (int r = 0; r < SETS; r++) {
        int32_t ids[SLOTS];
        for (int i = 0; i < SLOTS; i++) ids[i] = (r * SLOTS + i) % E;
        sel[r] = ds4_gpu_tensor_alloc(sizeof(ids));
        require_ok(ds4_gpu_tensor_write(sel[r], 0, ids, sizeof(ids)), "selected write");
    }
    const quant_type *shd = FF % shq->block ? &types[1] : shq;   /* Q5_0 down for 256-blocks */
    uint64_t sgb = 0, sdb = 0;
    require_ok(ds4_quant_row_bytes(shq->type, IN, &sgb) && ds4_quant_row_bytes(shd->type, FF, &sdb), "shared rows");
    const uint64_t sg = arena_alloc(a, FF * sgb), su = arena_alloc(a, FF * sgb), sd = arena_alloc(a, IN * sdb);
    fill_rows(shq, a->base + sg, FF, IN);
    fill_rows(shq, a->base + su, FF, IN);
    fill_rows(shd, a->base + sd, IN, FF);
    for (int which = 0; which < 2; which++) {
        double best = 1e30;
        for (int pass = 0; pass < 4; pass++) {
            const double t0 = now_s();
            require_ok(ds4_gpu_begin_commands(), "begin");
            for (int r = 0; r < REPS; r++) {
                if (which == 0) {
                    require_ok(ds4_gpu_qwen4_moe_mid_tensor(mid, gx, sel[r % SETS], a->base, a->size, goff, uoff, 42u, E, 1,
                                                            SLOTS, IN, FF, sg, su, shq->type), "moe mid");
                } else {
                    require_ok(ds4_gpu_qwen4_moe_down_tensor(part, mid, sel[r % SETS], a->base, a->size, doff, 42u, E, 1,
                                                             SLOTS, FF, IN, sd, shd->type), "moe down");
                }
            }
            require_ok(ds4_gpu_end_commands(), "end");
            require_ok(ds4_gpu_synchronize(), "sync");
            const double us = 1e6 * (now_s() - t0) / REPS;
            if (pass && us < best) best = us;
        }
        const double bytes = which == 0 ? (SLOTS + 1) * 2.0 * FF * (IN / 64) * 18 : (SLOTS + 1) * (double)IN * (FF / 64) * 18;
        printf("  moe %-5s q2_0 + %-6s shared %8.1f us %7.1f GB/s\n", which == 0 ? "mid" : "down", shq->name, best,
               bytes / best * 1e-3);
    }
    for (int r = 0; r < SETS; r++) ds4_gpu_tensor_free(sel[r]);
    ds4_gpu_tensor_free(gx);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part);
    free(x);
    a->used = 0;
}

int main(void) {
    static const uint64_t shapes[][2] = {
        { 2560, 640 }, { 6144, 96 }, { 640, 2560 }, { 512, 33 },
    };
    arena_t arena = { 0 };
    arena.size = (uint64_t)1 << 30;
    arena.base = mmap(NULL, arena.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (arena.base == MAP_FAILED) { perror("mmap"); return 1; }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(arena.base, arena.size), "model map registration");

    if (getenv("DS4_QUANT_BENCH")) {
        static const uint64_t bshapes[][2] = { { 2560, 10240 }, { 2560, 6144 }, { 6144, 2560 }, { 2560, 640 }, { 2560, 248320 }, { 10240, 320 }, { 2560, 512 } };
        const char *only = getenv("DS4_QUANT_BENCH");
        if (!strcmp(only, "moe") || !strcmp(only, "1")) {
            bench_moe(&arena, &types[0]);
            bench_moe(&arena, &types[3]);
            bench_moe(&arena, &ref_types[1]);
            bench_moe(&arena, &types[6]);
        }
        for (size_t i = 0; i < sizeof(types) / sizeof(types[0]) + 2; i++) {
            const quant_type *q = i < sizeof(types) / sizeof(types[0]) ? &types[i]
                                  : &ref_types[i - sizeof(types) / sizeof(types[0])];
            if (strcmp(only, "1") && strcmp(only, q->name)) continue;
            for (size_t s = 0; s < sizeof(bshapes) / sizeof(bshapes[0]); s++) {
                if (bshapes[s][0] % q->block == 0) bench_type(&arena, q, bshapes[s][0], bshapes[s][1]);
            }
        }
        ds4_gpu_cleanup();
        return 0;
    }

    if (getenv("DS4_QUANT_BENCH")) {
        static const uint64_t bshapes[][2] = { { 2560, 10240 }, { 2560, 6144 }, { 6144, 2560 }, { 2560, 640 }, { 2560, 248320 }, { 10240, 320 }, { 2560, 512 } };
        const char *only = getenv("DS4_QUANT_BENCH");
        if (!strcmp(only, "moe") || !strcmp(only, "1")) {
            bench_moe(&arena, &types[0]);
            bench_moe(&arena, &types[3]);
            bench_moe(&arena, &ref_types[1]);
            bench_moe(&arena, &types[6]);
        }
        for (size_t i = 0; i < sizeof(types) / sizeof(types[0]) + 2; i++) {
            const quant_type *q = i < sizeof(types) / sizeof(types[0]) ? &types[i]
                                  : &ref_types[i - sizeof(types) / sizeof(types[0])];
            if (strcmp(only, "1") && strcmp(only, q->name)) continue;
            for (size_t s = 0; s < sizeof(bshapes) / sizeof(bshapes[0]); s++) {
                if (bshapes[s][0] % q->block == 0) bench_type(&arena, q, bshapes[s][0], bshapes[s][1]);
            }
        }
        ds4_gpu_cleanup();
        return 0;
    }

    int ok = 1;
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        printf("%s\n", types[i].name);
        for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
            if (shapes[s][0] % types[i].block) continue;
            ok &= test_shape(&arena, &types[i], shapes[s][0], shapes[s][1]);
        }
    }
    ds4_gpu_cleanup();
    printf(ok ? "all quant type checks passed\n" : "quant type checks FAILED\n");
    return ok ? 0 : 1;
}
