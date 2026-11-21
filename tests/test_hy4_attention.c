#include "ds4_hy4_metal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { HEADS = 32, LATENT = 512, KV = 576, CTX = 2112 };

static uint32_t rng = 1;

static int test_head_mm_batch(void);

static ds4_gpu_tensor *random_tensor(size_t count, bool half_precision) {
    const size_t bytes = count * (half_precision ? sizeof(__fp16) : sizeof(float));
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(bytes);
    void *host = malloc(bytes);
    if (!t || !host) exit(1);
    for (size_t i = 0; i < count; i++) {
        rng = 1664525u * rng + 1013904223u;
        const float v = (int)(rng >> 8) / 4194304.0f - 2.0f;
        if (half_precision) ((__fp16 *)host)[i] = (__fp16)v;
        else ((float *)host)[i] = v;
    }
    if (!ds4_gpu_tensor_write(t, 0, host, bytes)) exit(1);
    free(host);
    return t;
}

#define B(t) { (t), 0, ds4_gpu_tensor_bytes(t) }
#define RUN(name, bindings, x, y) ds4_gpu_hy4_dispatch("kernel_hy4_" name, &a, NULL, 0, \
    bindings, sizeof(bindings) / sizeof(bindings[0]), x, y, 1, 128)

int main(void) {
    if (!ds4_gpu_init()) return 1;
    ds4_gpu_tensor *z = random_tensor(HEADS * LATENT, false);
    ds4_gpu_tensor *q = random_tensor(HEADS * 256, false);
    ds4_gpu_tensor *cache = random_tensor(CTX * KV, true);
    ds4_gpu_tensor *sink = random_tensor(HEADS, false);
    ds4_gpu_tensor *indices = ds4_gpu_tensor_alloc(511 * sizeof(int32_t));
    ds4_gpu_tensor *partial = ds4_gpu_tensor_alloc(HEADS * 64 * 514 * sizeof(float));
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(HEADS * LATENT * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(HEADS * LATENT * sizeof(float));
    if (!indices || !partial || !ref || !out) return 1;

    const ds4_hy4_binding single[] = { B(z), B(q), B(cache), B(indices), B(sink), B(ref) };
    const ds4_hy4_binding split[] = { B(z), B(q), B(cache), B(indices), B(sink), B(partial) };
    const ds4_hy4_binding merge[] = { B(partial), B(out) };
    const uint32_t positions[] = {511, 512, 769, 2047, 2048, 2049, 2050, 2051, 2111};
    float expected[HEADS * LATENT], actual[HEADS * LATENT];
    int ok = 1;
    for (unsigned i = 0; i < sizeof(positions) / sizeof(positions[0]) && ok; i++) {
        const uint32_t pos = positions[i];
        int32_t ids[511];
        // Walk backwards from the last complete group, then leave an unselected gap.
        for (unsigned j = 0; j < 511; j++) ids[j] = pos < 2048 ? -1 : (int32_t)(pos / 4 - 1 - j);
        const uint32_t keys = pos < 2048 ? pos + 1 : 2045 + pos % 4;
        const ds4_hy4_args a = {.rows = 1, .pos = pos, .heads = HEADS, .capacity = CTX,
                                .type = 1, .groups = (keys + 31) / 32};
        if (!ds4_gpu_tensor_write(indices, 0, ids, sizeof(ids)) || !ds4_gpu_begin_commands()) return 1;
        ok = RUN("attention", single, HEADS, 1) &&
             RUN("attention_partial", split, a.groups, HEADS / 4) && RUN("attention_merge", merge, HEADS, 1);
        ok = ds4_gpu_end_commands() && ok &&
             ds4_gpu_tensor_read(ref, 0, expected, sizeof(expected)) &&
             ds4_gpu_tensor_read(out, 0, actual, sizeof(actual));
        for (unsigned d = 0; ok && d < HEADS * LATENT; d++) {
            if (!isfinite(actual[d]) || !isfinite(expected[d]) || fabsf(actual[d] - expected[d]) > 1e-5f) {
                fprintf(stderr, "pos=%u element=%u expected=%g actual=%g\n", pos, d, expected[d], actual[d]);
                ok = 0;
            }
        }
    }
    const int head_mm_ok = test_head_mm_batch();
    ds4_gpu_tensor *owned[] = { z, q, cache, sink, indices, partial, ref, out };
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) ds4_gpu_tensor_free(owned[i]);
    ds4_gpu_cleanup();
    if (ok) puts("Hy4 split attention passed at all nine context positions");
    return ok && head_mm_ok ? 0 : 1;
}

/* Prefill head projections, including the second head and a partial row tile. */
static int test_head_mm_batch(void) {
    enum { ROWS = 35, TEST_HEADS = 2 };
    const char *kernels[] = {"kernel_hy4_head_mm_batch_q8_0", "kernel_hy4_head_mm_batch_q6_k"};
    int ok = 1;
    for (unsigned q6 = 0; q6 < 2 && ok; q6++) {
        const uint32_t in_dim = q6 ? 512 : 256, out_dim = q6 ? 256 : 512;
        const uint32_t block_bytes = q6 ? 210 : 34;
        const uint32_t row_bytes = q6 ? 2 * block_bytes : 6 * block_bytes;
        const size_t weight_bytes = TEST_HEADS * out_dim * row_bytes;
        const size_t count = ROWS * TEST_HEADS * out_dim;
        const ds4_hy4_args a = {.rows = ROWS, .width = q6 ? 512 : 192, .heads = TEST_HEADS,
                                .type = q6 ? 14 : 8, .row_bytes = row_bytes};
        ds4_gpu_tensor *x = random_tensor(ROWS * TEST_HEADS * in_dim, false);
        ds4_gpu_tensor *w = ds4_gpu_tensor_alloc(weight_bytes);
        ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(count * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(count * sizeof(float));
        if (!w || !ref || !out) return 0;
        uint8_t *packed = ds4_gpu_tensor_contents(w);
        for (size_t i = 0; i < weight_bytes; i++) {
            rng = 1664525u * rng + 1013904223u;
            packed[i] = rng >> 24;
        }
        for (size_t i = 0; i < weight_bytes; i += block_bytes) {
            /* Keep each fp16 block scale finite and small. */
            packed[i + (q6 ? 209 : 1)] = 0x08;
        }
        const ds4_hy4_binding single[] = { B(x), B(w), B(ref) };
        const ds4_hy4_binding batch[] = { B(x), B(w), B(out) };
        ok = ds4_gpu_hy4_dispatch("kernel_hy4_head_mm", &a, NULL, 0, single, 3,
                                  out_dim / 4, TEST_HEADS, ROWS, 128) &&
             ds4_gpu_hy4_dispatch(kernels[q6], &a, NULL, 0, batch, 3,
                                  out_dim / 32, TEST_HEADS, (ROWS + 31) / 32, 128);
        const float *expected = ds4_gpu_tensor_contents(ref), *actual = ds4_gpu_tensor_contents(out);
        for (size_t d = 0; ok && d < count; d++) {
            if (!isfinite(expected[d]) || !isfinite(actual[d]) ||
                fabsf(actual[d] - expected[d]) > 4e-5f + 2e-5f * fabsf(expected[d])) {
                fprintf(stderr, "%s element=%zu expected=%g actual=%g\n", kernels[q6], d, expected[d], actual[d]);
                ok = 0;
            }
        }
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(w);
        ds4_gpu_tensor_free(ref);
        ds4_gpu_tensor_free(out);
    }
    if (ok) puts("Hy4 batch head projections matched the row kernels");
    return ok;
}
