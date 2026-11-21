#include "ds4_hy4_metal.h"
#include "ds4_hy4_ane_schedule.h"
#include "ds4_hy4_rewind.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum { HEADS = 32, LATENT = 512, KV = 576, CTX = 2112 };

static uint32_t rng = 1;

static int test_expert_down_reduce(void);
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

enum { VERIFY_MAX_ROWS = 8, VERIFY_BENCH_ROWS = 3 };
enum verify_path { VERIFY_NEW, VERIFY_BATCH, VERIFY_SINGLE };

static void *verify_map;
static size_t verify_map_size;

static void *verify_weights(size_t bytes) {
    if (verify_map_size >= bytes) return verify_map;
    const size_t page = getpagesize();
    const size_t size = (bytes + page - 1) / page * page;
    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (map == MAP_FAILED) return NULL;
    if (!ds4_gpu_set_model_map_range(map, size, 0, size, bytes)) {
        munmap(map, size);
        return NULL;
    }
    if (verify_map) munmap(verify_map, verify_map_size);
    verify_map = map;
    verify_map_size = size;
    return map;
}

static void verify_random_quant(uint8_t *packed, size_t bytes, unsigned block, unsigned scale_hi, uint8_t scale) {
    for (size_t i = 0; i < bytes; i++) {
        rng = 1664525u * rng + 1013904223u;
        packed[i] = rng >> 24;
    }
    for (size_t i = 0; i < bytes; i += block) packed[i + scale_hi] = scale;
}

static int verify_compare(const char *name, uint32_t rows, ds4_gpu_tensor *ref,
                          ds4_gpu_tensor *out, size_t count, float atol, float rtol) {
    const float *expected = ds4_gpu_tensor_contents(ref), *actual = ds4_gpu_tensor_contents(out);
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t bad = count;
    for (size_t i = 0; i < count; i++) {
        const float delta = fabsf(actual[i] - expected[i]);
        max_abs = fmaxf(max_abs, delta);
        max_rel = fmaxf(max_rel, delta / fmaxf(fabsf(expected[i]), 1e-7f));
        if (bad == count && (!isfinite(actual[i]) || !isfinite(expected[i]) ||
                             delta > atol + rtol * fabsf(expected[i]))) bad = i;
    }
    printf("%s n=%u max_abs=%g max_rel=%g\n", name, rows, max_abs, max_rel);
    if (bad == count) return 1;
    fprintf(stderr, "%s n=%u element=%zu expected=%g actual=%g\n",
            name, rows, bad, expected[bad], actual[bad]);
    return 0;
}

typedef int (*verify_bench_fn)(void *context, enum verify_path path);

static int verify_bench(const char *name, verify_bench_fn dispatch, void *context) {
    enum { REPEATS = 12, TRIALS = 3 };
    double best[2] = {INFINITY, INFINITY};
    for (int trial = -1; trial < TRIALS; trial++) {
        for (enum verify_path path = VERIFY_NEW; path <= VERIFY_BATCH; path++) {
            if (!ds4_gpu_begin_commands()) return 0;
            const int repeats = trial < 0 ? 1 : REPEATS;
            int ok = 1;
            for (int i = 0; i < repeats && ok; i++) ok = dispatch(context, path);
            double ms = 0.0;
            if (!ds4_gpu_hy4_end_timed_commands(&ms) || !ok || ms <= 0.0) return 0;
            if (trial >= 0) best[path] = fmin(best[path], ms / repeats);
        }
    }
    printf("%s n=%d GPU old %.3f ms new %.3f ms (%.2fx)\n",
           name, VERIFY_BENCH_ROWS, best[VERIFY_BATCH], best[VERIFY_NEW],
           best[VERIFY_BATCH] / best[VERIFY_NEW]);
    return 1;
}

typedef int (*verify_projection_fn)(const ds4_hy4_args *a, const ds4_gpu_tensor *x,
                                    ds4_gpu_tensor *out, enum verify_path path);

struct verify_projection_bench {
    ds4_hy4_args args;
    const ds4_gpu_tensor *in;
    ds4_gpu_tensor *out;
    verify_projection_fn dispatch;
};

static int verify_projection_batch(void *context, enum verify_path path) {
    const struct verify_projection_bench *b = context;
    return b->dispatch(&b->args, b->in, b->out, path);
}

static int verify_projection(const char *name, ds4_hy4_args a, uint32_t in_cols, uint32_t out_cols,
                             verify_projection_fn dispatch, bool bench) {
    const uint32_t max_rows = bench ? VERIFY_BENCH_ROWS : VERIFY_MAX_ROWS;
    const size_t in_stride = in_cols * sizeof(float), out_stride = out_cols * sizeof(float);
    ds4_gpu_tensor *in = random_tensor(max_rows * in_cols, false);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(max_rows * out_stride);
    ds4_gpu_tensor *ref = bench ? NULL : ds4_gpu_tensor_alloc(max_rows * out_stride);
    int ok = out && (bench || ref);
    if (bench && ok) {
        a.rows = VERIFY_BENCH_ROWS;
        struct verify_projection_bench b = {a, in, out, dispatch};
        ok = verify_bench(name, verify_projection_batch, &b);
    }
    for (uint32_t n = 1; !bench && ok && n <= max_rows; n++) {
        a.rows = 1;
        for (uint32_t i = 0; ok && i < n; i++) {
            ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(in, i * in_stride, in_stride);
            ds4_gpu_tensor *ref_row = ds4_gpu_tensor_view(ref, i * out_stride, out_stride);
            ok = x_row && ref_row && dispatch(&a, x_row, ref_row, VERIFY_SINGLE);
            ds4_gpu_tensor_free(x_row);
            ds4_gpu_tensor_free(ref_row);
        }
        a.rows = n;
        ok = ok && ds4_gpu_tensor_fill_f32(out, NAN, n * out_cols) &&
             dispatch(&a, in, out, VERIFY_NEW) &&
             verify_compare(name, n, ref, out, n * out_cols, 1.5e-4f, 5e-5f);
    }
    ds4_gpu_tensor_free(in);
    ds4_gpu_tensor_free(ref);
    ds4_gpu_tensor_free(out);
    return ok;
}

static int verify_head_dispatch(const ds4_hy4_args *a, const ds4_gpu_tensor *in,
                                ds4_gpu_tensor *out, enum verify_path path) {
    const uint32_t out_dim = a->width == 192 ? 512 : 256;
    const ds4_hy4_binding b[] = {B(in), {NULL, 0, (uint64_t)a->heads * out_dim * a->row_bytes}, B(out)};
    if (path == VERIFY_NEW)
        return ds4_gpu_hy4_dispatch_rows("kernel_hy4_head_mv_rows", a, verify_map, verify_map_size,
                                         b, 3, out_dim / 16, a->heads, 128);
    if (path == VERIFY_SINGLE)
        return ds4_gpu_hy4_dispatch("kernel_hy4_head_mv", a, verify_map, verify_map_size,
                                    b, 3, out_dim / 16, a->heads, 1, 128);
    const char *batch = a->type == 14 ? "kernel_hy4_head_mm_batch_q6_k" : "kernel_hy4_head_mm_batch_q8_0";
    return ds4_gpu_hy4_dispatch(batch, a, verify_map, verify_map_size,
                                b, 3, out_dim / 32, a->heads, (a->rows + 31) / 32, 128);
}

static int test_verify_head(bool bench) {
    const char *names[] = {"MLA key Q8_0", "MLA value Q8_0", "MLA value Q6_K"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const bool key = i == 0, q6 = i == 2;
        const uint32_t heads = bench ? (q6 ? 32 : 64) : 2;
        const uint32_t width = key ? 192 : 512;
        const uint32_t in_dim = key ? 256 : 512, out_dim = key ? 512 : 256;
        const uint32_t row_bytes = q6 ? width / 256 * 210 : width / 32 * 34;
        const size_t bytes = (size_t)heads * out_dim * row_bytes;
        uint8_t *w = verify_weights(bytes);
        if (!w) return 0;
        verify_random_quant(w, bytes, q6 ? 210 : 34, q6 ? 209 : 1, 0x08);
        const ds4_hy4_args a = {.width = width, .heads = heads, .row_bytes = row_bytes, .type = q6 ? 14 : 8};
        if (!verify_projection(names[i], a, heads * in_dim, heads * out_dim, verify_head_dispatch, bench))
            return 0;
    }
    return 1;
}

static int verify_float_dispatch(const ds4_hy4_args *a, const ds4_gpu_tensor *in,
                                 ds4_gpu_tensor *out, enum verify_path path) {
    const uint64_t offset = 0;
    if (path == VERIFY_NEW) {
        const bool narrow = a->groups < 64;
        const ds4_hy4_binding b[] = {B(in), {NULL, offset, (uint64_t)a->groups * a->row_bytes}, B(out)};
        return ds4_gpu_hy4_dispatch_rows("kernel_hy4_float_mv_rows", a, verify_map, verify_map_size, b, 3,
                                         narrow ? a->groups : (a->groups + 3) / 4, 1, narrow ? 256 : 128);
    }
    if (path == VERIFY_SINGLE && a->type == 0 && a->groups >= 128)
        return ds4_gpu_qwen4_multi_gemv_tensor(in, 1, a->width, 1, &out, verify_map, verify_map_size,
                                               &offset, &a->type, &a->groups);
    return ds4_gpu_qwen4_dense_mm_tensor(out, in, verify_map, verify_map_size,
                                         offset, a->type, a->rows, a->width, a->groups);
}

static void verify_random_float(void *weights, size_t count, bool f16) {
    for (size_t i = 0; i < count; i++) {
        rng = 1664525u * rng + 1013904223u;
        const float value = (int)(rng >> 8) / 268435456.0f - 0.03125f;
        if (f16) ((__fp16 *)weights)[i] = (__fp16)value;
        else ((float *)weights)[i] = value;
    }
}

static int test_verify_float(bool bench) {
    const struct { const char *name; uint32_t width, out; bool f16, tail; } shapes[] = {
        {"F32 router", 3072, 512, false, false},
        {"F32 index key", 3072, 128, false, false},
        {"F32 index query", 1024, 2048, false, false},
        {"F32 index projection", 3072, 16, false, false},
        {"F32 compressor", 128, 128, false, false},
        {"F16 output tail", 280, 131, true, true},
    };
    for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        if (bench && shapes[i].tail) continue;
        const uint32_t width = shapes[i].width, out = shapes[i].out;
        const uint32_t row_bytes = width * (shapes[i].f16 ? sizeof(__fp16) : sizeof(float));
        void *w = verify_weights((size_t)out * row_bytes);
        if (!w) return 0;
        verify_random_float(w, (size_t)out * width, shapes[i].f16);
        const ds4_hy4_args a = {.width = width, .groups = out, .row_bytes = row_bytes, .type = shapes[i].f16};
        if (!verify_projection(shapes[i].name, a, width, out, verify_float_dispatch, bench)) return 0;
    }
    return 1;
}

struct verify_hc {
    ds4_hy4_args args;
    ds4_gpu_tensor *in, *scratch, *mix, *scale, *base, *out, *post;
};

static int verify_hc_dispatch(void *context, enum verify_path path) {
    const struct verify_hc *c = context;
    const ds4_hy4_args *a = &c->args;
    const ds4_hy4_binding p[] = {B(c->in), {NULL, 0, a->width * 12288 * sizeof(float)}, B(c->scratch)};
    const ds4_hy4_binding f[] = {B(c->in), B(c->scratch), B(c->scale), B(c->base), B(c->out), B(c->post)};
    const ds4_hy4_binding m[] = {B(c->in), B(c->mix), B(c->scale), B(c->base), B(c->out), B(c->post)};
    if (path == VERIFY_BATCH)
        return ds4_gpu_rms_norm_plain_rows_tensor(c->scratch, c->in, 12288, a->rows, 1e-5f) &&
               ds4_gpu_qwen4_dense_mm_tensor(c->mix, c->scratch, verify_map, verify_map_size,
                                             0, 0, a->rows, 12288, a->width) &&
               ds4_gpu_hy4_dispatch("kernel_hy4_hc_mix", a, NULL, 0, m, 6, 12, a->rows, 1, 256);
    const int partial = path == VERIFY_NEW ?
        ds4_gpu_hy4_dispatch_rows("kernel_hy4_hc_partial_rows", a, verify_map, verify_map_size,
                                  p, 3, a->groups, 1, a->width * 32) :
        ds4_gpu_hy4_dispatch("kernel_hy4_hc_partial", a, verify_map, verify_map_size,
                              p, 3, a->groups, 1, 1, a->width * 32);
    return partial && ds4_gpu_hy4_dispatch("kernel_hy4_hc_finish", a, NULL, 0, f, 6, 12, a->rows, 1, 256);
}

static int test_verify_hc(bool bench) {
    const size_t in_stride = 12288 * sizeof(float), out_stride = 3072 * sizeof(float);
    const size_t post_stride = 4 * sizeof(float);
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * out_stride);
    ds4_gpu_tensor *ref_post = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * post_stride);
    struct verify_hc c = {
        .args = {.rows = VERIFY_BENCH_ROWS, .groups = 16},
        .in = random_tensor(VERIFY_MAX_ROWS * 12288, false),
        .scratch = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * in_stride),
        .mix = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * 8 * sizeof(float)),
        .scale = random_tensor(2, false),
        .base = random_tensor(8, false),
        .out = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * out_stride),
        .post = ds4_gpu_tensor_alloc(VERIFY_MAX_ROWS * post_stride),
    };
    int ok = ref && ref_post && c.scratch && c.mix && c.out && c.post;
    for (uint32_t width = 4; ok && width <= 8; width += 4) {
        const char *name = width == 4 ? "output hyper-connection" : "layer hyper-connection";
        c.args.width = width;
        void *w = verify_weights(width * in_stride);
        if (!w) { ok = 0; break; }
        verify_random_float(w, width * 12288, false);
        if (bench) {
            ok = verify_bench(name, verify_hc_dispatch, &c);
            continue;
        }
        for (uint32_t n = 1; ok && n <= VERIFY_MAX_ROWS; n++) {
            struct verify_hc single = c;
            single.args.rows = 1;
            for (uint32_t i = 0; ok && i < n; i++) {
                single.in = ds4_gpu_tensor_view(c.in, i * in_stride, in_stride);
                single.out = ds4_gpu_tensor_view(ref, i * out_stride, out_stride);
                single.post = ds4_gpu_tensor_view(ref_post, i * post_stride, post_stride);
                ok = single.in && single.out && single.post && verify_hc_dispatch(&single, VERIFY_SINGLE);
                ds4_gpu_tensor_free(single.in);
                ds4_gpu_tensor_free(single.out);
                ds4_gpu_tensor_free(single.post);
            }
            c.args.rows = n;
            ok = ok && ds4_gpu_tensor_fill_f32(c.out, NAN, n * 3072) &&
                 ds4_gpu_tensor_fill_f32(c.post, NAN, n * 4) && verify_hc_dispatch(&c, VERIFY_NEW) &&
                 verify_compare(name, n, ref, c.out, n * 3072, 1e-5f, 1e-5f) &&
                 (width == 4 || verify_compare("hyper-connection post", n, ref_post, c.post, n * 4, 1e-5f, 1e-5f));
        }
    }
    ds4_gpu_tensor *owned[] = {ref, ref_post, c.in, c.scratch, c.mix, c.scale, c.base, c.out, c.post};
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) ds4_gpu_tensor_free(owned[i]);
    return ok;
}

static int test_verify(bool bench) {
    return test_verify_head(bench) &&
           test_verify_float(bench) && test_verify_hc(bench);
}

static int test_expert_mid_r4(void) {
    enum { DIM = 1536, SLOTS = 16, EXPERTS = 3, BLOCK = 66, ROW_BYTES = 6 * BLOCK };
    const ds4_hy4_args a = {.rows = 1, .type = 16, .row_bytes = ROW_BYTES};
    const size_t weight_bytes = EXPERTS * DIM * ROW_BYTES;
    const size_t count = SLOTS * DIM;
    ds4_gpu_tensor *x = random_tensor(DIM, false);
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(SLOTS * sizeof(int32_t));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(count * sizeof(float));
    ds4_gpu_tensor *owned[] = {x, ids, gate, up, ref, out};
    int ok = ids && gate && up && ref && out;
    if (ok) {
        int32_t *selected = ds4_gpu_tensor_contents(ids);
        for (unsigned s = 0; s < SLOTS; s++) selected[s] = (2 * s + 1) % EXPERTS;
        for (unsigned t = 2; t < 4; t++) {
            uint8_t *packed = ds4_gpu_tensor_contents(owned[t]);
            for (size_t i = 0; i < weight_bytes; i++) {
                rng = 1664525u * rng + 1013904223u;
                packed[i] = i % BLOCK == 1 ? 0x18 : rng >> 24;
            }
        }
        const ds4_hy4_binding scalar[] = {B(x), B(ids), B(gate), B(up), B(ref)};
        const ds4_hy4_binding wide[] = {B(x), B(ids), B(gate), B(up), B(out)};
        ok = ds4_gpu_hy4_dispatch("kernel_hy4_expert_mid", &a, NULL, 0, scalar, 5,
                                  DIM / 8, SLOTS, 1, 256) &&
             RUN("expert_mid_iq2_xxs_r4", wide, DIM / 16, SLOTS);
        const float *expected = ds4_gpu_tensor_contents(ref), *actual = ds4_gpu_tensor_contents(out);
        for (size_t i = 0; ok && i < count; i++) {
            if (!isfinite(expected[i]) || !isfinite(actual[i]) ||
                fabsf(actual[i] - expected[i]) > 6e-5f + 3e-5f * fabsf(expected[i])) {
                fprintf(stderr, "Hy4 expert mid r4 element=%zu expected=%g actual=%g\n",
                        i, expected[i], actual[i]);
                ok = 0;
            }
        }
    }
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) ds4_gpu_tensor_free(owned[i]);
    if (ok) puts("Hy4 four-row IQ2_XXS expert mid matched the scalar kernel");
    return ok;
}

static int test_ane_schedule(void) {
    const struct { uint32_t remaining, cap, a, b; } cases[] = {
        {3000, 512, 512, 512}, {900, 512, 512, 388}, {768, 512, 512, 256},
        {767, 512, 384, 383}, {512, 512, 256, 256}, {511, 512, 511, 0},
        {1200, 128, 128, 0}, {1200, 1024, 600, 600},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        uint32_t a;
        const uint32_t b = hy4_ane_pair_rows(cases[i].remaining, cases[i].cap, &a);
        if (a != cases[i].a || b != cases[i].b) {
            fprintf(stderr, "Hy4 ANE pair for %u rows with cap %u gave %u + %u\n",
                    cases[i].remaining, cases[i].cap, a, b);
            return 0;
        }
    }
    /* Compare both snapshot spans with the actual slots overwritten by
     * wrapped writes or writes covering the whole ring. */
    const uint32_t rings[] = {1023, 515, hy4_raw_ring_capacity(8192, 512),
                                       hy4_pending_ring_capacity(8192, 512)};
    for (unsigned k = 0; k < sizeof(rings) / sizeof(rings[0]); k++) {
        const uint32_t ring = rings[k];
        const bool pending = k % 2 != 0;
        const uint32_t pos[] = {63, 511, ring - 1, 2 * ring + 1};
        const uint32_t rows[] = {512, 600, 1024, ring + 1};
        for (unsigned p = 0; p < sizeof(pos) / sizeof(pos[0]); p++) {
            const uint32_t live = pending ? pos[p] % 4 : pos[p] - hy4_window_start(pos[p]);
            for (unsigned r = 0; r < sizeof(rows) / sizeof(rows[0]); r++) {
                bool written[ring];
                memset(written, 0, sizeof(written));
                for (uint32_t i = 0; i < rows[r]; i++) written[(pos[p] + i) % ring] = true;
                const uint32_t saved = hy4_ane_ring_overlap(live, ring, rows[r]);
                for (uint32_t i = 0; i < live; i++) {
                    if (written[(pos[p] - live + i) % ring] == (i < saved)) continue;
                    fprintf(stderr, "Hy4 ANE snapshot mismatch at pos %u for %u rows (ring %u)\n",
                            pos[p], rows[r], ring);
                    return 0;
                }
            }
        }
    }
    puts("Hy4 ANE chunk pairing and rolling-cache snapshot coverage passed");
    return 1;
}

static int test_rewind(void) {
    /* Model a verify batch accepting only its first row. Even with the
     * largest possible rejected suffix, a budget-sized rewind must work
     * for every group alignment, including a final-token logits replay. */
    const uint32_t cap = 512, ctx = 16384;
    const uint32_t raw_ring = hy4_raw_ring_capacity(ctx, cap);
    const uint32_t pending_ring = hy4_pending_ring_capacity(ctx, cap);
    for (uint32_t common = 6000; common < 6004; common++) {
        const uint32_t live = common + HY4_REWIND;
        const uint32_t end = live + cap - 1;
        const uint32_t raw = hy4_ring_after_write(0, live - 1, end, raw_ring);
        const uint32_t pending = hy4_ring_after_write(0, live - 1, end, pending_ring);
        if (hy4_rewind_start(live, live + 20, common, raw, pending) != common ||
            hy4_rewind_start(live, common, common, raw, pending) != common - 1 ||
            hy4_rewind_start(live, live + 20, common - 1, raw, pending) != 0) {
            fprintf(stderr, "Hy4 rewind budget or rejected-row accounting failed at %u\n", common);
            return 0;
        }
    }

    const struct { uint32_t live, prompt, common, raw, pending, start; } cases[] = {
        {8003, 8010, 8001, 7490, 8000, 8001},  /* Oldest required raw row fits. */
        {8003, 8010, 8001, 7491, 8000, 0},
        {8003, 8010, 8001, 7490, 8001, 0},     /* Beginning of the open group was lost. */
        {8003, 8010, 8000, 7489, 8002, 8000},  /* A group boundary needs no pending rows. */
        {8008, 8010, 8007, 7496, 8008, 0},     /* v1 snapshot at a group boundary. */
        {8008, 8010, 8007, 7496, 0, 8007},     /* Same window with no pool. */
        {8003, 8002, 8002, 7491, 8000, 0},     /* v1 snapshot one row short of replay. */
        {5, 4, 4, 0, 0, 3},
        {5, 1, 1, 0, 0, 0},
        {5, 10, 0, 0, 0, 0},
        {5, 5, 5, 0, 0, 0},                   /* Identical prompt needs no rewind. */
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const uint32_t start = hy4_rewind_start(cases[i].live, cases[i].prompt, cases[i].common,
                                                cases[i].raw, cases[i].pending);
        if (start != cases[i].start) {
            fprintf(stderr, "Hy4 rewind case %u expected %u, got %u\n", i, cases[i].start, start);
            return 0;
        }
    }
    if (hy4_raw_ring_capacity(256, 256) != 256 ||
        hy4_pending_ring_capacity(2500, 512) != 2500 ||
        hy4_ring_after_write(4000, 8000, 8001, raw_ring) != 4000 ||
        hy4_ring_after_write(8004, 8000, 8001, pending_ring) != 8000 ||
        hy4_ring_after_write(4000, 0, cap, raw_ring) != 0) {
        fputs("Hy4 ring capacity or retained-row tracking failed\n", stderr);
        return 0;
    }
    puts("Hy4 rewind budget, retained ring rows, and snapshot boundaries passed");
    return 1;
}

int main(int argc, char **argv) {
    if (!test_ane_schedule() || !test_rewind()) return 1;
    if (argc == 2 && strcmp(argv[1], "--host-only") == 0) return 0;
    const bool bench = argc == 2 && strcmp(argv[1], "--bench-verify") == 0;
    if (argc > 1 && !bench) {
        fprintf(stderr, "usage: %s [--host-only | --bench-verify]\n", argv[0]);
        return 1;
    }
    if (!ds4_gpu_init()) return 1;
    const int verify_ok = test_verify(bench);
    if (bench || !verify_ok) {
        ds4_gpu_cleanup();
        if (verify_map) munmap(verify_map, verify_map_size);
        return verify_ok ? 0 : 1;
    }
    if (!test_expert_mid_r4()) return 1;
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
        if (ok) {
            // Exercise both sparse indexing and a wrapped 512-key window.
            for (unsigned sparse = 0; sparse <= 1 && ok; sparse++) {
                ds4_hy4_args decode = a;
                decode.type = sparse;
                decode.capacity = sparse ? CTX : 512;
                decode.groups = sparse ? a.groups : ((pos < 512 ? pos + 1 : 512) + 31) / 32;
                if (!ds4_gpu_tensor_fill_f32(partial, NAN, (size_t)HEADS * decode.groups * 514) ||
                    !ds4_gpu_tensor_fill_f32(out, NAN, HEADS * LATENT) ||
                    !ds4_gpu_begin_commands()) return 1;
                ok =
                     ds4_gpu_hy4_dispatch("kernel_hy4_attention", &decode, NULL, 0, single, 6,
                                          HEADS, 1, 1, 128) &&
                     ds4_gpu_hy4_dispatch("kernel_hy4_attention_decode_partial", &decode, NULL, 0, split, 6,
                                          decode.groups, HEADS / 4, 1, 128) &&
                     ds4_gpu_hy4_dispatch("kernel_hy4_attention_merge", &decode, NULL, 0, merge, 2,
                                          HEADS, 1, 1, 128);
                ok = ds4_gpu_end_commands() && ok &&
                     ds4_gpu_tensor_read(ref, 0, expected, sizeof(expected)) &&
                     ds4_gpu_tensor_read(out, 0, actual, sizeof(actual));
                for (unsigned d = 0; ok && d < HEADS * LATENT; d++) {
                    if (!isfinite(actual[d]) || !isfinite(expected[d]) || fabsf(actual[d] - expected[d]) > 1e-5f) {
                        fprintf(stderr, "Hy4 decode attention pos=%u sparse=%u element=%u expected=%g actual=%g\n",
                                pos, sparse, d, expected[d], actual[d]);
                        ok = 0;
                    }
                }
            }
        }
    }
    const int head_mm_ok = test_head_mm_batch();
    ok = test_expert_down_reduce() && ok;
    ds4_gpu_tensor *owned[] = { z, q, cache, sink, indices, partial, ref, out };
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) ds4_gpu_tensor_free(owned[i]);
    ds4_gpu_cleanup();
    if (verify_map) munmap(verify_map, verify_map_size);
    if (ok) puts("Hy4 split attention passed at all nine context positions");
    return ok && head_mm_ok ? 0 : 1;
}

static int test_expert_down_reduce(void) {
    enum { DIM = 1536, SLOTS = 16, EXPERTS = 3, BLOCK = 66, ROW_BYTES = 6 * BLOCK };
    const ds4_hy4_args a = {.rows = 1, .row_bytes = ROW_BYTES};
    const size_t weight_bytes = EXPERTS * DIM * ROW_BYTES;
    ds4_gpu_tensor *mid = random_tensor(SLOTS * DIM, false);
    ds4_gpu_tensor *probs = random_tensor(SLOTS, false);
    ds4_gpu_tensor *ids = ds4_gpu_tensor_alloc(SLOTS * sizeof(int32_t));
    ds4_gpu_tensor *weight = ds4_gpu_tensor_alloc(weight_bytes);
    ds4_gpu_tensor *down = ds4_gpu_tensor_alloc(SLOTS * DIM * sizeof(float));
    ds4_gpu_tensor *ref = ds4_gpu_tensor_alloc(DIM * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(DIM * sizeof(float));
    ds4_gpu_tensor *owned[] = {mid, probs, ids, weight, down, ref, out};
    int ok = ids && weight && down && ref && out;
    if (ok) {
        int32_t *selected = ds4_gpu_tensor_contents(ids);
        float *router = ds4_gpu_tensor_contents(probs);
        for (unsigned s = 0; s < SLOTS; s++) {
            selected[s] = s % EXPERTS;
            router[s] = fabsf(router[s]) * 0.25f;
        }
        uint8_t *packed = ds4_gpu_tensor_contents(weight);
        for (size_t i = 0; i < weight_bytes; i++) {
            rng = 1664525u * rng + 1013904223u;
            packed[i] = i % BLOCK == 1 ? 0x10 : rng >> 24;
        }
        const ds4_hy4_binding separate[] = {B(mid), B(ids), B(weight), B(down)};
        const ds4_hy4_binding fused[] = {B(mid), B(ids), B(weight), B(probs), B(out)};
        ok = RUN("expert_down_iq2_xxs", separate, DIM / 8, SLOTS) &&
             ds4_gpu_qwen4_moe_reduce_tensor(ref, down, probs, NULL, NULL, NULL, NULL,
                                            1, SLOTS, SLOTS, DIM, 0) &&
             ds4_gpu_hy4_dispatch("kernel_hy4_expert_down_reduce_iq2_xxs", &a, NULL, 0, fused, 5,
                                  DIM / 4, 1, 1, 256);
        const float *expected = ds4_gpu_tensor_contents(ref), *actual = ds4_gpu_tensor_contents(out);
        for (size_t i = 0; ok && i < DIM; i++) {
            if (!isfinite(expected[i]) || !isfinite(actual[i]) ||
                fabsf(actual[i] - expected[i]) > 1e-5f + 1e-5f * fabsf(expected[i])) {
                fprintf(stderr, "Hy4 fused expert down element=%zu expected=%g actual=%g\n",
                        i, expected[i], actual[i]);
                ok = 0;
            }
        }
    }
    for (unsigned i = 0; i < sizeof(owned) / sizeof(owned[0]); i++) ds4_gpu_tensor_free(owned[i]);
    if (ok) puts("Hy4 fused IQ2_XXS expert down matched separate down and reduction");
    return ok;
}

/* Prefill head projections, including the second head and a partial row tile. */
static int test_head_mm_batch(void) {
    enum { ROWS = 35, TEST_HEADS = 2 };
    const char *kernels[] = {"kernel_hy4_head_mm_batch_q8_0", "kernel_hy4_head_mm_batch_q6_k"};
    int ok = 1;
    for (unsigned shape = 0; shape < 3 && ok; shape++) {
        const bool key = shape == 0, q6 = shape == 2;
        const uint32_t width = key ? 192 : 512;
        const uint32_t in_dim = key ? 256 : 512, out_dim = key ? 512 : 256;
        const uint32_t block_bytes = q6 ? 210 : 34;
        const uint32_t row_bytes = (width / (q6 ? 256 : 32)) * block_bytes;
        const size_t weight_bytes = TEST_HEADS * out_dim * row_bytes;
        const size_t count = ROWS * TEST_HEADS * out_dim;
        const ds4_hy4_args a = {.rows = ROWS, .width = width, .heads = TEST_HEADS,
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
        /* Reuse the first row, both heads and the original projection output. */
        ds4_hy4_args decode = a;
        decode.rows = 1;
        ok = ok && ds4_gpu_tensor_fill_f32(out, NAN, TEST_HEADS * out_dim) &&
             ds4_gpu_hy4_dispatch("kernel_hy4_head_mv", &decode, NULL, 0, batch, 3,
                                        out_dim / 16, TEST_HEADS, 1, 128);
        for (size_t d = 0; ok && d < TEST_HEADS * out_dim; d++) {
            if (!isfinite(actual[d]) ||
                fabsf(actual[d] - expected[d]) > 4e-5f + 2e-5f * fabsf(expected[d])) {
                fprintf(stderr, "Hy4 decode head mv type=%u element=%zu expected=%g actual=%g\n",
                        a.type, d, expected[d], actual[d]);
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
