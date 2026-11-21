/* Hy4 uses Qwen's two surface sets and stages each layer's weights from
 * the model. Each projection shape has a program with one slot per layer. */
#include "ds4_hy4_ane_schedule.h"

typedef enum {
    HY4_ANE_QA, HY4_ANE_QB, HY4_ANE_KV, HY4_ANE_GATE, HY4_ANE_ATTN_OUT,
    HY4_ANE_INDEX_Q, HY4_ANE_INDEX_K, HY4_ANE_INDEX_W,
    HY4_ANE_SHARED, HY4_ANE_LATENT_IN, HY4_ANE_LATENT_OUT, HY4_ANE_PROJECTIONS
} hy4_ane_projection;

typedef struct {
    int program;
    uint32_t slot;
    ds4_gpu_ane_weight weight[3];
} hy4_ane_op;

typedef struct hy4_ane {
    hy4_ane_op layer[HY4_LAYERS][HY4_ANE_PROJECTIONS];
} hy4_ane;

typedef struct {
    hy4_ane_projection id;
    uint32_t k, n, count;
    uint8_t layers[HY4_LAYERS];
    uint64_t work, copies, surfaces;
} hy4_ane_candidate;

static const ds4_tensor *hy4_ane_weights(const hy4_layer_weights *l, hy4_ane_projection id,
                                         const ds4_tensor *w[3]) {
    const ds4_tensor *weights[HY4_ANE_PROJECTIONS] = {
        [HY4_ANE_QA] = l->qa, [HY4_ANE_QB] = l->qb, [HY4_ANE_KV] = l->kv,
        [HY4_ANE_GATE] = l->attn_gate, [HY4_ANE_ATTN_OUT] = l->attn_out,
        [HY4_ANE_INDEX_Q] = l->index_q, [HY4_ANE_INDEX_K] = l->index_k, [HY4_ANE_INDEX_W] = l->index_w,
        [HY4_ANE_SHARED] = l->router ? l->ffn.gate : NULL,
        [HY4_ANE_LATENT_IN] = l->latent_in, [HY4_ANE_LATENT_OUT] = l->latent_out,
    };
    w[0] = weights[id];
    w[1] = id == HY4_ANE_SHARED ? l->ffn.up : NULL;
    w[2] = id == HY4_ANE_SHARED ? l->ffn.down : NULL;
    return w[0];
}

static uint64_t hy4_ane_align(uint64_t bytes, uint32_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

static uint64_t hy4_ane_surface_bytes(uint32_t rows, uint32_t width_bytes) {
    return hy4_ane_align(rows * hy4_ane_align(width_bytes, 64), getpagesize());
}

static uint64_t hy4_ane_copy_bytes(const ds4_tensor *w) {
    return hy4_ane_align(w->dim[0] * w->dim[1] * sizeof(uint16_t), getpagesize());
}

static int hy4_ane_candidate_order(const void *a, const void *b) {
    const hy4_ane_candidate *x = a, *y = b;
    return x->work > y->work ? -1 : x->work < y->work;
}

static uint32_t hy4_ane_plan(const ds4_hy4_model *h, uint32_t rows, hy4_ane_candidate *plan) {
    uint32_t count = 0;
    for (unsigned il = 0; il < HY4_LAYERS; il++) {
        for (hy4_ane_projection id = 0; id < HY4_ANE_PROJECTIONS; id++) {
            const ds4_tensor *w[3];
            if (!hy4_ane_weights(&h->layer[il], id, w)) continue;
            const bool shared = id == HY4_ANE_SHARED;
            const uint32_t k = w[0]->dim[0], n = w[0]->dim[1];
            uint64_t copies = 0;
            bool supported = true;
            for (unsigned j = 0; j < (shared ? 3u : 1u); j++) {
                const uint32_t type = w[j]->type;
                if (ds4_gpu_ane_type_ok(type)) continue;
                if ((type != DS4_TENSOR_Q6_K && type != DS4_TENSOR_Q8_0) ||
                    !ds4_gpu_ane_type_ok(DS4_TENSOR_F16)) {
                    supported = false;
                    break;
                }
                copies += hy4_ane_copy_bytes(w[j]);
            }
            if (!supported) continue;
            uint32_t i = 0;
            while (i < count && (plan[i].id != id || plan[i].k != k || plan[i].n != n)) i++;
            if (i == count) plan[count++] = (hy4_ane_candidate){ .id = id, .k = k, .n = n };
            hy4_ane_candidate *c = &plan[i];
            c->layers[c->count++] = il;
            c->copies += copies;
            c->work += (uint64_t)k * n * (shared ? 3u : 1u);
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        hy4_ane_candidate *c = &plan[i];
        const bool shared = c->id == HY4_ANE_SHARED;
        const uint32_t k = c->k, n = c->n, out = shared ? k : n;
        const uint64_t weights = shared ?
            2 * hy4_ane_surface_bytes(n, 2 * k) + hy4_ane_surface_bytes(k, 2 * n) :
            hy4_ane_surface_bytes(n, k);
        c->surfaces = 2 * (weights + hy4_ane_surface_bytes(rows, 2 * k) +
                           hy4_ane_surface_bytes(rows, 2 * out)) +
            hy4_ane_align((uint64_t)c->count * out * sizeof(float), getpagesize());
    }
    qsort(plan, count, sizeof(*plan), hy4_ane_candidate_order);
    return count;
}

typedef struct {
    const uint8_t *src;
    uint16_t *dst;
    uint64_t cols, row_bytes;
    uint32_t type;
    atomic_bool failed;
} hy4_ane_convert;

static void hy4_ane_convert_rows(void *ctx, uint64_t first, uint64_t end) {
    hy4_ane_convert *c = ctx;
    float row[c->cols];
    for (uint64_t i = first; i < end; i++) {
        if (!ds4_dequant_row(c->type, c->src + i * c->row_bytes, row, c->cols)) {
            atomic_store(&c->failed, true);
            return;
        }
        for (uint64_t j = 0; j < c->cols; j++) c->dst[i * c->cols + j] = f32_to_f16(row[j]);
    }
}

static bool hy4_ane_bind_weight(ds4_gpu_ane_weight *out, const ds4_model *m, const ds4_tensor *w) {
    const uint64_t row_bytes = w->bytes / w->dim[1];
    if (ds4_gpu_ane_type_ok(w->type)) {
        *out = (ds4_gpu_ane_weight){ .map = m->map, .size = m->size, .offset = w->abs_offset,
                                     .row_bytes = row_bytes, .type = w->type };
        return true;
    }
    ds4_gpu_tensor *copy = ds4_gpu_tensor_alloc(hy4_ane_copy_bytes(w));
    *out = (ds4_gpu_ane_weight){ .row_bytes = w->dim[0] * sizeof(uint16_t),
                                 .type = DS4_TENSOR_F16, .tensor = copy };
    if (!copy) return false;
    hy4_ane_convert c = { .src = m->map + w->abs_offset, .dst = ds4_gpu_tensor_contents(copy),
                          .cols = w->dim[0], .row_bytes = row_bytes, .type = w->type };
    if (!c.dst) return false;
    atomic_init(&c.failed, false);
    ds4_parallel_for_min_rows(w->dim[1], hy4_ane_convert_rows, &c, 16);
    return !atomic_load(&c.failed);
}

static void hy4_ane_close(ds4_hy4_model *h) {
    if (!h->ane) return;
    ds4_gpu_ane_discard_setup();
    for (unsigned il = 0; il < HY4_LAYERS; il++)
        for (unsigned i = 0; i < HY4_ANE_PROJECTIONS; i++)
            for (unsigned j = 0; j < 3; j++)
                ds4_gpu_tensor_free((ds4_gpu_tensor *)h->ane->layer[il][i].weight[j].tensor);
    free(h->ane);
    h->ane = NULL;
}

static void hy4_ane_open(ds4_hy4_model *h, const ds4_model *m, uint32_t cap) {
    const char *enabled = getenv("DS4_ANE");
    if (!enabled || strcmp(enabled, "1") != 0 || cap < HY4_ANE_MIN_ROWS) return;
    const double start = now_sec();
    const uint32_t rows = hy4_ane_align(cap, HY4_ANE_MIN_ROWS);
    const uint64_t budget = UINT64_C(1) << 30;
    const char *mask_env = getenv("DS4_HY4_ANE_MASK");
    /* attn_out, latent in/out and the index projections on the ANE slow the
     * M1 Max prefill down (the GPU then waits on joins); the default keeps
     * the four projections that measured faster. DS4_HY4_ANE_MASK overrides,
     * one bit per hy4_ane_projection. */
    const uint32_t default_mask = 1u << HY4_ANE_QA | 1u << HY4_ANE_KV |
                                  1u << HY4_ANE_GATE | 1u << HY4_ANE_SHARED;
    const uint32_t mask = mask_env ? (uint32_t)strtoul(mask_env, NULL, 0) : default_mask;
    hy4_ane_candidate plan[2 * HY4_ANE_PROJECTIONS];
    const uint32_t count = hy4_ane_plan(h, rows, plan);
    uint64_t copies = 0, surfaces = 0;
    uint32_t programs = 0;
    if (!ds4_gpu_ane_set_count(2)) goto fail;
    h->ane = calloc(1, sizeof(*h->ane));
    if (!h->ane) goto fail;
    for (unsigned il = 0; il < HY4_LAYERS; il++)
        for (unsigned i = 0; i < HY4_ANE_PROJECTIONS; i++) h->ane->layer[il][i].program = -1;
    for (uint32_t i = 0; i < count; i++) {
        const hy4_ane_candidate *c = &plan[i];
        if (mask && !(mask & (1u << c->id))) continue;
        if (c->copies + c->surfaces > budget - copies - surfaces) continue;
        const bool shared = c->id == HY4_ANE_SHARED;
        const int program = shared ? ds4_gpu_ane_create_ffn(c->k, c->n, rows, c->count) :
                                      ds4_gpu_ane_create(c->k, c->n, rows, c->count);
        if (program < 0) goto fail;
        for (uint32_t slot = 0; slot < c->count; slot++) {
            const uint32_t il = c->layers[slot];
            const ds4_tensor *w[3];
            hy4_ane_weights(&h->layer[il], c->id, w);
            hy4_ane_op *op = &h->ane->layer[il][c->id];
            op->program = program;
            op->slot = slot;
            for (unsigned j = 0; j < (shared ? 3u : 1u); j++)
                if (!hy4_ane_bind_weight(&op->weight[j], m, w[j])) goto fail;
            if (!shared && !ds4_gpu_ane_measure(program, slot, &op->weight[0])) goto fail;
        }
        copies += c->copies;
        surfaces += c->surfaces;
        programs++;
    }
    if (!programs || !ds4_gpu_ane_measure_done()) goto fail;
    fprintf(stderr, "ds4: Hy4 ANE prefill ready in %.2f s (%u programs, paired chunks, "
                    "%.1f MiB extra weight copies, %.1f MiB surfaces and scales)\n",
            now_sec() - start, programs, copies / 1048576.0, surfaces / 1048576.0);
    return;
fail:
    hy4_ane_close(h);
    fprintf(stderr, "ds4: Hy4 ANE setup failed; the GPU runs the prefill alone\n");
}

static bool hy4_ane_has(const hy4_ane_op *layer, hy4_ane_projection id) {
    return layer && layer[id].program >= 0;
}

/* Projections sharing the same activation also share one pack and input
 * ready signal. All weight staging goes into that command buffer. */
static bool hy4_ane_submit(const hy4_ane_op *layer, const ds4_gpu_tensor *in, uint32_t rows,
                           const hy4_ane_projection *ids, size_t count) {
    if (!layer) return true;
    int input = -1;
    for (size_t i = 0; i < count; i++) {
        const hy4_ane_projection id = ids[i];
        if (!hy4_ane_has(layer, id)) continue;
        const hy4_ane_op *op = &layer[id];
        if (input < 0) input = op->program;
        for (unsigned j = 0; j < (id == HY4_ANE_SHARED ? 3u : 1u); j++)
            if (!ds4_gpu_ane_stage(op->program, op->slot, j, &op->weight[j])) return false;
    }
    if (input < 0) return true;
    if (!ds4_gpu_ane_pack(input, in, rows)) return false;
    for (size_t i = 0; i < count; i++) {
        const hy4_ane_op *op = &layer[ids[i]];
        if (op->program >= 0 && !ds4_gpu_ane_eval(op->program, op->slot, input, rows)) return false;
    }
    return true;
}

static bool hy4_ane_join(const hy4_ane_op *layer, hy4_ane_projection id,
                         ds4_gpu_tensor *out, uint32_t rows, uint32_t cols) {
    const hy4_ane_op *op = &layer[id];
    return ds4_gpu_ane_unpack(op->program, op->slot, out, rows, cols, 0, NULL, 0);
}
