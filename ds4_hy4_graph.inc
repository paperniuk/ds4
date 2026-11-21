/* Hy4 Flash Metal graph. SwiGLU, weight projections and expert down/reduce
 * use the same DS4 quantized primitives as the Qwen graph. */
#include "ds4_hy4_metal.h"

#define HY4_SCRATCH_ROWS(X) \
    X(token_ids, 1, int32_t) \
    X(engram_valid, 1, int32_t) \
    X(streams, 4 * HY4_DIM, float) \
    X(flat_norm, 4 * HY4_DIM, float) \
    X(mix, 8, float) \
    X(post, 4, float) \
    X(norm, HY4_DIM, float) \
    X(branch, HY4_DIM, float) \
    X(shared, HY4_DIM, float) \
    X(qa, 1024, float) \
    X(q, 64 * 256, float) \
    X(kv, HY4_KV, float) \
    X(absorbed, 64 * 512, float) \
    X(attended, 64 * 512, float) \
    X(gate, 64 * 256, float) \
    X(ff_up, 13824, float) \
    X(ff_mid, 13824, float) \
    X(index_q, 16 * 128, float) \
    X(index_k, 128, float) \
    X(pool_v, 128, float) \
    X(pool_gate, 128, float) \
    X(index_w, 16, float) \
    X(selected, HY4_HIST_GROUPS, int32_t) \
    X(router, HY4_EXPERTS, float) \
    X(experts, HY4_ACTIVE, int32_t) \
    X(expert_lists, HY4_EXPERTS, int32_t) \
    X(expert_probs, HY4_ACTIVE, float) \
    X(latent, HY4_LATENT, float) \
    X(expert_mid, HY4_ACTIVE * HY4_LATENT, float) \
    X(expert_down, HY4_ACTIVE * HY4_LATENT, float) \
    X(engram, HY4_ENGRAM_LAYERS * HY4_LATENT, float)

typedef struct {
    ds4_gpu_tensor *kv, *pool, *pending;
} hy4_cache;

typedef struct {
    ds4_gpu_tensor *indices[HY4_DFLY_LAYERS];
    ds4_gpu_tensor *hidden, *states, *embed, *context, *context_embed, *context_hidden, *logits, *top;
    float *target_logits;
    uint32_t pos, row;
    uint32_t probes, pause;
    double single_sec, verify_sec_per_token;
} hy4_draft_graph;

static void hy4_draft_reset(hy4_draft_graph *d) {
    if (!d) return;
    d->pos = d->row = d->probes = d->pause = 0;
    d->single_sec = d->verify_sec_per_token = 0;
}

typedef struct {
    hy4_cache layer[HY4_LAYERS];
    ds4_gpu_tensor *scores, *attention_partials, *logits, *expert_counts;
    ds4_gpu_tensor *engram_layer[HY4_ENGRAM_LAYERS];
#define HY4_FIELD(name, columns, type) ds4_gpu_tensor *name;
    HY4_SCRATCH_ROWS(HY4_FIELD)
#undef HY4_FIELD
    float *engram_staging;
    uint32_t cap, ctx, ring, pending_ring;
    hy4_draft_graph *draft;
} ds4_hy4_graph;

static uint32_t hy4_prefill_cap(uint32_t ctx, uint32_t requested) {
    uint32_t cap = requested ? requested : HY4_PREFILL;
    if (cap > HY4_PREFILL_MAX) cap = HY4_PREFILL_MAX;
    return cap < ctx ? cap : ctx;
}

// The 32-head sparse layers select at most 2048 keys. Each partition
// carries its 512 value sums, softmax maximum and denominator.
static uint64_t hy4_attention_partial_bytes(uint32_t ctx) {
    if (ctx < 512) return 0;
    const uint32_t keys = ctx < 2048 ? ctx : 2048;
    return (uint64_t)32 * ((keys + 31) / 32) * 514 * sizeof(float);
}

static ds4_context_memory hy4_context_memory(uint32_t ctx, uint32_t requested) {
    ds4_context_memory m = {0};
    if (!ctx || ctx > HY4_CONTEXT) return m;
    m.prefill_cap = hy4_prefill_cap(ctx, requested);
    m.raw_cap = ctx < HY4_WINDOW + m.prefill_cap - 1 ? ctx : HY4_WINDOW + m.prefill_cap - 1;
    m.comp_cap = ctx / 4;
    const uint64_t cap = m.prefill_cap;
    m.raw_bytes = (uint64_t)30 * m.raw_cap * HY4_KV * sizeof(uint16_t);
    m.compressed_bytes = (uint64_t)12 * ctx * HY4_KV * sizeof(uint16_t);
    if (ctx > 2048) {
        m.compressed_bytes += (uint64_t)12 * (ctx / 4) * 128 * sizeof(uint16_t);
        m.compressed_bytes += (uint64_t)12 * (cap + 3) * 256 * sizeof(float);
        m.scratch_bytes += cap * (ctx / 4) * sizeof(float);
    }
#define HY4_PRICE(name, columns, type) m.scratch_bytes += cap * (columns) * sizeof(type);
    HY4_SCRATCH_ROWS(HY4_PRICE)
#undef HY4_PRICE
    m.scratch_bytes += (cap * HY4_ENGRAM_LAYERS * HY4_LATENT + HY4_VOCAB) * sizeof(float) + HY4_EXPERTS * sizeof(int32_t);
    m.scratch_bytes += hy4_attention_partial_bytes(ctx);
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

static void hy4_graph_free(ds4_hy4_graph *g) {
    if (!g) return;
    hy4_draft_graph *d = g->draft;
    if (d) {
        for (uint32_t i = 0; i < HY4_DFLY_LAYERS; i++) ds4_gpu_tensor_free(d->indices[i]);
        ds4_gpu_tensor_free(d->context_embed);
        ds4_gpu_tensor_free(d->context_hidden);
        ds4_gpu_tensor_free(d->context);
        ds4_gpu_tensor_free(d->states);
        ds4_gpu_tensor_free(d->hidden);
        ds4_gpu_tensor_free(d->embed);
        ds4_gpu_tensor_free(d->logits);
        ds4_gpu_tensor_free(d->top);
        free(d->target_logits);
        free(d);
    }
    for (unsigned i = 0; i < HY4_LAYERS; i++) {
        ds4_gpu_tensor_free(g->layer[i].kv);
        ds4_gpu_tensor_free(g->layer[i].pool);
        ds4_gpu_tensor_free(g->layer[i].pending);
    }
    ds4_gpu_tensor_free(g->scores);
    ds4_gpu_tensor_free(g->attention_partials);
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->expert_counts);
    for (unsigned i = 0; i < HY4_ENGRAM_LAYERS; i++) ds4_gpu_tensor_free(g->engram_layer[i]);
#define HY4_FREE(name, columns, type) ds4_gpu_tensor_free(g->name);
    HY4_SCRATCH_ROWS(HY4_FREE)
#undef HY4_FREE
    free(g->engram_staging);
    free(g);
}

static bool hy4_draft_alloc(ds4_hy4_graph *g) {
    const uint64_t row = HY4_DIM * sizeof(float);
    const uint32_t rows = g->cap < HY4_DFLY_ROWS ? g->cap : HY4_DFLY_ROWS;
    hy4_draft_graph *d = g->draft = xcalloc(1, sizeof(*d));
    d->hidden = ds4_gpu_tensor_alloc(rows * row);
    d->states = ds4_gpu_tensor_alloc(rows * row);
    d->embed = ds4_gpu_tensor_alloc(rows * row);
    d->logits = ds4_gpu_tensor_alloc((uint64_t)rows * HY4_VOCAB * sizeof(float));
    d->top = ds4_gpu_tensor_alloc(sizeof(int32_t));
    d->context = ds4_gpu_tensor_alloc(2 * row);
    if (d->context) {
        d->context_embed = ds4_gpu_tensor_view(d->context, 0, row);
        d->context_hidden = ds4_gpu_tensor_view(d->context, row, row);
    }
    if (!d->hidden || !d->states || !d->embed || !d->logits || !d->top ||
        !d->context_embed || !d->context_hidden) return false;
    d->target_logits = xmalloc((size_t)rows * HY4_VOCAB * sizeof(float));
    for (uint32_t i = 0; i < HY4_DFLY_LAYERS; i++) {
        d->indices[i] = ds4_gpu_tensor_alloc((uint64_t)rows * HY4_HIST_GROUPS * sizeof(int32_t));
        if (!d->indices[i]) return false;
    }
    return true;
}

static ds4_hy4_graph *hy4_graph_alloc(uint32_t ctx, uint32_t requested, bool draft) {
    if (!ctx || ctx > HY4_CONTEXT) return NULL;
    const ds4_context_memory plan = hy4_context_memory(ctx, requested);
    ds4_hy4_graph *g = xcalloc(1, sizeof(*g));
    g->ctx = ctx;
    g->cap = plan.prefill_cap;
    g->ring = plan.raw_cap;
    g->pending_ring = g->cap + 3;
    for (unsigned i = 0; i < HY4_LAYERS; i++) {
        hy4_cache *c = &g->layer[i];
        const bool sparse = hy4_sparse(i);
        c->kv = ds4_gpu_tensor_alloc((uint64_t)(sparse ? ctx : g->ring) * HY4_KV * sizeof(uint16_t));
        if (!c->kv) goto fail;
        if (sparse && ctx > 2048) {
            c->pool = ds4_gpu_tensor_alloc((uint64_t)(ctx / 4) * 128 * sizeof(uint16_t));
            c->pending = ds4_gpu_tensor_alloc((uint64_t)g->pending_ring * 256 * sizeof(float));
            if (!c->pool || !c->pending) goto fail;
        }
    }
    if (ctx > 2048) {
        g->scores = ds4_gpu_tensor_alloc((uint64_t)g->cap * (ctx / 4) * sizeof(float));
        if (!g->scores) goto fail;
    }
    if (hy4_attention_partial_bytes(ctx)) {
        g->attention_partials = ds4_gpu_tensor_alloc(hy4_attention_partial_bytes(ctx));
        if (!g->attention_partials) goto fail;
    }
    g->logits = ds4_gpu_tensor_alloc((uint64_t)HY4_VOCAB * sizeof(float));
    g->expert_counts = ds4_gpu_tensor_alloc(HY4_EXPERTS * sizeof(int32_t));
    if (!g->logits || !g->expert_counts) goto fail;
#define HY4_ALLOC(name, columns, type) \
    g->name = ds4_gpu_tensor_alloc((uint64_t)g->cap * (columns) * sizeof(type)); \
    if (!g->name) goto fail;
    HY4_SCRATCH_ROWS(HY4_ALLOC)
#undef HY4_ALLOC
    const uint64_t table_bytes = (uint64_t)g->cap * HY4_LATENT * sizeof(float);
    for (unsigned i = 0; i < HY4_ENGRAM_LAYERS; i++) {
        g->engram_layer[i] = ds4_gpu_tensor_view(g->engram, i * table_bytes, table_bytes);
        if (!g->engram_layer[i]) goto fail;
    }
    g->engram_staging = xmalloc(HY4_ENGRAM_LAYERS * table_bytes);
    if (draft && !hy4_draft_alloc(g)) goto fail;
    return g;
fail:
    hy4_graph_free(g);
    return NULL;
}

#undef HY4_SCRATCH_ROWS

#define HY4_T(t) { (t), 0, ds4_gpu_tensor_bytes(t) }
#define HY4_W(w) { NULL, (w)->abs_offset, (w)->bytes }
#define HY4_OP(name, a, x, y, z, threads, ...) \
    ds4_gpu_hy4_dispatch("kernel_hy4_" name, &(a), m->map, m->size, \
                         (const ds4_hy4_binding[]){__VA_ARGS__}, \
                         sizeof((const ds4_hy4_binding[]){__VA_ARGS__}) / sizeof(ds4_hy4_binding), \
                         (x), (y), (z), (threads))

static bool hy4_mm(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                   const ds4_gpu_tensor *in, uint32_t n) {
    if (n == 1 && w->type == DS4_TENSOR_Q6_K && w->dim[1] >= 3072 &&
        w->dim[1] % 16 == 0 && ds4_gpu_hy4_use_wide_q6()) {
        const ds4_hy4_args a = {.width = w->dim[0]};
        const bool wide = w->dim[1] >= 65536;
        return ds4_gpu_hy4_dispatch(wide ? "kernel_hy4_q6_r8" : "kernel_hy4_q6_r4", &a,
                                    m->map, m->size, (const ds4_hy4_binding[]){HY4_T(in), HY4_W(w), HY4_T(out)},
                                    3, w->dim[1] / (wide ? 16 : 8), 1, 1, 64);
    }
    if (n == 1 && w->type == DS4_TENSOR_F32 && w->dim[1] >= 128 &&
        getenv("DS4_HY4_TILED_F32") == NULL) {
        const uint32_t rows = w->dim[1];
        return ds4_gpu_qwen4_multi_gemv_tensor(in, 1, w->dim[0], 1, &out,
                                              m->map, m->size, &w->abs_offset, &w->type, &rows);
    }
    if (w->type == DS4_TENSOR_Q8_0)
        return ds4_gpu_qwen4_matmul_q8_0_tensor(out, m->map, m->size, w->abs_offset,
                                                 w->dim[0], w->dim[1], in, n);
    if (w->type == DS4_TENSOR_F32 || w->type == DS4_TENSOR_F16)
        return ds4_gpu_qwen4_dense_mm_tensor(out, in, m->map, m->size, w->abs_offset,
                                             w->type, n, w->dim[0], w->dim[1]);
    return ds4_gpu_matmul_quant_tensor(out, m->map, m->size, w->abs_offset,
                                       w->type, w->dim[0], w->dim[1], in, n);
}

static bool hy4_norm(const ds4_model *m, ds4_gpu_tensor *out, const ds4_tensor *w,
                     const ds4_gpu_tensor *in, uint32_t n) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out, in, m->map, m->size, w->abs_offset,
                                                w->dim[0], n, 1e-5f);
}

static bool hy4_hc(ds4_hy4_graph *g, const ds4_model *m, const hy4_hc_weights *h,
                   const ds4_gpu_tensor *streams, uint32_t n) {
    const ds4_hy4_args a = {.rows = n, .width = h->fn->dim[1], .groups = 16};
    if (n == 1 && h->fn->type == DS4_TENSOR_F32 && getenv("DS4_HY4_UNFUSED_IHC") == NULL) {
        return HY4_OP("hc_partial", a, a.groups, n, 1, a.width * 32,
                      HY4_T(streams), HY4_W(h->fn), HY4_T(g->flat_norm)) &&
               HY4_OP("hc_finish", a, HY4_DIM / 256, n, 1, 256,
                      HY4_T(streams), HY4_T(g->flat_norm), HY4_W(h->scale), HY4_W(h->base),
                      HY4_T(g->norm), HY4_T(g->post));
    }
    return ds4_gpu_rms_norm_plain_rows_tensor(g->flat_norm, streams, 4 * HY4_DIM, n, 1e-5f) &&
           hy4_mm(m, g->mix, h->fn, g->flat_norm, n) &&
           HY4_OP("hc_mix", a, HY4_DIM / 256, n, 1, 256, HY4_T(streams), HY4_T(g->mix),
                  HY4_W(h->scale), HY4_W(h->base), HY4_T(g->norm), HY4_T(g->post));
}

static bool hy4_hc_add(ds4_hy4_graph *g, const ds4_model *m, uint32_t n) {
    const ds4_hy4_args a = {.rows = n};
    return HY4_OP("hc_add", a, 4 * HY4_DIM / 256, n, 1, 256,
                  HY4_T(g->streams), HY4_T(g->branch), HY4_T(g->post));
}

static float hy4_fp8(uint8_t u) {
    const unsigned exponent = (u >> 3) & 15, mantissa = u & 7;
    const float x = exponent ? ldexpf((float)(8 + mantissa), (int)exponent - 10)
                             : ldexpf((float)mantissa, -9);
    return u & 128 ? -x : x;
}

typedef struct {
    const ds4_hy4_model *model;
    const int *history;
    const float *norm[HY4_ENGRAM_LAYERS];
    float fp8[256];
    float *out;
    uint32_t pos, stride, rows;
    atomic_bool failed;
} hy4_ngram_job;

static uint32_t hy4_mapped_token(const ds4_hy4_model *h, const int *history, uint32_t pos) {
    const int id = history[pos];
    if (id < 0 || id >= HY4_VOCAB || h->token_map[id] == UINT32_MAX) return h->mapped_pad;
    return h->token_map[id];
}

static bool hy4_read_row(const ds4_hy4_model *h, uint64_t index, uint8_t *raw) {
    const uint64_t offset = h->table_offset + index * h->row_bytes;
    size_t done = 0;
    while (done < h->row_bytes) {
        const ssize_t n = pread(h->fd, raw + done, h->row_bytes - done, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += (size_t)n;
    }
    return true;
}

static void hy4_ngram_rows(void *ctx, uint64_t first, uint64_t end) {
    hy4_ngram_job *job = ctx;
    const ds4_hy4_model *h = job->model;
    for (uint64_t i = first; i < end; i++) {
        const uint32_t layer = i / (job->rows * HY4_ENGRAM_HEADS);
        const uint32_t row = (i / HY4_ENGRAM_HEADS) % job->rows;
        const uint32_t pos = job->pos + row;
        const uint32_t head = i % HY4_ENGRAM_HEADS;
        const uint32_t col = layer * HY4_ENGRAM_HEADS + head;
        float *out = job->out + (layer * job->stride + row) * HY4_LATENT + head * HY4_ENGRAM_DIM;
        if (h->token_map[job->history[pos]] == UINT32_MAX) {
            memset(out, 0, HY4_ENGRAM_DIM * sizeof(float));
            continue;
        }
        const uint64_t *mult = h->multiplier + layer * 3;
        uint64_t hash = (uint64_t)hy4_mapped_token(h, job->history, pos) * mult[0];
        hash ^= (uint64_t)(pos >= 1 ? hy4_mapped_token(h, job->history, pos - 1) : h->mapped_pad) * mult[1];
        if (head >= 8)
            hash ^= (uint64_t)(pos >= 2 ? hy4_mapped_token(h, job->history, pos - 2) : h->mapped_pad) * mult[2];

        uint8_t raw[HY4_ENGRAM_DIM];
        if (!hy4_read_row(h, h->offsets[col] + hash % h->primes[col], raw)) {
            atomic_store(&job->failed, true);
            return;
        }
        const float *weight = job->norm[layer] + head * HY4_ENGRAM_DIM;
        float norm = 0;
        for (uint32_t d = 0; d < HY4_ENGRAM_DIM; d++) {
            float value;
            if (h->table_type == DS4_TENSOR_F8_E4M3) {
                if ((raw[d] & 127) == 127) {
                    atomic_store(&job->failed, true);
                    return;
                }
                value = job->fp8[raw[d]];
            } else {
                const uint8_t *b = raw + (d / 32) * 18;
                const uint32_t segment = d % 32;
                const uint8_t q = b[2 + segment % 16];
                const uint16_t scale = (uint16_t)b[0] | (uint16_t)b[1] << 8;
                value = f16_to_f32(scale) * ((segment < 16 ? q & 15 : q >> 4) - 8);
            }
            out[d] = value;
            norm += value * value;
        }
        const float inv = 1.0f / sqrtf(norm / HY4_ENGRAM_DIM + 1.1920929e-7f);
        for (uint32_t d = 0; d < HY4_ENGRAM_DIM; d++) out[d] *= inv * weight[d];
    }
}

static bool hy4_engram_stage(ds4_hy4_graph *g, const ds4_model *m, const ds4_hy4_model *h,
                             const int *history, uint32_t pos, uint32_t n) {
    hy4_ngram_job job = {.model = h, .history = history, .pos = pos, .stride = g->cap, .rows = n,
                         .out = g->engram_staging};
    for (unsigned i = 0; i < HY4_ENGRAM_LAYERS; i++) {
        const ds4_tensor *norm = h->layer[i * 10 + 1].engram_norm;
        job.norm[i] = (const float *)(m->map + norm->abs_offset);
    }
    if (h->table_type == DS4_TENSOR_F8_E4M3)
        for (unsigned i = 0; i < 256; i++) job.fp8[i] = hy4_fp8((uint8_t)i);
    atomic_init(&job.failed, false);
    ds4_parallel_for_min_rows((uint64_t)n * HY4_ENGRAM_LAYERS * HY4_ENGRAM_HEADS,
                              hy4_ngram_rows, &job, HY4_ENGRAM_HEADS);
    if (atomic_load(&job.failed)) return false;
    const uint64_t stride = (uint64_t)g->cap * HY4_LATENT;
    for (unsigned i = 0; i < HY4_ENGRAM_LAYERS; i++)
        if (!ds4_gpu_tensor_write(g->engram_layer[i], 0, g->engram_staging + i * stride,
                                  (uint64_t)n * HY4_LATENT * sizeof(float))) return false;

    int32_t valid[HY4_PREFILL_MAX];
    for (uint32_t i = 0; i < n; i++) valid[i] = h->token_map[history[pos+i]] != UINT32_MAX;
    return ds4_gpu_tensor_write(g->engram_valid, 0, valid, n * sizeof(valid[0]));
}

static bool hy4_engram_apply(ds4_hy4_graph *g, const ds4_model *m, const hy4_layer_weights *l,
                             uint32_t layer, uint32_t n) {
    const ds4_hy4_args a = {.rows = n};
    return hy4_mm(m, g->gate, l->engram_kv, g->engram_layer[layer / 10], n) &&
           HY4_OP("engram", a, 4, n, 1, 32, HY4_T(g->streams), HY4_T(g->gate),
                  HY4_W(l->engram_bias), HY4_W(l->engram_gate), HY4_W(l->engram_gamma),
                  HY4_W(l->engram_scale), HY4_T(g->engram_valid));
}

static bool hy4_index(ds4_hy4_graph *g, const ds4_model *m, const hy4_layer_weights *l,
                      hy4_cache *c, uint32_t pos, uint32_t n) {
    if (!c->pool) return true;
    ds4_hy4_args a = {.rows = n, .pos = pos, .capacity = g->pending_ring, .theta = 1e7f};
    if (!hy4_mm(m, g->index_k, l->index_k, g->norm, n) ||
        !hy4_norm(m, g->index_k, l->index_norm, g->index_k, n) ||
        !hy4_mm(m, g->pool_v, l->pool_v, g->index_k, n) ||
        !hy4_mm(m, g->pool_gate, l->pool_gate, g->index_k, n) ||
        !HY4_OP("pool_stage", a, 1, n, 1, 128, HY4_T(g->pool_v), HY4_T(g->pool_gate), HY4_T(c->pending)))
        return false;
    const uint32_t complete = (pos + n) / 4 - pos / 4;
    if (complete && !HY4_OP("pool", a, complete, 1, 1, 32,
                             HY4_T(c->pending), HY4_W(l->ape), HY4_W(l->pool_norm), HY4_T(c->pool))) return false;
    if (pos + n <= 2048) return true;

    a.groups = (pos + n - 1) / 4;
    a.heads = 16;
    a.width = 128;
    return hy4_mm(m, g->index_q, l->index_q, g->qa, n) &&
           HY4_OP("rope", a, 16, n, 1, 32, HY4_T(g->index_q)) &&
           hy4_mm(m, g->index_w, l->index_w, g->norm, n) &&
           HY4_OP("scores", a, a.groups, n, 1, 32, HY4_T(c->pool), HY4_T(g->index_q),
                  HY4_T(g->index_w), HY4_T(g->scores)) &&
           ds4_gpu_indexer_topk_tensor(g->selected, g->scores, a.groups, n, HY4_HIST_GROUPS);
}

static bool hy4_attention(ds4_hy4_graph *g, const ds4_model *m, const hy4_layer_weights *l,
                          uint32_t layer, uint32_t pos, uint32_t n) {
    const bool sparse = hy4_sparse(layer);
    const uint32_t heads = sparse ? 32 : 64;
    hy4_cache *c = &g->layer[layer];
    ds4_hy4_args a = {.rows = n, .pos = pos, .heads = heads, .width = 256,
                       .capacity = sparse ? g->ctx : g->ring, .theta = sparse ? 1e7f : 1e4f};
    if (!hy4_mm(m, g->qa, l->qa, g->norm, n) || !hy4_norm(m, g->qa, l->qa_norm, g->qa, n) ||
        !hy4_mm(m, g->q, l->qb, g->qa, n) || !hy4_mm(m, g->kv, l->kv, g->norm, n) ||
        !HY4_OP("rope", a, heads, n, 1, 32, HY4_T(g->q)) ||
        !HY4_OP("store", a, n, 1, 1, 32, HY4_T(g->kv), HY4_W(l->kv_norm), HY4_T(c->kv)) ||
        (sparse && !hy4_index(g, m, l, c, pos, n))) return false;
    a.width = 192;
    a.type = l->kb->type;
    a.row_bytes = l->kb->bytes / (heads * 512);
    if (!HY4_OP("head_mm", a, 128, heads, n, 128, HY4_T(g->q), HY4_W(l->kb), HY4_T(g->absorbed)))
        return false;
    a.type = sparse;
    if (sparse && n == 1 && pos >= 511) {
        const uint32_t count = pos < 2048 ? pos + 1 : 2045 + pos % 4;
        ds4_hy4_args split = a;
        split.groups = (count + 31) / 32;
        if (!HY4_OP("attention_partial", split, split.groups, heads / 4, 1, 128,
                    HY4_T(g->absorbed), HY4_T(g->q), HY4_T(c->kv), HY4_T(g->selected),
                    HY4_W(l->sink), HY4_T(g->attention_partials)) ||
            !HY4_OP("attention_merge", split, heads, 1, 1, 128,
                    HY4_T(g->attention_partials), HY4_T(g->attended))) return false;
    } else if (!HY4_OP("attention", a, heads, n, 1, 128, HY4_T(g->absorbed), HY4_T(g->q), HY4_T(c->kv),
                       HY4_T(g->selected), HY4_W(l->sink), HY4_T(g->attended))) return false;
    a.width = 512;
    a.type = l->vb->type;
    a.row_bytes = l->vb->bytes / (heads * 256);
    if (!HY4_OP("head_mm", a, 64, heads, n, 128, HY4_T(g->attended), HY4_W(l->vb), HY4_T(g->q)))
        return false;
    a.width = heads * 256;
    return hy4_mm(m, g->gate, l->attn_gate, g->norm, n) &&
           HY4_OP("gate", a, (n * a.width + 255) / 256, 1, 1, 256, HY4_T(g->q), HY4_T(g->gate)) &&
           hy4_mm(m, g->branch, l->attn_out, g->q, n) &&
           hy4_norm(m, g->branch, l->attn_post, g->branch, n);
}

static bool hy4_ffn(ds4_hy4_graph *g, const ds4_model *m, const hy4_layer_weights *l, uint32_t n) {
    const hy4_ffn_weights *f = &l->ffn;
    if (!hy4_mm(m, g->gate, f->gate, g->norm, n) || !hy4_mm(m, g->ff_up, f->up, g->norm, n) ||
        !ds4_gpu_swiglu_tensor(g->ff_mid, g->gate, g->ff_up, f->gate->dim[1] * n, 0, 1) ||
        !hy4_mm(m, g->shared, f->down, g->ff_mid, n)) return false;

    if (!l->router) return hy4_norm(m, g->branch, l->ffn_post, g->shared, n);
    const hy4_ffn_weights *e = &l->experts;
    const ds4_hy4_args a = {.rows = n, .type = e->gate->type,
                             .row_bytes = e->gate->bytes / (HY4_EXPERTS * HY4_LATENT)};
    if (!hy4_mm(m, g->router, l->router, g->norm, n) ||
        !HY4_OP("router", a, n, 1, 1, 32, HY4_T(g->router), HY4_W(l->router_bias),
                 HY4_T(g->experts), HY4_T(g->expert_probs)) ||
        !hy4_mm(m, g->latent, l->latent_in, g->norm, n)) return false;
    const bool scalar = getenv("DS4_HY4_SCALAR_EXPERTS") != NULL;
    const bool tiled = n >= 32 && !scalar;
    const bool codebook = !scalar && getenv("DS4_HY4_GENERIC_DECODE") == NULL;
    const ds4_hy4_args down = {.rows = n, .row_bytes = e->down->bytes / (HY4_EXPERTS * HY4_LATENT)};
    const bool experts_ok = tiled ?
        ds4_gpu_hy4_expert_prefill(g->expert_mid, g->expert_down, g->latent, g->experts,
                                    g->expert_lists, g->expert_counts, m->map, m->size,
                                    e->gate->abs_offset, e->up->abs_offset, e->down->abs_offset,
                                    e->gate->type, e->down->type, n) :
        (e->gate->type == DS4_TENSOR_IQ2_XXS && codebook ?
            ds4_gpu_hy4_expert_mid(g->expert_mid, g->latent, g->experts, m->map, m->size,
                                   e->gate->abs_offset, e->up->abs_offset, n) :
            HY4_OP("expert_mid", a, HY4_LATENT / 8, HY4_ACTIVE, n, 256,
                     HY4_T(g->latent), HY4_T(g->experts), HY4_W(e->gate), HY4_W(e->up), HY4_T(g->expert_mid))) &&
        (e->down->type == DS4_TENSOR_IQ2_XXS && codebook && getenv("DS4_HY4_GENERIC_EXPERT_DOWN") == NULL ?
            HY4_OP("expert_down_iq2_xxs", down, HY4_LATENT / 8, HY4_ACTIVE, n, 128,
                   HY4_T(g->expert_mid), HY4_T(g->experts), HY4_W(e->down), HY4_T(g->expert_down)) :
            ds4_gpu_qwen4_moe_down_tensor(g->expert_down, g->expert_mid, g->experts,
                                     m->map, m->size, e->down->abs_offset, e->down->type,
                                     HY4_EXPERTS, n, HY4_ACTIVE, HY4_LATENT, HY4_LATENT, 0, UINT32_MAX));
    return experts_ok &&
           ds4_gpu_qwen4_moe_reduce_tensor(g->latent, g->expert_down, g->expert_probs,
                                           NULL, NULL, NULL, NULL, n, HY4_ACTIVE, HY4_ACTIVE, HY4_LATENT, 0) &&
           hy4_mm(m, g->branch, l->latent_out, g->latent, n) &&
           ds4_gpu_add_tensor(g->branch, g->branch, g->shared, n * HY4_DIM) &&
           hy4_norm(m, g->branch, l->ffn_post, g->branch, n);
}

static bool hy4_forward_rows(ds4_hy4_graph *g, const ds4_engine *engine, const int *history,
                             uint32_t pos, uint32_t n, uint32_t output_rows, float *logits) {
    const ds4_model *m = &engine->model;
    const ds4_hy4_model *h = engine->hy4;
    const ds4_hy4_args a = {.rows = n};
    if (!n || n > g->cap || pos > g->ctx || n > g->ctx - pos || output_rows > n ||
        (output_rows > 1 && (!g->draft || output_rows > HY4_DFLY_ROWS))) return false;
    if (g->draft) g->draft->pos = 0;
    for (uint32_t i = 0; i < n; i++)
        if (history[pos+i] < 0 || history[pos+i] >= HY4_VOCAB) return false;
    if (!hy4_engram_stage(g, m, h, history, pos, n) ||
        !ds4_gpu_tensor_write(g->token_ids, 0, history + pos, (uint64_t)n * sizeof(int)) ||
        !ds4_gpu_begin_commands()) return false;
    bool ok = ds4_gpu_embed_tokens_quant_tensor(g->norm, g->token_ids, m->map, m->size,
                                                  h->embedding->abs_offset, h->embedding->type,
                                                  HY4_VOCAB, n, HY4_DIM) &&
              HY4_OP("repeat", a, 4 * HY4_DIM / 256, n, 1, 256, HY4_T(g->norm), HY4_T(g->streams));
    uint32_t source = 0;
    for (uint32_t i = 0; ok && i < HY4_LAYERS; i++) {
        const hy4_layer_weights *l = &h->layer[i];
        if (l->engram_kv) ok = hy4_engram_apply(g, m, l, i, n);
        ok = ok && hy4_hc(g, m, &l->attn_hc, g->streams, n) &&
             hy4_norm(m, g->norm, l->attn_norm, g->norm, n) &&
             hy4_attention(g, m, l, i, pos, n);
        if (g->draft && source < HY4_DFLY_LAYERS && h->draft->target[source] == i) {
            const uint64_t stride = HY4_HIST_GROUPS * sizeof(int32_t);
            if (ok && output_rows && pos + n > 2048)
                ok = ds4_gpu_tensor_copy(g->draft->indices[source], 0, g->selected,
                                          (n - output_rows) * stride, output_rows * stride);
            source++;
        }
        ok = ok && hy4_hc_add(g, m, n) &&
             hy4_hc(g, m, &l->ffn_hc, g->streams, n) &&
             hy4_norm(m, g->norm, l->ffn_norm, g->norm, n) &&
             hy4_ffn(g, m, l, n) && hy4_hc_add(g, m, n);
    }
    ds4_gpu_tensor *last = NULL;
    ds4_gpu_tensor *output = output_rows > 1 ? g->draft->logits : g->logits;
    if (ok && output_rows) {
        const uint64_t bytes = 4 * HY4_DIM * sizeof(float);
        last = ds4_gpu_tensor_view(g->streams, (n - output_rows) * bytes, output_rows * bytes);
        ok = last && hy4_hc(g, m, &h->head, last, output_rows) &&
             hy4_norm(m, g->norm, h->norm, g->norm, output_rows);
        if (ok && g->draft)
            ok = ds4_gpu_tensor_copy(g->draft->hidden, 0, g->norm, 0, output_rows * HY4_DIM * sizeof(float));
        ok = ok && hy4_mm(m, output, h->output, g->norm, output_rows);
    }
    const bool finished = ds4_gpu_end_commands();
    ds4_gpu_tensor_free(last);
    ok = ok && finished &&
         (!output_rows || ds4_gpu_tensor_read(output, 0, logits, (uint64_t)output_rows * HY4_VOCAB * sizeof(float)));
    if (ok && output_rows && g->draft) {
        g->draft->pos = pos + n;
        g->draft->row = output_rows - 1;
    }
    return ok;
}

static bool hy4_forward(ds4_hy4_graph *g, const ds4_engine *engine, const int *history,
                        uint32_t pos, uint32_t n, float *logits) {
    return hy4_forward_rows(g, engine, history, pos, n, logits ? 1 : 0, logits);
}

static bool hy4_draft_embed(ds4_hy4_graph *g, const ds4_engine *e, ds4_gpu_tensor *out,
                            const int *ids, uint32_t n) {
    const ds4_tensor *w = e->hy4->embedding;
    return ds4_gpu_tensor_write(g->token_ids, 0, ids, (uint64_t)n * sizeof(*ids)) &&
           ds4_gpu_embed_tokens_quant_tensor(out, g->token_ids, e->model.map, e->model.size,
                                              w->abs_offset, w->type, HY4_VOCAB, n, HY4_DIM);
}

static bool hy4_draft_layer(ds4_hy4_graph *g, const ds4_engine *e,
                            uint32_t i, uint32_t pos, uint32_t n) {
    const ds4_model *m = &e->mtp_model;
    const hy4_dfly_weights *w = e->hy4->draft;
    const hy4_layer_weights *l = &w->layer[i];
    hy4_draft_graph *d = g->draft;
    ds4_gpu_tensor *cache = g->layer[w->target[i]].kv;
    ds4_hy4_args a = {.rows = n, .pos = pos, .heads = 32, .width = 256,
                       .capacity = g->ctx, .theta = 1e7f};
    /* The proposals write only future sparse KV rows. The target fills all
     * rows it commits before reading them. The selected history belongs to
     * the last committed target position and is reused by every draft row. */
    if (!hy4_norm(m, g->norm, l->attn_norm, d->states, n) ||
        !hy4_mm(m, g->qa, l->qa, g->norm, n) || !hy4_norm(m, g->qa, l->qa_norm, g->qa, n) ||
        !hy4_mm(m, g->q, l->qb, g->qa, n) || !hy4_mm(m, g->kv, l->kv, g->norm, n) ||
        !HY4_OP("rope", a, 32, n, 1, 32, HY4_T(g->q)) ||
        !HY4_OP("store", a, n, 1, 1, 32, HY4_T(g->kv), HY4_W(l->kv_norm), HY4_T(cache)))
        return false;
    a.width = 192;
    a.type = DS4_TENSOR_F16;
    a.row_bytes = 192 * sizeof(uint16_t);
    if (!HY4_OP("head_mm", a, 128, 32, n, 128, HY4_T(g->q), HY4_T(w->keys[i]), HY4_T(g->absorbed)))
        return false;
    a.type = 2;
    a.groups = d->row;
    if (!HY4_OP("attention", a, 32, n, 1, 128, HY4_T(g->absorbed), HY4_T(g->q), HY4_T(cache),
                 HY4_T(d->indices[i]), HY4_W(l->sink), HY4_T(g->attended))) return false;
    a.width = 512;
    a.type = l->vb->type;
    a.groups = 192 + 256;
    a.row_bytes = l->vb->bytes / l->vb->dim[1];
    ds4_tensor vb = *l->vb;
    vb.abs_offset += 192 * a.row_bytes;
    vb.bytes -= 192 * a.row_bytes;
    if (!HY4_OP("head_mm", a, 64, 32, n, 128, HY4_T(g->attended), HY4_W(&vb), HY4_T(g->q)))
        return false;
    a.width = 32 * 256;
    return hy4_mm(m, g->gate, l->attn_gate, g->norm, n) &&
           HY4_OP("gate", a, n * a.width / 256, 1, 1, 256, HY4_T(g->q), HY4_T(g->gate)) &&
           hy4_mm(m, g->branch, l->attn_out, g->q, n) &&
           hy4_norm(m, g->branch, l->attn_post, g->branch, n) &&
           ds4_gpu_add_tensor(d->states, d->states, g->branch, n * HY4_DIM) &&
           hy4_norm(m, g->norm, l->ffn_norm, d->states, n) &&
           hy4_mm(m, g->gate, l->ffn.gate, g->norm, n) &&
           hy4_mm(m, g->ff_up, l->ffn.up, g->norm, n) &&
           ds4_gpu_swiglu_tensor(g->ff_mid, g->gate, g->ff_up, n * HY4_DFLY_FF, 0, 1) &&
           hy4_mm(m, g->branch, l->ffn.down, g->ff_mid, n) &&
           ds4_gpu_add_tensor(d->states, d->states, g->branch, n * HY4_DIM);
}

static bool hy4_draft_propose(ds4_hy4_graph *g, const ds4_engine *e,
                              uint32_t pos, int *ids, uint32_t n) {
    const ds4_model *m = &e->mtp_model;
    const hy4_dfly_weights *w = e->hy4->draft;
    hy4_draft_graph *d = g->draft;
    const uint64_t stride = HY4_DIM * sizeof(float);
    const ds4_hy4_args a = {.rows = n};
    for (uint32_t i = 1; i < n; i++) ids[i] = (int)w->mask;
    ds4_gpu_tensor *hidden = ds4_gpu_tensor_view(d->hidden, d->row * stride, stride);
    if (!hidden || !ds4_gpu_begin_commands()) {
        ds4_gpu_tensor_free(hidden);
        return false;
    }
    bool ok = hy4_draft_embed(g, e, d->states, ids, n) &&
              hy4_norm(m, d->context_embed, w->enorm, d->states, 1) &&
              hy4_norm(m, d->context_hidden, w->hnorm, hidden, 1) &&
              hy4_mm(m, g->branch, w->fuse, d->context, 1) &&
              HY4_OP("draft_fusion", a, HY4_DIM / 256, 1, 1, 256,
                      HY4_T(d->states), HY4_T(g->branch), HY4_W(w->gate));
    for (uint32_t i = 0; ok && i < HY4_DFLY_LAYERS; i++) ok = hy4_draft_layer(g, e, i, pos, n);
    ok = ok && hy4_norm(m, d->embed, w->norm, d->states, n);
    const bool finished = ds4_gpu_end_commands();
    ds4_gpu_tensor_free(hidden);
    if (!ok || !finished) return false;

    // Hidden correction is autoregressive over the proposals.
    for (uint32_t i = 1; i < n; i++) {
        ds4_gpu_tensor *row = ds4_gpu_tensor_view(d->embed, i * stride, stride);
        if (!row || !ds4_gpu_begin_commands()) {
            ds4_gpu_tensor_free(row);
            return false;
        }
        ok = hy4_draft_embed(g, e, g->branch, ids + i - 1, 1) &&
             hy4_norm(m, d->context_hidden, w->embed_norm, g->branch, 1) &&
             hy4_norm(m, d->context_embed, w->hidden_norm, row, 1) &&
             hy4_mm(m, g->gate, w->correction.gate, d->context, 1) &&
             hy4_mm(m, g->ff_up, w->correction.up, d->context, 1) &&
             ds4_gpu_swiglu_tensor(g->ff_mid, g->gate, g->ff_up, HY4_DIM, 0, 1) &&
             hy4_mm(m, g->branch, w->correction.down, g->ff_mid, 1) &&
             ds4_gpu_add_tensor(g->norm, row, g->branch, HY4_DIM) &&
             hy4_mm(&e->model, g->logits, e->hy4->output, g->norm, 1) &&
             ds4_gpu_qwen4_argmax_tensor(d->top, g->flat_norm, g->logits, HY4_VOCAB);
        const bool complete = ds4_gpu_end_commands();
        ds4_gpu_tensor_free(row);
        if (!ok || !complete || !ds4_gpu_tensor_read(d->top, 0, ids + i, sizeof(*ids)))
            return false;
        if (ids[i] < 0 || ids[i] >= HY4_VOCAB) return false;
    }
    return true;
}

#undef HY4_OP
#undef HY4_W
#undef HY4_T
