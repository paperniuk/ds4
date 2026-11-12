/* DFlash2 block drafter for Qwen3.8-Flash-Next: its context ring and one
 * draft pass.  Included by ds4.c after the Qwen graph helpers.
 *
 * The target captures the outputs of layers 5/15/25/35/45 (graph dfl_cap).
 * A committed position's capture becomes, per draft layer, a key and value
 * in a ring of 2048 slots.  A draft pass runs eight rows at positions
 * n..n+7 (the anchor token, then mask tokens) through five layers whose
 * attention sees the ring and the block, then scores rows 1..7 with the
 * target's output head and walks the candidate selector over each row's
 * top 16. */
#ifndef DS4_QWEN4_DFLASH_H
#define DS4_QWEN4_DFLASH_H

#include <dispatch/dispatch.h>

#define QWEN4_DFL_ROWS 8u
#define QWEN4_DFL_SCRATCH 32u          /* the Q8 batch products want 32 rows */
#define QWEN4_DFL_WIN 2048u
#define QWEN4_DFL_TOPK 16u
#define QWEN4_DFL_THETA 10000.0f
#define QWEN4_DFL_EPS 1e-6f
#define QWEN4_DFL_DRAFT (QWEN4_DFL_ROWS - 1u)
#define QWEN4_DFL_CAPW (QWEN4_DFL_NTARGET * QWEN4_DFL_HIDDEN)

typedef struct {
    int tok[QWEN4_DFL_DRAFT];          /* proposals for positions n+1.. */
    float conf[QWEN4_DFL_DRAFT];       /* selector max probability, temp 1 */
    int cand[QWEN4_DFL_DRAFT][QWEN4_DFL_TOPK];
    float score[QWEN4_DFL_DRAFT][QWEN4_DFL_TOPK]; /* unary + edge from the chosen predecessor */
    int raw[QWEN4_DFL_DRAFT];          /* each row's argmax, before the selector */
} qwen4_dfl_proposal;

typedef struct {
    bool ready;
    uint32_t ctx_n;                    /* positions [0, ctx_n) are in the ring */
    ds4_gpu_tensor *ring_k[QWEN4_DFL_LAYERS], *ring_v[QWEN4_DFL_LAYERS];
    ds4_gpu_tensor *h, *xn, *dyn, *c, *qkv, *q, *bk, *bv, *att, *o, *h1;
    ds4_gpu_tensor *gt, *up, *act, *mm, *fh, *logits, *sel;
    ds4_gpu_tensor *cin, *fco, *cn, *kv, *part;
    float *mask_row, *rows;
    double t_draft, t_commit, t_stage[5];
    uint64_t n_draft, n_commit;
} qwen4_dfl_runtime;

static qwen4_dfl_runtime g_qwen4_dfl_rt;

static ds4_gpu_tensor *qwen4_dfl_alloc_rows(uint32_t rows, uint32_t width) {
    return ds4_gpu_tensor_alloc((uint64_t)rows * width * sizeof(float));
}

static bool qwen4_dfl_runtime_init(const ds4_model *m, const ds4_weights *w) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    if (rt->ready) return true;
    if (!g_qwen4_dfl.loaded || g_qwen4_dfl.vocab != DS4_N_VOCAB) return false;
    const uint32_t S = QWEN4_DFL_SCRATCH, E = QWEN4_DFL_HIDDEN;
    const uint64_t ring = (uint64_t)QWEN4_DFL_KVDIM * QWEN4_DFL_WIN;
    bool ok = true;
    for (uint32_t l = 0; l < QWEN4_DFL_LAYERS && ok; l++) {
        rt->ring_k[l] = ds4_gpu_tensor_alloc(ring * sizeof(float));
        rt->ring_v[l] = ds4_gpu_tensor_alloc(ring * sizeof(float));
        ok = rt->ring_k[l] && rt->ring_v[l] && ds4_gpu_tensor_fill_f32(rt->ring_k[l], 0.0f, ring) &&
             ds4_gpu_tensor_fill_f32(rt->ring_v[l], 0.0f, ring);
    }
    if (ok) {
        rt->h = qwen4_dfl_alloc_rows(S, E);    rt->xn = qwen4_dfl_alloc_rows(S, E);
        rt->dyn = qwen4_dfl_alloc_rows(S, QWEN4_DFL_DYN);
        rt->c = qwen4_dfl_alloc_rows(S, E);    rt->qkv = qwen4_dfl_alloc_rows(S, QWEN4_DFL_QKV);
        rt->q = qwen4_dfl_alloc_rows(S, QWEN4_DFL_QDIM);
        rt->bk = qwen4_dfl_alloc_rows(S, QWEN4_DFL_KVDIM); rt->bv = qwen4_dfl_alloc_rows(S, QWEN4_DFL_KVDIM);
        rt->att = qwen4_dfl_alloc_rows(S, QWEN4_DFL_QDIM);
        rt->o = qwen4_dfl_alloc_rows(S, E);    rt->h1 = qwen4_dfl_alloc_rows(S, E);
        rt->gt = qwen4_dfl_alloc_rows(S, QWEN4_DFL_FF); rt->up = qwen4_dfl_alloc_rows(S, QWEN4_DFL_FF);
        rt->act = qwen4_dfl_alloc_rows(S, QWEN4_DFL_FF);
        rt->mm = qwen4_dfl_alloc_rows(S, E);   rt->fh = qwen4_dfl_alloc_rows(S, E);
        rt->logits = qwen4_dfl_alloc_rows(S, (uint32_t)DS4_N_VOCAB);
        rt->sel = qwen4_dfl_alloc_rows(S, QWEN4_DFL_RANK);
        rt->cin = qwen4_dfl_alloc_rows(S, QWEN4_DFL_CAPW);
        rt->fco = qwen4_dfl_alloc_rows(S, E);  rt->cn = qwen4_dfl_alloc_rows(S, E);
        rt->kv = qwen4_dfl_alloc_rows(S, 2u * QWEN4_DFL_KVDIM);
        rt->part = ds4_gpu_tensor_alloc(8ull * 32u * 32u * 130u * sizeof(float));
        rt->mask_row = malloc((size_t)E * sizeof(float));
        rt->rows = malloc((size_t)QWEN4_DFL_ROWS * E * sizeof(float));
        ok = rt->h && rt->xn && rt->dyn && rt->c && rt->qkv && rt->q && rt->bk && rt->bv && rt->att &&
             rt->o && rt->h1 && rt->gt && rt->up && rt->act && rt->mm && rt->fh && rt->logits &&
             rt->sel && rt->cin && rt->fco && rt->cn && rt->kv && rt->part && rt->mask_row && rt->rows;
    }
    if (getenv("DS4_QWEN4_DFL_PROFILE"))
        fprintf(stderr, "ds4: DFlash2 head type %u, %" PRIu64 " x %" PRIu64 "\n", w->output->type,
                w->output->dim[0], w->output->dim[1]);
    if (!ok) {
        fprintf(stderr, "ds4: DFlash2 draft: out of memory for its buffers\n");
        return false;
    }
    qwen4_ref_row(m, w->token_embd, QWEN4_DFL_MASK_TOKEN, rt->mask_row);
    rt->ctx_n = 0;
    rt->ready = true;
    return true;
}

/* Commit n capture rows of src (from src_row) as positions pos0..: their
 * keys and values in every layer's ring.  Encodes into the open batch. */
static bool qwen4_dfl_commit(const ds4_model *m, const ds4_gpu_tensor *src, uint32_t src_row,
                             uint32_t pos0, uint32_t n) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    const qwen4_dfl_weights *W = &g_qwen4_dfl;
    if (!rt->ready || pos0 != rt->ctx_n) return false;
    /* positions a full window behind the end are never read again */
    if (n > QWEN4_DFL_WIN) {
        const uint32_t skip = n - QWEN4_DFL_WIN;
        src_row += skip; pos0 += skip; n -= skip;
    }
    const uint64_t row_bytes = (uint64_t)QWEN4_DFL_CAPW * sizeof(float);
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; i += QWEN4_DFL_SCRATCH) {
        const uint32_t k = n - i < QWEN4_DFL_SCRATCH ? n - i : QWEN4_DFL_SCRATCH;
        ok = ds4_gpu_tensor_copy(rt->cin, 0, src, (uint64_t)(src_row + i) * row_bytes, (uint64_t)k * row_bytes) &&
             qwen4_gemv(rt->fco, m, &W->fc, rt->cin, k) &&
             ds4_gpu_qwen4_dfl_rms_tensor(rt->cn, rt->fco, m->map, m->size, W->hidden_norm.abs_offset,
                                          k, QWEN4_DFL_HIDDEN, QWEN4_DFL_EPS);
        for (uint32_t l = 0; l < QWEN4_DFL_LAYERS && ok; l++) {
            const qwen4_dfl_layer *L = &W->layer[l];
            ok = qwen4_gemv(rt->kv, m, &L->kv, rt->cn, k) &&
                 ds4_gpu_qwen4_dfl_ctx_kv_tensor(rt->ring_k[l], rt->ring_v[l], rt->kv, m->map, m->size,
                                                 L->k_norm.abs_offset, k, pos0 + i, QWEN4_DFL_THETA);
        }
    }
    if (ok) rt->ctx_n = pos0 + n;
    return ok;
}

/* DS4_QWEN4_DFL_ATTN_OLD=1: the per (row, head pair) attention kernel. */
static bool qwen4_dfl_attention(uint32_t l, uint32_t n) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    static int old = -1;
    if (old < 0) old = getenv("DS4_QWEN4_DFL_ATTN_OLD") != NULL;
    return old ? ds4_gpu_qwen4_dfl_attn_tensor(rt->att, rt->q, rt->ring_k[l], rt->ring_v[l], rt->bk, rt->bv,
                                               QWEN4_DFL_ROWS, n)
               : ds4_gpu_qwen4_dfl_attn_split_tensor(rt->att, rt->q, rt->ring_k[l], rt->ring_v[l], rt->bk, rt->bv,
                                                     rt->part, QWEN4_DFL_ROWS, n);
}

static bool qwen4_dfl_layer_pass(const ds4_model *m, const qwen4_dfl_layer *L, uint32_t l, uint32_t n) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    const uint32_t R = QWEN4_DFL_ROWS, E = QWEN4_DFL_HIDDEN;
    return ds4_gpu_qwen4_dfl_rms_tensor(rt->xn, rt->h, m->map, m->size, L->in_norm.abs_offset, R, E, QWEN4_DFL_EPS) &&
           qwen4_gemv(rt->dyn, m, &L->attn_dyn, rt->xn, R) &&
           ds4_gpu_qwen4_dfl_conv_tensor(rt->c, rt->xn, rt->dyn, m->map, m->size, L->attn_base.abs_offset,
                                         NULL, R, E, 0u) &&
           qwen4_gemv(rt->qkv, m, &L->qkv, rt->c, R) &&
           ds4_gpu_qwen4_dfl_qk_tensor(rt->q, rt->bk, rt->bv, rt->qkv, m->map, m->size, L->q_norm.abs_offset,
                                       L->k_norm.abs_offset, R, n, QWEN4_DFL_THETA) &&
           qwen4_dfl_attention(l, n) &&
           qwen4_gemv(rt->o, m, &L->o, rt->att, R) &&
           ds4_gpu_qwen4_dfl_conv_tensor(rt->h1, rt->o, rt->dyn, m->map, m->size, L->attn_base.abs_offset,
                                         rt->h, R, E, 1u) &&
           ds4_gpu_qwen4_dfl_rms_tensor(rt->xn, rt->h1, m->map, m->size, L->post_norm.abs_offset, R, E, QWEN4_DFL_EPS) &&
           qwen4_gemv(rt->dyn, m, &L->mlp_dyn, rt->xn, R) &&
           ds4_gpu_qwen4_dfl_conv_tensor(rt->c, rt->xn, rt->dyn, m->map, m->size, L->mlp_base.abs_offset,
                                         NULL, R, E, 0u) &&
           qwen4_gemv(rt->gt, m, &L->gate, rt->c, R) &&
           qwen4_gemv(rt->up, m, &L->up, rt->c, R) &&
           ds4_gpu_qwen4_dfl_swiglu_tensor(rt->act, rt->gt, rt->up, R * QWEN4_DFL_FF) &&
           qwen4_gemv(rt->mm, m, &L->down, rt->act, R) &&
           ds4_gpu_qwen4_dfl_conv_tensor(rt->h, rt->mm, rt->dyn, m->map, m->size, L->mlp_base.abs_offset,
                                         rt->h1, R, E, 1u);
}

static float qwen4_dfl_bf16f(uint16_t v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Top 16 of a row by value, highest first. */
static void qwen4_dfl_top16(const float *x, uint64_t n, int *ids, float *vals) {
    uint32_t k = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (k == QWEN4_DFL_TOPK && v <= vals[k - 1u]) continue;
        uint32_t j = k < QWEN4_DFL_TOPK ? k++ : k - 1u;
        while (j > 0 && vals[j - 1u] < v) { vals[j] = vals[j - 1u]; ids[j] = ids[j - 1u]; j--; }
        vals[j] = v; ids[j] = (int)i;
    }
}

/* Selector edge weights of predecessor `pred` to the 16 candidates. */
static void qwen4_dfl_edges(const float *sel, int pred, const int *cand, float *out) {
    const uint16_t *P = g_qwen4_dfl.pred_cb + (uint64_t)pred * QWEN4_DFL_RANK;
    float ps[QWEN4_DFL_RANK];
    for (uint32_t d = 0; d < QWEN4_DFL_RANK; d++) ps[d] = qwen4_dfl_bf16f(P[d]) * sel[d];
    for (uint32_t c = 0; c < QWEN4_DFL_TOPK; c++) {
        const uint16_t *S = g_qwen4_dfl.succ_cb + (uint64_t)cand[c] * QWEN4_DFL_RANK;
        float s = 0.0f;
        for (uint32_t d = 0; d < QWEN4_DFL_RANK; d++) s += ps[d] * qwen4_dfl_bf16f(S[d]);
        out[c] = s;
    }
}

/* One draft pass from the anchor token at position n (ctx_n == n): the
 * greedy selector walk over positions n+1..n+7. */
static bool qwen4_dfl_draft(const ds4_model *m, const ds4_weights *w, int anchor, uint32_t n,
                            qwen4_dfl_proposal *out) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    const qwen4_dfl_weights *W = &g_qwen4_dfl;
    const uint32_t R = QWEN4_DFL_ROWS, E = QWEN4_DFL_HIDDEN;
    if (!rt->ready || rt->ctx_n != n || anchor < 0 || anchor >= (int)DS4_N_VOCAB) return false;
    const double t0 = now_sec();
    qwen4_ref_row(m, w->token_embd, (uint64_t)anchor, rt->rows);
    for (uint32_t r = 1; r < R; r++) memcpy(rt->rows + (uint64_t)r * E, rt->mask_row, (size_t)E * sizeof(float));
    bool ok = ds4_gpu_tensor_write(rt->h, 0, rt->rows, (uint64_t)R * E * sizeof(float)) &&
              glm_graph_begin_commands_if_needed();
    const bool prof = getenv("DS4_QWEN4_DFL_PROFILE") != NULL;
    double tp = now_sec();
#define DFL_STAGE(i_) do { if (prof && ok) { ok = ds4_gpu_end_commands() && glm_graph_begin_commands_if_needed(); \
        const double t_ = now_sec(); rt->t_stage[i_] += t_ - tp; tp = t_; } } while (0)
    if (prof) { const double t_ = now_sec(); rt->t_stage[0] += t_ - tp; tp = t_; }
    for (uint32_t l = 0; l < QWEN4_DFL_LAYERS && ok; l++) ok = qwen4_dfl_layer_pass(m, &W->layer[l], l, n);
    DFL_STAGE(1);
    ok = ok && ds4_gpu_qwen4_dfl_rms_tensor(rt->fh, rt->h, m->map, m->size, W->norm.abs_offset, R, E, QWEN4_DFL_EPS) &&
         (W->head_rows ? qwen4_gemv(rt->logits, m, &W->head, rt->fh, R)
                       : qwen4_gemv(rt->logits, m, w->output, rt->fh, R));
    DFL_STAGE(2);
    ok = ok && qwen4_gemv(rt->sel, m, &W->sel, rt->fh, R);
    if (!ds4_gpu_end_commands()) ok = false;
    if (prof) { const double t_ = now_sec(); rt->t_stage[3] += t_ - tp; tp = t_; }
#undef DFL_STAGE
    if (!ok) return false;
    const float *logits = ds4_gpu_tensor_contents(rt->logits);
    const float *sel = ds4_gpu_tensor_contents(rt->sel);
    if (!logits || !sel) return false;
    const uint64_t head_n = W->head_rows ? W->head_rows : DS4_N_VOCAB;
    float uni[QWEN4_DFL_DRAFT][QWEN4_DFL_TOPK];
    float *uni_p = &uni[0][0];
    dispatch_apply(QWEN4_DFL_DRAFT, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
        qwen4_dfl_top16(logits + (uint64_t)(i + 1u) * head_n, head_n, out->cand[i], uni_p + i * QWEN4_DFL_TOPK);
    });
    int pred = anchor;
    for (uint32_t i = 0; i < QWEN4_DFL_DRAFT; i++) {
        float edge[QWEN4_DFL_TOPK];
        qwen4_dfl_edges(sel + (uint64_t)(i + 1u) * QWEN4_DFL_RANK, pred, out->cand[i], edge);
        uint32_t best = 0;
        float mx = -FLT_MAX;
        for (uint32_t c = 0; c < QWEN4_DFL_TOPK; c++) {
            out->score[i][c] = uni[i][c] + edge[c];
            if (out->score[i][c] > mx) { mx = out->score[i][c]; best = c; }
        }
        float sum = 0.0f;
        for (uint32_t c = 0; c < QWEN4_DFL_TOPK; c++) sum += expf(out->score[i][c] - mx);
        out->conf[i] = 1.0f / sum;
        out->raw[i] = out->cand[i][0];
        out->tok[i] = out->cand[i][best];
        pred = out->tok[i];
    }
    if (prof) rt->t_stage[4] += now_sec() - tp;
    rt->t_draft += now_sec() - t0;
    rt->n_draft++;
    return true;
}

/* The verify-row policy learns as it goes: the verify time of each row
 * count (EMA, seeded with M1 Max IQ3_XXS measurements at short context),
 * the draft time, and the acceptance rate of each selector-confidence
 * decile (seeded with a prior of ten observations at the decile's middle;
 * the selector is roughly calibrated only above 0.8). */
typedef struct {
    bool init;
    float row_ms[QWEN4_DFL_ROWS + 1u];
    float draft_ms;
    float cal_n[10], cal_ok[10];
} qwen4_dfl_policy;

static qwen4_dfl_policy g_qwen4_dfl_pol;

static void qwen4_dfl_policy_init(qwen4_dfl_policy *pol) {
    /* IQ3_XXS verify ms on M1 Max; five to eight rows with the mixed-quant
     * batch product (kernel_qwen4_batch_mm_kq) */
    static const float seed[QWEN4_DFL_ROWS + 1u] = { 0.0f, 31.0f, 42.4f, 61.7f, 75.2f, 86.0f, 93.0f, 100.0f, 107.0f };
    memcpy(pol->row_ms, seed, sizeof(seed));
    pol->draft_ms = 7.0f;
    for (uint32_t b = 0; b < 10u; b++) {
        pol->cal_n[b] = 10.0f;
        pol->cal_ok[b] = 10.0f * (0.05f + 0.1f * (float)b);
    }
    pol->init = true;
}

static float qwen4_dfl_accept_p(const qwen4_dfl_policy *pol, float conf) {
    uint32_t b = (uint32_t)(conf * 10.0f);
    if (b > 9u) b = 9u;
    return pol->cal_ok[b] / pol->cal_n[b];
}

/* DS4_QWEN4_DFL_ROWS=T fixes the verify rows; otherwise the count that
 * maximizes expected tokens per millisecond. */
static uint32_t qwen4_dfl_pick_rows(const qwen4_dfl_proposal *p, uint32_t max_rows) {
    qwen4_dfl_policy *pol = &g_qwen4_dfl_pol;
    if (!pol->init) qwen4_dfl_policy_init(pol);
    const char *env = getenv("DS4_QWEN4_DFL_ROWS");
    const uint32_t fixed = env && env[0] ? (uint32_t)strtoul(env, NULL, 10) : 0;
    if (fixed) return fixed < max_rows ? fixed : max_rows;
    uint32_t best_t = 1;
    float best = 1.0f / (pol->row_ms[1] + pol->draft_ms), e = 1.0f, surv = 1.0f;
    for (uint32_t t = 2; t <= max_rows; t++) {
        surv *= qwen4_dfl_accept_p(pol, p->conf[t - 2u]);
        e += surv;
        const float v = e / (pol->row_ms[t] + pol->draft_ms);
        if (v > best) { best = v; best_t = t; }
    }
    return best_t;
}

static void qwen4_dfl_policy_learn(const qwen4_dfl_proposal *p, uint32_t T, uint32_t k, double verify_s,
                                   double draft_s) {
    qwen4_dfl_policy *pol = &g_qwen4_dfl_pol;
    const float a = 0.1f;
    /* clamped samples: a first verify of a row count compiles pipelines
     * and allocates the tape, which would otherwise lock that count out */
    const float v = fminf(fmaxf((float)(1e3 * verify_s), 0.5f * pol->row_ms[T]), 1.5f * pol->row_ms[T]);
    const float d = fminf(fmaxf((float)(1e3 * draft_s), 0.5f * pol->draft_ms), 1.5f * pol->draft_ms);
    pol->row_ms[T] += a * (v - pol->row_ms[T]);
    pol->draft_ms += a * (d - pol->draft_ms);
    for (uint32_t i = 0; i + 1u < T && i <= k; i++) {
        uint32_t b = (uint32_t)(p->conf[i] * 10.0f);
        if (b > 9u) b = 9u;
        pol->cal_n[b] += 1.0f;
        if (i < k) pol->cal_ok[b] += 1.0f;
    }
}

typedef struct {
    uint64_t cycles, drafted_rows, accepted, emitted;
    double t_draft, t_verify, t_commit;
    uint64_t rows_hist[QWEN4_DFL_ROWS + 1u];
    double t_rows[QWEN4_DFL_ROWS + 1u];
    uint64_t cal_n[10], cal_ok[10];    /* verified drafts by confidence decile, and accepted */
} qwen4_dfl_stats;

/* Prefill commit: the capture rows of the forward that just ran from pos0. */
static bool qwen4_dfl_commit_forward(ds4_qwen4_gpu_graph *g, const ds4_model *m, uint32_t pos0) {
    if (g->pos <= pos0) return true;
    return glm_graph_begin_commands_if_needed() &&
           qwen4_dfl_commit(m, g->dfl_cap, 0, pos0, g->pos - pos0) &&
           ds4_gpu_end_commands();
}

/* One greedy DFlash2 cycle from `anchor` (sampled, not yet fed) at g->pos:
 * draft, verify T rows with the tape, keep the accepted prefix, commit its
 * captures.  out[] receives the accepted drafts then the target's next
 * token; returns how many (>= 1), 0 on failure. */
static uint32_t qwen4_dfl_cycle(ds4_qwen4_gpu_graph *g, const ds4_model *m, const ds4_weights *w,
                                int anchor, uint32_t max_rows, float *rows, int *out, qwen4_dfl_stats *st) {
    const uint32_t V = DS4_N_VOCAB, p = g->pos;
    qwen4_dfl_proposal prop;
    double t0 = now_sec();
    const double t_draft0 = t0;
    if (!qwen4_dfl_draft(m, w, anchor, p, &prop)) {
        fprintf(stderr, "ds4: DFlash2 draft failed (ring holds %u positions, anchor at %u)\n", g_qwen4_dfl_rt.ctx_n, p);
        return 0;
    }
    double t1 = now_sec();
    st->t_draft += t1 - t0;
    const uint32_t T = qwen4_dfl_pick_rows(&prop, max_rows);
    int toks[QWEN4_DFL_ROWS];
    toks[0] = anchor;
    for (uint32_t i = 1; i < T; i++) toks[i] = prop.tok[i - 1u];
    g->dfl_tape = T > 1u;
    if (g->dfl_tape && !qwen4_graph_ensure_tape(g)) { g->dfl_tape = false; return 0; }
    static int bmm3 = -1;
    /* opt-in: at three or four rows the padded batch product measured
     * slower than the exact rows kernel on M1 Max */
    if (bmm3 < 0) { const char *e = getenv("DS4_QWEN4_DFL_BMM3"); bmm3 = e && e[0] && e[0] != '0'; }
    if (bmm3 && T >= 3u) g_qwen4_batch_mm_min = 3u;
    static int kq_min = -1;
    if (kq_min < 0) { const char *e = getenv("DS4_QWEN4_DFL_KQ_MIN"); kq_min = e && e[0] ? atoi(e) : 5; }
    g_qwen4_batch_kq_min = kq_min > 0 ? (uint32_t)kq_min : 0u;
    ds4_gpu_qwen4_set_dfl_verify(T > 1u);
    const bool ok = qwen4_graph_forward_tokens(g, m, w, toks, T, rows, T > 1u);
    ds4_gpu_qwen4_set_dfl_verify(false);
    g_qwen4_batch_mm_min = 5u;
    g_qwen4_batch_kq_min = 0u;
    g->dfl_tape = false;
    if (!ok) { fprintf(stderr, "ds4: DFlash2 verify of %u rows failed\n", T); return 0; }
    t0 = now_sec();
    st->t_verify += t0 - t1;
    st->t_rows[T] += t0 - t1;
    uint32_t k = 0;
    while (k + 1u < T && sample_argmax(rows + (uint64_t)k * V, V) == prop.tok[k]) k++;
    for (uint32_t i = 0; i + 1u < T && i <= k; i++) {
        uint32_t b = (uint32_t)(prop.conf[i] * 10.0f);
        if (b > 9u) b = 9u;
        st->cal_n[b]++;
        if (i < k) st->cal_ok[b]++;
    }
    const uint32_t keep = k + 1u;
    if (keep < T && !qwen4_graph_tape_restore(g, p, keep)) {
        fprintf(stderr, "ds4: DFlash2 tape restore failed\n");
        return 0;
    }
    if (!glm_graph_begin_commands_if_needed() || !qwen4_dfl_commit(m, g->dfl_cap, 0, p, keep) ||
        !ds4_gpu_end_commands()) {
        fprintf(stderr, "ds4: DFlash2 commit failed\n");
        return 0;
    }
    const double t2 = now_sec();
    st->t_commit += t2 - t0;
    qwen4_dfl_policy_learn(&prop, T, k, t0 - t1, (t1 - t_draft0) + (t2 - t0));
    for (uint32_t i = 0; i < k; i++) out[i] = prop.tok[i];
    out[k] = sample_argmax(rows + (uint64_t)k * V, V);
    st->cycles++;
    st->drafted_rows += T - 1u;
    st->accepted += k;
    st->rows_hist[T]++;
    return k + 1u;
}

static void qwen4_dfl_report(const qwen4_dfl_stats *st) {
    if (!st->cycles) return;
    const double c = (double)st->cycles;
    fprintf(stderr, "ds4: DFlash2 %" PRIu64 " cycles, %.2f tokens/cycle, %.2f drafts verified/cycle, "
            "accepted %.1f%%; ms/cycle draft %.2f verify %.2f commit %.2f; rows",
            st->cycles, (double)(st->accepted + st->cycles) / c, (double)st->drafted_rows / c,
            st->drafted_rows ? 100.0 * (double)st->accepted / (double)st->drafted_rows : 0.0,
            1e3 * st->t_draft / c, 1e3 * st->t_verify / c, 1e3 * st->t_commit / c);
    for (uint32_t t = 1; t <= QWEN4_DFL_ROWS; t++) fprintf(stderr, " %" PRIu64, st->rows_hist[t]);
    fprintf(stderr, "\nds4: DFlash2 verify ms by rows:");
    for (uint32_t t = 1; t <= QWEN4_DFL_ROWS; t++)
        fprintf(stderr, " %.1f", st->rows_hist[t] ? 1e3 * st->t_rows[t] / (double)st->rows_hist[t] : 0.0);
    fprintf(stderr, "\nds4: DFlash2 acceptance by confidence decile:");
    for (uint32_t b = 0; b < 10u; b++)
        fprintf(stderr, " %.2f(%" PRIu64 ")", st->cal_n[b] ? (double)st->cal_ok[b] / (double)st->cal_n[b] : 0.0, st->cal_n[b]);
    fprintf(stderr, "\n");
}

/* DS4_QWEN4_DFL_BENCH: GPU time of the draft's pieces, 20 reps each. */
static void qwen4_dfl_bench(const ds4_model *m, const ds4_weights *w, uint32_t n) {
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    const qwen4_dfl_layer *L = &g_qwen4_dfl.layer[0];
    const uint32_t R = QWEN4_DFL_ROWS, E = QWEN4_DFL_HIDDEN;
    const int reps = 20;
    for (int k = 0; k < 14; k++) {
        bool ok = glm_graph_begin_commands_if_needed() && ds4_gpu_end_commands();
        const double t0 = now_sec();
        ok = glm_graph_begin_commands_if_needed();
        for (int i = 0; i < reps && ok; i++) {
            switch (k) {
            case 0: ok = qwen4_gemv(rt->gt, m, &L->gate, rt->c, R); break;
            case 1: ok = qwen4_gemv(rt->mm, m, &L->down, rt->act, R); break;
            case 2: ok = qwen4_gemv(rt->qkv, m, &L->qkv, rt->c, R); break;
            case 3: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, R); break;
            case 4: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, 1); break;
            case 5: ok = qwen4_dfl_attention(0, n); break;
            case 6: ok = qwen4_dfl_layer_pass(m, L, 0, n); break;
            case 7: ok = ds4_gpu_qwen4_dfl_conv_tensor(rt->c, rt->xn, rt->dyn, m->map, m->size, L->attn_base.abs_offset, NULL, R, E, 0u) &&
                         ds4_gpu_qwen4_dfl_rms_tensor(rt->xn, rt->h, m->map, m->size, L->in_norm.abs_offset, R, E, QWEN4_DFL_EPS); break;
            case 8: ok = qwen4_gemv(rt->gt, m, &L->gate, rt->c, 1); break;
            case 9: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, 9); break;
            case 10: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, 16); break;
            case 11: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, 2); break;
            case 12: ok = qwen4_gemv(rt->logits, m, w->output, rt->fh, 4); break;
            case 13: ok = !g_qwen4_dfl.head_rows || qwen4_gemv(rt->logits, m, &g_qwen4_dfl.head, rt->fh, R); break;
            }
        }
        ok = ok && ds4_gpu_end_commands();
        static const char *names[] = { "gate 8 rows", "down 8 rows", "qkv 8 rows", "head 8 rows", "head 1 row",
                                       "attention", "layer pass", "conv+rms", "gate 1 row", "head 9 rows",
                                       "head 16 rows", "head 2 rows", "head 4 rows", "q8 head 8 rows" };
        printf("dfl bench %-12s %.3f ms%s\n", names[k], 1e3 * (now_sec() - t0) / reps, ok ? "" : " FAILED");
    }
}

/* DS4_QWEN4_DFL_SELFTEST=<capture dump>[:start[:count]]: commit the dump's
 * rows position by position and draft at each, printing the proposals and
 * the greedy acceptance against the dump's own tokens.  For checking the
 * GPU drafter against the reference (ref.py) and timing it. */
static int qwen4_dfl_selftest(const ds4_model *m, const ds4_weights *w, const char *spec) {
    char path[1024];
    snprintf(path, sizeof(path), "%s", spec);
    uint32_t start = 64, count = 64;
    char *colon = strchr(path, ':');
    if (colon) {
        *colon = '\0';
        start = (uint32_t)strtoul(colon + 1, &colon, 10);
        if (*colon == ':') count = (uint32_t)strtoul(colon + 1, NULL, 10);
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "ds4: DFlash2 selftest: %s: %s\n", path, strerror(errno)); return 1; }
    const uint64_t rec = 8u + (uint64_t)QWEN4_DFL_CAPW * sizeof(float);
    fseek(fp, 0, SEEK_END);
    const uint64_t N = (uint64_t)ftell(fp) / rec;
    fseek(fp, 0, SEEK_SET);
    int *tok = malloc(N * sizeof(int));
    ds4_gpu_tensor *cap = ds4_gpu_tensor_alloc(N * QWEN4_DFL_CAPW * sizeof(float));
    float *dst = cap ? ds4_gpu_tensor_contents(cap) : NULL;
    bool ok = tok && dst && qwen4_dfl_runtime_init(m, w);
    for (uint64_t i = 0; i < N && ok; i++) {
        uint32_t pos;
        ok = fread(&pos, 4, 1, fp) == 1 && fread(&tok[i], 4, 1, fp) == 1 &&
             fread(dst + i * QWEN4_DFL_CAPW, sizeof(float), QWEN4_DFL_CAPW, fp) == QWEN4_DFL_CAPW && pos == i;
    }
    fclose(fp);
    if (!ok) { fprintf(stderr, "ds4: DFlash2 selftest: bad dump or init\n"); return 1; }
    if (start + count + QWEN4_DFL_ROWS > N) count = N > start + QWEN4_DFL_ROWS ? (uint32_t)(N - start - QWEN4_DFL_ROWS) : 0;
    qwen4_dfl_runtime *rt = &g_qwen4_dfl_rt;
    uint64_t hist[QWEN4_DFL_ROWS] = {0}, acc_sum = 0, raw_sum = 0;
    for (uint32_t n = start; n < start + count && ok; n++) {
        const double c0 = now_sec();
        ok = glm_graph_begin_commands_if_needed() &&
             qwen4_dfl_commit(m, cap, rt->ctx_n, rt->ctx_n, n - rt->ctx_n) &&
             ds4_gpu_end_commands();
        rt->t_commit += now_sec() - c0;
        rt->n_commit++;
        qwen4_dfl_proposal p;
        ok = ok && qwen4_dfl_draft(m, w, tok[n], n, &p);
        if (!ok) break;
        uint32_t a = 0, b = 0;
        while (a < QWEN4_DFL_DRAFT && p.tok[a] == tok[n + 1u + a]) a++;
        while (b < QWEN4_DFL_DRAFT && p.raw[b] == tok[n + 1u + b]) b++;
        hist[a]++; acc_sum += a; raw_sum += b;
        printf("dfl %u toks", n);
        for (uint32_t i = 0; i < QWEN4_DFL_DRAFT; i++) printf(" %d", p.tok[i]);
        printf(" conf");
        for (uint32_t i = 0; i < QWEN4_DFL_DRAFT; i++) printf(" %.3f", p.conf[i]);
        printf(" acc %u\n", a);
    }
    if (ok && getenv("DS4_QWEN4_DFL_BENCH")) qwen4_dfl_bench(m, w, rt->ctx_n);
    if (ok && count) {
        printf("dfl cycles %u mean accepted %.3f raw argmax %.3f hist", count, (double)acc_sum / count,
               (double)raw_sum / count);
        for (uint32_t i = 0; i < QWEN4_DFL_ROWS; i++) printf(" %" PRIu64, hist[i]);
        printf("\ndfl draft %.2f ms, commit %.2f ms (first commit included)\n",
               1e3 * rt->t_draft / (double)rt->n_draft, 1e3 * rt->t_commit / (double)rt->n_commit);
        printf("dfl stages ms: upload %.2f layers %.2f head %.2f sel %.2f cpu %.2f\n",
               1e3 * rt->t_stage[0] / rt->n_draft, 1e3 * rt->t_stage[1] / rt->n_draft,
               1e3 * rt->t_stage[2] / rt->n_draft, 1e3 * rt->t_stage[3] / rt->n_draft,
               1e3 * rt->t_stage[4] / rt->n_draft);
    }
    free(tok);
    ds4_gpu_tensor_free(cap);
    return ok ? 0 : 1;
}

#endif
