/* DFlash2 speculative decode for a Qwen3.8 session: the server path.
 * Included by ds4.c inside the Qwen GPU section, after the session struct
 * and the MTP cycle helpers.
 *
 * One cycle evaluates the token the caller sampled (the anchor) and up to
 * seven drafts in one verify, like ds4_session_qwen4_spec_cycle does for
 * the MTP block:
 *
 *   draft     the block drafter proposes positions p+1..p+7 from the anchor
 *   verify    the target runs [anchor, drafts] taped, all logit rows kept
 *   accept    greedy: drafts that are the target argmax; exact sampling at
 *             temperature > 0: each draft d drawn from the selector's q
 *             over its 16 candidates, kept with min(1, p(d)/q(d)); the first
 *             rejected one is replaced by a sample of max(0, p - q)
 *   keep      the tape restores the state after the last kept row and the
 *             kept rows' captures join the draft's ring
 *
 * The residual sample is not evaluated here: it becomes the token the next
 * ds4_session_sample returns, so it costs nothing extra and is fed as the
 * next cycle's anchor.  The output therefore follows the target
 * distribution exactly, as with sampled MTP. */
#ifndef DS4_QWEN4_DFLASH_SESSION_H
#define DS4_QWEN4_DFLASH_SESSION_H

/* The drafter runs a session when its weights loaded (DS4_QWEN_DFLASH) and
 * DS4_QWEN4_DFL is not 0. */
static bool qwen4_dfl_session_wanted(const ds4_engine *e) {
    const char *env = getenv("DS4_QWEN4_DFL");
    if (env && env[0] == '0') return false;
    return e->glm_mtp && g_qwen4_dfl.loaded &&
           qwen4_dfl_runtime_init(&e->model, &e->weights);
}

static void qwen4_dfl_session_init(ds4_session *s) {
    s->qwen4_forced_set = false;
    s->qwen4_dfl = qwen4_dfl_session_wanted(s->engine);
    if (!s->qwen4_dfl) return;
    s->qwen4_graph.dfl_capture = true;
    fprintf(stderr, "ds4: DFlash2 drafts this session's speculative decode\n");
}

/* The residual sample of a rejected draft, when the caller samples the next
 * token under the same knobs at the same place; anything else drops it. */
static bool qwen4_dfl_forced_take(ds4_session *s, float temperature, int top_k, float top_p, float min_p,
                                  int *token) {
    if (!s->qwen4_forced_set) return false;
    s->qwen4_forced_set = false;
    if (s->checkpoint.len != s->qwen4_forced_len || s->qwen4_graph.fwd_seq != s->qwen4_forced_seq ||
        temperature != s->qwen4_forced_temp || top_k != s->qwen4_forced_top_k ||
        top_p != s->qwen4_forced_top_p || min_p != s->qwen4_forced_min_p) return false;
    *token = s->qwen4_forced;
    return true;
}

static uint32_t qwen4_dfl_max_rows(const ds4_qwen4_gpu_graph *g, int accepted_cap) {
    uint32_t n = QWEN4_DFL_ROWS;
    if ((uint32_t)accepted_cap < n) n = (uint32_t)accepted_cap;
    if (g->ctx_cap - g->pos < n) n = g->ctx_cap - g->pos;
    if (g->cap_tokens < n) n = g->cap_tokens;
    if (g->n_logit_rows < n) n = g->n_logit_rows;
    return n ? n : 1u;
}

static int qwen4_dfl_session_fail(ds4_session *s, char *err, size_t errlen, const char *what) {
    if (errlen) snprintf(err, errlen, "Qwen3.8 DFlash2: %s", what);
    s->checkpoint_valid = false;
    return -1;
}

/* One cycle from first_token at the session's position; returns how many
 * tokens it committed (first_token, then the kept drafts), -1 on error.
 * s->logits are the target's next-token logits afterwards. */
static int ds4_session_qwen4_dfl_cycle(ds4_session *s, int first_token, int eos_token, float temperature,
                                       int top_k, float top_p, float min_p, uint64_t *rng, bool exact_sampling,
                                       int *accepted, int accepted_cap, char *err, size_t errlen) {
    ds4_engine *e = s->engine;
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    const ds4_model *m = &e->model;
    const ds4_weights *w = &e->weights;
    const uint32_t V = DS4_N_VOCAB;
    s->qwen4_forced_set = false;
    if (qwen4_session_replay_if_stale(s, err, errlen) != 0) return -1;
    const uint32_t p = g->pos;
    const bool sampled = exact_sampling && temperature > 0.0f && rng;
    qwen4_dfl_stats *st = &s->qwen4_dfl_st;
    const double t_step = now_sec();
    if (s->qwen4_dfl_last_start > 0.0)
        qwen4_dfl_note_step(s->qwen4_dfl_last_drafted, s->qwen4_dfl_last_tokens, t_step - s->qwen4_dfl_last_start);
    s->qwen4_dfl_last_start = t_step;
    s->qwen4_dfl_last_tokens = 0;
    if (!qwen4_dfl_should_draft()) {
        /* plain step: the forward's captures join the ring on their own */
        s->glm_spec_inside = 1;
        const int rc = ds4_session_eval_internal(s, first_token, false, err, errlen);
        s->glm_spec_inside = 0;
        if (rc != 0) return -1;
        st->plain++;
        s->qwen4_dfl_last_drafted = false;
        s->qwen4_dfl_last_tokens = 1;
        accepted[0] = first_token;
        return 1;
    }
    qwen4_dfl_ring_sync(g, s->checkpoint.v, p);
    qwen4_dfl_proposal prop;
    const double t0 = now_sec();
    if (!qwen4_dfl_draft(m, w, first_token, p, sampled ? temperature : 0.0f, sampled ? rng : NULL, &prop))
        return qwen4_dfl_session_fail(s, err, errlen, "draft failed");
    const double t1 = now_sec();
    uint32_t T = qwen4_dfl_pick_rows(&prop, qwen4_dfl_max_rows(g, accepted_cap));
    int toks[QWEN4_DFL_ROWS];
    toks[0] = first_token;
    for (uint32_t i = 1; i < T; i++) {
        toks[i] = prop.tok[i - 1u];
        if (toks[i] == eos_token) { T = i + 1u; break; }
    }
    if (!s->qwen4_dfl_rows) s->qwen4_dfl_rows = xmalloc((size_t)(QWEN4_DFL_ROWS + 1u) * V * sizeof(float));
    float *rows = s->qwen4_dfl_rows;
    /* the state before the verify, for a rewind to the block start (the
     * server resamples there when a block crosses a tool-syntax boundary) */
    s->qwen4_dfl_snap0 = e->dspark_exact_sampling && qwen4_graph_ensure_snap0(g) && qwen4_graph_state_copy0(g, true);
    g->snap0_valid = s->qwen4_dfl_snap0;
    if (s->qwen4_dfl_snap0) memcpy(rows + (size_t)QWEN4_DFL_ROWS * V, s->logits, (size_t)V * sizeof(float));
    g->snap_after_first = g->snap_after_second = false;
    g->snap_valid = g->snap2_valid = false;
    s->qwen4_lazy = false;
    s->glm_mtp_have = 0;
    s->glm_mtp_have2 = false;
    if (!qwen4_dfl_verify(g, m, w, toks, T, rows)) return qwen4_dfl_session_fail(s, err, errlen, "verify failed");
    const double t2 = now_sec();
    uint32_t k = 0;
    int replacement = -1;
    if (sampled) {
        for (; k + 1u < T; k++) {
            if (!sample_build_probabilities(rows + (size_t)k * V, V, temperature, top_k, top_p, min_p,
                                            s->sample_probs))
                return qwen4_dfl_session_fail(s, err, errlen, "target distribution failed");
            bool kept = false;
            const int t = speculative_sparse_step(s->sample_probs, V, prop.cand[k], prop.q[k], QWEN4_DFL_TOPK,
                                                  prop.pick[k], rng, &kept);
            if (t < 0) return qwen4_dfl_session_fail(s, err, errlen, "residual sampling failed");
            if (kept) continue;
            replacement = t;
            break;
        }
    } else {
        while (k + 1u < T && sample_argmax(rows + (size_t)k * V, V) == prop.tok[k]) k++;
    }
    const uint32_t keep = k + 1u;
    if (!qwen4_dfl_keep(g, m, p, keep, T, toks)) return qwen4_dfl_session_fail(s, err, errlen, "commit failed");
    const double t3 = now_sec();
    qwen4_dfl_policy_learn(&prop, T, k, t2 - t1, (t1 - t0) + (t3 - t2));
    for (uint32_t i = 0; i + 1u < T && i <= k; i++) {
        uint32_t b = (uint32_t)(prop.conf[i] * 10.0f);
        if (b > 9u) b = 9u;
        st->cal_n[b]++;
        if (i < k) st->cal_ok[b]++;
    }
    st->cycles++;
    st->drafted_rows += T - 1u;
    st->accepted += k;
    st->rows_hist[T]++;
    st->t_draft += t1 - t0;
    st->t_verify += t2 - t1;
    st->t_rows[T] += t2 - t1;
    st->t_commit += t3 - t2;
    for (uint32_t i = 0; i < keep; i++) token_vec_push(&s->checkpoint, toks[i]);
    s->checkpoint_valid = true;
    s->mtp_draft_valid = false;
    s->qwen4_rewound = false;
    memcpy(s->logits, rows + (size_t)k * V, (size_t)V * sizeof(float));
    s->qwen4_dfl_pos = p;
    s->qwen4_dfl_keep = keep;
    s->qwen4_dfl_seq = g->fwd_seq;
    if (replacement >= 0) {
        s->qwen4_forced_set = true;
        s->qwen4_forced = replacement;
        s->qwen4_forced_len = s->checkpoint.len;
        s->qwen4_forced_seq = g->fwd_seq;
        s->qwen4_forced_temp = temperature;
        s->qwen4_forced_top_k = top_k;
        s->qwen4_forced_top_p = top_p;
        s->qwen4_forced_min_p = min_p;
    }
    s->qwen4_dfl_last_drafted = true;
    s->qwen4_dfl_last_tokens = keep;
    for (uint32_t i = 0; i < keep; i++) accepted[i] = toks[i];
    return (int)keep;
}

/* ds4_session_rewind_speculative inside the last DFlash2 block: a position
 * the verify ran comes back from its tape, the block start from snap0.
 * Returns the logits of the restored position, or NULL. */
static const float *qwen4_dfl_rewind(ds4_session *s, uint32_t pos) {
    ds4_qwen4_gpu_graph *g = &s->qwen4_graph;
    if (!s->qwen4_dfl || !s->qwen4_dfl_rows || g->fwd_seq != s->qwen4_dfl_seq) return NULL;
    const uint32_t p = s->qwen4_dfl_pos, V = DS4_N_VOCAB;
    if (pos > p && pos < p + s->qwen4_dfl_keep) {
        if (!qwen4_graph_tape_restore(g, p, pos - p) || !ds4_gpu_end_commands()) return NULL;
        return s->qwen4_dfl_rows + (size_t)(pos - p - 1u) * V;
    }
    if (pos == p && s->qwen4_dfl_snap0 && g->snap0_valid && g->snap0_pos == p && qwen4_graph_state_copy0(g, false))
        return s->qwen4_dfl_rows + (size_t)QWEN4_DFL_ROWS * V;
    return NULL;
}

static void qwen4_dfl_session_report(const ds4_session *s) {
    if (!s->qwen4_dfl || !s->qwen4_dfl_st.cycles) return;
    qwen4_dfl_report(&s->qwen4_dfl_st);
}

#endif
