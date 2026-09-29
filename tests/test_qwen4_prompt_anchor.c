/* Model-backed check of the Qwen3.8 prompt anchor.
 *
 * A session prefills a prompt head, saves the anchor, runs on (the rest of
 * the prompt, greedy or MTP decoding, then a diverging transcript), and
 * restores the anchor.  Prefilling the prompt tail from there must give the
 * logits and greedy continuation of a session that prefilled the same head
 * and tail without the detour, byte for byte.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/qwen38.gguf make test-qwen4-prompt-anchor
 * with DS4_QWEN_NGRAM_GGUF / DS4_QWEN_MTP_GGUF set as for the server.
 * DS4_TEST_MTP=1 decodes with MTP between the save and the restore. */

#include "ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 8192
#define DECODE_STEPS 24
#define CHECK_STEPS 12

static int failures;

static void check(bool ok, const char *what) {
    printf("%s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

static void die(const char *what, const char *err) {
    fprintf(stderr, "FAIL: %s: %s\n", what, err ? err : "");
    exit(1);
}

static void sync_prefix(ds4_session *s, const ds4_tokens *p, int len) {
    char err[256] = "";
    ds4_tokens head = *p;
    head.len = len;
    if (ds4_session_sync(s, &head, err, sizeof(err)) != 0) die("sync", err);
}

static float *logits_copy(ds4_session *s, int n_vocab) {
    float *out = malloc((size_t)n_vocab * sizeof(float));
    if (!out || ds4_session_copy_logits(s, out, n_vocab) != n_vocab) die("copy logits", NULL);
    return out;
}

static void greedy(ds4_session *s, int *out, int n) {
    char err[256] = "";
    for (int i = 0; i < n; i++) {
        out[i] = ds4_session_argmax(s);
        if (ds4_session_eval(s, out[i], err, sizeof(err)) != 0) die("eval", err);
    }
}

/* Decode past the anchor the way the server does, MTP drafts included. */
static void run_on(ds4_session *s, ds4_engine *e, bool mtp) {
    char err[256] = "";
    int accepted[16];
    for (int done = 0; done < DECODE_STEPS;) {
        const int token = ds4_session_argmax(s);
        if (!mtp) {
            if (ds4_session_eval(s, token, err, sizeof(err)) != 0) die("eval", err);
            done++;
            continue;
        }
        const int n = ds4_session_eval_speculative_argmax(s, token, DECODE_STEPS - done,
                                                          ds4_token_eos(e), accepted, 16,
                                                          err, sizeof(err));
        if (n <= 0) die("speculative eval", err);
        done += n;
    }
}

static int last_turn_marker(ds4_engine *e, const ds4_tokens *p) {
    const int marker = ds4_token_turn_start(e);
    for (int i = p->len - 1; i > 0; i--) {
        if (p->v[i] == marker) return i;
    }
    return -1;
}

static void check_anchor(ds4_engine *e, const ds4_tokens *p, int anchor, bool mtp, const char *name) {
    const int n_vocab = ds4_engine_vocab_size(e);
    ds4_session *ref = NULL, *s = NULL;
    if (ds4_session_create(&ref, e, TEST_CTX) || ds4_session_create(&s, e, TEST_CTX))
        die("session create", NULL);

    sync_prefix(ref, p, anchor);
    float *head_logits = logits_copy(ref, n_vocab);
    sync_prefix(ref, p, p->len);
    float *want = logits_copy(ref, n_vocab);
    int want_tokens[CHECK_STEPS];
    greedy(ref, want_tokens, CHECK_STEPS);

    sync_prefix(s, p, anchor);
    check(ds4_session_anchor_save(s), "anchor saved");
    check(ds4_session_anchor_pos(s) == anchor, "anchor at the prompt head");
    sync_prefix(s, p, p->len);
    run_on(s, e, mtp);
    check(ds4_session_anchor_pos(s) == anchor, "anchor survives decoding");

    check(ds4_session_anchor_restore(s), "anchor restored");
    check(ds4_session_pos(s) == anchor, "session back at the anchor");
    sync_prefix(s, p, p->len);
    float *got = logits_copy(s, n_vocab);
    int got_tokens[CHECK_STEPS];
    greedy(s, got_tokens, CHECK_STEPS);
    char what[160];
    snprintf(what, sizeof(what), "%s: tail logits match a straight prefill", name);
    check(memcmp(got, want, (size_t)n_vocab * sizeof(float)) == 0, what);
    snprintf(what, sizeof(what), "%s: greedy continuation matches", name);
    check(memcmp(got_tokens, want_tokens, sizeof(want_tokens)) == 0, what);

    /* A restore with nothing left to prefill replays the head instead of
     * serving logits from the far end of the old transcript. */
    check(ds4_session_anchor_restore(s), "anchor restored again");
    sync_prefix(s, p, anchor);
    free(got);
    got = logits_copy(s, n_vocab);
    snprintf(what, sizeof(what), "%s: empty tail replays the head", name);
    check(memcmp(got, head_logits, (size_t)n_vocab * sizeof(float)) == 0, what);

    ds4_session_invalidate(s);
    check(ds4_session_anchor_pos(s) == -1, "invalidate drops the anchor");

    free(head_logits);
    free(want);
    free(got);
    ds4_session_free(ref);
    ds4_session_free(s);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: set DS4_TEST_MODEL to a Qwen3.8 GGUF\n");
        return 0;
    }
    const char *mtp_env = getenv("DS4_TEST_MTP");
    const bool mtp = mtp_env && strcmp(mtp_env, "0") != 0;
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .glm_mtp = mtp,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) die("engine open", model);
    if (ds4_token_turn_start(e) < 0) {
        fprintf(stderr, "SKIP: the model has no turn marker\n");
        ds4_engine_close(e);
        return 0;
    }

    /* A few thousand tokens, so the prefill spans several chunks. */
    char *text = malloc(64 * 1024);
    text[0] = '\0';
    for (int i = 0; i < 160; i++) {
        char line[160];
        snprintf(line, sizeof(line),
                 "Line %d: the recurrent state after token %d must survive a rewind.\n",
                 i, i * 17);
        strcat(text, line);
    }
    strcat(text, "Summarize the lines above in one sentence.");
    ds4_tokens p = {0};
    ds4_encode_chat_prompt(e, "You are a careful assistant.", text, DS4_THINK_HIGH, &p);
    free(text);

    const int marker = last_turn_marker(e, &p);
    printf("prompt %d tokens, last turn marker at %d, mtp %s\n", p.len, marker, mtp ? "on" : "off");
    check(marker > 0, "prompt has a turn marker");
    if (marker > 0) check_anchor(e, &p, marker, mtp, "turn marker");
    /* An anchor off the 4-token indexer block grid, with the block completed
     * again by the tail. */
    check_anchor(e, &p, p.len - 3 - (p.len - 3) % 4 + 1, mtp, "mid-block");

    ds4_tokens_free(&p);
    ds4_engine_close(e);
    printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
