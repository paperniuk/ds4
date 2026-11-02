/* Real-model check of the compact Qwen3.8 KV cache (int8 K/V, half indexer
 * keys) against f16 K/V on one long text.
 *
 * Both modes prefill the same N tokens, then teacher-force T more; each row
 * of next-token logits is compared (KL, top-1 agreement, target NLL).  The
 * int8 session is saved to a payload, reloaded into a fresh session (exact
 * logits), and the f16 engine must refuse that payload.
 *
 * Usage: test_qwen4_kv_q8 QWEN_GGUF TEXT_FILE [N] [T] */
#include "../ds4.c"
#include <assert.h>

enum { DEFAULT_N = 16384, DEFAULT_T = 128 };

typedef struct {
    double kl_mean, kl_max, nll_q8, nll_f16;
    uint32_t top1_same, rows;
} kv_q8_stats;

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(2); }
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *text = xmalloc((size_t)n + 1);
    if (fread(text, 1, (size_t)n, fp) != (size_t)n) { perror(path); exit(2); }
    text[n] = '\0';
    fclose(fp);
    return text;
}

static void log_softmax(const float *x, double *out) {
    double mx = -INFINITY, sum = 0.0;
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) if (x[i] > mx) mx = x[i];
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) sum += exp((double)x[i] - mx);
    const double lse = mx + log(sum);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) out[i] = (double)x[i] - lse;
}

static uint32_t argmax_row(const float *x) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < DS4_N_VOCAB; i++) if (x[i] > x[best]) best = i;
    return best;
}

static kv_q8_stats compare_rows(const float *f16, const float *q8, const int *targets, uint32_t rows) {
    kv_q8_stats st = {.rows = rows};
    double *lp = xmalloc(DS4_N_VOCAB * sizeof(double));
    double *lq = xmalloc(DS4_N_VOCAB * sizeof(double));
    for (uint32_t r = 0; r < rows; r++) {
        const float *a = f16 + (size_t)r * DS4_N_VOCAB, *b = q8 + (size_t)r * DS4_N_VOCAB;
        log_softmax(a, lp);
        log_softmax(b, lq);
        double kl = 0.0;
        for (uint32_t i = 0; i < DS4_N_VOCAB; i++) kl += exp(lp[i]) * (lp[i] - lq[i]);
        st.kl_mean += kl / rows;
        if (kl > st.kl_max) st.kl_max = kl;
        st.nll_f16 -= lp[targets[r]] / rows;
        st.nll_q8 -= lq[targets[r]] / rows;
        st.top1_same += argmax_row(a) == argmax_row(b);
    }
    free(lp);
    free(lq);
    return st;
}

static ds4_engine *open_engine(const char *model, int ctx, bool q8) {
    if (q8) assert(unsetenv("DS4_QWEN4_KV_F16") == 0);
    else assert(setenv("DS4_QWEN4_KV_F16", "1", 1) == 0);
    ds4_engine_options opt = {.model_path = model, .context_size = ctx, .backend = DS4_BACKEND_METAL};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine, &opt) == 0);
    assert(ds4_engine_is_qwen4(engine));
    assert(g_qwen4_kv_q8 == q8);
    return engine;
}

/* Prefill tokens[0..n) and teacher-force tokens[n..n+t), keeping the logits
 * that predict each forced token. */
static ds4_session *run_mode(ds4_engine *engine, const ds4_tokens *tokens, int n, int t,
                             int ctx, float *logits) {
    ds4_session *s = NULL;
    assert(ds4_session_create(&s, engine, ctx) == 0);
    ds4_tokens prompt = {.v = tokens->v, .len = n, .cap = n};
    char err[256] = {0};
    const double t0 = now_sec();
    if (ds4_session_sync(s, &prompt, err, sizeof(err))) { fprintf(stderr, "sync: %s\n", err); exit(1); }
    const double t1 = now_sec();
    for (int i = 0; i < t; i++) {
        memcpy(logits + (size_t)i * DS4_N_VOCAB, s->logits, DS4_N_VOCAB * sizeof(float));
        if (ds4_session_eval(s, tokens->v[n + i], err, sizeof(err))) { fprintf(stderr, "eval: %s\n", err); exit(1); }
    }
    const double t2 = now_sec();
    printf("  prefill %d tokens %.1f tok/s, decode %d tokens %.1f tok/s\n",
           n, n / (t1 - t0), t, t / (t2 - t1));
    fflush(stdout);
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s QWEN_GGUF TEXT_FILE [N] [T]\n", argv[0]);
        return 2;
    }
    const int n = argc > 3 ? atoi(argv[3]) : DEFAULT_N, t = argc > 4 ? atoi(argv[4]) : DEFAULT_T;
    const int ctx = n + t + 16;
    char *text = read_text(argv[2]);
    float *q8_logits = NULL, *f16_logits = NULL;
    ds4_tokens tokens = {0};
    char path[] = "/tmp/ds4-kv-q8-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);

    /* int8 first: its payload is then offered to the f16 engine */
    ds4_engine *engine = open_engine(argv[1], ctx, true);
    /* the vocabulary size is known only once the model is open */
    const size_t row_bytes = (size_t)DS4_N_VOCAB * sizeof(float);
    ds4_tokenize_text(engine, text, &tokens);
    assert(tokens.len >= n + t + 1);
    q8_logits = xmalloc((size_t)t * row_bytes);
    f16_logits = xmalloc((size_t)t * row_bytes);
    puts("int8 KV:");
    ds4_session *s = run_mode(engine, &tokens, n, t, ctx, q8_logits);
    char err[256] = {0};
    FILE *fp = fopen(path, "wb");
    assert(fp && ds4_session_save_payload(s, fp, err, sizeof(err)) == 0);
    fclose(fp);
    const uint64_t payload_bytes = ds4_session_payload_bytes(s);
    assert(ds4_session_eval(s, tokens.v[n + t], err, sizeof(err)) == 0);
    float *after = xmalloc(row_bytes);
    memcpy(after, s->logits, row_bytes);
    ds4_session_free(s);

    ds4_session *r = NULL;
    assert(ds4_session_create(&r, engine, ctx) == 0);
    fp = fopen(path, "rb");
    assert(fp && ds4_session_load_payload(r, fp, payload_bytes, err, sizeof(err)) == 0);
    fclose(fp);
    assert(ds4_session_eval(r, tokens.v[n + t], err, sizeof(err)) == 0);
    assert(!memcmp(after, r->logits, row_bytes));
    ds4_session_free(r);
    printf("  payload %.1f MiB reloads with exact logits\n", payload_bytes / 1048576.0);
    ds4_engine_close(engine);

    engine = open_engine(argv[1], ctx, false);
    puts("f16 KV:");
    s = run_mode(engine, &tokens, n, t, ctx, f16_logits);
    ds4_session_free(s);
    assert(ds4_session_create(&r, engine, ctx) == 0);
    fp = fopen(path, "rb");
    assert(fp);
    assert(ds4_session_load_payload(r, fp, payload_bytes, err, sizeof(err)) != 0);
    fclose(fp);
    ds4_session_free(r);
    printf("  int8 payload refused: %s\n", err);
    ds4_engine_close(engine);
    unlink(path);

    const kv_q8_stats st = compare_rows(f16_logits, q8_logits, tokens.v + n, (uint32_t)t);
    printf("int8 vs f16 over %u rows after %d tokens: KL mean %.5f max %.5f, top-1 same %u/%u, "
           "ppl f16 %.4f int8 %.4f\n", st.rows, n, st.kl_mean, st.kl_max, st.top1_same, st.rows,
           exp(st.nll_f16), exp(st.nll_q8));
    /* Loose gates: rounding noise, not a broken cache (which gives KL >> 1). */
    assert(st.kl_mean < 0.05);
    assert(st.top1_same * 10u >= st.rows * 9u);
    assert(exp(st.nll_q8) < exp(st.nll_f16) * 1.03);
    puts("Qwen int8 KV: OK");
    free(after);
    free(q8_logits);
    free(f16_logits);
    ds4_tokens_free(&tokens);
    free(text);
    return 0;
}
