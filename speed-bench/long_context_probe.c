/* Long-context retrieval probe for Qwen3.8.
 *
 * Builds one long document from a haystack file truncated to a token budget,
 * splices needle lines into it at chosen depths, prefills it once, and then
 * answers every question from that same state: the prompt anchor restores the
 * recurrent state at the end of the document, so a question costs only its own
 * few tokens.  Answers are greedy and written as TSV for an external grader.
 *
 * Spec file, tab separated:
 *   N <depth 0..1> <needle text>
 *   Q <id> <question> [<locator literal>]
 * The locator is searched in the final document and its token position is
 * reported, so the grader knows how deep the answer was.
 */

#include "ds4.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ITEMS 512

typedef struct {
    double depth;
    char *text;
} needle;

typedef struct {
    char *id, *text, *locator;
} question;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); exit(1); }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) { perror(path); exit(1); }
    buf[n] = 0;
    fclose(fp);
    return buf;
}

static void push_all(ds4_tokens *dst, const int *v, int n) {
    for (int i = 0; i < n; i++) ds4_tokens_push(dst, v[i]);
}

static void print_escaped(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n') fputs("\\n", fp);
        else if (s[i] == '\t') fputs("\\t", fp);
        else if (s[i] == '\r') continue;
        else fputc(s[i], fp);
    }
}

static int by_depth(const void *a, const void *b) {
    const double x = ((const needle *)a)->depth, y = ((const needle *)b)->depth;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    const char *model = NULL, *hay_path = NULL, *spec_path = NULL, *out_path = NULL;
    int doc_tokens = 0, n_predict = 48, chunk = 2048;
    bool mtp = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--haystack") && i + 1 < argc) hay_path = argv[++i];
        else if (!strcmp(argv[i], "--spec") && i + 1 < argc) spec_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--doc-tokens") && i + 1 < argc) doc_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mtp")) mtp = true;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (!model || !hay_path || !spec_path || !out_path || doc_tokens <= 0) {
        fprintf(stderr, "usage: long_context_probe -m MODEL --haystack FILE --spec FILE "
                        "--out FILE --doc-tokens N [-n N] [--prefill-chunk N] [--mtp]\n");
        return 2;
    }

    static needle needles[MAX_ITEMS];
    static question questions[MAX_ITEMS];
    int n_needles = 0, n_questions = 0;
    char *spec = read_file(spec_path);
    for (char *line = strtok(spec, "\n"); line; line = strtok(NULL, "\n")) {
        char *f[4] = {0};
        int nf = 0;
        for (char *p = line; nf < 4; nf++) {
            f[nf] = p;
            char *tab = strchr(p, '\t');
            if (!tab) { nf++; break; }
            *tab = 0;
            p = tab + 1;
        }
        if (nf >= 3 && !strcmp(f[0], "N") && n_needles < MAX_ITEMS) {
            needles[n_needles++] = (needle){atof(f[1]), f[2]};
        } else if (nf >= 3 && !strcmp(f[0], "Q") && n_questions < MAX_ITEMS) {
            questions[n_questions++] = (question){f[1], f[2], nf > 3 ? f[3] : NULL};
        }
    }
    qsort(needles, (size_t)n_needles, sizeof(needles[0]), by_depth);

    const int ctx = doc_tokens + 1024;
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .context_size = ctx,
        .prefill_chunk = chunk,
        .glm_mtp = mtp,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;

    /* The chat frame around a user message: what two different messages share. */
    ds4_tokens a = {0}, b = {0};
    ds4_encode_chat_prompt(engine, NULL, "alpha", DS4_THINK_NONE, &a);
    ds4_encode_chat_prompt(engine, NULL, "omega", DS4_THINK_NONE, &b);
    int head = 0, tail = 0;
    while (head < a.len && head < b.len && a.v[head] == b.v[head]) head++;
    while (tail < a.len - head && tail < b.len - head &&
           a.v[a.len - 1 - tail] == b.v[b.len - 1 - tail]) tail++;

    char *hay_text = read_file(hay_path);
    ds4_tokens hay = {0};
    ds4_tokenize_text(engine, hay_text, &hay);
    free(hay_text);

    /* Needles go on line boundaries of the haystack, each on its own line. */
    ds4_tokens nl = {0};
    ds4_tokenize_text(engine, "\n", &nl);
    ds4_tokens needle_tok[MAX_ITEMS];
    int needle_total = 0;
    for (int i = 0; i < n_needles; i++) {
        size_t len = strlen(needles[i].text);
        char *line = malloc(len + 2);
        memcpy(line, needles[i].text, len);
        line[len] = '\n';
        line[len + 1] = 0;
        needle_tok[i] = (ds4_tokens){0};
        ds4_tokenize_text(engine, line, &needle_tok[i]);
        free(line);
        needle_total += needle_tok[i].len;
    }
    const int hay_budget = doc_tokens - needle_total;
    if (hay_budget <= 0 || hay.len < hay_budget) {
        fprintf(stderr, "probe: haystack has %d tokens, need %d\n", hay.len, hay_budget);
        return 1;
    }

    ds4_tokens doc = {0};
    push_all(&doc, a.v, head);
    int next = 0;
    for (int i = 0; i < hay_budget; i++) {
        ds4_tokens_push(&doc, hay.v[i]);
        while (next < n_needles && nl.len == 1 && hay.v[i] == nl.v[0] &&
               (double)i >= needles[next].depth * (double)hay_budget) {
            push_all(&doc, needle_tok[next].v, needle_tok[next].len);
            next++;
        }
    }
    for (; next < n_needles; next++) {
        push_all(&doc, nl.v, nl.len);
        push_all(&doc, needle_tok[next].v, needle_tok[next].len);
    }

    /* Decoded document with the byte offset of every token, for the locators. */
    size_t text_len = 0, text_cap = 1 << 20;
    char *text = malloc(text_cap);
    size_t *tok_off = malloc((size_t)(doc.len + 1) * sizeof(tok_off[0]));
    for (int i = 0; i < doc.len; i++) {
        size_t len = 0;
        char *piece = ds4_token_text(engine, doc.v[i], &len);
        if (text_len + len + 1 > text_cap) {
            while (text_len + len + 1 > text_cap) text_cap *= 2;
            text = realloc(text, text_cap);
        }
        tok_off[i] = text_len;
        for (size_t k = 0; k < len; k++) text[text_len++] = piece[k] ? piece[k] : ' ';
        free(piece);
    }
    tok_off[doc.len] = text_len;
    text[text_len] = 0;

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, ctx) != 0) {
        fprintf(stderr, "probe: failed to create session\n");
        return 1;
    }
    char err[256];
    fprintf(stderr, "probe: document %d tokens (%d needles, frame %d+%d), %d questions\n",
            doc.len, n_needles, head, tail, n_questions);
    const double t0 = now_sec();
    if (ds4_session_sync(session, &doc, err, sizeof(err)) != 0) {
        fprintf(stderr, "probe: prefill failed: %s\n", err);
        return 1;
    }
    const double prefill_sec = now_sec() - t0;
    if (!ds4_session_anchor_save(session)) {
        fprintf(stderr, "probe: anchor save failed\n");
        return 1;
    }
    fprintf(stderr, "probe: prefill %.1f s, %.1f t/s\n", prefill_sec, (double)doc.len / prefill_sec);

    FILE *out = fopen(out_path, "w");
    if (!out) { perror(out_path); return 1; }
    fprintf(out, "#doc_tokens=%d prefill_tps=%.2f\n", doc.len, (double)doc.len / prefill_sec);
    fprintf(out, "id\tlocator_pos\tsync_ms\tdecode_tps\tanswer\n");
    const int eos = ds4_token_eos(engine);
    for (int q = 0; q < n_questions; q++) {
        int pos = -1;
        if (questions[q].locator) {
            const char *hit = strstr(text, questions[q].locator);
            if (hit) {
                const size_t off = (size_t)(hit - text);
                int lo = 0, hi = doc.len;
                while (lo + 1 < hi) {
                    const int mid = (lo + hi) / 2;
                    if (tok_off[mid] <= off) lo = mid; else hi = mid;
                }
                pos = lo;
            }
        }
        if (questions[q].locator && pos < 0) continue;   /* beyond this document */

        ds4_tokens prompt = {0}, qt = {0};
        size_t qlen = strlen(questions[q].text);
        char *qtext = malloc(qlen + 3);
        memcpy(qtext, "\n\n", 2);
        memcpy(qtext + 2, questions[q].text, qlen + 1);
        ds4_tokenize_text(engine, qtext, &qt);
        free(qtext);
        push_all(&prompt, doc.v, doc.len);
        push_all(&prompt, qt.v, qt.len);
        push_all(&prompt, a.v + a.len - tail, tail);

        const double q0 = now_sec();
        if (!ds4_session_anchor_restore(session) ||
            ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "probe: question %s failed: %s\n", questions[q].id, err);
            return 1;
        }
        const double q1 = now_sec();
        fprintf(out, "%s\t%d\t%.0f\t", questions[q].id, pos, (q1 - q0) * 1e3);
        char answer[8192];
        size_t alen = 0;
        int done = 0;
        bool stop = false;
        while (done < n_predict && !stop) {
            int toks[17];
            int ntok = 1;
            toks[0] = ds4_session_argmax(session);
            if (toks[0] < 0 || toks[0] == eos) break;
            if (mtp) {
                /* commits toks[0] and returns it with every accepted draft */
                ntok = ds4_session_eval_speculative_argmax(session, toks[0], n_predict - done, eos,
                                                           toks, 17, err, sizeof(err));
            } else if (ds4_session_eval(session, toks[0], err, sizeof(err)) != 0) {
                ntok = 0;
            }
            if (ntok <= 0) break;
            for (int j = 0; j < ntok && !stop; j++) {
                size_t len = 0;
                char *piece = ds4_token_text(engine, toks[j], &len);
                stop = toks[j] == eos || (len >= 2 && piece[0] == '<' && piece[1] == '|');
                if (!stop && alen + len < sizeof(answer)) { memcpy(answer + alen, piece, len); alen += len; }
                free(piece);
                if (!stop) done++;
            }
        }
        const double q2 = now_sec();
        fprintf(out, "%.2f\t", done > 0 ? (double)done / (q2 - q1) : 0.0);
        print_escaped(out, answer, alen);
        fputc('\n', out);
        fflush(out);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&qt);
    }
    fclose(out);
    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
