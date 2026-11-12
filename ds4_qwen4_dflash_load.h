/* DFlash2 draft weights for Qwen3.8-Flash-Next (incoai/Qwen3.8-Flash-Next-DFlash2).
 * Included by ds4.c before the MTP sidecar attach.  DS4_QWEN_DFLASH names
 * the checkpoint directory (config.json + model.safetensors).  The BF16
 * projections become Q8_0 and the norms and convolution kernels F32, mapped
 * right after the MTP block inside its reservation so the Metal model views
 * cover them like any other weight. */
#ifndef DS4_QWEN4_DFLASH_LOAD_H
#define DS4_QWEN4_DFLASH_LOAD_H

#define QWEN4_DFL_LAYERS 5u
#define QWEN4_DFL_HIDDEN 2560u
#define QWEN4_DFL_FF 7680u
#define QWEN4_DFL_QDIM 4096u           /* 32 query heads of 128 */
#define QWEN4_DFL_KVDIM 1024u          /* 8 KV heads of 128 */
#define QWEN4_DFL_QKV (QWEN4_DFL_QDIM + 2u * QWEN4_DFL_KVDIM)
#define QWEN4_DFL_DYN 640u             /* 2 stages x 2 taps x 160 groups */
#define QWEN4_DFL_RANK 256u
#define QWEN4_DFL_MASK_TOKEN 248077
#define QWEN4_DFL_NTARGET 5u

typedef struct {
    ds4_tensor in_norm, post_norm;     /* F32 [E] */
    ds4_tensor attn_base, mlp_base;    /* F32 [2][2][E] */
    ds4_tensor attn_dyn, mlp_dyn;      /* Q8_0 E -> 640 */
    ds4_tensor qkv, kv;                /* Q8_0 E -> 6144; kv: its k|v rows */
    ds4_tensor q_norm, k_norm;         /* F32 [128] */
    ds4_tensor o;                      /* Q8_0 4096 -> E */
    ds4_tensor gate, up, down;         /* Q8_0 */
} qwen4_dfl_layer;

typedef struct {
    bool loaded;
    qwen4_dfl_layer layer[QWEN4_DFL_LAYERS];
    ds4_tensor fc;                     /* Q8_0 5E -> E */
    ds4_tensor hidden_norm, norm;      /* F32 [E] */
    ds4_tensor sel;                    /* Q8_0 E -> 256 */
    ds4_tensor head;                   /* Q8_0 copy of the target head's first head_rows rows */
    uint32_t head_rows;                /* 0: the draft scores the full target head */
    const uint16_t *pred_cb, *succ_cb; /* BF16 [V][256], host-readable */
    uint64_t vocab;
} qwen4_dfl_weights;

static qwen4_dfl_weights g_qwen4_dfl;

/* The safetensors header entry of `name`: its BF16 data range and its
 * element count. */
static bool qwen4_dfl_st_find(const char *json, const char *name, uint64_t *off, uint64_t *bytes,
                              uint64_t *elements) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(json, key);
    if (!p) return false;
    const char *end = strchr(p, '}');
    const char *d = strstr(p, "\"dtype\"");
    const char *s = strstr(p, "\"shape\"");
    const char *o = strstr(p, "\"data_offsets\"");
    if (!end || !d || !s || !o || d > end || s > end || o > end) return false;
    const char *bf = strstr(d, "\"BF16\"");
    if (!bf || bf > end) return false;
    s = strchr(s, '[');
    if (!s) return false;
    uint64_t n = 1;
    while (*s && *s != ']') {
        s++;
        while (*s == ' ') s++;
        if (*s == ']') break;
        n *= strtoull(s, (char **)&s, 10);
        while (*s == ' ') s++;
    }
    o = strchr(o, '[');
    if (!o) return false;
    char *q = NULL;
    const uint64_t a = strtoull(o + 1, &q, 10);
    if (!q || *q != ',') return false;
    const uint64_t b = strtoull(q + 1, NULL, 10);
    if (b < a) return false;
    *off = a;
    *bytes = b - a;
    *elements = n;
    return true;
}

typedef struct {
    const uint8_t *map;     /* safetensors file */
    uint64_t data;          /* start of the tensor data */
    const char *json;
    uint8_t *out;           /* destination region */
    uint64_t used, room;
    uint64_t base;          /* model offset of out */
    ds4_model *m;
    bool ok;
} qwen4_dfl_builder;

static void qwen4_dfl_tensor(qwen4_dfl_builder *b, ds4_tensor *t, const char *name, uint32_t type,
                             uint64_t cols, uint64_t rows, uint64_t bytes) {
    memset(t, 0, sizeof(*t));
    t->name.ptr = name;
    t->name.len = strlen(name);
    t->ndim = rows > 1 ? 2 : 1;
    t->dim[0] = cols;
    t->dim[1] = rows;
    t->type = type;
    t->elements = cols * rows;
    t->bytes = bytes;
    t->abs_offset = b->base + b->used;
    t->rel_offset = t->abs_offset - b->m->tensor_data_pos;
}

static uint8_t *qwen4_dfl_take(qwen4_dfl_builder *b, uint64_t bytes) {
    const uint64_t at = align_up(b->used, 256u);
    if (at + bytes > b->room) { b->ok = false; return NULL; }
    b->used = at;
    return b->out + at;
}

static const uint16_t *qwen4_dfl_src(qwen4_dfl_builder *b, const char *name, uint64_t elements) {
    uint64_t off = 0, bytes = 0, n = 0;
    if (!qwen4_dfl_st_find(b->json, name, &off, &bytes, &n) || bytes != elements * 2u || n != elements) {
        fprintf(stderr, "ds4: DFlash2 tensor %s is missing or has an unexpected shape "
                "(%" PRIu64 " elements in %" PRIu64 " bytes, want %" PRIu64 ")\n", name, n, bytes, elements);
        b->ok = false;
        return NULL;
    }
    return (const uint16_t *)(b->map + b->data + off);
}

/* Q8_0 rows of one or more stacked BF16 [rows][cols] sources. */
static void qwen4_dfl_q8(qwen4_dfl_builder *b, ds4_tensor *t, const char *label, uint64_t cols,
                         const char *const *names, const uint64_t *rows, int n) {
    if (!b->ok) return;
    uint64_t total = 0;
    for (int i = 0; i < n; i++) total += rows[i];
    const uint64_t row_bytes = cols / 32u * 34u;
    uint8_t *dst = qwen4_dfl_take(b, total * row_bytes);
    if (!dst) return;
    qwen4_dfl_tensor(b, t, label, DS4_TENSOR_Q8_0, cols, total, total * row_bytes);
    uint64_t r0 = 0;
    for (int i = 0; i < n && b->ok; i++) {
        const uint16_t *src = qwen4_dfl_src(b, names[i], cols * rows[i]);
        if (!src) return;
        qwen4_hc_q8_ctx ctx = { .src = src, .dst = dst + r0 * row_bytes, .row_len = cols };
        ds4_parallel_for_min_rows(rows[i], qwen4_hc_q8_rows, &ctx, 64);
        r0 += rows[i];
    }
    b->used += total * row_bytes;
}

static void qwen4_dfl_f32(qwen4_dfl_builder *b, ds4_tensor *t, const char *label, const char *name, uint64_t n) {
    if (!b->ok) return;
    const uint16_t *src = qwen4_dfl_src(b, name, n);
    float *dst = (float *)qwen4_dfl_take(b, n * sizeof(float));
    if (!src || !dst) return;
    qwen4_dfl_tensor(b, t, label, DS4_TENSOR_F32, n, 1, n * sizeof(float));
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, sizeof(float));
    }
    b->used += n * sizeof(float);
}

static const uint16_t *qwen4_dfl_bf16_copy(qwen4_dfl_builder *b, const char *name, uint64_t elements) {
    if (!b->ok) return NULL;
    const uint16_t *src = qwen4_dfl_src(b, name, elements);
    uint8_t *dst = qwen4_dfl_take(b, elements * 2u);
    if (!src || !dst) return NULL;
    memcpy(dst, src, (size_t)(elements * 2u));
    b->used += elements * 2u;
    return (const uint16_t *)dst;
}

static uint64_t qwen4_dfl_bytes(uint64_t vocab) {
    const uint64_t E = QWEN4_DFL_HIDDEN, q8 = 34u, blk = 32u;
    uint64_t per_layer = (2u * QWEN4_DFL_DYN * E + QWEN4_DFL_QKV * E + E * QWEN4_DFL_QDIM +
                          2u * QWEN4_DFL_FF * E + E * QWEN4_DFL_FF) / blk * q8;
    per_layer += (2u * E + 8u * E + 2u * 128u) * sizeof(float) + 16u * 256u;
    uint64_t total = per_layer * QWEN4_DFL_LAYERS;
    total += (E * QWEN4_DFL_NTARGET * E + QWEN4_DFL_RANK * E) / blk * q8 + 2u * E * sizeof(float);
    total += 2u * vocab * QWEN4_DFL_RANK * 2u + 16u * 256u;
    return total;
}

static bool qwen4_dfl_q8_one(qwen4_dfl_builder *b, ds4_tensor *t, const char *label, uint64_t cols,
                             const char *name, uint64_t rows) {
    const char *s[1] = { name };
    const uint64_t r[1] = { rows };
    qwen4_dfl_q8(b, t, label, cols, s, r, 1);
    return b->ok;
}

static void qwen4_dfl_build_layer(qwen4_dfl_builder *b, uint32_t l) {
    static char labels[QWEN4_DFL_LAYERS][14][24];
    static const char *const parts[14] = {
        "in_norm", "post_norm", "attn_base", "mlp_base", "attn_dyn", "mlp_dyn", "qkv", "kv",
        "q_norm", "k_norm", "o", "gate", "up", "down" };
    for (int i = 0; i < 14; i++) snprintf(labels[l][i], sizeof(labels[l][i]), "dfl.%u.%s", l, parts[i]);
    qwen4_dfl_layer *L = &g_qwen4_dfl.layer[l];
    const uint64_t E = QWEN4_DFL_HIDDEN, row = E / 32u * 34u;
    char n[3][96];
#define DFL_NAME(i_, fmt_) snprintf(n[i_], sizeof(n[i_]), "layers.%u." fmt_, l)
    DFL_NAME(0, "input_layernorm.weight");          qwen4_dfl_f32(b, &L->in_norm, labels[l][0], n[0], E);
    DFL_NAME(0, "post_attention_layernorm.weight"); qwen4_dfl_f32(b, &L->post_norm, labels[l][1], n[0], E);
    DFL_NAME(0, "attention_conv.base_kernel");      qwen4_dfl_f32(b, &L->attn_base, labels[l][2], n[0], 4u * E);
    DFL_NAME(0, "mlp_conv.base_kernel");            qwen4_dfl_f32(b, &L->mlp_base, labels[l][3], n[0], 4u * E);
    DFL_NAME(0, "attention_conv.kernel_projection.weight");
    qwen4_dfl_q8_one(b, &L->attn_dyn, labels[l][4], E, n[0], QWEN4_DFL_DYN);
    DFL_NAME(0, "mlp_conv.kernel_projection.weight");
    qwen4_dfl_q8_one(b, &L->mlp_dyn, labels[l][5], E, n[0], QWEN4_DFL_DYN);
    DFL_NAME(0, "self_attn.q_proj.weight");
    DFL_NAME(1, "self_attn.k_proj.weight");
    DFL_NAME(2, "self_attn.v_proj.weight");
    {
        const char *s3[3] = { n[0], n[1], n[2] };
        const uint64_t r3[3] = { QWEN4_DFL_QDIM, QWEN4_DFL_KVDIM, QWEN4_DFL_KVDIM };
        qwen4_dfl_q8(b, &L->qkv, labels[l][6], E, s3, r3, 3);
    }
    /* the context writers' K|V rows: a view of the fused rows */
    L->kv = L->qkv;
    L->kv.name.ptr = labels[l][7];
    L->kv.name.len = strlen(labels[l][7]);
    L->kv.dim[1] = 2u * QWEN4_DFL_KVDIM;
    L->kv.elements = L->kv.dim[0] * L->kv.dim[1];
    L->kv.bytes = L->kv.dim[1] * row;
    L->kv.abs_offset += (uint64_t)QWEN4_DFL_QDIM * row;
    L->kv.rel_offset += (uint64_t)QWEN4_DFL_QDIM * row;
    DFL_NAME(0, "self_attn.q_norm.weight"); qwen4_dfl_f32(b, &L->q_norm, labels[l][8], n[0], 128u);
    DFL_NAME(0, "self_attn.k_norm.weight"); qwen4_dfl_f32(b, &L->k_norm, labels[l][9], n[0], 128u);
    DFL_NAME(0, "self_attn.o_proj.weight"); qwen4_dfl_q8_one(b, &L->o, labels[l][10], QWEN4_DFL_QDIM, n[0], E);
    DFL_NAME(0, "mlp.gate_proj.weight");    qwen4_dfl_q8_one(b, &L->gate, labels[l][11], E, n[0], QWEN4_DFL_FF);
    DFL_NAME(0, "mlp.up_proj.weight");      qwen4_dfl_q8_one(b, &L->up, labels[l][12], E, n[0], QWEN4_DFL_FF);
    DFL_NAME(0, "mlp.down_proj.weight");    qwen4_dfl_q8_one(b, &L->down, labels[l][13], QWEN4_DFL_FF, n[0], E);
#undef DFL_NAME
}

/* DS4_QWEN4_DFL_VOCAB: the draft scores the first N vocabulary ids only
 * (the tokenizer numbers tokens in merge order, so low ids are the frequent
 * ones), through a Q8_0 copy of those rows of the target's output head that
 * the batched kernel reads once for all eight rows.  0 keeps the full head
 * (exact candidates, slower).  Default 81920: the 32768 most frequent ids
 * lose ~10% of the accepted drafts on non-English text. */
static uint32_t qwen4_dfl_head_rows(uint64_t vocab) {
    const char *env = getenv("DS4_QWEN4_DFL_VOCAB");
    uint64_t n = env && env[0] ? strtoull(env, NULL, 10) : 81920u;
    if (n >= vocab) n = 0;
    return (uint32_t)(n / 32u * 32u);
}

typedef struct {
    const ds4_model *m;
    const ds4_tensor *src;
    uint8_t *dst;
    uint64_t cols;
} qwen4_dfl_head_ctx;

static void qwen4_dfl_head_q8_rows(void *vctx, uint64_t r0, uint64_t r1) {
    const qwen4_dfl_head_ctx *c = vctx;
    const uint64_t blocks = c->cols / 32u;
    const uint64_t src_row = c->src->bytes / c->src->dim[1];
    float *row = malloc((size_t)c->cols * sizeof(float));
    if (!row) return;
    for (uint64_t r = r0; r < r1; r++) {
        uint8_t *q = c->dst + r * blocks * 34u;
        if (!ds4_dequant_row(c->src->type, (const uint8_t *)c->m->map + c->src->abs_offset + r * src_row, row, c->cols)) {
            memset(q, 0, blocks * 34u);
            continue;
        }
        const float *x = row;
        for (uint64_t b = 0; b < blocks; b++, x += 32, q += 34) {
            float amax = 0.0f;
            for (int i = 0; i < 32; i++) if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
            const uint16_t dh = f32_to_f16(amax / 127.0f);
            const float d = f16_to_f32(dh);
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            memcpy(q, &dh, sizeof(dh));
            for (int i = 0; i < 32; i++) {
                const int iv = (int)lrintf(x[i] * id);
                q[2 + i] = (uint8_t)(int8_t)(iv > 127 ? 127 : iv < -127 ? -127 : iv);
            }
        }
    }
    free(row);
}

static void qwen4_dfl_build_head(qwen4_dfl_builder *b, uint32_t rows) {
    if (!b->ok || !rows) return;
    const ds4_tensor *src = model_find_tensor(b->m, "output.weight");
    const uint64_t cols = QWEN4_DFL_HIDDEN, row_bytes = cols / 32u * 34u;
    if (!src || src->ndim < 2 || src->dim[0] != cols || src->dim[1] < rows || (src->bytes % src->dim[1]) != 0) {
        fprintf(stderr, "ds4: DFlash2: the target output head does not match the draft\\n");
        b->ok = false;
        return;
    }
    uint8_t *dst = qwen4_dfl_take(b, rows * row_bytes);
    if (!dst) return;
    qwen4_dfl_tensor(b, &g_qwen4_dfl.head, "dfl.head", DS4_TENSOR_Q8_0, cols, rows, rows * row_bytes);
    qwen4_dfl_head_ctx ctx = { .m = b->m, .src = src, .dst = dst, .cols = cols };
    ds4_parallel_for_min_rows(rows, qwen4_dfl_head_q8_rows, &ctx, 256);
    b->used += rows * row_bytes;
    g_qwen4_dfl.head_rows = rows;
}

/* Maps the draft at `at` (inside the MTP reservation, `room` bytes free);
 * returns the bytes used, or 0 when DS4_QWEN_DFLASH is unset or unusable. */
static uint64_t model_map_qwen_dflash(ds4_model *m, uint8_t *at, uint64_t room) {
    const char *dir = getenv("DS4_QWEN_DFLASH");
    if (!dir || !dir[0]) return 0;
    char path[4096];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    const int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size < 16) {
        fprintf(stderr, "ds4: DFlash2 draft %s: %s\n", path, strerror(errno));
        if (fd >= 0) close(fd);
        return 0;
    }
    const uint8_t *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "ds4: DFlash2 draft %s: cannot map\n", path);
        return 0;
    }
    uint64_t hlen = 0;
    memcpy(&hlen, map, sizeof(hlen));
    /* the model shape is not set yet at attach time: the codebook gives the vocabulary */
    char *head = hlen && 8u + hlen <= (uint64_t)st.st_size ? xmalloc((size_t)hlen + 1u) : NULL;
    uint64_t cb_off = 0, cb_bytes = 0, cb_n = 0;
    if (head) { memcpy(head, map + 8, (size_t)hlen); head[hlen] = '\0'; }
    const bool cb_ok = head && qwen4_dfl_st_find(head, "candidate_selector.predecessor_codebook",
                                                 &cb_off, &cb_bytes, &cb_n) && cb_n % QWEN4_DFL_RANK == 0;
    free(head);
    const uint64_t vocab = cb_ok ? cb_n / QWEN4_DFL_RANK : 0;
    const uint32_t head_rows = vocab ? qwen4_dfl_head_rows(vocab) : 0;
    const uint64_t need = align_up(qwen4_dfl_bytes(vocab) + (uint64_t)head_rows * (QWEN4_DFL_HIDDEN / 32u * 34u) + 256u,
                                   (uint64_t)sysconf(_SC_PAGESIZE));
    if (!vocab || hlen == 0 || 8u + hlen > (uint64_t)st.st_size || need > room ||
        mmap(at, (size_t)need, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) == MAP_FAILED) {
        fprintf(stderr, "ds4: DFlash2 draft %s does not fit (%.2f GiB needed)\n", path,
                (double)need / (1024.0 * 1024.0 * 1024.0));
        munmap((void *)map, (size_t)st.st_size);
        return 0;
    }
    char *json = xmalloc((size_t)hlen + 1u);
    memcpy(json, map + 8, (size_t)hlen);
    json[hlen] = '\0';
    const double t0 = now_sec();
    qwen4_dfl_builder b = { .map = map, .data = 8u + hlen, .json = json, .out = at, .used = 0, .room = need,
                            .base = (uint64_t)(at - (uint8_t *)m->map), .m = m, .ok = true };
    const uint64_t E = QWEN4_DFL_HIDDEN;
    for (uint32_t l = 0; l < QWEN4_DFL_LAYERS && b.ok; l++) qwen4_dfl_build_layer(&b, l);
    qwen4_dfl_q8_one(&b, &g_qwen4_dfl.fc, "dfl.fc", QWEN4_DFL_NTARGET * E, "fc.weight", E);
    qwen4_dfl_f32(&b, &g_qwen4_dfl.hidden_norm, "dfl.hidden_norm", "hidden_norm.weight", E);
    qwen4_dfl_f32(&b, &g_qwen4_dfl.norm, "dfl.norm", "norm.weight", E);
    qwen4_dfl_q8_one(&b, &g_qwen4_dfl.sel, "dfl.sel", E, "candidate_selector.hidden_projection.weight",
                     QWEN4_DFL_RANK);
    g_qwen4_dfl.pred_cb = qwen4_dfl_bf16_copy(&b, "candidate_selector.predecessor_codebook", vocab * QWEN4_DFL_RANK);
    g_qwen4_dfl.succ_cb = qwen4_dfl_bf16_copy(&b, "candidate_selector.successor_codebook", vocab * QWEN4_DFL_RANK);
    qwen4_dfl_build_head(&b, head_rows);
    free(json);
    munmap((void *)map, (size_t)st.st_size);
    if (!b.ok || !g_qwen4_dfl.pred_cb || !g_qwen4_dfl.succ_cb) {
        fprintf(stderr, "ds4: DFlash2 draft %s could not be loaded\n", path);
        munmap(at, (size_t)need);
        memset(&g_qwen4_dfl, 0, sizeof(g_qwen4_dfl));
        return 0;
    }
    if (mprotect(at, (size_t)need, PROT_READ)) ds4_die_errno("cannot protect the DFlash2 draft", path);
    g_qwen4_dfl.vocab = vocab;
    g_qwen4_dfl.loaded = true;
    fprintf(stderr, "ds4: DFlash2 draft from %s (%.0f MiB, %.2f s)\n", dir,
            (double)b.used / (1024.0 * 1024.0), now_sec() - t0);
    return need;
}

#endif
