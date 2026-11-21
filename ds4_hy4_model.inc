/* Hy4 Flash model layout. The Engram shard has a private descriptor and is
 * never registered with Metal. Its rows are read on demand into prefill scratch. */
enum {
    HY4_LAYERS = 42, HY4_DIM = 3072, HY4_VOCAB = 120832, HY4_CONTEXT = 1048576,
    HY4_WINDOW = 512, HY4_KV = 576, HY4_LATENT = 1536, HY4_EXPERTS = 512,
    HY4_ACTIVE = 16, HY4_ENGRAM_LAYERS = 4, HY4_ENGRAM_HEADS = 16,
    HY4_ENGRAM_DIM = 96, HY4_ENGRAM_ROWS = 660006416,
    HY4_HIST_GROUPS = 511,
    HY4_PREFILL = 512, HY4_PREFILL_MAX = 1024
};

typedef struct {
    const ds4_tensor *fn, *scale, *base;
} hy4_hc_weights;

typedef struct {
    const ds4_tensor *gate, *up, *down;
} hy4_ffn_weights;

typedef struct {
    hy4_hc_weights attn_hc, ffn_hc;
    hy4_ffn_weights ffn, experts;
    const ds4_tensor *attn_norm, *qa, *qa_norm, *qb, *kv, *kv_norm, *kb, *vb;
    const ds4_tensor *attn_gate, *attn_out, *attn_post, *sink;
    const ds4_tensor *index_q, *index_k, *index_norm, *index_w, *pool_v, *pool_gate, *ape, *pool_norm;
    const ds4_tensor *ffn_norm, *ffn_post, *router, *router_bias, *latent_in, *latent_out;
    const ds4_tensor *engram_kv, *engram_bias, *engram_gate, *engram_norm, *engram_gamma, *engram_scale;
} hy4_layer_weights;

enum { HY4_DFLY_LAYERS = 5, HY4_DFLY_FF = 9216, HY4_DFLY_ROWS = 6 };

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
typedef struct {
    hy4_layer_weights layer[HY4_DFLY_LAYERS];
    ds4_gpu_tensor *keys[HY4_DFLY_LAYERS];
    const ds4_tensor *enorm, *hnorm, *fuse, *gate, *norm;
    const ds4_tensor *embed_norm, *hidden_norm;
    hy4_ffn_weights correction;
    uint32_t target[HY4_DFLY_LAYERS], mask, rows;
} hy4_dfly_weights;
#endif

typedef struct {
    hy4_layer_weights layer[HY4_LAYERS];
    hy4_hc_weights head;
    const ds4_tensor *embedding, *output, *norm;
    int fd;
    uint64_t table_offset;
    uint64_t multiplier[HY4_ENGRAM_LAYERS * 3];
    uint64_t offsets[HY4_ENGRAM_LAYERS * HY4_ENGRAM_HEADS];
    uint64_t primes[HY4_ENGRAM_LAYERS * HY4_ENGRAM_HEADS];
    uint32_t token_map[HY4_VOCAB], mapped_pad;
    uint32_t table_type, row_bytes;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    hy4_dfly_weights *draft;
#endif
} ds4_hy4_model;

static bool hy4_sparse(unsigned layer) { return layer == 0 || layer % 4 == 1; }
static bool hy4_engram_layer(unsigned layer) { return layer <= 31 && layer % 10 == 1; }

static const ds4_shape DS4_SHAPE_HY4 = {
    .name = "Tencent Hy4 Flash", .family = DS4_MODEL_FAMILY_HY4, .variant = DS4_VARIANT_HY4,
    .n_layer = HY4_LAYERS, .n_embd = HY4_DIM, .n_vocab = HY4_VOCAB, .n_head = 64,
    .n_head_kv = 1, .n_head_dim = HY4_KV, .n_value_dim = 512, .n_rot = 64,
    .n_lora_q = 1024, .n_kv_lora = 512, .n_expert = HY4_EXPERTS, .n_expert_used = HY4_ACTIVE,
    .n_expert_shared = 1, .n_ff_exp = HY4_LATENT, .n_leading_dense = 1,
    .n_indexer_head = 16, .n_indexer_head_dim = 128, .n_indexer_top_k = 2048, .n_hc = 4,
    .rms_eps = 1e-5f, .hc_eps = 1e-6f, .rope_orig_ctx = HY4_CONTEXT,
};

static bool ds4_model_is_hy4(void) { return DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_HY4; }

static void hy4_validate_config(const ds4_model *m) {
    static const struct { const char *key; uint32_t value; } config[] = {
        {"block_count", HY4_LAYERS}, {"context_length", HY4_CONTEXT},
        {"embedding_length", HY4_DIM}, {"expert_count", HY4_EXPERTS},
        {"expert_used_count", HY4_ACTIVE}, {"attention.kv_lora_rank", 512},
        {"attention.q_lora_rank", 1024}, {"attention.sliding_window", HY4_WINDOW},
        {"attention.indexer.kpool", 4}, {"attention.indexer.top_k", 2048},
        {"attention.indexer.head_count", 16}, {"rope.dimension_count", 64},
        {"hyper_connection.count", 4}, {"expert_latent_length", HY4_LATENT},
        {"ple.ngram_size", 3}, {"ple.heads_per_ngram", 8},
        {"embedding_length_per_layer_input", HY4_ENGRAM_DIM},
    };
    for (size_t i = 0; i < sizeof(config)/sizeof(config[0]); i++) {
        char key[96];
        snprintf(key, sizeof(key), "hy_v4_flash.%s", config[i].key);
        if (required_u32(m, key) != config[i].value) {
            fprintf(stderr, "ds4: unsupported Hy4 Flash metadata for %s\n", key);
            ds4_die("unsupported Hy4 Flash dimensions");
        }
    }
    g_ds4_shape = DS4_SHAPE_HY4;
}

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool hy4_split_field(const ds4_model *m, const char *key, uint16_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT16) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_read(&c, out, sizeof(*out));
}

static const ds4_tensor *hy4_weight(const ds4_model *m, int layer, const char *name,
                                    uint64_t d0, uint64_t d1, uint64_t d2) {
    char qualified[128];
    if (layer >= 0) {
        int len = snprintf(qualified, sizeof(qualified), "blk.%d.%s", layer, name);
        if (len < 0 || (size_t)len >= sizeof(qualified)) ds4_die("tensor name is too long");
        name = qualified;
    }
    const ds4_tensor *t = required_tensor(m, name);
    const uint32_t dims = d2 ? 3 : d1 ? 2 : 1;
    if (!t->bytes || t->ndim != dims || t->dim[0] != d0 ||
        (d1 && t->dim[1] != d1) || (d2 && t->dim[2] != d2) ||
        (!d1 && t->type != DS4_TENSOR_F32)) {
        fprintf(stderr, "ds4: invalid Hy4 tensor %.*s\n", (int)t->name.len, t->name.ptr);
        ds4_die("invalid Hy4 Flash tensor layout");
    }
    return t;
}

static hy4_hc_weights hy4_bind_hc(const ds4_model *m, int layer, const char *prefix, unsigned count) {
    char name[64];
    hy4_hc_weights w;
    snprintf(name, sizeof(name), "%s_fn.weight", prefix);
    w.fn = hy4_weight(m, layer, name, HY4_DIM * 4, count, 0);
    snprintf(name, sizeof(name), "%s_scale.weight", prefix);
    w.scale = hy4_weight(m, layer, name, count / 4, 0, 0);
    snprintf(name, sizeof(name), "%s_base.weight", prefix);
    w.base = hy4_weight(m, layer, name, count, 0, 0);
    return w;
}

static hy4_ffn_weights hy4_bind_ffn(const ds4_model *m, int layer, const char *suffix,
                                    unsigned hidden, unsigned mid, unsigned experts) {
    char name[64];
    hy4_ffn_weights w;
    snprintf(name, sizeof(name), "ffn_gate%s.weight", suffix);
    w.gate = hy4_weight(m, layer, name, hidden, mid, experts);
    snprintf(name, sizeof(name), "ffn_up%s.weight", suffix);
    w.up = hy4_weight(m, layer, name, hidden, mid, experts);
    snprintf(name, sizeof(name), "ffn_down%s.weight", suffix);
    w.down = hy4_weight(m, layer, name, mid, hidden, experts);
    return w;
}

static void hy4_metadata_array(const ds4_model *m, const char *key,
                                uint32_t type, void *out, size_t count, size_t element) {
    ds4_array_ref a;
    if (!model_get_array(m, key, &a) || a.len != count ||
        (a.type != type && !(type == GGUF_VALUE_UINT32 && a.type == GGUF_VALUE_INT32) &&
         !(type == GGUF_VALUE_UINT64 && a.type == GGUF_VALUE_INT64)))
        ds4_die("invalid Hy4 Engram metadata");
    ds4_cursor c = cursor_at(m, a.data_pos);
    if (!cursor_read(&c, out, count * element)) ds4_die(c.error);
}

static bool hy4_open_engram(ds4_hy4_model *h, const ds4_model *m, const char *path) {
    uint16_t index = 0, count = 0;
    if (!hy4_split_field(m, "split.no", &index) || !hy4_split_field(m, "split.count", &count) ||
        index != 0 || count != 2 || model_find_tensor(m, "per_layer_token_embd.weight")) {
        fprintf(stderr, "ds4: Hy4 Flash needs a separate GGUF Engram shard\n");
        return false;
    }
    static const char first[] = "-00001-of-00002.gguf";
    static const char second[] = "-00002-of-00002.gguf";
    const size_t len = strlen(path);
    if (len < sizeof(first) - 1 || strcmp(path + len - (sizeof(first) - 1), first)) {
        fprintf(stderr, "ds4: Hy4 Flash expects the first of two GGUF shards\n");
        return false;
    }

    char *next = ds4_strdup(path);
    memcpy(next + len - (sizeof(first) - 1), second, sizeof(second));
    if (access(next, R_OK) != 0) {
        fprintf(stderr, "ds4: missing Hy4 Engram shard %s\n", next);
        free(next);
        return false;
    }
    ds4_model shard;
    model_open(&shard, next, false, false);
    free(next);
    const ds4_tensor *t = model_find_tensor(&shard, "per_layer_token_embd.weight");
    uint16_t split_no = 0, num_splits = 0;
    ds4_str arch;
    const bool valid = model_get_string(&shard, "general.architecture", &arch) &&
        ds4_streq(arch, "hy_v4_flash") && hy4_split_field(&shard, "split.no", &split_no) &&
        hy4_split_field(&shard, "split.count", &num_splits) && split_no == 1 && num_splits == 2 &&
        shard.n_tensors == 1 && t && t->ndim == 2 && t->dim[0] == HY4_ENGRAM_DIM &&
        t->dim[1] == HY4_ENGRAM_ROWS && (t->type == DS4_TENSOR_F8_E4M3 || t->type == DS4_TENSOR_Q4_0);
    if (valid) {
        h->table_offset = t->abs_offset;
        h->table_type = t->type;
        h->row_bytes = t->type == DS4_TENSOR_F8_E4M3 ? 96 : 54;
        h->fd = dup(shard.fd);
        if (h->fd >= 0) {
            (void)fcntl(h->fd, F_SETFD, FD_CLOEXEC);
            (void)fcntl(h->fd, F_NOCACHE, 1);
            (void)fcntl(h->fd, F_RDAHEAD, 0);
        }
    }
    model_close(&shard);
    if (!valid || h->fd < 0) {
        fprintf(stderr, "ds4: invalid or unreadable Hy4 Engram shard\n");
        return false;
    }
    hy4_metadata_array(m, "hy_v4_flash.ple.layer_multipliers", GGUF_VALUE_UINT64,
                       h->multiplier, 12, sizeof(uint64_t));
    hy4_metadata_array(m, "hy_v4_flash.ple.head_offsets", GGUF_VALUE_UINT64,
                       h->offsets, 64, sizeof(uint64_t));
    hy4_metadata_array(m, "hy_v4_flash.ple.head_vocab_sizes", GGUF_VALUE_UINT64,
                       h->primes, 64, sizeof(uint64_t));
    hy4_metadata_array(m, "hy_v4_flash.engram.token_map", GGUF_VALUE_UINT32,
                       h->token_map, HY4_VOCAB, sizeof(uint32_t));
    for (unsigned i = 0; i < 64; i++)
        if (!h->primes[i] || h->offsets[i] >= HY4_ENGRAM_ROWS ||
            h->primes[i] > HY4_ENGRAM_ROWS - h->offsets[i])
            ds4_die("Hy4 Engram head exceeds its table");
    const uint32_t pad = required_u32(m, "hy_v4_flash.engram.pad_token_id");
    if (pad >= HY4_VOCAB || h->token_map[pad] == UINT32_MAX)
        ds4_die("invalid Hy4 Engram padding token");
    h->mapped_pad = h->token_map[pad];
    return true;
}

static void hy4_bind_model(ds4_hy4_model *h, const ds4_model *m) {
    h->embedding = hy4_weight(m, -1, "token_embd.weight", HY4_DIM, HY4_VOCAB, 0);
    h->output = hy4_weight(m, -1, "output.weight", HY4_DIM, HY4_VOCAB, 0);
    h->norm = hy4_weight(m, -1, "output_norm.weight", HY4_DIM, 0, 0);
    h->head = hy4_bind_hc(m, -1, "output_hc", 4);
    for (int i = 0; i < HY4_LAYERS; i++) {
        hy4_layer_weights *l = &h->layer[i];
        const unsigned heads = hy4_sparse(i) ? 32 : 64;
#define W(field, name, d0, d1, d2) l->field = hy4_weight(m, i, name, d0, d1, d2)
        l->attn_hc = hy4_bind_hc(m, i, "hc_attn", 8);
        l->ffn_hc = hy4_bind_hc(m, i, "hc_ffn", 8);
        W(attn_norm, "attn_norm.weight", HY4_DIM, 0, 0);
        W(qa, "attn_q_a.weight", HY4_DIM, 1024, 0);
        W(qa_norm, "attn_q_a_norm.weight", 1024, 0, 0);
        W(qb, "attn_q_b.weight", 1024, heads * 256, 0);
        W(kv, "attn_kv_a_mqa.weight", HY4_DIM, HY4_KV, 0);
        W(kv_norm, "attn_kv_a_norm.weight", 512, 0, 0);
        W(kb, "attn_k_b.weight", 192, 512, heads);
        W(vb, "attn_v_b.weight", 512, 256, heads);
        W(attn_gate, "attn_gate.weight", HY4_DIM, heads * 256, 0);
        W(attn_out, "attn_output.weight", heads * 256, HY4_DIM, 0);
        W(attn_post, "post_attention_norm.weight", HY4_DIM, 0, 0);
        W(sink, "attn_sinks.weight", heads, 0, 0);
        W(ffn_norm, "ffn_norm.weight", HY4_DIM, 0, 0);
        W(ffn_post, "post_ffw_norm.weight", HY4_DIM, 0, 0);
        l->ffn = hy4_bind_ffn(m, i, i ? "_shexp" : "", HY4_DIM, i ? HY4_LATENT : 13824, 0);
        if (i) {
            l->experts = hy4_bind_ffn(m, i, "_exps", HY4_LATENT, HY4_LATENT, HY4_EXPERTS);
            if (l->experts.gate->type != l->experts.up->type)
                ds4_die("Hy4 experts need matching gate and up tensor types");
            W(router, "ffn_gate_inp.weight", HY4_DIM, HY4_EXPERTS, 0);
            W(router_bias, "exp_probs_b.bias", HY4_EXPERTS, 0, 0);
            W(latent_in, "ffn_routed_down.weight", HY4_DIM, HY4_LATENT, 0);
            W(latent_out, "ffn_routed_up.weight", HY4_LATENT, HY4_DIM, 0);
        }
        if (hy4_sparse(i)) {
            W(index_q, "indexer.attn_q_b.weight", 1024, 16 * 128, 0);
            W(index_k, "indexer.attn_k.weight", HY4_DIM, 128, 0);
            W(index_norm, "indexer.k_norm.weight", 128, 0, 0);
            W(index_w, "indexer.proj.weight", HY4_DIM, 16, 0);
            W(pool_v, "indexer_compressor_value.weight", 128, 128, 0);
            W(pool_gate, "indexer_compressor_gate.weight", 128, 128, 0);
            W(ape, "indexer_compressor_ape.weight", 128, 4, 0);
            W(pool_norm, "indexer_compressor_norm.weight", 128, 0, 0);
        }
        if (hy4_engram_layer(i)) {
            W(engram_kv, "engram_kv.weight", 1536, HY4_DIM * 5, 0);
            W(engram_bias, "engram_kv_b.weight", HY4_DIM, 5, 0);
            W(engram_gate, "engram_gate.weight", HY4_DIM, 4, 0);
            W(engram_norm, "engram_lookup_norm.weight", HY4_ENGRAM_DIM, HY4_ENGRAM_HEADS, 0);
            W(engram_gamma, "engram_pre_gamma.weight", HY4_DIM, 0, 0);
            W(engram_scale, "engram_hc_scale.weight", HY4_DIM, 4, 0);
        }
#undef W
    }
}

static const ds4_tensor *hy4_draft_tensor(const ds4_model *m, const char *suffix,
                                         uint32_t in, uint32_t out) {
    char name[128];
    snprintf(name, sizeof(name), "model.dfly_layers.%s", suffix);
    const ds4_tensor *t = hy4_weight(m, -1, name, in, out, 0);
    if (out && t->type != DS4_TENSOR_Q6_K)
        ds4_die("unsupported Maya-XS DFly2 tensor type");
    return t;
}

static ds4_gpu_tensor *hy4_draft_transpose_keys(const ds4_model *m, const ds4_tensor *w) {
    enum { HEADS = 32, KEY = 192, VALUE = 256, RANK = 512 };
    const size_t count = HEADS * RANK * KEY;
    const size_t stride = w->bytes / (HEADS * (KEY + VALUE));
    ds4_gpu_tensor *keys = ds4_gpu_tensor_alloc(count * sizeof(uint16_t));
    if (!keys) return NULL;
    uint16_t *data = xmalloc(count * sizeof(*data));
    for (uint32_t head = 0; head < HEADS; head++) {
        for (uint32_t k = 0; k < KEY; k++) {
            float row[RANK];
            const void *src = m->map + w->abs_offset + (head * (KEY + VALUE) + k) * stride;
            if (!ds4_dequant_row(w->type, src, row, RANK)) ds4_die("invalid DFly2 key weights");
            for (uint32_t j = 0; j < RANK; j++)
                data[(head * RANK + j) * KEY + k] = f32_to_f16(row[j]);
        }
    }
    const bool ok = ds4_gpu_tensor_write(keys, 0, data, count * sizeof(*data));
    free(data);
    if (ok) return keys;
    ds4_gpu_tensor_free(keys);
    return NULL;
}

static bool hy4_open_draft(ds4_hy4_model *h, ds4_model *m, const char *path, int rows) {
    model_open(m, path, true, false);
    ds4_str arch, kind;
    uint32_t count = 0;
    const char *key_layers[] = {"hy_v4_flash_dfly.target_kv_layers"};
    const char *index_layers[] = {"hy_v4_flash_dfly.target_index_layers"};
    ds4_array_ref key_ref, index_ref;
    uint32_t index[HY4_DFLY_LAYERS];
    hy4_dfly_weights *d = h->draft = xcalloc(1, sizeof(*d));
    if (!model_get_string(m, "general.architecture", &arch) || !ds4_streq(arch, "hy_v4_flash_dfly") ||
        !model_get_string(m, "general.type", &kind) || !ds4_streq(kind, "drafter") ||
        required_u32(m, "hy_v4_flash_dfly.block_count") != HY4_DFLY_LAYERS ||
        !model_get_array(m, key_layers[0], &key_ref) || key_ref.len != HY4_DFLY_LAYERS ||
        !model_get_array(m, index_layers[0], &index_ref) || index_ref.len != HY4_DFLY_LAYERS ||
        !model_get_u32_array_any(m, key_layers, 1, d->target, HY4_DFLY_LAYERS, &count) ||
        count != HY4_DFLY_LAYERS ||
        !model_get_u32_array_any(m, index_layers, 1, index, HY4_DFLY_LAYERS, &count) ||
        count != HY4_DFLY_LAYERS || memcmp(index, d->target, sizeof(index)) != 0)
        return false;
    d->mask = required_u32(m, "hy_v4_flash_dfly.mask_token_id");
    if (d->mask >= HY4_VOCAB ||
        required_u32(m, "hy_v4_flash_dfly.embedding_length") != HY4_DIM ||
        required_u32(m, "hy_v4_flash_dfly.attention.head_count") != 32 ||
        required_u32(m, "hy_v4_flash_dfly.attention.kv_lora_rank") != 512 ||
        required_u32(m, "hy_v4_flash_dfly.attention.q_lora_rank") != 1024 ||
        required_u32(m, "hy_v4_flash_dfly.target_index_topk") != 2048 ||
        !ds4_gpu_set_model_map_range(m->map, m->size, m->tensor_data_pos,
                                    m->size - m->tensor_data_pos, m->max_tensor_bytes))
        return false;
    d->rows = rows > 1 && rows <= HY4_DFLY_ROWS ? (uint32_t)rows : 3;
    for (uint32_t i = 0; i < HY4_DFLY_LAYERS; i++) {
        if (d->target[i] >= HY4_LAYERS || !hy4_sparse(d->target[i]) ||
            (i && d->target[i] <= d->target[i-1])) return false;
        hy4_layer_weights *l = &d->layer[i];
#define DW(field, suffix, in, out) do { \
    char name[96]; \
    snprintf(name, sizeof(name), "%u.%s", i, suffix); \
    l->field = hy4_draft_tensor(m, name, in, out); \
} while (0)
        DW(attn_norm, "input_layernorm.weight", HY4_DIM, 0);
        DW(qa, "self_attn.q_a_proj.weight", HY4_DIM, 1024);
        DW(qa_norm, "self_attn.q_a_layernorm.weight", 1024, 0);
        DW(qb, "self_attn.q_b_proj.weight", 1024, 32 * 256);
        DW(kv, "self_attn.kv_a_proj_with_mqa.weight", HY4_DIM, HY4_KV);
        DW(kv_norm, "self_attn.kv_a_layernorm.weight", 512, 0);
        DW(vb, "self_attn.kv_b_proj.weight", 512, 32 * (192 + 256));
        DW(sink, "self_attn.learnable_sink_param", 32, 0);
        DW(attn_gate, "self_attn.linear_gate.weight", HY4_DIM, 32 * 256);
        DW(attn_out, "self_attn.o_proj.weight", 32 * 256, HY4_DIM);
        DW(attn_post, "self_attn.post_proj_norm.weight", HY4_DIM, 0);
        DW(ffn_norm, "post_attention_layernorm.weight", HY4_DIM, 0);
        DW(ffn.gate, "mlp.gate_proj.weight", HY4_DIM, HY4_DFLY_FF);
        DW(ffn.up, "mlp.up_proj.weight", HY4_DIM, HY4_DFLY_FF);
        DW(ffn.down, "mlp.down_proj.weight", HY4_DFLY_FF, HY4_DIM);
#undef DW
        d->keys[i] = hy4_draft_transpose_keys(m, l->vb);
        if (!d->keys[i]) return false;
    }
    d->enorm = hy4_draft_tensor(m, "mtp_fusion.enorm.weight", HY4_DIM, 0);
    d->hnorm = hy4_draft_tensor(m, "mtp_fusion.hnorm.weight", HY4_DIM, 0);
    d->fuse = hy4_draft_tensor(m, "mtp_fusion.eh_proj.weight", 2 * HY4_DIM, HY4_DIM);
    d->gate = hy4_draft_tensor(m, "mtp_fusion.gate", HY4_DIM, 0);
    d->norm = hy4_draft_tensor(m, "final_norm.weight", HY4_DIM, 0);
    d->embed_norm = hy4_draft_tensor(m, "hidden_correction.embed_norm.weight", HY4_DIM, 0);
    d->hidden_norm = hy4_draft_tensor(m, "hidden_correction.hidden_norm.weight", HY4_DIM, 0);
    d->correction.gate = hy4_draft_tensor(m, "hidden_correction.gate_proj.weight", 2 * HY4_DIM, HY4_DIM);
    d->correction.up = hy4_draft_tensor(m, "hidden_correction.up_proj.weight", 2 * HY4_DIM, HY4_DIM);
    d->correction.down = hy4_draft_tensor(m, "hidden_correction.down_proj.weight", HY4_DIM, HY4_DIM);
    return true;
}

static void hy4_draft_weights_free(hy4_dfly_weights *d) {
    if (!d) return;
    for (uint32_t i = 0; i < HY4_DFLY_LAYERS; i++) ds4_gpu_tensor_free(d->keys[i]);
    free(d);
}
#endif
