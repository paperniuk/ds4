/* A Hy4 snapshot contains only populated cache rows. Logical positions are
 * serialized in chronological order so a different prefill cap can restore
 * the same prefix into a differently sized ring. */
enum { HY4_PAYLOAD_MAGIC = 0x34594844, HY4_PAYLOAD_VERSION = 1, HY4_PAYLOAD_HEADER = 4 };

static uint64_t hy4_payload_bytes(uint32_t pos, bool pool) {
    const uint32_t window = pos < HY4_WINDOW ? pos : HY4_WINDOW;
    uint64_t bytes = (HY4_PAYLOAD_HEADER + (uint64_t)pos) * sizeof(uint32_t) + HY4_VOCAB * sizeof(float);
    bytes += ((uint64_t)30 * window + (uint64_t)12 * pos) * HY4_KV * sizeof(uint16_t);
    if (pool) bytes += 12 * ((uint64_t)(pos / 4) * 128 * sizeof(uint16_t) + (pos % 4) * 256 * sizeof(float));
    return bytes;
}

typedef struct {
    FILE *file;
    char *err;
    size_t errlen;
    uint8_t *scratch;
    size_t scratch_bytes;
    uint64_t *remaining;
} hy4_payload_io;

static int hy4_payload_rows(hy4_payload_io *io, ds4_gpu_tensor *tensor,
                            uint32_t row, uint32_t count, uint32_t capacity, uint64_t stride) {
    while (count) {
        const uint32_t start = row % capacity;
        const uint32_t take = count < capacity - start ? count : capacity - start;
        const uint64_t offset = start * stride;
        const uint64_t bytes = take * stride;
        const int rc = io->remaining ?
            payload_read_tensor_span(io->file, tensor, offset, bytes, io->scratch,
                                     io->scratch_bytes, io->remaining, io->err, io->errlen) :
            payload_write_tensor_span(io->file, tensor, offset, bytes, io->scratch,
                                      io->scratch_bytes, io->err, io->errlen);
        if (rc != 0) return rc;
        row += take;
        count -= take;
    }
    return 0;
}

static int hy4_payload_cache(hy4_payload_io *io, ds4_hy4_graph *g, uint32_t pos, bool pool) {
    const uint32_t window = pos < HY4_WINDOW ? pos : HY4_WINDOW;
    for (uint32_t i = 0; i < HY4_LAYERS; i++) {
        hy4_cache *cache = &g->layer[i];
        const bool sparse = hy4_sparse(i);
        if (hy4_payload_rows(io, cache->kv, sparse ? 0 : pos - window, sparse ? pos : window,
                              sparse ? g->ctx : g->ring, HY4_KV * sizeof(uint16_t))) return 1;
        if (pool && sparse &&
            (hy4_payload_rows(io, cache->pool, 0, pos / 4, g->ctx / 4, 128 * sizeof(uint16_t)) ||
             hy4_payload_rows(io, cache->pending, pos / 4 * 4, pos % 4, g->pending_ring, 256 * sizeof(float))))
            return 1;
    }
    return 0;
}

static int hy4_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    ds4_hy4_graph *g = s->hy4_graph;
    const bool pool = g->scores != NULL;
    const uint32_t pos = s->checkpoint.len;
    const uint32_t header[] = {HY4_PAYLOAD_MAGIC, HY4_PAYLOAD_VERSION, pos, pool};
    for (size_t i = 0; i < sizeof(header)/sizeof(header[0]); i++)
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    for (uint32_t i = 0; i < pos; i++)
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    if (payload_write_bytes(fp, s->logits, HY4_VOCAB * sizeof(float), err, errlen) != 0) return 1;
    hy4_payload_io io = {.file = fp, .err = err, .errlen = errlen, .scratch_bytes = 1u << 20};
    io.scratch = xmalloc(io.scratch_bytes);
    const int rc = hy4_payload_cache(&io, g, pos, pool);
    free(io.scratch);
    return rc;
}

static int hy4_load_payload(ds4_session *s, FILE *fp, uint64_t bytes, char *err, size_t errlen) {
    uint64_t remaining = bytes;
    uint32_t header[HY4_PAYLOAD_HEADER];
    for (size_t i = 0; i < sizeof(header)/sizeof(header[0]); i++)
        if (payload_read_u32(fp, &header[i], &remaining, err, errlen) != 0) return 1;
    const uint32_t pos = header[2], pool = header[3];
    ds4_hy4_graph *g = s->hy4_graph;
    if (header[0] != HY4_PAYLOAD_MAGIC || header[1] != HY4_PAYLOAD_VERSION ||
        !pos || pos >= g->ctx || pool != (g->scores != NULL) || bytes != hy4_payload_bytes(pos, pool)) {
        payload_set_err(err, errlen, "incompatible Hy4 Flash session payload");
        return 1;
    }
    token_vec tokens = {0};
    for (uint32_t i = 0; i < pos; i++) {
        uint32_t id;
        if (payload_read_u32(fp, &id, &remaining, err, errlen) != 0 || id >= HY4_VOCAB) {
            payload_set_err(err, errlen, "invalid token in Hy4 Flash session payload");
            token_vec_free(&tokens);
            return 1;
        }
        token_vec_push(&tokens, (int)id);
    }
    s->checkpoint_valid = false;
    if (payload_read_bytes(fp, s->logits, HY4_VOCAB * sizeof(float), &remaining, err, errlen) != 0) {
        token_vec_free(&tokens);
        return 1;
    }
    hy4_payload_io io = {.file = fp, .err = err, .errlen = errlen,
                          .scratch_bytes = 1u << 20, .remaining = &remaining};
    io.scratch = xmalloc(io.scratch_bytes);
    const int rc = hy4_payload_cache(&io, g, pos, pool);
    free(io.scratch);
    if (rc != 0 || remaining) {
        token_vec_free(&tokens);
        return 1;
    }
    token_vec_free(&s->checkpoint);
    s->checkpoint = tokens;
    hy4_draft_reset(g->draft);
    s->checkpoint_valid = true;
    return 0;
}
