#ifndef DS4_HY4_REWIND_H
#define DS4_HY4_REWIND_H

#include <stdbool.h>
#include <stdint.h>

enum { HY4_WINDOW = 512, HY4_REWIND = 4096 };

/* The normal allocation covers the history each chunk reads plus its own
 * writes. The extra rows cover rewinds even if up to cap - 1 rejected draft
 * rows and one extra token for logits lie between the two frontiers. */
static inline uint32_t hy4_raw_ring_capacity(uint32_t ctx, uint32_t cap) {
    const uint32_t rows = HY4_WINDOW - 1 + cap + HY4_REWIND;
    return rows < ctx ? rows : ctx;
}

static inline uint32_t hy4_pending_ring_capacity(uint32_t ctx, uint32_t cap) {
    const uint32_t rows = 3 + cap + HY4_REWIND;
    return rows < ctx ? rows : ctx;
}

static inline uint32_t hy4_window_start(uint32_t pos) {
    return pos < HY4_WINDOW ? 0 : pos - (HY4_WINDOW - 1);
}

/* A write at pos replaces the old suffix. Rows before pos survive only
 * if they were present already and the writes through end - 1 did not wrap
 * over them. Starting before first leaves no retained prefix in this ring. */
static inline uint32_t hy4_ring_after_write(uint32_t first, uint32_t pos,
                                           uint32_t end, uint32_t capacity) {
    if (first > pos) first = pos;
    const uint32_t oldest = end > capacity ? end - capacity : 0;
    return first > oldest ? first : oldest;
}

/* Attention needs the preceding 511 rows at most. The pool only needs
 * staged rows from a partially completed group. With no pool, pending_first
 * is zero. */
static inline bool hy4_cache_can_resume(uint32_t pos, uint32_t raw_first,
                                        uint32_t pending_first) {
    const uint32_t open = pos % 4;
    return pos == 0 || (raw_first <= hy4_window_start(pos) &&
                        (!open || pending_first <= pos - open));
}

static inline uint32_t hy4_rewind_start(uint32_t checkpoint, uint32_t prompt,
                                        uint32_t common, uint32_t raw_first,
                                        uint32_t pending_first) {
    if (!common || common >= checkpoint || common > prompt ||
        checkpoint - common > HY4_REWIND) return 0;
    /* A strict prefix needs its final row rerun to recover the logits. */
    const uint32_t start = common == prompt ? common - 1 : common;
    return hy4_cache_can_resume(start, raw_first, pending_first) ? start : 0;
}

#endif
