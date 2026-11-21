#ifndef DS4_HY4_ANE_SCHEDULE_H
#define DS4_HY4_ANE_SCHEDULE_H

#include <stdint.h>

enum { HY4_ANE_MIN_ROWS = 256 };

/* Keep full chunks when both can run, or balance a short final pair. */
static inline uint32_t hy4_ane_pair_rows(uint32_t remaining, uint32_t cap, uint32_t *first) {
    *first = remaining < cap ? remaining : cap;
    if (cap < HY4_ANE_MIN_ROWS || remaining < 2u * HY4_ANE_MIN_ROWS) return 0;
    if (remaining - *first < HY4_ANE_MIN_ROWS) *first = (remaining + 1u) / 2u;
    const uint32_t rest = remaining - *first;
    return rest < cap ? rest : cap;
}

/* The live prefix occupies the `live` rows just before pos. Sequential
 * writes overwrite the oldest of those rows first. */
static inline uint32_t hy4_ane_ring_overlap(uint32_t live, uint32_t ring, uint32_t writes) {
    const uint32_t untouched = writes < ring ? ring - writes : 0;
    return live > untouched ? live - untouched : 0;
}

#endif
