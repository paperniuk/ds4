#ifndef DS4_QUANTS_H
#define DS4_QUANTS_H

#include <stdbool.h>
#include <stdint.h>

/* CPU dequantization of one row of a GGUF tensor, the reference the Metal
 * kernels and the host-side embedding lookups use. */
bool ds4_quant_row_bytes(uint32_t type, uint64_t n, uint64_t *bytes);
bool ds4_dequant_row(uint32_t type, const void *src, float *dst, uint64_t n);

#endif
