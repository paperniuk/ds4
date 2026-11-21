#ifndef DS4_HY4_METAL_H
#define DS4_HY4_METAL_H

#include <stddef.h>

#include "ds4_gpu.h"

/* Bind the weight directly from the resident GGUF view when tensor is NULL. */
typedef struct {
    const ds4_gpu_tensor *tensor;
    uint64_t offset;
    uint64_t bytes;
} ds4_hy4_binding;

typedef struct {
    uint32_t rows, pos, width, heads, capacity, groups, type, row_bytes;
    float theta;
} ds4_hy4_args;

bool ds4_gpu_hy4_use_wide_q6(void);

int ds4_gpu_hy4_dispatch(const char *kernel, const ds4_hy4_args *args,
                         const void *map, uint64_t size,
                         const ds4_hy4_binding *bindings, size_t count,
                         uint32_t x, uint32_t y, uint32_t z, uint32_t threads);

/* Select a specialization for up to eight token rows. */
int ds4_gpu_hy4_dispatch_rows(const char *kernel, const ds4_hy4_args *args,
                              const void *map, uint64_t size,
                              const ds4_hy4_binding *bindings, size_t count,
                              uint32_t x, uint32_t y, uint32_t threads);

/* End a benchmark batch and return its Metal command buffer time. */
int ds4_gpu_hy4_end_timed_commands(double *gpu_ms);

int ds4_gpu_hy4_expert_mid(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x, const ds4_gpu_tensor *ids,
                          const void *map, uint64_t size, uint64_t gate, uint64_t up, uint32_t rows);

int ds4_gpu_hy4_expert_prefill(ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
                               const ds4_gpu_tensor *latent, const ds4_gpu_tensor *ids,
                               ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
                               const void *map, uint64_t size,
                               uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                               uint32_t mid_type, uint32_t down_type, uint32_t n);
#endif
