#pragma once
#include "common.cuh"
#include "ggml-cuda.h"

// Dequantize turbo4 block to fp16 with full WHT cycle (s2 → FWHT → normalize → s1)
// 128 threads per block, one block per turbo4 block. Uses shared memory for FWHT.
__global__ void k_convert_turbo4_to_fp16_orig(
    const block_turbo4_0 * __restrict__ src,
    half * __restrict__ dst,
    const int64_t n_blocks);

void ggml_cuda_mul_mat_turbo(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);
