#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml-common.h"

// === Shared constants ===
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

static __device__ __forceinline__
uint8_t turbo_find_nearest_3bit(float val) {
    if      (val < d_turbo_mid_3bit[0]) return 0;
    else if (val < d_turbo_mid_3bit[1]) return 1;
    else if (val < d_turbo_mid_3bit[2]) return 2;
    else if (val < d_turbo_mid_3bit[3]) return 3;
    else if (val < d_turbo_mid_3bit[4]) return 4;
    else if (val < d_turbo_mid_3bit[5]) return 5;
    else if (val < d_turbo_mid_3bit[6]) return 6;
    else                                return 7;
}

// === TURBO3: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO3_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    // NOTE: FWHT rotation omitted — no graph-level inverse on CUDA yet.
    // TODO: Add pre-rotate-Q in FA kernel + inverse-rotate output for optimal quality.
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO3;
        blk.norm = __float2half(grp_norm);
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// === TURBO3: GET_ROWS dequantize ===
#define QR_TURBO3_0 2
static __device__ __forceinline__
void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.x = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
    { const int j = iqs + 16;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.y = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
}

// === TURBO4: 3-bit unpack helper ===
static __device__ __forceinline__
uint8_t turbo4_unpack_3bit(const uint8_t * qs, int j) {
    int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
    return (uint8_t)((raw >> bit_pos) & 0x7);
}

// === TURBO4: SET_ROWS quantize ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst->norm = __float2half(norm);
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    for (int j = 0; j < 48; j++) dst->qs[j] = 0;
    for (int j = 0; j < 16; j++) dst->signs[j] = 0;
    float recon[128];
    for (int j = 0; j < 128; j++) {
        uint8_t idx = turbo_find_nearest_3bit(x[j]);
        recon[j] = d_turbo_centroids_3bit[idx];
        int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
        dst->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < 48)
            dst->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
    }
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float r = x[j] - recon[j]; rnorm_sq += r * r;
        if (r >= 0.0f) dst->signs[j / 8] |= (1 << (j % 8));
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));
}

// === TURBO4: GET_ROWS dequantize ===
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    const float rnorm = __half2float(x[ib].rnorm);
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;
    { const int j = iqs;
      uint8_t idx = turbo4_unpack_3bit(x[ib].qs, j);
      float s = (x[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
      v.x = (d_turbo_centroids_3bit[idx] + s * qjl_scale) * norm; }
    { const int j = iqs + 64;
      uint8_t idx = turbo4_unpack_3bit(x[ib].qs, j);
      float s = (x[ib].signs[j/8] & (1 << (j%8))) ? 1.0f : -1.0f;
      v.y = (d_turbo_centroids_3bit[idx] + s * qjl_scale) * norm; }
}
