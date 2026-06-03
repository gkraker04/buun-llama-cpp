#include "turbo-matmul.cuh"
#include "turbo-quant-cuda.cuh"
#include "turbo-wht.cuh"
#include "convert.cuh"
#include "quantize.cuh"
#include "common.cuh"

// ============================================================
// Activation pre-rotation via existing WHT kernel
// Activation pre-rotation via shared WHT wrapper
static void prerotate_activations(
    const float * src, float * dst, int64_t n_elements, cudaStream_t stream) {
    ggml_cuda_turbo_wht_forward(src, dst, n_elements, stream);
}

// ============================================================
// LUT for turbo4 4-bit indices → centroid values
// ============================================================
// Centroids for 4-bit turbo (from fattn-common.cuh d_turbo_centroids_4bit_fattn)
// We represent them as int8 LUT for efficient dp4a-style dot products
static constexpr __device__ float k_turbo4_centroids_f32[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

// ============================================================
// Path A2: Single-token decode with pre-quantized q8_1 activations
// ============================================================
static __global__ void k_turbo4_mul_mat_vec_q8(
    const block_turbo4_0 * __restrict__ vx,
    const block_q8_1 * __restrict__ vq8,
    float * __restrict__        dst,
    const int ncols_x,
    const int nrows_x) {

    const int row = blockIdx.x;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;  // 0..31

    const int blocks_per_row = ncols_x / QK_TURBO4;
    const int q8_per_block = QK_TURBO4 / QK8_1;  // 8

    const block_turbo4_0 * x_row = vx + (int64_t)row * blocks_per_row;
    float sumf = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const block_turbo4_0 * blk = &x_row[ib];
        const float norm = __half2float(blk->norm);
        const int blk_idx_q8 = ib * q8_per_block;

        for (int g = 0; g < q8_per_block; g++) {
            const block_q8_1 * q8_blk = &vq8[blk_idx_q8 + g];

            const int elem = g * QK8_1 + lane;
            const uint8_t byte_val = blk->qs[elem / 2];
            const int nibble_shift = (elem % 2) * 4;
            const uint8_t idx = (byte_val >> nibble_shift) & 0xF;
            const float w = k_turbo4_centroids_f32[idx];

            const signed char a_i8 = q8_blk->qs[lane];
            const float d = __low2float(q8_blk->ds);

            sumf += w * (float)((int)a_i8) * norm * d;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);

    if (lane == 0) dst[row] = sumf;
}

// ============================================================
// Path A1: Single-token decode — working simple kernel
static __global__ void k_turbo4_mul_mat_vec_simple(
    const block_turbo4_0 * __restrict__ vx,
    const float * __restrict__  vy,
    float * __restrict__        dst,
    const int ncols_x,
    const int nrows_x) {

    const int row = blockIdx.x;
    if (row >= nrows_x) return;
    const int tid = threadIdx.x;  // 0..255
    const int blocks_per_row = ncols_x / QK_TURBO4;
    const block_turbo4_0 * x_row = vx + (int64_t)row * blocks_per_row;
    float sumf = 0.0f;

    // Each thread handles a subset of elements (strided)
    for (int i = tid; i < ncols_x; i += blockDim.x) {
        const int blk_idx = i / QK_TURBO4;
        const int elem_in_blk = i % QK_TURBO4;
        const block_turbo4_0 * blk = &x_row[blk_idx];
        const float norm = __half2float(blk->norm);
        const uint8_t idx = (blk->qs[elem_in_blk / 2] >> ((elem_in_blk % 2) * 4)) & 0xF;
        const float w = d_turbo_centroids_4bit[idx] * norm;
        sumf += vy[i] * w;
    }

    // Warp-level reduction (32 threads per warp)
    const int lane = tid % 32;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);

    // Block-level reduction: warp leaders write to shared memory
    __shared__ float warp_sums[8];  // max 256 threads = 8 warps
    if (lane == 0) {
        const int warp_id = tid / 32;
        if (warp_id < 8)
            warp_sums[warp_id] = sumf;
    }
    __syncthreads();

    if (lane == 0 && tid / 32 == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++)
            total += warp_sums[w];
        dst[row] = total;
    }
}

// ============================================================
// Path A-original: Single-token decode (ne11 == 1) — parallel
// ============================================================
static __global__ void k_turbo4_mul_mat_vec(
    const block_turbo4_0 * __restrict__ vx,
    const float * __restrict__  vy,
    float * __restrict__        dst,
    const int ncols_x,
    const int nrows_x) {

    const int row = blockIdx.x;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TURBO4;
    const block_turbo4_0 * x_row = vx + (int64_t)row * blocks_per_row;
    float sumf = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const block_turbo4_0 * blk = &x_row[ib];
        const float norm = __half2float(blk->norm);
        const int block_base = ib * QK_TURBO4;

        #pragma unroll
        for (int je = 0; je < 4; je++) {
            const int elem = lane + je * 32;
            uint8_t idx = (blk->qs[elem / 2] >> ((elem % 2) * 4)) & 0xF;
            const float w = d_turbo_centroids_4bit[idx] * norm;
            sumf += vy[block_base + elem] * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);

    if (lane == 0) dst[row] = sumf;
}

// ============================================================
// Path B: Multi-token (ne11 <= 8)
// ============================================================
template<int ncols_dst>
static __global__ void k_turbo4_mul_mat_multi(
    const block_turbo4_0 * __restrict__ vx,
    const float * __restrict__  vy,
    float * __restrict__        dst,
    const int ncols_x, const int nrows_x,
    const int stride_y, const int stride_dst) {

    const int row = blockIdx.x;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TURBO4;
    const block_turbo4_0 * x_row = vx + (int64_t)row * blocks_per_row;
    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const block_turbo4_0 * blk = &x_row[ib];
        const float norm = __half2float(blk->norm);
        const int block_base = ib * QK_TURBO4;

        #pragma unroll
        for (int je = 0; je < 4; je++) {
            const int elem = lane + je * 32;
            uint8_t idx = (blk->qs[elem / 2] >> ((elem % 2) * 4)) & 0xF;
            const float w = d_turbo_centroids_4bit[idx] * norm;

            #pragma unroll
            for (int t = 0; t < ncols_dst; t++)
                sumf[t] += vy[t * stride_y + block_base + elem] * w;
        }
    }

    #pragma unroll
    for (int t = 0; t < ncols_dst; t++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sumf[t] += __shfl_xor_sync(0xffffffff, sumf[t], offset);
    }

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < ncols_dst; t++)
            dst[t * stride_dst + row] = sumf[t];
    }
}

template<int N>
static void launch_turbo4_multi(
    const void * vx, const float * vy, float * dst,
    int ncols_x, int nrows_x, int stride_y, int stride_dst,
    cudaStream_t stream) {
    const dim3 block(32, 1);
    const dim3 grid(nrows_x, 1);
    k_turbo4_mul_mat_multi<N><<<grid, block, 0, stream>>>(
        (const block_turbo4_0 *)vx, vy, dst,
        ncols_x, nrows_x, stride_y, stride_dst);
}

// ============================================================
// Path C: cuBLAS fallback via GPU dequant
// ============================================================
static __global__ void k_convert_turbo4_to_fp16(
    const block_turbo4_0 * __restrict__ src,
    half * __restrict__ dst,
    const int64_t n_blocks) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;
    turbo4_dequant_block_to_half(&src[block_idx], dst, (int64_t)block_idx * QK_TURBO4);
}

static void ggml_cuda_mul_mat_turbo_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_TURBO4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];
    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    // Pre-rotate activations
    ggml_cuda_pool_alloc<float> rotated(ctx.pool(id), ne00 * ne11);
    prerotate_activations((const float *)src1->data, rotated.get(), ne00 * ne11, stream);

    // Dequant weights to fp16
    const int64_t n_blocks = (int64_t)ne00 * ne01 / QK_TURBO4;
    ggml_cuda_pool_alloc<half> w_f16(ctx.pool(id), ne00 * ne01);
    {
        const int grid = (int)((n_blocks + 255) / 256);
        k_convert_turbo4_to_fp16<<<grid, 256, 0, stream>>>(
            (const block_turbo4_0 *)src0->data, w_f16.get(), n_blocks);
    }

    // Convert rotated activations to fp16
    ggml_cuda_pool_alloc<half> a_f16(ctx.pool(id), ne00 * ne11);
    {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16((const char *)rotated.get(), a_f16.get(), ne00 * ne11, stream);
    }

    // cuBLAS GEMM
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(id),
        CUBLAS_OP_T, CUBLAS_OP_N,
        ne01, ne11, ne00, &alpha,
        w_f16.get(), CUDA_R_16F, ne00,
        a_f16.get(), CUDA_R_16F, ne00, &beta,
        (float *)dst->data, CUDA_R_32F, dst->ne[0],
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================
// Main dispatch
// ============================================================
void ggml_cuda_mul_mat_turbo(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {

    GGML_ASSERT(src0->type == GGML_TYPE_TURBO4_0 ||
                src0->type == GGML_TYPE_TURBO3_0 ||
                src0->type == GGML_TYPE_TURBO2_0);
    const int ncols_x = src0->ne[0], nrows_x = src0->ne[1], ncols_dst = src1->ne[1];
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();

    // Step 1: Pre-rotate all activations
    const int64_t n_act = (int64_t)ncols_x * ncols_dst;
    ggml_cuda_pool_alloc<float> rotated(ctx.pool(id), n_act);
    prerotate_activations((const float *)src1->data, rotated.get(), n_act, stream);

    const void * src0_d = src0->data;
    float * dst_d = (float *)dst->data;
    const int stride_y = ncols_x, stride_dst = nrows_x;

    if (ncols_dst == 1) {
        // Single-token: quantize rotated activations to q8_1, then vec_dot
        const int64_t ne10_padded = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
        const int64_t n_q8_bytes = ne10_padded * (int64_t)sizeof(block_q8_1) / QK8_1;
        ggml_cuda_pool_alloc<char> q8_buf(ctx.pool(id), n_q8_bytes);
        {
            const int64_t s01 = ncols_x;  // stride between rows
            quantize_row_q8_1_cuda(rotated.get(), nullptr, q8_buf.get(),
                GGML_TYPE_F32, ncols_x, s01, ncols_x, ncols_x,
                ne10_padded, 1, 1, 1, stream);
        }
        k_turbo4_mul_mat_vec_q8<<<nrows_x, 32, 0, stream>>>(
            (const block_turbo4_0 *)src0_d, (const block_q8_1 *)q8_buf.get(),
            dst_d, ncols_x, nrows_x);
        CUDA_CHECK(cudaGetLastError());
    } else if (ncols_dst <= 8) {
        // Multi-token: use cuBLAS path (verified correct)
        ggml_cuda_mul_mat_turbo_cublas(ctx, src0, src1, dst);
    } else {
        // Large batch: cuBLAS
        ggml_cuda_mul_mat_turbo_cublas(ctx, src0, src1, dst);
    }
}
