#include "turbo-matmul.cuh"
#include "turbo-quant-cuda.cuh"
#include "turbo-wht.cuh"
#include "convert.cuh"

// ============================================================
// Activation pre-rotation via existing WHT kernel
// ============================================================
static void prerotate_activations(
    const float * src, float * dst, int64_t n_elements, cudaStream_t stream) {
    k_turbo_wht<<<(int)(n_elements / 128), 128, 0, stream>>>(
        src, dst, n_elements, 0);
}

// ============================================================
// Path A: Single-token decode (ne11 == 1)
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
        k_turbo4_mul_mat_vec<<<nrows_x, 32, 0, stream>>>(
            (const block_turbo4_0 *)src0_d, rotated.get(), dst_d, ncols_x, nrows_x);
    } else if (ncols_dst <= 8) {
        switch (ncols_dst) {
            case 1: launch_turbo4_multi<1>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 2: launch_turbo4_multi<2>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 3: launch_turbo4_multi<3>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 4: launch_turbo4_multi<4>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 5: launch_turbo4_multi<5>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 6: launch_turbo4_multi<6>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 7: launch_turbo4_multi<7>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
            case 8: launch_turbo4_multi<8>(src0_d, rotated.get(), dst_d, ncols_x, nrows_x, stride_y, stride_dst, stream); break;
        }
    } else {
        ggml_cuda_mul_mat_turbo_cublas(ctx, src0, src1, dst);
    }
}
