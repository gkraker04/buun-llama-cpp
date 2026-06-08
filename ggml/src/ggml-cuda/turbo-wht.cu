#include "turbo-wht.cuh"
#include "common.cuh"

// Sign arrays for FWHT rotation (from turbo-wht.h, seed=42)
static __constant__ float d_turbo_wht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float d_turbo_wht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// One block per 128-element group. 128 threads per block.
__global__ void k_turbo_wht(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    const float * s_first  = (direction == 0) ? d_turbo_wht_s1 : d_turbo_wht_s2;
    const float * s_second = (direction == 0) ? d_turbo_wht_s2 : d_turbo_wht_s1;

    __shared__ float buf[128];

    // Load and apply first signs
    if (threadIdx.x < 128) {
        buf[threadIdx.x] = src[offset + threadIdx.x] * s_first[threadIdx.x];
    }
    __syncthreads();

    // Parallel FWHT butterfly: 64 threads, 7 passes
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs, write output
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    if (threadIdx.x < 128) {
        dst[offset + threadIdx.x] = buf[threadIdx.x] * inv_sqrt_128 * s_second[threadIdx.x];
    }
}

// Raw-buffer wrapper: forward WHT rotation for activations (direction=0)
void ggml_cuda_turbo_wht_forward(const float * src, float * dst, int64_t n_elements, cudaStream_t stream) {
    const int64_t n_groups = (n_elements + 127) / 128;
    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src, dst, n_elements, 0);
}

// Raw-buffer wrapper: inverse WHT rotation (direction=1)
void ggml_cuda_turbo_wht_inverse(const float * src, float * dst, int64_t n_elements, cudaStream_t stream) {
    const int64_t n_groups = (n_elements + 127) / 128;
    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src, dst, n_elements, 1);
}

// Fused WHT forward + q8_1 quantization kernel
// Eliminates intermediate fp32 buffer: applies WHT in registers, then quantizes directly
// 128 threads per block (4 warps), each warp produces one q8_1 block (32 elements)
// Output: 4 q8_1 blocks per 128-element input group
__global__ void k_turbo_wht_q8_1(
        const float * __restrict__ src, void * __restrict__ dst,
        const int64_t n_elements) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    __shared__ float buf[128];

    // Load and apply first signs (direction=0: forward WHT)
    buf[threadIdx.x] = src[offset + threadIdx.x] * d_turbo_wht_s1[threadIdx.x];
    __syncthreads();

    // Parallel FWHT butterfly: h=1..64
    for (int h = 1; h < 128; h *= 2) {
        if (threadIdx.x < 64) {
            int j = (threadIdx.x / h) * (2 * h) + (threadIdx.x % h);
            float a = buf[j], b = buf[j + h];
            buf[j] = a + b; buf[j + h] = a - b;
        }
        __syncthreads();
    }

    // Normalize and apply second signs — result is in registers
    constexpr float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    float val = buf[threadIdx.x] * inv_sqrt_128 * d_turbo_wht_s2[threadIdx.x];

    // Now quantize to q8_1: each warp (32 threads) produces one q8_1 block
    const int warp_id = threadIdx.x / 32;  // 0..3
    const int lane_id = threadIdx.x % 32;  // 0..31

    // Warp-level reduction for amax and sum
    float amax = fabsf(val);
    float sum = val;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    // Quantize
    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(val / d);

    // Write output: 4 q8_1 blocks per 128-element group
    block_q8_1 * dst_blocks = (block_q8_1 *)dst;
    const int64_t block_idx = group * 4 + warp_id;  // 4 blocks per group

    dst_blocks[block_idx].qs[lane_id] = q;

    // Only lane 0 writes the scale/sum header
    if (lane_id == 0) {
        dst_blocks[block_idx].ds = make_half2(__float2half(d), __float2half(d * sum));
    }
}

// Launch fused WHT + q8_1 quantization
void ggml_cuda_turbo_wht_q8_1(const float * src, void * dst, int64_t n_elements, cudaStream_t stream) {
    const int64_t n_groups = (n_elements + 127) / 128;
    k_turbo_wht_q8_1<<<(int)n_groups, 128, 0, stream>>>(src, dst, n_elements);
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    k_turbo_wht<<<(int)n_groups, 128, 0, stream>>>(src_d, dst_d, n_elements, direction);
}
