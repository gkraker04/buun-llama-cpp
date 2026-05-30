#include "common.cuh"

// Forward declaration of WHT kernel (defined in turbo-wht.cu)
__global__ void k_turbo_wht(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_elements, const int direction);

// Raw-buffer wrapper for WHT rotation (non-static, cross-TU safe)
void ggml_cuda_turbo_wht_forward(const float * src, float * dst, int64_t n_elements, cudaStream_t stream);

// Operation wrapper for WHT op in graph (for tensor operations)
void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
