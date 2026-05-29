#include "common.cuh"

// Forward declaration of WHT kernel (defined in turbo-wht.cu)
__global__ void k_turbo_wht(
    const float * __restrict__ src, float * __restrict__ dst,
    const int64_t n_elements, const int direction);

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
