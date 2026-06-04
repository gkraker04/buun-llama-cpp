// TURBO4 vec_dot — MINIMAL TEST VERSION (returns 0)
#define VDR_TURBO4_0_Q8_1_MMVQ 8
#define VDR_TURBO4_0_Q8_1_MMQ  8

static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    return 0.0f;
}