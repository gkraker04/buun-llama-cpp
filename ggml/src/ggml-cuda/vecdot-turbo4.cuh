// === TURBO4: 4-bit Lloyd-Max centroids (symmetric, scaled to int8 for dp4a) ===
// vec_dot for MMVQ path. vec_dot for MMQ path identical when VDR is the same.

#define VDR_TURBO4_0_Q8_1_MMVQ 8
#define VDR_TURBO4_0_Q8_1_MMQ  8

// turbo4 centroids scaled to int8: round(centroid * 127 / 0.241556)
static constexpr __device__ int8_t kvalues_turbo4[16] = {
    -127, -96, -75, -58, -44, -31, -18, -6,
       6,  18,  31,  44,  58,  75,  96, 127,
};
#define TURBO4_SCALE_FACTOR 0.00190201575f  // 0.241556 / 127

// vec_dot for MMVQ: follows IQ4_NL pattern exactly
// - iqs indexes into BOTH qs arrays (Q4 and Q8)
// - Single q8 pointer set before the loop, indexed by l
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_TURBO4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_turbo4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 8], sumi);
    }

    return norm * (float)sumi * TURBO4_SCALE_FACTOR * __low2float(bq8_1->ds);
}
