// === TURBO4: 4-bit Lloyd-Max centroids — TWO vec_dot variants ===
// 1. vec_dot_turbo4_0_q8_1_iwht: applies iWHT to centroids, dots with unrotated q8_1 (~7 tok/s)
// 2. vec_dot_turbo4_0_q8_1_raw: NO iWHT, dots raw centroids with WHT-pre-rotated q8_1 (~30 tok/s target)
//
// Both use the SAME element mapping: centroid[i] dots with activation[i].
// Thread 0 (iqs=0): centroids 0-63, q8 blocks [0,1]
// Thread 1 (iqs=8): centroids 64-127, q8 blocks [2,3]

#define VDR_TURBO4_0_Q8_1_MMVQ 8
#define VDR_TURBO4_0_Q8_1_MMQ  4

// iWHT sign arrays (matching CPU ggml-turbo-quant.c) — only needed for iWHT vec_dot
static __constant__ float vd_iwht_s1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static __constant__ float vd_iwht_s2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// turbo4 centroids (float32, matching CPU CENTROIDS_4BIT)
static constexpr __device__ float t4c_f32[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

// === Helper: load 64 centroids from turbo4 block ===
// Sequential nibble packing: qs[0] = {idx[0] (lo), idx[1] (hi)}, qs[1] = {idx[2], idx[3]}, etc.
static __device__ __forceinline__ void load_turbo4_centroids(
    const block_turbo4_0 * bq, int elem_start, float * buf) {

    #pragma unroll
    for (int i = 0; i < 64; i += 8) {
        const int bo = elem_start / 2;
        const int b0 = bq->qs[bo + i/2 + 0];
        const int b1 = bq->qs[bo + i/2 + 1];
        const int b2 = bq->qs[bo + i/2 + 2];
        const int b3 = bq->qs[bo + i/2 + 3];
        buf[i+0] = t4c_f32[b0 & 0xF];
        buf[i+1] = t4c_f32[(b0>>4) & 0xF];
        buf[i+2] = t4c_f32[b1 & 0xF];
        buf[i+3] = t4c_f32[(b1>>4) & 0xF];
        buf[i+4] = t4c_f32[b2 & 0xF];
        buf[i+5] = t4c_f32[(b2>>4) & 0xF];
        buf[i+6] = t4c_f32[b3 & 0xF];
        buf[i+7] = t4c_f32[(b3>>4) & 0xF];
    }
}

// === Helper: dot 64 centroids with 2 q8_1 blocks (64 elements) ===
static __device__ __forceinline__ float dot_centroids_q8(
    const float * buf, const block_q8_1 * bq8_1, int q8_blk_off) {

    const float d0 = __low2float((bq8_1 + q8_blk_off + 0)->ds);
    const float d1 = __low2float((bq8_1 + q8_blk_off + 1)->ds);
    const int8_t * qs0 = (const int8_t *)(bq8_1 + q8_blk_off + 0)->qs;
    const int8_t * qs1 = (const int8_t *)(bq8_1 + q8_blk_off + 1)->qs;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += buf[i] * (float)qs0[i] * d0;
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        sum += buf[32 + i] * (float)qs1[i] * d1;
    }
    return sum;
}

// === VEC_DOT 1: iWHT on centroids, dot with unrotated q8_1 (~7 tok/s) ===
// This is the CORRECT baseline that matches CPU dequant exactly.
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1_iwht(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    const int elem_start = (iqs / 8) * 64;  // 0 or 64
    const bool is_first_half = (iqs == 0);

    // Load 64 centroids
    float buf[64];
    load_turbo4_centroids(bq, elem_start, buf);

    // Apply s2 signs
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        buf[i] *= vd_iwht_s2[elem_start + i];
    }

    // Butterfly passes 1-6 (h=1,2,4,8,16,32) — within 64 elements
    #pragma unroll
    for (int h = 1; h < 64; h *= 2) {
        for (int j = 0; j < 64; j += 2 * h) {
            for (int k = j; k < j + h; k++) {
                float a = buf[k], b = buf[k + h];
                buf[k]     = a + b;
                buf[k + h] = a - b;
            }
        }
    }

    // Butterfly pass 7 (h=64) — cross-thread exchange via warp shuffle
    const unsigned active_mask = __activemask();
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        float partner = __shfl_xor_sync(active_mask, buf[i], 1);
        if (is_first_half) {
            buf[i] = buf[i] + partner;
        } else {
            buf[i] = partner - buf[i];
        }
    }

    // Apply s1 signs * inv_sqrt_128
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        buf[i] *= inv_sqrt_128 * vd_iwht_s1[elem_start + i];
    }

    // Dot with unrotated q8_1 activations
    const int q8_blk_off = (iqs / 8) * 2;   // 0 or 2
    float sum = dot_centroids_q8(buf, bq8_1, q8_blk_off);

    return norm * sum;
}

// === VEC_DOT 2: NO iWHT, dot raw centroids with WHT-pre-rotated q8_1 (~30 tok/s target) ===
// For use with graph-side WHT forward rotation of activations.
// By Parseval's theorem: dot(iWHT(c), a) = dot(c, WHT_inv^T(a)) = dot(c, WHT_fwd(a))
// So we dot raw centroids (in WHT space) with WHT-rotated activations.
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1_raw(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    const int elem_start = (iqs / 8) * 64;  // 0 or 64

    // Load 64 centroids (NO iWHT applied)
    float buf[64];
    load_turbo4_centroids(bq, elem_start, buf);

    // Dot with WHT-pre-rotated q8_1 activations
    // Element mapping is IDENTICAL to iWHT version: centroid[i] dots with activation[i]
    const int q8_blk_off = (iqs / 8) * 2;   // 0 or 2
    float sum = dot_centroids_q8(buf, bq8_1, q8_blk_off);

    return norm * sum;
}

// === Default vec_dot with runtime mode switch ===
// Mode is set by mmvq.cu via cudaMemcpyToSymbolAsync before kernel launch:
//   0 = iWHT mode (for prefill or non-turbo: no WHT pre-rotation)
//   1 = raw mode (for turbo4 decode: WHT pre-rotation active)
__device__ int g_turbo4_raw_mode = 0;

static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    if (g_turbo4_raw_mode) {
        return vec_dot_turbo4_0_q8_1_raw(vbq, bq8_1, kbx, iqs);
    } else {
        return vec_dot_turbo4_0_q8_1_iwht(vbq, bq8_1, kbx, iqs);
    }
}
