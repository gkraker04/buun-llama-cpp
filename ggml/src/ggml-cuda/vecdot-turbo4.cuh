// === TURBO4: 4-bit Lloyd-Max centroids ===
// Two vec_dot variants:
// 1) vec_dot_turbo4_0_q8_1: on-the-fly iWHT (fallback, no pre-rotation)
// 2) vec_dot_turbo4_0_q8_1_rotated: pre-rotated activations (fast path)
//
// MMVQ thread layout: 2 threads per 128-element turbo4 block
// Thread 0 (iqs=0): centroids 0-63, q8 blocks [0,1]
// Thread 1 (iqs=8): centroids 64-127, q8 blocks [2,3]

#define VDR_TURBO4_0_Q8_1_MMVQ 8
#define VDR_TURBO4_0_Q8_1_MMQ  4

// iWHT sign arrays (matching CPU ggml-turbo-quant.c)
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

// turbo4 centroids as int8 (for dp4a)
// Scale factor: 127.0 / 0.241556 ≈ 525.8
static constexpr __device__ int8_t t4c_i8[16] = {
    -127, -96, -75, -58, -44, -31, -18, -6,
       6,  18,  31,  44,  58,  75,  96, 127
};
static constexpr float T4C_I8_SCALE = 1.0f / 525.8f;  // to recover original float values

// vec_dot for MMVQ — applies iWHT to centroids, dot with unrotated q8_1.
// Two threads cooperate per turbo4 block using warp shuffle:
//   Thread 0 (iqs=0): centroids 0-63, q8 blocks [0,1]
//   Thread 1 (iqs=8): centroids 64-127, q8 blocks [2,3]
//
// Pass 7 of the iWHT butterfly (h=64) crosses the 64-element boundary.
// Thread 0 computes: new = self + partner
// Thread 1 computes: new = partner - self
// Exchange via __shfl_xor_sync(active_mask, val, 1)
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    // Determine which half of the 128-element block this thread handles
    const int elem_start = (iqs / 8) * 64;  // 0 or 64
    const bool is_first_half = (iqs == 0);

    // === Step 1: load 64 centroids ===
    float buf[64];

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

    // === Step 2: apply s2 signs ===
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        buf[i] *= vd_iwht_s2[elem_start + i];
    }

    // === Step 3: butterfly passes 1-6 (h=1,2,4,8,16,32) ===
    // All within this thread's 64-element half
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

    // === Step 4: butterfly pass 7 (h=64) — cross-thread exchange ===
    // Thread 0: buf[i] += partner[i]  (new = A + B)
    // Thread 1: buf[i] = partner[i] - buf[i]  (new = A - B)
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

    // === Step 5: apply s1 signs * inv_sqrt_128 ===
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        buf[i] *= inv_sqrt_128 * vd_iwht_s1[elem_start + i];
    }

    // === Step 6: dot half with raw q8_1 activations ===
    const int q8_blk_off = (iqs / 8) * 2;   // 0 or 2
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

    return norm * sum;
}

// === Simplified vec_dot for pre-rotated activations ===
// Activations have already been WHT-rotated before q8_1 quantization.
// No iWHT butterfly needed — just load centroids and dot with q8_1.
// This is ~10x faster than vec_dot_turbo4_0_q8_1 (no butterfly, no shuffle).
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1_rotated(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    // Determine which half of the 128-element block this thread handles
    const int elem_start = (iqs / 8) * 64;  // 0 or 64

    // === Load 64 centroids from codebook ===
    float buf[64];

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

    // === Dot with pre-rotated q8_1 activations ===
    const int q8_blk_off = (iqs / 8) * 2;   // 0 or 2
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

    return norm * sum;
}

// === dp4a vec_dot for pre-rotated activations ===
// Uses int8 dot product (dp4a) for 4× throughput vs float multiply-add.
// Centroids are quantized to int8 at compile time (t4c_i8[]).
// Activations are already int8 in q8_1 blocks.
// dp4a computes: sum += Σ(centroid_i8[i] * activation_i8[i]) for 4 elements per call.
// Final result scaled by T4C_I8_SCALE * activation_scale * norm.
static __device__ __forceinline__ float vec_dot_turbo4_0_q8_1_dp4a(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_turbo4_0 * bq = (const block_turbo4_0 *) vbq + kbx;
    const float norm = __half2float(bq->norm);

    // Determine which half of the 128-element block this thread handles
    const int elem_start = (iqs / 8) * 64;  // 0 or 64

    // === Load 64 int8 centroids from codebook ===
    int8_t c_i8[64];
    #pragma unroll
    for (int i = 0; i < 64; i += 8) {
        const int bo = elem_start / 2;
        const int b0 = bq->qs[bo + i/2 + 0];
        const int b1 = bq->qs[bo + i/2 + 1];
        const int b2 = bq->qs[bo + i/2 + 2];
        const int b3 = bq->qs[bo + i/2 + 3];
        c_i8[i+0] = t4c_i8[b0 & 0xF];
        c_i8[i+1] = t4c_i8[(b0>>4) & 0xF];
        c_i8[i+2] = t4c_i8[b1 & 0xF];
        c_i8[i+3] = t4c_i8[(b1>>4) & 0xF];
        c_i8[i+4] = t4c_i8[b2 & 0xF];
        c_i8[i+5] = t4c_i8[(b2>>4) & 0xF];
        c_i8[i+6] = t4c_i8[b3 & 0xF];
        c_i8[i+7] = t4c_i8[(b3>>4) & 0xF];
    }

    // === dp4a dot product with q8_1 activations ===
    // Process 64 elements in 16 dp4a calls (4 elements per call)
    // Split into two q8_1 blocks (32 elements each)
    const int q8_blk_off = (iqs / 8) * 2;   // 0 or 2
    const float d0 = __low2float((bq8_1 + q8_blk_off + 0)->ds);
    const float d1 = __low2float((bq8_1 + q8_blk_off + 1)->ds);
    const int8_t * qs0 = (const int8_t *)(bq8_1 + q8_blk_off + 0)->qs;
    const int8_t * qs1 = (const int8_t *)(bq8_1 + q8_blk_off + 1)->qs;

    int sum_i32_0 = 0;  // accumulator for first q8_1 block (elements 0-31)
    int sum_i32_1 = 0;  // accumulator for second q8_1 block (elements 32-63)

    // Pack centroids and activations into int32, then dp4a
    #pragma unroll
    for (int i = 0; i < 32; i += 4) {
        // First q8_1 block (elements 0-31)
        int c_packed = (int)((uint8_t)c_i8[i+0] | ((uint8_t)c_i8[i+1] << 8) |
                             ((uint8_t)c_i8[i+2] << 16) | ((uint8_t)c_i8[i+3] << 24));
        int a_packed = (int)((uint8_t)qs0[i+0] | ((uint8_t)qs0[i+1] << 8) |
                             ((uint8_t)qs0[i+2] << 16) | ((uint8_t)qs0[i+3] << 24));
        sum_i32_0 = ggml_cuda_dp4a(c_packed, a_packed, sum_i32_0);

        // Second q8_1 block (elements 32-63)
        c_packed = (int)((uint8_t)c_i8[32+i+0] | ((uint8_t)c_i8[32+i+1] << 8) |
                         ((uint8_t)c_i8[32+i+2] << 16) | ((uint8_t)c_i8[32+i+3] << 24));
        a_packed = (int)((uint8_t)qs1[i+0] | ((uint8_t)qs1[i+1] << 8) |
                         ((uint8_t)qs1[i+2] << 16) | ((uint8_t)qs1[i+3] << 24));
        sum_i32_1 = ggml_cuda_dp4a(c_packed, a_packed, sum_i32_1);
    }

    // Convert to float and apply scales
    float sum = ((float)sum_i32_0 * d0 + (float)sum_i32_1 * d1) * T4C_I8_SCALE;

    return norm * sum;
}
