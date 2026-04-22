#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml-common.h"

// === InnerQ per-channel equalization ===
// Scale K channels before L2 norm + FWHT to reduce quantization error on anisotropic distributions.
// Inverse scale applied to Q in FA kernel to preserve dot products.
// Calibration: accumulate per-channel K^2, then set scale[i] = 1/sqrt(mean(K_i^2) * 128).
static __device__ float d_innerq_channel_scale[128];     // per-channel K scale (init to 1.0)
static __device__ float d_innerq_channel_scale_inv[128]; // per-channel Q inverse scale (init to 1.0)
static __device__ float d_innerq_channel_sq[128];        // calibration accumulator: sum of K_i^2
static __device__ float d_innerq_channel_max[128];       // calibration accumulator: max of |K_i| (for paper's formula)
static __device__ int   d_innerq_count;                  // calibration token count
static __device__ int   d_innerq_calibrate;              // 1 = accumulate stats, 0 = apply scales
static __device__ int   d_innerq_is_k;                   // 1 = current set_rows is K cache, 0 = V cache

// Forward declaration: fattn compilation unit has its own copy of inverse scales
extern void turbo_innerq_update_fattn_scales(const float * scale_inv);
extern void turbo_innerq_init_fattn();

// === Post-FWHT data extraction for empirical codebook computation ===
// Enabled by TURBO_EXTRACT=<max_samples> env var (e.g. TURBO_EXTRACT=2000000)
// Dumps post-rotation normalized values to /tmp/turbo_postrot.bin (float32)
// Device-visible extraction state
static __device__ float * d_extract_buf_ptr = nullptr;
static __device__ int   * d_extract_pos_ptr = nullptr;
static __device__ int     d_extract_max_val = 0;

// Host-side management
static float * h_extract_gpu_buf = nullptr;
static int   * h_extract_gpu_pos = nullptr;
static int     h_extract_max = 0;
static int     h_extract_state = 0;  // 0=uninit, 1=collecting, 2=done

static void turbo_extract_init(int max_samples) {
	cudaMalloc(&h_extract_gpu_buf, (size_t)max_samples * sizeof(float));
	cudaMalloc(&h_extract_gpu_pos, sizeof(int));
	int zero = 0;
	cudaMemcpy(h_extract_gpu_pos, &zero, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_extract_buf_ptr, &h_extract_gpu_buf, sizeof(float *));
	cudaMemcpyToSymbol(d_extract_pos_ptr, &h_extract_gpu_pos, sizeof(int *));
	cudaMemcpyToSymbol(d_extract_max_val, &max_samples, sizeof(int));
	h_extract_max = max_samples;
	h_extract_state = 1;
	fprintf(stderr, "TURBO_EXTRACT: collecting up to %d post-rotation samples\n", max_samples);
}

static void turbo_extract_check_done() {
	if (h_extract_state != 1) return;
	int pos;
	cudaMemcpy(&pos, h_extract_gpu_pos, sizeof(int), cudaMemcpyDeviceToHost);
	if (pos < h_extract_max) return;
	// Buffer full — dump to disk
	if (pos > h_extract_max) pos = h_extract_max;
	float * host_buf = (float *)malloc((size_t)pos * sizeof(float));
	cudaMemcpy(host_buf, h_extract_gpu_buf, (size_t)pos * sizeof(float), cudaMemcpyDeviceToHost);
	const char * path = "/tmp/turbo_postrot.bin";
	FILE * fp = fopen(path, "wb");
	if (fp) {
		fwrite(host_buf, sizeof(float), pos, fp);
		fclose(fp);
		fprintf(stderr, "TURBO_EXTRACT: wrote %d samples to %s (%.1f MB)\n",
				pos, path, (float)pos * sizeof(float) / (1024*1024));
	}
	free(host_buf);
	// Disable extraction (set device pointers to null)
	float * null_ptr = nullptr;
	int   * null_iptr = nullptr;
	int     zero_max = 0;
	cudaMemcpyToSymbol(d_extract_buf_ptr, &null_ptr, sizeof(float *));
	cudaMemcpyToSymbol(d_extract_pos_ptr, &null_iptr, sizeof(int *));
	cudaMemcpyToSymbol(d_extract_max_val, &zero_max, sizeof(int));
	cudaFree(h_extract_gpu_buf); h_extract_gpu_buf = nullptr;
	cudaFree(h_extract_gpu_pos); h_extract_gpu_pos = nullptr;
	h_extract_state = 2;
}

// Device-side: append 128 post-rotation values to extraction buffer
static __device__ void turbo_extract_append(const float * x) {
	if (!d_extract_buf_ptr || !d_extract_pos_ptr) return;
	int base = atomicAdd(d_extract_pos_ptr, 128);
	if (base + 128 <= d_extract_max_val) {
		for (int j = 0; j < 128; j++) d_extract_buf_ptr[base + j] = x[j];
	}
}

// Host-side init: set identity scales, zero accumulators
static void turbo_innerq_init() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    float zeros[128] = {};
    int zero = 0;
    cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
    cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_is_k, &zero, sizeof(zero));
    turbo_innerq_init_fattn();
}

// Host-side: set K/V flag before kernel launch (called from set-rows.cu)
static void turbo_innerq_set_is_k(int is_k) {
    cudaMemcpyToSymbol(d_innerq_is_k, &is_k, sizeof(int));
}

// Host-side: enable calibration mode
static void turbo_innerq_start_calibration() {
    float zeros[128] = {};
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_calibrate, &one, sizeof(one));
}

// Host-side: finalize calibration — compute scales from accumulated stats
static void turbo_innerq_finalize_calibration() {
    int zero = 0;
    cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));

    float sq[128], ch_max[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_innerq_channel_sq, sizeof(sq));
    cudaMemcpyFromSymbol(ch_max, d_innerq_channel_max, sizeof(ch_max));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(count));

    if (count == 0) return;

    // Mode: 0=RMS-based (default), 1=max-based (paper's formula: sqrt(max|K_i|))
    static const char * mode_env = getenv("TURBO_INNERQ_MODE");
    int mode = mode_env ? atoi(mode_env) : 0;

    static const char * strength_env = getenv("TURBO_INNERQ_STRENGTH");
    float strength = strength_env ? atof(strength_env) : 0.5f;
    float max_clamp = 2.0f;

    float scale[128], scale_inv[128];
    float max_ratio = 1.0f;

    if (mode == 1) {
        // Paper's formula: scale[i] = 1/sqrt(max(|K_{:,i}|))
        // This normalizes each channel so its max value becomes sqrt(max_val)
        fprintf(stderr, "InnerQ mode=1 (paper's max-based formula)\n");
        for (int i = 0; i < 128; i++) {
            if (ch_max[i] > 1e-10f) {
                float s = 1.0f / sqrtf(ch_max[i]);
                // Normalize so mean scale = 1 (preserve overall magnitude)
                scale[i] = s;
            } else {
                scale[i] = 1.0f;
            }
        }
        // Normalize scales to have geometric mean ≈ 1
        float log_sum = 0.0f;
        for (int i = 0; i < 128; i++) log_sum += logf(scale[i]);
        float geo_mean = expf(log_sum / 128.0f);
        for (int i = 0; i < 128; i++) {
            scale[i] /= geo_mean;
            if (scale[i] > max_clamp) scale[i] = max_clamp;
            if (scale[i] < 1.0f / max_clamp) scale[i] = 1.0f / max_clamp;
            scale_inv[i] = 1.0f / scale[i];
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    } else {
        // RMS-based: scale = (mean_rms/channel_rms)^strength
        float total_rms = 0.0f;
        float channel_rms[128];
        for (int i = 0; i < 128; i++) {
            channel_rms[i] = sqrtf(sq[i] / count);
            total_rms += channel_rms[i];
        }
        float mean_rms = total_rms / 128.0f;

        for (int i = 0; i < 128; i++) {
            if (channel_rms[i] > 1e-10f) {
                float raw = mean_rms / channel_rms[i];
                float s = powf(raw, strength);
                if (s > max_clamp) s = max_clamp;
                if (s < 1.0f / max_clamp) s = 1.0f / max_clamp;
                scale[i] = s;
                scale_inv[i] = 1.0f / s;
            } else {
                scale[i] = 1.0f;
                scale_inv[i] = 1.0f;
            }
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    }

    fprintf(stderr, "InnerQ calibration: %d tokens, mode=%d, strength=%.2f, max scale ratio: %.3f (clamped to %.1f)\n",
            count, mode, strength, max_ratio, max_clamp);

    // Auto-detect: if channels are already well-balanced, InnerQ won't help — skip
    if (max_ratio < 1.2f) {
        fprintf(stderr, "InnerQ: max ratio %.3f < 1.2 — channels already balanced, disabling (would hurt quality)\n", max_ratio);
        float ones[128];
        for (int i = 0; i < 128; i++) ones[i] = 1.0f;
        cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
        turbo_innerq_update_fattn_scales(ones);
        return;
    }

    // Print top-5 most affected channels
    float scale_copy[128];
    for (int i = 0; i < 128; i++) scale_copy[i] = scale[i];
    for (int k = 0; k < 5; k++) {
        float best = 0; int best_i = -1;
        for (int i = 0; i < 128; i++) {
            float r = fabsf(logf(scale_copy[i]));
            if (r > best) { best = r; best_i = i; }
        }
        if (best_i >= 0) {
            fprintf(stderr, "  channel %d: scale=%.4f (max=%.6f, rms=%.6f)\n",
                    best_i, scale[best_i], ch_max[best_i], sqrtf(sq[best_i] / count));
            scale_copy[best_i] = 1.0f; // mark as printed
        }
    }

    cudaMemcpyToSymbol(d_innerq_channel_scale, scale, sizeof(scale));
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv, scale_inv, sizeof(scale_inv));
    turbo_innerq_update_fattn_scales(scale_inv);
}

// === Shared constants ===
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// === TURBO2: 2-bit codebook (Lloyd-Max for N(0, 1/128)) ===
static __constant__ float d_turbo_centroids_2bit[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_mid_2bit[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// === TURBO4: 4-bit codebook (Lloyd-Max for N(0, 1/sqrt(128))) ===
static __constant__ float d_turbo_centroids_4bit[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static __constant__ float d_turbo_mid_4bit[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

// === FWHT rotation sign arrays (from turbo-wht.h, seed=42 rotation, seed=1042 QJL) ===
static __constant__ float d_turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
static __constant__ float d_turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};
// QJL sign arrays removed — turbo4 now uses pure 4-bit PolarQuant (no QJL correction)

// === FWHT rotation functions ===
static __device__ __forceinline__
void turbo_fwht_128_cuda(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

// Forward rotation: signs1 → FWHT → signs2
static __device__ __forceinline__
void turbo_rotate_forward_cuda(float * x, const float * s1, const float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128_cuda(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

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

static __device__ __forceinline__
uint8_t turbo_find_nearest_4bit(float val) {
    // Binary search over 15 midpoints for 16 centroids
    if (val < d_turbo_mid_4bit[7]) {
        if (val < d_turbo_mid_4bit[3]) {
            if (val < d_turbo_mid_4bit[1]) {
                return val < d_turbo_mid_4bit[0] ? 0 : 1;
            } else {
                return val < d_turbo_mid_4bit[2] ? 2 : 3;
            }
        } else {
            if (val < d_turbo_mid_4bit[5]) {
                return val < d_turbo_mid_4bit[4] ? 4 : 5;
            } else {
                return val < d_turbo_mid_4bit[6] ? 6 : 7;
            }
        }
    } else {
        if (val < d_turbo_mid_4bit[11]) {
            if (val < d_turbo_mid_4bit[9]) {
                return val < d_turbo_mid_4bit[8] ? 8 : 9;
            } else {
                return val < d_turbo_mid_4bit[10] ? 10 : 11;
            }
        } else {
            if (val < d_turbo_mid_4bit[13]) {
                return val < d_turbo_mid_4bit[12] ? 12 : 13;
            } else {
                return val < d_turbo_mid_4bit[14] ? 14 : 15;
            }
        }
    }
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
        const int innerq_is_k,
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
    // InnerQ: calibrate from both K and V, apply scaling to both
    if (d_innerq_calibrate) {
        for (int j = 0; j < 128; j++) {
            atomicAdd(&d_innerq_channel_sq[j], x[j] * x[j]);
            float abs_val = fabsf(x[j]);
            // atomicMax for float: CAS loop (no native float atomicMax)
            unsigned int * addr = (unsigned int *)&d_innerq_channel_max[j];
            unsigned int old_val = __float_as_uint(abs_val);
            unsigned int assumed;
            do {
                assumed = *addr;
                if (__uint_as_float(assumed) >= abs_val) break;
            } while (atomicCAS(addr, assumed, old_val) != assumed);
        }
        atomicAdd(&d_innerq_count, 1);
    }
    for (int j = 0; j < 128; j++) x[j] *= d_innerq_channel_scale[j];
    norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // Quantize and accumulate reconstruction norm for correction
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
            float c = d_turbo_centroids_3bit[idx];
            recon_norm_sq += c * c;
        }
    }
    // Norm correction: store corrected norm so dequant(x) has exact original L2 norm
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
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

// === TURBO4: SET_ROWS quantize (4-bit PolarQuant, no QJL) ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    // Forward FWHT rotation before quantization
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // 4-bit quantization: find nearest of 16 centroids, pack 2 per byte
    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = turbo_find_nearest_4bit(x[j]);
        uint8_t idx1 = turbo_find_nearest_4bit(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
    }
    // Norm correction: compute reconstruction norm in rotated space
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (j & 1) ? (dst->qs[j / 2] >> 4) : (dst->qs[j / 2] & 0xF);
        float r = d_turbo_centroids_4bit[idx];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    dst->norm = __float2half((recon_norm > 1e-10f) ? norm / recon_norm : norm);
}

// === TURBO4: GET_ROWS dequantize ===
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.x = d_turbo_centroids_4bit[idx] * norm; }
    { const int j = iqs + 64;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.y = d_turbo_centroids_4bit[idx] * norm; }
}

// === TURBO2: find nearest 2-bit centroid ===
static __device__ __forceinline__
uint8_t turbo_find_nearest_2bit(float val) {
    if      (val < d_turbo_mid_2bit[0]) return 0;
    else if (val < d_turbo_mid_2bit[1]) return 1;
    else if (val < d_turbo_mid_2bit[2]) return 2;
    else                                return 3;
}

// === TURBO2: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo2(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO2_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO2_GROUP;
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo2_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO2;
        for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = turbo_find_nearest_2bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            float c = d_turbo_centroids_2bit[idx];
            recon_norm_sq += c * c;
        }
    }
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
    }
}

// === TURBO2: GET_ROWS dequantize ===
#define QR_TURBO2_0 2
static __device__ __forceinline__
void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.x = d_turbo_centroids_2bit[idx] * norm; }
    { const int j = iqs + 16;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.y = d_turbo_centroids_2bit[idx] * norm; }
}

// === TURBO3_TCQ: Trellis-Coded Quantization (right-shift bitshift trellis, k=3, L=9) ===
// GLA-trained free-init TCQ codebook (512 entries) for N(0, 1/sqrt(128)) post-FWHT data
// MSE reduction: 37.6% vs Lloyd-Max 3-bit, +2.05 dB. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook[512] = {
    -0.19075318f, -0.12398477f, -0.08053825f, -0.04337945f, -0.02360115f, +0.01870265f, +0.07576828f, +0.15711791f,
    -0.17111190f, -0.12162214f, -0.08470646f, -0.04852028f, -0.01371993f, +0.02535509f, +0.08013468f, +0.14563999f,
    -0.23385642f, -0.13636887f, -0.07996625f, -0.04284568f, -0.01378520f, +0.02527046f, +0.08126875f, +0.19733478f,
    -0.17217710f, -0.12501276f, -0.08301722f, -0.04618388f, -0.01582557f, +0.01849815f, +0.05651660f, +0.11781682f,
    -0.26939890f, -0.11554235f, -0.07074665f, -0.03676226f, -0.01378042f, +0.02288926f, +0.07751006f, +0.21598307f,
    -0.16721224f, -0.12556323f, -0.08082666f, -0.04102167f, -0.01442464f, +0.02706698f, +0.06868703f, +0.12768870f,
    -0.17612142f, -0.12177497f, -0.07355501f, -0.04208433f, -0.01214733f, +0.02949718f, +0.07909346f, +0.15018134f,
    -0.23495452f, -0.12467323f, -0.07873887f, -0.04478245f, -0.01067369f, +0.02844658f, +0.07484870f, +0.14291016f,
    -0.20845117f, -0.12025491f, -0.07898818f, -0.03999034f, -0.00396196f, +0.03149235f, +0.07821322f, +0.14260191f,
    -0.18444445f, -0.11889985f, -0.07379119f, -0.03679606f, -0.00808100f, +0.02833046f, +0.07491008f, +0.13134058f,
    -0.19901366f, -0.12241073f, -0.07129523f, -0.03430970f, -0.00634336f, +0.03164584f, +0.06921050f, +0.12507342f,
    -0.22138300f, -0.11838018f, -0.07095155f, -0.03446699f, -0.00752457f, +0.02620806f, +0.07400409f, +0.15958642f,
    -0.16634685f, -0.10892222f, -0.06854335f, -0.02767931f, -0.00510447f, +0.03830038f, +0.09252869f, +0.13887878f,
    -0.21289924f, -0.11350111f, -0.06690028f, -0.03032817f, -0.00054839f, +0.03241062f, +0.07777942f, +0.14089005f,
    -0.16115880f, -0.11725200f, -0.07240758f, -0.03489496f, -0.00463092f, +0.03327753f, +0.07979671f, +0.13508332f,
    -0.18059183f, -0.11007259f, -0.06711663f, -0.02841142f, +0.00008600f, +0.03609043f, +0.08622773f, +0.18401953f,
    -0.15190504f, -0.10264046f, -0.06591309f, -0.03053302f, +0.00219368f, +0.03783871f, +0.08697283f, +0.17363742f,
    -0.16044058f, -0.10606719f, -0.06668835f, -0.02990519f, +0.00298238f, +0.04131254f, +0.09152508f, +0.16726999f,
    -0.16298678f, -0.10606801f, -0.06302952f, -0.02649282f, +0.00338007f, +0.03691096f, +0.08051851f, +0.19143041f,
    -0.15842708f, -0.10271062f, -0.06741970f, -0.02783111f, +0.00129675f, +0.04058053f, +0.08952771f, +0.12665890f,
    -0.14287122f, -0.10702290f, -0.06360254f, -0.02298262f, +0.00504083f, +0.03929205f, +0.07607899f, +0.17748189f,
    -0.15732529f, -0.10472551f, -0.06157213f, -0.02291222f, +0.00406915f, +0.04300021f, +0.09802638f, +0.19737541f,
    -0.16368793f, -0.10786568f, -0.06302504f, -0.02213908f, +0.00705703f, +0.04387142f, +0.09279074f, +0.17373691f,
    -0.15563499f, -0.09970366f, -0.05740117f, -0.02069011f, +0.00532867f, +0.04516702f, +0.09245405f, +0.15705084f,
    -0.22633528f, -0.11082206f, -0.06271142f, -0.02594333f, +0.00196982f, +0.03854224f, +0.07979941f, +0.13428254f,
    -0.20595677f, -0.10630489f, -0.06029190f, -0.02214403f, -0.00260620f, +0.03775614f, +0.07463138f, +0.13103214f,
    -0.25072671f, -0.10346837f, -0.06094402f, -0.02491104f, +0.00614344f, +0.04080280f, +0.08221361f, +0.13847503f,
    -0.20928229f, -0.10634761f, -0.05699658f, -0.02148475f, -0.00035151f, +0.03748212f, +0.07271124f, +0.12825825f,
    -0.18312579f, -0.09889935f, -0.06073723f, -0.02458788f, +0.00436764f, +0.04666018f, +0.09222218f, +0.14264482f,
    -0.25463980f, -0.10378968f, -0.05824099f, -0.02155519f, +0.00609332f, +0.04016074f, +0.08052604f, +0.13524376f,
    -0.20022215f, -0.09820325f, -0.05344592f, -0.02058924f, +0.00430976f, +0.04488201f, +0.08667631f, +0.14100030f,
    -0.23726417f, -0.10697613f, -0.05615639f, -0.01963419f, +0.00929481f, +0.04763221f, +0.08734125f, +0.14092055f,
    -0.13854847f, -0.08281066f, -0.04378172f, -0.00652702f, +0.02368154f, +0.05515453f, +0.10098024f, +0.21544034f,
    -0.13675106f, -0.08835772f, -0.04778416f, -0.01087520f, +0.01662638f, +0.05679985f, +0.09930499f, +0.25459621f,
    -0.13744516f, -0.07804402f, -0.04053756f, -0.00156069f, +0.01937795f, +0.05717912f, +0.10366104f, +0.19898203f,
    -0.12785788f, -0.08260384f, -0.04168846f, -0.00836940f, +0.02032687f, +0.05140464f, +0.09839836f, +0.17357632f,
    -0.14337727f, -0.07776439f, -0.04075604f, -0.00035689f, +0.02425877f, +0.06102493f, +0.10354523f, +0.26100360f,
    -0.13787537f, -0.08036437f, -0.03951768f, -0.00204148f, +0.02145062f, +0.05740400f, +0.10506784f, +0.19793756f,
    -0.12882150f, -0.07994786f, -0.04003095f, -0.00191794f, +0.02359812f, +0.06184931f, +0.10233122f, +0.23810753f,
    -0.14044366f, -0.07837795f, -0.04160599f, -0.00048596f, +0.02446058f, +0.05855361f, +0.10956655f, +0.22929512f,
    -0.17846599f, -0.09742940f, -0.04639398f, -0.01092025f, +0.02348794f, +0.05447743f, +0.09550074f, +0.15359668f,
    -0.17422996f, -0.08763111f, -0.04266620f, -0.00590155f, +0.02432001f, +0.06166173f, +0.10203922f, +0.15632069f,
    -0.16551951f, -0.09271351f, -0.04697642f, -0.00990860f, +0.02472535f, +0.06128802f, +0.10103604f, +0.14517386f,
    -0.17118861f, -0.08584806f, -0.03829585f, +0.00053346f, +0.02704928f, +0.06109060f, +0.09696287f, +0.15332595f,
    -0.12697297f, -0.08251215f, -0.04329925f, -0.00899454f, +0.02452956f, +0.06064569f, +0.11392346f, +0.18405104f,
    -0.19098167f, -0.09401987f, -0.03961263f, -0.00091159f, +0.02620175f, +0.06351430f, +0.10044691f, +0.14884785f,
    -0.15357839f, -0.08420967f, -0.03983079f, -0.00441110f, +0.02716057f, +0.06522659f, +0.11198404f, +0.16775683f,
    -0.19805412f, -0.09481380f, -0.04197457f, -0.00466698f, +0.02339645f, +0.06436768f, +0.11203527f, +0.16789078f,
    -0.13746277f, -0.08557623f, -0.03912223f, -0.00399355f, +0.03151713f, +0.06573500f, +0.11236197f, +0.18292049f,
    -0.14053986f, -0.08499924f, -0.03501216f, -0.00172963f, +0.02630023f, +0.06582417f, +0.11766521f, +0.19003936f,
    -0.13166662f, -0.07917286f, -0.03360028f, +0.00095822f, +0.02770623f, +0.07172356f, +0.11358009f, +0.18991790f,
    -0.23290175f, -0.08433987f, -0.03867760f, +0.00061902f, +0.03305846f, +0.06233019f, +0.10861871f, +0.15443935f,
    -0.12210833f, -0.06640679f, -0.02985525f, +0.00214670f, +0.02966577f, +0.07318296f, +0.11824244f, +0.21638604f,
    -0.15819124f, -0.08219178f, -0.03493502f, +0.00624893f, +0.03856357f, +0.07096187f, +0.11145671f, +0.15940793f,
    -0.12626326f, -0.07091254f, -0.02856854f, +0.00733897f, +0.03200106f, +0.07230481f, +0.12070683f, +0.21324470f,
    -0.13749853f, -0.07346727f, -0.03025852f, +0.00530487f, +0.03579740f, +0.07030963f, +0.11728036f, +0.17899297f,
    -0.18793107f, -0.07859394f, -0.03031515f, +0.01418602f, +0.04532805f, +0.07363716f, +0.12567619f, +0.19763788f,
    -0.12486269f, -0.07178514f, -0.02911957f, +0.00866743f, +0.03677420f, +0.07358893f, +0.11658713f, +0.16348342f,
    -0.18465906f, -0.08903159f, -0.03331701f, +0.00903627f, +0.04149811f, +0.07646608f, +0.12565799f, +0.22711519f,
    -0.16195340f, -0.07480428f, -0.01911557f, +0.01691384f, +0.03921197f, +0.07628624f, +0.11136164f, +0.16702954f,
    -0.12647923f, -0.07496141f, -0.03331255f, +0.01061243f, +0.04254632f, +0.07620428f, +0.12315008f, +0.25389046f,
    -0.12756266f, -0.07329518f, -0.02324664f, +0.01344221f, +0.04260113f, +0.08009208f, +0.12919118f, +0.18493628f,
    -0.19126476f, -0.07707876f, -0.02340527f, +0.01554000f, +0.04223934f, +0.08060503f, +0.11884624f, +0.16863864f,
    -0.13215958f, -0.06856741f, -0.01997532f, +0.01749025f, +0.04587398f, +0.08523111f, +0.14069217f, +0.23933266f
};

// TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis
// 512 threads per block (one per trellis state), one block per 128-element group
// Backtrace stored in shared memory (32KB, 4-bit packed)
template<typename idx_t>
static __global__ void __launch_bounds__(512, 1) k_set_rows_turbo3_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_tcq * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int64_t group = blockIdx.x;
    if (group >= ne_total_groups) return;

    const int sid = threadIdx.x; // state index 0..511

    // Compute source and destination pointers (same index math as turbo3)
    const int64_t i_base = group * QK_TURBO3_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_tcq * dst_blk = (block_turbo3_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                                  + (i00 / QK_TURBO3_TCQ);

    // Shared memory layout:
    // x[128]     : rotated+normalized input
    // cost[512]  : current path costs
    // bt[128][256]: backtrace, 4-bit packed (best predecessor index 0-7)
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ uint8_t bt[128][256]; // 32KB: bt[t][s/2] = (pred_s_even) | (pred_s_odd << 4)

    // Thread 0: read source, compute norm, apply InnerQ, normalize, FWHT
    // Compute directly in shared x[] to avoid register-heavy local array
    if (sid == 0) {
        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }

        // InnerQ scaling
        if (d_innerq_calibrate) {
            for (int j = 0; j < 128; j++) {
                atomicAdd(&d_innerq_channel_sq[j], x[j] * x[j]);
                float abs_val = fabsf(x[j]);
                unsigned int * addr = (unsigned int *)&d_innerq_channel_max[j];
                unsigned int old_val = __float_as_uint(abs_val);
                unsigned int assumed;
                do {
                    assumed = *addr;
                    if (__uint_as_float(assumed) >= abs_val) break;
                } while (atomicCAS(addr, assumed, old_val) != assumed);
            }
            atomicAdd(&d_innerq_count, 1);
        }
        for (int j = 0; j < 128; j++) x[j] *= d_innerq_channel_scale[j];

        // Recompute norm after InnerQ
        norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < 128; j++) x[j] *= inv_norm;

        // FWHT rotation (operates on shared memory x[] directly)
        turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);

        // Post-rotation extraction (if enabled)
        turbo_extract_append(x);

        // Store norm (reuse cost[0] temporarily)
        cost[0] = grp_norm;
    }
    __syncthreads();

    float saved_norm = cost[0];

    // Initialize Viterbi: free initial state (all states equally viable)
    cost[sid] = 0.0f;
    __syncthreads();

    // Forward pass: 128 time steps, fully parallel across 512 states
    for (int t = 0; t < 128; t++) {
        float xt = x[t];

        // For state sid: find best predecessor
        // Right-shift trellis: ns = (prev >> 3) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 3) | p, for p = 0..7
        int base_prev = (sid & 0x3F) << 3;
        float dist = xt - d_turbo3_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 8; p++) {
            float c = cost[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        __syncthreads();
        cost[sid] = best + dist;

        // Store backtrace: 4-bit packed, 2 entries per byte
        if (sid % 2 == 0) {
            bt[t][sid / 2] = (uint8_t)best_p;
        }
        __syncthreads();
        if (sid % 2 == 1) {
            bt[t][sid / 2] |= ((uint8_t)best_p) << 4;
        }
        __syncthreads();
    }

    // Thread 0: find best final state, backtrack, pack bitstream
    if (sid == 0) {
        // Find best final state
        float min_cost = cost[0];
        int min_state = 0;
        for (int s = 1; s < 512; s++) {
            if (cost[s] < min_cost) {
                min_cost = cost[s];
                min_state = s;
            }
        }

        // Backtrack: recover outputs (reuse x[] shared memory as byte array)
        uint8_t * outputs = (uint8_t *)x; // x[] no longer needed after forward pass
        int state = min_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6); // output = top 3 bits (right-shift trellis)
            int p = (bt[t][state / 2] >> ((state % 2) * 4)) & 0xF;
            state = ((state & 0x3F) << 3) | p; // reconstruct predecessor
        }

        // After backtrack, 'state' is the initial state chosen by Viterbi
        const int initial_state = state;

        // Compute reconstruction norm by replaying trellis from initial state
        float recon_norm_sq = 0.0f;
        int cur_state = initial_state;
        for (int t = 0; t < 128; t++) {
            cur_state = (cur_state >> 3) | (outputs[t] << 6);
            float c = d_turbo3_tcq_codebook[cur_state];
            recon_norm_sq += c * c;
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;

        // Pack bitstream: [6 prefix bits] [out_0 (3 bits)] ... [out_127 (3 bits)]
        for (int j = 0; j < 49; j++) dst_blk->qs[j] = 0;

        // Write initial state prefix (upper 6 bits = initial_state >> 3)
        dst_blk->qs[0] = (uint8_t)((initial_state >> 3) & 0x3F);

        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 3;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x7;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
            if (bit_off > 5) { // 3 bits cross byte boundary
                dst_blk->qs[byte_idx + 1] |= (uint8_t)(out >> (8 - bit_off));
            }
        }

        dst_blk->norm = __float2half(corrected_norm);
    }
}

// TCQ GET_ROWS dequantize (for non-FA paths)
#define QR_TURBO3_TCQ 2
static __device__ __forceinline__
void dequantize_turbo3_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs
    {
        const int t = iqs;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.x = d_turbo3_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.y = d_turbo3_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBO2_TCQ: 2-bit Trellis-Coded Quantization (k=2, L=8, 256 states, free initial state)
// =====================================================================================

// GLA-trained free-init 2-bit TCQ codebook (256 entries) for N(0, 1/sqrt(128)) post-FWHT data
// MSE reduction: 4.2% vs Lloyd-Max 2-bit, +0.18 dB. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turbo2_tcq_codebook[256] = {
    -0.17377298f, -0.08762707f, -0.01300744f, +0.10467077f, -0.15621400f, -0.07807468f, -0.00244975f, +0.10971872f,
    -0.17391683f, -0.07507965f, -0.00158784f, +0.08619211f, -0.22927522f, -0.08869821f, -0.01062877f, +0.08292897f,
    -0.14736083f, -0.07461455f, -0.00156869f, +0.08953861f, -0.20510331f, -0.07670021f, +0.00562693f, +0.09755767f,
    -0.17821995f, -0.07306755f, +0.01162439f, +0.12019700f, -0.14980938f, -0.06716545f, +0.01804089f, +0.11784229f,
    -0.17945849f, -0.06972521f, +0.00976605f, +0.11559892f, -0.16441021f, -0.06922967f, +0.00837952f, +0.09737813f,
    -0.15496514f, -0.06655134f, +0.01073252f, +0.09873007f, -0.16154034f, -0.06512384f, +0.01120347f, +0.09844273f,
    -0.16629047f, -0.07160361f, +0.01689301f, +0.10389574f, -0.15270690f, -0.06608909f, +0.01531757f, +0.10876989f,
    -0.15495242f, -0.06025202f, +0.02097986f, +0.12120320f, -0.21677839f, -0.06544403f, +0.01845107f, +0.12382485f,
    -0.16529795f, -0.06390794f, +0.01756180f, +0.10582994f, -0.17867196f, -0.06164099f, +0.02126243f, +0.11631798f,
    -0.14439308f, -0.06022475f, +0.01772231f, +0.11524636f, -0.16398476f, -0.05841067f, +0.02710701f, +0.12722188f,
    -0.14742752f, -0.05213630f, +0.02244631f, +0.10951075f, -0.14269118f, -0.05402560f, +0.02561049f, +0.11615862f,
    -0.14039113f, -0.05273549f, +0.02707237f, +0.13126772f, -0.15737704f, -0.05754378f, +0.02594541f, +0.10646760f,
    -0.14971745f, -0.05049292f, +0.03509529f, +0.13929558f, -0.14467933f, -0.05133092f, +0.03106021f, +0.12962434f,
    -0.16401061f, -0.05091477f, +0.02959540f, +0.11717260f, -0.14241236f, -0.04143231f, +0.04110209f, +0.15503085f,
    -0.14888643f, -0.04547486f, +0.03337607f, +0.12928898f, -0.13315155f, -0.04334711f, +0.03357259f, +0.12295390f,
    -0.13933571f, -0.04168339f, +0.04251146f, +0.14801516f, -0.12695345f, -0.04017735f, +0.03470594f, +0.12149578f,
    -0.13630760f, -0.03725725f, +0.04573099f, +0.14982770f, -0.13279556f, -0.03731158f, +0.03788514f, +0.14134987f,
    -0.14634417f, -0.03906009f, +0.04341434f, +0.13156858f, -0.11998180f, -0.03818642f, +0.04197899f, +0.12642762f,
    -0.15277894f, -0.03935205f, +0.04568923f, +0.16831640f, -0.11562648f, -0.03303958f, +0.04737825f, +0.12890437f,
    -0.13040864f, -0.03364901f, +0.04606153f, +0.14526574f, -0.13061834f, -0.03017139f, +0.05168760f, +0.14875662f,
    -0.12403387f, -0.03103612f, +0.04867485f, +0.12266303f, -0.10907682f, -0.02440896f, +0.05311224f, +0.15778596f,
    -0.11341729f, -0.02520524f, +0.05340497f, +0.15747784f, -0.11050928f, -0.02731021f, +0.05552406f, +0.13477354f,
    -0.11251016f, -0.02502996f, +0.05742991f, +0.15073479f, -0.12924648f, -0.02710250f, +0.05662459f, +0.16618961f,
    -0.12142910f, -0.02062330f, +0.06006443f, +0.14212358f, -0.12225247f, -0.01665350f, +0.05721657f, +0.16113346f,
    -0.10689972f, -0.01877897f, +0.06295932f, +0.15178648f, -0.11211861f, -0.01892951f, +0.06142450f, +0.16882628f,
    -0.09920592f, -0.01426363f, +0.06212827f, +0.15953216f, -0.14424184f, -0.01482532f, +0.06397840f, +0.15215315f,
    -0.10688859f, -0.01768018f, +0.06197682f, +0.13406777f, -0.10552422f, -0.01222899f, +0.06173200f, +0.16649240f,
    -0.11628240f, -0.01624644f, +0.06856942f, +0.16076413f, -0.08317817f, -0.00401934f, +0.07239269f, +0.17973306f,
    -0.09375231f, -0.00648847f, +0.06751947f, +0.18814264f, -0.10010364f, -0.00831303f, +0.07526674f, +0.15066913f,
    -0.11472419f, -0.01041994f, +0.07350467f, +0.16431492f, -0.10648406f, -0.00818389f, +0.07277713f, +0.17116972f,
    -0.10591904f, -0.00222131f, +0.07526167f, +0.15777809f, -0.09636197f, +0.00382409f, +0.07966353f, +0.15233697f,
    -0.09117776f, +0.00184235f, +0.07894982f, +0.21859670f, -0.07993965f, +0.00638250f, +0.09275463f, +0.19285717f
};

// 2-bit TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis (k=2, L=8)
template<typename idx_t>
static __global__ void __launch_bounds__(256, 1) k_set_rows_turbo2_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_tcq * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int iq_is_k,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    // Compute source and destination pointers (all threads, used by thread 0)
    const int64_t i_base = int64_t(grp) * QK_TURBO2_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_tcq * dst_blk = (block_turbo2_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBO2_TCQ);

    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ uint8_t bt[128][128]; // 256 states, 4-bit packed (2 per byte), safe even/odd serialization

    // Thread 0: load data, apply InnerQ, normalize, rotate (same order as turbo3_tcq)
    if (sid == 0) {
        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }

        // InnerQ scaling (on raw data, before normalization)
        if (d_innerq_calibrate) {
            for (int j = 0; j < 128; j++) {
                atomicAdd(&d_innerq_channel_sq[j], x[j] * x[j]);
                float abs_val = fabsf(x[j]);
                unsigned int * addr = (unsigned int *)&d_innerq_channel_max[j];
                unsigned int old_val = __float_as_uint(abs_val);
                unsigned int assumed;
                do {
                    assumed = *addr;
                    if (__uint_as_float(assumed) >= abs_val) break;
                } while (atomicCAS(addr, assumed, old_val) != assumed);
            }
            atomicAdd(&d_innerq_count, 1);
        }
        for (int j = 0; j < 128; j++) x[j] *= d_innerq_channel_scale[j];

        // Compute norm after InnerQ, then normalize
        norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
        for (int j = 0; j < 128; j++) x[j] *= inv_norm;

        // Forward FWHT
        turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);

        // Post-rotation extraction (if enabled)
        turbo_extract_append(x);

        cost[0] = grp_norm; // stash norm
    }
    __syncthreads();

    float saved_norm = cost[0];

    // Initialize Viterbi: free initial state (all 256 states equally viable)
    cost[sid] = 0.0f;
    __syncthreads();

    // Forward pass: 128 time steps, parallel across 256 states
    for (int t = 0; t < 128; t++) {
        float xt = x[t];

        // Right-shift trellis (k=2, L=8): ns = (prev >> 2) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 2) | p, for p = 0..3
        int base_prev = (sid & 0x3F) << 2;
        float dist = xt - d_turbo2_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 4; p++) {
            float c = cost[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        __syncthreads();
        cost[sid] = best + dist;

        // Store backtrace: 4-bit packed, 2 entries per byte (safe even/odd serialization)
        if (sid % 2 == 0) {
            bt[t][sid / 2] = (uint8_t)(best_p & 0x3);
        }
        __syncthreads();
        if (sid % 2 == 1) {
            bt[t][sid / 2] |= ((uint8_t)(best_p & 0x3)) << 4;
        }
        __syncthreads();
    }

    // Thread 0: find best final state, backtrack, pack bitstream
    if (sid == 0) {
        float min_cost = cost[0];
        int min_state = 0;
        for (int s = 1; s < 256; s++) {
            if (cost[s] < min_cost) {
                min_cost = cost[s];
                min_state = s;
            }
        }

        // Backtrack
        uint8_t * outputs = (uint8_t *)x;
        int state = min_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6); // output = top 2 bits (k=2)
            int p = (bt[t][state / 2] >> ((state % 2) * 4)) & 0x3;
            state = ((state & 0x3F) << 2) | p; // reconstruct predecessor
        }

        const int initial_state = state;

        // Compute reconstruction norm by replaying trellis from initial state
        float recon_norm_sq = 0.0f;
        int cur_state = initial_state;
        for (int t = 0; t < 128; t++) {
            cur_state = (cur_state >> 2) | (outputs[t] << 6);
            float c = d_turbo2_tcq_codebook[cur_state];
            recon_norm_sq += c * c;
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;

        // Pack bitstream: [6 prefix bits] [out_0 (2 bits)] ... [out_127 (2 bits)]
        for (int j = 0; j < 33; j++) dst_blk->qs[j] = 0;

        // Write initial state prefix (upper 6 bits = initial_state >> 2)
        dst_blk->qs[0] = (uint8_t)((initial_state >> 2) & 0x3F);

        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 2;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x3;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
            // 2 bits starting at bit_off: max bit_off=6, so 6+2=8 fits in one byte
            // But bit_off can be 0,2,4,6 (always even) so never crosses boundary
        }

        dst_blk->norm = __float2half(corrected_norm);
    }
}

// 2-bit TCQ GET_ROWS dequantize
#define QR_TURBO2_TCQ 2
static __device__ __forceinline__
void dequantize_turbo2_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 8-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.x = d_turbo2_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64
    {
        const int t = iqs + 64;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.y = d_turbo2_tcq_codebook[state] * norm;
    }
}
