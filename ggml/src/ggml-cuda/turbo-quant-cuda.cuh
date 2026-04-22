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
// GLA-trained codebook (512 entries) for N(0, 1/sqrt(128)) post-FWHT data
// MSE reduction: 8.0% vs Lloyd-Max 3-bit, +0.36 dB. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook[512] = {
    -0.06214560f, -0.12502260f, -0.07382884f, -0.03642227f, -0.03433620f, +0.01467172f, +0.06410509f, +0.13042919f,
    -0.18261877f, -0.12620314f, -0.07955874f, -0.04237626f, -0.02427942f, +0.01787884f, +0.06815198f, +0.15012592f,
    -0.27307857f, -0.13798780f, -0.08195644f, -0.04301426f, -0.01905152f, +0.02030195f, +0.07178343f, +0.17988904f,
    -0.21642926f, -0.12488871f, -0.08134518f, -0.04111187f, -0.01631510f, +0.01982895f, +0.06055766f, +0.11562954f,
    -0.17939871f, -0.11633642f, -0.07294229f, -0.03638248f, -0.02993495f, +0.01813853f, +0.07288168f, +0.19113147f,
    -0.14881812f, -0.11361916f, -0.07495335f, -0.03676215f, -0.02418072f, +0.02203872f, +0.06822619f, +0.12140875f,
    -0.13696862f, -0.11913544f, -0.07435404f, -0.03741877f, -0.02255596f, +0.02230758f, +0.07212648f, +0.14185024f,
    -0.16633589f, -0.11872256f, -0.07394852f, -0.03593235f, -0.01553523f, +0.02398467f, +0.06832753f, +0.13534149f,
    -0.03839786f, -0.12963653f, -0.07734107f, -0.03048394f, -0.01528207f, +0.02721232f, +0.07182979f, +0.15646212f,
    -0.17004525f, -0.11346669f, -0.06999251f, -0.03029514f, -0.01125266f, +0.02686655f, +0.07545448f, +0.13694111f,
    -0.18039437f, -0.11992071f, -0.06681578f, -0.02839880f, -0.00743602f, +0.02790579f, +0.07330064f, +0.12623274f,
    -0.20652470f, -0.11124615f, -0.06985663f, -0.02941486f, -0.01411294f, +0.02536411f, +0.07430467f, +0.17413373f,
    -0.12193601f, -0.10416105f, -0.06384080f, -0.02650684f, -0.00969252f, +0.03313519f, +0.08342334f, +0.13074860f,
    -0.16930850f, -0.11127753f, -0.06598979f, -0.02602533f, -0.00232888f, +0.03188136f, +0.07802786f, +0.15626628f,
    -0.18215934f, -0.11699776f, -0.06664142f, -0.02625479f, -0.00638505f, +0.03065578f, +0.08236100f, +0.13840177f,
    -0.19317177f, -0.10492659f, -0.06161392f, -0.02162120f, -0.01015966f, +0.03286023f, +0.08133359f, +0.17038604f,
    -0.14706501f, -0.10290411f, -0.06685282f, -0.02645278f, -0.00863619f, +0.02971112f, +0.08201745f, +0.14857379f,
    -0.17289503f, -0.11179614f, -0.06453714f, -0.02468797f, -0.00133530f, +0.03741648f, +0.08516291f, +0.15402340f,
    -0.15536236f, -0.10765649f, -0.06094400f, -0.02300114f, +0.00029752f, +0.03637884f, +0.07929008f, +0.17100569f,
    -0.17697978f, -0.10231667f, -0.06529455f, -0.02293338f, -0.00148205f, +0.03854367f, +0.08814529f, +0.13168870f,
    -0.13953234f, -0.10575779f, -0.05994583f, -0.02356202f, +0.00271234f, +0.03669587f, +0.07947033f, +0.16847364f,
    -0.15533706f, -0.10258552f, -0.05738193f, -0.01827062f, -0.00300421f, +0.03801045f, +0.08877521f, +0.17080041f,
    -0.15332940f, -0.10654955f, -0.05786463f, -0.02092363f, +0.00340581f, +0.03911386f, +0.08846670f, +0.16960522f,
    -0.14189870f, -0.10086122f, -0.05622103f, -0.01850897f, +0.00448368f, +0.03768524f, +0.09183349f, +0.15242210f,
    -0.19414657f, -0.11789841f, -0.06185782f, -0.02179393f, +0.00099846f, +0.03821497f, +0.08172760f, +0.13434775f,
    -0.20885529f, -0.10892752f, -0.05702097f, -0.01984956f, -0.01066291f, +0.03687306f, +0.07769896f, +0.13066619f,
    -0.20454105f, -0.10310126f, -0.06025568f, -0.02365155f, -0.00354930f, +0.03661468f, +0.08194148f, +0.13656945f,
    -0.21676115f, -0.10478667f, -0.05393605f, -0.01993795f, -0.00820763f, +0.04138235f, +0.07969661f, +0.13209077f,
    -0.17323336f, -0.10550813f, -0.05734613f, -0.01766877f, -0.02747631f, +0.04171953f, +0.08544579f, +0.13345307f,
    -0.23133917f, -0.10298422f, -0.05507982f, -0.01982192f, +0.00067949f, +0.03814183f, +0.08369409f, +0.13821313f,
    -0.19928116f, -0.11130875f, -0.05340059f, -0.01617996f, -0.01133017f, +0.04264435f, +0.08918515f, +0.13789825f,
    -0.21021035f, -0.11464639f, -0.05496802f, -0.01682054f, -0.00813110f, +0.04224558f, +0.08554578f, +0.13898933f,
    +0.05824982f, -0.08798827f, -0.04681230f, -0.00452132f, +0.01200696f, +0.05192965f, +0.09365089f, +0.23573873f,
    -0.12913230f, -0.08685162f, -0.04458628f, -0.00547155f, +0.01847301f, +0.05343122f, +0.10049206f, +0.22008317f,
    -0.12986002f, -0.08287955f, -0.03906727f, -0.00029259f, +0.01874791f, +0.05338605f, +0.10466762f, +0.20151090f,
    -0.13633583f, -0.08627453f, -0.04213292f, +0.00132776f, +0.01959301f, +0.05116458f, +0.10046281f, +0.17990007f,
    -0.00032820f, -0.08047196f, -0.04150430f, +0.00495035f, +0.02460073f, +0.05700675f, +0.10304886f, +0.24353701f,
    -0.11576355f, -0.07890392f, -0.04049335f, +0.00360325f, +0.02240071f, +0.05644501f, +0.10609626f, +0.21074118f,
    -0.07595720f, -0.07621986f, -0.03624055f, +0.00340502f, +0.02245199f, +0.05765819f, +0.10046056f, +0.22126861f,
    +0.00369544f, -0.08005535f, -0.04415584f, +0.01517149f, +0.02685620f, +0.05909380f, +0.10593831f, +0.22081248f,
    -0.06680196f, -0.10953058f, -0.03563205f, -0.00266230f, +0.01546759f, +0.05250149f, +0.10176462f, +0.14625405f,
    -0.15167430f, -0.08410123f, -0.03899765f, +0.00253949f, +0.00795206f, +0.05728987f, +0.10241884f, +0.16230208f,
    -0.16318576f, -0.08883854f, -0.03927768f, -0.00535170f, +0.02332669f, +0.05963313f, +0.10572989f, +0.15099563f,
    -0.16730580f, -0.08023939f, -0.03552054f, +0.00738248f, +0.02012303f, +0.05865037f, +0.10037655f, +0.15610783f,
    -0.11741391f, -0.07922334f, -0.03891004f, -0.00399082f, +0.01016806f, +0.05887343f, +0.11150557f, +0.19226326f,
    -0.13684212f, -0.07714037f, -0.03576919f, +0.00412331f, +0.02227807f, +0.06159182f, +0.10662748f, +0.15746773f,
    -0.15345299f, -0.08278162f, -0.03611321f, -0.00130648f, +0.02179412f, +0.06199090f, +0.11131456f, +0.17706025f,
    -0.14355213f, -0.08844274f, -0.03796425f, +0.00253411f, +0.01827505f, +0.05940008f, +0.10408053f, +0.17675605f,
    -0.05895919f, -0.11250976f, -0.02962677f, +0.00731344f, +0.02797250f, +0.06335406f, +0.11320216f, +0.17533616f,
    -0.16336080f, -0.07982693f, -0.03102628f, +0.00589648f, +0.02455771f, +0.06382181f, +0.11378996f, +0.18497091f,
    -0.13465274f, -0.07912506f, -0.03130402f, +0.00751457f, +0.02270424f, +0.06691534f, +0.11350100f, +0.18588624f,
    -0.19797648f, -0.07522168f, -0.03364683f, +0.00606670f, +0.03469620f, +0.06857832f, +0.11861626f, +0.17369726f,
    -0.11124050f, -0.06744437f, -0.02803577f, +0.00831613f, +0.01702440f, +0.06464885f, +0.11068181f, +0.16227544f,
    -0.14629919f, -0.07421948f, -0.02805495f, +0.01144904f, +0.02714360f, +0.07001255f, +0.11461623f, +0.16519872f,
    -0.12666686f, -0.07574533f, -0.02780315f, +0.00959052f, +0.03110829f, +0.06871058f, +0.12040406f, +0.21136766f,
    -0.11686254f, -0.07160583f, -0.02558218f, +0.01062428f, +0.03510914f, +0.06829799f, +0.11874134f, +0.18412868f,
    +0.03562390f, -0.09074698f, -0.02177646f, +0.02129744f, +0.03501983f, +0.07094062f, +0.11870168f, +0.18634565f,
    -0.13608137f, -0.06850736f, -0.02686231f, +0.01575120f, +0.03735312f, +0.07481827f, +0.12027726f, +0.16725661f,
    -0.14642444f, -0.07563780f, -0.02725398f, +0.00984016f, +0.03189267f, +0.07082478f, +0.12212736f, +0.21523769f,
    -0.17905528f, -0.06675190f, -0.01796044f, +0.02057233f, +0.04112163f, +0.07696854f, +0.11806865f, +0.16266986f,
    -0.08487759f, -0.05831570f, -0.02495721f, +0.01569513f, +0.02182579f, +0.07225720f, +0.11856424f, +0.24034521f,
    -0.11434156f, -0.06594269f, -0.02086656f, +0.02117146f, +0.04170140f, +0.07529766f, +0.12737427f, +0.18986302f,
    -0.17176220f, -0.07130082f, -0.01897010f, +0.02075581f, +0.03639790f, +0.07389685f, +0.11837092f, +0.18144726f,
    -0.13999806f, -0.07424239f, -0.01807416f, +0.03373757f, +0.01737867f, +0.07921319f, +0.13711509f, +0.23007135f,
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
    // cb[512]    : codebook (loaded from constant memory for faster access)
    // bt[128][256]: backtrace, 4-bit packed (best predecessor index 0-7)
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ float cb[512];
    __shared__ uint8_t bt[128][256]; // 32KB: bt[t][s/2] = (pred_s_even) | (pred_s_odd << 4)

    // Load codebook into shared memory
    cb[sid] = d_turbo3_tcq_codebook[sid];

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

    // Initialize Viterbi: start from state 0 only (matches decode's 6 zero-prefix bits)
    cost[sid] = (sid == 0) ? 0.0f : 1e30f;
    __syncthreads();

    // Forward pass: 128 time steps, fully parallel across 512 states
    for (int t = 0; t < 128; t++) {
        float xt = x[t];

        // For state sid: find best predecessor
        // Right-shift trellis: ns = (prev >> 3) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 3) | p, for p = 0..7
        int base_prev = (sid & 0x3F) << 3;
        float dist = xt - cb[sid];
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

        // Compute reconstruction norm for correction
        float recon_norm_sq = 0.0f;
        for (int t = 0; t < 128; t++) {
            int st;
            if (t == 0) {
                st = outputs[0] << 6;
            } else if (t == 1) {
                st = (outputs[1] << 6) | (outputs[0] << 3);
            } else {
                st = (outputs[t] << 6) | (outputs[t-1] << 3) | outputs[t-2];
            }
            float c = cb[st];
            recon_norm_sq += c * c;
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;

        // Pack bitstream: [6 zero bits] [out_0 (3 bits)] [out_1 (3 bits)] ... [out_127 (3 bits)]
        for (int j = 0; j < 49; j++) dst_blk->qs[j] = 0;

        for (int t = 0; t < 128; t++) {
            int out = outputs[t];
            int bit_pos = 6 + t * 3;
            for (int b = 0; b < 3; b++) {
                if (out & (1 << b)) {
                    int byte_idx = (bit_pos + b) / 8;
                    int bit_off = (bit_pos + b) % 8;
                    dst_blk->qs[byte_idx] |= (1 << bit_off);
                }
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
