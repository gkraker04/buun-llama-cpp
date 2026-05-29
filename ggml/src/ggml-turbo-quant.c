/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO3_0 (3-bit) and GGML_TYPE_TURBO4_0 (4-bit)
 * for use as --cache-type-k turbo3 --cache-type-v turbo3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#if defined(_WIN32)
#define _USE_MATH_DEFINES // for M_PI
#endif 

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* 4-bit: Lloyd-Max for N(0, 1/sqrt(128)), 16 centroids */
static const float CENTROIDS_4BIT[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static const float MIDPOINTS_4BIT[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, binary search on midpoints */
    if (val < MIDPOINTS_4BIT[7]) {
        if (val < MIDPOINTS_4BIT[3]) {
            if (val < MIDPOINTS_4BIT[1]) return val < MIDPOINTS_4BIT[0] ? 0 : 1;
            else                         return val < MIDPOINTS_4BIT[2] ? 2 : 3;
        } else {
            if (val < MIDPOINTS_4BIT[5]) return val < MIDPOINTS_4BIT[4] ? 4 : 5;
            else                         return val < MIDPOINTS_4BIT[6] ? 6 : 7;
        }
    } else {
        if (val < MIDPOINTS_4BIT[11]) {
            if (val < MIDPOINTS_4BIT[9])  return val < MIDPOINTS_4BIT[8] ? 8 : 9;
            else                          return val < MIDPOINTS_4BIT[10] ? 10 : 11;
        } else {
            if (val < MIDPOINTS_4BIT[13]) return val < MIDPOINTS_4BIT[12] ? 12 : 13;
            else                          return val < MIDPOINTS_4BIT[14] ? 14 : 15;
        }
    }
}

/* ---------- TURBO2_0: 2-bit PolarQuant, no QJL ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO2; j++) norm += x[i*QK_TURBO2 + j] * x[i*QK_TURBO2 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO2 / 4);
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_0: 2-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles quantize on GPU. CPU path is simplified.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBO3; j++) norm += x[i*QK_TURBO3 + j] * x[i*QK_TURBO3 + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, QK_TURBO3 / 4);
        memset(y[i].signs, 0, QK_TURBO3 / 8);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO3_TCQ: Trellis-Coded Quantization ---------- */

// TCQ codebook for 3-bit (512 entries), trained on Qwen3.5-27B KV activations
// Matches d_turbo3_tcq_codebook[512] in turbo-quant-cuda.cuh
static const float TURBO3_TCQ_CODEBOOK[512] = {
    -0.14559399f, -0.09062801f, -0.054925077f, -0.03699251f, -0.006363985f, +0.026264573f, +0.067378916f, +0.121981815f,
    -0.18648055f, -0.106522456f, -0.052047577f, -0.011695214f, +0.021953275f, +0.059698727f, +0.09831437f, +0.16083933f,
    -0.16390342f, -0.12639847f, -0.09513180f, -0.05938352f, -0.028396897f, +0.005973862f, +0.049104784f, +0.11334257f,
    -0.25952467f, -0.079778515f, -0.036024813f, +0.0003641268f, +0.031858794f, +0.073280424f, +0.11835553f, +0.19738495f,
    -0.14218009f, -0.10224814f, -0.062498566f, -0.027066832f, +0.00393002f, +0.04069300f, +0.08257346f, +0.14548601f,
    -0.18673635f, -0.13438253f, -0.088401966f, -0.05205436f, -0.02032501f, +0.012399545f, +0.05127183f, +0.10316186f,
    -0.10807011f, -0.065903045f, -0.032206114f, -0.0062006037f, +0.020679146f, +0.04422085f, +0.08313074f, +0.16821936f,
    -0.22979105f, -0.14431947f, -0.07689272f, -0.02755307f, +0.009225173f, +0.046684854f, +0.08834142f, +0.13766693f,
    -0.22114082f, -0.12612148f, -0.06890522f, -0.016128855f, +0.03691900f, +0.08474852f, +0.14940020f, +0.23229980f,
    -0.14933491f, -0.099693604f, -0.06738499f, -0.037100967f, -0.009332986f, +0.023535024f, +0.060272533f, +0.109464675f,
    -0.20200425f, -0.07398328f, -0.038700905f, -0.01714807f, +0.011161969f, +0.04528101f, +0.08902637f, +0.19573534f,
    -0.16645233f, -0.124482535f, -0.089342155f, -0.04427387f, -0.007353691f, +0.028033108f, +0.066108435f, +0.15552913f,
    -0.22295763f, -0.059887577f, -0.018804537f, +0.020141022f, +0.059682943f, +0.097920544f, +0.14080113f, +0.25698325f,
    -0.14248224f, -0.089685425f, -0.050101686f, -0.017257255f, +0.011412255f, +0.040830314f, +0.07400172f, +0.11997315f,
    -0.18649384f, -0.113997504f, -0.067775466f, -0.033394672f, +0.006586988f, +0.05312057f, +0.10433043f, +0.22344802f,
    -0.16138338f, -0.108194515f, -0.07600300f, -0.05135381f, -0.023365447f, +0.0087320795f, +0.045431953f, +0.09113002f,
    -0.12630440f, -0.07225349f, -0.032280035f, +0.0029231994f, +0.019239848f, +0.05081419f, +0.077840395f, +0.121695265f,
    -0.08928155f, -0.044983763f, -0.009889568f, +0.020831043f, +0.05684458f, +0.09409702f, +0.13867535f, +0.19084482f,
    -0.14182915f, -0.11380146f, -0.06904074f, -0.002002765f, +0.034864165f, +0.070399575f, +0.11403063f, +0.15394832f,
    -0.10876417f, -0.056122433f, -0.02267638f, +0.011113975f, +0.039639056f, +0.074084364f, +0.10155376f, +0.12540291f,
    -0.17693359f, -0.13940524f, -0.10049578f, -0.06796275f, -0.036915872f, +0.00062823476f, +0.042142134f, +0.17906062f,
    -0.09253492f, -0.04290128f, -0.006311852f, +0.023908244f, +0.049849935f, +0.078770354f, +0.10818172f, +0.15166481f,
    -0.12429565f, -0.07392063f, -0.029114135f, +0.0059440783f, +0.042675965f, +0.08425635f, +0.13836108f, +0.18634140f,
    -0.11795639f, -0.07033707f, -0.034163877f, -0.0008773357f, +0.03334606f, +0.07188203f, +0.12216825f, +0.17097956f,
    -0.18718453f, -0.14090346f, -0.097799584f, -0.059522875f, -0.019208657f, +0.03079176f, +0.09334672f, +0.15811224f,
    -0.27198875f, -0.16546582f, -0.11433405f, -0.06933013f, -0.04026183f, -0.0061146915f, +0.029263576f, +0.07322499f,
    -0.18471734f, -0.102074504f, -0.06492570f, -0.034418534f, -0.009636157f, +0.023043344f, +0.05751496f, +0.09905984f,
    -0.22826399f, -0.15946552f, -0.09913176f, -0.06585259f, -0.03252090f, +0.001313243f, +0.03556729f, +0.21612854f,
    -0.13243781f, -0.087299444f, -0.049820945f, -0.016216082f, +0.01799807f, +0.057916876f, +0.09001349f, +0.13221787f,
    -0.19516511f, -0.120894566f, -0.076130204f, -0.051442243f, -0.029535033f, -0.0020043184f, +0.029452588f, +0.075566076f,
    -0.27272871f, -0.15841717f, -0.105432935f, -0.06792948f, -0.024532158f, +0.014960791f, +0.054415092f, +0.101517834f,
    -0.21153601f, -0.15015371f, -0.08676790f, -0.04414934f, -0.0042129597f, +0.033762872f, +0.07589151f, +0.12768789f,
    -0.090428725f, -0.037582967f, +0.0013173596f, +0.03900247f, +0.06840049f, +0.116906695f, +0.16584939f, +0.25382105f,
    -0.13446195f, -0.07865091f, -0.039625354f, -0.0028398742f, +0.03019514f, +0.06799379f, +0.11850997f, +0.17521496f,
    -0.11350345f, -0.058599845f, -0.017512511f, +0.019431496f, +0.055897832f, +0.093173414f, +0.14820710f, +0.22092152f,
    -0.15165758f, -0.08869354f, -0.04974287f, -0.01705474f, +0.013134752f, +0.04367713f, +0.07733791f, +0.12430801f,
    -0.09329869f, -0.04673005f, -0.00045857552f, +0.042781368f, +0.07802363f, +0.11887439f, +0.16250038f, +0.28612965f,
    -0.12571070f, -0.07786012f, -0.03843933f, -0.0075433915f, +0.025822964f, +0.066053316f, +0.12021536f, +0.18341768f,
    -0.16079275f, -0.04921760f, -0.006114644f, +0.026215268f, +0.05699377f, +0.09813471f, +0.16080129f, +0.23786584f,
    -0.09980837f, -0.048535258f, -0.0096120685f, +0.025387142f, +0.05979822f, +0.09875251f, +0.14474337f, +0.20324114f,
    -0.15846540f, -0.09938028f, -0.061492465f, -0.03523542f, -0.0061364113f, +0.024916094f, +0.06037314f, +0.106796466f,
    -0.20557843f, -0.123237535f, -0.07734871f, -0.044549115f, -0.017114898f, +0.01616654f, +0.049574375f, +0.092319444f,
    -0.19221115f, -0.14642999f, -0.091701314f, -0.055265956f, -0.021026207f, +0.017720066f, +0.05786183f, +0.110154524f,
    -0.09956386f, -0.03870283f, +0.003052007f, +0.034851722f, +0.06256365f, +0.09628840f, +0.13979156f, +0.16582295f,
    -0.18026546f, -0.12448310f, -0.07424377f, -0.03954519f, -0.01221123f, +0.028641058f, +0.100819774f, +0.18240699f,
    -0.21520759f, -0.15573645f, -0.09820838f, -0.051450998f, -0.012993679f, +0.021135861f, +0.058727216f, +0.105848536f,
    -0.11207385f, -0.08335689f, -0.048542723f, -0.023198519f, +0.0039304253f, +0.037778318f, +0.07813917f, +0.13106476f,
    -0.17849164f, -0.120988995f, -0.078016765f, -0.043093704f, -0.016565649f, +0.015182641f, +0.050754096f, +0.09595712f,
    -0.22132620f, -0.13407415f, -0.065785654f, -0.013291034f, +0.032098345f, +0.07478225f, +0.12431934f, +0.19174045f,
    -0.095454164f, -0.051898945f, -0.015116375f, -0.012596778f, +0.018636847f, +0.05006925f, +0.087654814f, +0.13754296f,
    -0.15254061f, -0.09576059f, -0.052086458f, -0.01596074f, +0.017607626f, +0.04778498f, +0.08950204f, +0.14901252f,
    -0.26057002f, -0.12472382f, -0.074396215f, -0.03764066f, +0.0011168446f, +0.061569117f, +0.10793752f, +0.19771695f,
    -0.08661132f, -0.045195263f, -0.016098704f, +0.012780116f, +0.040476497f, +0.074102715f, +0.074102715f, +0.12635531f,
    -0.14047913f, -0.059587404f, -0.016261123f, +0.019801628f, +0.053541403f, +0.096650146f, +0.15005490f, +0.21051759f,
    -0.22986396f, -0.11964334f, -0.07266585f, -0.026522418f, +0.018169926f, +0.058630653f, +0.100647695f, +0.15919648f,
    -0.13251697f, -0.077567816f, -0.042766172f, -0.011389967f, +0.01831755f, +0.05304656f, +0.09620367f, +0.15567583f,
    -0.119819686f, -0.06772876f, -0.028123451f, +0.00876240f, +0.014405836f, +0.048829112f, +0.08422175f, +0.13823749f,
    -0.16379014f, -0.08956941f, -0.041652776f, +0.008921398f, +0.05473602f, +0.10037984f, +0.16022855f, +0.23457925f,
    -0.115844205f, -0.05939626f, -0.020390417f, +0.01374377f, +0.044976473f, +0.07873563f, +0.12207942f, +0.18412720f,
    -0.19048831f, -0.07587487f, -0.03220580f, -0.00011795067f, +0.02721784f, +0.04380719f, +0.07886723f, +0.13193911f,
    -0.13935551f, -0.092902906f, -0.052706074f, -0.017797327f, +0.015312965f, +0.056098964f, +0.11203423f, +0.24448302f,
    -0.17986591f, -0.10738580f, -0.06376371f, -0.026595421f, +0.00842492f, +0.04272362f, +0.08608052f, +0.15240218f,
    -0.10953678f, -0.057022586f, -0.012483291f, +0.024463262f, +0.06076792f, +0.09776234f, +0.12983681f, +0.18648379f,
    -0.16471463f, -0.089491285f, -0.037574016f, +0.004444791f, +0.039293647f, +0.07845859f, +0.12893885f, +0.23508036f
};

// TCQ codebook for 2-bit (256 entries), trained on Qwen3.5-27B KV activations
// Matches d_turbo2_tcq_codebook[256] in turbo-quant-cuda.cuh
static const float TURBO2_TCQ_CODEBOOK[256] = {
    -0.18030643f, -0.11009848f, -0.04742626f, +0.02894132f, -0.10523465f, -0.031312924f, +0.031491395f, +0.12263535f,
    -0.15660362f, -0.055477407f, +0.0046675834f, +0.06166081f, -0.07506216f, -0.016963918f, +0.043737844f, +0.116496615f,
    -0.08632783f, -0.022493735f, +0.041032985f, +0.10660284f, -0.06274858f, -0.0036939639f, +0.02095157f, +0.07539709f,
    -0.09802641f, -0.008419088f, +0.059072323f, +0.17311879f, -0.093109086f, -0.02654333f, +0.014827672f, +0.07793592f,
    -0.031235758f, +0.01271591f, +0.08752262f, +0.17246453f, -0.14595252f, -0.07227624f, +0.013628688f, +0.08131674f,
    -0.036909282f, +0.0018896917f, +0.05209119f, +0.12407892f, -0.13689458f, -0.06054520f, +0.0064648795f, +0.07551241f,
    -0.18980840f, -0.110128626f, -0.046503957f, +0.026387159f, -0.034967307f, +0.04810357f, +0.072072044f, +0.14355458f,
    -0.10182410f, -0.02907887f, +0.014033012f, +0.083419636f, -0.056140676f, +0.008405868f, +0.066070884f, +0.14037225f,
    -0.117427245f, -0.047159385f, +0.016928354f, +0.08142885f, -0.029359628f, +0.045608785f, +0.10559447f, +0.20061271f,
    -0.040425077f, +0.029068163f, +0.08408973f, +0.13628258f, -0.16633821f, -0.10711727f, -0.04196669f, +0.027895834f,
    -0.0054065837f, +0.058898676f, +0.12688550f, +0.18268861f, -0.16287325f, -0.11218357f, -0.07165227f, -0.009524379f,
    -0.24026902f, -0.073219374f, -0.0005165726f, +0.05959821f, -0.05532953f, +0.027044486f, +0.09425678f, +0.15356481f,
    -0.14381111f, -0.10563502f, -0.037867088f, +0.023611993f, -0.03624307f, +0.049588434f, +0.12192037f, +0.23462485f,
    -0.14990251f, -0.09659304f, -0.05886742f, +0.014878461f, -0.009889551f, +0.06910514f, +0.12120181f, +0.22596690f,
    -0.08290075f, -0.009009629f, +0.066151775f, +0.12188313f, -0.11591514f, -0.06952189f, -0.031633306f, +0.023740824f,
    -0.20510401f, -0.103369795f, +0.09148037f, +0.17268716f, -0.16597997f, -0.09207068f, -0.032810967f, +0.024847647f,
    -0.02487482f, +0.049298953f, +0.09624215f, +0.14217524f, -0.18418685f, -0.10147012f, -0.05841265f, +0.008057022f,
    -0.14269894f, -0.092456274f, -0.026881337f, +0.049792137f, -0.019881032f, +0.030333601f, +0.09736802f, +0.17764080f,
    -0.19579841f, -0.114739306f, -0.026823774f, +0.07466014f, -0.09001050f, -0.041468445f, +0.028473806f, +0.08870695f,
    -0.019396419f, +0.042828932f, +0.10885327f, +0.13335012f, -0.15005013f, -0.074581385f, -0.028608415f, +0.03848942f,
    -0.09687270f, -0.057059396f, +0.0077843578f, +0.06302297f, -0.23247094f, -0.14509225f, -0.032651436f, +0.027010715f,
    -0.047595482f, +0.06280303f, +0.114691675f, +0.17124057f, -0.21092793f, -0.13704823f, -0.07340412f, +0.0039013291f,
    -0.062834196f, +0.012601906f, +0.012601906f, +0.08721347f, -0.13256435f, -0.024173854f, +0.07723171f, +0.14801070f,
    -0.06471605f, -0.0017903054f, -0.0017903054f, +0.058302354f, -0.09731802f, -0.03400696f, +0.02762442f, +0.08986137f,
    -0.08288722f, -0.019051429f, +0.045709886f, +0.15211061f, -0.09507891f, -0.015612489f, +0.025347246f, +0.087257534f,
    -0.066236064f, -0.0047936034f, +0.06386274f, +0.15401669f, -0.105809286f, -0.051802177f, +0.01073050f, +0.08292137f,
    -0.11884470f, -0.04404144f, +0.02550729f, +0.02550729f, -0.01731189f, +0.062161792f, +0.12127554f, +0.21981733f,
    -0.17066145f, -0.11660990f, -0.049425896f, +0.021293938f, -0.04711412f, +0.026577346f, +0.055197213f, +0.12541275f,
    -0.028268812f, +0.015206398f, +0.09002519f, +0.12699963f, -0.10059831f, -0.026676945f, +0.059903253f, +0.13054545f,
    -0.09582803f, -0.033371232f, +0.010346129f, +0.066766635f, -0.09964944f, -0.028686784f, +0.021184925f, +0.09120017f,
    -0.16957201f, -0.07594450f, +0.04172865f, +0.18313301f, -0.051526368f, +0.011877304f, +0.011877304f, +0.07956263f,
    -0.13432936f, -0.05269006f, +0.03536416f, +0.117640756f, -0.022776067f, +0.042032316f, +0.10472976f, +0.18042557f
};

// WHT sign arrays (defined below, declared here for TCQ use)
extern const float turbo_wht_s1[128];
extern const float turbo_wht_s2[128];
extern void turbo_wht_forward(float * buf, int n);
extern void turbo_wht_inverse(float * buf, int n);

// Read 9-bit state from TCQ3 bitstream at element position t
static inline int turbo3_tcq_read_state(const uint8_t * qs, int t) {
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)qs[byte_idx] | ((uint16_t)qs[byte_idx + 1] << 8);
    return (raw >> bit_off) & 0x1FF;
}

// Read 8-bit state from TCQ2 bitstream at element position t
static inline int turbo2_tcq_read_state(const uint8_t * qs, int t) {
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)qs[byte_idx] | ((uint16_t)qs[byte_idx + 1] << 8);
    return (raw >> bit_off) & 0xFF;
}

// Viterbi encoder for TURBO3_TCQ: find optimal trellis path (3-bit right-shift, 512 states)
// State: 9 bits. Transition: state_t = ((state_{t-1} & 0x3F) << 3) | output_t
static void turbo3_tcq_viterbi_encode(
    const float * normalized, block_turbo3_tcq * blk, float norm) {

    // cost[2][512]: double-buffered trellis cost
    float cost[2][512];
    uint16_t bt[128][512]; // backtrace: predecessor index (3 bits of output in lower 3 bits)

    // Initialize costs to zero (all states equally likely)
    // Store for t=0 with uniform initialization
    for (int s = 0; s < 512; s++) {
        float x0 = normalized[0];
        cost[0][s] = (x0 - TURBO3_TCQ_CODEBOOK[s]) * (x0 - TURBO3_TCQ_CODEBOOK[s]);
        bt[0][s] = 0;
    }

    // Forward pass: Viterbi with right-shift trellis
    // State transition: state_t = ((state_{t-1} & 0x3F) << 3) | output_t
    // For current state s at time t, predecessor at t-1: pred = (s >> 3) | (p << 6) for p in 0..7
    for (int t = 1; t < 128; t++) {
        int cur = t & 1;
        int prev = 1 - cur;
        float xt = normalized[t];

        for (int s = 0; s < 512; s++) {
            int base_pred = s >> 3;
            float distortion = (xt - TURBO3_TCQ_CODEBOOK[s]) * (xt - TURBO3_TCQ_CODEBOOK[s]);

            float best_cost = INFINITY;
            int best_p = 0;
            for (int p = 0; p < 8; p++) {
                int pred = base_pred | (p << 6);
                float c = cost[prev][pred] + distortion;
                if (c < best_cost) {
                    best_cost = c;
                    best_p = p;
                }
            }
            cost[cur][s] = best_cost;
            bt[t][s] = (uint16_t)best_p;
        }
    }

    // Find best final state (minimum cost at t=127)
    int final_cost_idx = 0;
    float final_min = cost[127 & 1][0];
    for (int s = 1; s < 512; s++) {
        float c = cost[127 & 1][s];
        if (c < final_min) {
            final_min = c;
            final_cost_idx = s;
        }
    }

    // Backtrack to get output symbols (reversing CUDA k_set_rows_turbo3_tcq logic)
    // outputs[t] = state >> 6 extracts the 3-bit output symbol embedded in the 9-bit state
    // state = ((state & 0x3F) << 3) | p reconstructs predecessor
    uint8_t outputs[128];
    int state = final_cost_idx;
    for (int t = 127; t >= 0; t--) {
        outputs[t] = (uint8_t)(state >> 6);
        int p = (int)bt[t][state & 0x3F];
        state = ((state & 0x3F) << 3) | p;
    }
    int init_state = state;

    // Compute norm correction and pack bitstream
    // Rebuild state sequence from init_state + outputs to compute recon_norm
    {
        // Rebuild state sequence from initial state and outputs
        int st = init_state;
        float recon_sq_val = 0.0f;
        for (int t = 0; t < 128; t++) {
            st = ((st & 0x3F) << 3) | outputs[t];
            float c = TURBO3_TCQ_CODEBOOK[st];
            recon_sq_val += c * c;
        }
        float recon_norm = sqrtf(recon_sq_val);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;

        // Pack bitstream: 6-bit initial state + 128 × 3-bit outputs
        // init_bits = (init_state >> 3) & 0x3F (bits 3-8 of initial state after processing all outputs)
        // Actually from CUDA: const int init_bits = (shared_initial_state >> 3) & 0x3F;
        // where shared_initial_state is the state BEFORE processing any outputs
        int init_bits = (init_state >> 3) & 0x3F;

        for (int byte = 0; byte < 49; byte++) {
            uint8_t packed = 0;
            for (int bit = 0; bit < 8; bit++) {
                int pos = byte * 8 + bit;
                int v = 0;
                if (pos < 6) {
                    v = (init_bits >> pos) & 1;
                } else {
                    int sym_bit_pos = pos - 6;
                    int sym_idx = sym_bit_pos / 3;
                    if (sym_idx < 128) {
                        v = (outputs[sym_idx] >> (sym_bit_pos % 3)) & 1;
                    }
                }
                packed |= (uint8_t)(v << bit);
            }
            blk->qs[byte] = packed;
        }
        blk->norm = GGML_FP32_TO_FP16(corrected_norm);
    }
}

void quantize_row_turbo3_tcq_ref(const float * GGML_RESTRICT x, block_turbo3_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = (int)(k / QK_TURBO3_TCQ);
    const int d = QK_TURBO3_TCQ;
    float block_buf[TURBO_D];

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        // Step 1: compute norm
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        // Step 2: normalize
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        // Step 3: WHT rotate (matching k_turbo_wht on GPU)
        memcpy(block_buf, normalized, d * sizeof(float));
        for (int i = 0; i < d; i++) block_buf[i] *= turbo_wht_s1[i];
        for (int h = 1; h < d; h *= 2) {
            for (int i = 0; i < d; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = block_buf[j], b = block_buf[j + h];
                    block_buf[j] = a + b;
                    block_buf[j + h] = a - b;
                }
            }
        }
        const float inv_sqrt = 0.08838834764831845f;
        for (int i = 0; i < d; i++) normalized[i] = block_buf[i] * inv_sqrt * turbo_wht_s2[i];

        // Step 4: Viterbi encode
        turbo3_tcq_viterbi_encode(normalized, &y[block], norm);
    }
}

void dequantize_row_turbo3_tcq(const block_turbo3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3_TCQ == 0);
    const int nb = (int)(k / QK_TURBO3_TCQ);
    const int d = QK_TURBO3_TCQ;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        // Decode TCQ bitstream: read 9-bit state at each position, lookup codebook
        float rotated_recon[TURBO_D];
        for (int j = 0; j < d; j++) {
            int state = turbo3_tcq_read_state(x[block].qs, j);
            rotated_recon[j] = TURBO3_TCQ_CODEBOOK[state];
        }

        // Inverse WHT rotate
        for (int i = 0; i < d; i++) rotated_recon[i] *= turbo_wht_s2[i];
        for (int h = 1; h < d; h *= 2) {
            for (int i = 0; i < d; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = rotated_recon[j], b = rotated_recon[j + h];
                    rotated_recon[j] = a + b;
                    rotated_recon[j + h] = a - b;
                }
            }
        }
        const float inv_sqrt = 0.08838834764831845f;
        for (int i = 0; i < d; i++) rotated_recon[i] *= inv_sqrt * turbo_wht_s1[i];

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = rotated_recon[i] * norm;
        }
    }
}

size_t quantize_turbo3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBO3_TCQ) * sizeof(block_turbo3_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_tcq_ref(
            src + row * n_per_row,
            (block_turbo3_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_TCQ: 2-bit Trellis-Coded Quantization ---------- */

// Viterbi encoder for TURBO2_TCQ: 2-bit right-shift trellis, 256 states
// State: 8 bits. Transition: state_t = ((state_{t-1} & 0x3F) << 2) | output_t
static void turbo2_tcq_viterbi_encode(
    const float * normalized, block_turbo2_tcq * blk, float norm) {

    float cost[2][256];
    uint16_t bt[128][256];

    // Initialize t=0
    for (int s = 0; s < 256; s++) {
        float x0 = normalized[0];
        cost[0][s] = (x0 - TURBO2_TCQ_CODEBOOK[s]) * (x0 - TURBO2_TCQ_CODEBOOK[s]);
        bt[0][s] = 0;
    }

    // Forward pass
    for (int t = 1; t < 128; t++) {
        int cur = t & 1;
        int prev = 1 - cur;
        float xt = normalized[t];

        for (int s = 0; s < 256; s++) {
            // Predecessor: pred = (s >> 2) | (p << 6) for p in 0..3
            int base_pred = s >> 2;
            float distortion = (xt - TURBO2_TCQ_CODEBOOK[s]) * (xt - TURBO2_TCQ_CODEBOOK[s]);

            float best_cost = INFINITY;
            int best_p = 0;
            for (int p = 0; p < 4; p++) {
                int pred = base_pred | (p << 6);
                float c = cost[prev][pred] + distortion;
                if (c < best_cost) {
                    best_cost = c;
                    best_p = p;
                }
            }
            cost[cur][s] = best_cost;
            bt[t][s] = (uint16_t)best_p;
        }
    }

    // Find best final state
    int final_state = 0;
    float min_cost = cost[127 & 1][0];
    for (int s = 1; s < 256; s++) {
        float c = cost[127 & 1][s];
        if (c < min_cost) {
            min_cost = c;
            final_state = s;
        }
    }

    // Backtrack
    uint8_t outputs[128];
    int state = final_state;
    for (int t = 127; t >= 0; t--) {
        // For 2-bit TCQ, output symbol = state >> 6 (upper 2 bits of 8-bit state)
        outputs[t] = (uint8_t)(state >> 6);
        int p = (int)bt[t][state & 0x3F];
        state = ((state & 0x3F) << 2) | p;
    }
    int init_state = state;

    // Compute norm correction and pack bitstream
    {
        int st = init_state;
        float recon_sq = 0.0f;
        for (int t = 0; t < 128; t++) {
            st = ((st & 0x3F) << 2) | outputs[t];
            float c = TURBO2_TCQ_CODEBOOK[st];
            recon_sq += c * c;
        }
        float recon_norm = sqrtf(recon_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;

        // Pack bitstream: 6-bit init prefix + 128 × 2-bit outputs = 262 bits = 33 bytes
        int init_bits = (init_state >> 2) & 0x3F;

        for (int byte = 0; byte < 33; byte++) {
            uint8_t packed = 0;
            for (int bit = 0; bit < 8; bit++) {
                int pos = byte * 8 + bit;
                int v = 0;
                if (pos < 6) {
                    v = (init_bits >> pos) & 1;
                } else {
                    int sym_bit_pos = pos - 6;
                    int sym_idx = sym_bit_pos / 2;
                    if (sym_idx < 128) {
                        v = (outputs[sym_idx] >> (sym_bit_pos % 2)) & 1;
                    }
                }
                packed |= (uint8_t)(v << bit);
            }
            blk->qs[byte] = packed;
        }
        blk->norm = GGML_FP32_TO_FP16(corrected_norm);
    }
}

void quantize_row_turbo2_tcq_ref(const float * GGML_RESTRICT x, block_turbo2_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2_TCQ == 0);
    const int nb = (int)(k / QK_TURBO2_TCQ);
    const int d = QK_TURBO2_TCQ;
    float block_buf[TURBO_D];

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        // Step 1: compute norm
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        // Step 2: normalize
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        // Step 3: WHT rotate
        memcpy(block_buf, normalized, d * sizeof(float));
        for (int i = 0; i < d; i++) block_buf[i] *= turbo_wht_s1[i];
        for (int h = 1; h < d; h *= 2) {
            for (int i = 0; i < d; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = block_buf[j], b = block_buf[j + h];
                    block_buf[j] = a + b;
                    block_buf[j + h] = a - b;
                }
            }
        }
        const float inv_sqrt = 0.08838834764831845f;
        for (int i = 0; i < d; i++) normalized[i] = block_buf[i] * inv_sqrt * turbo_wht_s2[i];

        // Step 4: Viterbi encode
        turbo2_tcq_viterbi_encode(normalized, &y[block], norm);
    }
}

void dequantize_row_turbo2_tcq(const block_turbo2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2_TCQ == 0);
    const int nb = (int)(k / QK_TURBO2_TCQ);
    const int d = QK_TURBO2_TCQ;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        // Decode TCQ bitstream: read 8-bit state at each position, lookup codebook
        float rotated_recon[TURBO_D];
        for (int j = 0; j < d; j++) {
            int state = turbo2_tcq_read_state(x[block].qs, j);
            rotated_recon[j] = TURBO2_TCQ_CODEBOOK[state];
        }

        // Inverse WHT rotate (matching k_turbo_wht inverse)
        for (int i = 0; i < d; i++) rotated_recon[i] *= turbo_wht_s2[i];
        for (int h = 1; h < d; h *= 2) {
            for (int i = 0; i < d; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    float a = rotated_recon[j], b = rotated_recon[j + h];
                    rotated_recon[j] = a + b;
                    rotated_recon[j + h] = a - b;
                }
            }
        }
        const float inv_sqrt = 0.08838834764831845f;
        for (int i = 0; i < d; i++) rotated_recon[i] *= inv_sqrt * turbo_wht_s1[i];

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = rotated_recon[i] * norm;
        }
    }
}

size_t quantize_turbo2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBO2_TCQ) * sizeof(block_turbo2_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_tcq_ref(
            src + row * n_per_row,
            (block_turbo2_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 4-bit PolarQuant (16 centroids, no QJL) ---------- */


/* ---------- CPU WHT butterfly matching GPU k_turbo_wht ---------- */
#define TURBO_WHT_N 128
const float turbo_wht_s1[TURBO_WHT_N] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
const float turbo_wht_s2[TURBO_WHT_N] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

static void turbo_wht_forward(float * buf, int n) {
    for (int i = 0; i < n; i++) buf[i] *= turbo_wht_s1[i];
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = i; j < i + h; j++) {
                float a = buf[j], b = buf[j + h];
                buf[j] = a + b;
                buf[j + h] = a - b;
            }
        }
    }
    const    float inv_sqrt = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < n; i++) buf[i] *= inv_sqrt * turbo_wht_s2[i];
}

static void turbo_wht_inverse(float * buf, int n) {
    for (int i = 0; i < n; i++) buf[i] *= turbo_wht_s2[i];
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = i; j < i + h; j++) {
                float a = buf[j], b = buf[j + h];
                buf[j] = a + b;
                buf[j + h] = a - b;
            }
        }
    }
    const    float inv_sqrt = 0.08838834764831845f;
    for (int i = 0; i < n; i++) buf[i] *= inv_sqrt * turbo_wht_s1[i];
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: WHT rotate */
        turbo_wht_forward(normalized, d);

        /* Step 3: 4-bit quantization — find nearest of 16 centroids */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(normalized[i]);
        }

        /* Step 4: Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = CENTROIDS_4BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 4-bit indices: 2 per byte, low nibble first */
        for (int i = 0; i < d; i += 2) {
            y[block].qs[i / 2] = (uint8_t)((indices[i + 1] << 4) | (indices[i] & 0xF));
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 4-bit indices and reconstruct in rotated space */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (i & 1) ? (x[block].qs[i / 2] >> 4) : (x[block].qs[i / 2] & 0xF);
            rotated_recon[i] = CENTROIDS_4BIT[idx];
        }

        /* Inverse WHT rotate */
        turbo_wht_inverse(rotated_recon, d);
        float * dst = y + block * d;
        memcpy(dst, rotated_recon, d * sizeof(float));

        /* Scale by norm */
        for (int i = 0; i < d; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
