#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

static __constant__ float d_turbo_centroids_2bit_fattn[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_centroids_3bit_fattn[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_centroids_4bit_fattn[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

// GLA-trained TCQ codebook for fattn compilation unit (same values as turbo-quant-cuda.cuh)
static __constant__ float d_turbo3_tcq_codebook_fattn[512] = {
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

// FWHT rotation sign arrays for FA inline rotation (same values as turbo-quant-cuda.cuh)
static __constant__ float d_turbo_wht_signs1_fattn[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
static __constant__ float d_turbo_wht_signs2_fattn[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// InnerQ: per-channel inverse scale for Q pre-rotation (fattn compilation unit)
// Initialized to all 1.0 (identity). Updated by turbo_innerq_finalize_calibration().
static __device__ float d_innerq_channel_scale_inv_fattn[128];

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}


template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo2_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo2_0 * K_t2 = (const block_turbo2_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[4];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO2;
        const int j_start = elem0 % QK_TURBO2;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t2[ib].norm);
#pragma unroll
            for (int c = 0; c < 4; c++) {
                cn[c] = d_turbo_centroids_2bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        const uint8_t qs_lo = K_t2[ib].qs[j_start / 4];
        const uint8_t qs_hi = K_t2[ib].qs[j_start / 4 + 1];

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const uint8_t qs_b0 = (lj < 4) ? qs_lo : qs_hi;
            const int idx0 = (qs_b0 >> ((lj % 4) * 2)) & 0x3;
            const int lj1 = lj + 1;
            const uint8_t qs_b1 = (lj1 < 4) ? qs_lo : qs_hi;
            const int idx1 = (qs_b1 >> ((lj1 % 4) * 2)) & 0x3;
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += cn[idx0] * qf.x + cn[idx1] * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo3_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo3_0 * K_t3 = (const block_turbo3_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[8];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO3;
        const int j_start = elem0 % QK_TURBO3;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t3[ib].norm);
#pragma unroll
            for (int c = 0; c < 8; c++) {
                cn[c] = d_turbo_centroids_3bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        const uint8_t qs_lo = K_t3[ib].qs[j_start / 4];
        const uint8_t qs_hi = K_t3[ib].qs[j_start / 4 + 1];
        const uint8_t signs = K_t3[ib].signs[j_start / 8];

#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            int idx0, idx1;
            { const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
              const uint8_t low2 = (qs_b >> ((lj % 4) * 2)) & 0x3;
              const uint8_t hi1  = (signs >> lj) & 0x1;
              idx0 = low2 | (hi1 << 2); }
            { const int lj1 = lj + 1;
              const uint8_t qs_b = (lj1 < 4) ? qs_lo : qs_hi;
              const uint8_t low2 = (qs_b >> ((lj1 % 4) * 2)) & 0x3;
              const uint8_t hi1  = (signs >> lj1) & 0x1;
              idx1 = low2 | (hi1 << 2); }
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += cn[idx0] * qf.x + cn[idx1] * qf.y;
        }
    }
    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_turbo4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {
    const block_turbo4_0 * K_t4 = (const block_turbo4_0 *) K_c;
    GGML_UNUSED(Q_q8); GGML_UNUSED(Q_ds_v);
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;
    float sum = 0.0f;
    int prev_ib = -1;
    float cn[16];
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        const int base_f2 = k_KQ_0 + (threadIdx.x % nthreads) * cpy_ne;
        const int elem0 = base_f2 * 2;
        const int ib = elem0 / QK_TURBO4;
        const int j_start = elem0 % QK_TURBO4;

        if (ib != prev_ib) {
            const float norm = __half2float(K_t4[ib].norm);
#pragma unroll
            for (int c = 0; c < 16; c++) {
                cn[c] = d_turbo_centroids_4bit_fattn[c] * norm;
            }
            prev_ib = ib;
        }

        // 4-bit indices: 2 per byte, simple nibble extraction
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
            const int lj = k_KQ_1 * 2;
            const int j0 = j_start + lj;
            const uint8_t byte0 = K_t4[ib].qs[j0 / 2];
            const uint8_t byte1 = K_t4[ib].qs[(j0 + 1) / 2];
            const uint8_t idx0 = (j0 & 1) ? (byte0 >> 4) : (byte0 & 0xF);
            const uint8_t idx1 = ((j0 + 1) & 1) ? (byte1 >> 4) : (byte1 & 0xF);
            const float k0 = cn[idx0];
            const float k1 = cn[idx1];
#ifdef V_DOT2_F32_F16_AVAILABLE
            const float2 qf = __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            const float2 qf = ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1];
#endif
            sum += k0 * qf.x + k1 * qf.y;
        }
    }
    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}


template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo2_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const int64_t ib = i0 / QK_TURBO2;
    const int     j0 = (int)(i0 % QK_TURBO2);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float cn[4];
#pragma unroll
    for (int c = 0; c < 4; c++) cn[c] = d_turbo_centroids_2bit_fattn[c] * norm;
    const uint8_t qs_lo = x[ib].qs[j0 / 4];
    const uint8_t qs_hi = (ne > 4 || j0 % 4 + ne > 4) ? x[ib].qs[j0 / 4 + 1] : 0;
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int lj = j0 % 4 + l;
        const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
        const int idx = (qs_b >> ((lj % 4) * 2)) & 0x3;
        vals[l] = cn[idx];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo3_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const int64_t ib = i0 / QK_TURBO3;
    const int     j0 = (int)(i0 % QK_TURBO3);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    // Register-based centroid × norm LUT
    float cn[8];
#pragma unroll
    for (int c = 0; c < 8; c++) cn[c] = d_turbo_centroids_3bit_fattn[c] * norm;
    // Batch-load qs and signs bytes
    const uint8_t qs_lo = x[ib].qs[j0 / 4];
    const uint8_t qs_hi = (ne > 4 || j0 % 4 + ne > 4) ? x[ib].qs[j0 / 4 + 1] : 0;
    const uint8_t signs = x[ib].signs[j0 / 8];
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int lj = j0 % 4 + l;
        const uint8_t qs_b = (lj < 4) ? qs_lo : qs_hi;
        const uint8_t low2 = (qs_b >> ((lj % 4) * 2)) & 0x3;
        const uint8_t hi1  = (signs >> ((j0 % 8) + l)) & 0x1;
        vals[l] = cn[low2 | (hi1 << 2)];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_turbo4_0(
        const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const int64_t ib = i0 / QK_TURBO4;
    const int     j0 = (int)(i0 % QK_TURBO4);
    const float norm = __half2float(x[ib].norm);
    static_assert(ne == 2 || ne == 4 || ne == 8, "bad ne");
    float cn[16];
#pragma unroll
    for (int c = 0; c < 16; c++) cn[c] = d_turbo_centroids_4bit_fattn[c] * norm;
    float vals[ne];
#pragma unroll
    for (int l = 0; l < ne; l++) {
        const int j = j0 + l;
        const uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
        vals[l] = cn[idx];
    }
#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        for (int l0 = 0; l0 < ne; l0 += 2)
            ((half2 *)dst)[l0/2] = make_half2(__float2half(vals[l0]), __float2half(vals[l0+1]));
    } else
#endif
    if constexpr (std::is_same_v<T, float>) {
        for (int l = 0; l < ne; ++l) ((float *)dst)[l] = vals[l];
    } else { static_assert(std::is_same_v<T, void>, "bad type"); }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO2_0) {
        return vec_dot_fattn_vec_KQ_turbo2_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO3_0) {
        return vec_dot_fattn_vec_KQ_turbo3_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_TURBO4_0) {
        return vec_dot_fattn_vec_KQ_turbo4_0<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO2_0) {
        return dequantize_V_turbo2_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO3_0) {
        return dequantize_V_turbo3_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_TURBO4_0) {
        return dequantize_V_turbo4_0<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}


template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * __restrict__ dst,
        const float2 * __restrict__ dst_fixup,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            V_f16.alloc(ggml_nelements(V));
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16.ptr;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16.ptr;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int ntiles_KV = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_dst + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc) || tiles_efficiency_percent < 75;

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if ((int)blocks_num.x % ntiles_dst == 0 && (int)blocks_num.x > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = (int)blocks_num.x;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % blocks_num.x != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
