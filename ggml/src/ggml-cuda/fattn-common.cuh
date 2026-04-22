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

// GLA-trained free-init TCQ codebook for fattn compilation unit (same values as turbo-quant-cuda.cuh)
static __constant__ float d_turbo3_tcq_codebook_fattn[512] = {
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

// GLA-trained free-init 2-bit TCQ codebook for fattn compilation unit (same values as turbo-quant-cuda.cuh)
static __constant__ float d_turbo2_tcq_codebook_fattn[256] = {
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
