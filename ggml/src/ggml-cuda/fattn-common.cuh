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

// MSE reduction: 50.1% vs Lloyd-Max 3-bit, +3.02 dB. numpy GLA: n_train=4000, 100 iters, seed=99. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook_fattn[512] = {
    -0.24244059f, -0.12586778f, -0.06693592f, -0.02260770f, +0.01492950f, +0.05467265f, +0.10069778f, +0.18883320f,
    -0.19693744f, -0.14152811f, -0.09539399f, -0.06046141f, -0.02731707f, +0.01163860f, +0.05423523f, +0.11278591f,
    -0.11856443f, -0.06727399f, -0.02913110f, +0.00417571f, +0.03549468f, +0.07371171f, +0.11926779f, +0.18401266f,
    -0.25362726f, -0.15759121f, -0.10456934f, -0.06284792f, -0.01789622f, +0.03435958f, +0.08292559f, +0.14658904f,
    -0.16766223f, -0.09932603f, -0.04795861f, -0.00316137f, +0.03350896f, +0.07203513f, +0.12375449f, +0.24558071f,
    -0.21340639f, -0.11273975f, -0.05969454f, -0.02112451f, +0.01584557f, +0.05606037f, +0.09979239f, +0.18008010f,
    -0.14495838f, -0.08746232f, -0.05134764f, -0.02051995f, +0.00527687f, +0.03116450f, +0.06451037f, +0.11952747f,
    -0.20422562f, -0.11092815f, -0.05362599f, -0.00892124f, +0.02997769f, +0.07779223f, +0.13904265f, +0.22305706f,
    -0.17846867f, -0.11931835f, -0.08258449f, -0.04965282f, -0.01742237f, +0.01840734f, +0.06431029f, +0.13633489f,
    -0.16413829f, -0.09193468f, -0.05326458f, -0.01892892f, +0.02155236f, +0.07144441f, +0.12140869f, +0.16414391f,
    -0.16097247f, -0.11795594f, -0.07439681f, -0.03746189f, -0.00202306f, +0.03708065f, +0.07964572f, +0.25785580f,
    -0.21079398f, -0.08501953f, -0.04986830f, -0.02919976f, -0.00307999f, +0.01172143f, +0.04960671f, +0.10403022f,
    -0.13488377f, -0.08804465f, -0.05803939f, -0.02886309f, -0.00121364f, +0.03075502f, +0.07380433f, +0.14234027f,
    -0.17733476f, -0.11930768f, -0.08073044f, -0.05102654f, -0.02174008f, +0.01207697f, +0.05188434f, +0.10528153f,
    -0.27424359f, -0.14814219f, -0.09042648f, -0.04750653f, -0.00688004f, +0.03821837f, +0.08375420f, +0.15444848f,
    -0.17072668f, -0.11573062f, -0.07891619f, -0.04997802f, -0.02360069f, +0.00884181f, +0.04775132f, +0.09702020f,
    -0.13396922f, -0.08187833f, -0.03989934f, -0.00285008f, +0.03193579f, +0.06714261f, +0.10630646f, +0.19974332f,
    -0.10977794f, -0.05588607f, -0.01988312f, +0.00588292f, +0.02463065f, +0.04931722f, +0.08140395f, +0.11857282f,
    -0.11285258f, -0.06842930f, -0.03478571f, +0.00135103f, +0.04282236f, +0.08846240f, +0.14403294f, +0.18865710f,
    -0.16570802f, -0.13114756f, -0.08916780f, -0.01495983f, +0.02156897f, +0.05788230f, +0.10420620f, +0.15807896f,
    -0.09603385f, -0.05330852f, -0.01872682f, +0.01128407f, +0.04181543f, +0.07901397f, +0.12809893f, +0.20628030f,
    -0.12671234f, -0.07713382f, -0.04176560f, -0.01075576f, +0.02093381f, +0.05861618f, +0.10125964f, +0.16253341f,
    -0.12186792f, -0.07046833f, -0.02827389f, +0.00582941f, +0.03785510f, +0.07531738f, +0.13185618f, +0.20784822f,
    -0.11580890f, -0.06750206f, -0.03211596f, -0.00041264f, +0.02880913f, +0.06547855f, +0.11221221f, +0.17096693f,
    -0.20808545f, -0.15288957f, -0.09920800f, -0.05654906f, -0.02077297f, +0.01662349f, +0.06161885f, +0.11496038f,
    -0.25925224f, -0.12740968f, -0.07758909f, -0.03847224f, -0.00659505f, +0.02506258f, +0.05676728f, +0.15852313f,
    -0.20711072f, -0.15256361f, -0.09078260f, -0.04651003f, -0.01428200f, +0.02046691f, +0.06122406f, +0.11168941f,
    -0.29227489f, -0.10113064f, -0.06318919f, -0.04224788f, -0.01237292f, +0.01916771f, +0.05288843f, +0.08860565f,
    -0.18939137f, -0.13610712f, -0.07454449f, -0.03508454f, -0.00070383f, +0.03791586f, +0.07589655f, +0.12249952f,
    -0.23639095f, -0.16088664f, -0.10112434f, -0.05671202f, -0.02441828f, +0.01108337f, +0.04704943f, +0.08991196f,
    -0.18832129f, -0.10863860f, -0.06105676f, -0.02453765f, +0.00571738f, +0.03372482f, +0.06261604f, +0.10699216f,
    -0.22723058f, -0.15697967f, -0.09283338f, -0.04977757f, -0.00658221f, +0.03587560f, +0.07948679f, +0.13286804f,
    -0.10523734f, -0.05894853f, -0.01794997f, +0.01681460f, +0.05235468f, +0.08761997f, +0.12857652f, +0.27355651f,
    -0.16195214f, -0.08037403f, -0.03931148f, -0.00205822f, +0.03885316f, +0.08372425f, +0.14362726f, +0.19498548f,
    -0.11559617f, -0.06571012f, -0.02964597f, -0.00045770f, +0.02920656f, +0.06525015f, +0.11007642f, +0.23265810f,
    -0.12326290f, -0.06516271f, -0.02775460f, +0.00911453f, +0.03682196f, +0.07574877f, +0.13758171f, +0.19163566f,
    -0.09889923f, -0.05620999f, -0.01514455f, +0.01793674f, +0.05562053f, +0.10430135f, +0.16772700f, +0.28700828f,
    -0.14117142f, -0.08234416f, -0.03966629f, -0.00272311f, +0.03102731f, +0.07227346f, +0.13315912f, +0.20565525f,
    -0.09793990f, -0.05264642f, -0.01436317f, +0.01968854f, +0.05324087f, +0.09480734f, +0.16667446f, +0.25740325f,
    -0.14365898f, -0.07946859f, -0.03025317f, +0.01447767f, +0.05407316f, +0.09543498f, +0.14146231f, +0.20799392f,
    -0.16657843f, -0.10643959f, -0.06051657f, -0.02209583f, +0.01260932f, +0.04745538f, +0.09038523f, +0.16133716f,
    -0.21383845f, -0.13881313f, -0.09221762f, -0.05544837f, -0.02178388f, +0.01677356f, +0.05674765f, +0.10728363f,
    -0.17472305f, -0.11292139f, -0.06834519f, -0.03219563f, +0.00094835f, +0.03451309f, +0.07811368f, +0.14950613f,
    -0.21735978f, -0.14172379f, -0.09016410f, -0.05325706f, -0.02099085f, +0.01431495f, +0.05746740f, +0.10986551f,
    -0.16108559f, -0.09852512f, -0.05524211f, -0.01762269f, +0.01394665f, +0.05029779f, +0.09104291f, +0.15619289f,
    -0.18963714f, -0.12396694f, -0.07575205f, -0.03500398f, -0.00238001f, +0.03088680f, +0.06744511f, +0.11232874f,
    -0.15580266f, -0.11168178f, -0.07526547f, -0.04145918f, -0.00974866f, +0.03212880f, +0.07638067f, +0.13532050f,
    -0.18869418f, -0.12704822f, -0.07090112f, -0.03539131f, -0.00940597f, +0.01779585f, +0.05332254f, +0.10070462f,
    -0.10802900f, -0.05559649f, -0.01134203f, +0.02766773f, +0.06135347f, +0.09766156f, +0.13701990f, +0.20283839f,
    -0.15064489f, -0.08763143f, -0.05088234f, -0.01813378f, +0.01489159f, +0.05492927f, +0.10086069f, +0.16056357f,
    -0.10806098f, -0.05308804f, -0.01607634f, +0.01716060f, +0.04692414f, +0.08323829f, +0.12591397f, +0.19397456f,
    -0.14325471f, -0.07795846f, -0.03858727f, -0.01405432f, +0.01490734f, +0.04949452f, +0.09137284f, +0.14710410f,
    -0.08884655f, -0.04006506f, +0.00188640f, +0.03342239f, +0.06599921f, +0.10063411f, +0.13860981f, +0.21032288f,
    -0.12697365f, -0.06556813f, -0.02598349f, +0.00936226f, +0.04159310f, +0.07441500f, +0.11134150f, +0.15887714f,
    -0.24867819f, -0.09192626f, -0.04786854f, -0.01128089f, +0.02160387f, +0.05966909f, +0.10629620f, +0.19008696f,
    -0.10229637f, -0.05171858f, -0.01251010f, +0.01547078f, +0.03557770f, +0.06102134f, +0.09990518f, +0.15643583f,
    -0.22377374f, -0.14769778f, -0.08583978f, -0.04202831f, -0.00770624f, +0.02992121f, +0.07128810f, +0.12562713f,
    -0.11691130f, -0.05988247f, -0.02030350f, +0.01373520f, +0.04911441f, +0.09225391f, +0.15575270f, +0.23753035f,
    -0.18158022f, -0.09763482f, -0.05659763f, -0.02414636f, +0.00537057f, +0.03960274f, +0.07579990f, +0.12346489f,
    -0.26066830f, -0.12511689f, -0.06579913f, -0.00571045f, +0.03891529f, +0.08188405f, +0.12671439f, +0.18494135f,
    -0.15464623f, -0.08975428f, -0.04408587f, -0.01101469f, +0.02199709f, +0.05924839f, +0.10465596f, +0.26287255f,
    -0.19226125f, -0.11309006f, -0.07365324f, -0.03543059f, -0.00178878f, +0.03501295f, +0.07791925f, +0.14937145f,
    -0.12649273f, -0.06018269f, -0.01573098f, +0.02200219f, +0.05903495f, +0.09840808f, +0.13520581f, +0.18245036f,
    -0.16474872f, -0.09278035f, -0.04699890f, -0.00779894f, +0.03187623f, +0.07828258f, +0.13561429f, +0.23917313f
};

// MSE reduction: 33.1% vs Lloyd-Max 2-bit, +1.75 dB. numpy GLA: n_train=4000, 100 iters, 5 restarts. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turbo2_tcq_codebook_fattn[256] = {
    -0.08176727f, -0.00033508f, +0.06850938f, +0.16613583f, -0.14090237f, -0.05715980f, +0.01615283f, +0.11012612f,
    -0.10581727f, -0.04260033f, -0.00423828f, +0.06296677f, -0.17352516f, -0.07213694f, +0.02485547f, +0.10813029f,
    -0.12736021f, -0.06026637f, +0.00177779f, +0.06987048f, -0.08498892f, -0.01943354f, +0.06211906f, +0.01397950f,
    -0.17903381f, -0.01989968f, +0.03569642f, +0.09051796f, -0.09042171f, -0.02577177f, +0.02050355f, +0.10467158f,
    -0.21265116f, -0.11087410f, -0.04349163f, +0.01669601f, -0.12012258f, -0.01521601f, +0.07030928f, +0.13750617f,
    -0.06601736f, -0.04198077f, +0.02279012f, +0.10377382f, -0.07896508f, -0.00657534f, +0.06652649f, +0.17177304f,
    -0.07452555f, -0.00981928f, +0.04254026f, +0.11680857f, -0.12769225f, -0.04400226f, +0.01111500f, +0.08063783f,
    -0.05339707f, +0.01173677f, +0.07039803f, +0.14338760f, -0.12492259f, -0.05478338f, -0.01731757f, +0.04320757f,
    -0.00530445f, -0.15542837f, -0.06801344f, +0.04485723f, -0.07050634f, +0.01234248f, +0.11757696f, +0.22165567f,
    -0.01849510f, +0.04277446f, +0.08655161f, +0.15533215f, -0.10084474f, -0.00810490f, -0.03715962f, +0.04786975f,
    -0.02117090f, +0.04766359f, +0.08838871f, +0.16277327f, -0.24295192f, -0.12420259f, -0.05557786f, +0.12114887f,
    -0.12861997f, -0.06805481f, -0.05590313f, +0.01283404f, -0.01349204f, +0.05466014f, +0.10226475f, +0.19152307f,
    -0.09299547f, -0.02196216f, +0.03284279f, +0.09021873f, -0.07505369f, +0.08066312f, -0.03999974f, +0.04350512f,
    +0.00485651f, +0.05240202f, +0.12679257f, +0.19781399f, -0.18016882f, -0.11454904f, -0.06387294f, +0.01354196f,
    -0.17339253f, -0.10154387f, -0.03942726f, +0.03053090f, -0.01029367f, +0.05617156f, +0.10911176f, +0.18613949f,
    -0.21304886f, -0.11837386f, -0.06452254f, +0.01450099f, -0.03497068f, +0.03907030f, +0.06927501f, +0.13114283f,
    -0.15195946f, -0.06528903f, +0.00816301f, +0.09342197f, -0.00768985f, +0.08454979f, -0.06193831f, +0.04520382f,
    -0.18858465f, -0.12311971f, -0.08049614f, +0.00820490f, -0.03343302f, +0.04559230f, +0.09504822f, +0.16720207f,
    -0.08559455f, -0.00763808f, -0.07567421f, +0.03534968f, -0.03516657f, +0.07333340f, +0.00215530f, +0.06659426f,
    -0.02403073f, +0.04535064f, +0.10581165f, +0.14817812f, -0.16961506f, -0.10086726f, -0.04851092f, +0.02657260f,
    -0.03184498f, +0.03237205f, +0.09189106f, +0.14247570f, -0.18240723f, -0.09515552f, +0.01455373f, +0.24037592f,
    -0.13847726f, -0.10706620f, -0.04225504f, +0.02279146f, -0.02027496f, +0.06288219f, +0.14652734f, +0.24736365f,
    -0.01184501f, +0.06392768f, +0.12518647f, +0.20364036f, -0.06881002f, -0.14446024f, -0.04796625f, +0.02247028f,
    -0.11420977f, -0.03750149f, +0.03140424f, +0.10375965f, -0.15867621f, -0.07792078f, -0.00786463f, +0.07086110f,
    -0.05512634f, +0.01544903f, +0.08794563f, +0.18253894f, -0.12583706f, -0.04047658f, +0.03500937f, +0.12212106f,
    -0.07983117f, -0.02346017f, +0.02269844f, +0.09270003f, -0.14228862f, -0.05948335f, +0.01340374f, +0.08643699f,
    -0.17088441f, -0.08146483f, +0.01637994f, +0.11269872f, -0.12229883f, -0.02740963f, +0.06919862f, +0.17516392f,
    -0.23416011f, -0.08861073f, -0.00531799f, +0.04334467f, -0.07542395f, -0.00959691f, +0.03128058f, +0.11384328f,
    -0.12321154f, -0.05411436f, -0.00802293f, +0.04527715f, -0.02979034f, +0.01261100f, +0.08631871f, +0.14489119f,
    -0.06713610f, -0.01768748f, +0.04439952f, +0.08539781f, -0.10447017f, -0.03861764f, +0.01176727f, +0.08397588f,
    -0.09664737f, -0.03306058f, +0.01965956f, +0.08313737f, -0.15701702f, -0.03552708f, +0.03436711f, +0.12348684f,
    -0.07465987f, +0.03148096f, -0.01592258f, +0.07807118f, -0.08365041f, -0.00777653f, +0.06189138f, +0.16461129f
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
