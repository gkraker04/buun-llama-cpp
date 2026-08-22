// debug_gguf.cpp — dump raw tensor bytes from two GGUFs for comparison
#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <f16.gguf> <quant.gguf> <tensor-name>\n", argv[0]); return 1; }
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context * f16_ctx = gguf_init_from_file(argv[1], params);
    struct gguf_context * q_ctx   = gguf_init_from_file(argv[2], params);
    if (!f16_ctx || !q_ctx) { fprintf(stderr, "open failed\n"); return 1; }

    const char * name = argv[3];
    int if16 = gguf_find_tensor(f16_ctx, name);
    int iq   = gguf_find_tensor(q_ctx, name);
    fprintf(stderr, "f16 idx=%d q idx=%d\n", if16, iq);
    if (if16 < 0 || iq < 0) return 1;

    fprintf(stderr, "f16 type=%s off=%zu size=%zu\n",
        ggml_type_name(gguf_get_tensor_type(f16_ctx, if16)),
        gguf_get_tensor_offset(f16_ctx, if16),
        gguf_get_tensor_size(f16_ctx, if16));
    fprintf(stderr, "q   type=%s off=%zu size=%zu\n",
        ggml_type_name(gguf_get_tensor_type(q_ctx, iq)),
        gguf_get_tensor_offset(q_ctx, iq),
        gguf_get_tensor_size(q_ctx, iq));
    fprintf(stderr, "f16 data_offset=%zu q data_offset=%zu\n",
        gguf_get_data_offset(f16_ctx), gguf_get_data_offset(q_ctx));
    const int64_t * ne = gguf_get_tensor_ne(f16_ctx, if16);
    int64_t nelem = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) nelem *= ne[d];
    fprintf(stderr, "nelem=%lld\n", (long long) nelem);

    // open files and read at data_offset + tensor_offset
    FILE * f16_f = fopen(argv[1], "rb");
    FILE * q_f   = fopen(argv[2], "rb");
    size_t f16_pos = gguf_get_data_offset(f16_ctx) + gguf_get_tensor_offset(f16_ctx, if16);
    size_t q_pos   = gguf_get_data_offset(q_ctx)   + gguf_get_tensor_offset(q_ctx, iq);
    fprintf(stderr, "f16 abs pos=%zu q abs pos=%zu\n", f16_pos, q_pos);

    size_t f16_nbytes = gguf_get_tensor_size(f16_ctx, if16);
    size_t q_nbytes   = gguf_get_tensor_size(q_ctx, iq);
    std::vector<uint8_t> f16_raw(f16_nbytes), q_raw(q_nbytes);
#ifdef _WIN32
    _fseeki64(f16_f, (__int64) f16_pos, SEEK_SET);
    _fseeki64(q_f,   (__int64) q_pos,   SEEK_SET);
#else
    fseeko(f16_f, (off_t) f16_pos, SEEK_SET);
    fseeko(q_f,   (off_t) q_pos,   SEEK_SET);
#endif
    size_t r1 = fread(f16_raw.data(), 1, f16_nbytes, f16_f);
    size_t r2 = fread(q_raw.data(), 1, q_nbytes, q_f);
    fprintf(stderr, "read %zu / %zu bytes\n", r1, r2);

    // print first 8 f32 values
    fprintf(stderr, "f16 first f32s:");
    if (gguf_get_tensor_type(f16_ctx, if16) == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) f16_raw.data();
        for (int j = 0; j < 8 && j < nelem; ++j) fprintf(stderr, " %g", (double) ggml_fp16_to_fp32(h[j]));
    } else if (gguf_get_tensor_type(f16_ctx, if16) == GGML_TYPE_F32) {
        const float * f = (const float *) f16_raw.data();
        for (int j = 0; j < 8 && j < nelem; ++j) fprintf(stderr, " %g", (double) f[j]);
    }
    fprintf(stderr, "\nq   first f32s:");
    if (gguf_get_tensor_type(q_ctx, iq) == GGML_TYPE_F32) {
        const float * f = (const float *) q_raw.data();
        for (int j = 0; j < 8 && j < nelem; ++j) fprintf(stderr, " %g", (double) f[j]);
    } else {
        std::vector<float> tmp(nelem);
        switch (gguf_get_tensor_type(q_ctx, iq)) {
            case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) q_raw.data(), tmp.data(), nelem); break;
            case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) q_raw.data(), tmp.data(), nelem); break;
            default: fprintf(stderr, " (unhandled %s)", ggml_type_name(gguf_get_tensor_type(q_ctx, iq))); break;
        }
        for (int j = 0; j < 8 && j < nelem; ++j) fprintf(stderr, " %g", (double) tmp[j]);
    }
    fprintf(stderr, "\n");

    gguf_free(f16_ctx); gguf_free(q_ctx);
    fclose(f16_f); fclose(q_f);
    return 0;
}
