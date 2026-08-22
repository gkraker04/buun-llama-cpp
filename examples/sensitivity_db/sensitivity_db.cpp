// sensitivity_db.cpp — build the paper's linear-method sensitivity database.
//
// For each tensor in an F16 GGUF, and each target type, quantize the F16
// weights in-memory with ggml_quantize_chunk (the exact same code the
// inference path uses), dequantize with dequantize_row_*, and emit
// per-tensor normalized reconstruction error:
//     e_l(b) = ||W - W_hat(b)||^2_F / ||W||^2_F
//
// Pure per-(tensor, bitwidth) measurement — no GGUF round-trip, so the
// llama-quantize upgrade ladder (output->Q6_K min etc.) cannot contaminate
// the database.
//
// Usage: llama-sensitivity-db <f16.gguf> [tensor-filter]
// Output: CSV rows "name nelem f16_nbytes q_type q_nbytes rel_err" per
// (tensor, type) for all supported K/Q types, then TOTAL row per type.

#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- dequantize dispatch (mirrors ggml's own row-dequant API) ----
static void dequantize_row(enum ggml_type type, const void * x, float * y, int64_t k) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(y, x, k * sizeof(float));
            break;
        case GGML_TYPE_F16:
            { const ggml_fp16_t * h = (const ggml_fp16_t *) x;
              for (int64_t i = 0; i < k; ++i) y[i] = ggml_fp16_to_fp32(h[i]); }
            break;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) x, y, k); break;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K *) x, y, k); break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) x, y, k); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) x, y, k); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) x, y, k); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) x, y, k); break;
        default:
            fprintf(stderr, "unsupported quant type %s\n", ggml_type_name(type));
            exit(1);
    }
}

static bool type_supported(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <f16.gguf> [tensor-filter]\n", argv[0]); return 1; }

    const char * f16_path = argv[1];
    const char * filter   = argc > 2 ? argv[2] : nullptr;

    struct gguf_init_params params = { false, nullptr };
    struct gguf_context * f16_ctx = gguf_init_from_file(f16_path, params);
    if (!f16_ctx) { fprintf(stderr, "failed to open %s\n", f16_path); return 1; }

    const int n_tensors = gguf_get_n_tensors(f16_ctx);
    fprintf(stderr, "tensors: %d\n", n_tensors);

    // target types in bitwidth order
    const enum ggml_type target_types[] = {
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_0,
    };
    const int n_types = sizeof(target_types) / sizeof(target_types[0]);

    FILE * out = stdout;
    fprintf(out, "%-38s %12s %10s %8s %10s %12s\n", "name", "nelem", "f16_nbytes", "q_type", "q_nbytes", "rel_err");

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(f16_ctx, i);
        if (filter && !strstr(name, filter)) continue;

        enum ggml_type f16_type = gguf_get_tensor_type(f16_ctx, i);
        if (!type_supported(f16_type)) continue;

        const size_t f16_off = gguf_get_tensor_offset(f16_ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(f16_ctx, i);
        int64_t nelem = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) nelem *= ne[d];
        const size_t f16_nbytes = ggml_row_size(f16_type, nelem);

        // read reference tensor (f16 file, 64-bit seek)
        std::vector<uint8_t> f16_raw(f16_nbytes);
        {
            FILE * f = fopen(f16_path, "rb");
            if (!f) { fprintf(stderr, "fopen failed %s\n", f16_path); return 1; }
#ifdef _WIN32
            _fseeki64(f, (__int64) (gguf_get_data_offset(f16_ctx) + f16_off), SEEK_SET);
#else
            fseeko(f, (off_t) (gguf_get_data_offset(f16_ctx) + f16_off), SEEK_SET);
#endif
            if (fread(f16_raw.data(), 1, f16_nbytes, f) != f16_nbytes) { fprintf(stderr, "fread failed %s\n", name); fclose(f); continue; }
            fclose(f);
        }

        // reference f32
        std::vector<float> w_f32(nelem);
        dequantize_row(f16_type, f16_raw.data(), w_f32.data(), nelem);

        // reference norm
        double norm = 0.0;
        for (int64_t j = 0; j < nelem; ++j) norm += (double) w_f32[j] * (double) w_f32[j];

        for (int ti = 0; ti < n_types; ++ti) {
            enum ggml_type q_type = target_types[ti];

            // skip if same type as reference (F16 model has F32 tensors; they're never quantized)
            if (f16_type == q_type) continue;

            const size_t q_nbytes = ggml_row_size(q_type, nelem);
            std::vector<uint8_t> q_raw(q_nbytes);
            std::vector<float> wq_f32(nelem);

            // in-memory quantize: exact same code path as llama-quantize
            size_t written = ggml_quantize_chunk(q_type, w_f32.data(), q_raw.data(), 0, nelem / ggml_blck_size(q_type), ggml_blck_size(q_type), nullptr);
            (void) written;

            dequantize_row(q_type, q_raw.data(), wq_f32.data(), nelem);

            double ssd = 0.0;
            for (int64_t j = 0; j < nelem; ++j) {
                double d = (double) w_f32[j] - (double) wq_f32[j];
                ssd += d * d;
            }
            double rel = norm > 0.0 ? ssd / norm : 0.0;
            fprintf(out, "%-38s %12lld %10zu %8s %10zu %12.6e\n",
                name, (long long) nelem, f16_nbytes, ggml_type_name(q_type), q_nbytes, rel);
        }
    }

    gguf_free(f16_ctx);
    return 0;
}
