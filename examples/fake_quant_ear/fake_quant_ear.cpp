// fake_quant_ear.cpp — in-memory fake-quantization engine for Algorithm-1
// Multi-Bitwidth Shapley Estimation (SLQ, arXiv 2605.02404).
//
// Loads an F16 GGUF once, applies candidate quantization recipes to the live
// model tensors in RAM (quantize -> dequantize round-trip, no GGUF ever
// written), runs the perplexity-style forward pass over a calibration file,
// and writes logits in the exact binary format llama-perplexity's
// --logits-file uses (readable by tools/perplexity --kl-divergence and by
// _patches/ear_aggregate.py).
//
// The model stays F16-typed end to end; only the VALUES of the tensors named
// by a recipe are replaced with their quantize(dequantize(w)) reconstructions.
// This is standard PTQ fake-quantization and is exactly what Algorithm 1
// needs: one model load, many candidate evaluations.
//
// Modes:
//   A. single candidate: --recipe recipe.txt --logits-file out.logits
//   B. plan (Algorithm-1 walk): --plan plan.json --outdir dir
//
// Usage:
//   llama-fake-quant-ear -m model-f16.gguf -f calib.txt -c 4096 -t 6 \
//       --recipe ffn_up=Q4_K --logits-file step.logits
//   llama-fake-quant-ear -m model-f16.gguf -f calib.txt -c 4096 -t 6 \
//       --plan plan.json --outdir logits/ [--keep-f16-copy] [--imatrix imatrix.gguf]
//
// Recipe file format (whitespace-separated NAME=TYPE tokens):
//   ^blk\.0\.attn_gate\.weight$=Q4_K      per-tensor regex (llama-quantize
//                                         --tensor-type-file convention)
//   ffn_gate=Q4_K                         class token -> all tensors whose
//                                         name contains the token as a
//                                         delimited word (14 known classes)
//
// Plan file format (JSON array of step objects):
//   [ { "id": "p1_b4_step3",
//       "recipe": [ {"class": "ffn_up", "type": "Q4_K"},
//                   {"tensor": "^blk\\.0\\..*\\.weight$", "type": "Q5_K"} ],
//       "logits": "step_0003.logits" },
//     { "id": "p1_b4_step4",
//       "switch": { "class": "attn_qkv", "from": "Q8_0", "to": "Q4_K" },
//       "logits": "step_0004.logits" } ]
//
// "switch" steps are relative to the previous step's state (the natural
// Algorithm-1 walk: one group flips per step) and verify the "from" type.

#include "llama.h"
#include "llama-model.h" // llama_internal_get_tensor_map (internal, src/)

#include "arg.h"
#include "common.h"
#include "imatrix-loader.h"
#include "log.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h" // dequantize_row_* / block types (internal, ggml/src/)

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::ordered_json;

static const char * const CLASS_TOKENS[] = {
    "ffn_gate", "ffn_up", "ffn_down",
    "ssm_alpha", "ssm_beta", "ssm_out",
    "attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v", "attn_output",
    "output", "token_embd",
};

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------

static bool striequals(const char * a, const char * b) {
    while (*a && *b) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == 0 && *b == 0;
}

static ggml_type ggml_type_from_name(const std::string & name) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && striequals(n, name.c_str())) {
            return (ggml_type) i;
        }
    }
    return GGML_TYPE_COUNT;
}

// target types the tool knows how to fake-quantize. The list mirrors the
// fork's ggml_quantize_chunk weight palette restricted to the 2-8 bpw band
// (TQ deferred by the user; Bonsai Q2_0/Q2_0_G128 and the fp4 exotics
// MXFP4/NVFP4 excluded; the TURBO* KV-cache types are not weights).
// F16/F32/BF16 are the "restore/pristine or float" targets (they must match
// the tensor's original type when used as restore); BF16 is also a valid
// fake-quant target for other models.
static bool target_type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) ++b;
    while (e > b && std::isspace((unsigned char) s[e-1])) --e;
    return s.substr(b, e - b);
}

static bool is_class_token(const std::string & s) {
    for (const char * ct : CLASS_TOKENS) {
        if (s == ct) {
            return true;
        }
    }
    return false;
}

// class tokens match tensor names as delimited words: the character before
// and after the token must not be [A-Za-z0-9_]. This keeps "attn_q" from
// matching "blk.0.attn_qkv.weight" (followed by 'k') while still matching
// "blk.0.attn_q.weight".
static bool is_name_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

static bool contains_class_token(const std::string & name, const std::string & tok) {
    if (tok.empty()) {
        return false;
    }
    for (size_t p = 0; p + tok.size() <= name.size(); ++p) {
        if (name.compare(p, tok.size(), tok) != 0) {
            continue;
        }
        const bool prev_ok = p == 0 || !is_name_char(name[p-1]);
        const bool next_ok = p + tok.size() == name.size() || !is_name_char(name[p + tok.size()]);
        if (prev_ok && next_ok) {
            return true;
        }
    }
    return false;
}

// dequantize dispatch: mirrors the fork's row-dequant API for the 2-8 bpw
// weight palette.
static void dequantize_row(enum ggml_type type, const void * x, float * y, int64_t k) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(y, x, k * sizeof(float));
            break;
        case GGML_TYPE_F16:
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) x, y, k);
            break;
        case GGML_TYPE_BF16:    ggml_bf16_to_fp32_row((const ggml_bf16_t *) x, y, k); break;
        case GGML_TYPE_Q4_0:    dequantize_row_q4_0    ((const block_q4_0 *)     x, y, k); break;
        case GGML_TYPE_Q4_1:    dequantize_row_q4_1    ((const block_q4_1 *)     x, y, k); break;
        case GGML_TYPE_Q5_0:    dequantize_row_q5_0    ((const block_q5_0 *)     x, y, k); break;
        case GGML_TYPE_Q5_1:    dequantize_row_q5_1    ((const block_q5_1 *)     x, y, k); break;
        case GGML_TYPE_Q8_0:    dequantize_row_q8_0    ((const block_q8_0 *)     x, y, k); break;
        case GGML_TYPE_Q2_K:    dequantize_row_q2_K    ((const block_q2_K *)     x, y, k); break;
        case GGML_TYPE_Q3_K:    dequantize_row_q3_K    ((const block_q3_K *)     x, y, k); break;
        case GGML_TYPE_Q4_K:    dequantize_row_q4_K    ((const block_q4_K *)     x, y, k); break;
        case GGML_TYPE_Q5_K:    dequantize_row_q5_K    ((const block_q5_K *)     x, y, k); break;
        case GGML_TYPE_Q6_K:    dequantize_row_q6_K    ((const block_q6_K *)     x, y, k); break;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs ((const block_iq2_xxs *)  x, y, k); break;
        case GGML_TYPE_IQ2_XS:  dequantize_row_iq2_xs  ((const block_iq2_xs *)   x, y, k); break;
        case GGML_TYPE_IQ2_S:   dequantize_row_iq2_s   ((const block_iq2_s *)    x, y, k); break;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs ((const block_iq3_xxs *)  x, y, k); break;
        case GGML_TYPE_IQ3_S:   dequantize_row_iq3_s   ((const block_iq3_s *)    x, y, k); break;
        case GGML_TYPE_IQ4_NL:  dequantize_row_iq4_nl  ((const block_iq4_nl *)   x, y, k); break;
        case GGML_TYPE_IQ4_XS:  dequantize_row_iq4_xs  ((const block_iq4_xs *)   x, y, k); break;
        default:
            GGML_ASSERT(false && "unsupported type in dequantize dispatch");
    }
}

// ---------------------------------------------------------------------------
// logits writer — byte-identical to tools/perplexity/perplexity.cpp
// (--logits-file path). Format: "_logits_" magic, uint32 n_ctx, int n_vocab,
// int n_chunk, int32 tokens[n_chunk*n_ctx], then per chunk a block of
// uint16 log-probabilities (nv = 2*((n_vocab+1)/2)+4 per position).
// ---------------------------------------------------------------------------

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *) log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}

static void process_logits(std::ostream& out, int n_vocab, const float * logits, const int * tokens, int n_token,
        std::vector<std::thread> & workers, std::vector<uint16_t> & log_probs, double & nll, double & nll2) {
    std::mutex mutex;
    const int nv = 2*((n_vocab + 1)/2) + 4;
    int counter = 0;
    auto compute = [&mutex, &counter, &log_probs, &nll, &nll2, n_vocab, logits, tokens, n_token, nv] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const double v = log_softmax(n_vocab, logits + size_t(i)*n_vocab, log_probs.data() + size_t(i)*nv, tokens[i+1]);
            local_nll += v;
            local_nll2 += v*v;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
    out.write((const char *) log_probs.data(), size_t(n_token)*nv*sizeof(uint16_t));
}

// ---------------------------------------------------------------------------
// live-model tensor state
// ---------------------------------------------------------------------------

struct tensor_state {
    std::string name;
    ggml_tensor * t       = nullptr;
    ggml_type orig        = GGML_TYPE_COUNT; // type as loaded (F16, F32 or BF16)
    ggml_type cur         = GGML_TYPE_COUNT; // effective target type; == orig when pristine
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    int64_t nelem         = 0;
    size_t  nbytes        = 0;
    size_t  file_off      = 0;               // gguf data_offset + tensor offset in the F16 source
    std::vector<uint8_t> pristine;           // only populated with --keep-f16-copy
};

struct model_state {
    std::vector<tensor_state> tensors;
    std::unordered_map<std::string, std::vector<size_t>> class_tensors; // longest-token-wins
    bool keep_f16_copy = false;
    std::string src_path;
};

// register the live F16/F32/BF16 tensors of the loaded model and compute the
// class -> tensor expansion. Uses llama_internal_get_tensor_map (internal
// API, src/llama-model.h) — the same tensor objects the compute graph
// references, so in-place mutation of ->data is picked up by llama_decode.
static bool build_model_state(model_state & st, const llama_model * model, const std::string & src_path, bool keep_f16_copy) {
    st.src_path = src_path;
    st.keep_f16_copy = keep_f16_copy;

    const auto & tmap = llama_internal_get_tensor_map(model);
    int n_skipped = 0;
    for (const auto & [name, t] : tmap) {
        if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16) {
            ++n_skipped;
            continue;
        }
        tensor_state e;
        e.name   = name;
        e.t      = t;
        e.orig   = t->type;
        e.cur    = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            e.ne[d] = t->ne[d];
        }
        e.nelem  = ggml_nelements(t);
        e.nbytes = ggml_nbytes(t);
        st.tensors.push_back(std::move(e));
    }
    if (n_skipped > 0) {
        LOG_WRN("%s: skipped %d non-F16/F32/BF16 tensors (not fake-quantizable)", __func__, n_skipped);
    }
    if (st.tensors.empty()) {
        LOG_ERR("%s: no F16/F32/BF16 tensors found in the model", __func__);
        return false;
    }

    // class expansion with longest-token-wins disambiguation: e.g. both
    // "attn_output" and "output" match "blk.0.attn_output.weight", and the
    // more specific token wins; "output.weight" keeps class "output".
    for (size_t i = 0; i < st.tensors.size(); ++i) {
        const std::string & name = st.tensors[i].name;
        std::string best;
        for (const char * ct : CLASS_TOKENS) {
            if (contains_class_token(name, ct) && best.size() < strlen(ct)) {
                best = ct;
            }
        }
        if (!best.empty()) {
            st.class_tensors[best].push_back(i);
        }
    }

    if (keep_f16_copy) {
        double gb = 0.0;
        for (auto & e : st.tensors) {
            e.pristine.assign((const uint8_t *) e.t->data, (const uint8_t *) e.t->data + e.nbytes);
            gb += e.nbytes;
        }
        LOG_INF("%s: kept pristine copy of %zu tensors (%.1f GiB) for reset between steps",
                __func__, st.tensors.size(), gb/1024.0/1024.0/1024.0);
    } else {
        // metadata-only gguf read (no_alloc) to resolve per-tensor file offsets
        struct gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
        struct gguf_context * g = gguf_init_from_file(src_path.c_str(), ip);
        if (!g) {
            LOG_ERR("%s: failed to open %s for pristine reads", __func__, src_path.c_str());
            return false;
        }
        const size_t data_off = gguf_get_data_offset(g);
        std::unordered_map<std::string, size_t> offs;
        const int n = gguf_get_n_tensors(g);
        for (int i = 0; i < n; ++i) {
            offs.emplace(gguf_get_tensor_name(g, i), data_off + gguf_get_tensor_offset(g, i));
        }
        for (auto & e : st.tensors) {
            auto it = offs.find(e.name);
            if (it == offs.end()) {
                LOG_ERR("%s: tensor %s not found in source gguf", __func__, e.name.c_str());
                gguf_free(g);
                return false;
            }
            e.file_off = it->second;
        }
        gguf_free(g);
        LOG_INF("%s: pristine tensor data will be re-read from %s per step (pass --keep-f16-copy to keep it in RAM)",
                __func__, src_path.c_str());
    }
    return true;
}

// read the pristine (F16/F32/BF16) bytes of a tensor: RAM copy if available,
// otherwise a seek+read from the F16 source file. Per-call fopen keeps this
// thread-safe for the OpenMP loop.
static bool read_pristine_bytes(const tensor_state & e, const model_state & st, std::vector<uint8_t> & out) {
    if (st.keep_f16_copy) {
        out = e.pristine;
        return true;
    }
    out.resize(e.nbytes);
    FILE * f = fopen(st.src_path.c_str(), "rb");
    if (!f) {
        return false;
    }
#ifdef _WIN32
    const int rc = _fseeki64(f, (__int64) e.file_off, SEEK_SET);
#else
    const int rc = fseeko(f, (off_t) e.file_off, SEEK_SET);
#endif
    const bool ok = rc == 0 && fread(out.data(), 1, e.nbytes, f) == e.nbytes;
    fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// imatrix (for IQ4_XS and friends)
// ---------------------------------------------------------------------------

struct imatrix_data {
    bool loaded = false;
    std::unordered_map<std::string, std::vector<float>> data; // normalized per quantize.cpp
    mutable std::set<std::string> warned;

    // returns nullptr when the imatrix is unavailable for this tensor.
    // expected = ne[0]*ne[2] floats per llama-quant's size contract.
    const float * lookup(const std::string & name, int64_t expected) const {
        if (!loaded) {
            return nullptr;
        }
        auto it = data.find(name);
        if (it == data.end()) {
            if (warned.insert(name).second) {
                LOG_WRN("%s: no imatrix entry for %s (quantizing unweighted)", __func__, name.c_str());
            }
            return nullptr;
        }
        if ((int64_t) it->second.size() != expected) {
            if (warned.insert(name).second) {
                LOG_WRN("%s: imatrix size %d != expected %lld for %s (quantizing unweighted)",
                        __func__, (int) it->second.size(), (long long) expected, name.c_str());
            }
            return nullptr;
        }
        return it->second.data();
    }
};

// normalization mirrors tools/quantize/quantize.cpp load_imatrix:
// GGUF format divides the per-expert sums by their counts; legacy format
// divides by the single ncall.
static bool load_imatrix_data(const std::string & path, imatrix_data & out) {
    common_imatrix loaded;
    if (!common_imatrix_load(path, loaded)) {
        LOG_ERR("%s: failed to load imatrix from '%s'", __func__, path.c_str());
        return false;
    }
    if (!loaded.is_legacy && !loaded.has_metadata) {
        LOG_ERR("%s: missing imatrix metadata in file %s", __func__, path.c_str());
        return false;
    }
    for (const auto & [name, entry] : loaded.entries) {
        auto & e = out.data[name];
        e.resize(entry.sums.size());
        if (!loaded.is_legacy) {
            const int64_t ncounts = (int64_t) entry.counts.size();
            const int64_t ne0 = ncounts > 0 ? (int64_t) entry.sums.size() / ncounts : 0;
            for (int64_t j = 0; j < ncounts; ++j) {
                const float count = (float) entry.counts[j];
                if (count > 0.0f) {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = entry.sums[j*ne0 + i] / count;
                    }
                } else {
                    for (int64_t i = 0; i < ne0; ++i) {
                        e[j*ne0 + i] = 1;
                    }
                }
            }
        } else {
            const int64_t ncall = entry.counts.empty() ? 0 : entry.counts[0];
            if (ncall > 0) {
                for (size_t i = 0; i < entry.sums.size(); ++i) {
                    e[i] = entry.sums[i] / (float) ncall;
                }
            }
        }
    }
    out.loaded = true;
    LOG_INF("%s: loaded %d importance matrix entries from %s (%d chunks)",
            __func__, (int) out.data.size(), path.c_str(), loaded.chunk_count);
    return true;
}

// ---------------------------------------------------------------------------
// fake-quantization
// ---------------------------------------------------------------------------

struct requant_task {
    size_t idx;
    ggml_type target;
    const float * imatrix; // may be null
};

// Fake-quantize one tensor in place: read pristine bytes -> f32 -> quantize
// with ggml_quantize_chunk (the exact code path llama-quantize uses) ->
// dequantize -> write back into the live tensor's data buffer, converted to
// the tensor's original type (F16, F32 or BF16). The tensor TYPE is never changed.
static bool requantize_tensor(const tensor_state & e, const model_state & st, ggml_type target, const float * imatrix, std::string & err) {
    std::vector<uint8_t> file_buf;
    const uint8_t * raw = nullptr;
    if (st.keep_f16_copy) {
        raw = e.pristine.data();
    } else {
        if (!read_pristine_bytes(e, st, file_buf)) {
            err = "failed to read pristine data for " + e.name;
            return false;
        }
        raw = file_buf.data();
    }

    // restore targets (F16/F32/BF16): write the pristine bytes back. Only valid
    // when the target matches the tensor's original type (checked at parse).
    if (target == GGML_TYPE_F16 || target == GGML_TYPE_F32 || target == GGML_TYPE_BF16) {
        if (target != e.orig) {
            err = "tensor " + e.name + " is " + ggml_type_name(e.orig) + ", cannot restore to " + ggml_type_name(target);
            return false;
        }
        memcpy(e.t->data, raw, e.nbytes);
        return true;
    }

    const int64_t n_per_row = e.ne[0];
    const int64_t nrows     = e.ne[1];
    const int64_t n_planes  = e.ne[2]; // expert planes (ne[2]); dense weights have 1

    // the k/IQ quantizers require row lengths that are multiples of the
    // block size; refuse cleanly instead of tripping their GGML_ASSERT
    if (n_per_row % ggml_blck_size(target) != 0) {
        err = "tensor " + e.name + " has n_per_row " + std::to_string(n_per_row) +
              " which is not a multiple of " + std::to_string(ggml_blck_size(target)) +
              " (" + ggml_type_name(target) + "); refusing to quantize";
        return false;
    }

    std::vector<float> f32(e.nelem);
    dequantize_row(e.orig, raw, f32.data(), e.nelem);

    std::vector<uint8_t> q(ggml_row_size(target, e.nelem));

    // quantize each expert plane separately (imatrix advances by n_per_row
    // per plane — same layout llama-quant expects).
    for (int64_t i02 = 0; i02 < n_planes; ++i02) {
        const float * src = f32.data() + i02 * n_per_row * nrows;
        void * dst = q.data() + i02 * ggml_row_size(target, n_per_row) * nrows;
        const float * im = imatrix ? imatrix + i02 * n_per_row : nullptr;
        const size_t written = ggml_quantize_chunk(target, src, dst, 0, nrows, n_per_row, im);
        GGML_ASSERT(written == (size_t) nrows * ggml_row_size(target, n_per_row));
    }

    if (!ggml_validate_row_data(target, q.data(), q.size())) {
        err = "quantized data validation failed for " + e.name;
        return false;
    }

    dequantize_row(target, q.data(), f32.data(), e.nelem);

    if (e.orig == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f32.data(), (ggml_fp16_t *) e.t->data, e.nelem);
    } else if (e.orig == GGML_TYPE_BF16) {
        ggml_fp32_to_bf16_row(f32.data(), (ggml_bf16_t *) e.t->data, e.nelem);
    } else {
        memcpy(e.t->data, f32.data(), e.nbytes);
    }
    return true;
}

// ---------------------------------------------------------------------------
// recipes
// ---------------------------------------------------------------------------

struct recipe_entry {
    bool is_class = false;   // false => regex
    std::string key;         // class token or regex pattern
    std::regex re;
    ggml_type target = GGML_TYPE_COUNT;
};

struct recipe {
    std::vector<recipe_entry> entries;
};

static ggml_type parse_type_or_throw(const std::string & s) {
    const ggml_type t = ggml_type_from_name(s);
    if (t == GGML_TYPE_COUNT) {
        throw std::runtime_error("invalid quantization type '" + s + "'");
    }
    if (!target_type_supported(t)) {
        throw std::runtime_error("unsupported quantization type '" + s +
                "' (supported: F16 F32 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K "
                "Q6_K IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS)");
    }
    return t;
}

static void parse_recipe_token(const std::string & tok, recipe & rec) {
    const size_t eq = tok.find('=');
    if (eq == std::string::npos) {
        throw std::runtime_error("malformed recipe token '" + tok + "' (expected NAME=TYPE)");
    }
    const std::string name = trim(tok.substr(0, eq));
    const std::string type = trim(tok.substr(eq + 1));
    if (name.empty() || type.empty()) {
        throw std::runtime_error("malformed recipe token '" + tok + "' (expected NAME=TYPE)");
    }

    recipe_entry en;
    en.target = parse_type_or_throw(type);

    if (name[0] == '^') {
        // per-tensor regex (llama-quantize --tensor-type-file convention)
        en.is_class = false;
        en.key = name;
        try {
            en.re = std::regex(name);
        } catch (const std::regex_error & e) {
            throw std::runtime_error("invalid tensor regex '" + name + "': " + e.what());
        }
    } else {
        if (!is_class_token(name)) {
            throw std::runtime_error("unknown class name '" + name +
                    "' (expected one of the 14 classes: ffn_gate ffn_up ffn_down ssm_alpha ssm_beta ssm_out "
                    "attn_qkv attn_gate attn_q attn_k attn_v attn_output output token_embd; "
                    "use a ^regex$ pattern for per-tensor recipes)");
        }
        en.is_class = true;
        en.key = name;
    }
    rec.entries.push_back(std::move(en));
}

static recipe parse_recipe_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open recipe file '" + path + "'");
    }
    recipe rec;
    std::string tok;
    while (in >> tok) {
        parse_recipe_token(tok, rec);
    }
    if (rec.entries.empty()) {
        throw std::runtime_error("recipe file '" + path + "' contains no entries");
    }
    return rec;
}

static bool target_requires_imatrix(ggml_type t) {
    // mirrors llama-quant.cpp tensor_requires_imatrix: IQ4_XS (and IQ4_NL)
    // quantize without an imatrix in this fork; IQ2/3 family requires one.
    switch (t) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

static bool recipe_requires_imatrix(const recipe & rec) {
    for (const auto & en : rec.entries) {
        if (target_requires_imatrix(en.target)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// plan mode (Algorithm-1 batch)
// ---------------------------------------------------------------------------

struct plan_step {
    std::string id;
    std::string logits;
    bool is_switch = false;
    recipe rec;                        // absolute recipe (is_switch == false)
    std::string sw_class;
    ggml_type sw_from = GGML_TYPE_COUNT;
    ggml_type sw_to   = GGML_TYPE_COUNT;
};

static void json_to_recipe(const json & j, recipe & rec) {
    if (!j.is_array()) {
        throw std::runtime_error("plan step 'recipe' must be an array");
    }
    for (const auto & en : j) {
        recipe_entry e;
        if (en.contains("class")) {
            e.is_class = true;
            e.key = en.at("class").get<std::string>();
            if (!is_class_token(e.key)) {
                throw std::runtime_error("plan recipe class '" + e.key + "' is not a known class");
            }
        } else if (en.contains("tensor")) {
            e.key = en.at("tensor").get<std::string>();
            try {
                e.re = std::regex(e.key);
            } catch (const std::regex_error & re) {
                throw std::runtime_error("invalid plan recipe regex '" + e.key + "': " + re.what());
            }
        } else {
            throw std::runtime_error("plan recipe entries must have 'class' or 'tensor'");
        }
        e.target = parse_type_or_throw(en.at("type").get<std::string>());
        rec.entries.push_back(std::move(e));
    }
}

static std::vector<plan_step> parse_plan_file(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open plan file '" + path + "'");
    }
    json j;
    try {
        in >> j;
    } catch (const std::exception & e) {
        throw std::runtime_error("failed to parse plan JSON '" + path + "': " + e.what());
    }
    if (!j.is_array()) {
        throw std::runtime_error("plan must be a JSON array of step objects");
    }

    std::vector<plan_step> steps;
    for (const auto & s : j) {
        plan_step ps;
        ps.id     = s.value("id", "");
        ps.logits = s.at("logits").get<std::string>();
        if (s.contains("switch")) {
            const auto & sw = s.at("switch");
            ps.is_switch = true;
            ps.sw_class = sw.at("class").get<std::string>();
            if (!is_class_token(ps.sw_class)) {
                throw std::runtime_error("switch class '" + ps.sw_class + "' is not a known class");
            }
            ps.sw_from = parse_type_or_throw(sw.at("from").get<std::string>());
            ps.sw_to   = parse_type_or_throw(sw.at("to").get<std::string>());
        } else if (s.contains("recipe")) {
            json_to_recipe(s.at("recipe"), ps.rec);
            if (ps.rec.entries.empty()) {
                throw std::runtime_error("step '" + ps.id + "' has an empty recipe");
            }
        } else {
            throw std::runtime_error("step '" + ps.id + "' must have a 'recipe' or a 'switch'");
        }
        steps.push_back(std::move(ps));
    }
    if (steps.empty()) {
        throw std::runtime_error("plan file '" + path + "' contains no steps");
    }
    return steps;
}

// ---------------------------------------------------------------------------
// recipe application (the OpenMP-parallel quantize phase)
// ---------------------------------------------------------------------------

// Expand a recipe to the concrete list of tensors that need re-quantization
// (only those whose target type differs from the current state), then
// fake-quantize them in parallel.
static bool apply_recipe(const recipe & rec, model_state & st, const imatrix_data & imx) {
    std::vector<requant_task> tasks;

    for (const auto & en : rec.entries) {
        std::vector<size_t> idxs;
        if (en.is_class) {
            auto it = st.class_tensors.find(en.key);
            if (it == st.class_tensors.end()) {
                LOG_WRN("%s: class '%s' matched no tensors (is this the right model?)", __func__, en.key.c_str());
                continue;
            }
            idxs = it->second;
        } else {
            for (size_t i = 0; i < st.tensors.size(); ++i) {
                if (std::regex_search(st.tensors[i].name, en.re)) {
                    idxs.push_back(i);
                }
            }
            if (idxs.empty()) {
                LOG_WRN("%s: regex '%s' matched no tensors", __func__, en.key.c_str());
            }
        }

        for (size_t i : idxs) {
            tensor_state & e = st.tensors[i];
            if (e.cur == en.target) {
                continue; // no state change -> skip re-quantization
            }
            const float * im = imx.lookup(e.name, e.ne[0] * e.ne[2]);
            tasks.push_back({i, en.target, im});
        }
    }

    if (tasks.empty()) {
        LOG_INF("%s: recipe requires no tensor state change", __func__);
        return true;
    }
    int omp_threads = 1;
#ifdef _OPENMP
    omp_threads = omp_get_max_threads();
#endif
    LOG_INF("%s: re-quantizing %zu tensors (%d OpenMP threads)", __func__, tasks.size(), omp_threads);

    std::atomic<bool> failed{false};
    std::string err;
    std::mutex err_mu;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int ti = 0; ti < (int) tasks.size(); ++ti) {
        if (!failed.load(std::memory_order_relaxed)) {
            const requant_task & tk = tasks[ti];
            std::string local_err;
            if (!requantize_tensor(st.tensors[tk.idx], st, tk.target, tk.imatrix, local_err)) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(err_mu);
                if (err.empty()) {
                    err = local_err;
                }
            }
        }
    }

    if (failed) {
        LOG_ERR("%s: %s", __func__, err.c_str());
        return false;
    }
    for (const auto & tk : tasks) {
        st.tensors[tk.idx].cur = tk.target;
    }
    return true;
}

// relative step: switch one class from its current type to another,
// validating that the walk state matches "from".
static bool apply_switch(const plan_step & ps, model_state & st, const imatrix_data & imx) {
    auto it = st.class_tensors.find(ps.sw_class);
    if (it == st.class_tensors.end()) {
        LOG_ERR("%s: switch: class '%s' matched no tensors", __func__, ps.sw_class.c_str());
        return false;
    }
    for (size_t i : it->second) {
        if (st.tensors[i].cur != ps.sw_from) {
            LOG_ERR("%s: switch: tensor %s is at %s, expected %s (step %s)",
                    __func__, st.tensors[i].name.c_str(), ggml_type_name(st.tensors[i].cur),
                    ggml_type_name(ps.sw_from), ps.id.c_str());
            return false;
        }
    }
    recipe rec;
    recipe_entry en;
    en.is_class = true;
    en.key = ps.sw_class;
    en.target = ps.sw_to;
    rec.entries.push_back(std::move(en));
    return apply_recipe(rec, st, imx);
}

// ---------------------------------------------------------------------------
// forward pass + logits writer (mirrors tools/perplexity/perplexity.cpp)
// ---------------------------------------------------------------------------

static bool run_forward(llama_context * ctx, const common_params & params, const int32_t n_ctx,
        std::vector<llama_token> & tokens, const std::string & logits_path) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    GGML_ASSERT(!llama_vocab_get_add_eos(vocab));

    std::ofstream logits_stream;
    logits_stream.open(logits_path.c_str(), std::ios::binary);
    if (!logits_stream.is_open()) {
        LOG_ERR("%s: failed to open %s for writing", __func__, logits_path.c_str());
        return false;
    }
    LOG_INF("%s: saving all logits to %s", __func__, logits_path.c_str());
    logits_stream.write("_logits_", 8);
    logits_stream.write(reinterpret_cast<const char *>(&n_ctx), sizeof(n_ctx));

    if (int(tokens.size()) < 2*n_ctx) {
        LOG_ERR("%s: you need at least %d tokens to evaluate with a context of %d", __func__, 2*n_ctx, n_ctx);
        return false;
    }

    const int n_chunk_max = int(tokens.size()) / n_ctx;
    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_batch = params.n_batch;

    const int n_vocab = llama_vocab_n_tokens(vocab);

    int count = 0;
    double nll  = 0.0;
    double nll2 = 0.0;

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;
    const int n_seq = std::max(1, n_batch / n_ctx);

    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);
    GGML_ASSERT(params.n_ctx == n_seq * n_ctx);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx*n_seq), 0, 1);

    std::vector<float> logits;
    if (num_batches > 1) {
        logits.reserve(size_t(n_ctx) * n_vocab);
    }

    LOG_INF("%s: evaluating %d chunks, n_ctx=%d, batch_size=%d, n_seq=%d", __func__, n_chunk, n_ctx, n_batch, n_seq);

    const size_t n_workers = std::max(1u, std::thread::hardware_concurrency()) - 1;
    std::vector<std::thread> workers(n_workers);

    std::vector<uint16_t> log_probs;
    const int nv = 2*((n_vocab + 1)/2) + 4;

    // header: n_vocab, n_chunk and the full evaluation token block, exactly
    // as llama-perplexity writes it.
    logits_stream.write((const char *) &n_vocab, sizeof(n_vocab));
    logits_stream.write((const char *) &n_chunk, sizeof(n_chunk));
    logits_stream.write((const char *) tokens.data(), (size_t) n_chunk*n_ctx*sizeof(tokens[0]));
    log_probs.resize(size_t(n_ctx) * nv);

    // logits for the last half of each window are used (perplexity
    // convention), so the model always has some context to predict the token.
    const int first = n_ctx/2;

    for (int i = 0; i < n_chunk; i += n_seq) {
        const int start = i * n_ctx;
        const int end   = start + n_ctx;

        const int n_seq_batch = std::min(n_seq, n_chunk - i);

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            int n_outputs = 0;

            batch.n_tokens = 0;
            for (int seq = 0; seq < n_seq_batch; seq++) {
                int seq_start = batch_start + seq*n_ctx;

                // save original token and restore it after decode
                const auto token_org = tokens[seq_start];

                // add BOS token for the first batch of each chunk
                if (add_bos && j == 0) {
                    tokens[seq_start] = llama_vocab_bos(vocab);
                }

                for (int k = 0; k < batch_size; ++k) {
                    const int idx = seq*n_ctx + k;
                    batch.token   [idx]    = tokens[seq_start + k];
                    batch.pos     [idx]    = j*n_batch + k;
                    batch.n_seq_id[idx]    = 1;
                    batch.seq_id  [idx][0] = seq;
                    batch.logits  [idx]    = batch.pos[idx] >= first ? 1 : 0;

                    n_outputs += batch.logits[idx] != 0;
                }
                batch.n_tokens += batch_size;

                // restore the original token in case it was set to BOS
                tokens[seq_start] = token_org;
            }

            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: failed to decode", __func__);
                llama_batch_free(batch);
                return false;
            }

            if (num_batches > 1 && n_outputs > 0) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
            }
        }

        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int) (t_total*n_chunk/n_seq);
            if (total_seconds >= 60*60) {
                LOG("%d hours ", total_seconds / (60*60));
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        for (int seq = 0; seq < n_seq_batch; seq++) {
            const float * all_logits = num_batches > 1 ? logits.data() : llama_get_logits_ith(ctx, seq*n_ctx + first);

            llama_token * tokens_data = tokens.data() + start + seq*n_ctx + first;
            process_logits(logits_stream, n_vocab, all_logits, tokens_data, n_ctx - 1 - first,
                    workers, log_probs, nll, nll2);
            count += n_ctx - first - 1;

            LOG("[%d]%.4lf,", i + seq + 1, std::exp(nll / count));
        }

        logits.clear();
    }
    LOG("\n");

    llama_batch_free(batch);

    nll2 /= count;
    nll  /= count;
    const double ppl = exp(nll);
    nll2 -= nll * nll;
    if (nll2 > 0) {
        nll2 = sqrt(nll2/(count-1));
        LOG_INF("Final estimate: PPL = %.4lf +/- %.5lf", ppl, nll2*ppl);
    } else {
        LOG_ERR("Unexpected negative standard deviation of log(prob)");
    }
    return true;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct custom_cli {
    std::string recipe_path;
    std::string plan_path;
    std::string logits_file;
    std::string outdir;
    std::string imatrix_path;
    bool keep_f16_copy = false;
    bool allow_fa      = false;  // opt-in: keep CUDA backend (flash-attn on) for speed; candidates within ~1.3e-4 EAR of -dev none
};

static void usage(const char * exe) {
    fprintf(stderr,
        "usage: %s -m <f16.gguf> -f <calib.txt> -c <ctx> -t <threads>\n"
        "    [--recipe <recipe.txt>]           # one candidate: class->type or per-tensor recipe\n"
        "    [--plan <plan.json>]              # Algorithm-1 batch: list of step objects\n"
        "    [--logits-file <out.logits>]      # single-candidate logits output\n"
        "    [--imatrix <imatrix.gguf>]        # optional, required for IQ-family targets\n"
        "    [--outdir <dir>]                  # plan mode: per-step logits written here\n"
        "    [--keep-f16-copy]                 # retain pristine F16 data in RAM for reset between steps\n",
        "    [--allow-fa]                      # keep CUDA backend (flash-attn on) for 14x faster passes; candidates within ~1.3e-4 EAR of -dev none\n",
        exe);
}

// strip the tool-specific flags from argv (common_params_parse rejects
// unknown arguments) and collect their values.
static void extract_custom_args(int argc, char ** argv, custom_cli & cli, std::vector<char *> & rest) {
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--recipe") {
            if (i + 1 >= argc) throw std::runtime_error("--recipe requires a value");
            cli.recipe_path = argv[++i];
        } else if (a == "--plan") {
            if (i + 1 >= argc) throw std::runtime_error("--plan requires a value");
            cli.plan_path = argv[++i];
        } else if (a == "--logits-file" || a == "--save-all-logits" || a == "--kl-divergence-base") {
            if (i + 1 >= argc) throw std::runtime_error(a + " requires a value");
            cli.logits_file = argv[++i];
        } else if (a == "--outdir") {
            if (i + 1 >= argc) throw std::runtime_error("--outdir requires a value");
            cli.outdir = argv[++i];
        } else if (a == "--imatrix") {
            if (i + 1 >= argc) throw std::runtime_error("--imatrix requires a value");
            cli.imatrix_path = argv[++i];
        } else if (a == "--keep-f16-copy") {
            cli.keep_f16_copy = true;
        } else if (a == "--allow-fa") {
            cli.allow_fa = true;
        } else {
            rest.push_back(argv[i]);
        }
    }
}

// satisfies -Wmissing-declarations
int llama_fake_quant_ear(int argc, char ** argv);

int llama_fake_quant_ear(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 512;
    params.escape = false;

    custom_cli cli;
    std::vector<char *> rest;
    try {
        extract_custom_args(argc, argv, cli, rest);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0]);
        return 1;
    }

    common_init();

    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    try {
        // ---- mode validation -------------------------------------------------
        const bool single = !cli.recipe_path.empty();
        const bool plan   = !cli.plan_path.empty();
        if (single == plan) {
            throw std::runtime_error("exactly one of --recipe or --plan is required");
        }
        if (cli.logits_file.empty() && !params.logits_file.empty()) {
            cli.logits_file = params.logits_file; // --save-all-logits/--kl-divergence-base spelling
        }
        if (single && cli.logits_file.empty()) {
            throw std::runtime_error("--recipe mode requires --logits-file");
        }
        if (plan && cli.outdir.empty()) {
            throw std::runtime_error("--plan mode requires --outdir");
        }
        if (plan && !cli.logits_file.empty()) {
            throw std::runtime_error("--logits-file is not used in --plan mode (logits go to --outdir)");
        }
        if (single && cli.keep_f16_copy) {
            LOG_WRN("%s: --keep-f16-copy is a no-op in single-candidate mode", __func__);
        }
        if (params.model.path.empty()) {
            throw std::runtime_error("no model file specified (-m)");
        }
        if (params.prompt.empty()) {
            throw std::runtime_error("no calibration file specified (-f)");
        }

        const int32_t n_ctx = params.n_ctx;

        // ---- load recipes / plan (validates syntax and the imatrix rule)
        recipe single_rec;
        std::vector<plan_step> plan_steps;
        if (single) {
            single_rec = parse_recipe_file(cli.recipe_path);
        } else {
            plan_steps = parse_plan_file(cli.plan_path);
        }

        bool needs_imatrix = false;
        if (single) {
            needs_imatrix = recipe_requires_imatrix(single_rec);
        } else {
            for (const auto & ps : plan_steps) {
                if (ps.is_switch ? target_requires_imatrix(ps.sw_to) : recipe_requires_imatrix(ps.rec)) {
                    needs_imatrix = true;
                }
            }
        }
        if (needs_imatrix && cli.imatrix_path.empty()) {
            throw std::runtime_error("recipe requires --imatrix (IQ family target)");
        }

        // ---- context sizing (mirror llama_perplexity) ------------------------
        params.n_parallel = std::max(1, params.n_batch / n_ctx);
        params.n_ctx = params.n_parallel * n_ctx;
        params.n_batch = std::min(params.n_batch, params.n_ctx);

        // in-place fake-quant mutates tensor->data; with mmap that would write
        // through to the F16 source file. Force owned buffers.
        if (params.use_mmap) {
            params.use_mmap = false;
            LOG_INF("%s: forcing --no-mmap: in-place fake-quant must not write through to the source GGUF", __func__);
        }

        // the EAR protocol (climb pipeline) ran with `-dev none`: no CUDA device
        // exists, so flash-attention AUTO resolves off and the graph numerics are
        // the CPU (unfused) path. `-ngl 0` alone still leaves the CUDA backend
        // registered, which flips FA on and changes logits by ~3e-3 median —
        // enough to move EAR by ~4e-3. Force the same device-less state.
        // (--allow-fa opts out: candidates then sit within ~1.3e-4 EAR of the
        // -dev none state — measured on climb2 r10 — while running ~14x faster;
        // the pure-F16 baseline is NOT byte-identical under FA, so equivalence
        // proofs should keep the default.)
        if (!cli.allow_fa && (!params.devices.empty() || params.n_gpu_layers != 0)) {
            params.devices = { nullptr };
            LOG_INF("%s: forcing -dev none: EAR protocol equivalence requires no CUDA device (flash-attn off)", __func__);
        } else if (cli.allow_fa) {
            LOG_INF("%s: --allow-fa: keeping CUDA backend (flash-attn may engage); EAR within ~1.3e-4 of -dev none protocol", __func__);
        }

        // in-place fake-quant writes into tensor->data from the host. GPU
        // offload makes ->data a device pointer (or a host-mirror that is not
        // the compute buffer), so the write-back would fault or be invisible
        // to the graph. Force CPU tensors.
        if (params.n_gpu_layers != 0) {
            params.n_gpu_layers = 0;
            LOG_INF("%s: forcing -ngl 0: in-place fake-quant requires host-accessible tensor data (GPU offload would make ->data a device pointer)", __func__);
        }

        // dynamic VBR KV needs GPU buffers; with CPU tensors (above) the KV
        // would stay on CPU and VBR refuses to arm. Force a static KV type so
        // the forward pass works on CPU.
        if (params.vbr_cache_type_k || params.vbr_cache_type_v) {
            params.vbr_cache_type_k = false;
            params.vbr_cache_type_v = false;
            params.cache_type_k = GGML_TYPE_F16;
            params.cache_type_v = GGML_TYPE_F16;
            LOG_INF("%s: forcing static f16 KV cache: VBR dynamic KV requires GPU buffers, but this tool runs CPU tensors", __func__);
        }

        llama_backend_init();
        llama_numa_init(params.numa);

        auto llama_init = common_init_from_params(params);
        auto * model = llama_init->model();
        auto * ctx   = llama_init->context();
        if (model == nullptr) {
            throw std::runtime_error("unable to load model");
        }
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create context");
        }

        const int32_t n_ctx_train = llama_model_n_ctx_train(model);
        if (params.n_ctx > n_ctx_train) {
            LOG_WRN("%s: model was trained on only %d context tokens (%d specified)", __func__, n_ctx_train, params.n_ctx);
        }

        // ---- calibration tokens (mirror llama_perplexity) --------------------
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
        LOG_INF("%s: tokenized %zu tokens", __func__, tokens.size());
        if (int(tokens.size()) < 2*n_ctx) {
            throw std::runtime_error("calibration file tokenizes to fewer than 2*n_ctx tokens");
        }

        // ---- tensor state ----------------------------------------------------
        model_state st;
        if (!build_model_state(st, model, params.model.path, cli.keep_f16_copy)) {
            return 1;
        }
        LOG_INF("%s: registered %zu live tensors for fake-quant", __func__, st.tensors.size());
        for (const char * ct : CLASS_TOKENS) {
            auto it = st.class_tensors.find(ct);
            LOG_INF("  class %-12s -> %zu tensors", ct, it == st.class_tensors.end() ? 0 : it->second.size());
        }

        imatrix_data imx;
        if (!cli.imatrix_path.empty()) {
            if (!load_imatrix_data(cli.imatrix_path, imx)) {
                return 1;
            }
        }

        // bind the OpenMP quantize phase to the requested thread count
#ifdef _OPENMP
        if (params.cpuparams.n_threads > 0) {
            omp_set_num_threads(params.cpuparams.n_threads);
        }
#endif

        // ---- run -------------------------------------------------------------
        if (single) {
            LOG_INF("\n=== applying recipe '%s' ===", cli.recipe_path.c_str());
            if (!apply_recipe(single_rec, st, imx)) {
                return 1;
            }
            if (!run_forward(ctx, params, n_ctx, tokens, cli.logits_file)) {
                return 1;
            }
            LOG_INF("candidate done: %s", cli.logits_file.c_str());
        } else {
            std::error_code ec;
            std::filesystem::create_directories(cli.outdir, ec);
            if (ec) {
                LOG_ERR("%s: failed to create --outdir %s: %s", __func__, cli.outdir.c_str(), ec.message().c_str());
                return 1;
            }
            for (const auto & ps : plan_steps) {
                LOG_INF("\n=== step %s ===", ps.id.c_str());
                const bool ok = ps.is_switch ? apply_switch(ps, st, imx) : apply_recipe(ps.rec, st, imx);
                if (!ok) {
                    LOG_ERR("%s: step %s failed", __func__, ps.id.c_str());
                    return 1;
                }
                const std::string out = (std::filesystem::path(cli.outdir) / ps.logits).string();
                if (!run_forward(ctx, params, n_ctx, tokens, out)) {
                    return 1;
                }
                LOG_INF("step %s done: %s", ps.id.c_str(), out.c_str());
            }
            LOG_INF("\nplan complete: %zu steps -> %s", plan_steps.size(), cli.outdir.c_str());
        }

        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s", e.what());
        return 1;
    }
}

int main(int argc, char ** argv) {
    return llama_fake_quant_ear(argc, argv);
}
