#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

struct state_piece {
    int layer = -1;
    char kind = 0;
    std::vector<float> values;
};

struct replay_result {
    std::vector<state_piece> state;
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> greedy_tokens;
    std::vector<double> greedy_margins;
    size_t exact_bytes = 0;
    size_t codec_bytes = 0;
    uint64_t codec_ops = 0;
    ggml_type codec_last = GGML_TYPE_COUNT;
};

enum class continuation_mode {
    teacher_forced,
    greedy,
};

struct quality_summary {
    double state_nrmse_max = 0.0;
    double state_max_abs = 0.0;
    double kld_sum = 0.0;
    double kld_max = 0.0;
    int logit_rows = 0;
    int top1_equal = 0;
    size_t exact_bytes = 0;
    size_t codec_bytes = 0;
    std::vector<double> kld_values;
    std::vector<double> reference_margins;
    std::vector<uint8_t> top1_flips;
};

struct robust_kld_stats {
    double median = 0.0;
    double trim1_mean = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double frac_gt_1e4 = 0.0;
    double frac_gt_1e3 = 0.0;
    double frac_gt_1e2 = 0.0;
};

static uint32_t env_u32(
        const char * name,
        uint32_t     fallback,
        uint32_t     maximum) {
    const char * value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || *end != '\0' || parsed == 0 || parsed > maximum) {
        fprintf(stderr, "invalid %s=%s (expected 1..%u)\n",
            name, value, maximum);
        return 0;
    }
    return (uint32_t) parsed;
}

static bool env_size(
        const char * name,
        size_t       fallback,
        size_t &     result) {
    const char * value = std::getenv(name);
    if (!value || !*value) {
        result = fallback;
        return true;
    }
    if (*value < '0' || *value > '9') {
        fprintf(stderr, "invalid %s=%s\n", name, value);
        return false;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        fprintf(stderr, "invalid %s=%s\n", name, value);
        return false;
    }
    result = (size_t) parsed;
    return true;
}

static bool select_depths(
        uint32_t                n_verify,
        std::vector<uint32_t> & depths) {
    const char * value = std::getenv("G6_DEPTHS");
    if (!value || !*value) {
        const uint32_t candidates[] = {
            1, 4, 8, 16, 64, 136, 286, n_verify,
        };
        for (uint32_t depth : candidates) {
            if (depth <= n_verify &&
                std::find(depths.begin(), depths.end(), depth) ==
                    depths.end()) {
                depths.push_back(depth);
            }
        }
        return !depths.empty();
    }

    const std::string list(value);
    size_t begin = 0;
    while (begin < list.size()) {
        const size_t comma = list.find(',', begin);
        const size_t end =
            comma == std::string::npos ? list.size() : comma;
        const std::string item = list.substr(begin, end - begin);
        if (item.empty() || item[0] < '0' || item[0] > '9') {
            fprintf(stderr, "invalid G6_DEPTHS=%s\n", value);
            return false;
        }
        char * parsed_end = nullptr;
        const unsigned long parsed =
            std::strtoul(item.c_str(), &parsed_end, 10);
        if (!parsed_end || *parsed_end != '\0' ||
            parsed == 0 || parsed > n_verify) {
            fprintf(stderr,
                "invalid G6_DEPTHS item %s (expected 1..%u)\n",
                item.c_str(), n_verify);
            return false;
        }
        const uint32_t depth = (uint32_t) parsed;
        if (std::find(depths.begin(), depths.end(), depth) ==
            depths.end()) {
            depths.push_back(depth);
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return !depths.empty();
}

static llama_context_ptr make_ctx(
        const common_params & params,
        llama_model *         model,
        uint32_t              n_batch) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 2;
    cparams.n_rs_seq = 0;
    cparams.n_batch = std::max(cparams.n_batch, n_batch);
    cparams.n_ubatch = std::max(cparams.n_ubatch, n_batch);
    cparams.n_outputs_max = cparams.n_seq_max;
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

static llama_memory_recurrent * get_recurrent(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_recr();
    }
    return nullptr;
}

static bool trim_attention(
        llama_context * ctx,
        llama_pos       keep_end) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, keep_end, -1);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_rm(0, keep_end, -1);
    }
    return true;
}

static bool decode_range(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count) {
    llama_batch batch = llama_batch_init((int32_t) count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_token(
        llama_context * ctx,
        llama_token     token,
        llama_pos       pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool read_state(
        llama_memory_recurrent * mem,
        std::vector<state_piece> & out) {
    if (mem->size == 0 || mem->cells[0].tail < 0) {
        return false;
    }
    const int32_t tail = mem->cells[0].tail;
    const auto & cell = mem->cells[tail];
    const uint32_t base_row =
        cell.src >= 0 ? (uint32_t) cell.src : (uint32_t) tail;
    const uint32_t row = mem->rs_idx[0] * mem->size + base_row;

    out.clear();
    auto read_one = [&](ggml_tensor * tensor, int il, char kind) {
        if (!tensor) {
            return true;
        }
        if (tensor->type != GGML_TYPE_F32 ||
            tensor->nb[0] != sizeof(float)) {
            return false;
        }
        state_piece piece;
        piece.layer = il;
        piece.kind = kind;
        piece.values.resize((size_t) tensor->ne[0]);
        ggml_backend_tensor_get(
            tensor, piece.values.data(),
            (size_t) row * tensor->nb[1],
            piece.values.size() * sizeof(float));
        out.push_back(std::move(piece));
        return true;
    };

    for (int il = 0; il < (int) mem->r_l.size(); ++il) {
        if (!read_one(mem->r_l[il], il, 'r') ||
            !read_one(mem->s_l[il], il, 's')) {
            return false;
        }
    }
    return !out.empty();
}

static bool run_case(
        const common_params &             params,
        llama_model *                     model,
        const std::vector<llama_token> & tokens,
        uint32_t                         n_prefix,
        uint32_t                         n_verify,
        uint32_t                         n_accepted,
        uint32_t                         continuation,
        ggml_type                        codec,
        continuation_mode                mode,
        replay_result &                  result) {
    auto ctx = make_ctx(params, model, n_verify);
    if (!ctx) {
        return false;
    }
    auto * recurrent = get_recurrent(ctx.get());
    if (!recurrent) {
        return false;
    }

    llama_set_dflash_capture(ctx.get(), nullptr, 0);
    // The shipped verify tape remains capped at
    // LLAMA_DFLASH_MAX_VERIFY_TOKENS. This internal test deliberately sizes
    // its private tape to the quality horizon under test without changing the
    // production DFlash limit.
    ctx->allocate_tape_gpu(1, (int) n_verify);
    llama_set_tape_recording(ctx.get(), true);
    llama_set_tape_minimal_replay(ctx.get(), true);
    if (!llama_dflash_tape_replay_available(ctx.get()) ||
        !decode_range(ctx.get(), tokens, 0, n_prefix)) {
        return false;
    }
    llama_synchronize(ctx.get());
    if (!recurrent->try_seq_cp(0, 1, -1, -1) ||
        !decode_range(ctx.get(), tokens, n_prefix, n_verify)) {
        return false;
    }
    llama_synchronize(ctx.get());

    auto * capture = ctx->dflash_capture.get();
    auto * tape = capture ? capture->active_tape() : nullptr;
    if (!capture || !tape || !tape->minimal_packed ||
        !capture->tape_stage_minimal_packed ||
        capture->tape_stage_n_tokens != (int) n_verify) {
        return false;
    }
    result.exact_bytes =
        (size_t) tape->minimal_record_floats *
        n_verify * sizeof(float);

    const llama_pos keep_end =
        (llama_pos) (n_prefix + n_accepted);
    if (!trim_attention(ctx.get(), keep_end) ||
        !recurrent->seq_rm(0, -1, -1) ||
        !recurrent->try_seq_cp(1, 0, -1, -1)) {
        return false;
    }

    const uint64_t codec_ops_before =
        capture->approx_codec_roundtrips;
    if (codec != GGML_TYPE_F32 &&
        !ctx->dflash_tape_codec_roundtrip(codec)) {
        return false;
    }
    result.codec_ops =
        capture->approx_codec_roundtrips - codec_ops_before;
    result.codec_last = capture->approx_codec_last;
    result.codec_bytes =
        codec == GGML_TYPE_F32
            ? result.exact_bytes
            : capture->approx_codec_storage_bytes_last;

    if (!llama_tape_replay(ctx.get(), 0, (int) n_accepted) ||
        !llama_tape_replay_sync(ctx.get()) ||
        !capture->replay_minimal_last ||
        recurrent->seq_pos_max(0) != keep_end - 1 ||
        !read_state(recurrent, result.state)) {
        return false;
    }

    result.logits.clear();
    result.greedy_tokens.clear();
    result.greedy_margins.clear();
    const int n_vocab =
        llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (mode == continuation_mode::teacher_forced) {
        result.logits.reserve(continuation);
        for (uint32_t i = 0; i < continuation; ++i) {
            const uint32_t pos = (uint32_t) keep_end + i;
            if (!decode_range(ctx.get(), tokens, pos, 1)) {
                return false;
            }
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits) {
                return false;
            }
            result.logits.emplace_back(logits, logits + n_vocab);
        }
    } else {
        if ((size_t) keep_end >= tokens.size()) {
            return false;
        }
        result.greedy_tokens.reserve(continuation);
        result.greedy_margins.reserve(continuation);
        llama_token input = tokens[(size_t) keep_end];
        for (uint32_t i = 0; i < continuation; ++i) {
            const uint32_t pos = (uint32_t) keep_end + i;
            if (!decode_token(ctx.get(), input, (llama_pos) pos)) {
                return false;
            }
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            if (!logits) {
                return false;
            }
            llama_token best_token = 0;
            double best = -std::numeric_limits<double>::infinity();
            double second = -std::numeric_limits<double>::infinity();
            for (int token = 0; token < n_vocab; ++token) {
                const double value = logits[token];
                if (value > best) {
                    second = best;
                    best = value;
                    best_token = (llama_token) token;
                } else if (value > second) {
                    second = value;
                }
            }
            result.greedy_tokens.push_back(best_token);
            result.greedy_margins.push_back(best - second);
            input = best_token;
        }
    }
    return true;
}

// Exercise the persistent rolling-ring codec rather than the fixed-tape
// round-trip oracle. The F16 records are captured during normal decode,
// reconstructed through the window's detached F32 staging, installed into the
// live recurrent row, and then scored identically to run_case().
static bool run_f16_window_case(
        const common_params &             params,
        llama_model *                     model,
        const std::vector<llama_token> & tokens,
        uint32_t                         n_prefix,
        uint32_t                         n_verify,
        uint32_t                         n_accepted,
        uint32_t                         continuation,
        replay_result &                  result) {
    auto ctx = make_ctx(params, model, n_verify);
    if (!ctx) {
        return false;
    }
    auto * recurrent = get_recurrent(ctx.get());
    if (!recurrent) {
        return false;
    }

    llama_set_dflash_capture(ctx.get(), nullptr, 0);
    ctx->allocate_tape_gpu(1, (int) n_verify);
    llama_set_tape_recording(ctx.get(), true);
    llama_set_tape_minimal_replay(ctx.get(), true);
    const int advance_batch = std::min<int>((int) n_verify, 16);
    if (!llama_dflash_tape_replay_available(ctx.get()) ||
        !decode_range(ctx.get(), tokens, 0, n_prefix) ||
        !llama_dflash_window_enable_batched_f16(
            ctx.get(), 0, (int) n_verify, advance_batch) ||
        !decode_range(ctx.get(), tokens, n_prefix, n_verify)) {
        return false;
    }
    llama_synchronize(ctx.get());

    auto * capture = ctx->dflash_capture.get();
    auto * window =
        capture && !capture->windows.empty()
            ? capture->windows[0].get() : nullptr;
    if (!window || window->record_type != GGML_TYPE_F16 ||
        !window->record_packed ||
        window->record_packed->type != GGML_TYPE_F16 ||
        window->count != (int) n_verify ||
        window->frontier_pos !=
            (llama_pos) (n_prefix + n_verify - 1)) {
        return false;
    }

    const llama_pos target =
        (llama_pos) (n_prefix + n_accepted - 1);
    const llama_pos keep_end = target + 1;
    llama_dflash_window_info restored = {};
    if (!llama_dflash_window_restore_seq(
            ctx.get(), 0, target, keep_end) ||
        !llama_dflash_window_commit_branch(
            ctx.get(), 0, target) ||
        !llama_dflash_window_get_info(
            ctx.get(), 0, &restored) ||
        restored.codec != LLAMA_DFLASH_WINDOW_CODEC_F16 ||
        restored.boundary_pos != target ||
        restored.frontier_pos != target ||
        restored.record_count != 0 ||
        restored.capture_pending ||
        recurrent->seq_pos_max(0) != target ||
        !read_state(recurrent, result.state)) {
        return false;
    }

    result.exact_bytes =
        (size_t) window->record_floats *
        n_verify * sizeof(float);
    result.codec_bytes =
        (size_t) window->record_floats *
        n_verify * ggml_type_size(GGML_TYPE_F16);
    result.codec_ops = n_verify;
    result.codec_last = GGML_TYPE_F16;
    result.logits.clear();
    const int n_vocab =
        llama_vocab_n_tokens(llama_model_get_vocab(model));
    result.logits.reserve(continuation);
    for (uint32_t i = 0; i < continuation; ++i) {
        const uint32_t pos = (uint32_t) keep_end + i;
        if (!decode_range(ctx.get(), tokens, pos, 1)) {
            return false;
        }
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) {
            return false;
        }
        result.logits.emplace_back(logits, logits + n_vocab);
    }
    return true;
}

static bool bit_equal(
        const replay_result & lhs,
        const replay_result & rhs) {
    if (lhs.state.size() != rhs.state.size() ||
        lhs.logits.size() != rhs.logits.size() ||
        lhs.greedy_tokens != rhs.greedy_tokens ||
        lhs.greedy_margins.size() != rhs.greedy_margins.size() ||
        (lhs.greedy_margins.size() > 0 &&
         std::memcmp(
             lhs.greedy_margins.data(), rhs.greedy_margins.data(),
             lhs.greedy_margins.size() * sizeof(double)) != 0)) {
        return false;
    }
    for (size_t i = 0; i < lhs.state.size(); ++i) {
        const auto & a = lhs.state[i];
        const auto & b = rhs.state[i];
        if (a.layer != b.layer || a.kind != b.kind ||
            a.values.size() != b.values.size() ||
            std::memcmp(
                a.values.data(), b.values.data(),
                a.values.size() * sizeof(float)) != 0) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.logits.size(); ++i) {
        if (lhs.logits[i].size() != rhs.logits[i].size() ||
            std::memcmp(
                lhs.logits[i].data(), rhs.logits[i].data(),
                lhs.logits[i].size() * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

static double logits_kld(
        const std::vector<float> & reference,
        const std::vector<float> & candidate) {
    const double ref_max =
        *std::max_element(reference.begin(), reference.end());
    const double cand_max =
        *std::max_element(candidate.begin(), candidate.end());
    double ref_sum = 0.0;
    double cand_sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        ref_sum += std::exp((double) reference[i] - ref_max);
        cand_sum += std::exp((double) candidate[i] - cand_max);
    }
    const double ref_log_z = ref_max + std::log(ref_sum);
    const double cand_log_z = cand_max + std::log(cand_sum);
    double kld = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double log_p = (double) reference[i] - ref_log_z;
        const double p = std::exp(log_p);
        if (p != 0.0) {
            const double log_q =
                (double) candidate[i] - cand_log_z;
            kld += p * (log_p - log_q);
        }
    }
    return std::max(0.0, kld);
}

static bool accumulate_quality(
        const replay_result & reference,
        const replay_result & candidate,
        quality_summary &     summary) {
    if (reference.state.size() != candidate.state.size() ||
        reference.logits.size() != candidate.logits.size()) {
        return false;
    }
    for (size_t p = 0; p < reference.state.size(); ++p) {
        const auto & ref = reference.state[p];
        const auto & got = candidate.state[p];
        if (ref.layer != got.layer || ref.kind != got.kind ||
            ref.values.size() != got.values.size()) {
            return false;
        }
        double err2 = 0.0;
        double ref2 = 0.0;
        for (size_t i = 0; i < ref.values.size(); ++i) {
            const double delta =
                (double) got.values[i] - ref.values[i];
            err2 += delta * delta;
            ref2 += (double) ref.values[i] * ref.values[i];
            summary.state_max_abs =
                std::max(summary.state_max_abs, std::fabs(delta));
        }
        const double denom =
            std::max(ref2, std::numeric_limits<double>::min());
        summary.state_nrmse_max =
            std::max(summary.state_nrmse_max, std::sqrt(err2 / denom));
    }

    for (size_t row = 0; row < reference.logits.size(); ++row) {
        const auto & ref = reference.logits[row];
        const auto & got = candidate.logits[row];
        if (ref.size() != got.size() || ref.empty()) {
            return false;
        }
        const double kld = logits_kld(ref, got);
        if (!std::isfinite(kld)) {
            return false;
        }
        summary.kld_sum += kld;
        summary.kld_max = std::max(summary.kld_max, kld);
        summary.logit_rows++;
        size_t ref_top = 0;
        double ref_best = -std::numeric_limits<double>::infinity();
        double ref_second = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < ref.size(); ++i) {
            const double value = ref[i];
            if (value > ref_best) {
                ref_second = ref_best;
                ref_best = value;
                ref_top = i;
            } else if (value > ref_second) {
                ref_second = value;
            }
        }
        const size_t got_top =
            (size_t) std::distance(
                got.begin(), std::max_element(got.begin(), got.end()));
        const bool flip = ref_top != got_top;
        summary.top1_equal += flip ? 0 : 1;
        summary.kld_values.push_back(kld);
        summary.reference_margins.push_back(ref_best - ref_second);
        summary.top1_flips.push_back(flip ? 1 : 0);
    }
    summary.exact_bytes = reference.exact_bytes;
    summary.codec_bytes = candidate.codec_bytes;
    return true;
}

static double quantile_sorted(
        const std::vector<double> & sorted,
        double                      quantile) {
    if (sorted.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const double pos = quantile * (sorted.size() - 1);
    const size_t lo = (size_t) std::floor(pos);
    const size_t hi = (size_t) std::ceil(pos);
    const double weight = pos - lo;
    return sorted[lo] * (1.0 - weight) + sorted[hi] * weight;
}

static robust_kld_stats robust_stats(
        const quality_summary & summary) {
    robust_kld_stats result;
    if (summary.kld_values.empty()) {
        result.median = std::numeric_limits<double>::infinity();
        result.trim1_mean = std::numeric_limits<double>::infinity();
        result.p95 = result.p99 = result.p999 = result.median;
        return result;
    }
    std::vector<double> sorted = summary.kld_values;
    std::sort(sorted.begin(), sorted.end());
    result.median = quantile_sorted(sorted, 0.5);
    result.p95 = quantile_sorted(sorted, 0.95);
    result.p99 = quantile_sorted(sorted, 0.99);
    result.p999 = quantile_sorted(sorted, 0.999);

    const size_t trim =
        std::min(sorted.size() / 2, (size_t) std::floor(sorted.size() * 0.01));
    double trimmed_sum = 0.0;
    for (size_t i = trim; i < sorted.size() - trim; ++i) {
        trimmed_sum += sorted[i];
    }
    result.trim1_mean =
        trimmed_sum / (sorted.size() - 2 * trim);
    result.frac_gt_1e3 =
        (double) std::count_if(
            sorted.begin(), sorted.end(),
            [](double value) { return value > 1e-3; }) /
        sorted.size();
    result.frac_gt_1e4 =
        (double) std::count_if(
            sorted.begin(), sorted.end(),
            [](double value) { return value > 1e-4; }) /
        sorted.size();
    result.frac_gt_1e2 =
        (double) std::count_if(
            sorted.begin(), sorted.end(),
            [](double value) { return value > 1e-2; }) /
        sorted.size();
    return result;
}

static bool dump_quality(
        FILE *                  fp,
        size_t                  corpus_offset,
        uint32_t                n_prefix,
        uint32_t                depth,
        const char *            codec,
        const quality_summary & summary) {
    if (!fp) {
        return true;
    }
    if (summary.kld_values.size() != summary.reference_margins.size() ||
        summary.kld_values.size() != summary.top1_flips.size()) {
        return false;
    }
    for (size_t row = 0; row < summary.kld_values.size(); ++row) {
        const size_t corpus_token =
            corpus_offset + n_prefix + depth + row;
        fprintf(fp, "%zu\t%u\t%s\t%zu\t%zu\t%.17g\t%u\t%.17g\n",
            corpus_offset, depth, codec, row, corpus_token,
            summary.kld_values[row],
            (unsigned) summary.top1_flips[row],
            summary.reference_margins[row]);
    }
    return !ferror(fp);
}

static bool write_text_file(
        const std::string & path,
        const std::string & text) {
    FILE * fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        return false;
    }
    const bool ok =
        text.empty() ||
        std::fwrite(text.data(), 1, text.size(), fp) == text.size();
    return std::fclose(fp) == 0 && ok;
}

static bool dump_greedy(
        const std::string & prefix,
        const llama_vocab * vocab,
        const replay_result & exact,
        const replay_result & f16) {
    if (prefix.empty() ||
        exact.greedy_tokens.size() != exact.greedy_margins.size() ||
        f16.greedy_tokens.size() != f16.greedy_margins.size() ||
        exact.greedy_tokens.size() != f16.greedy_tokens.size()) {
        return false;
    }
    const std::string exact_text =
        common_detokenize(vocab, exact.greedy_tokens, false);
    const std::string f16_text =
        common_detokenize(vocab, f16.greedy_tokens, false);
    if (!write_text_file(prefix + ".exact.txt", exact_text) ||
        !write_text_file(prefix + ".f16.txt", f16_text)) {
        return false;
    }

    FILE * fp = std::fopen((prefix + ".tokens.tsv").c_str(), "w");
    if (!fp) {
        return false;
    }
    fprintf(fp, "step\texact_token\tf16_token\texact_margin\tf16_margin\n");
    for (size_t i = 0; i < exact.greedy_tokens.size(); ++i) {
        fprintf(fp, "%zu\t%d\t%d\t%.17g\t%.17g\n",
            i, exact.greedy_tokens[i], f16.greedy_tokens[i],
            exact.greedy_margins[i], f16.greedy_margins[i]);
    }
    return std::fclose(fp) == 0;
}

static void print_summary(
        const char *            name,
        const quality_summary & summary) {
    const double mean_kld =
        summary.logit_rows > 0
            ? summary.kld_sum / summary.logit_rows
            : std::numeric_limits<double>::infinity();
    const double top1 =
        summary.logit_rows > 0
            ? (double) summary.top1_equal / summary.logit_rows
            : 0.0;
    const double ratio =
        summary.exact_bytes > 0
            ? (double) summary.codec_bytes / summary.exact_bytes
            : 0.0;
    const robust_kld_stats robust = robust_stats(summary);
    fprintf(stdout,
        "G6_APPROX codec=%s state_nrmse_max=%.9g "
        "state_max_abs=%.9g mean_kld=%.9g median_kld=%.9g "
        "trim1_mean_kld=%.9g p95_kld=%.9g p99_kld=%.9g "
        "p999_kld=%.9g max_kld=%.9g "
        "frac_kld_gt_1e4=%.9g frac_kld_gt_1e3=%.9g "
        "frac_kld_gt_1e2=%.9g "
        "top1=%d/%d top1_rate=%.6f codec_bytes=%zu "
        "exact_bytes=%zu byte_ratio=%.6f\n",
        name, summary.state_nrmse_max, summary.state_max_abs,
        mean_kld, robust.median, robust.trim1_mean,
        robust.p95, robust.p99, robust.p999,
        summary.kld_max,
        robust.frac_gt_1e4, robust.frac_gt_1e3, robust.frac_gt_1e2,
        summary.top1_equal, summary.logit_rows, top1,
        summary.codec_bytes, summary.exact_bytes, ratio);
}

static void print_depth_summary(
        uint32_t                depth,
        const char *            name,
        const quality_summary & summary) {
    const double mean_kld =
        summary.logit_rows > 0
            ? summary.kld_sum / summary.logit_rows
            : std::numeric_limits<double>::infinity();
    const double top1 =
        summary.logit_rows > 0
            ? (double) summary.top1_equal / summary.logit_rows
            : 0.0;
    const double ratio =
        summary.exact_bytes > 0
            ? (double) summary.codec_bytes / summary.exact_bytes
            : 0.0;
    const robust_kld_stats robust = robust_stats(summary);
    fprintf(stdout,
        "G6_APPROX_DEPTH depth=%u codec=%s "
        "state_nrmse_max=%.9g state_max_abs=%.9g "
        "mean_kld=%.9g median_kld=%.9g "
        "trim1_mean_kld=%.9g p95_kld=%.9g p99_kld=%.9g "
        "p999_kld=%.9g max_kld=%.9g "
        "frac_kld_gt_1e4=%.9g frac_kld_gt_1e3=%.9g "
        "frac_kld_gt_1e2=%.9g "
        "top1=%d/%d top1_rate=%.6f codec_bytes=%zu "
        "exact_bytes=%zu byte_ratio=%.6f\n",
        depth, name,
        summary.state_nrmse_max, summary.state_max_abs,
        mean_kld, robust.median, robust.trim1_mean,
        robust.p95, robust.p99, robust.p999,
        summary.kld_max,
        robust.frac_gt_1e4, robust.frac_gt_1e3, robust.frac_gt_1e2,
        summary.top1_equal, summary.logit_rows, top1,
        summary.codec_bytes, summary.exact_bytes, ratio);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;
    common_init();
    if (!common_params_parse(
            argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();
    common_init_result_ptr init =
        common_init_from_params(params, /*model_only=*/true);
    llama_model * model = init->model();
    if (!model) {
        return 1;
    }
    if (!llama_model_is_recurrent(model) &&
        !llama_model_is_hybrid(model)) {
        fprintf(stderr, "skipping: G6 requires a recurrent/hybrid model\n");
        return 0;
    }

    const uint32_t n_verify =
        env_u32("G6_VERIFY_TOKENS", 16, 1024);
    const uint32_t continuation =
        env_u32("G6_CONTINUATION_TOKENS", 64, 1024);
    size_t corpus_offset = 0;
    size_t greedy_tokens_size = 0;
    if (n_verify == 0 || continuation == 0 ||
        !env_size("G6_CORPUS_OFFSET", 0, corpus_offset) ||
        !env_size("G6_GREEDY_TOKENS", 0, greedy_tokens_size) ||
        greedy_tokens_size > 1024) {
        if (greedy_tokens_size > 1024) {
            fprintf(stderr,
                "invalid G6_GREEDY_TOKENS=%zu (expected 0..1024)\n",
                greedy_tokens_size);
        }
        return 1;
    }
    const uint32_t greedy_tokens = (uint32_t) greedy_tokens_size;
    const char * greedy_dump_prefix =
        std::getenv("G6_GREEDY_DUMP_PREFIX");
    if (greedy_tokens > 0 &&
        (!greedy_dump_prefix || !*greedy_dump_prefix)) {
        fprintf(stderr,
            "G6_GREEDY_DUMP_PREFIX is required when "
            "G6_GREEDY_TOKENS > 0\n");
        return 1;
    }

    auto probe = make_ctx(params, model, n_verify);
    ggml_backend_t gpu = probe ? probe->find_gpu_backend() : nullptr;
    ggml_backend_dev_t dev =
        gpu ? ggml_backend_get_device(gpu) : nullptr;
    const char * dev_name =
        dev ? ggml_backend_dev_name(dev) : nullptr;
    if (!dev_name || !std::strstr(dev_name, "CUDA")) {
        fprintf(stderr, "skipping: G6 targets single-device CUDA\n");
        return 0;
    }
    probe.reset();

    constexpr uint32_t n_prefix = 8;
    std::vector<uint32_t> depths;
    if (!select_depths(n_verify, depths)) {
        return 1;
    }

    const int n_vocab =
        llama_vocab_n_tokens(llama_model_get_vocab(model));
    const size_t n_tokens_needed =
        (size_t) n_prefix + n_verify + continuation;
    const char * embedded_text =
        "A careful experiment separates what was measured from what was "
        "assumed. The first observation may suggest a pattern, but repeated "
        "trials reveal whether the pattern survives changes in scale, input, "
        "and execution order. Reliable systems preserve the old state until "
        "the new state is complete, validated, and safely published. ";
    const std::string corpus =
        params.prompt.empty() ? embedded_text : params.prompt;
    std::vector<llama_token> corpus_tokens = common_tokenize(
        llama_model_get_vocab(model), corpus,
        /*add_special=*/true, /*parse_special=*/false);
    const std::vector<llama_token> repeat = common_tokenize(
        llama_model_get_vocab(model), corpus,
        /*add_special=*/false, /*parse_special=*/false);
    if (corpus_tokens.empty() || repeat.empty()) {
        fprintf(stderr, "G6 corpus tokenization produced no tokens\n");
        return 1;
    }
    const size_t corpus_tokens_original = corpus_tokens.size();
    const bool external_corpus = !params.prompt.empty();
    const size_t corpus_end = corpus_offset + n_tokens_needed;
    if (corpus_end < corpus_offset) {
        fprintf(stderr, "G6 corpus offset overflows token range\n");
        return 1;
    }
    if (external_corpus && corpus_tokens.size() < corpus_end) {
        fprintf(stderr,
            "G6 external corpus has %zu tokens, offset %zu needs %zu; "
            "choose a non-overlapping in-range G6_CORPUS_OFFSET\n",
            corpus_tokens.size(), corpus_offset, corpus_end);
        return 1;
    }
    while (corpus_tokens.size() < corpus_end) {
        const size_t count =
            std::min(repeat.size(), corpus_end - corpus_tokens.size());
        corpus_tokens.insert(
            corpus_tokens.end(), repeat.begin(), repeat.begin() + count);
    }
    std::vector<llama_token> tokens(
        corpus_tokens.begin() + corpus_offset,
        corpus_tokens.begin() + corpus_end);

    FILE * dump_fp = nullptr;
    const char * dump_path = std::getenv("G6_KLD_DUMP");
    if (dump_path && *dump_path) {
        dump_fp = std::fopen(dump_path, "w");
        if (!dump_fp) {
            fprintf(stderr, "failed to open G6_KLD_DUMP=%s\n", dump_path);
            return 1;
        }
        fprintf(dump_fp,
            "corpus_offset\tdepth\tcodec\trow\tcorpus_token\t"
            "kld\ttop1_flip\treference_margin\n");
    }

    fprintf(stdout,
        "G6_CONFIG verify_tokens=%u continuation_tokens=%u depths=%zu "
        "corpus=%s corpus_tokens_total=%zu corpus_offset=%zu "
        "corpus_slice_tokens=%zu dump=%s greedy_tokens=%u "
        "greedy_dump=%s vocab=%d\n",
        n_verify, continuation, depths.size(),
        params.prompt.empty() ? "embedded_text" :
            (params.prompt_file.empty() ? "prompt" : "prompt_file"),
        corpus_tokens_original, corpus_offset, tokens.size(),
        dump_fp ? dump_path : "none", greedy_tokens,
        greedy_tokens > 0 ? greedy_dump_prefix : "none", n_vocab);

    quality_summary f16_summary;
    quality_summary t8_summary;
    for (uint32_t depth : depths) {
        replay_result reference;
        replay_result null_arm;
        replay_result f16;
        replay_result t8;
        quality_summary f16_depth;
        quality_summary t8_depth;
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, depth, continuation,
                GGML_TYPE_F32, continuation_mode::teacher_forced,
                reference) ||
            !run_case(
                params, model, tokens,
                n_prefix, n_verify, depth, continuation,
                GGML_TYPE_F32, continuation_mode::teacher_forced,
                null_arm) ||
            !bit_equal(reference, null_arm)) {
            fprintf(stderr,
                "G6 null arm is not bit-exact at depth %u\n", depth);
            return 1;
        }
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, depth, continuation,
                GGML_TYPE_F16, continuation_mode::teacher_forced,
                f16) ||
            f16.codec_ops != 1 ||
            f16.codec_last != GGML_TYPE_F16 ||
            f16.codec_bytes >= f16.exact_bytes ||
            !accumulate_quality(reference, f16, f16_depth) ||
            !accumulate_quality(reference, f16, f16_summary)) {
            fprintf(stderr,
                "G6 F16 codec arm failed structurally at depth %u\n",
                depth);
            return 1;
        }
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, depth, continuation,
                GGML_TYPE_TURBO8_0, continuation_mode::teacher_forced,
                t8) ||
            t8.codec_ops != 1 ||
            t8.codec_last != GGML_TYPE_TURBO8_0 ||
            t8.codec_bytes >= f16.codec_bytes ||
            !accumulate_quality(reference, t8, t8_depth) ||
            !accumulate_quality(reference, t8, t8_summary)) {
            fprintf(stderr,
                "G6 TURBO8 codec arm failed structurally at depth %u\n",
                depth);
            return 1;
        }
        if (!dump_quality(
                dump_fp, corpus_offset, n_prefix, depth,
                "f16", f16_depth) ||
            !dump_quality(
                dump_fp, corpus_offset, n_prefix, depth,
                "turbo8_qkv_f16_controls", t8_depth)) {
            fprintf(stderr,
                "G6 per-token dump failed at depth %u\n", depth);
            return 1;
        }
        fprintf(stdout,
            "G6_DEPTH depth=%u null_bit_exact=1 "
            "f16_bytes=%zu turbo8_mixed_bytes=%zu exact_bytes=%zu "
            "turbo8_qkv=turbo8 gate_beta=f16\n",
            depth, f16.codec_bytes, t8.codec_bytes,
            reference.exact_bytes);
        print_depth_summary(depth, "f16", f16_depth);
        print_depth_summary(
            depth, "turbo8_qkv_f16_controls", t8_depth);
    }

    print_summary("f16", f16_summary);
    print_summary("turbo8_qkv_f16_controls", t8_summary);

    const char * integrate_env =
        std::getenv("G6_WINDOW_INTEGRATION");
    if (integrate_env && std::atoi(integrate_env) != 0) {
        replay_result f16_oracle;
        replay_result f16_window;
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, n_verify, continuation,
                GGML_TYPE_F16, continuation_mode::teacher_forced,
                f16_oracle) ||
            !run_f16_window_case(
                params, model, tokens,
                n_prefix, n_verify, n_verify, continuation,
                f16_window) ||
            f16_window.codec_bytes * 2 != f16_window.exact_bytes ||
            !bit_equal(f16_oracle, f16_window)) {
            fprintf(stderr,
                "G6 persistent F16 ring does not reproduce the "
                "fixed-tape F16 oracle\n");
            return 1;
        }
        fprintf(stdout,
            "G6_WINDOW_INTEGRATION codec=f16 depth=%u "
            "oracle_bit_exact=1 byte_ratio=0.500000 "
            "advance_batch=%d\n",
            n_verify, std::min<int>((int) n_verify, 16));
    }

    if (greedy_tokens > 0) {
        replay_result exact_greedy;
        replay_result null_greedy;
        replay_result f16_greedy;
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, n_verify, greedy_tokens,
                GGML_TYPE_F32, continuation_mode::greedy,
                exact_greedy) ||
            !run_case(
                params, model, tokens,
                n_prefix, n_verify, n_verify, greedy_tokens,
                GGML_TYPE_F32, continuation_mode::greedy,
                null_greedy) ||
            !bit_equal(exact_greedy, null_greedy)) {
            fprintf(stderr,
                "G6 greedy exact null is not bit-identical\n");
            return 1;
        }
        if (!run_case(
                params, model, tokens,
                n_prefix, n_verify, n_verify, greedy_tokens,
                GGML_TYPE_F16, continuation_mode::greedy,
                f16_greedy) ||
            f16_greedy.codec_ops != 1 ||
            f16_greedy.codec_last != GGML_TYPE_F16 ||
            f16_greedy.codec_bytes >= f16_greedy.exact_bytes ||
            exact_greedy.greedy_tokens.size() != greedy_tokens ||
            f16_greedy.greedy_tokens.size() != greedy_tokens) {
            fprintf(stderr, "G6 F16 greedy arm failed structurally\n");
            return 1;
        }

        size_t survival = greedy_tokens;
        size_t positional_matches = 0;
        for (size_t i = 0; i < greedy_tokens; ++i) {
            const bool equal =
                exact_greedy.greedy_tokens[i] ==
                f16_greedy.greedy_tokens[i];
            positional_matches += equal ? 1 : 0;
            if (!equal && survival == greedy_tokens) {
                survival = i;
            }
        }
        const bool diverged = survival < greedy_tokens;
        const int first_exact =
            diverged ? exact_greedy.greedy_tokens[survival] : -1;
        const int first_f16 =
            diverged ? f16_greedy.greedy_tokens[survival] : -1;
        const double first_exact_margin =
            diverged ? exact_greedy.greedy_margins[survival] :
                std::numeric_limits<double>::infinity();
        const double first_f16_margin =
            diverged ? f16_greedy.greedy_margins[survival] :
                std::numeric_limits<double>::infinity();
        if (!dump_greedy(
                greedy_dump_prefix,
                llama_model_get_vocab(model),
                exact_greedy, f16_greedy)) {
            fprintf(stderr, "G6 greedy artifact dump failed\n");
            return 1;
        }
        fprintf(stdout,
            "G6_GREEDY depth=%u tokens=%u exact_null_bit_exact=1 "
            "survival=%zu diverged=%d first_divergence=%lld "
            "positional_matches=%zu/%u first_exact_token=%d "
            "first_f16_token=%d first_exact_margin=%.9g "
            "first_f16_margin=%.9g artifacts=%s\n",
            n_verify, greedy_tokens, survival, diverged ? 1 : 0,
            diverged ? (long long) survival : -1LL,
            positional_matches, greedy_tokens,
            first_exact, first_f16,
            first_exact_margin, first_f16_margin,
            greedy_dump_prefix);
    }

    fprintf(stdout,
        "G6_MEASUREMENT_COMPLETE depths=%zu continuation=%u "
        "quality_thresholds=UNSET approximate_namespace_only=1\n",
        depths.size(), continuation);
    if (dump_fp && std::fclose(dump_fp) != 0) {
        fprintf(stderr, "failed to close G6_KLD_DUMP=%s\n", dump_path);
        return 1;
    }
    return 0;
}
