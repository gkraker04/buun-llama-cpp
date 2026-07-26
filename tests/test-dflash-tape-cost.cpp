#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

using json = nlohmann::json;

struct depth_bin {
    int cached = 0;
    int depth = 0;
    int retained_records = 0;
    int replay_records = 0;
    int events = 0;
};

struct latency_bin {
    int cached = 0;
    int depth = 0;
    int retained_records = 0;
    int events = 0;
    int replay_records = 0;
    double plane_us = 0.0;
    double tape_us = 0.0;
    double full_forward_us = 0.0;
};

struct spec_bin {
    int draft = 0;
    int rollback = 0;
    int events = 0;
};

static llama_context_ptr make_ctx(
        const common_params & params,
        llama_model *         model,
        uint32_t              n_rs_seq,
        uint32_t              n_batch) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq = n_rs_seq;
    cparams.n_batch = std::max(cparams.n_batch, n_batch);
    cparams.n_ubatch = std::max(cparams.n_ubatch, n_batch);
    cparams.n_outputs_max = 1;
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

static bool decode_range(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count) {
    if (count == 0) {
        return true;
    }
    llama_batch batch = llama_batch_init((int32_t) count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        const llama_token token = tokens[pos % tokens.size()];
        common_batch_add(batch, token, pos, { 0 }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool boundary_graph_cache_enabled() {
    const char * disable = std::getenv("G5_DISABLE_BOUNDARY_GRAPH");
    return !disable || std::atoi(disable) == 0;
}

static bool f16_window_enabled() {
    const char * enable = std::getenv("G5_WINDOW_F16");
    return enable && std::atoi(enable) != 0;
}

static bool enable_window_batched(
        llama_context * ctx,
        llama_seq_id    seq_id,
        int             retained_depth,
        int             advance_batch) {
    return f16_window_enabled()
        ? llama_dflash_window_enable_batched_f16(
              ctx, seq_id, retained_depth, advance_batch)
        : llama_dflash_window_enable_batched(
              ctx, seq_id, retained_depth, advance_batch);
}

static bool enable_window(
        llama_context * ctx,
        llama_seq_id    seq_id,
        int             capacity) {
    return enable_window_batched(ctx, seq_id, capacity, 1);
}

static void reset_window_profile(dflash_window & window) {
    window.profile_append_us = 0;
    window.profile_advance_us = 0;
    window.profile_stage_us = 0;
    window.profile_apply_us = 0;
    window.profile_append_calls = 0;
    window.profile_advance_calls = 0;
    window.profile_apply_calls = 0;
}

static bool decode_chunks(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count,
        uint32_t                         chunk) {
    uint32_t done = 0;
    while (done < count) {
        const uint32_t n = std::min(chunk, count - done);
        if (!decode_range(ctx, tokens, begin + done, n)) {
            return false;
        }
        done += n;
    }
    return true;
}

template<typename F>
static double time_us(llama_context * ctx, F && fn) {
    llama_synchronize(ctx);
    const int64_t start = ggml_time_us();
    const bool ok = fn();
    llama_synchronize(ctx);
    const int64_t end = ggml_time_us();
    return ok ? (double) (end - start) : -1.0;
}

static size_t core_bytes(const llama_context * ctx) {
    size_t total = 0;
    for (const auto & entry : llama_get_memory_breakdown(ctx)) {
        total += entry.second.context + entry.second.compute;
    }
    return total;
}

static size_t vector_float_bytes(const std::vector<float> & values) {
    return values.capacity() * sizeof(float);
}

static size_t host_tape_bytes(const dflash_capture_data & capture) {
    size_t total = 0;
    auto add_layer = [&](const dflash_tape_layer & layer) {
        total += vector_float_bytes(layer.k);
        total += vector_float_bytes(layer.v);
        total += vector_float_bytes(layer.gate);
        total += vector_float_bytes(layer.beta);
        total += vector_float_bytes(layer.qkv_mixed);
    };
    for (const auto & layer : capture.tape_layers) {
        add_layer(layer);
    }
    for (const auto & layer : capture.window_staging.qkv_layers) {
        add_layer(layer);
    }
    for (const auto & layer : capture.window_pending.qkv_layers) {
        add_layer(layer);
    }
    total += capture.replay_zeros.capacity() * sizeof(float);
    return total;
}

struct tape_bytes {
    size_t fixed_tape = 0;
    size_t ring_boundary = 0;
    size_t replay_scratch = 0;
    size_t other_device = 0;
    size_t host_payload = 0;
    size_t host_metadata = 0;

    size_t device_total() const {
        return fixed_tape + ring_boundary + replay_scratch + other_device;
    }

    size_t total() const {
        return device_total() + host_payload + host_metadata;
    }
};

static tape_bytes measure_tape_bytes(const llama_context * ctx) {
    tape_bytes out;
    if (!ctx->dflash_capture) {
        return out;
    }
    const auto & capture = *ctx->dflash_capture;
    for (const auto & tape : capture.tapes) {
        if (tape && tape->buf) {
            out.fixed_tape += ggml_backend_buffer_get_size(tape->buf);
        }
    }
    for (const auto & window : capture.windows) {
        if (!window) {
            continue;
        }
        if (window->buf) {
            out.ring_boundary += ggml_backend_buffer_get_size(window->buf);
        }
        if (window->scratch) {
            out.replay_scratch += ggml_backend_buffer_get_size(window->scratch);
        }
        if (window->advance_scratch) {
            out.replay_scratch +=
                ggml_backend_buffer_get_size(window->advance_scratch);
        }
        if (window->codec_scratch) {
            out.replay_scratch +=
                ggml_backend_buffer_get_size(window->codec_scratch);
        }
        out.host_metadata += window->records.capacity() * sizeof(dflash_window_record);
        out.host_metadata += window->layers.capacity() * sizeof(dflash_window_layer);
        out.host_metadata += window->layer_ids.capacity() * sizeof(int32_t);
        out.host_metadata += window->gpu_layer_indices.capacity() * sizeof(int);
    }
    if (capture.replay_buf) {
        out.other_device += ggml_backend_buffer_get_size(capture.replay_buf);
    }
    for (ggml_backend_buffer_t buf : capture.replay_meta_bufs) {
        if (buf) {
            out.other_device += ggml_backend_buffer_get_size(buf);
        }
    }
    if (capture.stage_buf) {
        out.other_device += ggml_backend_buffer_get_size(capture.stage_buf);
    }
    out.host_payload = std::max(
        host_tape_bytes(capture),
        capture.window_host_staging_peak_bytes);
    out.host_metadata += capture.tape_layers.capacity() * sizeof(dflash_tape_layer);
    out.host_metadata += capture.tapes.capacity() * sizeof(std::unique_ptr<dflash_tape_gpu>);
    out.host_metadata += capture.windows.capacity() * sizeof(std::unique_ptr<dflash_window>);
    return out;
}

static bool load_histogram(
        const char *              path,
        std::vector<depth_bin> & bins,
        int &                     window,
        int &                     request_events,
        int64_t &                 appended_tokens,
        int                       advance_batch) {
    std::ifstream input(path);
    if (!input) {
        fprintf(stderr, "cannot open edit histogram: %s\n", path);
        return false;
    }
    json root;
    input >> root;
    const auto & hist = root.at("warm_edit_rewind").at("hist");
    window = 0;
    request_events = root.at("events").get<int>();
    appended_tokens = 0;
    int histogram_events = 0;
    for (const auto & sample : root.at("samples")) {
        appended_tokens +=
            (int64_t) sample.at("append").get<int>() *
            sample.at("events").get<int>();
    }
    for (const auto & item : hist) {
        const int depth = item.at("depth").get<int>();
        const int events = item.at("events").get<int>();
        if (depth <= 0 || events <= 0) {
            continue;
        }
        window = std::max(window, depth);
        histogram_events += events;
    }
    if (window <= 0 || advance_batch <= 0) {
        return false;
    }

    // Preserve the observed frontier jointly with rewind depth. A W-record
    // ring is not full during early turns, and after it fills a q-batched
    // boundary has a deterministic phase derived from the number of records
    // captured since token zero.
    std::map<std::tuple<int, int, int, int>, int> observed;
    for (const auto & sample : root.at("samples")) {
        const int cached = sample.at("cached").get<int>();
        const int depth = sample.at("rewind").get<int>();
        const int events = sample.at("events").get<int>();
        if (cached <= 0 || depth <= 0 || events <= 0) {
            continue;
        }
        const int total_records = cached - 1;
        const int retained =
            total_records < window
                ? total_records
                : window + (total_records - window) % advance_batch;
        const int replay = retained - depth;
        if (replay < 0) {
            fprintf(stderr,
                "observed rewind depth %d exceeds retained tape %d at cached=%d\n",
                depth, retained, cached);
            return false;
        }
        observed[{ cached, depth, retained, replay }] += events;
    }

    bins.clear();
    bins.reserve(observed.size());
    int observed_events = 0;
    for (const auto & item : observed) {
        const auto & [cached, depth, retained, replay] = item.first;
        bins.push_back({ cached, depth, retained, replay, item.second });
        observed_events += item.second;
    }
    if (observed_events != histogram_events) {
        fprintf(stderr,
            "joint edit samples account for %d events, depth histogram has %d\n",
            observed_events, histogram_events);
        return false;
    }
    return !bins.empty() && observed_events > 0;
}

static bool load_spec_histogram(
        const char *            path,
        std::vector<spec_bin> & bins,
        int &                   max_draft) {
    std::ifstream input(path);
    if (!input) {
        fprintf(stderr, "cannot open speculative histogram: %s\n", path);
        return false;
    }
    json root;
    input >> root;
    bins.clear();
    max_draft = 0;
    for (const auto & item : root.at("joint")) {
        spec_bin bin;
        bin.draft = item.at("draft").get<int>();
        bin.rollback = item.at("rollback").get<int>();
        bin.events = item.at("cycles").get<int>();
        if (bin.draft < 0 || bin.rollback < 0 ||
            bin.rollback > bin.draft || bin.events <= 0) {
            continue;
        }
        bins.push_back(bin);
        max_draft = std::max(max_draft, bin.draft);
    }
    return !bins.empty() && max_draft > 0;
}

static double weighted_mean(
        const std::vector<latency_bin> & bins,
        double latency_bin::*            field) {
    int64_t events = 0;
    double total = 0.0;
    for (const auto & bin : bins) {
        events += bin.events;
        total += bin.events * bin.*field;
    }
    return events > 0 ? total / events : 0.0;
}

static void reset_plane_frontier(
        llama_memory_recurrent * recurrent,
        llama_seq_id             seq_id,
        llama_pos                frontier,
        uint32_t                 valid_depth) {
    const int32_t tail = recurrent->cells[seq_id].tail;
    recurrent->cells[tail].pos = frontier;
    recurrent->rs_idx[seq_id] = 0;
    recurrent->rollback_valid_depth[seq_id] = valid_depth;
}

static int run_capture_profile(
        const common_params &             params,
        llama_model *                     model,
        const std::vector<llama_token> & tokens,
        int                               window,
        int                               advance_batch,
        const char *                      mode,
        int                               profile_tokens) {
    constexpr uint32_t capture_chunk = 16;
    const uint32_t batch = (uint32_t) window + 1;
    auto ctx = make_ctx(params, model, 0, batch);
    if (!ctx) {
        fprintf(stderr, "capture profile failed to create %s context\n", mode);
        return 1;
    }
    ggml_backend_t gpu = ctx->find_gpu_backend();
    ggml_backend_dev_t device = gpu ? ggml_backend_get_device(gpu) : nullptr;
    const char * device_name = device ? ggml_backend_dev_name(device) : nullptr;
    if (!device_name || !std::strstr(device_name, "CUDA")) {
        fprintf(stderr, "skipping capture profile: single-device CUDA required\n");
        return 0;
    }

    const bool tape_mode = std::strcmp(mode, "tape") == 0;
    if (!tape_mode && std::strcmp(mode, "baseline") != 0) {
        fprintf(stderr, "G5_PROFILE_CAPTURE must be baseline or tape\n");
        return 1;
    }

    if (tape_mode) {
        llama_set_dflash_capture(ctx.get(), nullptr, 0);
        llama_dflash_allocate_slots(ctx.get(), 1);
        llama_set_tape_recording(ctx.get(), true);
        llama_set_tape_minimal_replay(ctx.get(), true);
        if (!llama_dflash_tape_replay_available(ctx.get()) ||
            !decode_range(ctx.get(), tokens, 0, 1) ||
            !enable_window_batched(
                ctx.get(), 0, window, advance_batch) ||
            !decode_chunks(
                ctx.get(), tokens, 1, (uint32_t) window, capture_chunk)) {
            fprintf(stderr, "capture profile tape warm fill failed\n");
            return 1;
        }
        ctx->dflash_capture->windows[0]->advance_graph_cache_enabled =
            boundary_graph_cache_enabled();
    } else if (!decode_chunks(
            ctx.get(), tokens, 0, (uint32_t) window + 1, capture_chunk)) {
        fprintf(stderr, "capture profile baseline warm fill failed\n");
        return 1;
    }

    // Warm the identical one-token schedule and, for the tape arm, both
    // alternating private-boundary graphs through CUDA's direct/warmup/capture
    // sequence before the profiler range begins.
    const uint32_t warm_begin = (uint32_t) window + 1;
    const uint32_t warm_tokens = 4U * (uint32_t) advance_batch;
    for (uint32_t i = 0; i < warm_tokens; ++i) {
        if (!decode_range(ctx.get(), tokens, warm_begin + i, 1)) {
            fprintf(stderr, "capture profile steady warmup failed\n");
            return 1;
        }
    }
    llama_synchronize(ctx.get());
    dflash_window * profiled_window = nullptr;
    if (tape_mode) {
        profiled_window = ctx->dflash_capture->windows[0].get();
        reset_window_profile(*profiled_window);
        profiled_window->profile_timing = true;
    }

    using profiler_fn = int (*)(void);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto profiler_start = (profiler_fn)
        ggml_backend_reg_get_proc_address(reg, "dflash_cuda_profiler_start");
    auto profiler_stop = (profiler_fn)
        ggml_backend_reg_get_proc_address(reg, "dflash_cuda_profiler_stop");
    if (!profiler_start || !profiler_stop) {
        fprintf(stderr, "CUDA backend has no profiler range hooks\n");
        return 1;
    }
    if (profiler_start() != 0) {
        fprintf(stderr, "cudaProfilerStart failed\n");
        return 1;
    }

    const uint32_t measured_begin = warm_begin + warm_tokens;
    const int64_t start_us = ggml_time_us();
    bool ok = true;
    for (int i = 0; i < profile_tokens; ++i) {
        ok &= decode_range(
            ctx.get(), tokens, measured_begin + (uint32_t) i, 1);
        if (!ok) {
            break;
        }
    }
    llama_synchronize(ctx.get());
    if (profiled_window) {
        profiled_window->profile_timing = false;
    }
    const int64_t end_us = ggml_time_us();
    const int stop_status = profiler_stop();
    if (!ok || stop_status != 0) {
        fprintf(stderr, "capture profile measured phase failed\n");
        return 1;
    }

    if (tape_mode) {
        const auto * rolling = ctx->dflash_capture->windows[0].get();
        if (!rolling || rolling->count != window ||
            rolling->frontier_pos - rolling->boundary_pos != window ||
            ctx->dflash_capture->window_pending.active) {
            fprintf(stderr, "capture profile tape window lost invariants\n");
            return 1;
        }
    }

    fprintf(stdout,
        "G5_CAPTURE_PROFILE mode=%s tokens=%d elapsed_us=%" PRId64
        " us_per_token=%.3f window=%d advance_batch=%d boundary_graph=%d "
        "record_codec=%s\n",
        mode, profile_tokens, end_us - start_us,
        (double) (end_us - start_us) / profile_tokens,
        window, advance_batch,
        boundary_graph_cache_enabled() ? 1 : 0,
        f16_window_enabled() ? "f16" : "f32");
    if (profiled_window) {
        const double inv_tokens = 1.0 / profile_tokens;
        fprintf(stdout,
            "G5_CAPTURE_PHASE append_us_per_token=%.3f "
            "advance_us_per_token=%.3f stage_us_per_token=%.3f "
            "apply_us_per_token=%.3f append_calls=%" PRIu64 " "
            "advance_calls=%" PRIu64 " apply_calls=%" PRIu64 "\n",
            profiled_window->profile_append_us * inv_tokens,
            profiled_window->profile_advance_us * inv_tokens,
            profiled_window->profile_stage_us * inv_tokens,
            profiled_window->profile_apply_us * inv_tokens,
            profiled_window->profile_append_calls,
            profiled_window->profile_advance_calls,
            profiled_window->profile_apply_calls);
    }
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * hist_path = std::getenv("G5_EDIT_HIST");
    if (!hist_path || !hist_path[0]) {
        fprintf(stderr, "G5_EDIT_HIST must name the frozen edit histogram JSON\n");
        return 1;
    }
    const char * spec_hist_path = std::getenv("G5_SPEC_HIST");
    if (!spec_hist_path || !spec_hist_path[0]) {
        fprintf(stderr, "G5_SPEC_HIST must name the frozen speculative histogram JSON\n");
        return 1;
    }

    const char * advance_env = std::getenv("G5_ADVANCE_BATCH");
    const int advance_batch =
        advance_env ? std::max(1, std::atoi(advance_env)) : 16;
    std::vector<depth_bin> depth_bins;
    int window = 0;
    int request_events = 0;
    int64_t appended_tokens = 0;
    if (!load_histogram(
            hist_path, depth_bins, window,
            request_events, appended_tokens, advance_batch)) {
        return 1;
    }
    const int events = std::accumulate(
        depth_bins.begin(), depth_bins.end(), 0,
        [](int sum, const depth_bin & bin) { return sum + bin.events; });
    const char * generated_env = std::getenv("G5_GENERATED_PER_REQUEST");
    const int generated_per_request =
        generated_env ? std::max(0, std::atoi(generated_env)) : 128;
    std::vector<spec_bin> spec_bins;
    int spec_window = 0;
    if (!load_spec_histogram(spec_hist_path, spec_bins, spec_window)) {
        return 1;
    }
    const int spec_events = std::accumulate(
        spec_bins.begin(), spec_bins.end(), 0,
        [](int sum, const spec_bin & bin) { return sum + bin.events; });

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr init = common_init_from_params(params, /*model_only=*/true);
    llama_model * model = init->model();
    if (!model) {
        fprintf(stderr, "failed to initialize model\n");
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "skipping: Gate 5 requires a recurrent/hybrid model\n");
        return 0;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens((size_t) window + 64);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    const char * profile_capture = std::getenv("G5_PROFILE_CAPTURE");
    if (profile_capture && profile_capture[0]) {
        const char * profile_tokens_env =
            std::getenv("G5_PROFILE_CAPTURE_TOKENS");
        const int profile_tokens = profile_tokens_env
            ? std::max(1, std::atoi(profile_tokens_env))
            : 256;
        return run_capture_profile(
            params, model, tokens, window, advance_batch,
            profile_capture, profile_tokens);
    }

    constexpr uint32_t capture_chunk = 16;
    const char * steady_env = std::getenv("G5_STEADY_TOKENS");
    const uint32_t steady_requested = steady_env
        ? (uint32_t) std::max(1, std::atoi(steady_env))
        : 256U;
    const uint32_t steady_quantum = (uint32_t) advance_batch;
    const uint32_t steady_tokens =
        ((steady_requested + steady_quantum - 1) / steady_quantum) *
        steady_quantum;
    const uint32_t steady_warm_tokens = 4U * steady_quantum;
    const uint32_t batch = (uint32_t) window + 1;
    const int physical_capacity = window + advance_batch - 1;
    const double mean_retained_count =
        std::accumulate(
            depth_bins.begin(), depth_bins.end(), 0.0,
            [](double sum, const depth_bin & bin) {
                return sum + bin.retained_records * bin.events;
            }) / events;

    size_t baseline_core = 0;
    double baseline_plane_batch_us = 0.0;
    double baseline_fill_us = 0.0;
    double baseline_steady_us = 0.0;
    std::map<int, double> full_forward_us;

    // Baseline exact forward costs. Each replay length is warmed immediately
    // before measurement; the timed decode excludes the one-token boundary.
    {
        auto ctx = make_ctx(params, model, 0, batch);
        if (!ctx) {
            fprintf(stderr, "failed to create baseline context\n");
            return 1;
        }
        auto plane_batch_once = [&]() {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            return decode_range(ctx.get(), tokens, 0, batch);
        };
        if (!plane_batch_once()) {
            fprintf(stderr, "baseline plane-shape warmup failed\n");
            return 1;
        }
        baseline_plane_batch_us = time_us(ctx.get(), plane_batch_once);
        if (baseline_plane_batch_us < 0) {
            fprintf(stderr, "baseline plane-shape timing failed\n");
            return 1;
        }

        auto prepare_fill = [&]() {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            return decode_range(ctx.get(), tokens, 0, 1);
        };
        auto fill_only = [&]() {
            return decode_chunks(
                ctx.get(), tokens, 1, (uint32_t) window, capture_chunk);
        };
        if (!prepare_fill() || !fill_only() || !prepare_fill()) {
            fprintf(stderr, "baseline warm fill/reset failed\n");
            return 1;
        }
        baseline_fill_us = time_us(ctx.get(), fill_only);
        if (baseline_fill_us < 0) {
            fprintf(stderr, "baseline timed fill failed\n");
            return 1;
        }

        // Match the tape arm's one-token warm schedule.
        const uint32_t steady_warm_pos = (uint32_t) window + 1;
        for (uint32_t i = 0; i < steady_warm_tokens; ++i) {
            if (!decode_range(ctx.get(), tokens, steady_warm_pos + i, 1)) {
                fprintf(stderr, "baseline steady warmup failed\n");
                return 1;
            }
        }
        const uint32_t steady_begin =
            steady_warm_pos + steady_warm_tokens;
        baseline_steady_us = time_us(ctx.get(), [&]() {
            for (uint32_t i = 0; i < steady_tokens; ++i) {
                if (!decode_range(ctx.get(), tokens, steady_begin + i, 1)) {
                    return false;
                }
            }
            return true;
        });
        if (baseline_steady_us < 0) {
            fprintf(stderr, "baseline steady decode failed\n");
            return 1;
        }
        baseline_core = core_bytes(ctx.get());

        for (const auto & bin : depth_bins) {
            const int replay = bin.replay_records;
            if (full_forward_us.count(replay)) {
                continue;
            }
            if (replay == 0) {
                full_forward_us[replay] = 0.0;
                continue;
            }
            auto prepare_forward = [&]() {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                return decode_range(ctx.get(), tokens, 0, 1);
            };
            auto forward_replay = [&]() {
                return decode_range(
                    ctx.get(), tokens, 1, (uint32_t) replay);
            };
            if (!prepare_forward() || !forward_replay() ||
                !prepare_forward()) {
                fprintf(stderr, "full-forward warmup failed at %d records\n", replay);
                return 1;
            }
            const double elapsed = time_us(ctx.get(), forward_replay);
            if (elapsed < 0) {
                fprintf(stderr, "full-forward timing failed at %d records\n", replay);
                return 1;
            }
            full_forward_us[replay] = elapsed;
        }
    }

    size_t plane_core = 0;
    double plane_capture_us = -1.0;
    std::map<int, double> plane_select_us;
    bool plane_edit_capable = false;

    // Dense W-plane arm. A W+1-token batch can populate all snapshots, which
    // permits selection timing and actual allocation measurement. The same
    // planes are refresh-scoped: a following one-token normal decode sets valid
    // depth to zero, so this is not an online edit-window implementation.
    {
        auto ctx = make_ctx(params, model, (uint32_t) window, batch);
        if (!ctx) {
            fprintf(stderr, "dense plane context allocation failed at W=%d\n", window);
        } else {
            auto * recurrent = get_recurrent(ctx.get());
            if (!recurrent) {
                fprintf(stderr, "plane context exposes no recurrent memory\n");
                return 1;
            }
            auto capture_once = [&]() {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                return decode_range(ctx.get(), tokens, 0, batch);
            };
            if (!capture_once()) {
                fprintf(stderr, "plane capture warmup failed at W=%d\n", window);
                return 1;
            }
            plane_capture_us = time_us(ctx.get(), capture_once);
            if (plane_capture_us < 0 ||
                recurrent->rollback_valid_depth[0] != (uint32_t) window) {
                fprintf(stderr, "plane capture did not publish W=%d snapshots\n", window);
                return 1;
            }
            plane_core = core_bytes(ctx.get());
            plane_edit_capable = false;

            const llama_pos frontier = window;
            constexpr int selector_reps = 1000;
            for (const auto & bin : depth_bins) {
                if (plane_select_us.count(bin.depth)) {
                    continue;
                }
                reset_plane_frontier(
                    recurrent, 0, frontier, (uint32_t) window);
                const llama_pos p0 = frontier - bin.depth + 1;
                const int64_t start = ggml_time_us();
                bool ok = true;
                for (int rep = 0; rep < selector_reps; ++rep) {
                    reset_plane_frontier(
                        recurrent, 0, frontier, (uint32_t) window);
                    ok &= recurrent->seq_rm(0, p0, -1);
                }
                const int64_t end = ggml_time_us();
                if (!ok) {
                    fprintf(stderr, "plane selector rejected depth %d\n", bin.depth);
                    return 1;
                }
                plane_select_us[bin.depth] =
                    (double) (end - start) / selector_reps;
            }

            // Prove the refresh-scoped limitation rather than inferring it.
            reset_plane_frontier(
                recurrent, 0, frontier, (uint32_t) window);
            if (!decode_range(ctx.get(), tokens, batch, 1)) {
                fprintf(stderr, "plane one-token capability probe failed\n");
                return 1;
            }
            llama_synchronize(ctx.get());
            plane_edit_capable =
                recurrent->rollback_valid_depth[0] >= (uint32_t) window;
        }
    }

    size_t tape_core = 0;
    tape_bytes tape_extra;
    double tape_fill_us = 0.0;
    double tape_steady_us = 0.0;
    uint64_t tape_phase_append_us = 0;
    uint64_t tape_phase_advance_us = 0;
    uint64_t tape_phase_stage_us = 0;
    uint64_t tape_phase_apply_us = 0;
    uint64_t tape_phase_advance_calls = 0;
    std::map<std::pair<int, int>, double> tape_restore_us;

    {
        auto ctx = make_ctx(params, model, 0, batch);
        if (!ctx) {
            fprintf(stderr, "failed to create tape context\n");
            return 1;
        }
        auto * recurrent = get_recurrent(ctx.get());
        if (!recurrent) {
            fprintf(stderr, "tape context exposes no recurrent memory\n");
            return 1;
        }
        ggml_backend_t gpu = ctx->find_gpu_backend();
        ggml_backend_dev_t device = gpu ? ggml_backend_get_device(gpu) : nullptr;
        const char * device_name = device ? ggml_backend_dev_name(device) : nullptr;
        if (!device_name || !std::strstr(device_name, "CUDA")) {
            fprintf(stderr, "skipping: Gate 5 targets a single CUDA device\n");
            return 0;
        }

        llama_set_dflash_capture(ctx.get(), nullptr, 0);
        llama_dflash_allocate_slots(ctx.get(), 1);
        llama_set_tape_recording(ctx.get(), true);
        llama_set_tape_minimal_replay(ctx.get(), true);
        if (!llama_dflash_tape_replay_available(ctx.get())) {
            fprintf(stderr, "lossless GPU tape replay is unavailable\n");
            return 1;
        }

        auto prepare_window = [&]() {
            if (ctx->dflash_capture->window_pending.active) {
                return false;
            }
            if (!ctx->dflash_capture->windows.empty()) {
                ctx->dflash_capture->windows[0].reset();
            }
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            if (!decode_range(ctx.get(), tokens, 0, 1) ||
                !enable_window_batched(
                    ctx.get(), 0, window, advance_batch)) {
                return false;
            }
            ctx->dflash_capture->windows[0]->advance_graph_cache_enabled =
                boundary_graph_cache_enabled();
            return true;
        };
        auto fill_window = [&]() {
            return decode_chunks(
                ctx.get(), tokens, 1, (uint32_t) window, capture_chunk);
        };

        // Warm all capture graph shapes, then recreate the window outside the
        // timed region while retaining fixed-tape and scheduler allocations.
        if (!prepare_window() || !fill_window() ||
            !prepare_window()) {
            fprintf(stderr, "tape window warmup/reset failed\n");
            return 1;
        }
        tape_fill_us = time_us(ctx.get(), fill_window);
        if (tape_fill_us < 0) {
            fprintf(stderr, "timed tape window fill failed\n");
            return 1;
        }

        auto & window_state = *ctx->dflash_capture->windows[0];
        if (window_state.count != window ||
            window_state.frontier_pos - window_state.boundary_pos != window) {
            fprintf(stderr, "tape window did not retain W=%d records\n", window);
            return 1;
        }

        // Four complete q-token cycles alternate both private copies twice:
        // first execution/direct warmup, then CUDA capture+instantiate. Timed
        // steady-state therefore measures graph launches rather than setup.
        const uint32_t steady_warm_pos = (uint32_t) window + 1;
        for (uint32_t i = 0; i < steady_warm_tokens; ++i) {
            if (!decode_range(ctx.get(), tokens, steady_warm_pos + i, 1)) {
                fprintf(stderr, "tape steady boundary warmup failed\n");
                return 1;
            }
        }
        if (window_state.count != window) {
            fprintf(stderr, "tape steady boundary warmup failed\n");
            return 1;
        }
        const uint32_t steady_begin =
            steady_warm_pos + steady_warm_tokens;
        reset_window_profile(window_state);
        window_state.profile_timing = true;
        tape_steady_us = time_us(ctx.get(), [&]() {
            for (uint32_t i = 0; i < steady_tokens; ++i) {
                if (!decode_range(ctx.get(), tokens, steady_begin + i, 1)) {
                    return false;
                }
            }
            return true;
        });
        window_state.profile_timing = false;
        tape_phase_append_us = window_state.profile_append_us;
        tape_phase_advance_us = window_state.profile_advance_us;
        tape_phase_stage_us = window_state.profile_stage_us;
        tape_phase_apply_us = window_state.profile_apply_us;
        tape_phase_advance_calls = window_state.profile_advance_calls;
        if (tape_steady_us < 0) {
            fprintf(stderr, "tape steady decode/advance failed\n");
            return 1;
        }

        const auto & rolling = *ctx->dflash_capture->windows[0];
        if (rolling.count != window ||
            rolling.frontier_pos - rolling.boundary_pos != rolling.count ||
            rolling.capacity != physical_capacity) {
            fprintf(stderr, "steady tape window lost capacity\n");
            return 1;
        }

        // Recreate the exact ring age, q-phase, and physical head for every
        // observed cached frontier. This includes early turns where the W=289
        // ring has never filled.
        if (!prepare_window()) {
            fprintf(stderr, "could not reset tape for observed-state sweep\n");
            return 1;
        }
        std::map<int, std::vector<const depth_bin *>> observations;
        int max_total_records = 0;
        for (const auto & bin : depth_bins) {
            const int total_records = bin.cached - 1;
            observations[total_records].push_back(&bin);
            max_total_records = std::max(max_total_records, total_records);
        }
        for (int total_records = 1;
             total_records <= max_total_records;
             ++total_records) {
            if (!decode_range(
                    ctx.get(), tokens, (uint32_t) total_records, 1)) {
                fprintf(stderr,
                    "observed-state tape capture failed at record %d\n",
                    total_records);
                return 1;
            }
            auto found = observations.find(total_records);
            if (found == observations.end()) {
                continue;
            }
            const auto & observed_window = *ctx->dflash_capture->windows[0];
            for (const depth_bin * bin : found->second) {
                if (observed_window.count != bin->retained_records ||
                    observed_window.frontier_pos -
                            observed_window.boundary_pos !=
                        bin->retained_records) {
                    fprintf(stderr,
                        "observed ring disagrees at cached=%d depth=%d "
                        "(got %d records, expected %d)\n",
                        bin->cached, bin->depth, observed_window.count,
                        bin->retained_records);
                    return 1;
                }
                const llama_pos target =
                    observed_window.frontier_pos - bin->depth;
                const int replay_records =
                    (int) (target - observed_window.boundary_pos);
                if (replay_records != bin->replay_records) {
                    fprintf(stderr,
                        "observed replay count disagrees at cached=%d depth=%d\n",
                        bin->cached, bin->depth);
                    return 1;
                }
                // Warm this exact head/target before timing; graph descriptors
                // are rebuilt every call, while persistent scratch grows to
                // the maximum shape encountered by the sweep.
                if (!llama_dflash_window_reconstruct_seq(
                        ctx.get(), 0, target) ||
                    !llama_dflash_window_install_reconstructed(
                        ctx.get(), 0, target)) {
                    fprintf(stderr,
                        "tape restore warmup failed at cached=%d depth=%d\n",
                        bin->cached, bin->depth);
                    return 1;
                }
                constexpr int reps = 2;
                double best = std::numeric_limits<double>::infinity();
                for (int rep = 0; rep < reps; ++rep) {
                    const double elapsed = time_us(ctx.get(), [&]() {
                        return llama_dflash_window_reconstruct_seq(
                                   ctx.get(), 0, target) &&
                               llama_dflash_window_install_reconstructed(
                                   ctx.get(), 0, target);
                    });
                    if (elapsed < 0) {
                        fprintf(stderr,
                            "tape restore failed at cached=%d depth=%d\n",
                            bin->cached, bin->depth);
                        return 1;
                    }
                    best = std::min(best, elapsed);
                }
                tape_restore_us[{ bin->cached, bin->depth }] = best;
            }
            // A measured install moves the live recurrent row behind the
            // frontier. Put it back before capturing the next observed token.
            const llama_pos frontier = observed_window.frontier_pos;
            if (!llama_dflash_window_reconstruct_seq(
                    ctx.get(), 0, frontier) ||
                !llama_dflash_window_install_reconstructed(
                    ctx.get(), 0, frontier)) {
                fprintf(stderr,
                    "could not restore observed frontier at record %d\n",
                    total_records);
                return 1;
            }
        }

        // Account bytes only after every observed replay shape has run.
        tape_core = core_bytes(ctx.get());
        tape_extra = measure_tape_bytes(ctx.get());
    }

    std::vector<latency_bin> latency_bins;
    latency_bins.reserve(depth_bins.size());
    for (const auto & bin : depth_bins) {
        latency_bins.push_back({
            bin.cached,
            bin.depth,
            bin.retained_records,
            bin.events,
            bin.replay_records,
            plane_select_us.count(bin.depth) ? plane_select_us[bin.depth] : -1.0,
            tape_restore_us.at({ bin.cached, bin.depth }),
            full_forward_us.at(bin.replay_records),
        });
    }

    const size_t plane_incremental =
        plane_core > baseline_core ? plane_core - baseline_core : 0;
    const size_t tape_incremental_core =
        tape_core > baseline_core ? tape_core - baseline_core : 0;
    const size_t tape_incremental =
        tape_incremental_core + tape_extra.total();

    fprintf(stdout,
        "G5_CONFIG histogram=%s events=%d W=%d physical_capacity=%d "
        "advance_batch=%d boundary_graph=%d mean_retained_count=%.3f "
        "chunk=%u steady_tokens=%u record_codec=%s\n",
        hist_path, events, window, physical_capacity,
        advance_batch, boundary_graph_cache_enabled() ? 1 : 0,
        mean_retained_count,
        capture_chunk, steady_tokens,
        f16_window_enabled() ? "f16" : "f32");
    fprintf(stdout,
        "G5_CAPABILITY plane_refresh_scoped=1 plane_survives_one_token=%d "
        "tape_window_depth=%d\n",
        plane_edit_capable ? 1 : 0, window);
    fprintf(stdout,
        "G5_BYTES baseline_core=%zu plane_core=%zu plane_incremental=%zu "
        "tape_core=%zu tape_incremental_core=%zu tape_fixed=%zu "
        "tape_ring_boundary=%zu tape_scratch=%zu tape_other_device=%zu "
        "tape_host_payload=%zu tape_host_metadata=%zu tape_incremental_total=%zu\n",
        baseline_core, plane_core, plane_incremental,
        tape_core, tape_incremental_core, tape_extra.fixed_tape,
        tape_extra.ring_boundary, tape_extra.replay_scratch,
        tape_extra.other_device, tape_extra.host_payload,
        tape_extra.host_metadata, tape_incremental);
    fprintf(stdout,
        "G5_CAPTURE_US baseline_plane_batch=%.3f plane_fill=%.3f "
        "plane_overhead=%.3f baseline_tape_chunks=%.3f tape_fill=%.3f "
        "tape_fill_overhead=%.3f steady_tokens=%u baseline_steady_per_token=%.3f "
        "tape_steady_per_token=%.3f tape_steady_overhead_per_token=%.3f\n",
        baseline_plane_batch_us, plane_capture_us,
        plane_capture_us - baseline_plane_batch_us,
        baseline_fill_us, tape_fill_us, tape_fill_us - baseline_fill_us,
        steady_tokens,
        baseline_steady_us / steady_tokens,
        tape_steady_us / steady_tokens,
        (tape_steady_us - baseline_steady_us) / steady_tokens);
    fprintf(stdout,
        "G5_CAPTURE_PHASE append_us_per_token=%.3f "
        "advance_us_per_token=%.3f stage_us_per_token=%.3f "
        "apply_us_per_token=%.3f advance_calls=%" PRIu64 "\n",
        (double) tape_phase_append_us / steady_tokens,
        (double) tape_phase_advance_us / steady_tokens,
        (double) tape_phase_stage_us / steady_tokens,
        (double) tape_phase_apply_us / steady_tokens,
        tape_phase_advance_calls);

    for (const auto & bin : latency_bins) {
        fprintf(stdout,
            "G5_DEPTH cached=%d depth=%d retained_records=%d events=%d "
            "replay_records=%d "
            "plane_select_us=%.3f tape_restore_us=%.3f "
            "full_forward_us=%.3f\n",
            bin.cached, bin.depth, bin.retained_records, bin.events,
            bin.replay_records,
            bin.plane_us, bin.tape_us, bin.full_forward_us);
    }
    fprintf(stdout,
        "G5_WEIGHTED_US plane_select=%.3f tape_restore=%.3f "
        "full_forward=%.3f\n",
        weighted_mean(latency_bins, &latency_bin::plane_us),
        weighted_mean(latency_bins, &latency_bin::tape_us),
        weighted_mean(latency_bins, &latency_bin::full_forward_us));

    const double steady_overhead_per_token =
        (tape_steady_us - baseline_steady_us) / steady_tokens;
    const double captured_tokens_per_edit =
        events > 0
            ? ((double) appended_tokens +
               (double) generated_per_request * request_events) / events
            : 0.0;
    const double weighted_tape_restore =
        weighted_mean(latency_bins, &latency_bin::tape_us);
    const double weighted_full_forward =
        weighted_mean(latency_bins, &latency_bin::full_forward_us);
    const double tape_amortized_edit =
        weighted_tape_restore +
        steady_overhead_per_token * captured_tokens_per_edit;
    const double break_even_tokens =
        steady_overhead_per_token > 0.0
            ? (weighted_full_forward - weighted_tape_restore) /
                steady_overhead_per_token
            : std::numeric_limits<double>::infinity();
    fprintf(stdout,
        "G5_EDIT_AMORTIZED_US requests=%d warm_edits=%d appended_tokens=%" PRId64
        " generated_per_request=%d captured_tokens_per_edit=%.3f "
        "steady_overhead_per_token=%.3f tape_total_per_edit=%.3f "
        "full_forward_per_edit=%.3f break_even_tokens_per_edit=%.3f\n",
        request_events, events, appended_tokens,
        generated_per_request, captured_tokens_per_edit,
        steady_overhead_per_token, tape_amortized_edit,
        weighted_full_forward, break_even_tokens);

    if (tape_incremental == 0 || plane_edit_capable ||
        weighted_mean(latency_bins, &latency_bin::tape_us) <= 0.0) {
        fprintf(stderr, "Gate-5 harness invariant failed\n");
        return 1;
    }

    // Shallow control stratum. This compares complete recurrent-side cycles
    // (verify capture plus rollback/install when rejection is non-zero) using
    // the exact observed (draft, rejected suffix) weights.
    size_t spec_baseline_core = 0;
    size_t spec_plane_core = 0;
    size_t spec_tape_core = 0;
    tape_bytes spec_tape_extra;
    double spec_baseline_weighted = 0.0;
    double spec_plane_weighted = 0.0;
    double spec_tape_weighted = 0.0;
    const uint32_t spec_batch = (uint32_t) spec_window + 1;
    const int spec_capacity = spec_window + 1; // sampled row + draft rows

    {
        auto ctx = make_ctx(params, model, 0, spec_batch);
        if (!ctx) {
            fprintf(stderr, "failed to create speculative baseline context\n");
            return 1;
        }
        for (const auto & bin : spec_bins) {
            auto prepare = [&]() {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                return decode_range(ctx.get(), tokens, 0, 1);
            };
            auto verify = [&]() {
                return decode_range(
                    ctx.get(), tokens, 1, (uint32_t) bin.draft + 1);
            };
            if (!prepare() || !verify() || !prepare()) {
                fprintf(stderr, "spec baseline warmup failed at draft=%d\n", bin.draft);
                return 1;
            }
            const double elapsed = time_us(ctx.get(), verify);
            if (elapsed < 0) {
                fprintf(stderr, "spec baseline timing failed at draft=%d\n", bin.draft);
                return 1;
            }
            spec_baseline_weighted += elapsed * bin.events;
        }
        spec_baseline_core = core_bytes(ctx.get());
    }

    {
        auto ctx = make_ctx(
            params, model, (uint32_t) spec_window, spec_batch);
        if (!ctx) {
            fprintf(stderr, "failed to create speculative plane context\n");
            return 1;
        }
        auto * recurrent = get_recurrent(ctx.get());
        if (!recurrent) {
            fprintf(stderr, "spec plane context exposes no recurrent memory\n");
            return 1;
        }
        for (const auto & bin : spec_bins) {
            auto prepare = [&]() {
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                return decode_range(ctx.get(), tokens, 0, 1);
            };
            auto cycle = [&]() {
                if (!decode_range(
                        ctx.get(), tokens, 1, (uint32_t) bin.draft + 1)) {
                    return false;
                }
                if (recurrent->rollback_valid_depth[0] !=
                    (uint32_t) bin.draft) {
                    return false;
                }
                if (bin.rollback == 0) {
                    return true;
                }
                const llama_pos frontier = bin.draft + 1;
                return recurrent->seq_rm(
                    0, frontier - bin.rollback + 1, -1);
            };
            if (!prepare() || !cycle() || !prepare()) {
                fprintf(stderr,
                    "spec plane warmup failed at draft=%d rollback=%d\n",
                    bin.draft, bin.rollback);
                return 1;
            }
            const double elapsed = time_us(ctx.get(), cycle);
            if (elapsed < 0) {
                fprintf(stderr,
                    "spec plane timing failed at draft=%d rollback=%d\n",
                    bin.draft, bin.rollback);
                return 1;
            }
            spec_plane_weighted += elapsed * bin.events;
        }
        spec_plane_core = core_bytes(ctx.get());
    }

    {
        auto ctx = make_ctx(params, model, 0, spec_batch);
        if (!ctx) {
            fprintf(stderr, "failed to create speculative tape context\n");
            return 1;
        }
        llama_set_dflash_capture(ctx.get(), nullptr, 0);
        llama_dflash_allocate_slots(ctx.get(), 1);
        llama_set_tape_recording(ctx.get(), true);
        if (!llama_dflash_tape_replay_available(ctx.get())) {
            fprintf(stderr, "spec lossless GPU tape replay is unavailable\n");
            return 1;
        }

        auto prepare = [&]() {
            if (ctx->dflash_capture->window_pending.active) {
                return false;
            }
            dflash_window * rolling =
                ctx->dflash_capture->windows.empty()
                    ? nullptr : ctx->dflash_capture->windows[0].get();
            if (rolling) {
                rolling->enabled = false;
            }
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            if (!decode_range(ctx.get(), tokens, 0, 1)) {
                return false;
            }
            if (!rolling) {
                return enable_window(
                    ctx.get(), 0, spec_capacity);
            }

            // Capacity includes sampled+draft, so the warm cycle never advances
            // the published boundary. Reuse it and its replay scratch while
            // resetting only ring metadata for the identical prefix.
            rolling->head = 0;
            rolling->count = 0;
            rolling->boundary_pos = 0;
            rolling->frontier_pos = 0;
            rolling->reconstructed_idx = -1;
            rolling->reconstructed_pos = -1;
            rolling->last_publish_failed = false;
            for (auto & record : rolling->records) {
                record = {};
            }
            rolling->enabled = true;
            return true;
        };

        for (const auto & bin : spec_bins) {
            auto cycle = [&]() {
                if (!decode_range(
                        ctx.get(), tokens, 1, (uint32_t) bin.draft + 1)) {
                    return false;
                }
                if (bin.rollback == 0) {
                    return true;
                }
                auto * rolling =
                    ctx->dflash_capture->windows[0].get();
                if (!rolling) {
                    return false;
                }
                const llama_pos target =
                    rolling->frontier_pos - bin.rollback;
                return llama_dflash_window_reconstruct_seq(
                           ctx.get(), 0, target) &&
                       llama_dflash_window_install_reconstructed(
                           ctx.get(), 0, target);
            };
            if (!prepare() || !cycle() || !prepare()) {
                fprintf(stderr,
                    "spec tape warmup failed at draft=%d rollback=%d\n",
                    bin.draft, bin.rollback);
                return 1;
            }
            const double elapsed = time_us(ctx.get(), cycle);
            if (elapsed < 0) {
                fprintf(stderr,
                    "spec tape timing failed at draft=%d rollback=%d\n",
                    bin.draft, bin.rollback);
                return 1;
            }
            spec_tape_weighted += elapsed * bin.events;
        }
        spec_tape_core = core_bytes(ctx.get());
        spec_tape_extra = measure_tape_bytes(ctx.get());
    }

    spec_baseline_weighted /= spec_events;
    spec_plane_weighted /= spec_events;
    spec_tape_weighted /= spec_events;
    const size_t spec_plane_incremental =
        spec_plane_core > spec_baseline_core
            ? spec_plane_core - spec_baseline_core : 0;
    const size_t spec_tape_incremental =
        (spec_tape_core > spec_baseline_core
             ? spec_tape_core - spec_baseline_core : 0) +
        spec_tape_extra.total();

    fprintf(stdout,
        "G5_SPEC_CONFIG histogram=%s cycles=%d draft_max=%d tape_capacity=%d "
        "record_codec=%s\n",
        spec_hist_path, spec_events, spec_window, spec_capacity,
        f16_window_enabled() ? "f16" : "f32");
    fprintf(stdout,
        "G5_SPEC_BYTES baseline_core=%zu plane_incremental=%zu "
        "tape_incremental=%zu\n",
        spec_baseline_core, spec_plane_incremental,
        spec_tape_incremental);
    fprintf(stdout,
        "G5_SPEC_WEIGHTED_US baseline=%.3f plane=%.3f tape=%.3f "
        "plane_overhead=%.3f tape_overhead=%.3f\n",
        spec_baseline_weighted, spec_plane_weighted,
        spec_tape_weighted,
        spec_plane_weighted - spec_baseline_weighted,
        spec_tape_weighted - spec_baseline_weighted);

    if (spec_plane_incremental == 0 ||
        spec_tape_incremental == 0 ||
        spec_baseline_weighted <= 0.0 ||
        spec_plane_weighted <= 0.0 ||
        spec_tape_weighted <= 0.0) {
        fprintf(stderr, "Gate-5 speculative harness invariant failed\n");
        return 1;
    }

    fprintf(stdout,
        "G5_MEASURED PASS collection complete; gate verdict requires "
        "Dorei values, not a hard-coded threshold\n");
    return 0;
}
