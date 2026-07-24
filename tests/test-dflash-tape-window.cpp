#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama-context.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_context_ptr make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 0; // Gate 3 tests the replacement window, not rollback planes.
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) 16);
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) 16);
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
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         pos) {
    return decode_range(ctx, tokens, pos, 1);
}

struct state_piece {
    int layer;
    char kind;
    std::vector<float> values;
};

static bool read_tensor(
        ggml_tensor *              tensor,
        size_t                     offset,
        int                        layer,
        char                       kind,
        std::vector<state_piece> & out) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
        fprintf(stderr, "layer %d state %c is not plain F32\n", layer, kind);
        return false;
    }
    state_piece piece = { layer, kind, std::vector<float>((size_t) tensor->ne[0]) };
    ggml_backend_tensor_get(
        tensor, piece.values.data(), offset, piece.values.size() * sizeof(float));
    out.push_back(std::move(piece));
    return true;
}

static bool read_live_state(
        llama_context *            ctx,
        llama_memory_recurrent *   mem,
        std::vector<state_piece> & out) {
    auto & window = *ctx->dflash_capture->window;
    const int32_t tail = mem->cells[window.seq_id].tail;
    if (tail < 0) {
        return false;
    }
    const auto & cell = mem->cells[tail];
    const uint32_t base_row = cell.src >= 0 ? (uint32_t) cell.src : (uint32_t) tail;
    const uint32_t row = mem->rs_idx[window.seq_id] * mem->size + base_row;

    ctx->synchronize();
    out.clear();
    for (int il : window.layer_ids) {
        if (!read_tensor(mem->r_l[il], (size_t) row * mem->r_l[il]->nb[1], il, 'r', out) ||
            !read_tensor(mem->s_l[il], (size_t) row * mem->s_l[il]->nb[1], il, 's', out)) {
            return false;
        }
    }
    return !out.empty();
}

static bool read_window_state(
        llama_context *            ctx,
        int                        copy,
        std::vector<state_piece> & out) {
    auto & window = *ctx->dflash_capture->window;
    if (copy < 0 || copy > 1) {
        return false;
    }

    ctx->synchronize();
    out.clear();
    for (size_t li = 0; li < window.layers.size(); ++li) {
        const int il = window.layer_ids[li];
        auto & layer = window.layers[li];
        if (!read_tensor(layer.r[copy], 0, il, 'r', out) ||
            !read_tensor(layer.s[copy], 0, il, 's', out)) {
            return false;
        }
    }
    return !out.empty();
}

static bool states_bit_equal(
        const std::vector<state_piece> & lhs,
        const std::vector<state_piece> & rhs,
        const char *                     what) {
    if (lhs.size() != rhs.size()) {
        fprintf(stderr, "%s: state piece count differs (%zu != %zu)\n",
                what, lhs.size(), rhs.size());
        return false;
    }

    for (size_t p = 0; p < lhs.size(); ++p) {
        const auto & a = lhs[p];
        const auto & b = rhs[p];
        if (a.layer != b.layer || a.kind != b.kind || a.values.size() != b.values.size()) {
            fprintf(stderr, "%s: state metadata differs at piece %zu\n", what, p);
            return false;
        }
        if (std::memcmp(a.values.data(), b.values.data(),
                        a.values.size() * sizeof(float)) == 0) {
            continue;
        }

        size_t first = 0;
        while (first < a.values.size() &&
               std::memcmp(&a.values[first], &b.values[first], sizeof(float)) == 0) {
            ++first;
        }
        uint32_t a_bits = 0;
        uint32_t b_bits = 0;
        std::memcpy(&a_bits, &a.values[first], sizeof(a_bits));
        std::memcpy(&b_bits, &b.values[first], sizeof(b_bits));

        double max_abs = 0.0;
        for (size_t i = 0; i < a.values.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs((double) a.values[i] - (double) b.values[i]));
        }
        fprintf(stderr,
                "%s: layer %d state %c differs at float %zu: "
                "%.9g (0x%08x) != %.9g (0x%08x), max_abs=%g\n",
                what, a.layer, a.kind, first,
                (double) a.values[first], a_bits,
                (double) b.values[first], b_bits, max_abs);
        return false;
    }
    return true;
}

static bool ring_is(
        const dflash_window & window,
        llama_pos             boundary,
        int                   head,
        const std::vector<llama_pos> & positions) {
    if (window.boundary_pos != boundary ||
        window.head != head ||
        window.count != (int) positions.size()) {
        return false;
    }
    for (int i = 0; i < window.count; ++i) {
        const int slot = (window.head + i) % window.capacity;
        const auto & record = window.records[slot];
        if (!record.valid || record.seq_id != window.seq_id ||
            record.pos != positions[(size_t) i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params, /*model_only=*/true);
    llama_model * model = llama_init->model();
    if (!model) {
        fprintf(stderr, "%s: failed to init model\n", __func__);
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s: skipping for non-recurrent model\n", __func__);
        return 0;
    }

    auto ctx = make_ctx(params, model);
    if (!ctx) {
        fprintf(stderr, "%s: failed to init context\n", __func__);
        return 1;
    }
    auto * recurrent = get_recurrent(ctx.get());
    if (!recurrent) {
        fprintf(stderr, "%s: skipping because no recurrent cache is exposed\n", __func__);
        return 0;
    }

    ggml_backend_t gpu_backend = ctx->find_gpu_backend();
    ggml_backend_dev_t gpu_device = gpu_backend ? ggml_backend_get_device(gpu_backend) : nullptr;
    const char * gpu_name = gpu_device ? ggml_backend_dev_name(gpu_device) : nullptr;
    if (!gpu_name || !std::strstr(gpu_name, "CUDA")) {
        fprintf(stderr, "%s: skipping because Gate 3 targets single-device CUDA\n", __func__);
        return 0;
    }

    llama_set_dflash_capture(ctx.get(), nullptr, 0);
    llama_dflash_allocate_slots(ctx.get(), 1);
    llama_set_tape_recording(ctx.get(), true);
    if (!llama_dflash_tape_replay_available(ctx.get())) {
        fprintf(stderr, "%s: skipping because lossless single-device GPU tape replay is unavailable\n",
                __func__);
        return 0;
    }

    constexpr uint32_t prefix = 3;
    constexpr int capacity = 3;
    constexpr uint32_t final_pos = 8;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(final_pos + 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    if (!decode_range(ctx.get(), tokens, 0, prefix)) {
        fprintf(stderr, "%s: prefix decode failed\n", __func__);
        return 1;
    }
    llama_synchronize(ctx.get());
    if (!ctx->get_cparams().fused_gdn_ar || !ctx->get_cparams().fused_gdn_ch) {
        fprintf(stderr, "%s: skipping because the CUDA fused-GDN path is not active\n", __func__);
        return 0;
    }
    if (!llama_dflash_window_enable(ctx.get(), 0, capacity)) {
        fprintf(stderr, "%s: failed to enable rolling tape window\n", __func__);
        return 1;
    }

    auto & window = *ctx->dflash_capture->window;
    if (window.boundary_pos != (llama_pos) prefix - 1 ||
        window.frontier_pos != (llama_pos) prefix - 1 ||
        window.head != 0 || window.count != 0) {
        fprintf(stderr, "%s: initial window metadata is wrong\n", __func__);
        return 1;
    }

    std::vector<state_piece> initial_live;
    std::vector<state_piece> initial_boundary;
    if (!read_live_state(ctx.get(), recurrent, initial_live) ||
        !read_window_state(ctx.get(), window.published_idx, initial_boundary) ||
        !states_bit_equal(initial_live, initial_boundary, "initial boundary snapshot")) {
        return 1;
    }

    // Fill the ring exactly: logical records [3,4,5] occupy physical slots [0,1,2].
    for (uint32_t pos = prefix; pos <= 5; ++pos) {
        if (!decode_one(ctx.get(), tokens, pos) ||
            llama_dflash_window_capture_pending(ctx.get())) {
            fprintf(stderr, "%s: normal-decode capture failed at pos %u\n", __func__, pos);
            return 1;
        }
    }
    if (!ring_is(window, 2, 0, { 3, 4, 5 })) {
        fprintf(stderr, "%s: full ring layout is wrong before fault\n", __func__);
        return 1;
    }

    // Snapshot every published bit and all commit metadata. The one-shot fault fires
    // after private b+1 computation and its GPU fence, immediately before publication.
    const int old_published_idx = window.published_idx;
    const llama_pos old_boundary_pos = window.boundary_pos;
    const int old_head = window.head;
    const int old_count = window.count;
    const auto old_record = window.records[window.head];
    std::vector<state_piece> before_fault;
    if (!read_window_state(ctx.get(), old_published_idx, before_fault)) {
        fprintf(stderr, "%s: failed to read pre-fault boundary\n", __func__);
        return 1;
    }

    llama_dflash_window_inject_publish_failure(ctx.get());
    if (!decode_one(ctx.get(), tokens, 6)) {
        fprintf(stderr, "%s: model decode failed while injecting publication fault\n", __func__);
        return 1;
    }
    if (!llama_dflash_window_capture_pending(ctx.get()) ||
        !window.last_publish_failed ||
        window.published_idx != old_published_idx ||
        window.boundary_pos != old_boundary_pos ||
        window.head != old_head ||
        window.count != old_count ||
        !window.records[old_head].valid ||
        window.records[old_head].pos != old_record.pos ||
        window.records[old_head].seq_id != old_record.seq_id) {
        fprintf(stderr, "%s: fault path mutated publication or retired the oldest record\n", __func__);
        return 1;
    }

    std::vector<state_piece> after_fault;
    if (!read_window_state(ctx.get(), window.published_idx, after_fault) ||
        !states_bit_equal(before_fault, after_fault, "failed publication boundary")) {
        return 1;
    }
    if (decode_one(ctx.get(), tokens, 7)) {
        fprintf(stderr, "%s: pending capture did not protect fixed-tape staging\n", __func__);
        return 1;
    }

    // A clean retry first publishes b=3, retires record 3, and writes pending pos 6
    // into the freed physical slot 0. Decoding 7 and 8 then advances head 1->2->0:
    // both record storage and the left edge cross the circular wrap.
    if (!llama_dflash_window_retry_capture(ctx.get()) ||
        llama_dflash_window_capture_pending(ctx.get()) ||
        !ring_is(window, 3, 1, { 4, 5, 6 })) {
        fprintf(stderr, "%s: clean retry did not commit the first wrapped append\n", __func__);
        return 1;
    }
    if (!decode_one(ctx.get(), tokens, 7) ||
        !ring_is(window, 4, 2, { 5, 6, 7 }) ||
        !decode_one(ctx.get(), tokens, 8) ||
        !ring_is(window, 5, 0, { 6, 7, 8 })) {
        fprintf(stderr, "%s: left-edge wraparound produced the wrong logical ring\n", __func__);
        return 1;
    }

    std::vector<state_piece> live_final;
    if (!read_live_state(ctx.get(), recurrent, live_final) ||
        !llama_dflash_window_reconstruct(ctx.get(), final_pos)) {
        fprintf(stderr, "%s: failed to reconstruct the frontier after wraparound\n", __func__);
        return 1;
    }
    if (window.reconstructed_pos != (llama_pos) final_pos ||
        window.reconstructed_idx == window.published_idx) {
        fprintf(stderr, "%s: reconstruction was not isolated in the private boundary copy\n", __func__);
        return 1;
    }

    std::vector<state_piece> reconstructed;
    if (!read_window_state(ctx.get(), window.reconstructed_idx, reconstructed) ||
        !states_bit_equal(live_final, reconstructed, "post-wrap frontier reconstruction")) {
        return 1;
    }

    fprintf(stderr,
            "%s: PASS (fault retained b=2 + record 3; wrapped window [5,8] "
            "reconstructed live state bit-for-bit on %s)\n",
            __func__, gpu_name);
    return 0;
}
