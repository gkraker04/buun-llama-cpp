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
    cparams.n_seq_max = 2;
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
        uint32_t                         count,
        llama_seq_id                     seq_id = 0) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        const size_t token_idx =
            (pos + (size_t) std::max(seq_id, 0) * 5) % tokens.size();
        common_batch_add(
            batch, tokens[token_idx], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_two_equal_ranges(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin_0,
        uint32_t                         begin_1,
        uint32_t                         count,
        bool                             reverse_owners = false) {
    llama_batch batch = llama_batch_init(2 * count, 0, 1);
    auto add_range = [&](llama_seq_id seq_id, uint32_t begin) {
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t pos = begin + i;
            const size_t token_idx =
                (pos + (size_t) std::max(seq_id, 0) * 5) % tokens.size();
            common_batch_add(
                batch, tokens[token_idx], pos, { seq_id }, i + 1 == count);
        }
    };
    if (reverse_owners) {
        add_range(1, begin_1);
        add_range(0, begin_0);
    } else {
        add_range(0, begin_0);
        add_range(1, begin_1);
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
        llama_seq_id                seq_id,
        std::vector<state_piece> & out) {
    auto & window = *ctx->dflash_capture->windows[seq_id];
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
        llama_seq_id                seq_id,
        int                        copy,
        std::vector<state_piece> & out) {
    auto & window = *ctx->dflash_capture->windows[seq_id];
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

static bool host_qkv_is_packed(
        llama_context *                    ctx,
        int                                n_tokens,
        const std::vector<llama_seq_id> & seq_ids) {
    if (ctx->dflash_capture->tape_layers.empty()) {
        return false;
    }
    for (const auto & tape : ctx->dflash_capture->tape_layers) {
        if (tape.n_tokens != n_tokens ||
            tape.n_seqs != (int) seq_ids.size() ||
            tape.conv_channels <= 0 ||
            tape.qkv_mixed.size() !=
                (size_t) tape.conv_channels * tape.n_tokens * tape.n_seqs) {
            return false;
        }
        for (llama_seq_id seq_id : seq_ids) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                found |= tape.seq_ids[s] == seq_id;
            }
            if (!found) {
                return false;
            }
        }
    }
    return true;
}

static bool ownership_payload_matches(
        llama_context *       ctx,
        const dflash_window & window,
        int                   logical_record,
        int                   source_token) {
    if (logical_record < 0 || logical_record >= window.count) {
        return false;
    }
    const int slot = (window.head + logical_record) % window.capacity;
    auto * tape_gpu = ctx->dflash_capture->tapes[window.seq_id].get();
    if (!tape_gpu) {
        return false;
    }
    for (size_t li = 0; li < window.layers.size(); ++li) {
        const auto & host = ctx->dflash_capture->tape_layers[li];
        const int gpu_li = window.gpu_layer_indices[li];
        if (gpu_li < 0 || source_token < 0 ||
            source_token >= host.n_tokens) {
            return false;
        }
        const auto & src = tape_gpu->layers[gpu_li];
        const auto & dst = window.layers[li];
        const size_t qkv_n = (size_t) host.conv_channels;
        std::vector<float> expected_qkv(qkv_n);
        std::vector<float> got_qkv(qkv_n);
        if (src.qkv) {
            ggml_backend_tensor_get(
                src.qkv, expected_qkv.data(),
                (size_t) source_token * src.qkv->nb[1],
                qkv_n * sizeof(float));
        } else {
            int seq_axis = -1;
            for (int s = 0; s < host.n_seqs; ++s) {
                if (host.seq_ids[s] == window.seq_id) {
                    seq_axis = s;
                    break;
                }
            }
            if (seq_axis < 0) {
                return false;
            }
            const size_t qkv_off =
                ((size_t) seq_axis * host.n_tokens + source_token) * qkv_n;
            std::memcpy(
                expected_qkv.data(), host.qkv_mixed.data() + qkv_off,
                qkv_n * sizeof(float));
        }
        ggml_backend_tensor_get(
            dst.qkv, got_qkv.data(), (size_t) slot * dst.qkv->nb[1],
            qkv_n * sizeof(float));
        if (std::memcmp(
                got_qkv.data(), expected_qkv.data(),
                qkv_n * sizeof(float)) != 0) {
            return false;
        }

        const size_t h_v = (size_t) dst.gate->ne[0];
        std::vector<float> src_values(h_v);
        std::vector<float> dst_values(h_v);
        for (int kind = 0; kind < 2; ++kind) {
            ggml_tensor * src_tensor = kind == 0 ? src.gate : src.beta;
            ggml_tensor * dst_tensor = kind == 0 ? dst.gate : dst.beta;
            ggml_backend_tensor_get(
                src_tensor, src_values.data(),
                (size_t) source_token * src_tensor->nb[2],
                h_v * sizeof(float));
            ggml_backend_tensor_get(
                dst_tensor, dst_values.data(),
                (size_t) slot * dst_tensor->nb[1],
                h_v * sizeof(float));
            if (std::memcmp(
                    src_values.data(), dst_values.data(),
                    h_v * sizeof(float)) != 0) {
                return false;
            }
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
    ggml_backend_t meta_backend = ctx->find_meta_backend();
    ggml_backend_dev_t gpu_device = gpu_backend ? ggml_backend_get_device(gpu_backend) : nullptr;
    const char * gpu_name = gpu_device ? ggml_backend_dev_name(gpu_device) : nullptr;
    const bool tensor_split = meta_backend != nullptr;
    if ((!gpu_name || !std::strstr(gpu_name, "CUDA")) && !tensor_split) {
        fprintf(stderr, "%s: skipping because Gate 3/4 targets CUDA\n", __func__);
        return 0;
    }

    llama_set_dflash_capture(ctx.get(), nullptr, 0);
    llama_dflash_allocate_slots(ctx.get(), 2);
    llama_set_tape_recording(ctx.get(), true);
    if (!llama_dflash_tape_replay_available(ctx.get())) {
        fprintf(stderr, "%s: skipping because lossless single-device GPU tape replay is unavailable\n",
                __func__);
        return 0;
    }

    constexpr uint32_t prefix = 3;
    const int capacity = tensor_split ? 8 : 3;
    constexpr uint32_t final_pos = 8;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(14);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    if (!decode_range(ctx.get(), tokens, 0, prefix)) {
        fprintf(stderr, "%s: seq-0 prefix decode failed\n", __func__);
        return 1;
    }
    if (!decode_range(ctx.get(), tokens, 0, prefix, 1)) {
        fprintf(stderr, "%s: seq-1 prefix decode failed\n", __func__);
        return 1;
    }
    llama_synchronize(ctx.get());
    if (!ctx->get_cparams().fused_gdn_ar || !ctx->get_cparams().fused_gdn_ch) {
        fprintf(stderr, "%s: skipping because the CUDA fused-GDN path is not active\n", __func__);
        return 0;
    }
    if (!llama_dflash_window_enable(ctx.get(), 0, capacity)) {
        fprintf(stderr, "%s: failed to enable seq-0 rolling tape window\n", __func__);
        return 1;
    }
    if (!llama_dflash_window_enable(ctx.get(), 1, capacity)) {
        fprintf(stderr, "%s: failed to enable seq-1 rolling tape window\n", __func__);
        return 1;
    }

    auto & window = *ctx->dflash_capture->windows[0];
    auto & window_1 = *ctx->dflash_capture->windows[1];
    if (window.boundary_pos != (llama_pos) prefix - 1 ||
        window.frontier_pos != (llama_pos) prefix - 1 ||
        window.head != 0 || window.count != 0 ||
        window_1.boundary_pos != (llama_pos) prefix - 1 ||
        window_1.frontier_pos != (llama_pos) prefix - 1 ||
        window_1.head != 0 || window_1.count != 0) {
        fprintf(stderr, "%s: initial window metadata is wrong\n", __func__);
        return 1;
    }

    if (tensor_split) {
        if (!window.ownership_only || !window_1.ownership_only) {
            fprintf(stderr, "%s: tensor-split windows did not enter ownership-trace mode\n",
                    __func__);
            return 1;
        }

        // Multi-device Gate-4 arm: validate the actual meta-tape gather/transfer
        // against both host ring payloads, not only the echoed input metadata.
        if (!decode_two_equal_ranges(ctx.get(), tokens, 3, 3, 2, true) ||
            llama_dflash_window_capture_pending(ctx.get()) ||
            !ring_is(window, 2, 0, { 3, 4 }) ||
            !ring_is(window_1, 2, 0, { 3, 4 }) ||
            !ownership_payload_matches(ctx.get(), window, 0, 0) ||
            !ownership_payload_matches(ctx.get(), window, 1, 1) ||
            !ownership_payload_matches(ctx.get(), window_1, 0, 0) ||
            !ownership_payload_matches(ctx.get(), window_1, 1, 1)) {
            fprintf(stderr, "%s: tensor-split payload ownership/transfer mismatch\n",
                    __func__);
            return 1;
        }

        llama_dflash_window_set_speculative(ctx.get(), true);
        if (!decode_two_equal_ranges(ctx.get(), tokens, 5, 5, 3, true) ||
            !llama_dflash_window_capture_pending(ctx.get()) ||
            !llama_dflash_window_commit(ctx.get(), 0, 2) ||
            !llama_dflash_window_capture_pending(ctx.get()) ||
            !llama_dflash_window_commit(ctx.get(), 1, 1) ||
            llama_dflash_window_capture_pending(ctx.get()) ||
            !ring_is(window, 2, 0, { 3, 4, 5, 6 }) ||
            !ring_is(window_1, 2, 0, { 3, 4, 5 }) ||
            !ownership_payload_matches(ctx.get(), window, 2, 0) ||
            !ownership_payload_matches(ctx.get(), window, 3, 1) ||
            !ownership_payload_matches(ctx.get(), window_1, 2, 0)) {
            fprintf(stderr, "%s: tensor-split speculative ownership mismatch\n", __func__);
            return 1;
        }
        llama_dflash_window_set_speculative(ctx.get(), false);
        fprintf(stderr,
                "%s: PASS (tensor-split normal/speculative multi-sequence "
                "positions and minimal-F32 payload transfers preserve ownership)\n",
                __func__);
        return 0;
    }

    std::vector<state_piece> initial_live;
    std::vector<state_piece> initial_boundary;
    if (!read_live_state(ctx.get(), recurrent, 0, initial_live) ||
        !read_window_state(ctx.get(), 0, window.published_idx, initial_boundary) ||
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
    if (!read_window_state(ctx.get(), 0, old_published_idx, before_fault)) {
        fprintf(stderr, "%s: failed to read pre-fault boundary\n", __func__);
        return 1;
    }

    llama_dflash_window_inject_publish_failure_seq(ctx.get(), 0);
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
    if (!read_window_state(ctx.get(), 0, window.published_idx, after_fault) ||
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
    if (!read_live_state(ctx.get(), recurrent, 0, live_final) ||
        !llama_dflash_window_reconstruct_seq(ctx.get(), 0, final_pos)) {
        fprintf(stderr, "%s: failed to reconstruct the frontier after wraparound\n", __func__);
        return 1;
    }
    if (window.reconstructed_pos != (llama_pos) final_pos ||
        window.reconstructed_idx == window.published_idx) {
        fprintf(stderr, "%s: reconstruction was not isolated in the private boundary copy\n", __func__);
        return 1;
    }

    std::vector<state_piece> reconstructed;
    if (!read_window_state(ctx.get(), 0, window.reconstructed_idx, reconstructed) ||
        !states_bit_equal(live_final, reconstructed, "post-wrap frontier reconstruction")) {
        return 1;
    }

    // Gate 4, normal path: one decode carries two contiguous tokens for each of
    // two sequences. The fixed tape packs QKV sequence-major and routes gate/beta
    // to per-sequence GPU tapes; both rolling rings must retain the exact owner
    // and absolute position after their independent left-edge advances.
    // Reverse first-owner order deliberately: recurrent split_equal emits one
    // ubatch per owner, so the callback accumulator must join both QKV chunks
    // into one transaction-wide packed image.
    if (!decode_two_equal_ranges(ctx.get(), tokens, 9, 3, 2, true) ||
        llama_dflash_window_capture_pending(ctx.get()) ||
        !host_qkv_is_packed(ctx.get(), 2, { 0, 1 }) ||
        !ring_is(window, 7, 2, { 8, 9, 10 }) ||
        !ring_is(window_1, 2, 0, { 3, 4 })) {
        fprintf(stderr, "%s: multi-token/multi-sequence normal capture routed incorrectly\n",
                __func__);
        return 1;
    }

    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        auto & seq_window = *ctx->dflash_capture->windows[seq_id];
        std::vector<state_piece> live;
        std::vector<state_piece> rebuilt;
        if (!read_live_state(ctx.get(), recurrent, seq_id, live) ||
            !llama_dflash_window_reconstruct_seq(
                ctx.get(), seq_id, seq_window.frontier_pos) ||
            !read_window_state(
                ctx.get(), seq_id, seq_window.reconstructed_idx, rebuilt) ||
            !states_bit_equal(live, rebuilt, "multi-sequence frontier reconstruction")) {
            return 1;
        }
    }

    // Gate 4, speculative path: decode three candidates for each owner, then
    // commit different accepted prefixes. An undecided owner keeps staging
    // live and blocks overwrite; rejected suffixes never receive ring metadata.
    llama_dflash_window_set_speculative(ctx.get(), true);
    if (!decode_two_equal_ranges(ctx.get(), tokens, 11, 5, 3, true) ||
        !llama_dflash_window_capture_pending(ctx.get()) ||
        !ring_is(window, 7, 2, { 8, 9, 10 }) ||
        !ring_is(window_1, 2, 0, { 3, 4 })) {
        fprintf(stderr, "%s: speculative candidates published before acceptance\n", __func__);
        return 1;
    }
    if (decode_one(ctx.get(), tokens, 11)) {
        fprintf(stderr, "%s: unresolved speculative staging did not block overwrite\n", __func__);
        return 1;
    }
    if (!llama_dflash_window_commit(ctx.get(), 0, 2) ||
        !llama_dflash_window_capture_pending(ctx.get()) ||
        !host_qkv_is_packed(ctx.get(), 3, { 0, 1 }) ||
        !ring_is(window, 9, 1, { 10, 11, 12 }) ||
        !ring_is(window_1, 2, 0, { 3, 4 })) {
        fprintf(stderr, "%s: first speculative owner committed the wrong prefix\n", __func__);
        return 1;
    }
    if (!llama_dflash_window_commit(ctx.get(), 1, 1) ||
        llama_dflash_window_capture_pending(ctx.get()) ||
        !ring_is(window, 9, 1, { 10, 11, 12 }) ||
        !ring_is(window_1, 2, 0, { 3, 4, 5 })) {
        fprintf(stderr, "%s: speculative accept/reject ownership is wrong\n", __func__);
        return 1;
    }
    llama_dflash_window_set_speculative(ctx.get(), false);

    fprintf(stderr,
            "%s: PASS (fault retained b=2 + record 3; wrapped window [5,8] "
            "reconstructed live state bit-for-bit; normal multi-token/multi-seq "
            "and speculative accept/reject preserved ownership on %s)\n",
            __func__, gpu_name);
    return 0;
}
