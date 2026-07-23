#include "arg.h"
#include "common.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint32_t n_seq_max = 1) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  n_seq_max * (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, n_seq_max * (cparams.n_rs_seq + 1));
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

static llama_pos get_attention_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    return -1;
}

static bool check_depth(llama_context * ctx, llama_seq_id seq_id, uint32_t expected, const char * label) {
    const auto * recurrent = get_recurrent(ctx);
    if (recurrent == nullptr || seq_id < 0 || (size_t) seq_id >= recurrent->rollback_valid_depth.size()) {
        fprintf(stderr, "%s : cannot read rollback depth for sequence %d\n", label, seq_id);
        return false;
    }
    const uint32_t actual = recurrent->rollback_valid_depth[seq_id];
    if (actual != expected) {
        fprintf(stderr, "%s : rollback depth mismatch for sequence %d (%u != %u)\n",
                label, seq_id, actual, expected);
        return false;
    }
    return true;
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
        common_batch_add(batch, tokens[pos], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_equal_split(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         n_seq_tokens,
        uint32_t                         n_seqs) {
    llama_batch batch = llama_batch_init(n_seq_tokens * n_seqs, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t pos = 0; pos < n_seq_tokens; ++pos) {
            const uint32_t i = s * n_seq_tokens + pos;
            common_batch_add(batch, tokens[i], pos, { (llama_seq_id) s }, pos + 1 == n_seq_tokens);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> save_seq(
        llama_context *       ctx,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq_id, flags));
    const size_t n = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (n != state.size()) {
        state.clear();
    }
    return state;
}

static bool load_seq(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id seq_id) {
    return !state.empty() && llama_state_seq_set_data(ctx, state.data(), state.size(), seq_id) == state.size();
}

static bool seq_state_payload_equal(
        const std::vector<uint8_t> & lhs,
        const std::vector<uint8_t> & rhs) {
    // The in-memory sequence envelope starts with magic + source seq_id. The memory
    // payload that follows must be identical when comparing two different sequence ids.
    constexpr size_t envelope_size = sizeof(uint32_t) + sizeof(llama_seq_id);
    return lhs.size() == rhs.size() && lhs.size() >= envelope_size &&
        std::equal(lhs.begin() + envelope_size, lhs.end(), rhs.begin() + envelope_size);
}

static std::vector<float> copy_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    return logits == nullptr ? std::vector<float>() : std::vector<float>(logits, logits + n_vocab);
}

static bool logits_equal(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const char *               label) {
    constexpr float eps = 1e-5f;
    if (lhs.size() != rhs.size() || lhs.empty()) {
        fprintf(stderr, "%s : missing or differently sized logits\n", label);
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %zu (%g != %g)\n",
                    label, i, (double) lhs[i], (double) rhs[i]);
            return false;
        }
    }
    return true;
}

static bool abort_decode(void *) {
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

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    auto ctx_src      = make_ctx(params, model);
    auto ctx_test     = make_ctx(params, model);
    auto ctx_ref      = make_ctx(params, model);
    auto ctx_parallel = make_ctx(params, model, 3);
    if (!ctx_src || !ctx_test || !ctx_ref || !ctx_parallel) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    auto * recurrent = get_recurrent(ctx_test.get());
    if (recurrent == nullptr || recurrent->n_rs_seq < 3) {
        fprintf(stderr, "%s : skipping because recurrent rollback depth is less than 3\n", __func__);
        return 0;
    }
    const uint32_t n_rs_seq = recurrent->n_rs_seq;

    // Preserve the original regression: a selected rollback plane must serialize as the
    // logical active row and replay identically after restore.
    if (!decode_range(ctx_src.get(), tokens, 0, 4) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : rolled-back checkpoint setup failed\n", __func__);
        return 1;
    }
    const auto rolled_back_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), rolled_back_state, 0) ||
        !decode_range(ctx_src.get(),  tokens, 3, 1) ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !logits_equal(copy_logits(ctx_src.get(), n_vocab), copy_logits(ctx_test.get(), n_vocab),
                "rolled-back checkpoint restore")) {
        fprintf(stderr, "%s : rolled-back checkpoint did not replay identically\n", __func__);
        return 1;
    }
    llama_memory_clear(llama_get_memory(ctx_src.get()),  true);
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);

    // A restore installs only the active row. A following one-token decode writes group 0,
    // so rollback one must fail without changing serialized state or the current logits.
    if (!decode_range(ctx_src.get(), tokens, 0, 3)) {
        fprintf(stderr, "%s : source prefix decode failed\n", __func__);
        return 1;
    }
    const auto restored_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), restored_state, 0) ||
        !check_depth(ctx_test.get(), 0, 0, "restore") ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "restore then one-token decode")) {
        fprintf(stderr, "%s : restore setup failed\n", __func__);
        return 1;
    }

    const auto state_before_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_before_rm = copy_logits(ctx_test.get(), n_vocab);
    const int32_t tail_before_rm = get_recurrent(ctx_test.get())->cells[0].tail;
    const uint32_t rs_before_rm  = get_recurrent(ctx_test.get())->rs_idx[0];
    if (!load_seq(ctx_ref.get(), state_before_rm, 0)) {
        fprintf(stderr, "%s : failed to retain pre-rm reference\n", __func__);
        return 1;
    }
    // Exercise the composite memory operation. For hybrid models, a rejected
    // recurrent rollback must not remove the corresponding attention entries.
    if (llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : stale rollback unexpectedly succeeded after restore + decode\n", __func__);
        return 1;
    }
    const auto state_after_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_after_rm = copy_logits(ctx_test.get(), n_vocab);
    const bool state_unchanged  = state_before_rm == state_after_rm;
    const bool logits_unchanged = logits_equal(logits_before_rm, logits_after_rm, "failed rollback");
    const llama_pos pos_after_rm = llama_memory_seq_pos_max(llama_get_memory(ctx_test.get()), 0);
    const auto * recurrent_after_rm = get_recurrent(ctx_test.get());
    const bool metadata_unchanged = recurrent_after_rm->cells[0].tail == tail_before_rm &&
        recurrent_after_rm->rs_idx[0] == rs_before_rm;
    if (!state_unchanged || !logits_unchanged || !metadata_unchanged || pos_after_rm != 3) {
        fprintf(stderr, "%s : failed-rm details: state=%s, logits=%s, metadata=%s, pos=%d\n",
                __func__, state_unchanged ? "same" : "changed",
                logits_unchanged ? "same" : "changed",
                metadata_unchanged ? "same" : "changed", pos_after_rm);
        fprintf(stderr, "%s : failed rollback mutated state, logits, or position\n", __func__);
        return 1;
    }
    if (!decode_range(ctx_test.get(), tokens, 4, 1) ||
        !decode_range(ctx_ref.get(),  tokens, 4, 1) ||
        !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-failed-rm continuation")) {
        fprintf(stderr, "%s : failed rollback changed continuation\n", __func__);
        return 1;
    }

    // A four-token verify writes the active plane plus rollback planes 1..3.
    // Each rollback is checked independently against a retained prefix reference.
    for (uint32_t rollback = 1; rollback <= 3; ++rollback) {
        llama_memory_clear(llama_get_memory(ctx_test.get()), true);
        llama_memory_clear(llama_get_memory(ctx_ref.get()),  true);

        if (!decode_range(ctx_test.get(), tokens, 0, 2) ||
            !decode_range(ctx_test.get(), tokens, 2, 4) ||
            !check_depth(ctx_test.get(), 0, 3, "four-token verify")) {
            fprintf(stderr, "%s : four-token verify setup failed for rollback %u\n", __func__, rollback);
            return 1;
        }

        const uint32_t rollback_pos = 6 - rollback;
        if (!decode_range(ctx_ref.get(), tokens, 0, rollback_pos) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, rollback_pos, -1) ||
            !decode_range(ctx_test.get(), tokens, rollback_pos, 1) ||
            !decode_range(ctx_ref.get(),  tokens, rollback_pos, 1) ||
            !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                    "bounded rollback reference")) {
            fprintf(stderr, "%s : rollback %u did not match retained reference\n", __func__, rollback);
            return 1;
        }
    }

    // A narrow decode replaces, rather than extends, the valid-depth assignment.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "verify before narrow decode") ||
        !decode_range(ctx_test.get(), tokens, 4, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "verify then narrow decode")) {
        fprintf(stderr, "%s : verify then narrow decode setup failed\n", __func__);
        return 1;
    }
    const auto narrow_state = save_seq(ctx_test.get(), 0);
    if (get_recurrent(ctx_test.get())->seq_rm(0, 4, -1) ||
        narrow_state != save_seq(ctx_test.get(), 0)) {
        fprintf(stderr, "%s : stale rollback succeeded or mutated after narrow decode\n", __func__);
        return 1;
    }

    // The assignment clamps at n_rs_seq, and full removal is never rejected by the guard.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, n_rs_seq + 1) ||
        !check_depth(ctx_test.get(), 0, n_rs_seq, "clamped verify") ||
        !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, -1, -1) ||
        !check_depth(ctx_test.get(), 0, 0, "full removal")) {
        fprintf(stderr, "%s : clamp or full-removal check failed\n", __func__);
        return 1;
    }

    // A successful detectable copy into an empty destination copies the composite state.
    // Give the empty destination synthetic rollback metadata to prove that the copied active
    // row resets it rather than inheriting the source's valid planes.
    if (!decode_range(ctx_parallel.get(), tokens, 0, 4, 1) ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        fprintf(stderr, "%s : successful seq_cp setup failed\n", __func__);
        return 1;
    }
    const auto source_state = save_seq(ctx_parallel.get(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    auto * recurrent_parallel = get_recurrent(ctx_parallel.get());
    recurrent_parallel->set_rs_idx(0, 3);
    recurrent_parallel->rollback_valid_depth[0] = 3;
    const bool copy_succeeded = llama_get_memory(ctx_parallel.get())->try_seq_cp(1, 0, -1, -1);
    const auto destination_state = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (!copy_succeeded ||
        !seq_state_payload_equal(source_state, destination_state) ||
        !check_depth(ctx_parallel.get(), 0, 0, "seq_cp destination") ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        size_t first_diff = 0;
        while (first_diff < source_state.size() &&
               first_diff < destination_state.size() &&
               source_state[first_diff] == destination_state[first_diff]) {
            ++first_diff;
        }
        fprintf(stderr, "%s : successful seq_cp details: result=%s, source=%zu B, destination=%zu B, first_diff=%zu\n",
                __func__, copy_succeeded ? "true" : "false",
                source_state.size(), destination_state.size(), first_diff);
        fprintf(stderr, "%s : successful seq_cp did not copy state or reset destination depth\n", __func__);
        return 1;
    }

    // Fill every recurrent cell, then exercise the recurrent implementation directly.
    // Reserve-before-clear must report exhaustion while preserving the occupied destination
    // byte-for-byte, including its rollback metadata.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 3)) {
        fprintf(stderr, "%s : seq_cp exhaustion setup failed\n", __func__);
        return 1;
    }
    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const auto exhausted_dst_state = save_seq(ctx_parallel.get(), 0);
    const int32_t exhausted_dst_tail = recurrent_parallel->cells[0].tail;
    const llama_pos exhausted_dst_pos = recurrent_parallel->seq_pos_max(0);
    const uint32_t exhausted_dst_rs_idx = recurrent_parallel->rs_idx[0];
    const uint32_t exhausted_dst_depth = recurrent_parallel->rollback_valid_depth[0];
    const uint32_t exhausted_used = recurrent_parallel->used;
    if (recurrent_parallel->try_seq_cp(1, 0, -1, -1) ||
        exhausted_dst_state != save_seq(ctx_parallel.get(), 0) ||
        recurrent_parallel->cells[0].tail != exhausted_dst_tail ||
        recurrent_parallel->seq_pos_max(0) != exhausted_dst_pos ||
        recurrent_parallel->rs_idx[0] != exhausted_dst_rs_idx ||
        recurrent_parallel->rollback_valid_depth[0] != exhausted_dst_depth ||
        recurrent_parallel->used != exhausted_used) {
        fprintf(stderr, "%s : exhausted recurrent seq_cp succeeded or mutated destination\n", __func__);
        return 1;
    }

    // The hybrid fallback cannot roll back the attention copy, so detected failure
    // invalidates both children. The destination is empty and therefore never selectable
    // as a half-copied composite checkpoint.
    llama_memory_t mem_parallel = llama_get_memory(ctx_parallel.get());
    if (dynamic_cast<llama_memory_hybrid *>(mem_parallel) != nullptr ||
        dynamic_cast<llama_memory_hybrid_iswa *>(mem_parallel) != nullptr) {
        if (mem_parallel->try_seq_cp(1, 0, -1, -1) ||
            recurrent_parallel->get_cell_count(0) != 0 ||
            recurrent_parallel->seq_pos_max(0) != -1 ||
            get_attention_pos_max(ctx_parallel.get(), 0) != -1 ||
            llama_memory_seq_pos_max(mem_parallel, 0) != -1 ||
            !check_depth(ctx_parallel.get(), 0, 0, "failed hybrid seq_cp destination")) {
            fprintf(stderr, "%s : failed hybrid seq_cp left an incoherent destination\n", __func__);
            return 1;
        }
    }

    // Equal-split batching still publishes each participating sequence independently.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 2) ||
        !check_depth(ctx_parallel.get(), 0, 3, "equal split seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "equal split seq 1") ||
        !decode_range(ctx_parallel.get(), tokens, 4, 1, 0) ||
        !check_depth(ctx_parallel.get(), 0, 0, "narrow seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "untouched seq 1")) {
        fprintf(stderr, "%s : per-sequence equal-split assignment failed\n", __func__);
        return 1;
    }

    // Invalidation happens before graph submission, so an aborted decode cannot leave a
    // positive depth that is not backed by a successfully committed write.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "pre-abort verify")) {
        fprintf(stderr, "%s : abort setup failed\n", __func__);
        return 1;
    }
    llama_set_abort_callback(ctx_test.get(), abort_decode, nullptr);
    llama_batch abort_batch = llama_batch_init(1, 0, 1);
    common_batch_add(abort_batch, tokens[4], 4, { 0 }, true);
    const int32_t abort_result = llama_decode(ctx_test.get(), abort_batch);
    llama_batch_free(abort_batch);
    llama_set_abort_callback(ctx_test.get(), nullptr, nullptr);
    if (abort_result != 2 || !check_depth(ctx_test.get(), 0, 0, "aborted decode")) {
        fprintf(stderr, "%s : aborted decode returned %d or retained positive depth\n",
                __func__, abort_result);
        return 1;
    }

    fprintf(stderr, "%s : recurrent rollback-plane validity checks passed\n", __func__);
    return 0;
}
