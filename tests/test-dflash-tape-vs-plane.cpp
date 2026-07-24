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
    cparams.n_seq_max = 2; // live sequence + immutable pre-verify backup
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  2 * (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, 2 * (cparams.n_rs_seq + 1));
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
        common_batch_add(batch, tokens[pos], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct state_piece {
    int layer;
    char kind;
    std::vector<float> values;
};

static bool read_state(
        llama_memory_recurrent * mem,
        llama_seq_id             seq_id,
        std::vector<state_piece> & out) {
    if (seq_id < 0 || (uint32_t) seq_id >= mem->size) {
        return false;
    }

    const int32_t tail = mem->cells[seq_id].tail;
    if (tail < 0) {
        return false;
    }

    const auto & cell = mem->cells[tail];
    const uint32_t base_row = cell.src >= 0 ? (uint32_t) cell.src : (uint32_t) tail;
    const uint32_t row = mem->rs_idx[seq_id] * mem->size + base_row;

    out.clear();
    auto read_row = [&](ggml_tensor * tensor, int il, char kind) {
        if (tensor == nullptr) {
            return true;
        }
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
            fprintf(stderr, "layer %d state %c is not plain F32\n", il, kind);
            return false;
        }

        state_piece piece = { il, kind, std::vector<float>((size_t) tensor->ne[0]) };
        ggml_backend_tensor_get(
                tensor,
                piece.values.data(),
                (size_t) row * tensor->nb[1],
                piece.values.size() * sizeof(float));
        out.push_back(std::move(piece));
        return true;
    };

    for (int il = 0; il < (int) mem->r_l.size(); ++il) {
        if (!read_row(mem->r_l[il], il, 'r') || !read_row(mem->s_l[il], il, 's')) {
            return false;
        }
    }
    return !out.empty();
}

static bool states_bit_equal(
        const std::vector<state_piece> & plane,
        const std::vector<state_piece> & replay) {
    if (plane.size() != replay.size()) {
        fprintf(stderr, "state piece count differs (%zu != %zu)\n", plane.size(), replay.size());
        return false;
    }

    for (size_t p = 0; p < plane.size(); ++p) {
        const auto & lhs = plane[p];
        const auto & rhs = replay[p];
        if (lhs.layer != rhs.layer || lhs.kind != rhs.kind || lhs.values.size() != rhs.values.size()) {
            fprintf(stderr, "state piece metadata differs at index %zu\n", p);
            return false;
        }
        if (std::memcmp(lhs.values.data(), rhs.values.data(), lhs.values.size() * sizeof(float)) == 0) {
            continue;
        }

        size_t first = 0;
        while (first < lhs.values.size() &&
               std::memcmp(&lhs.values[first], &rhs.values[first], sizeof(float)) == 0) {
            ++first;
        }

        uint32_t lhs_bits = 0;
        uint32_t rhs_bits = 0;
        std::memcpy(&lhs_bits, &lhs.values[first], sizeof(lhs_bits));
        std::memcpy(&rhs_bits, &rhs.values[first], sizeof(rhs_bits));

        double max_abs = 0.0;
        for (size_t i = 0; i < lhs.values.size(); ++i) {
            max_abs = std::max(max_abs, std::fabs((double) lhs.values[i] - (double) rhs.values[i]));
        }
        fprintf(stderr,
                "layer %d state %c differs at float %zu: %.9g (0x%08x) != %.9g (0x%08x), max_abs=%g\n",
                lhs.layer, lhs.kind, first,
                (double) lhs.values[first], lhs_bits,
                (double) rhs.values[first], rhs_bits,
                max_abs);
        return false;
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

    // Avoid allocating the ordinary context: this test needs n_rs_seq and a backup sequence.
    common_init_result_ptr llama_init = common_init_from_params(params, /*model_only=*/true);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    auto ctx = make_ctx(params, model);
    if (!ctx) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return 1;
    }

    auto * recurrent = get_recurrent(ctx.get());
    if (recurrent == nullptr || recurrent->n_rs_seq < 3) {
        fprintf(stderr, "%s : skipping because recurrent rollback depth is less than 3\n", __func__);
        return 0;
    }

    ggml_backend_t gpu_backend = ctx->find_gpu_backend();
    ggml_backend_dev_t gpu_device = gpu_backend ? ggml_backend_get_device(gpu_backend) : nullptr;
    const char * gpu_name = gpu_device ? ggml_backend_dev_name(gpu_device) : nullptr;
    if (gpu_name == nullptr || std::strstr(gpu_name, "CUDA") == nullptr) {
        fprintf(stderr, "%s : skipping because this gate is for the single-device CUDA path\n", __func__);
        return 0;
    }

    // No target hidden layers are needed, but set_dflash_capture creates the capture owner
    // required by tape setup. The GPU tape must be installed before the first decode.
    llama_set_dflash_capture(ctx.get(), nullptr, 0);
    llama_dflash_allocate_slots(ctx.get(), 1);
    llama_set_tape_recording(ctx.get(), true);

    if (!llama_dflash_tape_replay_available(ctx.get()) ||
        !ctx->dflash_capture ||
        !ctx->dflash_capture->active_tape()) {
        fprintf(stderr, "%s : skipping because lossless single-device GPU tape replay is unavailable\n", __func__);
        return 0;
    }

    constexpr uint32_t n_prefix   = 3;
    constexpr uint32_t n_verify   = 4;
    constexpr uint32_t n_accepted = 2;
    constexpr uint32_t rollback   = n_verify - n_accepted;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(n_prefix + n_verify);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    if (!decode_range(ctx.get(), tokens, 0, n_prefix)) {
        fprintf(stderr, "%s : prefix decode failed\n", __func__);
        return 1;
    }
    llama_synchronize(ctx.get());

    // Mirror production's backup-sequence dance. Recurrent seq_cp deliberately does not
    // migrate planes; the backup is the active pre-verify row that replay must start from.
    if (!recurrent->try_seq_cp(0, 1, -1, -1)) {
        fprintf(stderr, "%s : failed to create pre-verify recurrent backup\n", __func__);
        return 1;
    }

    if (!decode_range(ctx.get(), tokens, n_prefix, n_verify)) {
        fprintf(stderr, "%s : verify decode failed\n", __func__);
        return 1;
    }
    llama_synchronize(ctx.get());

    if (!ctx->get_cparams().fused_gdn_ar || !ctx->get_cparams().fused_gdn_ch) {
        fprintf(stderr, "%s : skipping because the CUDA fused-GDN path is not active\n", __func__);
        return 0;
    }
    if (recurrent->rollback_valid_depth[0] < rollback) {
        fprintf(stderr, "%s : verify produced rollback depth %u, need %u\n",
                __func__, recurrent->rollback_valid_depth[0], rollback);
        return 1;
    }
    if (ctx->dflash_capture->tape_stage_n_tokens != (int) n_verify) {
        fprintf(stderr, "%s : GPU tape captured %d tokens, expected %u\n",
                __func__, ctx->dflash_capture->tape_stage_n_tokens, n_verify);
        return 1;
    }
    if (ctx->dflash_capture->tape_layers.empty()) {
        fprintf(stderr, "%s : recurrent model exposed no DFlash tape layers\n", __func__);
        return 1;
    }
    for (size_t li = 0; li < ctx->dflash_capture->tape_layers.size(); ++li) {
        const auto & tape = ctx->dflash_capture->tape_layers[li];
        if (tape.n_tokens != (int) n_verify || tape.qkv_mixed.empty()) {
            fprintf(stderr, "%s : layer %zu has incomplete conv tape (%d tokens, %zu values)\n",
                    __func__, li, tape.n_tokens, tape.qkv_mixed.size());
            return 1;
        }
    }

    // Plane path: trim rejected verify tokens. The recurrent child selects rollback plane
    // `rollback`; read_state resolves that logical row exactly as recurrent state_write does.
    const llama_pos keep_end = (llama_pos) (n_prefix + n_accepted);
    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, keep_end, -1)) {
        fprintf(stderr, "%s : plane rollback failed\n", __func__);
        return 1;
    }
    if (recurrent->rs_idx[0] != rollback || recurrent->seq_pos_max(0) != keep_end - 1) {
        fprintf(stderr, "%s : plane rollback selected row %u at pos %d, expected row %u at pos %d\n",
                __func__, recurrent->rs_idx[0], recurrent->seq_pos_max(0),
                rollback, (int) keep_end - 1);
        return 1;
    }

    std::vector<state_piece> plane_state;
    if (!read_state(recurrent, 0, plane_state)) {
        fprintf(stderr, "%s : failed to read plane state\n", __func__);
        return 1;
    }

    // Tape path: discard the live recurrent cell, restore the immutable pre-verify row,
    // then replay only the accepted prefix of the verify tape.
    if (!recurrent->seq_rm(0, -1, -1) ||
        !recurrent->try_seq_cp(1, 0, -1, -1)) {
        fprintf(stderr, "%s : failed to restore pre-verify recurrent backup\n", __func__);
        return 1;
    }

    llama_tape_replay(ctx.get(), 0, n_accepted);
    llama_tape_replay_sync(ctx.get());

    if (recurrent->rs_idx[0] != 0 || recurrent->seq_pos_max(0) != keep_end - 1) {
        fprintf(stderr, "%s : tape replay selected row %u at pos %d, expected row 0 at pos %d\n",
                __func__, recurrent->rs_idx[0], recurrent->seq_pos_max(0), (int) keep_end - 1);
        return 1;
    }

    std::vector<state_piece> redundant_state;
    if (!read_state(recurrent, 0, redundant_state)) {
        fprintf(stderr, "%s : failed to read redundant-tape-replayed state\n", __func__);
        return 1;
    }

    if (!states_bit_equal(plane_state, redundant_state)) {
        fprintf(stderr, "%s : F32 tape replay is not bit-exact with the rollback plane on %s\n",
                __func__, gpu_name);
        return 1;
    }

    // Gate 2: restore the same immutable boundary again, poison redundant K/V,
    // and replay from qkv_mixed + gate + beta only. Poisoning makes an accidental
    // read from either redundant tensor fail loudly rather than yielding a false pass.
    if (!recurrent->seq_rm(0, -1, -1) ||
        !recurrent->try_seq_cp(1, 0, -1, -1)) {
        fprintf(stderr, "%s : failed to restore pre-verify backup for minimal replay\n", __func__);
        return 1;
    }

    auto * active_tape = ctx->dflash_capture->active_tape();
    if (active_tape == nullptr) {
        fprintf(stderr, "%s : active GPU tape disappeared before minimal replay\n", __func__);
        return 1;
    }
    for (auto & layer : active_tape->layers) {
        ggml_backend_tensor_memset(layer.k, 0xa5, 0, ggml_nbytes(layer.k));
        ggml_backend_tensor_memset(layer.v, 0xa5, 0, ggml_nbytes(layer.v));
    }

    llama_set_tape_minimal_replay(ctx.get(), true);
    llama_tape_replay(ctx.get(), 0, n_accepted);
    llama_tape_replay_sync(ctx.get());

    if (!ctx->dflash_capture->replay_minimal_last) {
        fprintf(stderr, "%s : requested minimal-F32 replay fell back to the redundant path\n", __func__);
        return 1;
    }
    if (recurrent->rs_idx[0] != 0 || recurrent->seq_pos_max(0) != keep_end - 1) {
        fprintf(stderr, "%s : minimal replay selected row %u at pos %d, expected row 0 at pos %d\n",
                __func__, recurrent->rs_idx[0], recurrent->seq_pos_max(0), (int) keep_end - 1);
        return 1;
    }

    std::vector<state_piece> minimal_state;
    if (!read_state(recurrent, 0, minimal_state)) {
        fprintf(stderr, "%s : failed to read minimal-F32 replay state\n", __func__);
        return 1;
    }
    if (!states_bit_equal(redundant_state, minimal_state)) {
        fprintf(stderr, "%s : minimal-F32 conv reconstruction is not bit-exact with redundant tape replay on %s\n",
                __func__, gpu_name);
        return 1;
    }

    fprintf(stderr, "%s : PASS (plane == redundant F32 tape == minimal-F32 conv replay bit-for-bit on %s)\n",
            __func__, gpu_name);
    return 0;
}
