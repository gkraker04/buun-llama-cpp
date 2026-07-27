#include "llama-vbr-checkpoint.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"
#include "llama-vbr-checkpoint-compose.inc"
#include "llama-vbr-generation.h"
#include "llama-vbr-generation-oracle.h"

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

// ODR NOTE: this definition is mirrored token-identically in
// tests/test-checkpoint-shadow-lifecycle.cpp (the test-only record factory — no factory symbol
// is compiled into production libllama). CI compares the two definitions.
struct llama_vbr_checkpoint_shadow {
    vbr_checkpoint_generation_record record;
};

namespace {

// One memory-tree child in fixed pre-order, with its serializer-derived dependency mode under
// LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY (the §5.2 table, source-confirmed):
//   plain KV      -> [self: payload_complete]   (state_write serializes its KV regardless of flags)
//   iSWA          -> [base: live_guarded, swa: payload_complete]   (iswa state_write omits base)
//   hybrid        -> [attn: live_guarded]       (hybrid state_write omits attention, keeps recurrent)
//   hybrid+iSWA   -> [base: live_guarded, swa: payload_complete]   (flags forwarded through iswa)
//   recurrent     -> no controller row          (its state is entirely in the checkpoint payload)
struct walk_child {
    const llama_kv_cache * cache        = nullptr;
    bool                   live_guarded = false;
};

bool collect_children(llama_memory_i * mem, std::vector<walk_child> & out) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        out.push_back({ iswa->get_base(), /*live_guarded=*/true  });
        out.push_back({ iswa->get_swa(),  /*live_guarded=*/false });
        return true;
    }
    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return collect_children(hybrid_iswa->get_mem_attn(), out);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        out.push_back({ hybrid->get_mem_attn(), /*live_guarded=*/true });
        return true;
    }
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        out.push_back({ kv, /*live_guarded=*/false });
        return true;
    }
    return dynamic_cast<llama_memory_recurrent *>(mem) != nullptr;
}

struct child_capture_ctx {
    const llama_kv_cache * cache         = nullptr;
    llama_seq_id           seq_id        = -1;
    llama_pos              frontier      = -1;  // exclusive
    bool *                 oracle_failed = nullptr;  // shared: compose fails fast on first child
};

// Armed live_guarded capture: the landed adapter (ownership-index enumeration, double-read
// stability) plus — when the debug oracle gate is enabled — an independent full-set comparison
// built from canonical observations WITHOUT the ownership index. An oracle disagreement fails
// the capture (reason-coded); it is never an admission input for the evaluator.
bool capture_from_cache(void * p, uint32_t child_id, vbr_checkpoint_generation_controller & out) {
    auto * ctx = static_cast<child_capture_ctx *>(p);
    if (!ctx->cache->vbr_generation_capture_live_guarded(child_id, ctx->seq_id, ctx->frontier, out)) {
        return false;
    }
    if (vbr_generation_oracle_enabled()) {
        std::vector<vbr_generation_oracle_cell> observations;
        if (!ctx->cache->vbr_generation_oracle_observations(ctx->seq_id, observations)) {
            *ctx->oracle_failed = true;
            return false;
        }
        const auto baseline = vbr_generation_oracle_capture(ctx->frontier, observations);
        for (const auto & stream : out.streams) {
            const auto audit = vbr_generation_oracle_audit(ctx->frontier, observations, baseline, stream);
            if (!audit.complete || !audit.set_equal) {
                *ctx->oracle_failed = true;
                return false;
            }
        }
    }
    return true;
}

size_t record_size(const vbr_checkpoint_generation_record & record) {
    size_t result = sizeof(vbr_checkpoint_generation_record);
    for (const auto & controller : record.controllers) {
        result += sizeof(controller);
        result += controller.units.size() * sizeof(vbr_checkpoint_unit_generation);
        for (const auto & stream : controller.streams) {
            result += sizeof(stream);
            result += stream.pages.size() * sizeof(vbr_generation_page_ref);
        }
    }
    return result;
}

}  // namespace

void llama_vbr_checkpoint_shadow_capture(
        llama_memory_t                         mem,
        llama_seq_id                           seq_id,
        const vbr_checkpoint_frontier_fields * frontier,
        llama_vbr_checkpoint_capture_result *  result) noexcept {
    if (result == nullptr) {
        return;
    }
    result->handle = nullptr;
    result->reason = vbr_checkpoint_capture_reason::internal_error;
    try {
        if (mem == nullptr || frontier == nullptr || seq_id < 0) {
            result->reason = vbr_checkpoint_capture_reason::invalid_arguments;
            return;
        }

        std::vector<walk_child> walk;
        if (!collect_children(mem, walk)) {
            result->reason = vbr_checkpoint_capture_reason::not_applicable;
            return;
        }

        bool oracle_failed = false;
        std::vector<child_capture_ctx> contexts(walk.size());
        std::vector<vbr_checkpoint_child_input> children;
        children.reserve(walk.size());
        for (size_t i = 0; i < walk.size(); ++i) {
            const auto * cache = walk[i].cache;
            vbr_checkpoint_child_input input;
            input.live_guarded = walk[i].live_guarded;
            input.armed        = cache != nullptr && cache->vbr_operation_armed();
            if (input.armed) {
                input.pool_uuid = cache->vbr_pool_id();
            }
            if (input.live_guarded && input.armed) {
                contexts[i] = { cache, seq_id, frontier->next_position, &oracle_failed };
                input.capture     = capture_from_cache;
                input.capture_ctx = &contexts[i];
            } else if (input.live_guarded && cache != nullptr) {
                // Any cell of the checkpoint sequence below the exclusive frontier is live
                // coverage the (unarmed) shadow cannot represent.
                const llama_pos pos_min = cache->seq_pos_min(seq_id);
                input.live_covered = pos_min >= 0 && pos_min < frontier->next_position;
            }
            children.push_back(input);
        }

        vbr_checkpoint_generation_record record;
        auto reason = vbr_checkpoint_compose(children, *frontier, record);
        if (reason == vbr_checkpoint_capture_reason::child_capture_failed && oracle_failed) {
            reason = vbr_checkpoint_capture_reason::oracle_mismatch;
        }
        result->reason = reason;
        if (reason != vbr_checkpoint_capture_reason::ok) {
            return;
        }
        result->handle = new llama_vbr_checkpoint_shadow{ std::move(record) };
    } catch (...) {
        result->handle = nullptr;
        result->reason = vbr_checkpoint_capture_reason::internal_error;
    }
}

void llama_vbr_checkpoint_shadow_free(llama_vbr_checkpoint_shadow * shadow) noexcept {
    delete shadow;
}

bool llama_vbr_checkpoint_shadow_equal(
        const llama_vbr_checkpoint_shadow * a,
        const llama_vbr_checkpoint_shadow * b) noexcept {
    return a != nullptr && b != nullptr &&
           a->record.status == vbr_checkpoint_generation_status::complete &&
           b->record.status == vbr_checkpoint_generation_status::complete &&
           a->record == b->record;
}

size_t llama_vbr_checkpoint_shadow_size(const llama_vbr_checkpoint_shadow * shadow) noexcept {
    return shadow != nullptr ? record_size(shadow->record) : 0;
}

vbr_checkpoint_generation_status llama_vbr_checkpoint_shadow_status(
        const llama_vbr_checkpoint_shadow * shadow) noexcept {
    return shadow != nullptr ? shadow->record.status : vbr_checkpoint_generation_status::generation_unknown;
}

const char * llama_vbr_checkpoint_shadow_reason_name(vbr_checkpoint_capture_reason reason) noexcept {
    switch (reason) {
        case vbr_checkpoint_capture_reason::ok:                   return "ok";
        case vbr_checkpoint_capture_reason::not_applicable:       return "not_applicable";
        case vbr_checkpoint_capture_reason::invalid_arguments:    return "invalid_arguments";
        case vbr_checkpoint_capture_reason::unarmed_live_covered: return "unarmed_live_covered";
        case vbr_checkpoint_capture_reason::child_capture_failed: return "child_capture_failed";
        case vbr_checkpoint_capture_reason::oracle_mismatch:      return "oracle_mismatch";
        case vbr_checkpoint_capture_reason::internal_error:       return "internal_error";
    }
    return "unknown";
}
