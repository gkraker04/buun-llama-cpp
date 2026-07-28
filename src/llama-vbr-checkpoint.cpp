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
    std::vector<vbr_checkpoint_oracle_sidecar_entry> oracle_sidecar;
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
    const llama_kv_cache * cache          = nullptr;
    llama_seq_id           seq_id         = -1;
    llama_pos              frontier       = -1;  // exclusive
    bool *                 oracle_failed  = nullptr;  // shared: compose fails fast on first child
    bool *                 global_failure = nullptr;
    std::vector<vbr_checkpoint_oracle_sidecar_entry> * oracle_sidecar = nullptr;
};

// Armed live_guarded capture: the landed adapter (ownership-index enumeration, double-read
// stability) plus — when the debug oracle gate is enabled — an independent full-set comparison
// built from canonical observations WITHOUT the ownership index. An oracle disagreement fails
// the capture (reason-coded); it is never an admission input for the evaluator.
bool capture_from_cache(void * p, uint32_t child_id, vbr_checkpoint_generation_controller & out) {
    auto * ctx = static_cast<child_capture_ctx *>(p);
    if (!ctx->cache->vbr_generation_capture_live_guarded(child_id, ctx->seq_id, ctx->frontier, out)) {
        if (ctx->cache->vbr_generation_shadow_globally_unavailable()) {
            *ctx->global_failure = true;
        }
        return false;
    }
    if (vbr_generation_oracle_enabled()) {
        std::vector<vbr_generation_oracle_cell> observations;
        if (!ctx->cache->vbr_generation_oracle_observations(
                    ctx->seq_id, ctx->frontier, observations)) {
            if (ctx->cache->vbr_generation_shadow_globally_unavailable()) {
                *ctx->global_failure = true;
            } else {
                *ctx->oracle_failed = true;
            }
            return false;
        }
        const auto baseline = vbr_generation_oracle_capture(ctx->frontier, observations);
        for (const auto & stream : out.streams) {
            const auto audit = vbr_generation_oracle_audit(ctx->frontier, observations, baseline, stream);
            if (!audit.complete || !audit.set_equal || !audit.bytes_equal) {
                *ctx->oracle_failed = true;
                return false;
            }
            ctx->oracle_sidecar->push_back(
                    { child_id, stream.stream_index, ctx->frontier, baseline });
        }
    }
    if (ctx->cache->vbr_generation_shadow_globally_unavailable()) {
        *ctx->global_failure = true;
        return false;
    }
    return true;
}

size_t record_size(const llama_vbr_checkpoint_shadow & shadow) {
    const auto & record = shadow.record;
    size_t result = sizeof(llama_vbr_checkpoint_shadow);
    for (const auto & controller : record.controllers) {
        result += sizeof(controller);
        result += controller.units.size() * sizeof(vbr_checkpoint_unit_generation);
        for (const auto & stream : controller.streams) {
            result += sizeof(stream);
            result += stream.pages.size() * sizeof(vbr_generation_page_ref);
        }
    }
    for (const auto & entry : shadow.oracle_sidecar) {
        result += sizeof(entry);
        result += entry.baseline.dependency_cells.size() * sizeof(uint32_t);
    }
    return result;
}

vbr_checkpoint_reset_scope reset_scope_for(vbr_checkpoint_capture_reason reason) {
    switch (reason) {
        case vbr_checkpoint_capture_reason::ok:
        case vbr_checkpoint_capture_reason::not_applicable:
            return vbr_checkpoint_reset_scope::none;
        case vbr_checkpoint_capture_reason::controller_unavailable:
            return vbr_checkpoint_reset_scope::global;
        case vbr_checkpoint_capture_reason::invalid_arguments:
        case vbr_checkpoint_capture_reason::unarmed_live_covered:
        case vbr_checkpoint_capture_reason::child_capture_failed:
        case vbr_checkpoint_capture_reason::oracle_mismatch:
        case vbr_checkpoint_capture_reason::internal_error:
            return vbr_checkpoint_reset_scope::capturing_slot;
        case vbr_checkpoint_capture_reason::_count:
            break;
    }
    return vbr_checkpoint_reset_scope::capturing_slot;
}

vbr_checkpoint_shadow_category map_category(vbr_checkpoint_eligibility_category value) {
    switch (value) {
        case vbr_checkpoint_eligibility_category::not_applicable:
            return vbr_checkpoint_shadow_category::not_applicable;
        case vbr_checkpoint_eligibility_category::generation_unknown:
            return vbr_checkpoint_shadow_category::generation_unknown;
        case vbr_checkpoint_eligibility_category::strict_accept:
            return vbr_checkpoint_shadow_category::strict_accept;
        case vbr_checkpoint_eligibility_category::live_rebased_shadow_accept:
            return vbr_checkpoint_shadow_category::live_rebased_shadow_accept;
        case vbr_checkpoint_eligibility_category::strict_reject:
            return vbr_checkpoint_shadow_category::strict_reject;
    }
    return vbr_checkpoint_shadow_category::strict_reject;
}

vbr_checkpoint_shadow_reason map_reason(vbr_checkpoint_eligibility_reason value) {
    switch (value) {
        case vbr_checkpoint_eligibility_reason::none: return vbr_checkpoint_shadow_reason::none;
        case vbr_checkpoint_eligibility_reason::capability_not_applicable:
            return vbr_checkpoint_shadow_reason::capability_not_applicable;
        case vbr_checkpoint_eligibility_reason::record_unknown:
            return vbr_checkpoint_shadow_reason::record_unknown;
        case vbr_checkpoint_eligibility_reason::record_version:
            return vbr_checkpoint_shadow_reason::record_version;
        case vbr_checkpoint_eligibility_reason::identity_or_frontier:
            return vbr_checkpoint_shadow_reason::identity_or_frontier;
        case vbr_checkpoint_eligibility_reason::controller_shape:
            return vbr_checkpoint_shadow_reason::controller_shape;
        case vbr_checkpoint_eligibility_reason::child_order:
            return vbr_checkpoint_shadow_reason::child_order;
        case vbr_checkpoint_eligibility_reason::dependency_mode:
            return vbr_checkpoint_shadow_reason::dependency_mode;
        case vbr_checkpoint_eligibility_reason::controller_inactive:
            return vbr_checkpoint_shadow_reason::controller_inactive;
        case vbr_checkpoint_eligibility_reason::controller_unstable:
            return vbr_checkpoint_shadow_reason::controller_unstable;
        case vbr_checkpoint_eligibility_reason::pool_uuid:
            return vbr_checkpoint_shadow_reason::pool_uuid;
        case vbr_checkpoint_eligibility_reason::global_generation:
            return vbr_checkpoint_shadow_reason::global_generation;
        case vbr_checkpoint_eligibility_reason::unit_shape:
            return vbr_checkpoint_shadow_reason::unit_shape;
        case vbr_checkpoint_eligibility_reason::unit_unstable:
            return vbr_checkpoint_shadow_reason::unit_unstable;
        case vbr_checkpoint_eligibility_reason::unit_generation:
            return vbr_checkpoint_shadow_reason::unit_generation;
        case vbr_checkpoint_eligibility_reason::live_rebased_transition:
            return vbr_checkpoint_shadow_reason::live_rebased_transition;
        case vbr_checkpoint_eligibility_reason::stream_shape:
            return vbr_checkpoint_shadow_reason::stream_shape;
        case vbr_checkpoint_eligibility_reason::stream_order:
            return vbr_checkpoint_shadow_reason::stream_order;
        case vbr_checkpoint_eligibility_reason::malformed_page_refs:
            return vbr_checkpoint_shadow_reason::malformed_page_refs;
        case vbr_checkpoint_eligibility_reason::page_out_of_range:
            return vbr_checkpoint_shadow_reason::page_out_of_range;
        case vbr_checkpoint_eligibility_reason::dependency_changed:
            return vbr_checkpoint_shadow_reason::dependency_changed;
        case vbr_checkpoint_eligibility_reason::dependency_membership_lost:
            return vbr_checkpoint_shadow_reason::dependency_membership_lost;
        case vbr_checkpoint_eligibility_reason::dependency_cardinality:
            return vbr_checkpoint_shadow_reason::dependency_cardinality;
    }
    return vbr_checkpoint_shadow_reason::controller_unstable;
}

vbr_checkpoint_shadow_observation map_observation(vbr_observation_class value) {
    switch (value) {
        case vbr_observation_class::trivial_append:
            return vbr_checkpoint_shadow_observation::trivial_append;
        case vbr_observation_class::boundary_refined:
            return vbr_checkpoint_shadow_observation::boundary_refined;
        case vbr_observation_class::destructive:
            return vbr_checkpoint_shadow_observation::destructive;
        case vbr_observation_class::import_refined:
            return vbr_checkpoint_shadow_observation::import_refined;
    }
    return vbr_checkpoint_shadow_observation::trivial_append;
}

vbr_checkpoint_shadow_tombstone map_tombstone(vbr_expected_tombstone_class value) {
    switch (value) {
        case vbr_expected_tombstone_class::none:
            return vbr_checkpoint_shadow_tombstone::none;
        case vbr_expected_tombstone_class::restore_one_behind:
            return vbr_checkpoint_shadow_tombstone::restore_one_behind;
        case vbr_expected_tombstone_class::swa_wrap:
            return vbr_checkpoint_shadow_tombstone::swa_wrap;
        case vbr_expected_tombstone_class::explicit_destructive_trim:
            return vbr_checkpoint_shadow_tombstone::explicit_destructive_trim;
        case vbr_expected_tombstone_class::dependency_seq_removed:
            return vbr_checkpoint_shadow_tombstone::dependency_seq_removed;
        case vbr_expected_tombstone_class::unexplained:
            return vbr_checkpoint_shadow_tombstone::unexplained;
    }
    return vbr_checkpoint_shadow_tombstone::unexplained;
}

bool build_live_view(
        llama_memory_i *                       mem,
        llama_seq_id                           seq_id,
        const vbr_checkpoint_frontier_fields & frontier,
        vbr_generation_live_view &             live,
        std::vector<walk_child> &              walk) {
    if (!vbr_checkpoint_frontier_valid(frontier) || seq_id < 0 || !collect_children(mem, walk)) {
        return false;
    }

    bool any_armed = false;
    std::vector<vbr_checkpoint_child_policy> policy;
    policy.reserve(walk.size());
    live.controllers.reserve(walk.size());
    for (uint32_t child_id = 0; child_id < walk.size(); ++child_id) {
        const auto * cache = walk[child_id].cache;
        const bool armed = cache != nullptr && cache->vbr_operation_armed();
        any_armed = any_armed || armed;
        const auto mode = walk[child_id].live_guarded
                ? checkpoint_child_dependency_mode::live_guarded
                : checkpoint_child_dependency_mode::payload_complete;
        const vbr_pool_uuid pool = armed ? cache->vbr_pool_id() : vbr_pool_uuid{};
        policy.push_back({ child_id, mode, pool });

        vbr_generation_live_controller_view controller;
        if (walk[child_id].live_guarded && armed) {
            if (!cache->vbr_generation_live_guarded_view(
                        child_id, seq_id, frontier.next_position, controller)) {
                // Still invoke the sole evaluator exactly once: a missing live view is a
                // capability/controller-unavailable result, never a server-invented view.
                live.capability_applicable = false;
                controller.child_id        = child_id;
                controller.dependency_mode = mode;
            }
        } else {
            controller.child_id        = child_id;
            controller.dependency_mode = mode;
        }
        live.controllers.push_back(std::move(controller));
    }
    live.legacy_eligible              = false;  // G-only bridge: never imports P/L authority
    live.identity_frontier_eligible   = true;
    live.capability_applicable        = live.capability_applicable && any_armed;
    live.identity_policy_order_digest = vbr_checkpoint_identity_digest(frontier, policy);
    return true;
}

const vbr_checkpoint_oracle_sidecar_entry * find_oracle_baseline(
        const llama_vbr_checkpoint_shadow & shadow, uint32_t child_id, uint32_t stream_index) {
    const auto it = std::find_if(
            shadow.oracle_sidecar.begin(), shadow.oracle_sidecar.end(),
            [&](const auto & entry) {
                return entry.child_id == child_id && entry.stream_index == stream_index;
            });
    return it == shadow.oracle_sidecar.end() ? nullptr : &*it;
}

vbr_checkpoint_oracle_outcome audit_oracle(
        const llama_vbr_checkpoint_shadow & shadow,
        const std::vector<walk_child> &     walk,
        llama_seq_id                       seq_id,
        const vbr_checkpoint_eligibility & eligibility) {
    if (!vbr_generation_oracle_enabled()) {
        return vbr_checkpoint_oracle_outcome::disabled;
    }
    const bool crossing =
        eligibility.observation_class == vbr_observation_class::destructive ||
        eligibility.observation_class == vbr_observation_class::import_refined;
    if (!vbr_generation_oracle_audit_due(
                crossing, shadow.record.identity_policy_order_digest,
                vbr_generation_oracle_audit_forced())) {
        return vbr_checkpoint_oracle_outcome::not_due;
    }

    bool set_equal   = true;
    bool bytes_equal = true;
    for (const auto & controller : shadow.record.controllers) {
        if (controller.dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
                controller.child_id >= walk.size() || walk[controller.child_id].cache == nullptr) {
            continue;
        }
        for (const auto & stream : controller.streams) {
            std::vector<vbr_generation_oracle_cell> observations;
            if (!walk[controller.child_id].cache->vbr_generation_oracle_observations(
                        seq_id, stream.computation_frontier, observations)) {
                return vbr_checkpoint_oracle_outcome::unavailable;
            }
            const auto * sidecar =
                find_oracle_baseline(shadow, controller.child_id, stream.stream_index);
            if (sidecar == nullptr || sidecar->computation_frontier != stream.computation_frontier) {
                // Includes late-enable: no capture-time sidecar is ever fabricated.
                return vbr_checkpoint_oracle_outcome::unavailable;
            }
            auto audit = vbr_generation_oracle_audit(
                    stream.computation_frontier, observations, sidecar->baseline, stream);
            vbr_generation_oracle_inject(audit);
            if (!audit.complete) {
                return vbr_checkpoint_oracle_outcome::unavailable;
            }
            set_equal   = set_equal && audit.set_equal;
            bytes_equal = bytes_equal && audit.bytes_equal;
        }
    }
    if (!set_equal && !bytes_equal) {
        return vbr_checkpoint_oracle_outcome::set_and_byte_mismatch;
    }
    if (!set_equal) {
        return vbr_checkpoint_oracle_outcome::set_mismatch;
    }
    if (!bytes_equal) {
        return vbr_checkpoint_oracle_outcome::byte_mismatch;
    }
    return vbr_checkpoint_oracle_outcome::pass;
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
    result->handle      = nullptr;
    result->reason      = vbr_checkpoint_capture_reason::internal_error;
    result->reset_scope = vbr_checkpoint_reset_scope::capturing_slot;
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

        bool oracle_failed  = false;
        bool global_failure = false;
        std::vector<vbr_checkpoint_oracle_sidecar_entry> oracle_sidecar;
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
                contexts[i] = {
                    cache, seq_id, frontier->next_position, &oracle_failed,
                    &global_failure, &oracle_sidecar
                };
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
        if (reason == vbr_checkpoint_capture_reason::child_capture_failed) {
            if (global_failure) {
                reason = vbr_checkpoint_capture_reason::controller_unavailable;
            } else if (oracle_failed) {
                reason = vbr_checkpoint_capture_reason::oracle_mismatch;
            }
        }
        result->reason      = reason;
        result->reset_scope = reset_scope_for(reason);
        if (reason != vbr_checkpoint_capture_reason::ok) {
            return;
        }
        result->handle = new llama_vbr_checkpoint_shadow{
            std::move(record), std::move(oracle_sidecar)
        };
    } catch (...) {
        result->handle      = nullptr;
        result->reason      = vbr_checkpoint_capture_reason::internal_error;
        result->reset_scope = vbr_checkpoint_reset_scope::capturing_slot;
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
    return shadow != nullptr ? record_size(*shadow) : 0;
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
        case vbr_checkpoint_capture_reason::controller_unavailable: return "controller_unavailable";
        case vbr_checkpoint_capture_reason::oracle_mismatch:      return "oracle_mismatch";
        case vbr_checkpoint_capture_reason::internal_error:       return "internal_error";
        case vbr_checkpoint_capture_reason::_count:               break;
    }
    return "unknown";
}

void llama_vbr_checkpoint_shadow_evaluate(
        const llama_vbr_checkpoint_shadow *       shadow,
        llama_memory_t                            mem,
        llama_seq_id                              seq_id,
        const vbr_checkpoint_frontier_fields *    frontier,
        llama_vbr_checkpoint_shadow_evaluation *  result) noexcept {
    if (result == nullptr) {
        return;
    }
    *result = {};
    try {
        if (shadow == nullptr || mem == nullptr || frontier == nullptr || seq_id < 0) {
            return;
        }

        vbr_generation_live_view live;
        std::vector<walk_child> walk;
        if (!build_live_view(mem, seq_id, *frontier, live, walk)) {
            return;
        }

        // The only call to the raw comparison authority in this bridge.
        const auto eligibility = checkpoint_vbr_eligibility(shadow->record, live);
        result->evaluator_invocations = 1;
        result->strict                = eligibility.strict;
        result->live_rebased_shadow   = eligibility.live_rebased_shadow;
        result->category              = map_category(eligibility.category);
        result->reason                = map_reason(eligibility.reason);
        result->observation_class     = map_observation(eligibility.observation_class);
        result->tombstone_class       = map_tombstone(eligibility.tombstone_class);
        result->refinement_used       = eligibility.refinement_used;
        result->rejecting_cells       = eligibility.rejecting_cells;
        result->oracle_outcome        = vbr_checkpoint_oracle_outcome::disabled;
        if (eligibility.strict && eligibility.refinement_used) {
            result->oracle_outcome = audit_oracle(*shadow, walk, seq_id, eligibility);
        } else if (vbr_generation_oracle_enabled()) {
            result->oracle_outcome = vbr_checkpoint_oracle_outcome::not_due;
        }
    } catch (...) {
        // Record-free fail-closed default was installed above. No exception crosses the bridge.
        *result = {};
    }
}
