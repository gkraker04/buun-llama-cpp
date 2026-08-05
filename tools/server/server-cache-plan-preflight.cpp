#include "server-cache-plan-preflight-internal.h"

#include "server-cache-plan-authority.h"

#include <algorithm>

bool server_cache_plan_local_source_registry::get_or_assign(
        uintptr_t instance,
        int32_t & source_id) {
    auto [it, inserted] = source_ids_.emplace(instance, -1);
    (void) inserted;
    return server_cache_plan_assign_source_id(
        it->second, next_source_id_, source_id);
}

bool server_cache_plan_local_source_registry::find(
        uintptr_t instance,
        int32_t & source_id) const noexcept {
    const auto found = source_ids_.find(instance);
    if (found == source_ids_.end() || found->second < 0) {
        source_id = -1;
        return false;
    }
    source_id = found->second;
    return true;
}

server_cache_plan_stage1_semantics server_cache_plan_stage1_semantics_for(
        bool is_preflight,
        bool native_completion,
        bool update_cache,
        bool prompt_cache_available,
        bool adapter_matches) noexcept {
    server_cache_plan_stage1_semantics out;
    out.completion_semantics = is_preflight || native_completion;
    out.host_lookup_enabled = update_cache && prompt_cache_available &&
                              out.completion_semantics && adapter_matches;
    out.recovery_citation = prompt_cache_available && out.completion_semantics
        ? common_cache_plan_recovery_citation::prospective
        : common_cache_plan_recovery_citation::unavailable;
    return out;
}

static bool tier_enabled(const common_cache_plan_record & rec) noexcept {
    const auto decision = server_cache_plan_level_of(rec.selection);
    return decision != common_cache_plan_authority_level::off &&
           decision != common_cache_plan_authority_level::_count &&
           server_cache_plan_level_enabled(
               rec.authority.configured_level, decision);
}

server_cache_plan_preflight_expected_path
server_cache_plan_preflight_derive_expected_path(
        const common_cache_plan_record & rec,
        bool planner_inputs_current) noexcept {
    if (!planner_inputs_current ||
        rec.planner_status != common_cache_plan_planner_status::ok ||
        !server_cache_plan_shadow_choice_valid(rec)) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    if (!tier_enabled(rec)) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    if (rec.destruction.plan_candidate == rec.shadow_choice &&
        rec.destruction.effects != 0 &&
        rec.destruction.state ==
            common_cache_plan_destruction_state::quoted) {
        return server_cache_plan_preflight_expected_path::
            conditional_on_destruction_certification;
    }
    if (rec.destruction.plan_candidate == rec.shadow_choice &&
        rec.destruction.effects != 0) {
        return server_cache_plan_preflight_expected_path::legacy;
    }
    return server_cache_plan_preflight_expected_path::
        planner_if_still_current;
}

static llama_cache_acct_value term_raw(
        const common_cache_plan_candidate & candidate,
        llama_cache_acct_cost_kind kind) noexcept {
    return candidate.cost_terms[size_t(kind)].raw;
}

bool server_cache_plan_preflight_build_view(
        const common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        bool planner_inputs_current,
        server_cache_plan_preflight_view & out) noexcept {
    try {
        out = {};
        out.status = server_cache_plan_preflight_status::ok;
        out.planner_status = rec.planner_status;
        out.configured_level = rec.authority.configured_level;
        out.selection_tier = rec.selection;
        out.fallback_reason = rec.authority.fallback_reason;
        if (rec.planner_status == common_cache_plan_planner_status::ok &&
            !planner_inputs_current) {
            out.fallback_reason =
                common_cache_plan_authority_fallback::stale_capability;
        } else if (rec.planner_status ==
                       common_cache_plan_planner_status::ok &&
                   !tier_enabled(rec)) {
            out.fallback_reason =
                common_cache_plan_authority_fallback::tier_not_enabled;
        }
        out.prompt_tokens = rec.n_prompt_tokens;
        out.expected_path = server_cache_plan_preflight_derive_expected_path(
            rec, planner_inputs_current);

        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & candidate = rec.inventory[i];
            if (candidate.reason == COMMON_CACHE_PLAN_REASON_NONE) {
                continue;
            }
            auto found = std::find_if(
                out.miss_reasons.begin(), out.miss_reasons.end(),
                [&](const auto & row) {
                    return row.provider == candidate.provider &&
                           row.reason == candidate.reason;
                });
            if (found == out.miss_reasons.end()) {
                out.miss_reasons.push_back({
                    candidate.provider, candidate.reason, 1,
                });
            } else {
                found->count++;
            }
        }

        if (rec.planner_status != common_cache_plan_planner_status::ok ||
            !server_cache_plan_shadow_choice_valid(rec)) {
            return true;
        }
        const auto & selected = rec.inventory[size_t(rec.shadow_choice)];
        out.provider = selected.provider;
        out.provider_available = true;
        out.target_relation = rec.selection == common_cache_plan_selection::by_id
            ? server_cache_plan_preflight_target_relation::forced_slot
            : (selected.target_slot_id == legacy_target_slot_id
                ? server_cache_plan_preflight_target_relation::same_as_legacy
                : server_cache_plan_preflight_target_relation::retarget);
        out.cost_terms = selected.cost_terms;
        out.predicted_replay_tokens = term_raw(
            selected, llama_cache_acct_cost_kind::replay);
        out.predicted_restore_bytes = term_raw(
            selected, llama_cache_acct_cost_kind::restore);
        out.predicted_ttft_us = selected.predicted_total_us;
        if (out.prompt_tokens.state == llama_cache_acct_known::known &&
            out.predicted_replay_tokens.state ==
                llama_cache_acct_known::known &&
            out.predicted_replay_tokens.value <= out.prompt_tokens.value) {
            out.predicted_reuse_tokens = llama_cache_acct_value::measured(
                out.prompt_tokens.value - out.predicted_replay_tokens.value);
        }
        if (selected.provider == common_cache_plan_provider::cold_replay) {
            out.cache_hit = server_cache_plan_preflight_cache_hit::miss;
        } else if (out.predicted_replay_tokens.state ==
                       llama_cache_acct_known::known) {
            out.cache_hit = out.predicted_replay_tokens.value == 0
                ? server_cache_plan_preflight_cache_hit::full
                : server_cache_plan_preflight_cache_hit::partial;
        }

        if (rec.destruction.plan_candidate == rec.shadow_choice ||
            rec.destruction.state ==
                common_cache_plan_destruction_state::not_required) {
            out.destruction.state = rec.destruction.state;
            out.destruction.reason = rec.destruction.reason;
            out.destruction.effects = rec.destruction.effects;
            out.destruction.protection = rec.destruction.lease_verdict;
            out.destruction.displaced_fate = rec.destruction.displaced_fate;
            out.destruction.recovery = rec.destruction.recovery_citation;
            uint64_t projected = 0;
            const auto quote = std::find_if(
                rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
                [&](const auto & candidate) {
                    return candidate.receipt.plan_candidate == rec.shadow_choice;
                });
            if (quote != rec.destruction_quotes.end() &&
                common_cache_plan_projected_release_bytes(
                    quote->projected_domains, projected)) {
                out.destruction.projected_release_bytes =
                    llama_cache_acct_value::measured(projected);
            }
            out.destruction.estimated_destruction_us =
                selected.cost_terms[size_t(
                    llama_cache_acct_cost_kind::eviction)].estimated_us;
        }
        return true;
    } catch (...) {
        out = {};
        out.status = server_cache_plan_preflight_status::internal_fault;
        return false;
    }
}
