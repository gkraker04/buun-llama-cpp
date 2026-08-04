#pragma once

#include "server-cache-yield.h"
#include "server-cache-plan-authority.h"

#include <functional>
#include <vector>

struct server_cache_destruction_artifact {
    server_cache_yield_candidate candidate;
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;
    int32_t host_source_id = -1;
    common_retention_pool pool = common_retention_pool::attention;
    bool mandatory_anchor = false;
};

using server_cache_destruction_preview_callback = std::function<bool(
    const std::vector<llama_cache_acct_op_id> &,
    uint64_t,
    llama_cache_acct_release_set_preview &)>;

using server_cache_destruction_projection_callback = std::function<bool(
    const llama_cache_acct_release_set_preview &,
    std::vector<common_cache_plan_yield_domain> &)>;

struct server_cache_destruction_quote_options {
    bool lifecycle_available = false;
    common_cache_plan_recovery_citation recovery_citation =
        common_cache_plan_recovery_citation::unavailable;
    uint64_t admission_sequence = 0;
};

// D-A0a's bounded pre-minimization shadow pass. `artifacts` is one normalized
// retention inventory: identities and leases were each inspected exactly once.
// Quotes are memoized by canonical victim-manifest digest; no mutation, lease
// advancement, or accounting claim occurs.
bool server_cache_destruction_quote_all(
    common_cache_plan_record & rec,
    int32_t legacy_plan_candidate,
    const std::vector<server_cache_destruction_artifact> & artifacts,
    uint64_t accounting_serial,
    const server_cache_destruction_preview_callback & preview,
    const server_cache_destruction_projection_callback & project,
    const server_cache_destruction_quote_options & options,
    common_cache_plan_destruction_counters & counters) noexcept;

void server_cache_destruction_select_quote(
    common_cache_plan_record & rec,
    common_cache_plan_destruction_counters & counters) noexcept;

void server_cache_destruction_finalize_projection(
    common_cache_plan_record & rec,
    const server_cache_yield_result & yield) noexcept;

bool server_cache_destruction_effect_matches(
    const common_cache_plan_destruction_receipt & quote,
    const common_cache_plan_destruction_effect_digest & current_effect,
    const std::vector<common_cache_plan_yield_domain> & quoted_domains,
    const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept;

// Forward contract for D-A0b's mutation-boundary certify-time recheck. The
// quote serial is evidence only; exact union/digest/domain equality decides.
common_cache_plan_destruction_reason server_cache_destruction_effect_recheck(
    const common_cache_plan_destruction_receipt & quote,
    const common_cache_plan_destruction_effect_digest & current_effect,
    const std::vector<common_cache_plan_yield_domain> & quoted_domains,
    const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept;

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
    const common_cache_plan_record & rec,
    int32_t candidate,
    int32_t legacy_candidate) noexcept;

bool server_cache_destruction_has_effect(
    const common_cache_plan_record & rec,
    int32_t legacy_candidate) noexcept;
