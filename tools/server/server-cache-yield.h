#pragma once

#include "server-cache-lease.h"
#include "server-retention-sidecar.h"
#include "../../src/llama-cache-budget.h"

#include <array>
#include <functional>
#include <vector>

constexpr uint32_t SERVER_CACHE_YIELD_POLICY_VERSION = 1;
constexpr size_t SERVER_CACHE_YIELD_MAX_CANDIDATES = 8192;

enum class server_cache_yield_status : uint8_t {
    fits = 0,
    insufficient_yield,
    unsupported_required,
    unavailable,
    _count,
};

const char * server_cache_yield_status_name(
    server_cache_yield_status status) noexcept;

struct server_cache_yield_candidate {
    llama_cache_acct_artifact_id artifact_id;
    common_retention_artifact_record record;
    server_retention_candidate_availability availability =
        server_retention_candidate_availability::backing_missing_or_stale;
    server_cache_lease_evaluation lease;
    bool identity_known = false;
    std::vector<llama_cache_acct_op_id> release_ops;
    bool has_unsupported_host_spill = false;
};

struct server_cache_yield_result {
    server_cache_yield_status status =
        server_cache_yield_status::unavailable;
    uint32_t yield_policy_version = SERVER_CACHE_YIELD_POLICY_VERSION;
    uint64_t accounting_serial = 0;
    std::array<std::vector<llama_cache_acct_artifact_id>,
               size_t(common_retention_pool::_count)> selected;
    std::vector<llama_cache_budget_plan_entry> plan;
    std::vector<llama_cache_acct_artifact_id> unsupported;
    // Winning D-S2 fit projection. Its domain rows are the canonical source of
    // resident/before/released/reserved/after at accounting_serial; D-S7 lowers
    // them to accounting-only wire types.
    llama_cache_budget_result projected_fit;
};

using server_cache_yield_preview_callback = std::function<bool(
    const std::vector<llama_cache_acct_op_id> &,
    uint64_t,
    llama_cache_acct_release_set_preview &)>;
using server_cache_yield_fits_callback = std::function<llama_cache_budget_result(
    const llama_cache_budget_plan &)>;
using server_cache_yield_candidate_resolver = std::function<void(
    const server_retention_candidate &,
    server_cache_yield_candidate &,
    server_cache_lease_identity &,
    bool & identity_known)>;

// Impure server-side half: joins the catalog value with live backing/identity
// through one injected resolver, then performs exactly one lease evaluation.
bool server_cache_yield_assemble(
    const std::vector<server_retention_candidate> & catalog,
    server_cache_lease_table & leases,
    const server_cache_yield_candidate_resolver & resolver,
    std::vector<server_cache_yield_candidate> & out) noexcept;

// Pure yield-ladder planner. Candidates already carry one evaluated lease result
// and validated backing operation ids; all impure ledger/budget access is injected.
server_cache_yield_result server_cache_yield_plan(
    const std::vector<server_cache_yield_candidate> & candidates,
    uint64_t accounting_serial,
    const server_cache_yield_preview_callback & preview,
    const server_cache_yield_fits_callback & fits,
    uint32_t policy_version = SERVER_CACHE_YIELD_POLICY_VERSION) noexcept;
