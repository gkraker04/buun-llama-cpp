#pragma once

#include <cstdint>
#include <vector>

// ZC1's behavior-neutral policy kernel. It owns no checkpoint storage and
// performs no certification or mutation. The eventual server adapter supplies
// explicit, reviewed constants and asks the existing D-A4 door to certify one
// member from exclusion_order. The production adapters remain responsible for
// evidence collection, recovery proofs, and every mutation.

enum class server_cache_retention_identity : uint8_t {
    current = 0,
    stale_conclusive_refused,
    identity_unknown,
};

enum class server_cache_retention_protection : uint8_t {
    none = 0,
    mandatory_anchor,
    hard_lease,
    recovery_pin,
    current_task,
};

struct server_cache_retention_member {
    uint32_t                          ordinal    = 0;
    uint64_t                          stable_id  = 0;
    uint64_t                          frontier   = 0;
    server_cache_retention_identity   identity   = server_cache_retention_identity::current;
    server_cache_retention_protection protection = server_cache_retention_protection::none;
    bool                              incoming   = false;
};

struct server_cache_retention_policy_config {
    uint32_t capacity           = 0;
    uint32_t recent_floor       = 0;
    uint32_t minimum_historical = 0;
    uint64_t bucket_base        = 0;
    uint32_t bucket_growth      = 0;
};

enum class server_cache_retention_policy_status : uint8_t {
    ok = 0,
    invalid_config,
    incomplete_evidence,
    protected_over_capacity,
    capacity_unavailable,
};

constexpr bool server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status status) noexcept {
    return status ==
               server_cache_retention_policy_status::incomplete_evidence ||
           status ==
               server_cache_retention_policy_status::capacity_unavailable;
}

struct server_cache_retention_policy_result {
    server_cache_retention_policy_status status            = server_cache_retention_policy_status::invalid_config;
    uint32_t                             recent_quota      = 0;
    uint32_t                             historical_quota  = 0;
    bool                                 incoming_selected = false;
    std::vector<uint32_t>                desired;
    std::vector<uint32_t>                exclusion_order;
};

// minimum_historical has one exact meaning in this kernel: for capacity >= 2,
// reserve K=min(minimum_historical, capacity-1) historical labels before
// assigning recent labels. At least one label of each lane remains. This makes
// the preregistered constant executable; it is not a shipping default.
server_cache_retention_policy_result server_cache_plan_retention_set(
    const server_cache_retention_policy_config &       config,
    uint64_t                                           current_frontier,
    const std::vector<server_cache_retention_member> & members) noexcept;

enum class server_cache_host_pressure_resource : uint8_t {
    bytes = 0,
    tokens,
};

struct server_cache_host_retention_member {
    uint32_t ordinal = 0;
    uint64_t stable_id = 0;
    uint64_t last_use_epoch = 0;
    uint64_t release_bytes = 0;
    uint64_t release_tokens = 0;
    bool soft_leased = false;
    bool main_family = false;
    bool exactly_redundant = false;
    bool eligible = false;
    bool incoming = false;
};

enum class server_cache_host_retention_status : uint8_t {
    selected = 0,
    no_eligible_progress,
    incomplete_evidence,
};

struct server_cache_host_retention_result {
    server_cache_host_retention_status status =
        server_cache_host_retention_status::incomplete_evidence;
    uint32_t ordinal = UINT32_MAX;
    uint64_t projected_pressure_steps = 0;
};

// Calibration-free host-pressure ordering for ZC1. The caller performs one
// immutable inventory pass and marks hard-leased, recovery-pinned, busy, and
// staged-incoming rows ineligible. This pure helper chooses only among rows
// that make strict progress in the currently violated resource; it owns no
// lease evaluator, release capability, or physical eraser.
server_cache_host_retention_result server_cache_plan_host_retention_victim(
    server_cache_host_pressure_resource resource,
    uint64_t need,
    const std::vector<server_cache_host_retention_member> & members) noexcept;

// Frozen counterfactual corpus event. parent_node identifies the exact prompt
// prefix from which this request extends; UINT32_MAX is the root. Node IDs are
// unique and parents must precede children. The simulator publishes one
// checkpoint opportunity after each request and uses the same allocator above.
struct server_cache_retention_sim_event {
    uint32_t node        = 0;
    uint32_t parent_node = UINT32_MAX;
    uint64_t frontier    = 0;
    bool     deep_rewind = false;
};

struct server_cache_retention_sim_config {
    server_cache_retention_policy_config policy;
    uint64_t                             recent_replay_cap     = 0;
    uint64_t                             historical_replay_cap = 0;
};

// The frozen production ZC1 tuple. `minimum_step == 0` is normalized to one,
// matching the server's checkpoint-min-step boundary. Training/grid callers
// remain free to construct arbitrary simulation configs directly.
server_cache_retention_sim_config server_cache_zc1_retention_config(
    uint32_t capacity,
    uint64_t minimum_step) noexcept;

// One spelling of the frozen lane replay envelope, shared by the production
// checkpoint adapter and the preregistered counterfactual evaluator.
uint64_t server_cache_retention_replay_cap(
    const server_cache_retention_sim_config & config,
    uint64_t current_frontier,
    uint64_t victim_frontier) noexcept;

struct server_cache_retention_sim_score {
    bool                  valid                      = false;
    uint64_t              total_replay_tokens        = 0;
    uint64_t              p95_replay_tokens          = 0;
    uint64_t              p99_replay_tokens          = 0;
    uint64_t              max_replay_tokens          = 0;
    uint64_t              zero_coverage_deep_rewinds = 0;
    uint64_t              checkpoint_mutations       = 0;
    uint64_t              publication_skips          = 0;
    std::vector<uint64_t> replay_samples;
};

// Training-only counterfactual evaluator. It is deterministic, state-local,
// performs no server mutation, and treats every synthetic member as unprotected.
server_cache_retention_sim_score server_cache_simulate_retention(
    const server_cache_retention_sim_config &             config,
    const std::vector<server_cache_retention_sim_event> & events) noexcept;

// The counterfactual comparator used by the preregistered gate: publish every
// event, drop the oldest member at capacity, and otherwise use the same prefix
// tree/replay accounting as the ZC simulation.
server_cache_retention_sim_score server_cache_simulate_fifo(
    uint32_t                                              capacity,
    const std::vector<server_cache_retention_sim_event> & events) noexcept;
