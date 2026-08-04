#pragma once

#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-yield.h"
#include "server-retention-sidecar.h"
#include "../../common/common-cache-plan.h"
#include "../../src/llama-cache-authority.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct server_prompt_cache_state;

// D-A3's process-local policy seams. The weight callback is deliberately a
// dimensionless fixed-point multiplier: fitted B restore cost remains the
// economic base, while E1/capstone policy can replace the provisional
// automatic weight without changing the victim ladder. A weight callback that
// returns false, or returns true with weight_milli == 0, refuses pricing for
// that victim (fail-closed); zero never means free-to-evict. The recovery callback
// is invoked only for an already-authorized durable source; it may never
// enumerate or widen tenant authorization.
constexpr uint32_t SERVER_CACHE_HOST_WEIGHT_SCALE = 1000;
constexpr uint32_t SERVER_CACHE_HOST_SOFT_LEASE_WEIGHT = 2000;
constexpr uint32_t SERVER_CACHE_HOST_MAIN_FAMILY_WEIGHT = 2000;

using server_cache_host_retention_weight_fn = bool (*)(
    void * context,
    const server_prompt_cache_state & victim,
    uint32_t & weight_milli) noexcept;

struct server_cache_host_recovery_evidence {
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin pin;
    common_cache_plan_displaced_fate fate =
        common_cache_plan_displaced_fate::unavailable;
};

using server_cache_host_recovery_fn = bool (*)(
    void * context,
    const server_prompt_cache_state & victim,
    server_cache_host_recovery_evidence & out) noexcept;

// P2 F0 authority substrate. The debug observer is only a serialization layer over this
// independently-owned state; --cache-lifecycle can therefore enforce accounting with debug off.
// Member order is lifetime order: retention releases lease memberships and accounting operations,
// so the ledger and leases must outlive it.
struct server_cache_authority {
    struct device_binding {
        ggml_backend_dev_t               device = nullptr;
        llama_cache_acct_resource_domain domain;
    };

    llama_cache_acct_ledger ledger;
    server_cache_lease_table leases;
    server_retention_sidecar_store retention;
    server_cache_destruction_observer destruction;
    llama_cache_budget_coordinator budget;
    server_cache_yield_result last_yield;
    common_cache_plan_destruction_counters destruction_counters;

    // Immutable bridge from load-time placement to ledger-local device domains.
    std::vector<device_binding> live_device_domains;
    // Fixed-at-reserve-time compute rows. Physical capacity is sampled at observation/admission.
    std::vector<llama_cache_budget_device_input> budget_devices;
    llama_cache_budget_config budget_config;

    uint64_t admission_retries   = 0;
    uint64_t admission_refusals  = 0;
    uint64_t admission_commits   = 0;
    uint64_t admission_rollbacks = 0;
    uint64_t destruction_quote_sequence = 0;
    std::string calibration_profile;
    void * host_retention_weight_context = nullptr;
    server_cache_host_retention_weight_fn host_retention_weight = nullptr;
    void * host_recovery_context = nullptr;
    server_cache_host_recovery_fn host_recovery = nullptr;
    bool configured = true;
    bool summary_emitted = false;

    // Construct a point-in-time budget input. pending_host_bytes are already allocated in the
    // detached host-cache node, so they are added back to the sampled CPU free-memory headroom.
    bool sample_budget(
            llama_cache_budget_config & config,
            uint64_t pending_host_bytes = 0) noexcept;

    // Lower one exact accounting union into the capacity domains that its
    // release would affect. D-A2+ share this one projection door.
    bool project_release(
            const llama_cache_acct_release_set_preview & release,
            std::vector<common_cache_plan_yield_domain> & out) noexcept;

    // F0b's first authoritative producer: admit/stage/commit all host-entry payload leaves as one
    // all-or-nothing server transaction. Publication itself remains the caller's no-throw splice.
    bool admit_host_entry(server_prompt_cache_state & entry) noexcept;

    // Bounded process-local D-A receipt publication for destruction work that
    // occurs during host-cache maintenance rather than one B request record.
    void observe_host_destruction(
        common_cache_plan_destruction_receipt receipt,
        bool observe_classification = true) noexcept;
};
