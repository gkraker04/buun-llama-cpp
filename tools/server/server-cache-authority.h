#pragma once

#include "server-cache-lease.h"
#include "server-cache-yield.h"
#include "server-retention-sidecar.h"
#include "../../src/llama-cache-authority.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

struct server_prompt_cache_state;

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

    // Immutable bridge from load-time placement to ledger-local device domains.
    std::vector<device_binding> live_device_domains;
    // Fixed-at-reserve-time compute rows. Physical capacity is sampled at observation/admission.
    std::vector<llama_cache_budget_device_input> budget_devices;
    llama_cache_budget_config budget_config;

    uint64_t admission_retries   = 0;
    uint64_t admission_refusals  = 0;
    uint64_t admission_commits   = 0;
    uint64_t admission_rollbacks = 0;
    bool configured = true;
    bool summary_emitted = false;

    // Construct a point-in-time budget input. pending_host_bytes are already allocated in the
    // detached host-cache node, so they are added back to the sampled CPU free-memory headroom.
    bool sample_budget(
            llama_cache_budget_config & config,
            uint64_t pending_host_bytes = 0) noexcept;

    // F0b's first authoritative producer: admit/stage/commit all host-entry payload leaves as one
    // all-or-nothing server transaction. Publication itself remains the caller's no-throw splice.
    bool admit_host_entry(server_prompt_cache_state & entry) noexcept;
};
