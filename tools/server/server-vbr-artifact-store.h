#pragma once

#include "../../src/llama-cache-authority.h"
#include "../../src/llama-vbr-artifact-catalog.h"
#include "../../src/llama-vbr-explicit-capture.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class server_vbr_artifact_capture_status : uint8_t {
    ok = 0,
    unsupported,
    unavailable,
    invalid_slot,
    slot_processing,
    stale_frontier,
    identity_unavailable,
    unauthorized,
    required_companion_unavailable,
    admission_refused,
    source_changed,
    internal_error,
    _count,
};

const char * server_vbr_artifact_capture_status_name(
    server_vbr_artifact_capture_status status) noexcept;

enum class server_vbr_artifact_store_create_failure : uint8_t {
    none = 0,
    ledger_missing,
    budget_sampler_missing,
    topology_missing,
    pool_binding_missing,
    lane_missing,
    attention_child_missing,
    ring_size_invalid,
    chunk_size_invalid,
    budget_sample_failed,
    ring_create_failed,
    internal_error,
    _count,
};

const char * server_vbr_artifact_store_create_failure_name(
    server_vbr_artifact_store_create_failure failure) noexcept;

struct server_vbr_artifact_store_create_diagnostics {
    server_vbr_artifact_store_create_failure failure =
        server_vbr_artifact_store_create_failure::none;
    vbr_capture_stream_status ring_status =
        vbr_capture_stream_status::_count;
    vbr_capture_ring_create_failure ring_failure =
        vbr_capture_ring_create_failure::none;
    uint64_t requested_ring_bytes = 0;
    uint64_t attempted_ring_bytes = 0;
    uint64_t constructed_ring_bytes = 0;
    size_t chunk_bytes = 0;
    size_t lane_count = 0;
    uint32_t attention_children = 0;
};

struct server_vbr_artifact_store_config {
    using sample_budget_fn = bool (*)(
        void * context,
        llama_cache_budget_config & output) noexcept;

    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain pinned_domain;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<vbr_capture_lane> lanes;
    uint32_t attention_children = 0;
    uint64_t ring_bytes = 0;
    size_t chunk_bytes = 0;
    void * budget_context = nullptr;
    sample_budget_fn sample_budget = nullptr;
};

// Observe the artifact machinery's empty capacity rows after the one-shot
// manifest and before that domain's producer is certified. Device-scoped live
// rows are deliberately left to the live-memory observer.
bool server_vbr_artifact_store_observe_empty_accounting(
    llama_cache_acct_ledger & ledger,
    const llama_cache_acct_resource_domain & domain) noexcept;

// Configuration-owner hook for the store's dedicated pinned-host domain.
// No other producer has pinned state, so this composes empty observation,
// retention-sidecar certification, and the exact-domain proof.
bool server_vbr_artifact_store_configure_pinned_accounting(
    llama_cache_acct_ledger & ledger,
    const llama_cache_acct_resource_domain & domain) noexcept;

// Prove that every budget-participating cell the capture can price is known
// and certified in each exact topology-qualified domain. This is read-only and
// runs after the ordinary host/live producers have certified their rows.
bool server_vbr_artifact_store_verify_accounting(
    llama_cache_acct_ledger & ledger,
    const std::vector<llama_cache_acct_resource_domain> & domains) noexcept;

struct server_vbr_artifact_capture_output {
    server_vbr_artifact_capture_status status =
        server_vbr_artifact_capture_status::internal_error;
    vbr_explicit_capture_status library_status =
        vbr_explicit_capture_status::internal_error;
    vbr_explicit_capture_phase phase =
        vbr_explicit_capture_phase::validation;
    vbr_capture_stream_status inner_stream_status =
        vbr_capture_stream_status::_count;
    vbr_explicit_generation_failure generation_failure =
        vbr_explicit_generation_failure::none;
    vbr_explicit_size_failure size_failure =
        vbr_explicit_size_failure::none;
    vbr_capture_begin_diagnostics begin_diagnostics;
    std::string reference;
    vbr_artifact_consistency_kind consistency =
        vbr_artifact_consistency_kind::capture_exact;
    uint32_t controllers = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    bool dedup = false;
};

struct server_vbr_artifact_store_counters {
    uint64_t requested = 0;
    uint64_t exact_published = 0;
    uint64_t refused = 0;
    uint64_t unavailable = 0;
    uint64_t internal_error = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t chunks = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t dedup_hits = 0;
    uint64_t dedup_misses = 0;
    uint64_t staging_overlap_refusals = 0;
    std::array<uint64_t,
        size_t(vbr_explicit_capture_status::_count)> capture_outcomes = {};
};

// F3.3 server owner for the internal catalog/ring. The library capture is the
// only producer of authorization masks and content identities; this layer
// returns a tenant-bound opaque handle but intentionally exposes no resolution
// surface until F4 supplies validated tenant authorization.
class server_vbr_artifact_store {
public:
    static std::unique_ptr<server_vbr_artifact_store> create(
        const server_vbr_artifact_store_config & config,
        server_vbr_artifact_capture_status & status,
        server_vbr_artifact_store_create_diagnostics * diagnostics =
            nullptr) noexcept;

    ~server_vbr_artifact_store();
    server_vbr_artifact_store(const server_vbr_artifact_store &) = delete;
    server_vbr_artifact_store & operator=(
        const server_vbr_artifact_store &) = delete;

    server_vbr_artifact_capture_output capture(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept;

    const server_vbr_artifact_store_counters & counters() const noexcept;
    uint32_t attention_children() const noexcept;

private:
    struct impl;
    explicit server_vbr_artifact_store(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};
