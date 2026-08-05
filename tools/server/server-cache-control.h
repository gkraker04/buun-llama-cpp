#pragma once

#include "../../common/common-cache-family.h"
#include "server-cache-lease.h"
#include "server-retention-sidecar.h"

#include <array>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

class server_vbr_artifact_store;

enum class server_cache_control_status : uint8_t {
    ok = 0,
    invalid_request,
    not_supported,
    not_found,
    identity_unavailable,
    subject_busy,
    fallback_unavailable,
    fallback_invalid,
    hard_lease_blocked,
    lease_conflict,
    lease_expired,
    partially_stale,
    subject_lost,
    orphaned,
    already_released,
    profile_unfitted,
    capacity_refused,
    stale_capability,
    internal_fault,
    _count,
};

const char * server_cache_control_status_name(
    server_cache_control_status status) noexcept;

// Scheduler task-door precheck. E1's two-copy guarantee relies on lifecycle
// publication/floor enforcement; debug-only authority is observability and
// must refuse rather than construct a lease whose pin its erasers ignore.
server_cache_control_status server_cache_control_task_precheck(
    bool request_present,
    bool lifecycle_available,
    bool substrate_available) noexcept;

struct server_cache_control_token {
    uint64_t high = 0;
    uint64_t low = 0;
    explicit operator bool() const noexcept { return high != 0 && low != 0; }
};

inline bool operator==(
        server_cache_control_token a,
        server_cache_control_token b) noexcept {
    return a.high == b.high && a.low == b.low;
}

enum class server_cache_control_subject_kind : uint8_t {
    live_prefix = 0,
    host_snapshot,
    vbr_reference,
    live_checkpoint, // closed v1 rejection for both subjects and fallbacks.
    _count,
};

enum class server_cache_control_operation : uint8_t {
    holder_create = 0,
    holder_close,
    holder_reattach,
    family_register,
    family_bind,
    lease_acquire,
    lease_inspect,
    lease_renew,
    lease_release,
    events,
    _count,
};

enum class server_cache_control_protection_state : uint8_t {
    current = 0,
    partially_stale,
    subject_lost,
    orphaned,
    released,
    _count,
};

struct server_cache_control_selector {
    server_cache_control_subject_kind kind =
        server_cache_control_subject_kind::live_prefix;
    // E1.1a is scheduler-internal. E1.2 converts semantic wire selectors into
    // this exact association; raw keys never cross the HTTP boundary.
    server_retention_instance_key retention_key;
    std::string reference;
    std::string tenant_key;
    server_cache_lease_identity identity;
    server_cache_lease_frontier frontier;
};

struct server_cache_control_request {
    server_cache_control_token holder;
    server_cache_control_token recovery;
    server_cache_control_token lease;
    server_cache_control_token family;
    common_cache_family_role family_role = common_cache_family_role::_count;
    // E1.2 supplies a bounded client idempotency digest. Zero is allowed only
    // for scheduler-internal tests and receives no response-loss replay.
    uint64_t idempotency_key = 0;
    server_cache_lease_class requested_class = server_cache_lease_class::soft;
    uint64_t ttl_ns = 0;
    server_cache_control_selector subject;
    server_cache_control_selector fallback;
};

struct server_cache_control_event_view {
    uint64_t ordinal = 0;
    server_cache_control_operation operation =
        server_cache_control_operation::holder_create;
    server_cache_control_status status = server_cache_control_status::ok;
    server_cache_lease_class cls = server_cache_lease_class::none;
};

struct server_cache_control_result {
    server_cache_control_status status =
        server_cache_control_status::internal_fault;
    server_cache_control_token holder;
    server_cache_control_token holder_recovery;
    server_cache_control_token lease;
    server_cache_control_token family;
    server_cache_control_token family_binding;
    // Scheduler-internal resolved value. E1.2 serializes only opaque handles.
    common_cache_family_binding cache_family;
    server_cache_lease_class granted_class = server_cache_lease_class::none;
    server_cache_control_protection_state protection =
        server_cache_control_protection_state::released;
    server_cache_lease_frontier lease_frontier;
    server_cache_lease_frontier proven_frontier;
    uint64_t expires_at_ns = 0;
    std::vector<server_cache_control_event_view> events;
};

class server_cache_control_token_source {
public:
    virtual ~server_cache_control_token_source() = default;
    virtual bool next(server_cache_control_token & out) noexcept = 0;
};

struct server_cache_control_config {
    using refresh_subject_fn = bool (*)(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier) noexcept;
    using resolve_vbr_fn = server_cache_control_status (*)(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_subject & subject,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier,
        server_cache_durable_fallback_proof & pin) noexcept;
    using acquire_host_proof_fn = server_cache_durable_fallback_proof (*)(
        void * context,
        const server_cache_control_selector & selector) noexcept;
    server_cache_lease_table * leases = nullptr;
    server_retention_sidecar_store * retention = nullptr;
    server_vbr_artifact_store * artifacts = nullptr;
    server_cache_lease_clock * clock = nullptr;
    server_cache_control_token_source * tokens = nullptr;
    void * refresh_context = nullptr;
    refresh_subject_fn refresh_subject = nullptr;
    void * resolve_vbr_context = nullptr;
    resolve_vbr_fn resolve_vbr = nullptr;
    void * host_proof_context = nullptr;
    acquire_host_proof_fn acquire_host_proof = nullptr;
    size_t max_holders = 64;
    size_t max_leases = 1024;
    size_t max_families = 1024;
    size_t max_family_bindings = 4096;
    // Model-free allocation-fault seams. Production must leave both defaults;
    // the E1 contract scan forbids assignments outside tests.
    size_t test_fail_note_after = std::numeric_limits<size_t>::max();
    bool test_fail_remember = false;
};

// Scheduler-owned E1 authority. It is also the lease table's one fallback
// provider: a proof is staged only while one scheduler transaction calls the
// existing grant/renew door, then consumed exactly once by acquire().
class server_cache_control_authority final :
        private server_cache_lease_fallback_provider {
public:
    explicit server_cache_control_authority(
        const server_cache_control_config & config) noexcept;
    ~server_cache_control_authority();
    server_cache_control_authority(const server_cache_control_authority &) = delete;
    server_cache_control_authority & operator=(
        const server_cache_control_authority &) = delete;

    server_cache_control_result execute(
        server_cache_control_operation operation,
        const server_cache_control_request & request) noexcept;
    // Completion launch resolves the opaque binding on the scheduler thread.
    // Closed/expired holders and unknown handles are indistinguishable misses.
    server_cache_control_status resolve_family_binding(
        server_cache_control_token token,
        common_cache_family_binding & out) noexcept;
    void lifecycle_point() noexcept;
    bool available() const noexcept;

private:
    struct impl;
    server_cache_durable_fallback_proof acquire(
        const server_cache_lease_subject & subject,
        const server_cache_lease_identity & identity) noexcept override;
    std::unique_ptr<impl> state_;
};
