#pragma once

#include "../../src/llama-cache-accounting.h"

#include <array>
#include <cstddef>
#include <cstdint>

// D-S4's one closed inventory. The second argument is the physical choke-point allowed to
// contain the corresponding raw primitive; the third is its logical admission owner. Two
// full-slot classes deliberately share one manifest builder/admission owner. CI extracts the
// mapping, so adding a class or raw primitive without extending it fails the contract scan.
#define SERVER_CACHE_DESTRUCTION_INVENTORY(X) \
    X(slot_drop,                server_cache_slot_drop_impl,                observe_full_slot) \
    X(live_range_drop,          server_cache_live_range_drop_impl,          observe_live_range_drop) \
    X(host_artifact_drop,       server_prompt_cache_destroy_entry_impl,     server_prompt_cache_observe_drop) \
    X(checkpoint_drop,          server_cache_checkpoint_drop_impl,          checkpoint_drop) \
    X(token_ledger_truncate,    server_cache_token_ledger_truncate_impl,    token_ledger_truncate) \
    X(mandatory_recovery_reset, server_cache_mandatory_recovery_reset_impl, observe_full_slot)

enum class server_cache_destruction_class : uint8_t {
#define SERVER_CACHE_DESTRUCTION_CLASS(name, symbol, admission) name,
    SERVER_CACHE_DESTRUCTION_INVENTORY(SERVER_CACHE_DESTRUCTION_CLASS)
#undef SERVER_CACHE_DESTRUCTION_CLASS
    _count,
};

enum class server_cache_destruction_reason : uint8_t {
    slot_rebind = 0,
    child_release,
    idle_reclaim,
    live_prefix_replace,
    context_shift,
    checkpoint_invalidated,
    checkpoint_thin,
    checkpoint_capacity,
    host_dedup,
    host_capacity,
    host_token_limit,
    host_consumed_restore,
    host_shutdown,
    low_lcp_reset,
    restore_failure,
    trim_rejection,
    transient_speculative,
    _count,
};

enum class server_cache_destruction_target_kind : uint8_t {
    live_target = 0,
    live_draft,
    token_ledger,
    checkpoint_ring,
    rolling_window,
    typed_accelerator,
    host_artifact,
    _count,
};

enum class server_cache_destruction_verdict : uint8_t {
    admit_unleased = 0,
    _count,
};

enum class server_cache_destruction_execution : uint8_t {
    pass_through = 0,
    _count,
};

struct server_cache_destruction_target {
    server_cache_destruction_target_kind kind =
        server_cache_destruction_target_kind::live_target;
    int32_t slot_id = -1;
    llama_cache_acct_artifact_id artifact;
    bool artifact_known = false;
};

struct server_cache_destruction_yield {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    bool domain_known = false;
    llama_cache_acct_measure measure =
        llama_cache_acct_measure::logical_payload;
    llama_cache_acct_value value = {
        0, llama_cache_acct_known::unavailable,
    };
};

constexpr size_t SERVER_CACHE_DESTRUCTION_MAX_TARGETS = 8;
constexpr size_t SERVER_CACHE_DESTRUCTION_MAX_YIELDS  = 16;
constexpr size_t SERVER_CACHE_DESTRUCTION_EVENT_RING  = 64;

struct server_cache_destruction_request {
    server_cache_destruction_class cls =
        server_cache_destruction_class::slot_drop;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::slot_rebind;
    std::array<server_cache_destruction_target,
               SERVER_CACHE_DESTRUCTION_MAX_TARGETS> targets = {};
    std::array<server_cache_destruction_yield,
               SERVER_CACHE_DESTRUCTION_MAX_YIELDS> yields = {};
    uint8_t n_targets = 0;
    uint8_t n_yields  = 0;
    bool overflowed   = false;

    void add_target(server_cache_destruction_target_kind kind, int32_t slot_id) noexcept {
        if (n_targets >= targets.size()) {
            overflowed = true;
            return;
        }
        auto & target = targets[n_targets++];
        target.kind    = kind;
        target.slot_id = slot_id;
    }

    void add_yield(const server_cache_destruction_yield & value) noexcept {
        if (n_yields >= yields.size()) {
            overflowed = true;
            return;
        }
        yields[n_yields++] = value;
    }

    void add_yield(llama_cache_acct_category category) noexcept {
        server_cache_destruction_yield value;
        value.category = category;
        add_yield(value);
    }
};

// A logical operation keeps this small token across split physical phases. D-S4 and D-S5
// execute pass-through; D-A can later change execution authority without re-cutting joined
// operations such as low-LCP reset.
struct server_cache_destruction_admission {
    server_cache_destruction_class cls =
        server_cache_destruction_class::slot_drop;
    server_cache_destruction_reason reason =
        server_cache_destruction_reason::slot_rebind;
    server_cache_destruction_verdict verdict =
        server_cache_destruction_verdict::admit_unleased;
    server_cache_destruction_execution execution =
        server_cache_destruction_execution::pass_through;
    uint64_t sequence = 0;
    bool issued = false;
    bool observer_recorded = false;

    bool covers(
            server_cache_destruction_class expected_class,
            server_cache_destruction_reason expected_reason) const noexcept {
        return issued && cls == expected_class && reason == expected_reason;
    }
};

struct server_cache_destruction_event {
    server_cache_destruction_request request;
    server_cache_destruction_verdict verdict =
        server_cache_destruction_verdict::admit_unleased;
    server_cache_destruction_execution execution =
        server_cache_destruction_execution::pass_through;
    uint64_t sequence = 0;
};

// Transient, process-local D-S4 observer. It is deliberately absent from cache-plan JSON
// until D-S7. Recording is fixed-capacity and noexcept; overwriting the oldest detail never
// loses the monotone per-class totals.
struct server_cache_destruction_observer {
    std::array<server_cache_destruction_event,
               SERVER_CACHE_DESTRUCTION_EVENT_RING> events = {};
    std::array<uint64_t, size_t(server_cache_destruction_class::_count)> totals = {};
    uint64_t n_events   = 0;
    uint64_t overflows  = 0;

    uint64_t observe(const server_cache_destruction_request & request) noexcept {
        const size_t cls = size_t(request.cls);
        if (cls >= totals.size()) {
            overflows++;
            return 0;
        }
        server_cache_destruction_event & event =
            events[size_t(n_events % events.size())];
        event.request   = request;
        event.verdict   = server_cache_destruction_verdict::admit_unleased;
        event.execution = server_cache_destruction_execution::pass_through;
        event.sequence  = n_events + 1;
        totals[cls]++;
        n_events++;
        if (request.overflowed) {
            overflows++;
        }
        return event.sequence;
    }
};

// The ONE retention-admission API. D-S4 has no lease state, so the simulated verdict is
// admit_unleased and execution is always pass-through. D-S5 replaces only the simulation;
// D-A is the later authority flip.
inline server_cache_destruction_admission server_cache_retention_admit(
        server_cache_destruction_observer * observer,
        const server_cache_destruction_request & request) noexcept {
    server_cache_destruction_admission admission;
    admission.cls    = request.cls;
    admission.reason = request.reason;
    admission.issued = true;
    if (observer) {
        admission.sequence = observer->observe(request);
        admission.observer_recorded = admission.sequence != 0;
    }
    return admission;
}
