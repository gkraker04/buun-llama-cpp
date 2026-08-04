#pragma once

#include "common-cache-plan.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>

// B-A pre-mutation decision substrate. It is process-local and contains no
// shipped cache state. B-A0b only dual-runs: execution remains legacy.
struct server_cache_plan_authority {
    common_cache_plan_authority_level configured_level =
        common_cache_plan_authority_level::off;
    common_cache_plan_authority_counters counters;
    std::string calibration_profile;

    explicit server_cache_plan_authority(
        common_cache_plan_authority_level level) noexcept : configured_level(level) {}

    // Runs the existing cost planner against the complete, target-qualified
    // pre-mutation inventory. Capability is sampled on both sides of the call;
    // drift refuses qualification rather than changing the shipped path.
    void plan_before_mutation(
        common_cache_plan_record & rec,
        uint64_t capability_before,
        uint64_t capability_after) noexcept;

    void fail_closed(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason =
            common_cache_plan_authority_fallback::internal_fault) noexcept;

    // Legacy execution has completed. Preserve the pre-mutation counterfactual
    // and fill only the legacy/executed sides of the schema-v5 receipt.
    void finalize_legacy_execution(common_cache_plan_record & rec) noexcept;
};

// Stable, allocation-free capability fold used at the immediately-pre-mutation
// revalidation seam. Callers fold exactly the state that made candidates usable.
uint64_t server_cache_plan_capability_fold(
    uint64_t hash,
    uint64_t value) noexcept;

constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE = 1000000;
constexpr int32_t SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE = 10000;
constexpr int32_t SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID =
    (INT32_MAX - SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE -
     (SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE - 1)) /
    SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE;

constexpr int32_t server_cache_plan_host_checkpoint_source_id(
        int32_t host_source_id,
        int32_t checkpoint_ordinal = 0) noexcept {
    return host_source_id < 0 ||
           host_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
           checkpoint_ordinal < 0 ||
           checkpoint_ordinal >= SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE
        ? -1
        : SERVER_CACHE_PLAN_HOST_CHECKPOINT_BASE +
          host_source_id*SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE +
          checkpoint_ordinal;
}

// Observer-only host-state identity. The fixed request-local registry assigns
// small wire ids to list-node addresses without serializing or hashing the
// pointer. Surviving nodes retain identity across save dedup/eviction; the new
// staged node has a distinct address and takes the next id.
bool server_cache_plan_find_or_assign_source_id(
    const void * instance,
    std::array<const void *, COMMON_CACHE_PLAN_MAX_CANDIDATES> & instances,
    uint32_t & n_instances,
    int32_t & source_id) noexcept;

// Checkpoint containers are enumerated forward by the authority inventory but
// reverse by the shipped selector. Translate reverse visit position back to
// the forward ordinal used by both live and host-composed inventory rows.
int32_t server_cache_plan_checkpoint_source_id_from_reverse(
    size_t checkpoint_count,
    uint32_t reverse_ordinal,
    int32_t host_source_id = -1) noexcept;

constexpr bool server_cache_plan_viable(
        common_cache_plan_reason reason) noexcept {
    return reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
}

struct server_cache_plan_live_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    float sim = 0.0f;
};

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
    bool busy,
    bool has_payload,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens) noexcept;

void server_cache_plan_apply_live(
    common_cache_plan_candidate * row,
    const server_cache_plan_live_evaluation & evaluation) noexcept;

struct server_cache_plan_host_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
    float sim = 0.0f;
    float f_keep = 0.0f;
};

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
    bool payload_present,
    bool identity_matches,
    uint64_t lcp_tokens,
    uint64_t prompt_tokens,
    uint64_t source_tokens,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_host(
    common_cache_plan_candidate * row,
    const server_cache_plan_host_evaluation & evaluation) noexcept;

struct server_cache_plan_checkpoint_evaluation {
    common_cache_plan_reason reason = COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE;
    uint64_t lcp_tokens = 0;
    uint64_t payload_bytes = 0;
};

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
    bool payload_present,
    bool frontier_current,
    bool recurrent,
    bool representation_matches,
    int64_t pos_min,
    int64_t pos_max,
    int64_t next_position,
    int64_t min_position_threshold,
    uint64_t payload_bytes) noexcept;

void server_cache_plan_apply_checkpoint(
    common_cache_plan_candidate * row,
    const server_cache_plan_checkpoint_evaluation & evaluation) noexcept;
