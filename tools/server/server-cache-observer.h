#pragma once

#include "common-cache-plan.h"
#include "server-cache-fingerprint.h"

#include <array>
#include <cstdint>
#include <limits>

// ZC2 is deliberately an observation-only substrate.  These types contain no
// coefficient solver, persistence handle, or authority bit.  ZC3 supplies the
// stable execution identity and ZC4 is the first unit allowed to consume the
// sufficient state below for a fitted estimate.

enum class server_cache_observation_operation : uint8_t {
    replay = 0,
    restore,
    durability_prepare,
    destruction_apply,
    _count,
};

enum class server_cache_observation_terminal : uint8_t {
    accepted = 0,
    diagnostic,
    operation_unavailable,
    _count,
};

enum class server_cache_observation_reason : uint8_t {
    none = 0,
    identity_unavailable,
    mixed_slots,
    multiple_submissions,
    no_completion_fence,
    operation_failed,
    warmup_unsettled,
    invalid_geometry,
    instance_capacity,
    numeric_overflow,
    _count,
};

// Persisted authority-relevant estimator state is carried by the live
// instance even before ZC4 consumes it. This prevents the persistence seam
// from silently dropping future non-tail terminals on restart.
enum class server_cache_calibration_authority_terminal : uint8_t {
    none = 0,
    tail_exceeded,
    confidence_budget_exhausted,
    ordinal_exhausted,
    numeric_fault,
    _count,
};

const char * server_cache_observation_operation_name(
    server_cache_observation_operation value) noexcept;
const char * server_cache_observation_terminal_name(
    server_cache_observation_terminal value) noexcept;
const char * server_cache_observation_reason_name(
    server_cache_observation_reason value) noexcept;

bool server_cache_observation_replay_feature(
    uint64_t tokens,
    uint8_t & size_family,
    std::array<double, 4> & feature) noexcept;
bool server_cache_observation_byte_feature(
    uint64_t bytes,
    uint8_t & size_family,
    std::array<double, 4> & feature) noexcept;
uint64_t server_cache_observation_response_cap_us(
    server_cache_observation_operation operation,
    uint8_t size_family) noexcept;

struct server_cache_observation_key {
    server_cache_observation_operation operation =
        server_cache_observation_operation::replay;
    common_cache_plan_provider provider = common_cache_plan_provider::live_slot;
    uint8_t restore_kind = 0;
    uint8_t prepare_shape = 0;
    uint8_t contention_bucket = 0;
    uint8_t start_bucket = 0;
    uint8_t batch_bucket = 0;
    uint8_t ubatch_bucket = 0;
    uint8_t size_family = 0;
    uint8_t feature_dim = 0;
    // Process/profile identity is deliberately separate from the operated-on
    // state. ZC4 supplies participant and representation digests at the real
    // provider seam; a global execution root must never impersonate either.
    std::array<uint8_t, 32> profile_execution_digest = {};
    std::array<uint8_t, 32> participant_execution_digest = {};
    std::array<uint8_t, 32> adapter_application_digest = {};
    std::array<uint8_t, 32> representation_digest = {};
    std::array<uint8_t, 32> effect_action_shape_digest = {};
    bool adapter_application_complete = false;
    bool identity_complete = false;
    bool identity_exact = false;

    bool operator==(const server_cache_observation_key & other) const noexcept;
};

bool server_cache_observation_key_valid(
    const server_cache_observation_key & key) noexcept;

struct server_cache_observation_record {
    server_cache_observation_key key;
    std::array<double, 4> feature = {};
    int32_t slot_id = -1;
    uint32_t prompt_slots = 0;
    uint32_t active_slots = 0;
    uint32_t submissions = 0;
    uint32_t tokens = 0;
    int64_t start_position = -1;
    uint32_t effective_batch = 0;
    uint32_t effective_ubatch = 0;
    bool target_participates = false;
    bool draft_participates = false;
    bool speculative_participates = false;
    bool warmup = false;
    uint64_t payload_bytes = 0;
    uint64_t owned_cpu_us = 0;
    uint64_t backend_service_us = 0;
    uint64_t owned_service_us = 0;
    uint64_t capped_service_us = 0;
    bool tail_exceeded = false;
    uint64_t fence_serial_before = 0;
    uint64_t fence_serial_after = 0;
    server_cache_observation_terminal terminal =
        server_cache_observation_terminal::diagnostic;
    server_cache_observation_reason reason =
        server_cache_observation_reason::none;
};

struct server_cache_observation_instance {
    static constexpr size_t residual_capacity = 8;

    bool used = false;
    server_cache_observation_key key;
    std::array<std::array<double, 4>, 4> v = {};
    std::array<double, 4> b = {};
    std::array<uint64_t, residual_capacity> response_reservoir = {};
    uint64_t n_success = 0;
    uint64_t reservoir_seen = 0;
    bool tail_exceeded = false;
    uint64_t tail_actual_max_us = 0;
    uint64_t fit_generation = 0;
    server_cache_calibration_authority_terminal authority_terminal =
        server_cache_calibration_authority_terminal::none;
    std::array<double, 4> feature_min = {};
    std::array<double, 4> feature_max = {};
    uint64_t qualified_execution_ordinal = 0;
    std::array<double, 6> log_wealth = {};
    uint64_t n_validation = 0;
    std::array<uint64_t, 8> fit_region_minutes = {};
    uint8_t fit_region_count = 0;
    std::array<uint64_t, 8> validation_region_minutes = {};
    uint8_t validation_region_count = 0;
    uint64_t safe_measurable_opportunities = 0;
    uint64_t opportunity_at_last_validation = 0;
    uint64_t last_fit_unix_ms = 0;
    uint64_t last_validation_unix_ms = 0;
};

struct server_cache_observation_counters {
    uint64_t accepted = 0;
    uint64_t diagnostic = 0;
    uint64_t operation_unavailable = 0;
    uint64_t instance_capacity = 0;
    uint64_t numeric_fault = 0;
};

class server_cache_observation_store {
public:
    static constexpr size_t instance_capacity = 128;
    static constexpr size_t diagnostic_capacity = 256;

    // Qualifies in place before the row is retained. Callers therefore emit
    // the same typed terminal that counters and sufficient state observed.
    bool observe(server_cache_observation_record & record) noexcept;

    const server_cache_observation_counters & counters() const noexcept {
        return counters_;
    }
    const std::array<server_cache_observation_instance,
                     instance_capacity> & instances() const noexcept {
        return instances_;
    }
    const std::array<server_cache_observation_record,
                     diagnostic_capacity> & recent_records() const noexcept {
        return recent_records_;
    }
    uint64_t records_seen() const noexcept { return records_seen_; }
    uint64_t mutation_generation() const noexcept { return mutation_generation_; }
    void set_execution_fingerprint(
        const server_cache_execution_fingerprint & value) noexcept;
    void apply_execution_fingerprint(
        server_cache_observation_key & key) const noexcept;
    const server_cache_execution_fingerprint & execution_fingerprint() const noexcept {
        return execution_fingerprint_;
    }
    bool restore_persisted_instances(
        const std::array<server_cache_observation_instance,
                         instance_capacity> & instances,
        uint64_t mutation_generation) noexcept;

private:
    std::array<server_cache_observation_instance, instance_capacity> instances_ = {};
    std::array<server_cache_observation_record, diagnostic_capacity> recent_records_ = {};
    server_cache_observation_counters counters_;
    uint64_t records_seen_ = 0;
    uint64_t mutation_generation_ = 0;
    server_cache_execution_fingerprint execution_fingerprint_;
};

server_cache_observation_key server_cache_observation_cpu_key(
    server_cache_observation_operation operation,
    common_cache_plan_provider provider,
    uint8_t prepare_shape) noexcept;

// The tooling-visible serializer has one owner and is unable to perturb the
// operation terminal. Debug allocation or serialization failure is swallowed.
void server_cache_emit_observation_noexcept(
    const server_cache_observation_record & record,
    bool enabled) noexcept;

bool server_cache_observe_cpu_operation(
    server_cache_observation_store * store,
    server_cache_observation_key key,
    int32_t slot_id,
    uint64_t operation_extent_bytes,
    int64_t owned_start_us,
    int64_t owned_end_us,
    bool success,
    server_cache_observation_record * emitted = nullptr) noexcept;

struct server_cache_sync_fence_snapshot {
    uint64_t serial = 0;
    int64_t completed_us = 0;
};

struct server_cache_observation_submission {
    int64_t submission_us = 0;
    server_cache_sync_fence_snapshot fence_before;
    uint32_t prompt_slots = 0;
    uint32_t active_slots = 0;
    uint32_t tokens = 0;
    uint64_t payload_bytes = 0;
    int64_t start_position = -1;
    uint32_t effective_batch = 0;
    uint32_t effective_ubatch = 0;
    bool target_participates = false;
    bool draft_participates = false;
    bool speculative_participates = false;
    bool warmup = false;
    uint8_t contention_bucket = 0;
    uint8_t start_bucket = 0;
    uint8_t batch_bucket = 0;
    uint8_t ubatch_bucket = 0;
    uint8_t size_family = 0;
    std::array<double, 4> feature = {};
};

class server_cache_calibration_epoch {
public:
    void arm(
        int32_t slot_id,
        server_cache_observation_key key,
        uint64_t initial_owned_cpu_us = 0) noexcept;
    void bind_provider(
        server_cache_observation_key key,
        uint64_t payload_bytes,
        uint64_t initial_owned_cpu_us = 0) noexcept;
    void begin_owned_cpu(int64_t start_us) noexcept;
    void pause_owned_cpu(int64_t end_us) noexcept;
    void mark_mixed(server_cache_observation_reason reason) noexcept;
    void note_submission(const server_cache_observation_submission & value) noexcept;
    bool latch_fence(server_cache_sync_fence_snapshot fence_after) noexcept;
    void mark_operation_terminal() noexcept { terminal_ready_ = active_; }
    bool finish(
        server_cache_observation_store & store,
        server_cache_observation_record * emitted = nullptr) noexcept;
    bool abandon(
        server_cache_observation_reason reason,
        server_cache_observation_store & store,
        server_cache_observation_record * emitted = nullptr) noexcept;

    bool active() const noexcept { return active_; }
    bool submitted() const noexcept { return submissions_ != 0; }
    bool terminal_ready() const noexcept { return terminal_ready_; }
    int32_t slot_id() const noexcept { return slot_id_; }
    uint32_t submissions() const noexcept { return submissions_; }
    uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    server_cache_observation_operation operation() const noexcept {
        return key_.operation;
    }
    void reset() noexcept { *this = {}; }

private:
    server_cache_observation_record make_record(
        server_cache_sync_fence_snapshot fence_after,
        server_cache_observation_reason forced_reason) const noexcept;

    server_cache_observation_key key_;
    std::array<double, 4> feature_ = {};
    int32_t slot_id_ = -1;
    uint64_t owned_cpu_us_ = 0;
    int64_t owned_cpu_segment_start_us_ = 0;
    int64_t first_submission_us_ = 0;
    server_cache_sync_fence_snapshot fence_before_;
    server_cache_sync_fence_snapshot first_fence_;
    bool has_fence_ = false;
    bool terminal_ready_ = false;
    uint32_t prompt_slots_ = 0;
    uint32_t active_slots_ = 0;
    uint32_t submissions_ = 0;
    uint32_t tokens_ = 0;
    uint64_t payload_bytes_ = 0;
    int64_t start_position_ = -1;
    uint32_t effective_batch_ = 0;
    uint32_t effective_ubatch_ = 0;
    bool target_participates_ = false;
    bool draft_participates_ = false;
    bool speculative_participates_ = false;
    bool warmup_ = false;
    server_cache_observation_reason preexisting_reason_ =
        server_cache_observation_reason::none;
    bool active_ = false;
};
