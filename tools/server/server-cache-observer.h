#pragma once

#include "common-cache-plan.h"
#include "server-cache-fingerprint.h"

#include <algorithm>
#include <array>
#include <cstddef>
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

// Version-2 estimator classes are frozen independently of the broad
// operation enum. In particular, flat/scaled is a pre-outcome provider
// contract and never an empirical choice made after seeing a response.
enum class server_cache_calibration_model_kind : uint8_t {
    replay_scaled = 0,
    restore_scaled,
    restore_flat,
    durability_prepare_scaled,
    durability_prepare_flat,
    destruction_apply_scaled,
    destruction_apply_flat,
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
    drifted,
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
uint8_t server_cache_observation_batch_bucket(uint32_t batch) noexcept;
uint8_t server_cache_observation_start_bucket(int64_t position) noexcept;
bool server_cache_observation_replay_chain_geometry(
    uint64_t tokens,
    uint32_t max_effective_batch,
    uint8_t & size_family,
    uint8_t & batch_bucket,
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
    server_cache_calibration_model_kind model_kind =
        server_cache_calibration_model_kind::replay_scaled;
    uint64_t operation_extent_bytes = 0;
    uint8_t target_draft_spec_composition = 0;
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
bool server_cache_observation_apply_restore_geometry(
    server_cache_observation_key & key,
    int64_t reference_frontier,
    uint32_t effective_batch,
    uint32_t effective_ubatch) noexcept;

// Admission time is captured at the first provider-owned operation seam,
// before any response is available. Estimator admission must never resample
// either clock from the completion-side observer.
struct server_cache_observation_admission_clock {
    bool valid = false;
    int64_t steady_us = 0;
    uint64_t unix_ms = 0;
};

struct server_cache_observation_cpu_start {
    int64_t owned_start_us = 0;
    server_cache_observation_admission_clock admission_clock;
};

server_cache_observation_admission_clock
server_cache_observation_capture_admission_clock() noexcept;
server_cache_observation_cpu_start
server_cache_observation_capture_cpu_start(bool enabled) noexcept;

struct server_cache_observation_record {
    server_cache_observation_key key;
    server_cache_observation_admission_clock admission_clock;
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
    uint32_t calibration_instance_slot = UINT32_MAX;
    // Closed internal assignment codes: 0 none, 1 fit, 2 validation,
    // 3 validation-unavailable, 4 fit-rate-limited.
    uint8_t calibration_assignment = 0;
    uint64_t calibration_n_fit = 0;
    uint64_t calibration_n_validation = 0;
    // Closed internal state code: 0 unavailable, otherwise enum+1 from the
    // ZC4 model. Claim ordinals are debug evidence, never a replayable ticket.
    uint8_t calibration_profile_state = 0;
    bool calibration_claim_available = false;
    uint64_t calibration_boot_claim_ordinal = 0;
    uint64_t calibration_profile_generation = 0;
    uint64_t calibration_fit_generation = 0;
    bool calibration_prediction_available = false;
    double calibration_prediction_us = 0.0;
    double calibration_radius_us = 0.0;
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
    // Physical registry identity is process-local and reconstructed from the
    // fixed array index on restore.  Confidence claims must bind this value as
    // well as g; a caller cannot spend another slot's confidence allocation.
    uint32_t estimator_slot = UINT32_MAX;
    server_cache_observation_key key;
    std::array<std::array<double, 4>, 4> v = {};
    std::array<double, 4> b = {};
    // Estimator v3 stores absolute pre-update held-out residuals here.  The
    // stable wire field name is retained so the estimator-version gate, rather
    // than an ad-hoc schema fork, owns the semantic transition from v2's raw
    // responses.
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
    // Process-local admission currency. It deliberately resets at boot and is
    // never serialized as a cross-restart wall-clock claim.
    uint64_t last_fit_steady_second = std::numeric_limits<uint64_t>::max();
    std::array<double, 6> log_wealth = {};
    uint64_t n_validation = 0;
    std::array<uint64_t, 8> fit_region_minutes = {};
    uint8_t fit_region_count = 0;
    std::array<uint64_t, 8> validation_region_minutes = {};
    uint8_t validation_region_count = 0;
    uint64_t safe_measurable_opportunities = 0;
    // Process-local de-duplication key for read-only inventory observations.
    uint64_t last_opportunity_inventory_ordinal = UINT64_MAX;
    uint64_t opportunity_at_last_validation = 0;
    uint64_t last_fit_unix_ms = 0;
    uint64_t last_validation_unix_ms = 0;
};

struct server_cache_calibration_claim_identity {
    bool available = false;
    uint64_t boot_claim_ordinal = 0;
    uint64_t profile_generation_ordinal = 0;
    uint32_t estimator_slot = 0;
    uint64_t fit_generation = 0;
};

struct server_cache_observation_counters {
    uint64_t accepted = 0;
    uint64_t diagnostic = 0;
    uint64_t operation_unavailable = 0;
    uint64_t instance_capacity = 0;
    uint64_t numeric_fault = 0;
    uint64_t fit_rows = 0;
    uint64_t validation_rows = 0;
    uint64_t validation_unavailable = 0;
    uint64_t fit_rate_limited = 0;
    uint64_t drifted = 0;
};

struct server_cache_calibration_principal_cell {
    bool used = false;
    uint32_t estimator_slot = 0;
    uint64_t principal_hash = 0;
    uint64_t window = 0;
    uint8_t fit_rows = 0;
};

enum class server_cache_resume_validation_outcome_kind : uint8_t {
    unavailable = 1,
    succeeded = 2,
};

struct server_cache_resume_validation_outcome {
    uint32_t estimator_slot = UINT32_MAX;
    server_cache_resume_validation_outcome_kind kind =
        server_cache_resume_validation_outcome_kind::unavailable;
};

using server_cache_resume_validation_flags = std::array<bool, 128>;

class server_cache_observation_store {
public:
    static constexpr size_t instance_capacity = 128;
    static constexpr size_t diagnostic_capacity = 256;
    static constexpr size_t slot_scratch_capacity = 4096;

    // Qualifies in place before the row is retained. Callers therefore emit
    // the same typed terminal that counters and sufficient state observed.
    bool observe(server_cache_observation_record & record) noexcept;
    bool note_safe_measurable_opportunity(
        const server_cache_observation_key & key,
        uint64_t inventory_ordinal) noexcept;

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
    uint64_t authority_currency_serial() const noexcept {
        return authority_currency_serial_;
    }
    bool calibration_claim(
        uint32_t estimator_slot,
        server_cache_calibration_claim_identity & out) const noexcept;
    bool authority_admission_allowed(uint32_t estimator_slot) const noexcept;
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
    void set_calibration_claim_identity(
        bool available,
        uint64_t boot_claim_ordinal,
        uint64_t profile_generation_ordinal) noexcept;
    void set_committed_profile_mutation_generation(uint64_t value) noexcept;
    void set_operation_identity(
        bool complete,
        const std::array<uint8_t, 32> & representation_digest,
        uint8_t target_draft_spec_composition) noexcept;
    void set_resume_state(
                          const server_cache_resume_validation_flags & validation_pending,
                          const server_cache_resume_validation_flags & authority_validation_required,
                          int64_t fit_barrier_started_us) noexcept;
    size_t take_resume_validation_outcomes(
        server_cache_resume_validation_outcome * out,
        size_t capacity) noexcept;
    void set_scheduler_active_slots(uint32_t value) noexcept {
        scheduler_active_slots_ = value;
    }
    bool cpu_operation_isolated() const noexcept {
        return scheduler_active_slots_ <= 1;
    }
    bool prepare_slot_scratch(size_t count) noexcept {
        if (count > slot_scratch_capacity) return false;
        slot_scratch_count_ = count;
        reset_slot_scratch();
        return true;
    }
    void reset_slot_scratch() noexcept {
        std::fill_n(slot_batch_tokens_.begin(), slot_scratch_count_, 0);
        std::fill_n(slot_first_positions_.begin(), slot_scratch_count_,
                    std::numeric_limits<int32_t>::max());
    }
    bool note_slot_submission(
            int32_t slot_id, uint32_t tokens, int32_t first_position) noexcept {
        if (slot_id < 0 || size_t(slot_id) >= slot_scratch_count_ ||
            tokens > UINT32_MAX - slot_batch_tokens_[size_t(slot_id)]) return false;
        slot_batch_tokens_[size_t(slot_id)] += tokens;
        slot_first_positions_[size_t(slot_id)] = std::min(
            slot_first_positions_[size_t(slot_id)], first_position);
        return true;
    }
    uint32_t slot_batch_tokens(size_t slot_id) const noexcept {
        return slot_id < slot_scratch_count_ ? slot_batch_tokens_[slot_id] : 0;
    }
    int32_t slot_first_position(size_t slot_id) const noexcept {
        return slot_id < slot_scratch_count_
            ? slot_first_positions_[slot_id]
            : std::numeric_limits<int32_t>::max();
    }
    size_t slot_scratch_count() const noexcept { return slot_scratch_count_; }

private:
    std::array<server_cache_observation_instance, instance_capacity> instances_ = {};
    std::array<server_cache_observation_record, diagnostic_capacity> recent_records_ = {};
    std::array<server_cache_calibration_principal_cell, 64> principal_cells_ = {};
    server_cache_observation_counters counters_;
    uint64_t records_seen_ = 0;
    uint64_t mutation_generation_ = 0;
    uint64_t authority_currency_serial_ = 1;
    server_cache_execution_fingerprint execution_fingerprint_;
    bool claim_identity_available_ = false;
    uint64_t boot_claim_ordinal_ = 0;
    uint64_t profile_generation_ordinal_ = 0;
    uint64_t committed_profile_mutation_generation_ = UINT64_MAX;
    std::array<uint64_t, instance_capacity>
        generation_started_mutation_ = {};
    bool operation_identity_complete_ = false;
    std::array<uint8_t, 32> operation_representation_digest_ = {};
    uint8_t target_draft_spec_composition_ = 0;
    server_cache_resume_validation_flags resume_validation_pending_ = {};
    server_cache_resume_validation_flags
        resume_authority_validation_required_ = {};
    std::array<server_cache_resume_validation_outcome, instance_capacity>
        resume_validation_outcomes_ = {};
    uint16_t resume_validation_outcome_count_ = 0;
    int64_t resume_fit_barrier_started_us_ = 0;
    uint32_t scheduler_active_slots_ = 0;
    std::array<uint32_t, slot_scratch_capacity> slot_batch_tokens_ = {};
    std::array<int32_t, slot_scratch_capacity> slot_first_positions_ = {};
    size_t slot_scratch_count_ = 0;

    void note_authority_mutation() noexcept;
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
    server_cache_observation_cpu_start owned_start,
    int64_t owned_end_us,
    bool success,
    bool isolated = true,
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

bool server_cache_observation_same_chain_geometry(
    const server_cache_observation_submission & first,
    const server_cache_observation_submission & next) noexcept;

class server_cache_calibration_epoch {
public:
    void arm(
        int32_t slot_id,
        server_cache_observation_key key,
        uint64_t initial_owned_cpu_us = 0,
        server_cache_observation_admission_clock admission_clock =
            server_cache_observation_admission_clock{}) noexcept;
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
    server_cache_observation_admission_clock admission_clock_;
    std::array<double, 4> feature_ = {};
    int32_t slot_id_ = -1;
    uint64_t owned_cpu_us_ = 0;
    int64_t owned_cpu_segment_start_us_ = 0;
    bool owned_cpu_segment_overlapped_ = false;
    int64_t pending_submission_us_ = 0;
    server_cache_sync_fence_snapshot fence_before_;
    server_cache_sync_fence_snapshot pending_fence_before_;
    server_cache_sync_fence_snapshot last_fence_;
    uint64_t backend_service_us_ = 0;
    uint32_t fenced_submissions_ = 0;
    bool submission_pending_ = false;
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
