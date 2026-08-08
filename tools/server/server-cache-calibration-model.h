#pragma once

#include "server-cache-observer.h"
#include "server-cache-lifecycle.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

// ZC4's numerical procedure is deliberately independent of the scheduler and
// persistence owners.  All inputs are fixed before the observed response is
// inspected, and every routine is allocation-free and noexcept.

struct server_cache_calibration_claim_identity {
    bool available = false;
    uint64_t boot_claim_ordinal = 0;
    uint64_t profile_generation_ordinal = 0;
    uint32_t estimator_slot = 0;
    uint64_t fit_generation = 0;
};

enum class server_cache_calibration_prediction_status : uint8_t {
    ok = 0,
    learning,
    out_of_coverage,
    confidence_budget_exhausted,
    numeric_fault,
};

struct server_cache_calibration_prediction {
    server_cache_calibration_prediction_status status =
        server_cache_calibration_prediction_status::learning;
    double point_us = 0.0;
    double radius_us = 0.0;
    double lower_us = 0.0;
    double upper_us = 0.0;
    double condition_number = 0.0;
    double log_determinant = 0.0;
};

enum class server_cache_calibration_instance_state : uint8_t {
    unseen = 0,
    learning,
    provisional,
    active,
    drifted,
    quarantined,
};
const char * server_cache_calibration_instance_state_name(
    server_cache_calibration_instance_state value) noexcept;

bool server_cache_calibration_predict(
    const server_cache_observation_instance & instance,
    const server_cache_calibration_claim_identity & claim,
    const std::array<double, 4> & feature,
    server_cache_calibration_prediction & out) noexcept;
server_cache_calibration_instance_state server_cache_calibration_state(
    const server_cache_observation_instance & instance,
    const server_cache_calibration_claim_identity & claim,
    const std::array<double, 4> & feature,
    uint64_t now_unix_ms,
    server_cache_calibration_prediction * prediction = nullptr,
    bool authority_admission_allowed = true) noexcept;

enum class server_cache_calibration_contribution_side : uint8_t {
    baseline = 0,
    challenger,
};

struct server_cache_calibration_contribution {
    const server_cache_observation_instance * instance = nullptr;
    server_cache_calibration_claim_identity claim;
    std::array<double, 4> feature = {};
    uint32_t weight_milli = 1000;
    server_cache_calibration_contribution_side side =
        server_cache_calibration_contribution_side::baseline;
    uint64_t now_unix_ms = 0;
    bool authority_admission_allowed = true;
};

struct server_cache_calibration_direct_bound {
    server_cache_calibration_prediction_status status =
        server_cache_calibration_prediction_status::learning;
    double benefit_us = 0.0;
    double radius_us = 0.0;
    double benefit_lower_us = 0.0;
};

bool server_cache_calibration_bound_direct_difference(
    const server_cache_calibration_contribution * contributions,
    size_t count,
    server_cache_calibration_direct_bound & out) noexcept;

bool server_cache_calibration_representation_digest_v1(
    const void * bytes,
    size_t size,
    std::array<uint8_t, 32> & out) noexcept;

struct server_cache_calibration_participant_v1 {
    std::array<uint8_t, 32> adapter_application_digest = {};
    uint16_t media_runtime_class = 0;
    std::array<uint8_t, 32> representation_digest = {};
    uint16_t target_draft_spec_composition = 0;
};

bool server_cache_calibration_participant_digest_v1(
    const server_cache_calibration_participant_v1 * participants,
    size_t count,
    std::array<uint8_t, 32> & out) noexcept;
bool server_cache_calibration_single_participant_digest_v1(
    const std::array<uint8_t, 32> & adapter_application_digest,
    const std::array<uint8_t, 32> & representation_digest,
    uint8_t target_draft_spec_composition,
    std::array<uint8_t, 32> & out) noexcept;

struct server_cache_calibration_effect_action_v1 {
    common_cache_plan_destruction_effect effect =
        common_cache_plan_destruction_effect::none;
    server_cache_destruction_class destruction_class =
        server_cache_destruction_class::_count;
    server_cache_destruction_release_owner release_owner =
        server_cache_destruction_release_owner::none;
};

bool server_cache_calibration_effect_action_digest_v1(
    const server_cache_calibration_effect_action_v1 * actions,
    size_t count,
    std::array<uint8_t, 32> & out) noexcept;
bool server_cache_calibration_apply_shape_digest_v1(
    common_cache_plan_destruction_effect_set effects,
    server_cache_destruction_class destruction_class,
    server_cache_destruction_release_owner release_owner,
    std::array<uint8_t, 32> & out) noexcept;

enum class server_cache_calibration_assignment : uint8_t {
    fit = 0,
    validation,
    validation_unavailable,
    fit_rate_limited,
};

// Exactly one pre-outcome validation assignment per block of eight. The
// varying within-block position prevents periodic workloads from permanently
// donating one feature shape to validation and another only to fitting.
bool server_cache_calibration_validation_assignment(
    uint64_t qualified_execution_ordinal) noexcept;

struct server_cache_calibration_update_context {
    // Process-local one-second admission block and persisted Unix diversity
    // region are intentionally separate currencies.
    uint64_t steady_second = 0;
    uint64_t unix_minute = 0;
    uint64_t unix_ms = 0;
    uint64_t principal_hash = 0; // zero is the trusted-local anonymous principal
    bool force_validation = false;
    bool fit_admission_allowed = true;
    bool principal_admission_allowed = true;
    server_cache_calibration_claim_identity claim;
};

struct server_cache_calibration_update_result {
    server_cache_calibration_assignment assignment =
        server_cache_calibration_assignment::fit_rate_limited;
    bool moments_changed = false;
    bool validation_changed = false;
    bool drifted = false;
    bool tail_latched = false;
    // Validation evidence is always the prediction captured before the
    // response mutation. Fit rows never populate these fields.
    bool validation_prediction_available = false;
    server_cache_calibration_prediction validation_prediction;
};

struct server_cache_calibration_preassignment {
    bool valid = false;
    uint64_t qualified_execution_ordinal = 0;
    uint64_t fit_generation = 0;
    server_cache_calibration_assignment assignment =
        server_cache_calibration_assignment::fit_rate_limited;
    bool validation_prediction_available = false;
    server_cache_calibration_prediction validation_prediction;
};

bool server_cache_calibration_preassign(
    server_cache_observation_instance & instance,
    const std::array<double, 4> & feature,
    const server_cache_calibration_update_context & context,
    server_cache_calibration_preassignment & out) noexcept;
bool server_cache_calibration_complete(
    server_cache_observation_instance & instance,
    const server_cache_observation_record & record,
    const server_cache_calibration_update_context & context,
    const server_cache_calibration_preassignment & assignment,
    server_cache_calibration_update_result & out) noexcept;
bool server_cache_calibration_abandon(
    const server_cache_observation_instance & instance,
    const server_cache_calibration_preassignment & assignment) noexcept;

bool server_cache_calibration_update(
    server_cache_observation_instance & instance,
    const server_cache_observation_record & record,
    const server_cache_calibration_update_context & context,
    server_cache_calibration_update_result & out) noexcept;

// One exact descriptor freezes the 24-MiB ZC4 host budget. The allocation is
// made once for learn/auto; its regions are offsets rather than independently
// growing containers. Placement of the live owners into these regions is
// guarded by the size/alignment checks below.
struct server_cache_calibration_arena_layout {
    static constexpr size_t mib = 1024 * 1024;
    static constexpr size_t profile_slots_begin = 0;
    static constexpr size_t profile_slots_size = 16 * mib;
    static constexpr size_t snapshots_begin = profile_slots_begin + profile_slots_size;
    static constexpr size_t snapshots_size = 2 * mib;
    static constexpr size_t global_tables_begin = snapshots_begin + snapshots_size;
    static constexpr size_t global_tables_size = 2 * mib;
    static constexpr size_t fingerprint_begin = global_tables_begin + global_tables_size;
    static constexpr size_t fingerprint_size = 1 * mib;
    static constexpr size_t codec_scratch_begin = fingerprint_begin + fingerprint_size;
    static constexpr size_t codec_scratch_size = 2 * mib;
    static constexpr size_t reserve_begin = codec_scratch_begin + codec_scratch_size;
    static constexpr size_t reserve_size = 1 * mib;
    static constexpr size_t total_size = reserve_begin + reserve_size;
    static constexpr size_t alignment = 64;
};

static_assert(server_cache_calibration_arena_layout::total_size == 24 * 1024 * 1024);

class server_cache_calibration_arena {
public:
    server_cache_calibration_arena() = default;
    ~server_cache_calibration_arena();
    server_cache_calibration_arena(const server_cache_calibration_arena &) = delete;
    server_cache_calibration_arena & operator=(const server_cache_calibration_arena &) = delete;

    bool allocate() noexcept;
    void reset() noexcept;
    bool ready() const noexcept { return storage_ != nullptr; }
    void * region(size_t offset, size_t size, size_t alignment) noexcept;
    const void * region(size_t offset, size_t size, size_t alignment) const noexcept;
    static bool layout_valid() noexcept;

    template <typename T, typename... Args>
    T * construct(size_t offset, size_t size, Args && ... args) noexcept {
        static_assert(std::is_nothrow_destructible_v<T>);
        void * storage = region(offset, size, alignof(T));
        if (!storage || sizeof(T) > size) return nullptr;
        try {
            return ::new (storage) T(std::forward<Args>(args)...);
        } catch (...) {
            return nullptr;
        }
    }

private:
    std::byte * storage_ = nullptr;
};

template <typename T>
struct server_cache_calibration_arena_deleter {
    void operator()(T * value) const noexcept {
        if (value) value->~T();
    }
};

template <typename T>
using server_cache_calibration_arena_ptr =
    std::unique_ptr<T, server_cache_calibration_arena_deleter<T>>;
