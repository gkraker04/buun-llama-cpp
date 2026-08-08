#include "server-cache-observer.h"
#include "server-cache-calibration-model.h"
#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>

namespace {

template <typename T>
bool add_checked(T a, T b, T & out) noexcept {
    if (b > std::numeric_limits<T>::max() - a) {
        out = 0;
        return false;
    }
    out = a + b;
    return true;
}

bool finite_feature(const server_cache_observation_record & record) noexcept {
    if (record.key.feature_dim == 0 || record.key.feature_dim > 4) {
        return false;
    }
    for (uint8_t i = 0; i < record.key.feature_dim; ++i) {
        if (!std::isfinite(record.feature[i]) || record.feature[i] < 0.0) {
            return false;
        }
    }
    return true;
}

bool valid_enum_and_terminal(
        const server_cache_observation_record & record) noexcept {
    if (!server_cache_observation_key_valid(record.key) ||
        record.terminal >= server_cache_observation_terminal::_count ||
        record.reason >= server_cache_observation_reason::_count) {
        return false;
    }
    return (record.terminal == server_cache_observation_terminal::accepted) ==
           (record.reason == server_cache_observation_reason::none);
}

void increment_saturating(uint64_t & value) noexcept {
    if (value != std::numeric_limits<uint64_t>::max()) {
        ++value;
    }
}

void store_recent(
        std::array<server_cache_observation_record,
                   server_cache_observation_store::diagnostic_capacity> & rows,
        uint64_t & seen,
        const server_cache_observation_record & record) noexcept {
    rows[seen % rows.size()] = record;
    increment_saturating(seen);
}

} // namespace

const char * server_cache_observation_operation_name(
        server_cache_observation_operation value) noexcept {
    switch (value) {
        case server_cache_observation_operation::replay: return "replay";
        case server_cache_observation_operation::restore: return "restore";
        case server_cache_observation_operation::durability_prepare: return "durability_prepare";
        case server_cache_observation_operation::destruction_apply: return "destruction_apply";
        case server_cache_observation_operation::_count: break;
    }
    return "invalid";
}

const char * server_cache_observation_terminal_name(
        server_cache_observation_terminal value) noexcept {
    switch (value) {
        case server_cache_observation_terminal::accepted: return "accepted";
        case server_cache_observation_terminal::diagnostic: return "diagnostic";
        case server_cache_observation_terminal::operation_unavailable: return "operation_unavailable";
        case server_cache_observation_terminal::_count: break;
    }
    return "invalid";
}

const char * server_cache_observation_reason_name(
        server_cache_observation_reason value) noexcept {
    switch (value) {
        case server_cache_observation_reason::none: return "none";
        case server_cache_observation_reason::identity_unavailable: return "identity_unavailable";
        case server_cache_observation_reason::mixed_slots: return "mixed_slots";
        case server_cache_observation_reason::no_completion_fence: return "no_completion_fence";
        case server_cache_observation_reason::operation_failed: return "operation_failed";
        case server_cache_observation_reason::warmup_unsettled: return "warmup_unsettled";
        case server_cache_observation_reason::invalid_geometry: return "invalid_geometry";
        case server_cache_observation_reason::instance_capacity: return "instance_capacity";
        case server_cache_observation_reason::numeric_overflow: return "numeric_overflow";
        case server_cache_observation_reason::_count: break;
    }
    return "invalid";
}

bool server_cache_observation_replay_feature(
        uint64_t tokens,
        uint8_t & size_family,
        std::array<double, 4> & feature) noexcept {
    static constexpr uint64_t caps[] = { 512, 2048, 8192, 65536 };
    feature = {};
    if (tokens == 0 || tokens > caps[3]) {
        return false;
    }
    size_t family = 0;
    while (family + 1 < std::size(caps) && tokens > caps[family]) {
        ++family;
    }
    size_family = uint8_t(family);
    const double u = double(tokens) / double(caps[family]);
    static constexpr double knots[] = { 0.0, 0.125, 0.5, 1.0 };
    if (u <= knots[1]) {
        const double t = u / knots[1];
        feature[0] = 1.0 - t;
        feature[1] = t;
    } else if (u <= knots[2]) {
        const double t = (u - knots[1]) / (knots[2] - knots[1]);
        feature[1] = 1.0 - t;
        feature[2] = t;
    } else {
        const double t = (u - knots[2]) / (knots[3] - knots[2]);
        feature[2] = 1.0 - t;
        feature[3] = t;
    }
    return true;
}

bool server_cache_observation_replay_chain_geometry(
        uint64_t tokens,
        uint32_t max_effective_batch,
        uint8_t & size_family,
        uint8_t & batch_bucket,
        std::array<double, 4> & feature) noexcept {
    if (max_effective_batch == 0 ||
        !server_cache_observation_replay_feature(
            tokens, size_family, feature)) {
        return false;
    }
    batch_bucket = server_cache_observation_batch_bucket(max_effective_batch);
    return true;
}

uint8_t server_cache_observation_batch_bucket(uint32_t batch) noexcept {
    if (batch <= 32) return 0;
    if (batch <= 128) return 1;
    if (batch <= 512) return 2;
    return 3;
}

bool server_cache_observation_byte_feature(
        uint64_t bytes,
        uint8_t & size_family,
        std::array<double, 4> & feature) noexcept {
    static constexpr uint64_t MIB = 1024ULL * 1024ULL;
    static constexpr uint64_t caps[] = {
        64 * MIB, 256 * MIB, 1024 * MIB, 4096 * MIB,
    };
    feature = {};
    if (bytes > caps[3]) {
        return false;
    }
    size_t family = 0;
    while (family + 1 < std::size(caps) && bytes > caps[family]) {
        ++family;
    }
    size_family = uint8_t(family);
    const double v = double(bytes) / double(caps[family]);
    feature[0] = 1.0 - v;
    feature[1] = v;
    return true;
}

uint64_t server_cache_observation_response_cap_us(
        server_cache_observation_operation operation,
        uint8_t size_family) noexcept {
    if (size_family >= 4) {
        return 0;
    }
    static constexpr uint64_t replay_caps[] = {
        2000000, 5000000, 20000000, 120000000,
    };
    static constexpr uint64_t byte_caps[] = {
        2000000, 5000000, 20000000, 60000000,
    };
    return operation == server_cache_observation_operation::replay
        ? replay_caps[size_family]
        : byte_caps[size_family];
}

bool server_cache_observation_key::operator==(
        const server_cache_observation_key & other) const noexcept {
    return operation == other.operation && provider == other.provider &&
        restore_kind == other.restore_kind &&
        prepare_shape == other.prepare_shape &&
        contention_bucket == other.contention_bucket &&
        start_bucket == other.start_bucket &&
        batch_bucket == other.batch_bucket &&
        ubatch_bucket == other.ubatch_bucket &&
        size_family == other.size_family &&
        feature_dim == other.feature_dim &&
        model_kind == other.model_kind &&
        operation_extent_bytes == other.operation_extent_bytes &&
        target_draft_spec_composition ==
            other.target_draft_spec_composition &&
        profile_execution_digest == other.profile_execution_digest &&
        participant_execution_digest == other.participant_execution_digest &&
        adapter_application_digest == other.adapter_application_digest &&
        representation_digest == other.representation_digest &&
        effect_action_shape_digest == other.effect_action_shape_digest &&
        identity_exact == other.identity_exact;
}

bool server_cache_observation_key_valid(
        const server_cache_observation_key & key) noexcept {
    const auto nonzero = [](const std::array<uint8_t, 32> & digest) {
        return std::any_of(digest.begin(), digest.end(),
                           [](uint8_t byte) { return byte != 0; });
    };
    if (key.operation >= server_cache_observation_operation::_count ||
        key.provider >= common_cache_plan_provider::_count ||
        key.model_kind >= server_cache_calibration_model_kind::_count ||
        key.restore_kind > 4 || key.prepare_shape > 4 ||
        key.contention_bucket > 1 || key.start_bucket > 3 ||
        key.batch_bucket > 3 || key.ubatch_bucket > 3 ||
        key.size_family > 3 || key.feature_dim == 0 ||
        key.target_draft_spec_composition > 3 ||
        key.feature_dim > 4) return false;
    const bool replay = key.operation == server_cache_observation_operation::replay;
    const bool restore = key.operation == server_cache_observation_operation::restore;
    const bool prepare = key.operation ==
        server_cache_observation_operation::durability_prepare;
    const bool apply = key.operation ==
        server_cache_observation_operation::destruction_apply;
    const bool flat = key.model_kind ==
            server_cache_calibration_model_kind::restore_flat ||
        key.model_kind ==
            server_cache_calibration_model_kind::durability_prepare_flat ||
        key.model_kind ==
            server_cache_calibration_model_kind::destruction_apply_flat;
    if ((replay && (key.model_kind !=
                        server_cache_calibration_model_kind::replay_scaled ||
                    key.feature_dim != 4 || key.restore_kind != 0 ||
                    key.prepare_shape != 0)) ||
        (restore && !((key.model_kind ==
                           server_cache_calibration_model_kind::restore_scaled ||
                       key.model_kind ==
                           server_cache_calibration_model_kind::restore_flat) &&
                      key.restore_kind >= 1 && key.prepare_shape == 0)) ||
        (prepare && !((key.model_kind ==
                           server_cache_calibration_model_kind::durability_prepare_scaled ||
                       key.model_kind ==
                           server_cache_calibration_model_kind::durability_prepare_flat) &&
                      key.restore_kind == 0 && key.prepare_shape >= 1)) ||
        (apply && !((key.model_kind ==
                         server_cache_calibration_model_kind::destruction_apply_scaled ||
                     key.model_kind ==
                         server_cache_calibration_model_kind::destruction_apply_flat) &&
                    key.restore_kind == 0 && key.prepare_shape == 0)) ||
        (!replay && key.feature_dim != (flat ? 1 : 2)) ||
        (!flat && key.operation_extent_bytes != 0)) return false;
    if (!key.identity_complete) return true;
    if (!key.adapter_application_complete ||
        !nonzero(key.profile_execution_digest) ||
        !nonzero(key.participant_execution_digest) ||
        !nonzero(key.adapter_application_digest) ||
        !nonzero(key.representation_digest) ||
        !nonzero(key.effect_action_shape_digest)) return false;
    return true;
}

server_cache_observation_admission_clock
server_cache_observation_capture_admission_clock() noexcept {
    using namespace std::chrono;
    server_cache_observation_admission_clock out;
    // Use the same monotonic currency as resume_fit_barrier_started_us_.
    const int64_t steady = ggml_time_us();
    const auto wall_ms = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    if (steady < 0 || wall_ms < 0) {
        return out;
    }
    out.valid = true;
    out.steady_us = steady;
    out.unix_ms = uint64_t(wall_ms);
    return out;
}

server_cache_observation_cpu_start
server_cache_observation_capture_cpu_start(bool enabled) noexcept {
    server_cache_observation_cpu_start out;
    if (!enabled) {
        return out;
    }
    out.admission_clock = server_cache_observation_capture_admission_clock();
    out.owned_start_us = out.admission_clock.steady_us;
    return out;
}

void server_cache_observation_store::set_execution_fingerprint(
        const server_cache_execution_fingerprint & value) noexcept {
    const bool changed_root =
        execution_fingerprint_.execution_root != value.execution_root;
    if (changed_root) {
        // Model/profile transitions are atomic at the scheduler seam. Never
        // retain instances keyed to the previous execution root.
        instances_ = {};
        principal_cells_ = {};
        generation_started_mutation_ = {};
        mutation_generation_ = 0;
    } else if (execution_fingerprint_.complete != value.complete ||
        execution_fingerprint_.exact != value.exact) {
        increment_saturating(mutation_generation_);
    }
    execution_fingerprint_ = value;
}

bool server_cache_observation_store::restore_persisted_instances(
        const std::array<server_cache_observation_instance,
                         instance_capacity> & instances,
        uint64_t mutation_generation) noexcept {
    if (!execution_fingerprint_.complete || mutation_generation == 0) {
        return false;
    }
    for (const auto & instance : instances) {
        if (!instance.used) continue;
        if (!instance.key.identity_complete ||
            instance.key.profile_execution_digest !=
                execution_fingerprint_.execution_root ||
            instance.key.feature_dim == 0 || instance.key.feature_dim > 4) {
            return false;
        }
    }
    instances_ = instances;
    generation_started_mutation_ = {};
    for (uint32_t slot = 0; slot < instances_.size(); ++slot) {
        if (instances_[slot].used) instances_[slot].estimator_slot = slot;
    }
    mutation_generation_ = mutation_generation;
    return true;
}

void server_cache_observation_store::set_calibration_claim_identity(
        bool available,
        uint64_t boot_claim_ordinal,
        uint64_t profile_generation_ordinal) noexcept {
    claim_identity_available_ = available;
    boot_claim_ordinal_ = available ? boot_claim_ordinal : 0;
    profile_generation_ordinal_ = available ? profile_generation_ordinal : 0;
}

void server_cache_observation_store::set_operation_identity(
        bool complete,
        const std::array<uint8_t, 32> & representation_digest,
        uint8_t target_draft_spec_composition) noexcept {
    operation_identity_complete_ = complete &&
        target_draft_spec_composition <= 3;
    operation_representation_digest_ = operation_identity_complete_
        ? representation_digest : std::array<uint8_t, 32>{};
    target_draft_spec_composition_ = operation_identity_complete_
        ? target_draft_spec_composition : 0;
}

void server_cache_observation_store::set_resume_state(
        const server_cache_resume_validation_flags & validation_pending,
        const server_cache_resume_validation_flags & authority_validation_required,
        int64_t fit_barrier_started_us) noexcept {
    resume_validation_pending_ = validation_pending;
    resume_authority_validation_required_ = authority_validation_required;
    for (uint32_t slot = 0; slot < instances_.size(); ++slot) {
        if (instances_[slot].used) continue;
        resume_validation_pending_[slot] = false;
        resume_authority_validation_required_[slot] = false;
    }
    resume_validation_outcome_count_ = 0;
    resume_fit_barrier_started_us_ = std::max<int64_t>(
        0, fit_barrier_started_us);
}

size_t server_cache_observation_store::take_resume_validation_outcomes(
        server_cache_resume_validation_outcome * out,
        size_t capacity) noexcept {
    if ((!out && capacity != 0) || capacity == 0 ||
        resume_validation_outcome_count_ == 0) return 0;
    const size_t count = std::min<size_t>(
        capacity, resume_validation_outcome_count_);
    std::copy_n(resume_validation_outcomes_.begin(), count, out);
    std::move(resume_validation_outcomes_.begin() + count,
              resume_validation_outcomes_.begin() +
                  resume_validation_outcome_count_,
              resume_validation_outcomes_.begin());
    resume_validation_outcome_count_ -= uint16_t(count);
    return count;
}

void server_cache_observation_store::apply_execution_fingerprint(
        server_cache_observation_key & key) const noexcept {
    if (!execution_fingerprint_.complete) return;
    key.profile_execution_digest = execution_fingerprint_.execution_root;
    key.identity_complete = false;
    key.identity_exact = execution_fingerprint_.exact;
    if (!operation_identity_complete_ || !key.adapter_application_complete ||
        key.prepare_shape == 3) return;
    key.representation_digest = operation_representation_digest_;
    key.target_draft_spec_composition = target_draft_spec_composition_;
    if (!server_cache_calibration_single_participant_digest_v1(
            key.adapter_application_digest, key.representation_digest,
            key.target_draft_spec_composition,
            key.participant_execution_digest)) return;
    const bool effect_ready = key.operation !=
        server_cache_observation_operation::destruction_apply ||
        std::any_of(key.effect_action_shape_digest.begin(),
                    key.effect_action_shape_digest.end(),
                    [](uint8_t value) { return value != 0; });
    key.identity_complete = effect_ready;
}

bool server_cache_observation_store::note_safe_measurable_opportunity(
        const server_cache_observation_key & key,
        uint64_t inventory_ordinal) noexcept {
    if (!key.identity_complete || inventory_ordinal == UINT64_MAX ||
        mutation_generation_ == UINT64_MAX) return false;
    for (auto & instance : instances_) {
        if (!instance.used || !(instance.key == key)) continue;
        if (instance.authority_terminal !=
                server_cache_calibration_authority_terminal::none ||
            instance.safe_measurable_opportunities == UINT64_MAX) return false;
        if (instance.last_opportunity_inventory_ordinal == inventory_ordinal) {
            return true;
        }
        instance.last_opportunity_inventory_ordinal = inventory_ordinal;
        ++instance.safe_measurable_opportunities;
        ++mutation_generation_;
        return true;
    }
    return false;
}

bool server_cache_observation_store::observe(
        server_cache_observation_record & record) noexcept {
    if (!valid_enum_and_terminal(record)) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::numeric_overflow;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }
    if (record.terminal == server_cache_observation_terminal::diagnostic) {
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.diagnostic);
        return false;
    }
    const bool outcome_available = record.terminal ==
        server_cache_observation_terminal::accepted;
    if (!record.key.identity_complete) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::identity_unavailable;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.diagnostic);
        return false;
    }
    if (record.warmup) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::warmup_unsettled;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.diagnostic);
        return false;
    }
    if (!finite_feature(record)) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::invalid_geometry;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }
    if (!record.admission_clock.valid || record.admission_clock.steady_us < 0) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::invalid_geometry;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.diagnostic);
        return false;
    }
    server_cache_observation_instance * instance = nullptr;
    uint32_t instance_slot = 0;
    for (auto & candidate : instances_) {
        if (candidate.used && candidate.key == record.key) {
            instance = &candidate;
            instance_slot = uint32_t(&candidate - instances_.data());
            break;
        }
    }
    if (mutation_generation_ == std::numeric_limits<uint64_t>::max()) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::numeric_overflow;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }
    // A validation-triggered drift terminal freezes that exact claim stream.
    // The next pre-outcome row may allocate one fresh generation in the same
    // bounded physical slot; no moments or ordinals cross the generation.
    bool rotated_generation = false;
    if (instance && instance->authority_terminal ==
            server_cache_calibration_authority_terminal::drifted) {
        if (instance->fit_generation == UINT64_MAX) {
            instance->authority_terminal =
                server_cache_calibration_authority_terminal::ordinal_exhausted;
            record.terminal = server_cache_observation_terminal::diagnostic;
            record.reason = server_cache_observation_reason::numeric_overflow;
            store_recent(recent_records_, records_seen_, record);
            increment_saturating(counters_.numeric_fault);
            increment_saturating(mutation_generation_);
            return false;
        }
        server_cache_observation_instance fresh;
        fresh.used = true;
        fresh.estimator_slot = instance_slot;
        fresh.key = record.key;
        fresh.fit_generation = instance->fit_generation + 1;
        for (uint8_t i = 0; i < record.key.feature_dim; ++i) {
            fresh.v[i][i] = 1.0;
        }
        *instance = fresh;
        generation_started_mutation_[instance_slot] = mutation_generation_ + 1;
        rotated_generation = true;
    }
    if (!instance) {
        const bool capacity_available = std::any_of(
            instances_.begin(), instances_.end(),
            [](const auto & candidate) { return !candidate.used; });
        if (!capacity_available) {
            record.terminal = server_cache_observation_terminal::diagnostic;
            record.reason = server_cache_observation_reason::instance_capacity;
            store_recent(recent_records_, records_seen_, record);
            increment_saturating(counters_.instance_capacity);
            return false;
        }
    }
    bool registered = false;
    if (!instance) {
        for (auto & candidate : instances_) {
            if (!candidate.used) {
                candidate.used = true;
                candidate.key = record.key;
                for (uint8_t i = 0; i < record.key.feature_dim; ++i) {
                    candidate.v[i][i] = 1.0;
                }
                instance = &candidate;
                instance_slot = uint32_t(&candidate - instances_.data());
                candidate.estimator_slot = instance_slot;
                generation_started_mutation_[instance_slot] =
                    mutation_generation_ + 1;
                registered = true;
                break;
            }
        }
    }
    if (!instance) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::instance_capacity;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.instance_capacity);
        return false;
    }
    server_cache_calibration_update_context context;
    context.steady_second = uint64_t(record.admission_clock.steady_us) / 1000000;
    context.unix_ms = record.admission_clock.unix_ms;
    context.unix_minute = context.unix_ms / 60000;
    context.claim.available = claim_identity_available_ &&
        committed_profile_mutation_generation_ >=
            generation_started_mutation_[instance_slot];
    context.claim.boot_claim_ordinal = boot_claim_ordinal_;
    context.claim.profile_generation_ordinal = profile_generation_ordinal_;
    context.claim.estimator_slot = instance_slot;
    context.claim.fit_generation = instance->fit_generation;
    // A different/new operation cannot discharge the restored profile's
    // one-shot validation obligation. Only a row owned by a persisted model
    // instance may consume it; an immature persisted instance still advances
    // as validation_unavailable exactly once.
    context.force_validation = resume_validation_pending_[instance_slot] &&
        !registered && !rotated_generation;
    const bool any_resume_pending = std::any_of(
        resume_validation_pending_.begin(), resume_validation_pending_.end(),
        [](bool value) { return value; });
    context.fit_admission_allowed = !any_resume_pending &&
        (resume_fit_barrier_started_us_ == 0 ||
         (record.admission_clock.steady_us >= resume_fit_barrier_started_us_ &&
          record.admission_clock.steady_us - resume_fit_barrier_started_us_ >=
              60000000));
    server_cache_calibration_principal_cell * principal_cell = nullptr;
    const uint64_t principal_window = context.steady_second / 60;
    const bool ordinary_fit_assignment = !context.force_validation &&
        !server_cache_calibration_validation_assignment(
            instance->qualified_execution_ordinal);
    if (ordinary_fit_assignment) {
        for (auto & cell : principal_cells_) {
            if (cell.used && cell.estimator_slot == instance_slot &&
                cell.principal_hash == context.principal_hash) {
                principal_cell = &cell;
                break;
            }
        }
        if (!principal_cell) {
            // Expired cells are deterministic reusable capacity. Prefer the
            // first free cell, then the first cell from an older minute.
            for (auto & cell : principal_cells_) {
                if (!cell.used || cell.window != principal_window) {
                    cell = {};
                    cell.used = true;
                    cell.estimator_slot = instance_slot;
                    cell.principal_hash = context.principal_hash;
                    cell.window = principal_window;
                    principal_cell = &cell;
                    break;
                }
            }
        }
        if (principal_cell && principal_cell->window != principal_window) {
            principal_cell->window = principal_window;
            principal_cell->fit_rows = 0;
        }
        context.principal_admission_allowed = principal_cell &&
            principal_cell->fit_rows < 4;
    }
    server_cache_calibration_preassignment assignment;
    if (!server_cache_calibration_preassign(
            *instance, record.feature, context, assignment)) {
        if (registered && instance->authority_terminal ==
                server_cache_calibration_authority_terminal::none) {
            *instance = {};
        }
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::numeric_overflow;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }
    if (assignment.assignment == server_cache_calibration_assignment::fit &&
        principal_cell && principal_cell->fit_rows < UINT8_MAX) {
        ++principal_cell->fit_rows;
    }

    server_cache_calibration_update_result result;
    result.assignment = assignment.assignment;
    result.validation_prediction_available =
        assignment.validation_prediction_available;
    result.validation_prediction = assignment.validation_prediction;
    if (outcome_available) {
        uint64_t service_sum = 0;
        const uint64_t response_cap = server_cache_observation_response_cap_us(
            record.key.operation, record.key.size_family);
        const bool response_valid =
            add_checked(record.owned_cpu_us, record.backend_service_us,
                        service_sum) &&
            service_sum == record.owned_service_us && response_cap != 0 &&
            record.capped_service_us ==
                std::min(record.owned_service_us, response_cap) &&
            record.tail_exceeded == (record.owned_service_us > response_cap);
        if (!response_valid || !server_cache_calibration_complete(
                *instance, record, context, assignment, result)) {
            instance->authority_terminal =
                server_cache_calibration_authority_terminal::numeric_fault;
            record.terminal = server_cache_observation_terminal::diagnostic;
            record.reason = server_cache_observation_reason::numeric_overflow;
            increment_saturating(mutation_generation_);
            store_recent(recent_records_, records_seen_, record);
            increment_saturating(counters_.numeric_fault);
            return false;
        }
    } else if (!server_cache_calibration_abandon(*instance, assignment)) {
        instance->authority_terminal =
            server_cache_calibration_authority_terminal::numeric_fault;
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::numeric_overflow;
        increment_saturating(mutation_generation_);
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }
    switch (result.assignment) {
        case server_cache_calibration_assignment::fit:
            if (outcome_available) increment_saturating(counters_.fit_rows);
            break;
        case server_cache_calibration_assignment::validation:
            if (outcome_available) increment_saturating(counters_.validation_rows);
            break;
        case server_cache_calibration_assignment::validation_unavailable:
            increment_saturating(counters_.validation_unavailable);
            break;
        case server_cache_calibration_assignment::fit_rate_limited:
            increment_saturating(counters_.fit_rate_limited);
            break;
    }
    if (result.drifted) increment_saturating(counters_.drifted);
    auto queue_resume_outcome = [&](server_cache_resume_validation_outcome_kind kind) {
        for (uint16_t i = 0; i < resume_validation_outcome_count_; ++i) {
            if (resume_validation_outcomes_[i].estimator_slot == instance_slot) {
                if (kind == server_cache_resume_validation_outcome_kind::succeeded) {
                    resume_validation_outcomes_[i].kind = kind;
                }
                return;
            }
        }
        if (resume_validation_outcome_count_ < resume_validation_outcomes_.size()) {
            resume_validation_outcomes_[resume_validation_outcome_count_++] =
                { instance_slot, kind };
        }
    };
    if (context.force_validation) {
        resume_validation_pending_[instance_slot] = false;
        if (outcome_available && result.assignment ==
                server_cache_calibration_assignment::validation) {
            resume_authority_validation_required_[instance_slot] = false;
            queue_resume_outcome(
                server_cache_resume_validation_outcome_kind::succeeded);
        } else {
            queue_resume_outcome(
                server_cache_resume_validation_outcome_kind::unavailable);
        }
    } else if (outcome_available &&
               resume_authority_validation_required_[instance_slot] &&
               result.assignment == server_cache_calibration_assignment::validation) {
        resume_authority_validation_required_[instance_slot] = false;
        queue_resume_outcome(server_cache_resume_validation_outcome_kind::succeeded);
    }
    record.calibration_instance_slot = instance_slot;
    record.calibration_assignment = uint8_t(result.assignment) + 1;
    record.calibration_n_fit = instance->n_success;
    record.calibration_n_validation = instance->n_validation;
    record.calibration_claim_available = context.claim.available;
    record.calibration_boot_claim_ordinal = context.claim.boot_claim_ordinal;
    record.calibration_profile_generation =
        context.claim.profile_generation_ordinal;
    record.calibration_fit_generation = instance->fit_generation;
    const auto profile_state = server_cache_calibration_state(
        *instance, context.claim, record.feature, context.unix_ms,
        nullptr, !resume_authority_validation_required_[instance_slot]);
    record.calibration_profile_state = uint8_t(profile_state) + 1;
    if (result.validation_prediction_available) {
        record.calibration_prediction_available = true;
        record.calibration_prediction_us = result.validation_prediction.point_us;
        record.calibration_radius_us = result.validation_prediction.radius_us;
    }
    increment_saturating(mutation_generation_);
    store_recent(recent_records_, records_seen_, record);
    if (outcome_available) {
        increment_saturating(counters_.accepted);
        return true;
    }
    increment_saturating(counters_.operation_unavailable);
    return false;
}

server_cache_observation_key server_cache_observation_cpu_key(
        server_cache_observation_operation operation,
        common_cache_plan_provider provider,
        uint8_t prepare_shape) noexcept {
    server_cache_observation_key key;
    key.operation = operation;
    key.provider = provider;
    key.prepare_shape = prepare_shape;
    if (operation == server_cache_observation_operation::restore) {
        key.restore_kind = provider == common_cache_plan_provider::host_cache_entry
            ? 1 : provider ==
                    common_cache_plan_provider::live_context_checkpoint
                ? 3 : 0;
    }
    key.feature_dim = 2;
    key.model_kind = operation ==
            server_cache_observation_operation::restore
        ? server_cache_calibration_model_kind::restore_scaled
        : operation == server_cache_observation_operation::durability_prepare
            ? server_cache_calibration_model_kind::durability_prepare_scaled
            : operation == server_cache_observation_operation::destruction_apply
                ? server_cache_calibration_model_kind::destruction_apply_scaled
                : server_cache_calibration_model_kind::replay_scaled;
    (void) server_cache_calibration_effect_action_digest_v1(
        nullptr, 0, key.effect_action_shape_digest);
    key.identity_complete = false;
    return key;
}

void server_cache_emit_observation_noexcept(
        const server_cache_observation_record & record,
        bool enabled) noexcept {
    if (!enabled) {
        return;
    }
    try {
        const json payload = {
            { "operation", server_cache_observation_operation_name(record.key.operation) },
            { "terminal", server_cache_observation_terminal_name(record.terminal) },
            { "reason", server_cache_observation_reason_name(record.reason) },
            { "provider", common_cache_plan_provider_name(record.key.provider) },
            { "calibration_model_kind", uint8_t(record.key.model_kind) },
            { "size_family", record.key.size_family },
            { "contention_bucket", record.key.contention_bucket },
            { "start_bucket", record.key.start_bucket },
            { "batch_bucket", record.key.batch_bucket },
            { "ubatch_bucket", record.key.ubatch_bucket },
            { "slot_id", record.slot_id },
            { "submissions", record.submissions },
            { "prompt_slots", record.prompt_slots },
            { "active_slots", record.active_slots },
            { "tokens", record.tokens },
            { "payload_bytes", record.payload_bytes },
            { "owned_cpu_us", record.owned_cpu_us },
            { "backend_service_us", record.backend_service_us },
            { "owned_service_us", record.owned_service_us },
            { "capped_service_us", record.capped_service_us },
            { "tail_exceeded", record.tail_exceeded },
            { "calibration_instance_slot", record.calibration_instance_slot },
            { "calibration_assignment", record.calibration_assignment },
            { "calibration_n_fit", record.calibration_n_fit },
            { "calibration_n_validation", record.calibration_n_validation },
            { "calibration_profile_state",
              record.calibration_profile_state == 0
                  ? "unavailable"
                  : server_cache_calibration_instance_state_name(
                        server_cache_calibration_instance_state(
                            record.calibration_profile_state - 1)) },
            { "calibration_claim_available", record.calibration_claim_available },
            { "calibration_boot_claim_ordinal", record.calibration_boot_claim_ordinal },
            { "calibration_profile_generation", record.calibration_profile_generation },
            { "calibration_fit_generation", record.calibration_fit_generation },
            { "calibration_prediction_available", record.calibration_prediction_available },
            { "calibration_prediction_us", record.calibration_prediction_us },
            { "calibration_radius_us", record.calibration_radius_us },
            { "fence_serial_before", record.fence_serial_before },
            { "fence_serial_after", record.fence_serial_after },
        };
        SRV_INF("CACHE_OPTIMIZER_OBSERVATION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug serialization never changes an operation terminal.
    }
}

bool server_cache_observe_cpu_operation(
        server_cache_observation_store * store,
        server_cache_observation_key key,
        int32_t slot_id,
        uint64_t operation_extent_bytes,
        server_cache_observation_cpu_start owned_start,
        int64_t owned_end_us,
        bool success,
        bool isolated,
        server_cache_observation_record * emitted) noexcept {
    if (!store) {
        return false;
    }
    server_cache_observation_record record;
    record.key = key;
    record.admission_clock = owned_start.admission_clock;
    record.slot_id = slot_id;
    record.payload_bytes = operation_extent_bytes;
    record.prompt_slots = 1;
    record.active_slots = isolated ? 1 : 2;
    if (!server_cache_observation_byte_feature(
            operation_extent_bytes, record.key.size_family,
            record.feature) || owned_end_us < owned_start.owned_start_us) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::invalid_geometry;
    } else if (!isolated) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::mixed_slots;
    } else if (!success) {
        record.terminal =
            server_cache_observation_terminal::operation_unavailable;
        record.reason = server_cache_observation_reason::operation_failed;
    } else {
        record.owned_cpu_us = uint64_t(
            owned_end_us - owned_start.owned_start_us);
        record.owned_service_us = record.owned_cpu_us;
        const uint64_t response_cap = server_cache_observation_response_cap_us(
            record.key.operation, record.key.size_family);
        record.capped_service_us = std::min(
            record.owned_service_us, response_cap);
        record.tail_exceeded = record.owned_service_us > response_cap;
        record.terminal = key.identity_complete
            ? server_cache_observation_terminal::accepted
            : server_cache_observation_terminal::diagnostic;
        record.reason = key.identity_complete
            ? server_cache_observation_reason::none
            : server_cache_observation_reason::identity_unavailable;
    }
    const bool accepted = store->observe(record);
    if (emitted) {
        *emitted = record;
    }
    return accepted;
}

void server_cache_calibration_epoch::arm(
        int32_t slot_id,
        server_cache_observation_key key,
        uint64_t initial_owned_cpu_us,
        server_cache_observation_admission_clock admission_clock) noexcept {
    *this = {};
    slot_id_ = slot_id;
    key_ = key;
    owned_cpu_us_ = initial_owned_cpu_us;
    // Never upgrade a failed first-operation capture at this later seam. In
    // particular, host/checkpoint restore arms only after provider work has
    // returned; retrying the clock here would make outcome latency control its
    // own estimator assignment. A missing tuple remains fail-closed in
    // observe().
    admission_clock_ = admission_clock;
    active_ = slot_id >= 0;
}

void server_cache_calibration_epoch::bind_provider(
        server_cache_observation_key key,
        uint64_t payload_bytes,
        uint64_t initial_owned_cpu_us) noexcept {
    if (!active_) {
        return;
    }
    key_ = key;
    payload_bytes_ = payload_bytes;
    owned_cpu_us_ = initial_owned_cpu_us;
}

void server_cache_calibration_epoch::begin_owned_cpu(int64_t start_us) noexcept {
    if (!active_ || start_us <= 0 || owned_cpu_segment_start_us_ != 0) {
        return;
    }
    owned_cpu_segment_start_us_ = start_us;
    // CPU preparation before an unresolved async fence lies inside that
    // fence-chain's critical path. Track it for attribution, but never add it
    // again to the non-overlapping CPU component.
    owned_cpu_segment_overlapped_ = submission_pending_;
}

void server_cache_calibration_epoch::pause_owned_cpu(int64_t end_us) noexcept {
    if (!active_ || owned_cpu_segment_start_us_ == 0) {
        return;
    }
    if (end_us < owned_cpu_segment_start_us_) {
        mark_mixed(server_cache_observation_reason::numeric_overflow);
    } else if (!owned_cpu_segment_overlapped_ &&
               uint64_t(end_us - owned_cpu_segment_start_us_) >
                   std::numeric_limits<uint64_t>::max() - owned_cpu_us_) {
        mark_mixed(server_cache_observation_reason::numeric_overflow);
    } else if (!owned_cpu_segment_overlapped_) {
        owned_cpu_us_ += uint64_t(end_us - owned_cpu_segment_start_us_);
    }
    owned_cpu_segment_start_us_ = 0;
    owned_cpu_segment_overlapped_ = false;
}

void server_cache_calibration_epoch::mark_mixed(
        server_cache_observation_reason reason) noexcept {
    if (active_ && reason != server_cache_observation_reason::none &&
        preexisting_reason_ == server_cache_observation_reason::none) {
        preexisting_reason_ = reason;
    }
}

bool server_cache_observation_same_chain_geometry(
        const server_cache_observation_submission & first,
        const server_cache_observation_submission & next) noexcept {
    return first.prompt_slots == next.prompt_slots &&
           first.active_slots == next.active_slots &&
           first.payload_bytes == next.payload_bytes &&
           first.effective_ubatch == next.effective_ubatch &&
           first.target_participates == next.target_participates &&
           first.draft_participates == next.draft_participates &&
           first.speculative_participates == next.speculative_participates &&
           first.contention_bucket == next.contention_bucket &&
           first.ubatch_bucket == next.ubatch_bucket;
}

void server_cache_calibration_epoch::note_submission(
        const server_cache_observation_submission & value) noexcept {
    if (!active_) {
        return;
    }
    // A required synchronization may have completed the previous submission
    // group after the outer close hook. Consume that passive timestamp before
    // opening a new group so idle CPU/wall time cannot enter its backend span.
    if (submission_pending_) {
        if (value.fence_before.serial > pending_fence_before_.serial) {
            (void) latch_fence(value.fence_before);
        } else if (value.fence_before.serial < pending_fence_before_.serial) {
            mark_mixed(server_cache_observation_reason::invalid_geometry);
        }
    }
    if (submissions_ == std::numeric_limits<uint32_t>::max() ||
        value.tokens > std::numeric_limits<uint32_t>::max() - tokens_) {
        mark_mixed(server_cache_observation_reason::numeric_overflow);
        return;
    }
    ++submissions_;
    if (submissions_ == 1) {
        fence_before_ = value.fence_before;
        prompt_slots_ = value.prompt_slots;
        active_slots_ = value.active_slots;
        payload_bytes_ = value.payload_bytes;
        start_position_ = value.start_position;
        effective_batch_ = value.effective_batch;
        effective_ubatch_ = value.effective_ubatch;
        target_participates_ = value.target_participates;
        draft_participates_ = value.draft_participates;
        speculative_participates_ = value.speculative_participates;
        warmup_ = value.warmup;
        feature_ = value.feature;
        key_.contention_bucket = value.contention_bucket;
        key_.start_bucket = value.start_bucket;
        key_.batch_bucket = value.batch_bucket;
        key_.ubatch_bucket = value.ubatch_bucket;
    } else {
        server_cache_observation_submission first;
        first.prompt_slots = prompt_slots_;
        first.active_slots = active_slots_;
        first.payload_bytes = payload_bytes_;
        first.effective_ubatch = effective_ubatch_;
        first.target_participates = target_participates_;
        first.draft_participates = draft_participates_;
        first.speculative_participates = speculative_participates_;
        first.contention_bucket = key_.contention_bucket;
        first.ubatch_bucket = key_.ubatch_bucket;
        if (!server_cache_observation_same_chain_geometry(first, value)) {
            mark_mixed(server_cache_observation_reason::mixed_slots);
        }
        effective_batch_ = std::max(effective_batch_, value.effective_batch);
        key_.batch_bucket = std::max(key_.batch_bucket, value.batch_bucket);
        warmup_ = warmup_ || value.warmup;
    }
    tokens_ += value.tokens;
    if (key_.operation == server_cache_observation_operation::replay) {
        if (!server_cache_observation_replay_chain_geometry(
                tokens_, effective_batch_, key_.size_family,
                key_.batch_bucket, feature_)) {
            mark_mixed(server_cache_observation_reason::invalid_geometry);
        }
    } else {
        key_.size_family = value.size_family;
        feature_ = value.feature;
    }
    if (!submission_pending_) {
        pending_submission_us_ = value.submission_us;
        pending_fence_before_ = value.fence_before;
        submission_pending_ = true;
    }
}

bool server_cache_calibration_epoch::latch_fence(
        server_cache_sync_fence_snapshot fence_after) noexcept {
    if (!active_ || !submission_pending_ ||
        fence_after.serial <= pending_fence_before_.serial) {
        return false;
    }
    if (pending_submission_us_ <= 0 ||
        fence_after.completed_us < pending_submission_us_ ||
        uint64_t(fence_after.completed_us - pending_submission_us_) >
            std::numeric_limits<uint64_t>::max() - backend_service_us_) {
        mark_mixed(server_cache_observation_reason::invalid_geometry);
    } else {
        backend_service_us_ += uint64_t(
            fence_after.completed_us - pending_submission_us_);
    }
    last_fence_ = fence_after;
    submission_pending_ = false;
    pending_submission_us_ = 0;
    // One already-required fence completes every submission queued since the
    // previous fence. Their continuous critical path is charged exactly once;
    // a later independently fenced group starts a new span.
    fenced_submissions_ = submissions_;
    return true;
}

server_cache_observation_record server_cache_calibration_epoch::make_record(
        server_cache_sync_fence_snapshot fence_after,
        server_cache_observation_reason forced_reason) const noexcept {
    server_cache_observation_record record;
    record.key = key_;
    record.admission_clock = admission_clock_;
    record.feature = feature_;
    record.slot_id = slot_id_;
    record.prompt_slots = prompt_slots_;
    record.active_slots = active_slots_;
    record.submissions = submissions_;
    record.tokens = tokens_;
    record.payload_bytes = payload_bytes_;
    record.start_position = start_position_;
    record.effective_batch = effective_batch_;
    record.effective_ubatch = effective_ubatch_;
    record.target_participates = target_participates_;
    record.draft_participates = draft_participates_;
    record.speculative_participates = speculative_participates_;
    record.warmup = warmup_;
    record.fence_serial_before = fence_before_.serial;
    record.fence_serial_after = fence_after.serial;

    server_cache_observation_reason reason = forced_reason !=
            server_cache_observation_reason::none
        ? forced_reason
        : preexisting_reason_;
    if (reason == server_cache_observation_reason::none &&
        (prompt_slots_ != 1 || active_slots_ != 1)) {
        reason = server_cache_observation_reason::mixed_slots;
    }
    if (reason == server_cache_observation_reason::none &&
        !key_.identity_complete) {
        reason = server_cache_observation_reason::identity_unavailable;
    }
    if (reason == server_cache_observation_reason::none &&
        (submission_pending_ || fenced_submissions_ != submissions_ ||
         fence_after.serial < fence_before_.serial)) {
        reason = server_cache_observation_reason::invalid_geometry;
    }

    record.owned_cpu_us = owned_cpu_us_;
    record.backend_service_us = backend_service_us_;
    if (!add_checked(record.owned_cpu_us, record.backend_service_us,
                     record.owned_service_us)) {
        reason = server_cache_observation_reason::numeric_overflow;
    }
    const uint64_t response_cap = server_cache_observation_response_cap_us(
        record.key.operation, record.key.size_family);
    record.capped_service_us = std::min(
        record.owned_service_us, response_cap);
    record.tail_exceeded = response_cap != 0 &&
        record.owned_service_us > response_cap;
    record.reason = reason;
    record.terminal = reason == server_cache_observation_reason::none
        ? server_cache_observation_terminal::accepted
        : server_cache_observation_terminal::diagnostic;
    return record;
}

bool server_cache_calibration_epoch::finish(
        server_cache_observation_store & store,
        server_cache_observation_record * emitted) noexcept {
    if (!active_ || !terminal_ready_ || submissions_ == 0 || submission_pending_ ||
        fenced_submissions_ != submissions_) {
        return false;
    }
    auto record = make_record(last_fence_,
                              server_cache_observation_reason::none);
    (void) store.observe(record);
    if (emitted) {
        *emitted = record;
    }
    reset();
    return true;
}

bool server_cache_calibration_epoch::abandon(
        server_cache_observation_reason reason,
        server_cache_observation_store & store,
        server_cache_observation_record * emitted) noexcept {
    if (!active_) {
        return false;
    }
    auto record = make_record({}, reason);
    record.terminal = server_cache_observation_terminal::operation_unavailable;
    (void) store.observe(record);
    if (emitted) {
        *emitted = record;
    }
    reset();
    return true;
}
