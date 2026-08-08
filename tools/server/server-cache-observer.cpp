#include "server-cache-observer.h"
#include "server-common.h"

#include <algorithm>
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
    if (record.key.operation >= server_cache_observation_operation::_count ||
        record.key.provider >= common_cache_plan_provider::_count ||
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
        case server_cache_observation_reason::multiple_submissions: return "multiple_submissions";
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
        profile_execution_digest == other.profile_execution_digest &&
        participant_execution_digest == other.participant_execution_digest &&
        adapter_application_digest == other.adapter_application_digest &&
        representation_digest == other.representation_digest &&
        effect_action_shape_digest == other.effect_action_shape_digest;
}

void server_cache_observation_store::set_execution_fingerprint(
        const server_cache_execution_fingerprint & value) noexcept {
    execution_fingerprint_ = value;
}

void server_cache_observation_store::apply_execution_fingerprint(
        server_cache_observation_key & key) const noexcept {
    if (!execution_fingerprint_.complete) {
        return;
    }
    key.profile_execution_digest = execution_fingerprint_.execution_root;
    // Participant and representation identity are operation-local facts.
    // ZC3a only supplies the profile root and immutable adapter carrier, so
    // observations remain diagnostic until ZC4 binds those two digests.
    key.identity_complete = false;
    key.identity_exact = execution_fingerprint_.exact;
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
    if (record.terminal ==
            server_cache_observation_terminal::operation_unavailable) {
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.operation_unavailable);
        return false;
    }
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
    uint64_t service_sum = 0;
    const uint64_t response_cap = server_cache_observation_response_cap_us(
        record.key.operation, record.key.size_family);
    if (!add_checked(record.owned_cpu_us, record.backend_service_us,
                     service_sum) ||
        service_sum != record.owned_service_us || response_cap == 0 ||
        record.capped_service_us !=
            std::min(record.owned_service_us, response_cap) ||
        record.tail_exceeded != (record.owned_service_us > response_cap)) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::numeric_overflow;
        store_recent(recent_records_, records_seen_, record);
        increment_saturating(counters_.numeric_fault);
        return false;
    }

    server_cache_observation_instance * instance = nullptr;
    for (auto & candidate : instances_) {
        if (candidate.used && candidate.key == record.key) {
            instance = &candidate;
            break;
        }
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

    // Compute the update before reserving a new instance. A numeric fault
    // cannot consume a capacity slot with a half-initialized key.
    std::array<std::array<double, 4>, 4> next_v = {};
    std::array<double, 4> next_b = {};
    if (instance) {
        next_v = instance->v;
        next_b = instance->b;
    } else {
        for (uint8_t i = 0; i < record.key.feature_dim; ++i) {
            next_v[i][i] = 1.0;
        }
    }
    for (uint8_t i = 0; i < record.key.feature_dim; ++i) {
        for (uint8_t j = 0; j < record.key.feature_dim; ++j) {
            next_v[i][j] += record.feature[i] * record.feature[j];
            if (!std::isfinite(next_v[i][j])) {
                record.terminal = server_cache_observation_terminal::diagnostic;
                record.reason = server_cache_observation_reason::numeric_overflow;
                store_recent(recent_records_, records_seen_, record);
                increment_saturating(counters_.numeric_fault);
                return false;
            }
        }
        next_b[i] += record.feature[i] * double(record.capped_service_us);
        if (!std::isfinite(next_b[i])) {
            record.terminal = server_cache_observation_terminal::diagnostic;
            record.reason = server_cache_observation_reason::numeric_overflow;
            store_recent(recent_records_, records_seen_, record);
            increment_saturating(counters_.numeric_fault);
            return false;
        }
    }
    if (!instance) {
        for (auto & candidate : instances_) {
            if (!candidate.used) {
                candidate.used = true;
                candidate.key = record.key;
                instance = &candidate;
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
    instance->v = next_v;
    instance->b = next_b;
    instance->response_reservoir[
        instance->reservoir_seen % instance->residual_capacity] =
        record.capped_service_us;
    increment_saturating(instance->reservoir_seen);
    increment_saturating(instance->n_success);
    instance->tail_exceeded = instance->tail_exceeded || record.tail_exceeded;
    store_recent(recent_records_, records_seen_, record);
    increment_saturating(counters_.accepted);
    return true;
}

server_cache_observation_key server_cache_observation_cpu_key(
        server_cache_observation_operation operation,
        common_cache_plan_provider provider,
        uint8_t prepare_shape) noexcept {
    server_cache_observation_key key;
    key.operation = operation;
    key.provider = provider;
    key.prepare_shape = prepare_shape;
    key.feature_dim = 2;
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
            { "slot_id", record.slot_id },
            { "submissions", record.submissions },
            { "prompt_slots", record.prompt_slots },
            { "active_slots", record.active_slots },
            { "tokens", record.tokens },
            { "payload_bytes", record.payload_bytes },
            { "owned_cpu_us", record.owned_cpu_us },
            { "backend_service_us", record.backend_service_us },
            { "owned_service_us", record.owned_service_us },
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
        int64_t owned_start_us,
        int64_t owned_end_us,
        bool success,
        server_cache_observation_record * emitted) noexcept {
    if (!store) {
        return false;
    }
    server_cache_observation_record record;
    record.key = key;
    record.slot_id = slot_id;
    record.payload_bytes = operation_extent_bytes;
    record.prompt_slots = 1;
    record.active_slots = 1;
    if (!server_cache_observation_byte_feature(
            operation_extent_bytes, record.key.size_family,
            record.feature) || owned_end_us < owned_start_us) {
        record.terminal = server_cache_observation_terminal::diagnostic;
        record.reason = server_cache_observation_reason::invalid_geometry;
    } else if (!success) {
        record.terminal =
            server_cache_observation_terminal::operation_unavailable;
        record.reason = server_cache_observation_reason::operation_failed;
    } else {
        record.owned_cpu_us = uint64_t(owned_end_us - owned_start_us);
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
        uint64_t initial_owned_cpu_us) noexcept {
    *this = {};
    slot_id_ = slot_id;
    key_ = key;
    owned_cpu_us_ = initial_owned_cpu_us;
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
}

void server_cache_calibration_epoch::pause_owned_cpu(int64_t end_us) noexcept {
    if (!active_ || owned_cpu_segment_start_us_ == 0) {
        return;
    }
    if (end_us < owned_cpu_segment_start_us_ ||
        uint64_t(end_us - owned_cpu_segment_start_us_) >
            std::numeric_limits<uint64_t>::max() - owned_cpu_us_) {
        mark_mixed(server_cache_observation_reason::numeric_overflow);
    } else {
        owned_cpu_us_ += uint64_t(end_us - owned_cpu_segment_start_us_);
    }
    owned_cpu_segment_start_us_ = 0;
}

void server_cache_calibration_epoch::mark_mixed(
        server_cache_observation_reason reason) noexcept {
    if (active_ && reason != server_cache_observation_reason::none &&
        preexisting_reason_ == server_cache_observation_reason::none) {
        preexisting_reason_ = reason;
    }
}

void server_cache_calibration_epoch::note_submission(
        const server_cache_observation_submission & value) noexcept {
    if (!active_) {
        return;
    }
    ++submissions_;
    if (submissions_ == 1) {
        first_submission_us_ = value.submission_us;
        fence_before_ = value.fence_before;
        prompt_slots_ = value.prompt_slots;
        active_slots_ = value.active_slots;
        tokens_ = value.tokens;
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
        key_.size_family = value.size_family;
    } else if (preexisting_reason_ == server_cache_observation_reason::none) {
        preexisting_reason_ =
            server_cache_observation_reason::multiple_submissions;
    }
}

bool server_cache_calibration_epoch::latch_fence(
        server_cache_sync_fence_snapshot fence_after) noexcept {
    if (!active_ || submissions_ == 0 || has_fence_ ||
        fence_after.serial <= fence_before_.serial) {
        return false;
    }
    first_fence_ = fence_after;
    has_fence_ = true;
    return true;
}

server_cache_observation_record server_cache_calibration_epoch::make_record(
        server_cache_sync_fence_snapshot fence_after,
        server_cache_observation_reason forced_reason) const noexcept {
    server_cache_observation_record record;
    record.key = key_;
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
    if (reason == server_cache_observation_reason::none && submissions_ != 1) {
        reason = server_cache_observation_reason::multiple_submissions;
    }
    if (reason == server_cache_observation_reason::none &&
        !key_.identity_complete) {
        reason = server_cache_observation_reason::identity_unavailable;
    }
    if (reason == server_cache_observation_reason::none &&
        (!has_fence_ || fence_after.completed_us < first_submission_us_)) {
        reason = server_cache_observation_reason::invalid_geometry;
    }

    record.owned_cpu_us = owned_cpu_us_;
    if (fence_after.completed_us >= first_submission_us_) {
        record.backend_service_us = uint64_t(
            fence_after.completed_us - first_submission_us_);
    }
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
    if (!active_ || submissions_ == 0 || !has_fence_) {
        return false;
    }
    auto record = make_record(first_fence_,
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
