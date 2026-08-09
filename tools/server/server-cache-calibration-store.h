#pragma once

#include "server-cache-observer.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

bool server_cache_calibration_secure_random(
    std::array<uint8_t, 32> & out) noexcept;

template <typename T, size_t Capacity, typename Size = uint16_t>
struct server_cache_calibration_bounded_array {
    static_assert(Capacity <= size_t(std::numeric_limits<Size>::max()));

    std::array<T, Capacity> values = {};
    Size count = 0;

    constexpr size_t size() const noexcept { return count; }
    constexpr bool empty() const noexcept { return count == 0; }
    constexpr size_t capacity() const noexcept { return Capacity; }
    T * begin() noexcept { return values.data(); }
    const T * begin() const noexcept { return values.data(); }
    T * end() noexcept { return values.data() + count; }
    const T * end() const noexcept { return values.data() + count; }
    T & operator[](size_t index) noexcept { return values[index]; }
    const T & operator[](size_t index) const noexcept { return values[index]; }
    T & front() noexcept { return values[0]; }
    const T & front() const noexcept { return values[0]; }
    T & back() noexcept { return values[count - 1]; }
    const T & back() const noexcept { return values[count - 1]; }

    void clear() noexcept { count = 0; }
    bool push_back(const T & value) noexcept {
        if (count == Capacity) return false;
        values[count++] = value;
        return true;
    }
    bool push_back(T && value) noexcept {
        if (count == Capacity) return false;
        values[count++] = std::move(value);
        return true;
    }
    T * erase(T * position) noexcept {
        if (position < begin() || position >= end()) return end();
        std::move(position + 1, end(), position);
        --count;
        values[count] = {};
        return position;
    }
};

struct server_cache_calibration_instance_snapshot {
    uint32_t slot = 0;
    server_cache_observation_key key;
    uint64_t fit_generation = 0;
    server_cache_calibration_authority_terminal authority_terminal =
        server_cache_calibration_authority_terminal::none;
    uint64_t tail_actual_max_us = 0;
    std::array<std::array<double, 4>, 4> v = {};
    std::array<double, 4> b = {};
    uint64_t n_fit = 0;
    std::array<double, 4> feature_min = {};
    std::array<double, 4> feature_max = {};
    uint64_t qualified_execution_ordinal = 0;
    std::array<double, 6> log_wealth = {};
    uint64_t n_validation = 0;
    server_cache_calibration_bounded_array<uint64_t, 8, uint8_t>
        fit_region_minutes;
    server_cache_calibration_bounded_array<uint64_t, 8, uint8_t>
        validation_region_minutes;
    uint64_t safe_measurable_opportunities = 0;
    uint64_t opportunity_at_last_validation = 0;
    uint64_t last_fit_unix_ms = 0;
    uint64_t last_validation_unix_ms = 0;
    std::array<uint64_t,
               server_cache_observation_instance::residual_capacity>
        response_reservoir = {};
    uint64_t reservoir_seen = 0;
};

struct server_cache_calibration_profile_snapshot {
    uint64_t profile_generation_ordinal = 0;
    uint64_t profile_file_generation = 0;
    uint64_t persisted_prune_recency = 0;
    uint64_t mutation_generation = 0;
    std::array<uint8_t, 32> profile_identity_digest = {};
    bool identity_exact = false;
    server_cache_calibration_bounded_array<
        server_cache_calibration_instance_snapshot,
        server_cache_observation_store::instance_capacity> instances;
    uint64_t profile_last_update_unix_ms = 0;
    // Process-local transport provenance; deliberately absent from the wire.
    // The coordinator sets it only on images loaded by the current boot.
    bool persisted_seed = false;
    // A restored wall-clock anomaly clears validation/age authority without
    // discarding fit moments. This marker is process-local and is never
    // serialized; it only authenticates that exceptional same-writer reset.
    bool clock_authority_reset = false;
};

// Estimator sufficient-state validation has one owner shared by decode,
// same-writer replacement, observer snapshot, and restart restore. Passing a
// predecessor additionally enforces the monotonic locked-writer transition.
bool server_cache_calibration_validate_profile(
    const server_cache_calibration_profile_snapshot & value,
    const server_cache_calibration_profile_snapshot * previous = nullptr) noexcept;

struct server_cache_calibration_profile_reference {
    uint64_t profile_generation_ordinal = 0;
    std::array<uint8_t, 32> profile_identity_digest = {};
    uint64_t profile_file_generation = 0;
    std::array<uint8_t, 32> profile_payload_digest = {};
    uint64_t persisted_prune_recency = 0;
};

struct server_cache_calibration_manifest {
    std::array<uint8_t, 32> store_lineage_id = {};
    uint64_t next_boot_claim_ordinal = 0;
    uint64_t next_profile_generation_ordinal = 0;
    uint64_t next_persisted_prune_epoch = 0;
    uint64_t next_immutable_file_ordinal = 0;
    uint64_t generation = 0;
    server_cache_calibration_bounded_array<
        server_cache_calibration_profile_reference, 16, uint8_t> profiles;
    uint64_t last_update_unix_ms = 0;
};

using server_cache_calibration_profile_set =
    server_cache_calibration_bounded_array<
        server_cache_calibration_profile_snapshot, 16, uint8_t>;

static_assert(std::is_standard_layout_v<server_cache_calibration_profile_snapshot>);
static_assert(sizeof(server_cache_calibration_profile_snapshot) <= 1024 * 1024);

// The arena reserves exactly two immutable single-profile handoff slots: one
// coalesced pending image and one worker-owned in-flight image.
class alignas(64) server_cache_calibration_snapshot_workspace {
public:
    static constexpr size_t slot_size = 1024 * 1024;

    server_cache_calibration_snapshot_workspace() noexcept {
        ::new (pending_storage_) server_cache_calibration_profile_snapshot;
        ::new (inflight_storage_) server_cache_calibration_profile_snapshot;
    }
    ~server_cache_calibration_snapshot_workspace() {
        pending().~server_cache_calibration_profile_snapshot();
        inflight().~server_cache_calibration_profile_snapshot();
    }
    server_cache_calibration_snapshot_workspace(
        const server_cache_calibration_snapshot_workspace &) = delete;
    server_cache_calibration_snapshot_workspace & operator=(
        const server_cache_calibration_snapshot_workspace &) = delete;

    server_cache_calibration_profile_snapshot & pending() noexcept {
        return *reinterpret_cast<server_cache_calibration_profile_snapshot *>(
            pending_storage_);
    }
    server_cache_calibration_profile_snapshot & inflight() noexcept {
        return *reinterpret_cast<server_cache_calibration_profile_snapshot *>(
            inflight_storage_);
    }

private:
    alignas(64) std::byte pending_storage_[slot_size];
    alignas(64) std::byte inflight_storage_[slot_size];
};

static_assert(sizeof(server_cache_calibration_snapshot_workspace) ==
              2 * 1024 * 1024);

enum class server_cache_calibration_load_status : uint8_t {
    ok = 0,
    missing,
    busy,
    capacity,
    corrupt,
    unsupported,
    io_fault,
    ordinal_exhausted,
    _count,
};

enum class server_cache_calibration_writer_health : uint8_t {
    idle = 0,
    starting,
    healthy,
    quarantined,
    stopped,
};

// Internal fault seam used only by the model-free crash/retry tests. Contract
// scans forbid every production caller.
enum class server_cache_calibration_test_fault : uint8_t {
    none = 0,
    profile_write_once,
    manifest_replace_once,
    referencing_manifest_replace_once,
};
void server_cache_calibration_set_test_fault(
    server_cache_calibration_test_fault value) noexcept;
uint64_t server_cache_calibration_test_fault_hits() noexcept;

struct server_cache_calibration_commit_ack {
    std::array<uint8_t, 32> profile_identity_digest = {};
    uint64_t profile_generation_ordinal = 0;
    uint64_t mutation_generation = 0;
    uint64_t profile_file_generation = 0;
    uint64_t root_generation = 0;
    std::array<uint64_t,
               server_cache_observation_store::instance_capacity>
        fit_generations = {};
    std::array<bool,
               server_cache_observation_store::instance_capacity>
        fit_generation_used = {};
};

struct server_cache_calibration_profile_currency {
    std::array<uint8_t, 32> profile_identity_digest = {};
    uint64_t last_enqueued_mutation_generation = 0;
    uint64_t committed_mutation_generation = 0;
    uint64_t committed_profile_generation_ordinal = 0;
    uint64_t committed_profile_file_generation = 0;
    uint64_t committed_root_generation = 0;
    std::array<uint64_t,
               server_cache_observation_store::instance_capacity>
        committed_fit_generations = {};
    std::array<bool,
               server_cache_observation_store::instance_capacity>
        committed_fit_generation_used = {};
    uint64_t profile_last_use_epoch = 0;
    // The persistence cadence belongs to this exact profile.  A model/profile
    // switch must not lend elapsed dirty time to a different currency.
    int64_t dirty_since_us = 0;
    uint8_t profile_state_rank = 0;
    bool committed_ack_seen = false;
    // A persisted seed remains validation-pending across process-local
    // profile switches. Only the future validation owner may clear it.
    server_cache_resume_validation_flags resume_validation_pending = {};
    server_cache_resume_validation_flags
        resume_authority_validation_required = {};
    int64_t resume_started_us = 0;
    bool clock_authority_reset = false;
    bool persisted_origin = false;
};

bool server_cache_calibration_profile_persistence_due(
    server_cache_calibration_profile_currency & currency,
    uint64_t mutation_generation,
    int64_t now_us) noexcept;

// Envelope helpers are public only to the model-free persistence tests. The
// payload parser is bounded before JSON allocation and rejects duplicate keys.
bool server_cache_calibration_encode_manifest(
    const server_cache_calibration_manifest & value,
    std::vector<uint8_t> & out) noexcept;
bool server_cache_calibration_decode_manifest(
    const uint8_t * data,
    size_t size,
    server_cache_calibration_manifest & out) noexcept;
bool server_cache_calibration_encode_profile(
    const std::array<uint8_t, 32> & store_lineage_id,
    const server_cache_calibration_profile_snapshot & value,
    std::vector<uint8_t> & out) noexcept;
bool server_cache_calibration_decode_profile(
    const uint8_t * data,
    size_t size,
    const std::array<uint8_t, 32> & expected_lineage_id,
    server_cache_calibration_profile_snapshot & out) noexcept;
bool server_cache_calibration_encode_profile_with_scratch_for_test(
    const std::array<uint8_t, 32> & store_lineage_id,
    const server_cache_calibration_profile_snapshot & value,
    void * scratch,
    size_t scratch_size,
    std::vector<uint8_t> & out,
    size_t & high_water) noexcept;
bool server_cache_calibration_decode_profile_with_scratch_for_test(
    const uint8_t * data,
    size_t size,
    const std::array<uint8_t, 32> & expected_lineage_id,
    void * scratch,
    size_t scratch_size,
    server_cache_calibration_profile_snapshot & out,
    size_t & high_water) noexcept;

bool server_cache_calibration_snapshot_observer(
    const server_cache_observation_store & store,
    server_cache_calibration_profile_snapshot & out) noexcept;
bool server_cache_calibration_restore_observer(
    const server_cache_calibration_profile_snapshot & value,
    server_cache_observation_store & store) noexcept;

// One process owns one directory lock. Root generations and immutable profile
// ordinals are reserved durably before they can label authority-visible state.
class server_cache_calibration_store {
public:
    server_cache_calibration_store() = default;
    server_cache_calibration_store(void * codec_scratch,
                                   size_t codec_scratch_size) noexcept
        : codec_scratch_(codec_scratch),
          codec_scratch_size_(codec_scratch_size) {}
    ~server_cache_calibration_store();
    server_cache_calibration_store(const server_cache_calibration_store &) = delete;
    server_cache_calibration_store & operator=(const server_cache_calibration_store &) = delete;

    server_cache_calibration_load_status open(
        const std::string & directory,
        const std::string & secure_state_root = {}) noexcept;
    void close() noexcept;
    bool is_open() const noexcept {
        return directory_descriptor_ >= 0 && lock_descriptor_ >= 0 && !failed_;
    }

    const server_cache_calibration_manifest & manifest() const noexcept {
        return manifest_;
    }
    uint64_t boot_claim_ordinal() const noexcept { return boot_claim_ordinal_; }
    size_t codec_high_water() const noexcept { return codec_high_water_; }

    server_cache_calibration_load_status load_profiles(
        server_cache_calibration_profile_set & out) noexcept;
    server_cache_calibration_load_status commit_profile(
        server_cache_calibration_profile_snapshot value) noexcept;

private:
    bool commit_manifest() noexcept;
    bool create_lineage() noexcept;
    bool garbage_collect_profiles() noexcept;
    bool load_referenced_profile(
        const server_cache_calibration_profile_reference & reference,
        server_cache_calibration_profile_snapshot & out) noexcept;

    std::string directory_;
    int directory_descriptor_ = -1;
    int lock_descriptor_ = -1;
    bool failed_ = false;
    uint64_t boot_claim_ordinal_ = 0;
    server_cache_calibration_manifest manifest_;
    void * codec_scratch_ = nullptr;
    size_t codec_scratch_size_ = 0;
    size_t codec_high_water_ = 0;
};

// Scheduler-facing persistence door. start/poll/enqueue never perform file
// I/O. One coalesced immutable snapshot is owned by the worker; a newer
// mutation generation replaces a pending older generation.
class server_cache_calibration_writer {
public:
    server_cache_calibration_writer() = default;
    server_cache_calibration_writer(
        server_cache_calibration_snapshot_workspace * snapshots,
        void * codec_scratch,
        size_t codec_scratch_size) noexcept
        : snapshots_(snapshots), codec_scratch_(codec_scratch),
          codec_scratch_size_(codec_scratch_size) {}
    ~server_cache_calibration_writer();
    server_cache_calibration_writer(const server_cache_calibration_writer &) = delete;
    server_cache_calibration_writer & operator=(const server_cache_calibration_writer &) = delete;

    // Production must pass its exact state root. Model-free tests that own an
    // already-isolated temporary leaf pass an explicit empty root.
    bool start(std::string directory,
               std::string secure_state_root) noexcept;
    void stop() noexcept;
    bool enqueue(const server_cache_calibration_profile_snapshot & value) noexcept;
    bool poll_loaded(
        server_cache_calibration_load_status & status,
        server_cache_calibration_profile_set & profiles) noexcept;
    bool poll_committed(server_cache_calibration_commit_ack & out) noexcept;
    server_cache_calibration_writer_health health() const noexcept {
        return health_.load(std::memory_order_acquire);
    }
    bool boot_claim_ready() const noexcept {
        return boot_claim_ready_.load(std::memory_order_acquire);
    }
    uint64_t boot_claim_ordinal() const noexcept {
        return boot_claim_ordinal_.load(std::memory_order_acquire);
    }

private:
    void run(std::string directory, std::string secure_state_root) noexcept;
    server_cache_calibration_profile_snapshot & pending_profile() noexcept;
    server_cache_calibration_profile_snapshot & inflight_profile() noexcept;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    bool ever_started_ = false;
    bool started_ = false;
    bool stop_ = false;
    bool load_ready_ = false;
    bool load_delivered_ = false;
    server_cache_calibration_load_status load_status_ =
        server_cache_calibration_load_status::missing;
    // The load handoff stays in the coordinator's profile region; pending and
    // commit-inflight images live in the two exact snapshot-region slots. The
    // worker never allocates another production full-profile handoff.
    server_cache_calibration_profile_set loaded_profiles_;
    bool pending_ = false;
    server_cache_calibration_snapshot_workspace * snapshots_ = nullptr;
    std::unique_ptr<server_cache_calibration_snapshot_workspace> owned_snapshots_;
    void * codec_scratch_ = nullptr;
    size_t codec_scratch_size_ = 0;
    std::unique_ptr<std::byte[]> owned_codec_scratch_;
    server_cache_calibration_bounded_array<
        server_cache_calibration_commit_ack, 16, uint8_t> committed_acks_;
    std::array<uint8_t, 32> last_enqueued_identity_ = {};
    uint64_t last_enqueued_mutation_generation_ = 0;
    std::atomic<server_cache_calibration_writer_health> health_ =
        server_cache_calibration_writer_health::idle;
    std::atomic<uint8_t> committed_ack_count_ = 0;
    std::atomic<bool> boot_claim_ready_ = false;
    std::atomic<uint64_t> boot_claim_ordinal_ = 0;
};

// Scheduler-owned coordinator. It atomically joins the asynchronously loaded
// store with the stable execution fingerprint, owns cadence/dirty currency,
// and exposes one model-sleep flush door without doing file I/O.
class server_cache_calibration_coordinator {
public:
    server_cache_calibration_coordinator() = default;
    server_cache_calibration_coordinator(
        server_cache_calibration_snapshot_workspace * snapshots,
        void * codec_scratch,
        size_t codec_scratch_size) noexcept
        : writer_(snapshots, codec_scratch, codec_scratch_size) {}
    bool start(std::string directory,
               std::string secure_state_root) noexcept;
    bool resolve_load(const server_cache_execution_fingerprint & fingerprint,
                      server_cache_observation_store & observer) noexcept;
    void lifecycle(server_cache_observation_store & observer) noexcept;
    void note_profile_use() noexcept;
    void flush_latest(server_cache_observation_store & observer) noexcept;
    void drain_latest_for_shutdown(
        server_cache_observation_store & observer) noexcept;
    void stop() noexcept;
    server_cache_calibration_writer_health health() const noexcept {
        return writer_.health();
    }
    bool resume_pending() const noexcept {
        return std::any_of(resume_pending_.begin(), resume_pending_.end(),
                           [](bool value) { return value; });
    }
    int64_t resume_started_us() const noexcept { return resume_started_us_; }
    bool profile_persisted_origin() const noexcept;
    void complete_resume_validation(
        uint32_t estimator_slot, bool succeeded) noexcept;

private:
    bool enqueue_latest(server_cache_observation_store & observer) noexcept;
    bool enqueue_one_cached_dirty() noexcept;
    bool has_cached_dirty() const noexcept;
    bool consume_acks() noexcept;
    void apply_claim_identity(server_cache_observation_store & observer) noexcept;
    bool cache_snapshot(
        const server_cache_calibration_profile_snapshot & value) noexcept;

    server_cache_calibration_writer writer_;
    server_cache_calibration_profile_set loaded_profiles_;
    bool load_resolved_ = false;
    std::array<uint8_t, 32> profile_identity_digest_ = {};
    server_cache_calibration_bounded_array<
        server_cache_calibration_profile_currency, 16, uint8_t> profile_currencies_;
    uint64_t profile_last_use_epoch_ = 0;
    bool profile_reuse_disabled_ = false;
    server_cache_resume_validation_flags resume_pending_ = {};
    server_cache_resume_validation_flags
        resume_authority_validation_required_ = {};
    int64_t resume_started_us_ = 0;
    bool cached_dirty_ = false;
    int64_t cached_retry_us_ = 0;
    // Scheduler-safe-point staging lives at stable object altitude rather than
    // consuming a ~96 KiB request-thread stack frame or a by-value temporary.
    server_cache_calibration_profile_snapshot snapshot_buffer_;
    server_cache_calibration_profile_snapshot overflow_snapshot_;
    bool overflow_dirty_ = false;
    std::optional<server_cache_execution_fingerprint> deferred_fingerprint_;
};
