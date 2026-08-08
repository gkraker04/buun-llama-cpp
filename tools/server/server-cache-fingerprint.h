#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

struct common_params;
struct common_cache_plan_vbr_regime;
struct common_adapter_lora_info;

// ZC3a's stable execution identity is deliberately independent of the
// process-random cache lineage/frontier identities.  The codec below is the
// single production owner of the frozen v1 byte format in the design.

enum class server_cache_fingerprint_artifact_role : uint16_t {
    target  = 1,
    draft   = 2,
    mmproj  = 3,
    adapter = 4,
};

enum class server_cache_fingerprint_field_type : uint8_t {
    bool_u8  = 1,
    enum_u16 = 2,
    u32      = 3,
    u64      = 4,
    binary64 = 5,
    utf8     = 6,
    bytes    = 7,
    digest32 = 8,
};

struct server_cache_fingerprint_artifact {
    server_cache_fingerprint_artifact_role role =
        server_cache_fingerprint_artifact_role::target;
    uint32_t ordinal = 0;
    uint64_t byte_length = 0;
    std::array<uint8_t, 32> content_sha256 = {};
    // Exact is an integrity property, not a statement that hashing finished.
    // Mutable descriptor hashing produces a stable shadow seed with false.
    bool exact = false;
};

struct server_cache_fingerprint_field {
    uint16_t id = 0;
    server_cache_fingerprint_field_type type =
        server_cache_fingerprint_field_type::bytes;
    std::vector<uint8_t> payload;
    bool exact = true;
};

struct server_cache_execution_fingerprint {
    std::array<uint8_t, 32> artifact_root = {};
    std::array<uint8_t, 32> config_root = {};
    std::array<uint8_t, 32> execution_root = {};
    bool complete = false;
    bool exact = false;
};

// Canonical little-endian payload constructors. Floating input is rejected
// unless finite; -0 is normalized to +0 before its bits are emitted.
server_cache_fingerprint_field server_cache_fingerprint_bool(
    uint16_t id, bool value);
server_cache_fingerprint_field server_cache_fingerprint_enum(
    uint16_t id, uint16_t value);
server_cache_fingerprint_field server_cache_fingerprint_u32(
    uint16_t id, uint32_t value);
server_cache_fingerprint_field server_cache_fingerprint_u64(
    uint16_t id, uint64_t value);
bool server_cache_fingerprint_binary64(
    uint16_t id, double value, server_cache_fingerprint_field & out) noexcept;
bool server_cache_fingerprint_utf8(
    uint16_t id, const char * data, size_t size,
    server_cache_fingerprint_field & out) noexcept;
server_cache_fingerprint_field server_cache_fingerprint_bytes(
    uint16_t id, const void * data, size_t size);
server_cache_fingerprint_field server_cache_fingerprint_digest(
    uint16_t id, const std::array<uint8_t, 32> & value);

// Requires the exact v1 field set 1..32 with the frozen adjacent type table.
// Inputs must already be in canonical order; duplicates, malformed payloads,
// unknown codes, overflow, or a noncanonical artifact tuple fail closed.
bool server_cache_execution_fingerprint_v1(
    std::vector<server_cache_fingerprint_artifact> artifacts,
    std::vector<server_cache_fingerprint_field> fields,
    server_cache_execution_fingerprint & out) noexcept;

// Production lowering of the complete frozen 1..32 configuration table.
// Missing driver/device UUID/topology facts are encoded deterministically but
// mark their fields non-exact; they still produce a stable shadow seed and can
// never confer restart authority.
bool server_cache_fingerprint_fields_v1(
    const common_params & params,
    const common_cache_plan_vbr_regime & vbr,
    uint32_t effective_n_gpu_layers,
    uint16_t pipeline_mode,
    uint16_t allocator_vmm_regime,
    std::vector<server_cache_fingerprint_field> & out) noexcept;

struct server_cache_adapter_application_entry {
    uint32_t ordinal = 0;
    double scale = 0.0;
    uint16_t application_mode = 0; // 0 ordinary, 1 aLoRA
};

bool server_cache_adapter_application_entries_digest_v1(
    const std::vector<server_cache_adapter_application_entry> & entries,
    std::array<uint8_t, 32> & out) noexcept;

bool server_cache_adapter_application_digest_v1(
    const std::vector<common_adapter_lora_info> & adapters,
    std::array<uint8_t, 32> & out) noexcept;

// Loader-owned descriptor handed to the background verifier. The worker owns
// and closes descriptor. `integrity_exact` is true only when the descriptor's
// backing object was already immutable (fs-verity or sealed object).
struct server_cache_fingerprint_descriptor {
    server_cache_fingerprint_artifact_role role =
        server_cache_fingerprint_artifact_role::target;
    uint32_t ordinal = 0;
    int descriptor = -1;
    uint64_t byte_length = 0;
    bool integrity_exact = false;
};

void server_cache_fingerprint_descriptors_close(
    std::vector<server_cache_fingerprint_descriptor> & descriptors) noexcept;

// One cancelable, bounded worker. It never touches the scheduler or planner.
// The scheduler may advertise demand; hashing then pauses between bounded
// reads. The complete object, including its read buffer, fits the ZC4 1-MiB
// fingerprint arena region.
class server_cache_fingerprint_worker {
public:
    static constexpr size_t descriptor_capacity = 256;
    static constexpr size_t fixed_artifact_capacity = 256;

    server_cache_fingerprint_worker() = default;
    ~server_cache_fingerprint_worker();
    server_cache_fingerprint_worker(const server_cache_fingerprint_worker &) = delete;
    server_cache_fingerprint_worker & operator=(const server_cache_fingerprint_worker &) = delete;

    bool start(
        std::vector<server_cache_fingerprint_descriptor> descriptors,
        std::vector<server_cache_fingerprint_field> fields,
        std::vector<server_cache_fingerprint_artifact> fixed_artifacts = {}) noexcept;
    // Production admission writes directly into the arena-owned worker. The
    // configuration root is streamed through the frozen v1 codec, so neither
    // input tables nor canonical byte vectors escape the 1-MiB region.
    bool configure(
        const common_params & params,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime) noexcept;
    bool configure(
        const common_params & params,
        const float * effective_tensor_split,
        size_t effective_tensor_split_count,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime) noexcept;
    bool add_descriptor(server_cache_fingerprint_descriptor value) noexcept;
    bool add_fixed_artifact(server_cache_fingerprint_artifact value) noexcept;
    bool launch() noexcept;
    void set_scheduler_demand(bool value) noexcept {
        scheduler_demand_.store(value, std::memory_order_relaxed);
    }
    bool poll(server_cache_execution_fingerprint & out) noexcept;
    void stop() noexcept;
    bool configured_exact() const noexcept { return config_exact_; }
    uint32_t configured_inexact_fields() const noexcept {
        return config_inexact_fields_;
    }

private:
    static constexpr size_t arena_region_bytes = 1024 * 1024;
    static constexpr size_t arena_state_reserve_bytes = 64 * 1024;
    static constexpr size_t hash_buffer_bytes =
        arena_region_bytes - arena_state_reserve_bytes;

    void run() noexcept;

    std::atomic<bool> cancel_{false};
    std::atomic<bool> scheduler_demand_{false};
    std::thread thread_;
    std::mutex mutex_;
    server_cache_execution_fingerprint result_;
    bool ready_ = false;
    bool delivered_ = false;
    bool started_ = false;
    std::array<server_cache_fingerprint_descriptor, descriptor_capacity>
        descriptors_ = {};
    std::array<std::array<uint8_t, 32>, descriptor_capacity>
        descriptor_digests_ = {};
    std::array<server_cache_fingerprint_artifact, fixed_artifact_capacity>
        fixed_artifacts_ = {};
    size_t descriptor_count_ = 0;
    size_t fixed_artifact_count_ = 0;
    std::array<uint8_t, 32> config_root_ = {};
    bool config_ready_ = false;
    bool config_exact_ = false;
    uint32_t config_inexact_fields_ = 0;
    alignas(64) std::array<uint8_t, hash_buffer_bytes> hash_buffer_;
};
