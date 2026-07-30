#pragma once

#include "llama-cache-authority.h"
#include "llama-vbr-artifact.h"

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// F3.1 bounded streaming substrate. These types are internal to libllama:
// no live KV/cache or server policy enters this unit.
static constexpr uint64_t VBR_CAPTURE_PINNED_RING_MAX_BYTES =
    256ull*1024*1024;

enum class vbr_capture_stream_status : uint8_t {
    ok = 0,
    invalid_argument,
    ring_unavailable,
    transfer_failed,
    short_read,
    duplicate_segment,
    missing_segment,
    late_segment,
    hash_mismatch,
    format_rejected,
    accounting_unavailable,
    accounting_refused,
    stage_failed,
    commit_failed,
    publication_failed,
    internal_error,
    _count,
};

const char * vbr_capture_stream_status_name(
    vbr_capture_stream_status status) noexcept;

struct artifact_segment {
    std::shared_ptr<const std::vector<uint8_t>> storage;
    uint64_t offset = 0;
    uint64_t length = 0;
};

// Immutable segmented pageable backing. Each append allocates at most one
// capture chunk; source() supports arbitrary reads across segment boundaries
// and never concatenates the complete artifact.
class artifact_segment_chain {
public:
    artifact_segment_chain();
    ~artifact_segment_chain();

    artifact_segment_chain(const artifact_segment_chain &) = delete;
    artifact_segment_chain & operator=(const artifact_segment_chain &) = delete;
    artifact_segment_chain(artifact_segment_chain &&) noexcept;
    artifact_segment_chain & operator=(artifact_segment_chain &&) noexcept;

    bool append(const uint8_t * data, size_t size) noexcept;
    uint64_t size() const noexcept;
    size_t segment_count() const noexcept;
    size_t max_segment_size() const noexcept;
    bool read(uint64_t offset, uint8_t * destination, size_t size) const noexcept;
    vbr_artifact_byte_source source() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

std::array<uint8_t, 32> vbr_capture_stream_digest(
    const artifact_segment_chain & chain) noexcept;

struct vbr_capture_lane {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    // Test/backend capability seam: skip optional event creation and use the
    // synchronized fallback required on devices without event support.
    bool force_synchronous = false;
};

struct vbr_capture_ring_accounting {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain domain;
    const llama_cache_budget_config * budget = nullptr;
};

struct vbr_capture_stream_source {
    using read_fn = bool (*)(
        const void * context,
        uint64_t offset,
        uint8_t * destination,
        size_t size) noexcept;

    uint32_t lane = 0;
    uint64_t size = 0;

    // Exactly one source shape is used. tensor != nullptr selects backend D2H;
    // otherwise read supplies deterministic CPU/synthetic bytes.
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t device = nullptr;
    const ggml_tensor * tensor = nullptr;
    uint64_t tensor_offset = 0;
    const void * context = nullptr;
    read_fn read = nullptr;

    // Deterministic synthetic completion-fault seam. Production callers keep
    // UINT64_MAX; tests prove a failed completion drains the ring and exposes
    // no verified segment.
    uint64_t fail_completion_at = UINT64_MAX;
};

struct vbr_capture_stream_stats {
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t backpressure_waits = 0;
    uint64_t event_completions = 0;
    uint64_t synchronous_fallbacks = 0;
    size_t max_segment_size = 0;
    std::array<uint8_t, 32> streaming_digest = {};
};

// One globally-bounded ring split across per-device lanes. A null device lane
// is the deterministic CPU test path. Real lanes allocate that device's host
// buffer type and use optional backend events; no event means a synchronized
// fallback, never an unbounded allocation.
class vbr_pinned_chunk_ring {
public:
    static std::unique_ptr<vbr_pinned_chunk_ring> create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting =
            nullptr) noexcept;

    ~vbr_pinned_chunk_ring();
    vbr_pinned_chunk_ring(const vbr_pinned_chunk_ring &) = delete;
    vbr_pinned_chunk_ring & operator=(const vbr_pinned_chunk_ring &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;

    vbr_capture_stream_status stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept;

private:
    struct impl;
    explicit vbr_pinned_chunk_ring(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

struct vbr_verified_segment {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    bool clean_stash = false;
    std::shared_ptr<const artifact_segment_chain> bytes;
    std::array<uint8_t, 32> streaming_digest = {};
};

struct vbr_capture_sink_result {
    vbr_capture_stream_status status =
        vbr_capture_stream_status::internal_error;
    llama_cache_acct_artifact_id reference_artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id reference_lineage;
    bool adopted = false;
};

class vbr_unit_build {
public:
    virtual ~vbr_unit_build() = default;
    virtual vbr_capture_stream_status accept_verified_segment(
        const vbr_verified_segment & segment) noexcept = 0;
    virtual vbr_capture_stream_status seal_unit() noexcept = 0;
};

class vbr_capture_build {
public:
    virtual ~vbr_capture_build() = default;
    virtual std::unique_ptr<vbr_unit_build> begin_unit(
        uint32_t unit_index,
        vbr_capture_stream_status & status) noexcept = 0;
    virtual vbr_capture_sink_result publish_reference() noexcept = 0;
};

class vbr_unit_version_sink {
public:
    virtual ~vbr_unit_version_sink() = default;
    virtual std::unique_ptr<vbr_capture_build> begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status) noexcept = 0;
};
