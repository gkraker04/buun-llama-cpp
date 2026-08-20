#pragma once

#include "llama-cache-budget.h"

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// One process-wide ceiling shared by capture (D2H) and adoption (H2D)
// instances. Each ring is independently bounded below this ceiling.
static constexpr uint64_t VBR_PINNED_RING_MAX_BYTES = 256ull*1024*1024;

enum class vbr_pinned_ring_create_failure : uint8_t {
    none = 0,
    invalid_geometry,
    invalid_accounting_binding,
    existing_ring_charge,
    accounting_update_failed,
    budget_reset_failed,
    budget_unavailable,
    budget_exceeded,
    global_capacity_exceeded,
    invalid_lane_binding,
    duplicate_device_lane,
    host_buffer_type_unavailable,
    host_buffer_allocation_failed,
    host_buffer_too_small,
    host_buffer_base_unavailable,
    lane_underprovisioned,
    accounting_charge_failed,
    internal_error,
    _count,
};

struct vbr_pinned_ring_lane {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    bool force_synchronous = false;
};

struct vbr_pinned_ring_accounting {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain domain;
    const llama_cache_budget_config * budget = nullptr;
    llama_cache_acct_category category =
        llama_cache_acct_category::pinned_preimage_ring;

    // Deterministic test seam for a ledger fault between physical allocation
    // and the final ring gauge. Production leaves both fields null.
    void * charge_fault_context = nullptr;
    void (*inject_charge_fault)(void * context) noexcept = nullptr;
};

class vbr_bounded_pinned_ring_core;

// Move-only ownership of one ring chunk between acquire and release. The
// adapter that submits a transfer is responsible for waiting before release.
class vbr_pinned_chunk_lease {
public:
    vbr_pinned_chunk_lease() = default;
    vbr_pinned_chunk_lease(vbr_pinned_chunk_lease && other) noexcept;
    vbr_pinned_chunk_lease & operator=(vbr_pinned_chunk_lease && other) noexcept;
    ~vbr_pinned_chunk_lease() = default;

    vbr_pinned_chunk_lease(const vbr_pinned_chunk_lease &) = delete;
    vbr_pinned_chunk_lease & operator=(const vbr_pinned_chunk_lease &) = delete;

    explicit operator bool() const noexcept { return chunk_ != nullptr; }
    uint8_t * data() const noexcept { return data_; }
    size_t capacity() const noexcept { return capacity_; }
    size_t valid() const noexcept { return valid_; }

private:
    vbr_bounded_pinned_ring_core * owner_ = nullptr;
    void * chunk_ = nullptr;
    uint8_t * data_ = nullptr;
    size_t capacity_ = 0;
    size_t valid_ = 0;

    void reset() noexcept;
    friend class vbr_bounded_pinned_ring_core;
};

// Direction-neutral bounded lane/chunk/event/backpressure core. It does not
// know whether bytes flow device->host or host->device and never hashes or
// retains artifact bytes.
class vbr_bounded_pinned_ring_core {
public:
    static std::unique_ptr<vbr_bounded_pinned_ring_core> create(
        const std::vector<vbr_pinned_ring_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        const vbr_pinned_ring_accounting * accounting,
        vbr_pinned_ring_create_failure & failure) noexcept;

    ~vbr_bounded_pinned_ring_core();
    vbr_bounded_pinned_ring_core(const vbr_bounded_pinned_ring_core &) = delete;
    vbr_bounded_pinned_ring_core & operator=(const vbr_bounded_pinned_ring_core &) = delete;

    uint64_t capacity_bytes() const noexcept;
    size_t chunk_bytes() const noexcept;
    size_t lane_count() const noexcept;
    const vbr_pinned_ring_lane * lane_binding(uint32_t lane) const noexcept;

    // Returns an empty lease with would_block=true when the lane's next chunk
    // is still owned by an outstanding transfer. The adapter then completes
    // its oldest pending lease and retries; acquire never waits implicitly.
    vbr_pinned_chunk_lease acquire(
        uint32_t lane, bool & would_block) noexcept;

    // Marks a filled chunk in flight and records an event when available.
    // Without an event the backend is synchronized before returning.
    bool submit(
        vbr_pinned_chunk_lease & lease,
        size_t valid,
        ggml_backend_t backend,
        bool & synchronous_fallback) noexcept;

    // Completes a submitted chunk. event_completion reports which completion
    // path was used so the direction adapter preserves its historical stats.
    bool wait(
        vbr_pinned_chunk_lease & lease,
        bool & event_completion) noexcept;

    void release(vbr_pinned_chunk_lease & lease) noexcept;

private:
    struct impl;
    explicit vbr_bounded_pinned_ring_core(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};
