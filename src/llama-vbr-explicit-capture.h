#pragma once

#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-checkpoint-types.h"

#include <array>
#include <cstdint>
#include <vector>

class llama_memory_i;

enum class vbr_explicit_capture_status : uint8_t {
    ok = 0,
    not_armed,
    unsupported_layout,
    slot_not_idle,
    identity_unavailable,
    generation_unavailable,
    registry_busy,
    recovery_pending,
    // Reserved for F3.3's route-level geometry diagnostics. F3.2 maps all
    // private-hook geometry refusals to generation_unavailable.
    geometry_mismatch,
    stash_inconsistent,
    required_companion_unavailable,
    size_overflow,
    ring_unavailable,
    admission_refused,
    transfer_failed,
    short_read,
    // Reserved for a backend that can report asynchronous event failure.
    // Today's ggml completion API is void, so F3.2 detects it by digest/length.
    event_failed,
    source_changed,
    hash_mismatch,
    dedup_validation_failed,
    accounting_failed,
    publication_failed,
    internal_error,
    _count,
};

const char * vbr_explicit_capture_status_name(
    vbr_explicit_capture_status status) noexcept;

// One runtime pool-to-portable-topology binding. Device ordinals are portable
// only within the cited topology; lane identifies the F3.1 D2H ring lane.
struct vbr_explicit_capture_pool_binding {
    vbr_pool_uuid pool_uuid;
    int device = -1;
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
    uint32_t lane = UINT32_MAX;
};

struct vbr_explicit_representation_identity {
    uint32_t codec_id = 0;
    uint32_t codec_version = 0;
    std::array<uint8_t, 32> codebook_digest = {};
    std::array<uint8_t, 32> rotation_digest = {};
    std::array<uint8_t, 32> meansub_digest = {};
};

struct vbr_explicit_companion_provider {
    using capture_fn = bool (*)(
        const void * context,
        llama_seq_id sequence,
        std::vector<uint8_t> & output) noexcept;
    using size_fn = bool (*)(
        const void * context,
        llama_seq_id sequence,
        uint64_t & output) noexcept;

    vbr_artifact_companion_kind kind =
        vbr_artifact_companion_kind::typed_accelerator;
    uint32_t format_version = 1;
    std::array<uint8_t, 32> build_identity_digest = {};
    vbr_artifact_portable_domain domain;
    bool required = true;
    const void * context = nullptr;
    size_fn size = nullptr;
    capture_fn capture = nullptr;
};

struct vbr_explicit_capture_request {
    using representation_identity_fn = bool (*)(
        const void * context,
        int32_t current_type,
        bool value_side,
        vbr_explicit_representation_identity & output) noexcept;

    llama_seq_id sequence = -1;
    vbr_checkpoint_frontier_fields frontier;
    vbr_artifact_identity_block identity;
    std::array<uint8_t, 32> identity_policy_order_digest = {};
    bool idle_decode_thread = false;
    vbr_pinned_chunk_ring * ring = nullptr;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<vbr_explicit_companion_provider> companions;
    const void * representation_context = nullptr;
    representation_identity_fn representation_identity = nullptr;
};

struct vbr_explicit_capture_accounting {
    using prepare_fn = bool (*)(
        void * context,
        const vbr_artifact_package & package) noexcept;

    const llama_cache_budget_config * budget = nullptr;
    llama_cache_transaction_fault fault;
    void * context = nullptr;
    // Called after the exact package accounting manifest exists and before
    // begin_capture. A catalog binding/configuration adapter lives here
    // rather than weakening the generic sink interface.
    prepare_fn prepare = nullptr;
};

struct vbr_explicit_capture_result {
    vbr_explicit_capture_status status =
        vbr_explicit_capture_status::internal_error;
    vbr_capture_sink_result sink;
    uint32_t controllers = 0;
    uint32_t units = 0;
    uint32_t companions = 0;
    uint64_t payload_bytes = 0;
    uint64_t stash_bytes = 0;
    uint64_t companion_bytes = 0;
};

vbr_explicit_capture_result vbr_capture_explicit_manifest(
    llama_memory_i & memory,
    const vbr_explicit_capture_request & request,
    vbr_unit_version_sink & sink,
    const vbr_explicit_capture_accounting & accounting) noexcept;
