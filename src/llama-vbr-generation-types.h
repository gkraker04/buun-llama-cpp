#pragma once

#include "llama.h"
#include "llama-vbr-checkpoint-types.h"

#include <array>
#include <cstdint>
#include <vector>

// Immutable Revision-9 record vocabulary shared with the disabled oracle trust domain. Live
// tracker/index/mutation helpers deliberately do not appear in this header.
constexpr uint32_t VBR_GENERATION_PAGE_CELLS = 256;
constexpr uint32_t VBR_GENERATION_MASK_WORDS = VBR_GENERATION_PAGE_CELLS / (8u * sizeof(uint64_t));

struct vbr_pool_uuid {
    uint64_t hi = 0;
    uint64_t lo = 0;
};

inline bool operator==(const vbr_pool_uuid & lhs, const vbr_pool_uuid & rhs) {
    return lhs.hi == rhs.hi && lhs.lo == rhs.lo;
}

inline bool operator!=(const vbr_pool_uuid & lhs, const vbr_pool_uuid & rhs) {
    return !(lhs == rhs);
}

// The one definition of a real (non-sentinel) pool identity; matches the tracker's active()
// convention.
inline bool vbr_pool_uuid_is_set(const vbr_pool_uuid & uuid) {
    return uuid.hi != 0 && uuid.lo != 0;
}

enum class vbr_repr_domain : uint8_t {
    full,
    tapped,
};

// Durable admission provenance. Unlike the diagnostic operation history, this value is part of
// the immutable unit tuple compared by checkpoint_vbr_eligibility().
enum class vbr_repr_transition : uint8_t {
    initial,
    degrade_f16_to_t8_admitted,
    degrade_other,
    promote,
    partial_import,
    whole_import,
    explicit_restore,
    manifest_adopt,
    full_reset,
    recovery_invalidate,
};

// Immutable checkpoint tuple. Process-local stability/coordination fields (publish_seq, flags)
// are intentionally absent rather than copied-and-zeroed.
struct vbr_checkpoint_unit_generation {
    uint64_t            repr_gen         = 0;
    int32_t             current_type     = -1;
    int32_t             last_source_type = -1;
    vbr_repr_domain     domain           = vbr_repr_domain::full;
    uint8_t             promote_hops     = 0;
    vbr_repr_transition last_transition  = vbr_repr_transition::initial;
};

enum class checkpoint_child_dependency_mode : uint8_t {
    absent,
    payload_complete,
    live_guarded,
};

struct vbr_generation_page_ref {
    uint32_t                                        page_index        = 0;
    uint32_t                                        captured_page_gen = 0;
    std::array<uint64_t, VBR_GENERATION_MASK_WORDS> covered_mask      = {};
};
static_assert(sizeof(vbr_generation_page_ref) == 40,
              "Revision-9 checkpoint page references are fixed at 40 bytes");

struct vbr_checkpoint_generation_stream {
    uint32_t                             stream_index              = 0;
    llama_seq_id                         dependency_seq_id         = -1;
    llama_pos                            computation_frontier      = -1;
    uint32_t                             captured_dependency_count = 0;
    std::vector<vbr_generation_page_ref> pages;
};

struct vbr_checkpoint_generation_controller {
    uint32_t                                      child_id          = 0;
    checkpoint_child_dependency_mode              dependency_mode   = checkpoint_child_dependency_mode::absent;
    vbr_pool_uuid                                 pool_uuid         = {};
    uint64_t                                      global_generation = 0;
    std::vector<vbr_checkpoint_unit_generation>   units;
    std::vector<vbr_checkpoint_generation_stream> streams;
};

struct vbr_checkpoint_generation_record {
    uint32_t                                          version = 1;
    vbr_checkpoint_generation_status                  status  = vbr_checkpoint_generation_status::generation_unknown;
    std::array<uint8_t, 32>                           identity_policy_order_digest = {};
    std::vector<vbr_checkpoint_generation_controller> controllers;
};

// Record-vs-record deep equality (§9.2 dedup relation). This compares two captured checkpoint
// records with each other — never a captured record against live tracker state, which remains
// the exclusive province of checkpoint_vbr_eligibility().
inline bool operator==(const vbr_checkpoint_unit_generation & lhs, const vbr_checkpoint_unit_generation & rhs) {
    return lhs.repr_gen == rhs.repr_gen && lhs.current_type == rhs.current_type &&
           lhs.last_source_type == rhs.last_source_type && lhs.domain == rhs.domain &&
           lhs.promote_hops == rhs.promote_hops && lhs.last_transition == rhs.last_transition;
}

inline bool operator==(const vbr_generation_page_ref & lhs, const vbr_generation_page_ref & rhs) {
    return lhs.page_index == rhs.page_index && lhs.captured_page_gen == rhs.captured_page_gen &&
           lhs.covered_mask == rhs.covered_mask;
}

inline bool operator==(const vbr_checkpoint_generation_stream & lhs, const vbr_checkpoint_generation_stream & rhs) {
    return lhs.stream_index == rhs.stream_index && lhs.dependency_seq_id == rhs.dependency_seq_id &&
           lhs.computation_frontier == rhs.computation_frontier &&
           lhs.captured_dependency_count == rhs.captured_dependency_count && lhs.pages == rhs.pages;
}

inline bool operator==(const vbr_checkpoint_generation_controller & lhs, const vbr_checkpoint_generation_controller & rhs) {
    return lhs.child_id == rhs.child_id && lhs.dependency_mode == rhs.dependency_mode &&
           lhs.pool_uuid == rhs.pool_uuid && lhs.global_generation == rhs.global_generation &&
           lhs.units == rhs.units && lhs.streams == rhs.streams;
}

inline bool operator==(const vbr_checkpoint_generation_record & lhs, const vbr_checkpoint_generation_record & rhs) {
    return lhs.version == rhs.version && lhs.status == rhs.status &&
           lhs.identity_policy_order_digest == rhs.identity_policy_order_digest &&
           lhs.controllers == rhs.controllers;
}
