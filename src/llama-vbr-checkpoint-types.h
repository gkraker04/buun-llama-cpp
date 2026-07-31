#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

// Record-free leaf vocabulary for the checkpoint shadow bridge (F4, verify round): the opaque
// bridge header includes ONLY this file, so the composite generation record can never become
// transitively visible to the common holder TU. llama-vbr-generation-types.h re-exports these
// for the record-typed implementation files.

enum class vbr_checkpoint_generation_status : uint8_t {
    complete,
    generation_unknown,
};

// §9.1 closed capture outcome. `not_applicable` (no armed controller anywhere in the memory
// tree) is distinct from failure: it produces no counters and no lifecycle logs.
enum class vbr_checkpoint_capture_reason : uint8_t {
    ok,
    not_applicable,
    invalid_arguments,
    unarmed_live_covered,
    child_capture_failed,
    oracle_mismatch,
    internal_error,
    controller_unavailable,
    _count,
};

// A2 commit-3 reset transport. The producer authenticates the scope; consumers must never
// infer it from a generic child_capture_failed result.
enum class vbr_checkpoint_reset_scope : uint8_t {
    none,
    capturing_slot,
    global,
    _count,
};

// Record-free mirror of the sole generation evaluator's closed leaf vocabulary. Keep these
// separate from llama-vbr-generation.h so the opaque bridge cannot expose the process-local
// checkpoint record or live tracker types to common/server consumers.
enum class vbr_checkpoint_shadow_category : uint8_t {
    not_applicable,
    generation_unknown,
    strict_accept,
    live_rebased_shadow_accept,
    strict_reject,
    _count,
};

enum class vbr_checkpoint_shadow_reason : uint8_t {
    none,
    capability_not_applicable,
    record_unknown,
    record_version,
    identity_or_frontier,
    controller_shape,
    child_order,
    dependency_mode,
    controller_inactive,
    controller_unstable,
    // Frozen schema-4 wire ordinal/spelling; semantic meaning is lineage mismatch.
    pool_uuid,
    global_generation,
    unit_shape,
    unit_unstable,
    unit_generation,
    live_rebased_transition,
    stream_shape,
    stream_order,
    malformed_page_refs,
    page_out_of_range,
    dependency_changed,
    dependency_membership_lost,
    dependency_cardinality,
    _count,
};

enum class vbr_checkpoint_shadow_tombstone : uint8_t {
    none,
    restore_one_behind,
    swa_wrap,
    explicit_destructive_trim,
    dependency_seq_removed,
    unexplained,
    _count,
};

enum class vbr_checkpoint_shadow_observation : uint8_t {
    trivial_append,
    boundary_refined,
    destructive,
    import_refined,
    _count,
};

// One outcome per audited observation. This is deliberately rich enough for the server to
// increment its set/hash/unavailable counters exactly once without src becoming a second
// durable counter authority.
enum class vbr_checkpoint_oracle_outcome : uint8_t {
    disabled,
    not_due,
    pass,
    set_mismatch,
    byte_mismatch,
    set_and_byte_mismatch,
    unavailable,
    _count,
};

// Append detection for every closed enum mirrored by common. Common pins the prior last value
// numerically; these sentinels make an unmirrored append fail here as well.
static_assert(static_cast<uint8_t>(vbr_checkpoint_capture_reason::_count) == 8);
static_assert(static_cast<uint8_t>(vbr_checkpoint_reset_scope::_count) == 3);
static_assert(static_cast<uint8_t>(vbr_checkpoint_shadow_category::_count) == 5);
static_assert(static_cast<uint8_t>(vbr_checkpoint_shadow_reason::_count) == 23);
static_assert(static_cast<uint8_t>(vbr_checkpoint_shadow_tombstone::_count) == 6);
static_assert(static_cast<uint8_t>(vbr_checkpoint_shadow_observation::_count) == 4);
static_assert(static_cast<uint8_t>(vbr_checkpoint_oracle_outcome::_count) == 7);

// D-A2-8: server-layer checkpoint identity/frontier fields bound into the record digest.
// next_position is the EXCLUSIVE computation frontier (dependencies are pos < next_position),
// matching the capture-side filter — never pos_max.
struct vbr_checkpoint_frontier_fields {
    const char * execution_identity          = nullptr;
    size_t       execution_identity_len      = 0;
    const char * adapter_config_identity     = nullptr;
    size_t       adapter_config_identity_len = 0;
    const char * media_content_identity      = nullptr;
    size_t       media_content_identity_len  = 0;
    uint64_t     sequence_epoch = 0;
    int64_t      token_count    = 0;
    llama_pos    next_position  = -1;
};
