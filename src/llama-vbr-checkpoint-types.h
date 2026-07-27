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
};

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
