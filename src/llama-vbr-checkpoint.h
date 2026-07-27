#pragma once

#include "llama.h"
#include "llama-vbr-checkpoint-types.h"

#include <cstddef>

// D-A2-7/D-A2-8 checkpoint shadow bridge. INTERNAL header — never installed, not part of
// include/llama.h. Functions are LLAMA_API so shared/Windows builds export them to the common
// library (precedent: src/llama-ext.h), and every exported operation is noexcept: internal
// exceptions are caught and fail closed at the libllama/common boundary. The handle stays
// opaque — this header includes only the record-free leaf vocabulary, so no consumer of the
// bridge (in particular the common holder TU) can even transitively see the composite
// generation record (F4, verify round; CI bans any generation-header include here).

struct llama_vbr_checkpoint_shadow;

struct llama_vbr_checkpoint_capture_result {
    // null == no shadow (generation unknown by representation); reason says why
    struct llama_vbr_checkpoint_shadow * handle = nullptr;
    vbr_checkpoint_capture_reason        reason = vbr_checkpoint_capture_reason::internal_error;
};

// §9.1 composite capture over the memory tree for one checkpoint sequence. A live handle is
// always a COMPLETE record. frontier->next_position is the exclusive computation frontier.
LLAMA_API void llama_vbr_checkpoint_shadow_capture(
        llama_memory_t                          mem,
        llama_seq_id                            seq_id,
        const vbr_checkpoint_frontier_fields *  frontier,
        llama_vbr_checkpoint_capture_result *   result) noexcept;

LLAMA_API void llama_vbr_checkpoint_shadow_free(struct llama_vbr_checkpoint_shadow * shadow) noexcept;

// Deep record equality (§9.2 dedup relation). True only when BOTH handles hold complete records
// that compare equal; any null/incomplete side is false.
LLAMA_API bool llama_vbr_checkpoint_shadow_equal(
        const struct llama_vbr_checkpoint_shadow * a,
        const struct llama_vbr_checkpoint_shadow * b) noexcept;

// Resident bytes attributable to the record (deterministic in its content).
LLAMA_API size_t llama_vbr_checkpoint_shadow_size(const struct llama_vbr_checkpoint_shadow * shadow) noexcept;

// null -> generation_unknown.
LLAMA_API vbr_checkpoint_generation_status llama_vbr_checkpoint_shadow_status(
        const struct llama_vbr_checkpoint_shadow * shadow) noexcept;

LLAMA_API const char * llama_vbr_checkpoint_shadow_reason_name(vbr_checkpoint_capture_reason reason) noexcept;
