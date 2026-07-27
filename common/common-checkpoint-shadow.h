#pragma once

#include "common.h"

#include <cstdint>
#include <vector>

// §9.1–9.3 checkpoint shadow lifecycle helpers (D-A2-7). Everything here operates on
// common_prompt_checkpoint; the underlying opaque record handle is named only inside the
// single bridge TU. All functions fail closed and never alter legacy checkpoint behavior.

enum class common_checkpoint_shadow_reason {
    ok,
    not_applicable,        // no armed VBR controller anywhere: zero counters, zero logs
    invalid_arguments,
    unarmed_live_covered,  // §11.1 row 16
    child_capture_failed,
    oracle_mismatch,
    internal_error,
};

// §9.1 capture: attach a fresh composite generation record to `ckpt` (replacing any). On any
// non-ok outcome the shadow is null — generation unknown by representation.
common_checkpoint_shadow_reason common_checkpoint_shadow_capture(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier);

// True iff a complete generation record is attached.
bool common_checkpoint_shadow_complete(const common_prompt_checkpoint & ckpt);

// §9.2 dedup relation: true only when BOTH checkpoints hold complete records that are deeply
// equal. Absence/unknown on either side is false, never a semantic opinion.
bool common_checkpoint_shadow_equal(const common_prompt_checkpoint & a, const common_prompt_checkpoint & b);

// §9.3 step 5: move src's shadow into dst (dst's old record is destroyed). Only shadow state
// is touched — no payload or accelerator bytes.
void common_checkpoint_shadow_adopt(common_prompt_checkpoint & dst, common_prompt_checkpoint & src);

// Process-wide count of shadows dropped by checkpoint copies (host-cache staging, clones).
uint64_t common_checkpoint_shadow_dropped_on_copy();

const char * common_checkpoint_shadow_reason_name(common_checkpoint_shadow_reason reason);

// §9.3 refresh byte-proof observation: the current state, serialized detached by the caller.
// A null vector pointer means the component could not be reproduced.
struct common_checkpoint_refresh_observation {
    const std::vector<uint8_t> * tgt  = nullptr;
    const std::vector<uint8_t> * dft  = nullptr;
    const std::vector<uint8_t> * ring = nullptr;
    const std::vector<uint8_t> * spec = nullptr;
    bool dft_applicable  = false;  // a draft context exists
    bool ring_applicable = false;  // the slot can speculate (ring state reproducible)
    bool spec_applicable = false;  // speculative impl state reproducible
};

enum class common_checkpoint_refresh_verdict {
    proven,
    refused_cannot_reproduce,
    refused_byte_mismatch,  // counted as shadow_refresh_nondeterministic_byte_mismatch
};

// Pure §9.3 proof: exact byte equality against EVERY retained payload — data_tgt; data_dft when
// a draft context or a retained draft payload exists; each applicable typed accelerator payload
// (accel.ring, accel.spec). A retained/applicable component that cannot be reproduced refuses.
// Touches no shadow, payload, or accelerator state.
common_checkpoint_refresh_verdict common_checkpoint_shadow_refresh_proof(
        const common_prompt_checkpoint &             retained,
        const common_checkpoint_refresh_observation & current);
