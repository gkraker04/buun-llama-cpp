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
    controller_unavailable,  // authenticated pool/controller-wide availability latch
};

// Commit-3 closed reset-scope transport: the producer (bridge) authenticates the scope;
// consumers must never infer global-ness from a generic child_capture_failed.
enum class common_checkpoint_reset_scope {
    none,
    capturing_slot,
    global,
};

// §9.1 capture: attach a fresh composite generation record to `ckpt` (replacing any). On any
// non-ok outcome the shadow is null — generation unknown by representation.
common_checkpoint_shadow_reason common_checkpoint_shadow_capture(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier);

// Capture variant reporting the producer-authenticated reset scope alongside the reason.
common_checkpoint_shadow_reason common_checkpoint_shadow_capture_scoped(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier,
        common_checkpoint_reset_scope &     reset_scope);

// Record-free mirrors of the bridge's closed evaluation vocabulary (value-for-value; the
// bridge TU pins the correspondence with static_asserts). The G evaluation is deliberately
// independent of the server's P and F predicates.
enum class common_checkpoint_shadow_category {
    not_applicable,
    generation_unknown,
    strict_accept,
    live_rebased_shadow_accept,
    strict_reject,
};

enum class common_checkpoint_shadow_eval_reason {
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
};

enum class common_checkpoint_shadow_tombstone {
    none,
    restore_one_behind,
    swa_wrap,
    explicit_destructive_trim,
    dependency_seq_removed,
    unexplained,
};

enum class common_checkpoint_shadow_observation {
    trivial_append,
    boundary_refined,
    destructive,
    import_refined,
};

enum class common_checkpoint_oracle_outcome {
    disabled,
    not_due,
    pass,
    set_mismatch,
    byte_mismatch,
    set_and_byte_mismatch,
    unavailable,
};

struct common_checkpoint_shadow_evaluation {
    bool                                 strict              = false;
    bool                                 live_rebased_shadow = false;
    common_checkpoint_shadow_category    category = common_checkpoint_shadow_category::generation_unknown;
    common_checkpoint_shadow_eval_reason reason   = common_checkpoint_shadow_eval_reason::record_unknown;
    common_checkpoint_shadow_observation observation_class =
            common_checkpoint_shadow_observation::trivial_append;
    common_checkpoint_shadow_tombstone tombstone_class = common_checkpoint_shadow_tombstone::none;
    bool                               refinement_used = false;
    uint32_t                           rejecting_cells = 0;
    common_checkpoint_oracle_outcome   oracle_outcome  = common_checkpoint_oracle_outcome::disabled;
    bool                               evaluated       = false;  // one sole-evaluator invocation happened
};

// Commit-3 G-only evaluation through the opaque bridge (the ONE common call site). Fails
// closed: a null/absent record or any precondition failure returns the default
// generation-unknown result with evaluated=false.
common_checkpoint_shadow_evaluation common_checkpoint_shadow_evaluate(
        const common_prompt_checkpoint &    ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier);

const char * common_checkpoint_shadow_eval_reason_name(common_checkpoint_shadow_eval_reason reason);
const char * common_checkpoint_shadow_tombstone_name(common_checkpoint_shadow_tombstone tombstone);
const char * common_checkpoint_shadow_observation_name(common_checkpoint_shadow_observation observation);
const char * common_checkpoint_oracle_outcome_name(common_checkpoint_oracle_outcome outcome);
const char * common_checkpoint_reset_scope_name(common_checkpoint_reset_scope scope);

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
