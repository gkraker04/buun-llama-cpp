#pragma once

#include "common-checkpoint-shadow.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

// §8.3–8.5 pre-flip qualification coordinator (commit 3). Pure logic over one immutable
// snapshot of the reverse checkpoint list: candidate computation, relation evidence, and the
// disposition table. NOTHING here touches shipped selection, either live authority bit, or
// the selected read path — commit 3 computes qualified_pending only; the first G flip is A3.

// Evidence schema version: a stored evidence object with a different version (or a different
// server authority_generation) is stale and lazily clears before its next observation. Stale
// evidence is discarded, never upgraded.
constexpr uint32_t COMMON_SHADOW_EVIDENCE_VERSION = 1;

// §8.4 qualification minima. Parameterized so CPU rows can drive tiny ladders; production
// consumers use the defaults below (1024 / 64 / 16 / zero-tolerance).
struct common_shadow_qualification_minima {
    uint64_t consecutive_agreements = 1024;
    uint64_t per_class_observations = 64;   // every applicable nontrivial class
    uint64_t boundary_refinements   = 16;   // boundary observations with actual mask refinement
};

// One relation's versioned streak/class evidence (WS-4/L, WS-4/G per slot; WS-7/P, WS-7/F
// server-wide). §8.4 zero-tolerance for unexplained disagreement / exception / oracle failure
// is enforced STRUCTURALLY: any such fault resets this object, so an accumulation window that
// reaches the consecutive minimum contained zero faults by construction. Fault totals are
// server-level observability counters, not evidence fields (they survive evidence resets).
struct common_shadow_relation_evidence {
    uint32_t evidence_version     = COMMON_SHADOW_EVIDENCE_VERSION;
    uint64_t authority_generation = 0;

    uint64_t agreement_streak     = 0;
    uint64_t agreements_total     = 0;
    uint64_t disagreements_total  = 0;
    // indexed by common_checkpoint_shadow_observation (trivial_append never qualifies but is
    // still counted for observability)
    std::array<uint64_t, 4> class_counts = {};
    uint64_t boundary_refinements = 0;

    bool qualified = false;

    void reset(uint64_t new_authority_generation);
    // Lazy version/authority clearing before the next observation (Q5.1: discard, never
    // upgrade). Returns true when the object was cleared.
    bool ensure_current(uint64_t current_authority_generation);
    void recompute_qualified(const common_shadow_qualification_minima & minima);
};

// One reverse-list row of the immutable snapshot (index 0 = newest checkpoint). The caller
// separates positional predicates from legacy validity so the G axes never inherit L checks.
struct common_shadow_candidate_row {
    bool p_pos      = false;  // physical/legacy positional predicate
    bool f_pos      = false;  // logical frontier positional predicate (incl. frontier currency)
    bool l_valid    = false;  // legacy validity (non-positional legacy checks)
    bool has_record = false;  // a complete generation record is attached
};

// Closed per-axis G verdict for candidate scanning, derived from the bridge evaluation.
enum class common_shadow_g_verdict {
    strict,             // strict accept: supplies the candidate
    expected_tombstone, // availability non-event: pauses the axis, never falls through
    expected_unknown,   // unknown/not-evaluable/absent record: pauses (unknown-skip is A3)
    live_rebased,       // shadow-only observation: pauses strict qualification in A2
    unexplained,        // unexplained disagreement: resets relations on this axis
    oracle_failure,     // audit mismatch: resets relations on this axis
};

struct common_shadow_axis_result {
    int64_t                 candidate = -1;  // reverse-list index, -1 = none
    bool                    paused    = false;
    bool                    failed    = false;  // unexplained or oracle_failure
    // True only when this axis made an ELIGIBLE observation (a candidate, a pause, or a
    // failure at a positionally eligible row). A scan with nothing positionally eligible is
    // not an observation at all — relations touching an unobserved axis pause, they never
    // mint a vacuous agreement from two absent candidates.
    bool                    observed  = false;
    common_shadow_g_verdict verdict   = common_shadow_g_verdict::expected_unknown;
    int64_t                 verdict_row = -1;   // row that produced the verdict (or -1)
};

struct common_shadow_candidates {
    common_shadow_axis_result p_l;
    common_shadow_axis_result f_l;
    common_shadow_axis_result p_g;
    common_shadow_axis_result f_g;
    uint32_t g_evaluations = 0;  // bridge invocations this scan (<=1 per row by construction)
};

common_shadow_g_verdict common_shadow_classify_evaluation(
        const common_checkpoint_shadow_evaluation & evaluation);

// Compute all four candidates from one snapshot. `evaluate` is invoked lazily, AT MOST ONCE
// per row (memoized internally), only for rows some G axis actually reaches. The G scan stops
// at the first positionally-eligible row: a non-strict verdict pauses or fails the axis, it
// never falls through to an older checkpoint (no fake agreement).
common_shadow_candidates common_shadow_compute_candidates(
        const std::vector<common_shadow_candidate_row> &                    rows,
        const std::function<common_checkpoint_shadow_evaluation(size_t)> &  evaluate);

enum class common_shadow_relation { ws4_l, ws4_g, ws7_p, ws7_f };

// Pin-8 disposition per relation for one scan.
enum class common_shadow_disposition {
    advance,   // ordinary agreement: streak/class accounting advances
    disagree,  // ordinary disagreement: streak resets, totals advance
    pause,     // expected tombstone/unknown/live_rebased: availability non-event
    reset,     // unexplained disagreement or oracle failure: scoped evidence reset
};

struct common_shadow_scan_outcome {
    std::array<common_shadow_disposition, 4> disposition = {};  // indexed by common_shadow_relation
    bool all_qualified = false;  // all four evidence objects qualified after this scan
};

// Per-relation observation metadata for one scan. Each relation is credited only with the
// class of the evaluation that actually establishes ITS agreement (WS-7/P from the P/G
// evaluation, WS-7/F from F/G, WS-4/G from the shared agreed row; WS-4/L conservatively from
// the agreed row's evaluation when one happened this scan, else trivial_append = no credit).
struct common_shadow_relation_observation {
    common_checkpoint_shadow_observation observation =
            common_checkpoint_shadow_observation::trivial_append;
    bool boundary_refined = false;
};

// Apply one scan to the four relation evidence objects (order: ws4_l, ws4_g, ws7_p, ws7_f).
// Callers pass the CURRENT authority generation; stale evidence lazily clears first.
common_shadow_scan_outcome common_shadow_apply_scan(
        std::array<common_shadow_relation_evidence *, 4> &        evidence,
        const common_shadow_candidates &                          candidates,
        const std::array<common_shadow_relation_observation, 4> & observations,
        uint64_t                                                  authority_generation,
        const common_shadow_qualification_minima &                minima);

const char * common_shadow_relation_name(common_shadow_relation relation);
const char * common_shadow_disposition_name(common_shadow_disposition disposition);
const char * common_shadow_g_verdict_name(common_shadow_g_verdict verdict);
