#pragma once

#include "common-cache-plan.h"

#include <vector>

// common-cache-plan-estimate.h — B shadow-planner estimators (§7.5), schema v2.
//
// Versioned, policy-free cost estimation over the schema-v2 candidate inventory. SHADOW
// ONLY: fills per-candidate cost terms, predicted totals, and the shadow choice/tie set on
// a finalized-being record; never touches a shipped decision. Runs strictly inside the
// finalize planner boundary (A2) — throwing is tolerated there (the boundary clears planner
// outputs), but these functions avoid allocation and do not throw on their own.
//
// B-2 discipline: coefficients come ONLY from a fitted calibration profile. No profile →
// no estimates (typed-unavailable), never a default coefficient. Measured actuals are
// separate record fields and are never substituted.

struct common_cache_plan_calib {
    const char * profile;            // stable id: "{model class}/{hardware class}/b{batch}"
    uint32_t     estimator_version;  // bumped on ANY coefficient or formula change
    // fitted coefficients (dorei microbench sweep; see tools/server/bench/cache-plan-calibrate.py)
    double replay_us_per_token;      // forward replay cost per prompt token
    double restore_us_per_byte;      // pageable-host -> device state install, per byte
    double workspace_setup_us;       // fixed per-restore setup overhead
};

// Checked-in fitted table lookup (data reviewed like code). Returns nullptr when the
// profile has no fitted entry — the caller then leaves planner outputs unavailable.
const common_cache_plan_calib * common_cache_plan_calib_find(const std::string & profile);

// THE profile-composition rule (single producer-side spelling, tested): lowercases and
// squashes everything outside [a-z0-9/.] to '-'. The server composes records with this;
// fitted table entries must key on exactly this output — a drifted spelling fails silently
// (estimators legally refuse forever), so no consumer hand-rolls it. kv_desc names the KV
// cache regime ("k<ctk>-v<ctv>"): replay/restore costs differ across KV codecs (VBR,
// turbo encode paths), so an f16 run and a vbr run must never share coefficients.
std::string common_cache_plan_calib_profile(const std::string & model_stem,
                                            const std::string & hw_desc, int n_batch,
                                            const std::string & kv_desc);

// EFFECTIVE VBR regime identity (D-pins r2 finding 2): requested CLI strings are NOT the
// regime — `auto` resolves to a selected family/policy/schedule, the aggregate floor
// resolves to a capacity, and documented developer env overrides (VBR_BUDGET_MIB,
// VBR_MIN_BITS, VBR_POLICY_LADDER) can move the controller's budget AFTER CLI resolution.
// Two runs whose requested strings match but whose resolved regimes differ must NOT share
// fitted coefficients. Every field below is an effective/resolved value; `overrides` carries
// the raw override tokens so an overridden run keys distinctly (empty when none are set).
// `unrepresented_override` means the effective state could not be established — the caller
// then has NO profile (refuse) rather than a possibly-aliased one.
struct common_cache_plan_vbr_regime {
    bool        armed = false;
    bool        side_k = false;      // K took the vbr alias
    bool        side_v = false;
    std::string budget_mode;         // resolved budget mode (dynamic/fixed tier)
    std::string family;              // vbr_selected_family
    std::string policy;              // vbr_selected_policy
    std::string schedule;            // vbr_selected_schedule (path/name identity)
    double      capacity_bits = 0.0; // resolved aggregate floor (bits/value)
    double      selected_bpv  = 0.0; // measured BPV of the selected rung
    uint64_t    vram_budget_bytes = 0; // resolved explicit budget, 0 == auto
    float       reclaim_floor_bpv = 0.0f;
    float       reset_keep_frac   = 0.0f;
    std::string overrides;           // raw env-override tokens, empty when none
    bool        unrepresented_override = false;
};

// THE KV-regime segment, pure and tested. Non-VBR runs key on the ggml type names only.
// Returns an EMPTY string when the regime is armed but unrepresentable — the profile is
// then empty and the planner refuses (no_profile), never a possibly-aliased match.
std::string common_cache_plan_calib_kv(const common_cache_plan_vbr_regime & vbr,
                                       const std::string & type_k, const std::string & type_v);

// THE placement-key (hardware class) construction, pure and tested: POSITIONAL device
// order (main_gpu / tensor_split index into it — reversed heterogeneous orders must
// produce distinct keys), effective ngl (0 or no devices → "cpu"), and for multi-GPU the
// split mode / main gpu / per-device tensor-split percents. `tensor_split` may be null
// (treated as all-zero / auto).
std::string common_cache_plan_calib_hw(const std::vector<std::string> & gpu_descs,
                                       int n_gpu_layers_eff, int split_mode, int main_gpu,
                                       const float * tensor_split);

// Tie resolution floor (planner-owned, deterministic): candidates whose predicted totals
// are within max(5% of the minimum, 100us) of the minimum form the tie set. The recorded
// shadow choice is the tie-set member with the smallest (provider, source_id, ordinal) key
// — NEVER a function of the shipped choice (r2 finding 3); agreement is computed offline
// as shipped-in-tie-set.
constexpr double COMMON_CACHE_PLAN_TIE_REL_FLOOR = 0.05;
constexpr double COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US = 100.0;

// Estimate every valid candidate (disposition accepted / valid_not_chosen_cost; chain rows
// composed from their components; component_only rows are estimated as components but
// EXCLUDED from the root optimum) and fill shadow_choice + shadow_tie_set. Returns a
// closed status; anything but `ok` leaves ALL planner outputs typed-unavailable (the
// all-or-nothing clear is this function's postcondition):
//   invalid_calibration — profile mismatch vs rec.calibration_profile, version 0, or
//       non-finite/negative coefficients (validated HERE, at the trust boundary);
//   incomplete_evidence — provider overflow, a dropped derived plan, an unresolved visited
//       candidate (disposition unavailable), a valid row missing the scalars estimation
//       needs, unknown n_prompt, or an empty participant set — never a partial optimum.
// B-covered terms are restore/replay/workspace; transfer and eviction stay unavailable
// until D.
common_cache_plan_planner_status common_cache_plan_estimate_and_choose(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib);
