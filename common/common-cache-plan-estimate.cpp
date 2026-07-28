#include "common-cache-plan-estimate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <tuple>

// Fitted calibration table. Filled ONLY by the dorei microbench sweep + offline fit
// (tools/server/bench/cache-plan-calibrate.py); entries are data reviewed like code.
// Profiles without an entry refuse (B-2: no default coefficients, ever).

// dorei 2026-07-28 campaign (36 cold records over 64-2048 tokens, 6 composed host->ckpt
// restores over 20-50 MiB payloads; replay intercept ~13.3 ms absorbed by the fit)
static const common_cache_plan_calib CALIB_DOREI_QWEN35_2B_3090_B512 = {
    "qwen35-2b-q4-k---medium/nvidia-geforce-rtx-3090-ngl99/b512/kf16-vf16", 1,
    75.688, 0.000010, 7184.4,
};

// dorei 2026-07-28 campaign, buun's 27B-VBR serving config under PINNED -ngl 99 and the
// DEFAULT VBR ladder regime (dynamic budget / auto floor / auto vram / auto policy /
// reclaim 8.125 / reset-keep 0.25 — the key names it; a different ladder config is a
// different regime and must be re-fitted). (the
// auto-fit placement varied per launch — ngl64 measured 979 us/tok, all-GPU measures
// 861.5: the profile key distinguishing placements is load-bearing). 23 cold records,
// 5 checkpoint restores — hybrid payloads constant at 156.9 MiB, restore cost FLAT
// ~158 ms carried by workspace; crossover vs replay ~= 183 tokens.
static const common_cache_plan_calib CALIB_DOREI_QWEN36_27B_VBR_3090 = {
    "qwen35-27b-q6-k/nvidia-geforce-rtx-3090-ngl99/b2048/kvbr-vvbr-vbr-dynamic-dynamic-runtime-controller--1.25-1.25-0-8.125-0.25", 1,
    861.510, 0.0, 158069.2,
};

static const common_cache_plan_calib * const calib_table[] = {
    &CALIB_DOREI_QWEN35_2B_3090_B512,
    &CALIB_DOREI_QWEN36_27B_VBR_3090,
    nullptr, // sentinel so the array is never empty; skipped by the scan below
};

const common_cache_plan_calib * common_cache_plan_calib_find(const std::string & profile) {
    for (const auto * entry : calib_table) {
        if (entry && profile == entry->profile) {
            return entry;
        }
    }
    return nullptr;
}

std::string common_cache_plan_calib_profile(const std::string & model_stem,
                                            const std::string & hw_desc, int n_batch,
                                            const std::string & kv_desc) {
    std::string prof = model_stem + "/" + hw_desc + "/b" + std::to_string(n_batch)
                     + "/" + kv_desc;
    for (char & ch : prof) {
        ch = ch >= 'A' && ch <= 'Z' ? char(ch - 'A' + 'a') : ch;
        if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '/' || ch == '.')) {
            ch = '-';
        }
    }
    return prof;
}

std::string common_cache_plan_calib_kv(const common_cache_plan_vbr_regime & vbr,
                                       const std::string & type_k, const std::string & type_v) {
    const std::string k = (vbr.armed && vbr.side_k) ? "vbr" : type_k;
    const std::string v = (vbr.armed && vbr.side_v) ? "vbr" : type_v;
    const std::string kv = "k" + k + "-v" + v;
    if (!vbr.armed) {
        return kv;
    }
    if (vbr.unrepresented_override) {
        return ""; // effective regime unknown -> no profile -> planner refuses
    }
    // every dimension that moves the ladder's cost, in RESOLVED form; separators stay in
    // the profile sanitizer's squash class so the key renders legibly
    char nums[128];
    snprintf(nums, sizeof(nums), "%.4g %.4g %llu %.4g %.4g",
             vbr.capacity_bits, vbr.selected_bpv,
             (unsigned long long) vbr.vram_budget_bytes,
             (double) vbr.reclaim_floor_bpv, (double) vbr.reset_keep_frac);
    std::string out = kv + " vbr " + vbr.budget_mode + " " + vbr.family + " " + vbr.policy +
                      " " + vbr.schedule + " " + nums;
    if (!vbr.overrides.empty()) {
        out += " ovr " + vbr.overrides;
    }
    return out;
}

std::string common_cache_plan_calib_hw(const std::vector<std::string> & gpu_descs,
                                       int n_gpu_layers_eff, int split_mode, int main_gpu,
                                       const float * tensor_split) {
    if (gpu_descs.empty() || n_gpu_layers_eff == 0) {
        return "cpu";
    }
    std::string hw;
    for (const auto & d : gpu_descs) {
        hw += (hw.empty() ? "" : "+") + d;
    }
    hw += " ngl" + std::to_string(n_gpu_layers_eff);
    if (gpu_descs.size() > 1) {
        hw += " sm" + std::to_string(split_mode) + " mg" + std::to_string(main_gpu);
        std::string ts;
        for (size_t i = 0; i < gpu_descs.size(); i++) {
            const int pct = tensor_split ? (int) (tensor_split[i] * 100.0f) : 0;
            ts += (ts.empty() ? "" : "-") + std::to_string(pct);
        }
        hw += " ts" + ts;
    }
    return hw;
}

// does this row participate in the shadow optimum? (valid candidates only; validity
// precedes economics — the planner NEVER re-derives validity)
static bool cache_plan_row_valid(const common_cache_plan_candidate & c) {
    return c.disposition == common_cache_plan_disposition::accepted ||
           c.disposition == common_cache_plan_disposition::valid_not_chosen_cost;
}

// root-optimum membership: valid AND independently executable from the request's starting
// state — a component-only row (checkpoint exposed by a delivered host entry) is priced as
// a chain component but can never win on its own (verify-r1 finding 1)
static bool cache_plan_row_participates(const common_cache_plan_candidate & c) {
    return cache_plan_row_valid(c) && !c.component_only;
}

// the ONE place terms are written and versions stamped: restore + replay + workspace
// (B-covered) and the predicted total; transfer/eviction stay typed-unavailable until D
static void cache_plan_fill_terms(common_cache_plan_candidate & c,
                                  const common_cache_plan_calib & calib,
                                  uint64_t restore_bytes, double restore_us,
                                  uint64_t replay_tokens, double replay_us,
                                  double workspace_us) {
    const auto set_term = [&](llama_cache_acct_cost_kind kind, uint64_t raw, double us) {
        auto & term = c.cost_terms[size_t(kind)];
        term.raw               = llama_cache_acct_value::measured(raw);
        term.estimated_us      = llama_cache_acct_value::measured((uint64_t) std::llround(us));
        term.estimator_version = calib.estimator_version;
    };
    set_term(llama_cache_acct_cost_kind::restore,   restore_bytes, restore_us);
    set_term(llama_cache_acct_cost_kind::replay,    replay_tokens, replay_us);
    set_term(llama_cache_acct_cost_kind::workspace, 0,             workspace_us);
    c.predicted_total_us = llama_cache_acct_value::measured(
        (uint64_t) std::llround(restore_us + replay_us + workspace_us));
}

// estimate one non-chain row; false = a needed scalar is missing (typed-unknown lcp/bytes)
static bool cache_plan_estimate_row(common_cache_plan_candidate & c, uint64_t n_prompt,
                                    const common_cache_plan_calib & calib) {
    uint64_t restore_bytes = 0;
    uint64_t replay_tokens = 0;
    bool     has_restore   = false;

    switch (c.provider) {
        case common_cache_plan_provider::cold_replay:
            replay_tokens = n_prompt;
            break;
        case common_cache_plan_provider::live_slot:
            // state is already installed: reuse the prefix, replay the rest
            if (c.lcp_tokens.state != llama_cache_acct_known::known) {
                return false;
            }
            replay_tokens = n_prompt > c.lcp_tokens.value ? n_prompt - c.lcp_tokens.value : 0;
            break;
        case common_cache_plan_provider::host_cache_entry:
        case common_cache_plan_provider::live_context_checkpoint:
            if (c.lcp_tokens.state    != llama_cache_acct_known::known ||
                c.payload_bytes.state != llama_cache_acct_known::known) {
                return false;
            }
            restore_bytes = c.payload_bytes.value;
            replay_tokens = n_prompt > c.lcp_tokens.value ? n_prompt - c.lcp_tokens.value : 0;
            has_restore   = true;
            break;
        default:
            return false;
    }

    cache_plan_fill_terms(c, calib,
                          restore_bytes, (double) restore_bytes * calib.restore_us_per_byte,
                          replay_tokens, (double) replay_tokens * calib.replay_us_per_token,
                          has_restore ? calib.workspace_setup_us : 0.0);
    return true;
}

static common_cache_plan_planner_status cache_plan_estimate_impl(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib) {
    // trust boundary (verify-r1 finding 6): the estimator validates its calibration —
    // exact profile match against the record, a reviewed (nonzero) version, and
    // finite/nonnegative coefficients. A false match here would fabricate economics.
    if (calib.profile == nullptr || rec.calibration_profile != calib.profile ||
        calib.estimator_version == 0 ||
        !std::isfinite(calib.replay_us_per_token) || calib.replay_us_per_token < 0.0 ||
        !std::isfinite(calib.restore_us_per_byte) || calib.restore_us_per_byte < 0.0 ||
        !std::isfinite(calib.workspace_setup_us)  || calib.workspace_setup_us  < 0.0) {
        return common_cache_plan_planner_status::invalid_calibration;
    }

    // completeness over the DECLARED domain: overflow means the observed inventory lost
    // rows, and a dropped derived plan means the plan SET is incomplete — never an optimum
    // over a partial set (A1/A2). Truncation is fine: the domain is the shipped-visited
    // set by construction.
    for (const auto st : rec.inventory_states) {
        if (st == common_cache_plan_inventory_state::overflowed) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }
    if (rec.derived_plans_incomplete) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    if (rec.n_prompt_tokens.state != llama_cache_acct_known::known) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    const uint64_t n_prompt = rec.n_prompt_tokens.value;

    // pass 0 (verify-r1 finding 2): a visited candidate whose shipped phase established
    // neither validity nor invalidity (disposition unavailable — e.g. an LRU-only slot
    // whose reuse was never evaluated) makes the whole optimum unavailable. Under B-a the
    // observer may not resolve it itself; honest refusal beats silent omission.
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        if (rec.inventory[i].disposition == common_cache_plan_disposition::unavailable) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }

    // pass 1: estimate every VALID non-chain row (component_only rows included — chains
    // compose from them); any valid row the calibration cannot cover leaves the whole
    // shadow result unavailable (an optimum that silently skipped a valid candidate would
    // be a fabricated verdict)
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        auto & c = rec.inventory[i];
        if (!cache_plan_row_valid(c) || c.is_chain()) {
            continue;
        }
        if (!cache_plan_estimate_row(c, n_prompt, calib)) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
    }

    // pass 2: chain rows compose from their components — restore/workspace add, replay is
    // the DEEPEST component's (the chain replays only past its furthest frontier)
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        auto & c = rec.inventory[i];
        if (!c.is_chain() || !cache_plan_row_valid(c)) {
            continue;
        }
        uint64_t restore_bytes = 0, replay_tokens = UINT64_MAX;
        double   restore_us = 0.0, workspace_us = 0.0, replay_us = 0.0;
        bool     ok = false;
        for (const int32_t comp : c.component_ids) {
            if (comp < 0 || uint32_t(comp) >= rec.n_inventory) {
                continue;
            }
            const auto & cc = rec.inventory[size_t(comp)];
            if (cc.predicted_total_us.state != llama_cache_acct_known::known) {
                // a chain over unestimated components has no honest total
                return common_cache_plan_planner_status::incomplete_evidence;
            }
            const auto & rest = cc.cost_terms[size_t(llama_cache_acct_cost_kind::restore)];
            const auto & repl = cc.cost_terms[size_t(llama_cache_acct_cost_kind::replay)];
            const auto & work = cc.cost_terms[size_t(llama_cache_acct_cost_kind::workspace)];
            restore_bytes += rest.raw.value;
            restore_us    += (double) rest.estimated_us.value;
            workspace_us  += (double) work.estimated_us.value;
            if (repl.raw.value < replay_tokens) {
                replay_tokens = repl.raw.value;
                replay_us     = (double) repl.estimated_us.value;
            }
            ok = true;
        }
        if (!ok) {
            return common_cache_plan_planner_status::incomplete_evidence;
        }
        cache_plan_fill_terms(c, calib, restore_bytes, restore_us,
                              replay_tokens, replay_us, workspace_us);
    }

    // pass 3: minimum + tie set + planner-owned stable choice
    uint64_t min_total  = UINT64_MAX;
    bool     any        = false;
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        const auto & c = rec.inventory[i];
        if (cache_plan_row_participates(c) &&
            c.predicted_total_us.state == llama_cache_acct_known::known) {
            min_total = std::min(min_total, c.predicted_total_us.value);
            any = true;
        }
    }
    if (!any) {
        return common_cache_plan_planner_status::incomplete_evidence;
    }
    const double floor_us = std::max((double) min_total * COMMON_CACHE_PLAN_TIE_REL_FLOOR,
                                     COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US);

    rec.n_shadow_ties = 0;
    int32_t choice = -1;
    for (uint32_t i = 0; i < rec.n_inventory; i++) {
        const auto & c = rec.inventory[i];
        if (!cache_plan_row_participates(c) ||
            c.predicted_total_us.state != llama_cache_acct_known::known) {
            continue;
        }
        if ((double) c.predicted_total_us.value <= (double) min_total + floor_us) {
            rec.shadow_tie_set[rec.n_shadow_ties++] = (int32_t) i;
            // stable planner-owned key: (provider, source_id, ordinal), never the shipped choice
            const auto & best = choice >= 0 ? rec.inventory[size_t(choice)] : c;
            if (choice < 0 ||
                std::make_tuple(uint8_t(c.provider), c.source_id, (int32_t) i) <
                std::make_tuple(uint8_t(best.provider), best.source_id, choice)) {
                choice = (int32_t) i;
            }
        }
    }
    rec.shadow_choice = choice;
    return choice >= 0 ? common_cache_plan_planner_status::ok
                       : common_cache_plan_planner_status::incomplete_evidence;
}

common_cache_plan_planner_status common_cache_plan_estimate_and_choose(
        common_cache_plan_record & rec, const common_cache_plan_calib & calib) {
    // all-or-nothing is THIS function's postcondition, not a call-site convention: a
    // refusal mid-pass leaves earlier rows carrying committed estimates, which would be
    // exactly the half-estimated evidence A2/B-3 forbid — clear before returning non-ok
    const auto status = cache_plan_estimate_impl(rec, calib);
    if (status != common_cache_plan_planner_status::ok) {
        rec.clear_planner_outputs();
    }
    return status;
}
