#include "common.h"
#include "common-checkpoint-shadow.h"
#include "common-checkpoint-coordinator.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-vbr-checkpoint.h"
#include "llama-vbr-checkpoint-compose.inc"
#include "llama-vbr-generation.h"
#include "llama-vbr-generation-oracle.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-operation.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Test-only record factory (F12): the ONLY handle-minting path outside bridge capture, compiled
// solely into this test target — production libllama exports no factory symbol and this file
// never recompiles the production bridge TU. The struct definition below mirrors
// src/llama-vbr-checkpoint.cpp token-identically; CI compares the two definitions.
struct llama_vbr_checkpoint_shadow {
    vbr_checkpoint_generation_record record;
    std::vector<vbr_checkpoint_oracle_sidecar_entry> oracle_sidecar;
};

// Test seam into the one common holder bridge TU (defined in common-checkpoint-shadow.cpp).
void common_checkpoint_shadow_attach(common_prompt_checkpoint & ckpt, llama_vbr_checkpoint_shadow * handle);

static llama_vbr_checkpoint_shadow * make_handle(vbr_checkpoint_generation_record record) {
    return new llama_vbr_checkpoint_shadow{ std::move(record), {} };
}

static vbr_checkpoint_generation_record make_complete_record() {
    vbr_checkpoint_generation_record record;
    record.status = vbr_checkpoint_generation_status::complete;
    record.identity_policy_order_digest.fill(0x5a);

    vbr_checkpoint_generation_stream stream;
    stream.stream_index              = 0;
    stream.dependency_seq_id         = 0;
    stream.computation_frontier      = 128;
    stream.captured_dependency_count = 2;
    vbr_generation_page_ref page;
    page.page_index        = 0;
    page.captured_page_gen = 7;
    page.covered_mask[0]   = (uint64_t(1) << 10) | (uint64_t(1) << 20);
    stream.pages.push_back(page);

    vbr_checkpoint_generation_controller controller;
    controller.child_id          = 0;
    controller.dependency_mode   = checkpoint_child_dependency_mode::live_guarded;
    controller.lineage_uuid      = { 0x1111, 0x2222 };
    controller.global_generation = 3;
    controller.streams.push_back(std::move(stream));

    record.controllers.push_back(std::move(controller));
    return record;
}

static common_prompt_checkpoint make_checkpoint() {
    common_prompt_checkpoint ckpt;
    ckpt.clear();
    ckpt.n_tokens  = 128;
    ckpt.id_task   = 42;
    ckpt.pos_min   = 0;
    ckpt.pos_max   = 127;
    ckpt.data_tgt  = { 1, 2, 3, 4 };
    ckpt.data_dft  = {};
    ckpt.accel.ring = { 9, 9 };
    return ckpt;
}

static bool run_holder_tests() {
    common_prompt_checkpoint a = make_checkpoint();
    if (common_checkpoint_shadow_complete(a) || a.size() != a.size_without_shadow()) {
        fprintf(stderr, "shadow-less checkpoint was not clean\n");
        return false;
    }

    common_checkpoint_shadow_attach(a, make_handle(make_complete_record()));
    if (!common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "attached complete record did not report complete\n");
        return false;
    }
    const size_t legacy_size = a.size_without_shadow();
    if (legacy_size != a.data_tgt.size() + a.data_dft.size() + a.accel.size() ||
            a.size() <= legacy_size) {
        fprintf(stderr, "size accounting: live size must be legacy + resident shadow bytes\n");
        return false;
    }

    // copy drops the shadow (fresh generation-unknown) and counts the drop; legacy fields copy
    const uint64_t drops_before = common_checkpoint_shadow_dropped_on_copy();
    common_prompt_checkpoint b(a);
    if (common_checkpoint_shadow_complete(b) || b.shadow != nullptr ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 1 ||
            b.n_tokens != a.n_tokens || b.id_task != a.id_task ||
            b.data_tgt != a.data_tgt || b.accel.ring != a.accel.ring ||
            b.size() != a.size_without_shadow()) {
        fprintf(stderr, "copy construction did not drop the shadow with counted legacy fidelity\n");
        return false;
    }
    common_prompt_checkpoint c = make_checkpoint();
    c = a;
    if (common_checkpoint_shadow_complete(c) ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 2) {
        fprintf(stderr, "copy assignment did not drop the shadow\n");
        return false;
    }
    // a still holds its shadow after being copied from
    if (!common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "copy source lost its shadow\n");
        return false;
    }

    // self-assignment is guarded: state (including the shadow) is unchanged, nothing counted
    auto * self = &a;
    a = *self;
    if (!common_checkpoint_shadow_complete(a) ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 2) {
        fprintf(stderr, "self-assignment was not a guarded no-op\n");
        return false;
    }

    // moves transfer the shadow
    common_prompt_checkpoint d(std::move(a));
    if (!common_checkpoint_shadow_complete(d) || common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "move construction did not transfer the shadow\n");
        return false;
    }
    common_prompt_checkpoint e;
    e.clear();
    e = std::move(d);
    if (!common_checkpoint_shadow_complete(e) || common_checkpoint_shadow_complete(d)) {
        fprintf(stderr, "move assignment did not transfer the shadow\n");
        return false;
    }

    // clear destroys; double-clear is a no-op
    e.clear();
    e.clear();
    if (common_checkpoint_shadow_complete(e) || e.size() != 0) {
        fprintf(stderr, "clear did not destroy the shadow\n");
        return false;
    }

    // host-cache staging parity (F3): admission prices size_without_shadow() on the LIVE list,
    // and the invalidate-first copies then really are exactly that size
    std::vector<common_prompt_checkpoint> live;
    live.push_back(make_checkpoint());
    common_checkpoint_shadow_attach(live.back(), make_handle(make_complete_record()));
    live.push_back(make_checkpoint());
    size_t priced = 0;
    for (const auto & ckpt : live) {
        priced += ckpt.size_without_shadow();
    }
    std::vector<common_prompt_checkpoint> staged_copy = live;
    size_t staged_size = 0;
    for (const auto & ckpt : staged_copy) {
        staged_size += ckpt.size();
    }
    if (priced != staged_size) {
        fprintf(stderr, "staging admission price diverged from the staged copy's size\n");
        return false;
    }

    fprintf(stderr, "holder lifecycle rows PASS\n");
    return true;
}

static bool run_equality_tests() {
    common_prompt_checkpoint a = make_checkpoint();
    common_prompt_checkpoint b = make_checkpoint();
    common_checkpoint_shadow_attach(a, make_handle(make_complete_record()));
    common_checkpoint_shadow_attach(b, make_handle(make_complete_record()));
    if (!common_checkpoint_shadow_equal(a, b) || !common_checkpoint_shadow_equal(a, a)) {
        fprintf(stderr, "equal records did not compare equal (or not reflexive)\n");
        return false;
    }

    // a single covered-mask bit difference is detected
    auto flipped = make_complete_record();
    flipped.controllers[0].streams[0].pages[0].covered_mask[0] ^= uint64_t(1) << 20;
    common_prompt_checkpoint c = make_checkpoint();
    common_checkpoint_shadow_attach(c, make_handle(std::move(flipped)));
    if (common_checkpoint_shadow_equal(a, c)) {
        fprintf(stderr, "a covered-mask bit difference was not detected\n");
        return false;
    }

    // absence/unknown is availability, never equality
    common_prompt_checkpoint none = make_checkpoint();
    auto unknown_record   = make_complete_record();
    unknown_record.status = vbr_checkpoint_generation_status::generation_unknown;
    common_prompt_checkpoint unknown = make_checkpoint();
    common_checkpoint_shadow_attach(unknown, make_handle(std::move(unknown_record)));
    if (common_checkpoint_shadow_equal(a, none) || common_checkpoint_shadow_equal(none, none) ||
            common_checkpoint_shadow_equal(a, unknown) ||
            common_checkpoint_shadow_complete(unknown)) {
        fprintf(stderr, "absent/unknown records leaked into the equality relation\n");
        return false;
    }

    // §9.3 step 5 adopt: only shadow state moves
    common_prompt_checkpoint fresh = make_checkpoint();
    common_checkpoint_shadow_attach(fresh, make_handle(make_complete_record()));
    const std::vector<uint8_t> retained_payload = c.data_tgt;
    common_checkpoint_shadow_adopt(c, fresh);
    if (!common_checkpoint_shadow_equal(a, c) || common_checkpoint_shadow_complete(fresh) ||
            c.data_tgt != retained_payload) {
        fprintf(stderr, "adopt did not swap exactly the shadow record\n");
        return false;
    }

    fprintf(stderr, "equality/adopt rows PASS\n");
    return true;
}

// §11.1 rows 11/12: the refresh byte-proof over every retained payload.
static bool run_refresh_proof_tests() {
    common_prompt_checkpoint retained = make_checkpoint();
    common_checkpoint_shadow_attach(retained, make_handle(make_complete_record()));

    std::vector<uint8_t> cur_tgt  = retained.data_tgt;
    std::vector<uint8_t> cur_ring = retained.accel.ring;
    common_checkpoint_refresh_observation obs;
    obs.tgt             = &cur_tgt;
    obs.ring            = &cur_ring;
    obs.ring_applicable = true;

    // row 11: byte-identical reproduction of every retained payload proves the refresh
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::proven) {
        fprintf(stderr, "row 11: byte-identical state did not prove the refresh\n");
        return false;
    }

    // row 12: a mismatching reproduction refuses as nondeterminism evidence
    cur_tgt[0] ^= 0xff;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_byte_mismatch) {
        fprintf(stderr, "row 12: a target byte mismatch did not refuse the refresh\n");
        return false;
    }
    cur_tgt[0] ^= 0xff;

    // a retained payload that cannot be reproduced refuses (never mutates anything)
    obs.tgt = nullptr;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "an unreproducible target payload did not refuse the refresh\n");
        return false;
    }
    obs.tgt = &cur_tgt;

    // F1: applicable accelerator payloads are part of the proof — ring mismatch refuses
    cur_ring.push_back(1);
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_byte_mismatch) {
        fprintf(stderr, "F1: an accel.ring mismatch did not refuse the refresh\n");
        return false;
    }
    cur_ring.pop_back();

    // retained ring with no reproduction path refuses
    obs.ring            = nullptr;
    obs.ring_applicable = false;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "F1: a retained ring without a save path did not refuse\n");
        return false;
    }
    obs.ring            = &cur_ring;
    obs.ring_applicable = true;

    // an applicable component with nothing retained cannot be proven against nonempty state
    std::vector<uint8_t> cur_dft = { 7 };
    obs.dft            = &cur_dft;
    obs.dft_applicable = true;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "a draft context with no retained draft payload did not refuse\n");
        return false;
    }
    cur_dft.clear();
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::proven) {
        fprintf(stderr, "an empty applicable component was not vacuous\n");
        return false;
    }

    // F1 (verify round): an APPLICABLE component whose observation is null must refuse even
    // when nothing is retained (e.g. the spec-state getter failed under can_speculate)
    obs.spec_applicable = true;
    obs.spec            = nullptr;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "F1: a null observation for an applicable empty component was proven\n");
        return false;
    }
    obs.spec_applicable = false;

    fprintf(stderr, "refresh byte-proof rows PASS\n");
    return true;
}

// ---- commit-3 §8.3–8.5 coordinator rows (pure common logic; no bridge, no model) ----

static common_checkpoint_shadow_evaluation coord_eval(
        common_checkpoint_shadow_category  category,
        common_checkpoint_shadow_tombstone tombstone = common_checkpoint_shadow_tombstone::none,
        common_checkpoint_oracle_outcome   oracle    = common_checkpoint_oracle_outcome::disabled) {
    common_checkpoint_shadow_evaluation e;
    e.evaluated       = true;
    e.category        = category;
    e.strict          = category == common_checkpoint_shadow_category::strict_accept;
    e.tombstone_class = tombstone;
    e.oracle_outcome  = oracle;
    return e;
}

static bool run_coordinator_candidate_tests() {
    using row = common_shadow_candidate_row;

    // asymmetric matrix: newest row (0) satisfies only F, row 1 satisfies only P; row 0 has a
    // strict record, row 1 has a strict record too — all four candidates land distinctly and
    // the sole evaluator runs at most once per row
    {
        std::vector<row> rows = {
            { /*p_pos=*/false, /*f_pos=*/true,  /*l_valid=*/true, /*has_record=*/true },
            { /*p_pos=*/true,  /*f_pos=*/false, /*l_valid=*/true, /*has_record=*/true },
        };
        std::vector<int> calls(rows.size(), 0);
        const auto candidates = common_shadow_compute_candidates(rows, [&](size_t i) {
            calls[i]++;
            return coord_eval(common_checkpoint_shadow_category::strict_accept);
        });
        if (candidates.p_l.candidate != 1 || candidates.f_l.candidate != 0 ||
                candidates.p_g.candidate != 1 || candidates.f_g.candidate != 0 ||
                candidates.g_evaluations != 2 || calls[0] != 1 || calls[1] != 1) {
            fprintf(stderr, "coordinator: asymmetric four-candidate matrix failed\n");
            return false;
        }
    }

    // expected tombstone at the FIRST eligible row pauses the axis — it never falls through to
    // the older strict row (that would mint a fake agreement)
    {
        std::vector<row> rows = {
            { true, true, true, true },   // tombstoned
            { true, true, true, true },   // strict, must NOT be reached
        };
        std::vector<int> calls(rows.size(), 0);
        const auto candidates = common_shadow_compute_candidates(rows, [&](size_t i) {
            calls[i]++;
            return i == 0 ? coord_eval(common_checkpoint_shadow_category::strict_reject,
                                       common_checkpoint_shadow_tombstone::swa_wrap)
                          : coord_eval(common_checkpoint_shadow_category::strict_accept);
        });
        if (!candidates.p_g.paused || !candidates.f_g.paused ||
                candidates.p_g.candidate != -1 || calls[1] != 0 ||
                candidates.p_g.verdict != common_shadow_g_verdict::expected_tombstone) {
            fprintf(stderr, "coordinator: tombstone did not pause without fall-through\n");
            return false;
        }
    }

    // a missing record at the first eligible row pauses as expected_unknown WITHOUT invoking
    // the evaluator at all
    {
        std::vector<row> rows = { { true, true, true, /*has_record=*/false } };
        const auto candidates = common_shadow_compute_candidates(rows, [&](size_t) {
            fprintf(stderr, "coordinator: evaluator invoked for a record-less row\n");
            return coord_eval(common_checkpoint_shadow_category::strict_accept);
        });
        if (!candidates.p_g.paused || candidates.g_evaluations != 0 ||
                candidates.p_g.verdict != common_shadow_g_verdict::expected_unknown) {
            fprintf(stderr, "coordinator: record-less row was not an unknown pause\n");
            return false;
        }
    }

    // classification mapping (verify r1 findings 1+2): the closed category+reason+tombstone
    // table — only the four named §5.5 classes are expected tombstones; strict rejects with
    // tombstone none (identity, generation drift...) are unexplained; pinned transient reasons
    // are expected-unknown; ALL FOUR oracle non-pass outcomes incl. unavailable are failures
    {
        auto identity_reject = coord_eval(common_checkpoint_shadow_category::strict_reject);
        identity_reject.reason = common_checkpoint_shadow_eval_reason::identity_or_frontier;
        auto unstable_reject = coord_eval(common_checkpoint_shadow_category::strict_reject);
        unstable_reject.reason = common_checkpoint_shadow_eval_reason::controller_unstable;
        const common_checkpoint_shadow_tombstone expected_classes[4] = {
            common_checkpoint_shadow_tombstone::restore_one_behind,
            common_checkpoint_shadow_tombstone::swa_wrap,
            common_checkpoint_shadow_tombstone::explicit_destructive_trim,
            common_checkpoint_shadow_tombstone::dependency_seq_removed,
        };
        for (const auto cls : expected_classes) {
            if (common_shadow_classify_evaluation(coord_eval(
                        common_checkpoint_shadow_category::strict_reject, cls)) !=
                    common_shadow_g_verdict::expected_tombstone) {
                fprintf(stderr, "coordinator: named tombstone class was not an expected pause\n");
                return false;
            }
        }
        if (common_shadow_classify_evaluation(identity_reject) !=
                common_shadow_g_verdict::unexplained ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::strict_reject)) !=
                common_shadow_g_verdict::unexplained ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::strict_reject,
                    common_checkpoint_shadow_tombstone::unexplained)) !=
                common_shadow_g_verdict::unexplained ||
            common_shadow_classify_evaluation(unstable_reject) !=
                common_shadow_g_verdict::expected_unknown ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::live_rebased_shadow_accept)) !=
                common_shadow_g_verdict::live_rebased ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::generation_unknown)) !=
                common_shadow_g_verdict::expected_unknown ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::strict_accept,
                    common_checkpoint_shadow_tombstone::none,
                    common_checkpoint_oracle_outcome::byte_mismatch)) !=
                common_shadow_g_verdict::oracle_failure ||
            common_shadow_classify_evaluation(coord_eval(
                    common_checkpoint_shadow_category::strict_accept,
                    common_checkpoint_shadow_tombstone::none,
                    common_checkpoint_oracle_outcome::unavailable)) !=
                common_shadow_g_verdict::oracle_failure ||
            common_shadow_classify_evaluation(common_checkpoint_shadow_evaluation{}) !=
                common_shadow_g_verdict::expected_unknown) {
            fprintf(stderr, "coordinator: evaluation classification mapping diverged\n");
            return false;
        }
    }

    // verify r1 finding 3: a scan with nothing positionally eligible makes no observation —
    // no axis observes, so no relation can mint a vacuous agreement
    {
        std::vector<row> rows = { { false, false, true, true } };
        const auto candidates = common_shadow_compute_candidates(rows, [&](size_t) {
            fprintf(stderr, "coordinator: evaluator invoked with nothing eligible\n");
            return coord_eval(common_checkpoint_shadow_category::strict_accept);
        });
        if (candidates.p_l.observed || candidates.f_g.observed || candidates.g_evaluations != 0) {
            fprintf(stderr, "coordinator: ineligible scan claimed an observation\n");
            return false;
        }
        common_shadow_relation_evidence e0, e1, e2, e3;
        std::array<common_shadow_relation_evidence *, 4> evidence = { &e0, &e1, &e2, &e3 };
        const std::array<common_shadow_relation_observation, 4> no_obs = {};
        const auto outcome = common_shadow_apply_scan(
            evidence, candidates, no_obs, 0, common_shadow_qualification_minima{ 1, 0, 0 });
        for (const auto d : outcome.disposition) {
            if (d != common_shadow_disposition::pause) {
                fprintf(stderr, "coordinator: empty scan advanced a relation\n");
                return false;
            }
        }
        if (e0.agreement_streak != 0 || e3.agreements_total != 0 || outcome.all_qualified) {
            fprintf(stderr, "coordinator: empty scan accumulated vacuous evidence\n");
            return false;
        }
    }

    fprintf(stderr, "coordinator candidate rows PASS\n");
    return true;
}

static bool run_coordinator_evidence_tests() {
    // production defaults are the §8.4 minima
    {
        const common_shadow_qualification_minima defaults;
        if (defaults.consecutive_agreements != 1024 || defaults.per_class_observations != 64 ||
                defaults.boundary_refinements != 16) {
            fprintf(stderr, "coordinator: production minima diverged from §8.4\n");
            return false;
        }
    }

    const common_shadow_qualification_minima tiny{ /*consecutive=*/3, /*per_class=*/1, /*boundary=*/1 };
    common_shadow_relation_evidence ws4_l, ws4_g, ws7_p, ws7_f;
    std::array<common_shadow_relation_evidence *, 4> evidence = { &ws4_l, &ws4_g, &ws7_p, &ws7_f };

    const auto agreeing_candidates = [] {
        common_shadow_candidates c;
        c.p_l.candidate = c.f_l.candidate = c.p_g.candidate = c.f_g.candidate = 0;
        c.p_l.observed = c.f_l.observed = c.p_g.observed = c.f_g.observed = true;
        c.p_g.verdict = c.f_g.verdict = common_shadow_g_verdict::strict;
        return c;
    };
    const auto uniform_obs = [](common_checkpoint_shadow_observation cls, bool boundary) {
        std::array<common_shadow_relation_observation, 4> obs;
        obs.fill({ cls, boundary });
        return obs;
    };

    // three agreeing scans across the three applicable nontrivial classes reach the tiny
    // minima on all four relations
    {
        const common_checkpoint_shadow_observation classes[3] = {
            common_checkpoint_shadow_observation::boundary_refined,
            common_checkpoint_shadow_observation::destructive,
            common_checkpoint_shadow_observation::import_refined,
        };
        common_shadow_scan_outcome outcome;
        for (int i = 0; i < 3; ++i) {
            outcome = common_shadow_apply_scan(
                evidence, agreeing_candidates(),
                uniform_obs(classes[i],
                            classes[i] == common_checkpoint_shadow_observation::boundary_refined),
                /*authority_generation=*/0, tiny);
        }
        if (!outcome.all_qualified || !ws4_l.qualified || ws7_f.agreement_streak != 3 ||
                ws4_g.boundary_refinements != 1) {
            fprintf(stderr, "coordinator: agreeing ladder did not reach tiny minima\n");
            return false;
        }
    }

    // verify r1 finding 6: relation-local classes — a destructive P observation and an
    // import-refined F observation credit their own relations, never each other's
    {
        common_shadow_relation_evidence a0, a1, a2, a3;
        std::array<common_shadow_relation_evidence *, 4> asym = { &a0, &a1, &a2, &a3 };
        std::array<common_shadow_relation_observation, 4> obs = {};
        obs[2] = { common_checkpoint_shadow_observation::destructive, false };      // ws7_p <- P/G
        obs[3] = { common_checkpoint_shadow_observation::import_refined, false };   // ws7_f <- F/G
        common_shadow_apply_scan(asym, agreeing_candidates(), obs, 0, tiny);
        const auto destructive = size_t(common_checkpoint_shadow_observation::destructive);
        const auto import_ref  = size_t(common_checkpoint_shadow_observation::import_refined);
        if (a2.class_counts[destructive] != 1 || a2.class_counts[import_ref] != 0 ||
                a3.class_counts[import_ref] != 1 || a3.class_counts[destructive] != 0 ||
                a0.class_counts[destructive] != 0) {
            fprintf(stderr, "coordinator: observation class leaked across relations\n");
            return false;
        }
    }

    // a paused scan is an availability non-event: nothing advances, nothing resets
    {
        auto paused        = agreeing_candidates();
        paused.f_g.paused  = true;
        paused.f_g.candidate = -1;
        paused.f_g.verdict = common_shadow_g_verdict::expected_tombstone;
        const auto outcome = common_shadow_apply_scan(
            evidence, paused,
            uniform_obs(common_checkpoint_shadow_observation::trivial_append, false), 0, tiny);
        // relations touching F/G pause; WS-4/L and WS-7/P still advance
        if (outcome.disposition[1] != common_shadow_disposition::pause ||
                outcome.disposition[3] != common_shadow_disposition::pause ||
                outcome.disposition[0] != common_shadow_disposition::advance ||
                ws4_g.agreement_streak != 3 || ws7_f.agreement_streak != 3 ||
                ws4_l.agreement_streak != 4) {
            fprintf(stderr, "coordinator: pause was not an availability non-event\n");
            return false;
        }
    }

    // an unexplained axis resets the relations that touch it and disqualifies them
    {
        auto failed        = agreeing_candidates();
        failed.p_g.failed  = true;
        failed.p_g.candidate = -1;
        failed.p_g.verdict = common_shadow_g_verdict::unexplained;
        const auto outcome = common_shadow_apply_scan(
            evidence, failed,
            uniform_obs(common_checkpoint_shadow_observation::trivial_append, false), 0, tiny);
        if (outcome.disposition[1] != common_shadow_disposition::reset ||
                outcome.disposition[2] != common_shadow_disposition::reset ||
                ws4_g.agreement_streak != 0 || ws4_g.qualified || ws7_p.agreements_total != 0 ||
                ws4_l.agreement_streak != 5) {
            fprintf(stderr, "coordinator: unexplained axis did not reset its relations\n");
            return false;
        }
    }

    // verify r1 finding 1: an oracle-unavailable axis is a failure exactly like a mismatch —
    // relations touching it reset and cannot remain qualified
    {
        common_shadow_relation_evidence q0, q1, q2, q3;
        std::array<common_shadow_relation_evidence *, 4> qual = { &q0, &q1, &q2, &q3 };
        for (int i = 0; i < 3; ++i) {
            common_shadow_apply_scan(
                qual, agreeing_candidates(),
                uniform_obs(common_checkpoint_shadow_observation::boundary_refined, true), 0,
                common_shadow_qualification_minima{ 3, 0, 1 });
        }
        if (!q1.qualified) {
            fprintf(stderr, "coordinator: oracle row setup failed to qualify\n");
            return false;
        }
        auto unavailable = agreeing_candidates();
        unavailable.f_g.failed    = true;
        unavailable.f_g.candidate = -1;
        unavailable.f_g.verdict   = common_shadow_g_verdict::oracle_failure;
        const auto outcome = common_shadow_apply_scan(
            qual, unavailable,
            uniform_obs(common_checkpoint_shadow_observation::trivial_append, false), 0,
            common_shadow_qualification_minima{ 3, 0, 1 });
        if (outcome.disposition[1] != common_shadow_disposition::reset ||
                outcome.disposition[3] != common_shadow_disposition::reset ||
                q1.qualified || q3.qualified || q1.agreement_streak != 0 ||
                outcome.all_qualified) {
            fprintf(stderr, "coordinator: oracle unavailability did not reset/disqualify\n");
            return false;
        }
    }

    // ordinary disagreement: streak resets, totals persist, no evidence clearing
    {
        auto disagreeing = agreeing_candidates();
        disagreeing.f_l.candidate = 1;
        const auto outcome = common_shadow_apply_scan(
            evidence, disagreeing,
            uniform_obs(common_checkpoint_shadow_observation::trivial_append, false), 0, tiny);
        if (outcome.disposition[0] != common_shadow_disposition::disagree ||
                ws4_l.agreement_streak != 0 || ws4_l.agreements_total != 5 ||
                ws4_l.disagreements_total != 1 || ws4_l.qualified) {
            fprintf(stderr, "coordinator: ordinary disagreement accounting diverged\n");
            return false;
        }
    }

    // Q5.1 version clearing: stale evidence_version or authority_generation lazily clears
    // before the next observation — discarded, never upgraded
    {
        common_shadow_relation_evidence stale;
        stale.agreements_total = 7;
        stale.evidence_version = COMMON_SHADOW_EVIDENCE_VERSION + 1;
        if (!stale.ensure_current(0) || stale.agreements_total != 0 ||
                stale.evidence_version != COMMON_SHADOW_EVIDENCE_VERSION) {
            fprintf(stderr, "coordinator: stale evidence_version did not clear\n");
            return false;
        }
        stale.agreements_total = 9;
        if (!stale.ensure_current(2) || stale.agreements_total != 0 ||
                stale.authority_generation != 2 || stale.ensure_current(2)) {
            fprintf(stderr, "coordinator: authority_generation mismatch did not clear\n");
            return false;
        }
    }

    fprintf(stderr, "coordinator evidence rows PASS\n");
    return true;
}

static bool run_bridge_boundary_tests() {
    // null memory / null frontier fail closed with a closed reason, never a crash
    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(nullptr, 0, nullptr, &result);
    if (result.handle != nullptr ||
            result.reason != vbr_checkpoint_capture_reason::invalid_arguments ||
            result.reset_scope != vbr_checkpoint_reset_scope::capturing_slot) {
        fprintf(stderr, "null capture arguments were not refused\n");
        return false;
    }
    llama_vbr_checkpoint_shadow_capture(nullptr, 0, nullptr, nullptr);

    if (llama_vbr_checkpoint_shadow_status(nullptr) !=
            vbr_checkpoint_generation_status::generation_unknown ||
            llama_vbr_checkpoint_shadow_size(nullptr) != 0 ||
            llama_vbr_checkpoint_shadow_equal(nullptr, nullptr)) {
        fprintf(stderr, "null handles were not generation-unknown/zero/unequal\n");
        return false;
    }
    llama_vbr_checkpoint_shadow_free(nullptr);

    llama_vbr_checkpoint_shadow_evaluation evaluation;
    llama_vbr_checkpoint_shadow_evaluate(nullptr, nullptr, 0, nullptr, &evaluation);
    if (evaluation.evaluator_invocations != 0 ||
            evaluation.category != vbr_checkpoint_shadow_category::generation_unknown ||
            evaluation.reason != vbr_checkpoint_shadow_reason::record_unknown) {
        fprintf(stderr, "null G-only evaluation was not closed generation-unknown\n");
        return false;
    }
    llama_vbr_checkpoint_shadow_evaluate(nullptr, nullptr, 0, nullptr, nullptr);

    for (const auto reason : {
                 vbr_checkpoint_capture_reason::ok,
                 vbr_checkpoint_capture_reason::not_applicable,
                 vbr_checkpoint_capture_reason::invalid_arguments,
                 vbr_checkpoint_capture_reason::unarmed_live_covered,
                 vbr_checkpoint_capture_reason::child_capture_failed,
                 vbr_checkpoint_capture_reason::oracle_mismatch,
                 vbr_checkpoint_capture_reason::internal_error,
                 vbr_checkpoint_capture_reason::controller_unavailable,
         }) {
        const char * name = llama_vbr_checkpoint_shadow_reason_name(reason);
        if (name == nullptr || strlen(name) == 0) {
            fprintf(stderr, "capture reason without a name\n");
            return false;
        }
    }

    fprintf(stderr, "bridge boundary rows PASS\n");
    return true;
}

// --- C2-P9 rung-1 rows (f)-(h): real armed iSWA composite capture (GPU fixture) ---------------
// Run with a gemma-4 iSWA model argument on the dorei gate box; the no-arg ctest invocation
// runs only the CPU rows above.

// Same-named friend as the epoch test (each test binary defines its own shim): tracker access
// for row (g)'s !stable() and shadow-unavailable refusals.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }
    static vbr_generation_tracker * tracker_mut(llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_mut();
    }
};

static bool gpu_decode_at(llama_context * ctx, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, 1, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (ok) {
        // decode is asynchronous; the fixture inspects/mutates memory next, so fence to the
        // completed-state boundary (this is also where submitted evidence commits)
        llama_synchronize(ctx);
    }
    return ok;
}

static vbr_checkpoint_frontier_fields gpu_frontier(int64_t n_past) {
    static const std::string exec_id    = "shadow-fixture-exec";
    static const std::string adapter_id = "shadow-fixture-adapter";
    static const std::string media_id   = "shadow-fixture-media";
    vbr_checkpoint_frontier_fields frontier;
    frontier.execution_identity          = exec_id.c_str();
    frontier.execution_identity_len      = exec_id.size();
    frontier.adapter_config_identity     = adapter_id.c_str();
    frontier.adapter_config_identity_len = adapter_id.size();
    frontier.media_content_identity      = media_id.c_str();
    frontier.media_content_identity_len  = media_id.size();
    frontier.sequence_epoch = 1;
    frontier.token_count    = n_past;
    frontier.next_position  = (llama_pos) n_past;
    return frontier;
}

static llama_vbr_checkpoint_shadow * gpu_capture(llama_memory_t mem, int64_t n_past,
                                                 vbr_checkpoint_capture_reason & reason,
                                                 vbr_checkpoint_reset_scope * reset_scope = nullptr) {
    const auto frontier = gpu_frontier(n_past);
    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(mem, 0, &frontier, &result);
    reason = result.reason;
    if (reset_scope != nullptr) {
        *reset_scope = result.reset_scope;
    }
    return result.handle;
}

static int run_gpu_fixture_rows(const char * model_path) {
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                 = 128;
    cparams.n_batch               = 32;
    cparams.n_ubatch              = 32;
    cparams.n_seq_max             = 1;
    cparams.n_threads             = 2;
    cparams.n_threads_batch       = 2;
    cparams.type_k                = GGML_TYPE_F16;
    cparams.type_v                = GGML_TYPE_F16;
    cparams.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.vbr_dynamic           = true;
    cparams.vbr_budget_explicit   = true;
    cparams.vbr_vram_budget_bytes = 64ull * 1024 * 1024;

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create CUDA VBR context\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx.get());

    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        base = iswa->get_base();
        swa  = iswa->get_swa();
    } else if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        base = hybrid->get_mem_attn()->get_base();
        swa  = hybrid->get_mem_attn()->get_swa();
    } else {
        fprintf(stderr, "fixture did not create an iSWA attention cache\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(base) ||
            !llama_kv_cache_vbr_epoch_test::active(swa)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for both children\n");
        return 0;
    }

    int64_t n_past = 0;
    for (; n_past < 4; ++n_past) {
        if (!gpu_decode_at(ctx.get(), (llama_pos) n_past)) {
            fprintf(stderr, "seed decode failed at pos %" PRId64 "\n", n_past);
            return 1;
        }
    }

    // row (f): composite bridge capture on the real armed iSWA pair — complete record, child
    // order [base, swa], serializer-derived modes, armed pool identities, shared digest helper
    vbr_checkpoint_capture_reason reason;
    llama_vbr_checkpoint_shadow * first = gpu_capture(mem, n_past, reason);
    if (first == nullptr || reason != vbr_checkpoint_capture_reason::ok) {
        fprintf(stderr, "row f: armed iSWA capture failed (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    {
        const auto & record = first->record;
        if (record.status != vbr_checkpoint_generation_status::complete ||
                record.controllers.size() != 2 ||
                record.controllers[0].child_id != 0 ||
                record.controllers[0].dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
                record.controllers[1].child_id != 1 ||
                record.controllers[1].dependency_mode != checkpoint_child_dependency_mode::payload_complete ||
                !record.controllers[1].streams.empty()) {
            fprintf(stderr, "row f: composite child order/modes did not match [base=live_guarded, swa=payload_complete]\n");
            return 1;
        }
        if (record.controllers[0].lineage_uuid != base->vbr_lineage_id() ||
                record.controllers[1].lineage_uuid != swa->vbr_lineage_id() ||
                record.controllers[1].lineage_uuid.hi == 0) {
            fprintf(stderr, "row f: armed children did not record their exact lineages\n");
            return 1;
        }
        std::vector<vbr_checkpoint_child_policy> policy;
        for (const auto & controller : record.controllers) {
            policy.push_back({ controller.child_id, controller.dependency_mode, controller.lineage_uuid });
        }
        if (vbr_checkpoint_identity_digest(gpu_frontier(n_past), policy) !=
                record.identity_policy_order_digest) {
            fprintf(stderr, "row f: record digest diverged from the shared digest helper\n");
            return 1;
        }
    }

    // row (h): no-op recapture is equal; a real mutation (frontier advance, and a same-frontier
    // destructive trim) makes captures unequal
    llama_vbr_checkpoint_shadow * again = gpu_capture(mem, n_past, reason);
    if (again == nullptr || !llama_vbr_checkpoint_shadow_equal(first, again)) {
        fprintf(stderr, "row h: no-op recapture was not equal\n");
        return 1;
    }
    if (!gpu_decode_at(ctx.get(), (llama_pos) n_past)) {
        fprintf(stderr, "row h: mutation decode failed\n");
        return 1;
    }
    ++n_past;
    llama_vbr_checkpoint_shadow * moved = gpu_capture(mem, n_past, reason);
    if (moved == nullptr || llama_vbr_checkpoint_shadow_equal(first, moved)) {
        fprintf(stderr, "row h: a frontier-advancing decode did not change the record\n");
        return 1;
    }
    if (!llama_memory_seq_rm(mem, 0, (llama_pos) (n_past - 1), -1)) {
        fprintf(stderr, "row h: tail trim was rejected\n");
        return 1;
    }
    --n_past;
    llama_vbr_checkpoint_shadow * trimmed = gpu_capture(mem, n_past, reason);
    if (trimmed == nullptr || llama_vbr_checkpoint_shadow_equal(first, trimmed)) {
        fprintf(stderr, "row h: a same-shape destructive trim did not change the record\n");
        return 1;
    }

    // row (g): capture refuses while a child is mid-mutation (!stable()) and while its shadow
    // is latched unavailable
    {
        auto * tracker = llama_kv_cache_vbr_epoch_test::tracker_mut(base);
        if (tracker == nullptr) {
            fprintf(stderr, "row g: base tracker unavailable\n");
            return 1;
        }
        vbr_scoped_operation op(vbr_mutation_binding(
                vbr_operation_kind::decode, 0, 0,
                std::numeric_limits<llama_pos>::max(),
                vbr_operation_class::ordinary_decode,
                tracker->runtime_instance()));
        if (!op.id()) {
            fprintf(stderr, "row g: test operation failed to register\n");
            return 1;
        }
        {
            auto event = tracker->begin_event(
                    vbr_mutation_registrant::apply_ubatch_append,
                    vbr_operation_class::ordinary_decode,
                    0,
                    vbr_generation_stamp_kind::dependency,
                    op.id());
            if (!event) {
                fprintf(stderr, "row g: could not open a mid-mutation event\n");
                return 1;
            }
            vbr_checkpoint_reset_scope reset_scope;
            llama_vbr_checkpoint_shadow * refused =
                gpu_capture(mem, n_past, reason, &reset_scope);
            if (refused != nullptr ||
                    reason != vbr_checkpoint_capture_reason::child_capture_failed ||
                    reset_scope != vbr_checkpoint_reset_scope::capturing_slot) {
                fprintf(stderr, "row g: capture was not refused mid-mutation (reason=%s)\n",
                        llama_vbr_checkpoint_shadow_reason_name(reason));
                llama_vbr_checkpoint_shadow_free(refused);
                return 1;
            }
            if (!event.finish()) {
                fprintf(stderr, "row g: mid-mutation event did not close\n");
                return 1;
            }
        }
        tracker->set_shadow_unavailable();
        vbr_checkpoint_reset_scope reset_scope;
        llama_vbr_checkpoint_shadow * latched = gpu_capture(mem, n_past, reason, &reset_scope);
        if (latched != nullptr ||
                reason != vbr_checkpoint_capture_reason::controller_unavailable ||
                reset_scope != vbr_checkpoint_reset_scope::global) {
            fprintf(stderr, "row g: capture was not refused while shadow-unavailable (reason=%s)\n",
                    llama_vbr_checkpoint_shadow_reason_name(reason));
            llama_vbr_checkpoint_shadow_free(latched);
            return 1;
        }
    }

    llama_vbr_checkpoint_shadow_free(first);
    llama_vbr_checkpoint_shadow_free(again);
    llama_vbr_checkpoint_shadow_free(moved);
    llama_vbr_checkpoint_shadow_free(trimmed);

    printf("GPU armed-iSWA composite rows (f)-(h) PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (!run_holder_tests() || !run_equality_tests() || !run_refresh_proof_tests() ||
            !run_coordinator_candidate_tests() || !run_coordinator_evidence_tests() ||
            !run_bridge_boundary_tests()) {
        return 1;
    }
    printf("checkpoint shadow lifecycle PASS\n");
    if (argc == 2) {
        return run_gpu_fixture_rows(argv[1]);
    }
    return 0;
}
