#include "common-checkpoint-coordinator.h"

#include <algorithm>

void common_shadow_relation_evidence::reset(uint64_t new_authority_generation) {
    *this                = common_shadow_relation_evidence{};
    authority_generation = new_authority_generation;
}

bool common_shadow_relation_evidence::ensure_current(uint64_t current_authority_generation) {
    if (evidence_version != COMMON_SHADOW_EVIDENCE_VERSION ||
            authority_generation != current_authority_generation) {
        reset(current_authority_generation);
        return true;
    }
    return false;
}

void common_shadow_relation_evidence::recompute_qualified(
        const common_shadow_qualification_minima & minima) {
    // §8.4: 1,024 consecutive qualifying agreements; 64 observations per applicable
    // nontrivial class; >=16 boundary observations with actual refinement. Zero-tolerance for
    // unexplained/exception/oracle faults is structural: any fault reset this object, so
    // reaching the consecutive minimum implies a fault-free window. iSWA and host movement are
    // not applicable in current server dynamic VBR, so the applicable nontrivial classes are
    // boundary_refined, destructive, and import_refined. Trivial appends never satisfy
    // nontrivial minima.
    const bool classes_met =
        class_counts[size_t(common_checkpoint_shadow_observation::boundary_refined)] >=
                minima.per_class_observations &&
        class_counts[size_t(common_checkpoint_shadow_observation::destructive)] >=
                minima.per_class_observations &&
        class_counts[size_t(common_checkpoint_shadow_observation::import_refined)] >=
                minima.per_class_observations;
    qualified = agreement_streak >= minima.consecutive_agreements &&
                classes_met &&
                boundary_refinements >= minima.boundary_refinements;
}

common_shadow_g_verdict common_shadow_classify_evaluation(
        const common_checkpoint_shadow_evaluation & evaluation) {
    if (!evaluation.evaluated) {
        return common_shadow_g_verdict::expected_unknown;
    }
    // §6.2 verify r1 finding 1: an unavailable stable read is an oracle failure exactly like a
    // mismatch — it must reset affected evidence and can never qualify. Only disabled/not_due/
    // pass are non-failures.
    switch (evaluation.oracle_outcome) {
        case common_checkpoint_oracle_outcome::set_mismatch:
        case common_checkpoint_oracle_outcome::byte_mismatch:
        case common_checkpoint_oracle_outcome::set_and_byte_mismatch:
        case common_checkpoint_oracle_outcome::unavailable:
            return common_shadow_g_verdict::oracle_failure;
        case common_checkpoint_oracle_outcome::disabled:
        case common_checkpoint_oracle_outcome::not_due:
        case common_checkpoint_oracle_outcome::pass:
            break;
    }
    switch (evaluation.category) {
        case common_checkpoint_shadow_category::strict_accept:
            return common_shadow_g_verdict::strict;
        case common_checkpoint_shadow_category::live_rebased_shadow_accept:
            return common_shadow_g_verdict::live_rebased;
        case common_checkpoint_shadow_category::not_applicable:
        case common_checkpoint_shadow_category::generation_unknown:
            return common_shadow_g_verdict::expected_unknown;
        case common_checkpoint_shadow_category::strict_reject:
            // Verify r1 finding 2 (closed category+reason+tombstone mapping): ONLY the four
            // named §5.5 tombstone classes are availability non-events. A strict reject with
            // tombstone none (identity/frontier, pool UUID, generation drift, shape...) is an
            // unexplained disagreement — except the pinned transient reasons, which are
            // not-evaluable-now, not evidence of disagreement.
            switch (evaluation.tombstone_class) {
                case common_checkpoint_shadow_tombstone::restore_one_behind:
                case common_checkpoint_shadow_tombstone::swa_wrap:
                case common_checkpoint_shadow_tombstone::explicit_destructive_trim:
                case common_checkpoint_shadow_tombstone::dependency_seq_removed:
                    return common_shadow_g_verdict::expected_tombstone;
                case common_checkpoint_shadow_tombstone::none:
                case common_checkpoint_shadow_tombstone::unexplained:
                    break;
            }
            if (evaluation.reason == common_checkpoint_shadow_eval_reason::controller_unstable ||
                    evaluation.reason == common_checkpoint_shadow_eval_reason::unit_unstable) {
                return common_shadow_g_verdict::expected_unknown;
            }
            return common_shadow_g_verdict::unexplained;
    }
    return common_shadow_g_verdict::expected_unknown;
}

namespace {

// Scan one G axis: stop at the FIRST positionally eligible row; its verdict decides the axis.
// Never fall through to an older checkpoint (that would mint a fake agreement).
common_shadow_axis_result scan_g_axis(
        const std::vector<common_shadow_candidate_row> &                          rows,
        bool common_shadow_candidate_row::*                                       pos,
        const std::function<common_shadow_g_verdict(size_t)> &                    verdict_at) {
    common_shadow_axis_result result;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!(rows[i].*pos)) {
            continue;
        }
        result.observed    = true;
        result.verdict_row = int64_t(i);
        if (!rows[i].has_record) {
            result.verdict = common_shadow_g_verdict::expected_unknown;
            result.paused  = true;
            return result;
        }
        result.verdict = verdict_at(i);
        switch (result.verdict) {
            case common_shadow_g_verdict::strict:
                result.candidate = int64_t(i);
                return result;
            case common_shadow_g_verdict::expected_tombstone:
            case common_shadow_g_verdict::expected_unknown:
            case common_shadow_g_verdict::live_rebased:
                result.paused = true;
                return result;
            case common_shadow_g_verdict::unexplained:
            case common_shadow_g_verdict::oracle_failure:
                result.failed = true;
                return result;
        }
        return result;
    }
    // no positionally eligible row at all: not an observation (verify r1 finding 3)
    return result;
}

common_shadow_axis_result scan_l_axis(
        const std::vector<common_shadow_candidate_row> & rows,
        bool common_shadow_candidate_row::*              pos) {
    common_shadow_axis_result result;
    result.verdict = common_shadow_g_verdict::strict;
    for (size_t i = 0; i < rows.size(); ++i) {
        if ((rows[i].*pos) && rows[i].l_valid) {
            result.candidate   = int64_t(i);
            result.verdict_row = int64_t(i);
            result.observed    = true;
            return result;
        }
    }
    return result;
}

}  // namespace

common_shadow_candidates common_shadow_compute_candidates(
        const std::vector<common_shadow_candidate_row> &                   rows,
        const std::function<common_checkpoint_shadow_evaluation(size_t)> & evaluate) {
    common_shadow_candidates out;

    // memoized G verdicts: the sole evaluator runs at most once per row
    std::vector<int8_t>                  have(rows.size(), 0);
    std::vector<common_shadow_g_verdict> memo(rows.size(), common_shadow_g_verdict::expected_unknown);
    const auto verdict_at = [&](size_t i) {
        if (!have[i]) {
            have[i] = 1;
            out.g_evaluations++;
            memo[i] = common_shadow_classify_evaluation(evaluate(i));
        }
        return memo[i];
    };

    out.p_l = scan_l_axis(rows, &common_shadow_candidate_row::p_pos);
    out.f_l = scan_l_axis(rows, &common_shadow_candidate_row::f_pos);
    out.p_g = scan_g_axis(rows, &common_shadow_candidate_row::p_pos, verdict_at);
    out.f_g = scan_g_axis(rows, &common_shadow_candidate_row::f_pos, verdict_at);
    return out;
}

namespace {

common_shadow_disposition relation_disposition(
        const common_shadow_axis_result & lhs,
        const common_shadow_axis_result & rhs) {
    if (lhs.failed || rhs.failed) {
        return common_shadow_disposition::reset;
    }
    // verify r1 finding 3: an axis that made no eligible observation cannot supply either an
    // agreement or a disagreement — the relation pauses. Two absent candidates never agree.
    if (!lhs.observed || !rhs.observed) {
        return common_shadow_disposition::pause;
    }
    if (lhs.paused || rhs.paused) {
        return common_shadow_disposition::pause;
    }
    return lhs.candidate == rhs.candidate ? common_shadow_disposition::advance
                                          : common_shadow_disposition::disagree;
}

}  // namespace

common_shadow_scan_outcome common_shadow_apply_scan(
        std::array<common_shadow_relation_evidence *, 4> &        evidence,
        const common_shadow_candidates &                          candidates,
        const std::array<common_shadow_relation_observation, 4> & observations,
        uint64_t                                                  authority_generation,
        const common_shadow_qualification_minima &                minima) {
    common_shadow_scan_outcome out;

    const std::array<std::pair<const common_shadow_axis_result *, const common_shadow_axis_result *>, 4>
            relations = { {
                    { &candidates.p_l, &candidates.f_l },  // ws4_l
                    { &candidates.p_g, &candidates.f_g },  // ws4_g
                    { &candidates.p_l, &candidates.p_g },  // ws7_p
                    { &candidates.f_l, &candidates.f_g },  // ws7_f
            } };

    out.all_qualified = true;
    for (size_t r = 0; r < relations.size(); ++r) {
        auto * ev = evidence[r];
        ev->ensure_current(authority_generation);
        const auto disposition = relation_disposition(*relations[r].first, *relations[r].second);
        out.disposition[r]     = disposition;
        switch (disposition) {
            case common_shadow_disposition::advance:
                ev->agreement_streak++;
                ev->agreements_total++;
                ev->class_counts[size_t(observations[r].observation)]++;
                if (observations[r].boundary_refined) {
                    ev->boundary_refinements++;
                }
                break;
            case common_shadow_disposition::disagree:
                ev->agreement_streak = 0;
                ev->disagreements_total++;
                break;
            case common_shadow_disposition::pause:
                // availability non-event: removed from equality accounting entirely
                break;
            case common_shadow_disposition::reset:
                // scoped evidence reset; the triggering fault is counted by the server-level
                // observability counters, not inside the cleared evidence object
                ev->reset(authority_generation);
                break;
        }
        ev->recompute_qualified(minima);
        out.all_qualified = out.all_qualified && ev->qualified;
    }
    return out;
}

const char * common_shadow_relation_name(common_shadow_relation relation) {
    switch (relation) {
        case common_shadow_relation::ws4_l: return "ws4_l";
        case common_shadow_relation::ws4_g: return "ws4_g";
        case common_shadow_relation::ws7_p: return "ws7_p";
        case common_shadow_relation::ws7_f: return "ws7_f";
    }
    return "unknown";
}

const char * common_shadow_disposition_name(common_shadow_disposition disposition) {
    switch (disposition) {
        case common_shadow_disposition::advance:  return "advance";
        case common_shadow_disposition::disagree: return "disagree";
        case common_shadow_disposition::pause:    return "pause";
        case common_shadow_disposition::reset:    return "reset";
    }
    return "unknown";
}

const char * common_shadow_g_verdict_name(common_shadow_g_verdict verdict) {
    switch (verdict) {
        case common_shadow_g_verdict::strict:             return "strict";
        case common_shadow_g_verdict::expected_tombstone: return "expected_tombstone";
        case common_shadow_g_verdict::expected_unknown:   return "expected_unknown";
        case common_shadow_g_verdict::live_rebased:       return "live_rebased";
        case common_shadow_g_verdict::unexplained:        return "unexplained";
        case common_shadow_g_verdict::oracle_failure:     return "oracle_failure";
    }
    return "unknown";
}
