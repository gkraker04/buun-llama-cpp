#include "server-cache-plan-authority.h"

#include "common-cache-plan-estimate.h"
#include "../../src/llama-sha256.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool decode_sha256_hex(
        const std::string & value,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    if (value.size() != 64) return false;
    const auto nibble = [](char c) noexcept -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int hi = nibble(value[2*i]);
        const int lo = nibble(value[2*i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}

} // namespace

server_cache_plan_local_authority_latch::
server_cache_plan_local_authority_latch(
        server_cache_plan_local_authority_latch && other) noexcept {
    move_from(other);
}

server_cache_plan_local_authority_latch &
server_cache_plan_local_authority_latch::operator=(
        server_cache_plan_local_authority_latch && other) noexcept {
    if (this != &other) {
        reset();
        move_from(other);
    }
    return *this;
}

void server_cache_plan_local_authority_latch::reset() noexcept {
    state_ = common_cache_optimizer_authority_state::not_attempted;
    certified_once_ = false;
    coefficient_source_ = common_cache_optimizer_profile_source::none;
    candidate_ = -1;
    boot_claim_ordinal_ = 0;
    profile_generation_ = 0;
    authority_currency_serial_ = 0;
    instance_generation_digest_ = {};
    procedure_version_ = 0;
}

void server_cache_plan_local_authority_latch::move_from(
        server_cache_plan_local_authority_latch & other) noexcept {
    state_ = other.state_;
    certified_once_ = other.certified_once_;
    coefficient_source_ = other.coefficient_source_;
    candidate_ = other.candidate_;
    boot_claim_ordinal_ = other.boot_claim_ordinal_;
    profile_generation_ = other.profile_generation_;
    authority_currency_serial_ = other.authority_currency_serial_;
    instance_generation_digest_ = other.instance_generation_digest_;
    procedure_version_ = other.procedure_version_;
    other.reset();
}

void server_cache_plan_local_authority_latch::clear() noexcept {
    reset();
}

bool server_cache_plan_local_authority_latch::prequalify(
        const common_cache_optimizer_authority_receipt & receipt) noexcept {
    if (state_ != common_cache_optimizer_authority_state::not_attempted ||
        receipt.state != common_cache_optimizer_authority_state::prequalified) {
        return false;
    }
    state_ = common_cache_optimizer_authority_state::prequalified;
    return true;
}

bool server_cache_plan_local_authority_latch::certify(
        const common_cache_optimizer_authority_receipt & receipt) noexcept {
    if (state_ != common_cache_optimizer_authority_state::prequalified ||
        receipt.state != common_cache_optimizer_authority_state::certified ||
        !receipt.certified_once ||
        receipt.coefficient_source !=
            common_cache_optimizer_profile_source::local_fit ||
        receipt.candidate < 0 ||
        receipt.boot_claim_ordinal.state != llama_cache_acct_known::known ||
        receipt.profile_generation.state != llama_cache_acct_known::known ||
        receipt.authority_currency_serial.state !=
            llama_cache_acct_known::known ||
        receipt.procedure_version == 0 ||
        !decode_sha256_hex(
            receipt.instance_generation_digest,
            instance_generation_digest_)) {
        return false;
    }
    state_ = common_cache_optimizer_authority_state::certified;
    certified_once_ = true;
    coefficient_source_ = receipt.coefficient_source;
    candidate_ = receipt.candidate;
    boot_claim_ordinal_ = receipt.boot_claim_ordinal.value;
    profile_generation_ = receipt.profile_generation.value;
    authority_currency_serial_ = receipt.authority_currency_serial.value;
    procedure_version_ = receipt.procedure_version;
    return true;
}

bool server_cache_plan_local_authority_latch::certified_for(
        const common_cache_optimizer_authority_receipt & receipt) const noexcept {
    std::array<uint8_t, 32> digest = {};
    return state_ == common_cache_optimizer_authority_state::certified &&
        certified_once_ && receipt.certified_once &&
        receipt.state == common_cache_optimizer_authority_state::certified &&
        receipt.coefficient_source == coefficient_source_ &&
        receipt.candidate == candidate_ &&
        receipt.boot_claim_ordinal.state == llama_cache_acct_known::known &&
        receipt.boot_claim_ordinal.value == boot_claim_ordinal_ &&
        receipt.profile_generation.state == llama_cache_acct_known::known &&
        receipt.profile_generation.value == profile_generation_ &&
        receipt.authority_currency_serial.state ==
            llama_cache_acct_known::known &&
        receipt.authority_currency_serial.value == authority_currency_serial_ &&
        receipt.procedure_version == procedure_version_ &&
        decode_sha256_hex(receipt.instance_generation_digest, digest) &&
        digest == instance_generation_digest_;
}

bool server_cache_plan_local_authority_latch::fallback(
        common_cache_optimizer_authority_receipt & receipt,
        common_cache_optimizer_fallback_reason reason) noexcept {
    if (state_ == common_cache_optimizer_authority_state::fallback ||
        state_ == common_cache_optimizer_authority_state::executed) {
        return false;
    }
    state_ = common_cache_optimizer_authority_state::fallback;
    receipt.state = common_cache_optimizer_authority_state::fallback;
    receipt.certified_once = certified_once_;
    receipt.reason = reason;
    return true;
}

bool server_cache_plan_local_authority_latch::execute(
        common_cache_optimizer_authority_receipt & receipt) noexcept {
    if (!certified_for(receipt)) return false;
    state_ = common_cache_optimizer_authority_state::executed;
    receipt.state = common_cache_optimizer_authority_state::executed;
    receipt.certified_once = true;
    receipt.reason = common_cache_optimizer_fallback_reason::none;
    return true;
}

bool server_cache_plan_authority::set_profile_display_salt(
        const std::array<uint8_t, 32> & salt) noexcept {
    if (profile_display_salt_ready_) {
        return salt == profile_display_salt_;
    }
    profile_display_salt_ = salt;
    profile_display_entries_ = {};
    profile_display_salt_ready_ = true;
    return true;
}

bool server_cache_plan_authority::profile_display_label(
        const std::array<uint8_t, 32> & execution_root,
        std::string & out) noexcept {
    out.clear();
    if (!profile_display_salt_ready_) return false;
    try {
        static constexpr char domain[] = "llama.cache.profile-display.v1\0";
        static constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyz234567";
        llama_sha256 hash;
        hash.update(domain, sizeof(domain) - 1);
        hash.update(profile_display_salt_.data(), profile_display_salt_.size());
        hash.update(execution_root.data(), execution_root.size());
        const auto digest = hash.finish();

        profile_display_entry * entry = nullptr;
        profile_display_entry * free = nullptr;
        for (auto & item : profile_display_entries_) {
            if (!item.used) {
                if (!free) free = &item;
                continue;
            }
            if (item.execution_root == execution_root) {
                entry = &item;
                break;
            }
            if (std::equal(digest.begin(), digest.begin() + 12,
                           item.digest.begin())) {
                item.collision = true;
            }
        }
        if (!entry) {
            if (!free) return false;
            free->execution_root = execution_root;
            std::copy_n(digest.begin(), free->digest.size(), free->digest.begin());
            free->used = true;
            for (const auto & item : profile_display_entries_) {
                if (&item != free && item.used &&
                    std::equal(digest.begin(), digest.begin() + 12,
                               item.digest.begin())) {
                    free->collision = true;
                    break;
                }
            }
            entry = free;
        }

        const size_t input_bytes = entry->collision ? 16 : 12;
        out.reserve(6 + (input_bytes*8 + 4)/5);
        out = "local-";
        uint32_t bits = 0;
        unsigned n_bits = 0;
        for (size_t i = 0; i < input_bytes; ++i) {
            bits = (bits << 8) | entry->digest[i];
            n_bits += 8;
            while (n_bits >= 5) {
                n_bits -= 5;
                out.push_back(alphabet[(bits >> n_bits) & 31]);
            }
        }
        if (n_bits != 0) {
            out.push_back(alphabet[(bits << (5 - n_bits)) & 31]);
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

int32_t server_cache_plan_host_source(
        const common_cache_plan_record & rec,
        int32_t candidate) noexcept {
    if (candidate < 0 || uint32_t(candidate) >= rec.n_inventory) {
        return -1;
    }
    const auto & row = rec.inventory[size_t(candidate)];
    if (row.is_chain()) {
        const int32_t base = row.component_ids[0];
        return base >= 0 && uint32_t(base) < rec.n_inventory
            ? rec.inventory[size_t(base)].source_id : -1;
    }
    return row.provider == common_cache_plan_provider::host_cache_entry
        ? row.source_id : -1;
}

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    if (candidate < 0 || legacy_candidate < 0 ||
        uint32_t(candidate) >= rec.n_inventory ||
        uint32_t(legacy_candidate) >= rec.n_inventory) {
        return 0;
    }
    const auto & planned = rec.inventory[size_t(candidate)];
    const auto & legacy = rec.inventory[size_t(legacy_candidate)];
    common_cache_plan_destruction_effect_set effects = 0;
    if (planned.target_slot_id != legacy.target_slot_id) {
        if (rec.selection == common_cache_plan_selection::similarity &&
            planned.provider == common_cache_plan_provider::live_slot &&
            planned.f_keep_known && planned.f_keep >= 1.0) {
            // The sole B-A zero-destruction cross-target case.
        } else {
            effects |= common_cache_plan_destruction_effect_bit(
                rec.selection == common_cache_plan_selection::similarity
                    ? common_cache_plan_destruction_effect::
                          destructive_similarity_retarget
                    : common_cache_plan_destruction_effect::
                          cross_target_displacement);
        }
    }
    const bool legacy_uses_live_target =
        common_cache_plan_provider_is_live(legacy.provider);
    const bool destruction_certification_available =
        (permitted_effects &
         server_cache_plan_nonconsuming_host_effects(true)) != 0;
    if (planned.target_slot_id == legacy.target_slot_id &&
        ((planned.provider == common_cache_plan_provider::cold_replay &&
          legacy.provider != common_cache_plan_provider::cold_replay) ||
         (destruction_certification_available &&
          (planned.provider == common_cache_plan_provider::host_cache_entry ||
           planned.is_chain()) && legacy_uses_live_target))) {
        // Cold replacement is the frozen B-A4 effect. D-A5 adds host restore
        // to the same physical class only when lifecycle certification exists;
        // lifecycle-off therefore preserves B-A4's previously-authorized
        // same-target host-restore behavior byte-for-byte.
        // The schema-v6 name predates non-consuming host restore. Its physical
        // class is the stable contract: any certified same-target whole-state
        // replacement destroys the live slot, whether replacement bytes come
        // from cold replay or an immutable host snapshot.
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::same_target_cold_replacement);
    }
    const int32_t planned_host = server_cache_plan_host_source(rec, candidate);
    const int32_t legacy_host = server_cache_plan_host_source(rec, legacy_candidate);
    if (planned_host >= 0 && planned_host != legacy_host) {
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::different_host_source_consumption);
    }
    // Physical non-effects (lifecycle's non-consuming host restore) and
    // mutation-boundary D-A certificates share this single row-opening mask.
    return effects & ~permitted_effects;
}

uint64_t server_cache_plan_capability_fold(
        uint64_t hash,
        uint64_t value) noexcept {
    // FNV-1a with an explicit value delimiter. This is a process-local drift
    // detector, not a durable/content identity.
    for (unsigned i = 0; i < 8; ++i) {
        hash = (hash ^ uint8_t(value >> (8*i))) * 1099511628211ull;
    }
    return (hash ^ 0xffu) * 1099511628211ull;
}

bool server_cache_plan_assign_source_id(
        int32_t & instance_source_id,
        int32_t & next_source_id,
        int32_t & source_id) noexcept {
    if (instance_source_id >= 0) {
        source_id = instance_source_id;
        return true;
    }
    if (next_source_id < 0 ||
        next_source_id > SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID ||
        next_source_id >= int32_t(COMMON_CACHE_PLAN_MAX_CANDIDATES)) {
        source_id = -1;
        return false;
    }
    instance_source_id = next_source_id++;
    source_id = instance_source_id;
    return true;
}

void server_cache_plan_authority::plan_before_mutation(
        common_cache_plan_record & rec,
        uint64_t capability_before,
        uint64_t capability_after) noexcept {
    rec.authority = {};
    common_cache_plan_authority_fallback fallback =
        common_cache_plan_authority_fallback::none;
    if (capability_before != capability_after) {
        rec.clear_planner_outputs();
        rec.planner_status = common_cache_plan_planner_status::incomplete_evidence;
        fallback = common_cache_plan_authority_fallback::stale_capability;
    } else {
        common_cache_plan_run_planner(rec);
    }
    common_cache_plan_derive_shadow_authority(rec, configured_level, fallback);
    rec.authority_prequalified =
        rec.planner_status == common_cache_plan_planner_status::ok &&
        capability_before == capability_after;
    rec.planner_precomputed = true;
}

#if defined(SERVER_CACHE_LOCAL_AUTHORITY)

namespace {

common_cache_optimizer_fallback_reason local_prediction_reason(
        server_cache_calibration_prediction_status status) noexcept {
    switch (status) {
        case server_cache_calibration_prediction_status::learning:
            return common_cache_optimizer_fallback_reason::incomplete_evidence;
        case server_cache_calibration_prediction_status::out_of_coverage:
            return common_cache_optimizer_fallback_reason::out_of_coverage;
        case server_cache_calibration_prediction_status::confidence_budget_exhausted:
            return common_cache_optimizer_fallback_reason::insufficient_confidence;
        case server_cache_calibration_prediction_status::numeric_fault:
            return common_cache_optimizer_fallback_reason::internal_fault;
        case server_cache_calibration_prediction_status::ok:
            break;
    }
    return common_cache_optimizer_fallback_reason::none;
}

common_cache_optimizer_coverage_class local_prediction_coverage(
        server_cache_calibration_prediction_status status) noexcept {
    switch (status) {
        case server_cache_calibration_prediction_status::out_of_coverage:
            return common_cache_optimizer_coverage_class::out_of_coverage;
        case server_cache_calibration_prediction_status::numeric_fault:
            return common_cache_optimizer_coverage_class::unavailable;
        case server_cache_calibration_prediction_status::ok:
            return common_cache_optimizer_coverage_class::complete;
        case server_cache_calibration_prediction_status::learning:
        case server_cache_calibration_prediction_status::confidence_budget_exhausted:
            return common_cache_optimizer_coverage_class::confidence_inactive;
    }
    return common_cache_optimizer_coverage_class::unavailable;
}

void local_refuse(
        common_cache_plan_record & rec,
        common_cache_optimizer_fallback_reason reason,
        common_cache_optimizer_coverage_class coverage,
        common_cache_optimizer_profile_state state,
        common_cache_optimizer_disposition disposition =
            common_cache_optimizer_disposition::refused,
        server_cache_plan_local_authority_latch * latch = nullptr) noexcept {
    rec.optimizer.economic_disposition = disposition;
    rec.optimizer.local_fallback_reason = reason;
    rec.optimizer.coverage_class = coverage;
    rec.optimizer.profile_state = state;
    rec.optimizer.local_authority.state = disposition ==
            common_cache_optimizer_disposition::not_attempted
        ? common_cache_optimizer_authority_state::not_attempted
        : common_cache_optimizer_authority_state::fallback;
    rec.optimizer.local_authority.reason = reason;
    if (latch && disposition !=
            common_cache_optimizer_disposition::not_attempted) {
        (void) latch->fallback(rec.optimizer.local_authority, reason);
    }
    rec.authority_prequalified = false;
}

struct local_profile_reduction {
    common_cache_optimizer_profile_state state =
        common_cache_optimizer_profile_state::active;
    common_cache_optimizer_coverage_class coverage =
        common_cache_optimizer_coverage_class::complete;
    common_cache_optimizer_fallback_reason reason =
        common_cache_optimizer_fallback_reason::none;
    common_cache_optimizer_disposition disposition =
        common_cache_optimizer_disposition::certified_improvement;
    bool all_points = true;

    void note(const server_cache_calibration_snapshot_lookup & lookup) noexcept {
        const auto terminal = lookup.instance
            ? lookup.instance->authority_terminal
            : server_cache_calibration_authority_terminal::none;
        const auto state_rank = [](common_cache_optimizer_profile_state value) {
            switch (value) {
                case common_cache_optimizer_profile_state::quarantined: return 5;
                case common_cache_optimizer_profile_state::drifted: return 4;
                case common_cache_optimizer_profile_state::learning: return 3;
                case common_cache_optimizer_profile_state::provisional: return 2;
                case common_cache_optimizer_profile_state::active: return 1;
                default: return 6;
            }
        };
        common_cache_optimizer_profile_state item_state =
            common_cache_optimizer_profile_state::learning;
        if (terminal == server_cache_calibration_authority_terminal::numeric_fault) {
            item_state = common_cache_optimizer_profile_state::quarantined;
        } else if (terminal ==
                server_cache_calibration_authority_terminal::drifted) {
            item_state = common_cache_optimizer_profile_state::drifted;
        } else if (terminal !=
                server_cache_calibration_authority_terminal::none) {
            // Coverage and confidence terminals are typed separately below;
            // they are not corrupt fits and must not masquerade as quarantine.
            item_state = common_cache_optimizer_profile_state::provisional;
        } else {
            switch (lookup.state) {
                case server_cache_calibration_instance_state::active:
                    item_state = common_cache_optimizer_profile_state::active; break;
                case server_cache_calibration_instance_state::provisional:
                    item_state = common_cache_optimizer_profile_state::provisional; break;
                case server_cache_calibration_instance_state::drifted:
                    item_state = common_cache_optimizer_profile_state::drifted; break;
                case server_cache_calibration_instance_state::quarantined:
                    item_state = common_cache_optimizer_profile_state::quarantined; break;
                case server_cache_calibration_instance_state::unseen:
                case server_cache_calibration_instance_state::learning:
                    item_state = common_cache_optimizer_profile_state::learning; break;
            }
        }
        if (state_rank(item_state) > state_rank(state)) state = item_state;
        all_points = all_points && lookup.point_available;

        if (terminal == server_cache_calibration_authority_terminal::numeric_fault) {
            coverage = common_cache_optimizer_coverage_class::unavailable;
            reason = common_cache_optimizer_fallback_reason::internal_fault;
            disposition = common_cache_optimizer_disposition::refused;
        } else if (terminal ==
                server_cache_calibration_authority_terminal::drifted) {
            // Drift is instance-local. Preserve it in the request's diagnostic
            // profile-state reduction, but let the direct-bound door decide
            // whether a selected baseline/challenger contribution is usable.
            coverage = common_cache_optimizer_coverage_class::confidence_inactive;
        } else if (terminal ==
                server_cache_calibration_authority_terminal::tail_exceeded ||
            lookup.prediction.status ==
                server_cache_calibration_prediction_status::out_of_coverage) {
            // Coverage is likewise required only for the terms in the final
            // comparison. A point-complete unselected row must not veto it.
            coverage = common_cache_optimizer_coverage_class::out_of_coverage;
        } else if (terminal ==
                server_cache_calibration_authority_terminal::confidence_budget_exhausted ||
            terminal == server_cache_calibration_authority_terminal::ordinal_exhausted ||
            lookup.prediction.status == server_cache_calibration_prediction_status::
                confidence_budget_exhausted) {
            // Confidence is request-local: a point-complete third candidate
            // may remain provisional without blocking a certified comparison
            // between an active baseline and challenger. The direct-bound
            // door below checks confidence for exactly those required terms.
            coverage = common_cache_optimizer_coverage_class::confidence_inactive;
        } else if (!lookup.point_available) {
            if (reason == common_cache_optimizer_fallback_reason::none) {
                coverage = common_cache_optimizer_coverage_class::point_estimate_incomplete;
                reason = common_cache_optimizer_fallback_reason::incomplete_evidence;
                disposition = common_cache_optimizer_disposition::learning;
            }
        } else if (lookup.state !=
                server_cache_calibration_instance_state::active &&
            reason == common_cache_optimizer_fallback_reason::none) {
            coverage = common_cache_optimizer_coverage_class::confidence_inactive;
        }
    }
};

bool local_fill_candidate(
        common_cache_plan_candidate & row,
        const server_cache_calibration_snapshot_lookup & lookup,
        uint64_t prompt_tokens) noexcept {
    if (!std::isfinite(lookup.prediction.point_us) ||
        lookup.prediction.point_us < 0.0 ||
        lookup.prediction.point_us > double(UINT64_MAX)) {
        return false;
    }
    const uint64_t replay_tokens = row.provider ==
            common_cache_plan_provider::cold_replay
        ? prompt_tokens
        : row.lcp_tokens.state == llama_cache_acct_known::known &&
              prompt_tokens > row.lcp_tokens.value
            ? prompt_tokens - row.lcp_tokens.value : 0;
    const uint64_t restore_bytes =
        row.provider == common_cache_plan_provider::host_cache_entry ||
        row.provider == common_cache_plan_provider::live_context_checkpoint
            ? row.payload_bytes.value : 0;
    const uint64_t estimate = uint64_t(std::llround(lookup.prediction.point_us));
    for (auto & term : row.cost_terms) term = {};
    auto set = [&](llama_cache_acct_cost_kind kind, uint64_t raw, uint64_t us) {
        auto & term = row.cost_terms[size_t(kind)];
        term.raw = llama_cache_acct_value::measured(raw);
        term.estimated_us = llama_cache_acct_value::measured(us);
        term.estimator_version =
            SERVER_CACHE_CALIBRATION_ESTIMATOR_VERSION;
    };
    if (lookup.instance->key.operation ==
            server_cache_observation_operation::restore) {
        // The measured restore response includes its replay tail. Charge that
        // total exactly once in the restore term; workspace is known zero for
        // local fits and replay remains descriptive only.
        set(llama_cache_acct_cost_kind::restore, restore_bytes, estimate);
        set(llama_cache_acct_cost_kind::replay, replay_tokens, 0);
    } else {
        set(llama_cache_acct_cost_kind::restore, 0, 0);
        set(llama_cache_acct_cost_kind::replay, replay_tokens, estimate);
    }
    set(llama_cache_acct_cost_kind::workspace, 0, 0);
    row.predicted_total_us = llama_cache_acct_value::measured(estimate);
    return true;
}

} // namespace

void server_cache_plan_authority::plan_local_before_mutation(
        common_cache_plan_record & rec,
        const server_cache_plan_local_inventory & evidence,
        const server_cache_observation_store & observations,
        int32_t legacy_plan_candidate,
        uint64_t capability_before,
        uint64_t capability_after,
        uint64_t now_unix_ms,
        common_cache_plan_destruction_effect_set /* permitted_effects */,
        server_cache_plan_local_authority_latch * latch) noexcept {
    rec.authority = {};
    rec.optimizer.baseline_plan_candidate = legacy_plan_candidate;
    rec.optimizer.local_authority.state =
        common_cache_optimizer_authority_state::not_attempted;
    rec.authority.legacy_plan_candidate = legacy_plan_candidate;
    rec.planner_precomputed = true;
    try {
        if (local_observations != &observations) {
            rec.clear_planner_outputs();
            rec.planner_status = common_cache_plan_planner_status::internal_fault;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::currency_changed,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::unavailable,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::stale_capability);
            return;
        }
        if (capability_before != capability_after) {
            rec.clear_planner_outputs();
            rec.planner_status =
                common_cache_plan_planner_status::incomplete_evidence;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::currency_changed,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::unavailable,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::stale_capability);
            return;
        }
        if (rec.selection != common_cache_plan_selection::by_id) {
            rec.clear_planner_outputs();
            rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::none,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::unseen,
                common_cache_optimizer_disposition::not_attempted, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::tier_not_enabled);
            return;
        }
        server_cache_calibration_authority_snapshot snapshot;
        if (!server_cache_calibration_capture_snapshot(
                observations, now_unix_ms, snapshot)) {
            rec.clear_planner_outputs();
            rec.planner_status = common_cache_plan_planner_status::profile_unfitted;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::profile_unfitted,
                common_cache_optimizer_coverage_class::point_estimate_incomplete,
                common_cache_optimizer_profile_state::learning,
                common_cache_optimizer_disposition::learning, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::profile_unfitted);
            return;
        }
        rec.optimizer.local_authority.authority_currency_serial =
            llama_cache_acct_value::measured(snapshot.authority_currency_serial);
        if (!profile_display_label(
                observations.execution_fingerprint().execution_root,
                rec.optimizer.profile_identity)) {
            rec.clear_planner_outputs();
            rec.planner_status = common_cache_plan_planner_status::internal_fault;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::internal_fault,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::quarantined,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::internal_fault);
            return;
        }
        rec.optimizer.profile_source =
            common_cache_optimizer_profile_source::local_fit;

        if (rec.inventory_saturated() || rec.derived_plans_incomplete ||
            rec.n_prompt_tokens.state != llama_cache_acct_known::known ||
            legacy_plan_candidate < 0 ||
            uint32_t(legacy_plan_candidate) >= rec.n_inventory) {
            rec.clear_planner_outputs();
            rec.planner_status =
                common_cache_plan_planner_status::incomplete_evidence;
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::incomplete_evidence,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::unavailable,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }

        std::array<server_cache_calibration_snapshot_lookup,
                   COMMON_CACHE_PLAN_MAX_CANDIDATES> lookups = {};
        std::array<std::array<server_cache_calibration_snapshot_lookup, 3>,
                   COMMON_CACHE_PLAN_MAX_CANDIDATES> consequence_lookups = {};
        local_profile_reduction reduction;
        const auto price_consequences = [&](
                uint32_t i, common_cache_plan_candidate & row) {
            const auto & candidate_evidence = evidence.candidates[i];
            if (candidate_evidence.requires_d_consequences &&
                candidate_evidence.consequence_count == 0) {
                return false;
            }
            if (candidate_evidence.consequence_count >
                    candidate_evidence.consequences.size()) {
                return false;
            }
            bool complete = true;
            for (uint8_t d = 0;
                 d < candidate_evidence.consequence_count; ++d) {
                const auto & item = candidate_evidence.consequences[d];
                auto & lookup = consequence_lookups[i][d];
                if (!item.measurable || item.weight_milli == 0 ||
                    item.weight_milli > 1000000 ||
                    !server_cache_calibration_snapshot_lookup_exact(
                        snapshot, item.key, item.feature, lookup)) {
                    lookup.state =
                        server_cache_calibration_instance_state::unseen;
                }
                reduction.note(lookup);
                if (!lookup.point_available) {
                    complete = false;
                    continue;
                }
                const long double weighted =
                    (long double) lookup.prediction.point_us *
                    item.weight_milli / 1000.0L;
                const uint64_t estimate = std::isfinite(weighted) &&
                        weighted >= 0.0L &&
                        weighted <= (long double) UINT64_MAX
                    ? uint64_t(std::llround(weighted)) : UINT64_MAX;
                auto & cost = row.cost_terms[size_t(item.cost_kind)];
                const uint64_t prior = cost.estimated_us.state ==
                        llama_cache_acct_known::known
                    ? cost.estimated_us.value : 0;
                if (estimate == UINT64_MAX ||
                    prior > UINT64_MAX - estimate ||
                    row.predicted_total_us.state !=
                        llama_cache_acct_known::known ||
                    row.predicted_total_us.value > UINT64_MAX - estimate) {
                    complete = false;
                    continue;
                }
                cost.raw = llama_cache_acct_value::measured(0);
                cost.estimated_us =
                    llama_cache_acct_value::measured(prior + estimate);
                cost.estimator_version =
                    SERVER_CACHE_CALIBRATION_ESTIMATOR_VERSION;
                row.predicted_total_us.value += estimate;
            }
            return complete;
        };
        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            auto & row = rec.inventory[i];
            if (row.disposition == common_cache_plan_disposition::unavailable) {
                rec.clear_planner_outputs();
                rec.planner_status =
                    common_cache_plan_planner_status::incomplete_evidence;
                local_refuse(rec,
                    common_cache_optimizer_fallback_reason::incomplete_evidence,
                    common_cache_optimizer_coverage_class::unavailable,
                    common_cache_optimizer_profile_state::unavailable,
                    common_cache_optimizer_disposition::refused, latch);
                common_cache_plan_derive_shadow_authority(
                    rec, configured_level,
                    common_cache_plan_authority_fallback::incomplete_evidence);
                return;
            }
            if (!row.viable() || row.is_chain()) continue;
            const auto & candidate_evidence = evidence.candidates[i];
            if (!candidate_evidence.measurable ||
                !server_cache_calibration_snapshot_lookup_exact(
                    snapshot, candidate_evidence.key,
                    candidate_evidence.feature, lookups[i])) {
                lookups[i].state =
                    server_cache_calibration_instance_state::unseen;
            }
            reduction.note(lookups[i]);
            if (lookups[i].point_available &&
                !local_fill_candidate(
                    row, lookups[i], rec.n_prompt_tokens.value)) {
                lookups[i].point_available = false;
                reduction.all_points = false;
            }
            reduction.all_points =
                price_consequences(i, row) && reduction.all_points;
        }

        if (reduction.all_points) {
            reduction.all_points =
                common_cache_plan_compose_preestimated_chains(
                    rec, SERVER_CACHE_CALIBRATION_ESTIMATOR_VERSION) ==
                common_cache_plan_planner_status::ok;
        }
        if (reduction.all_points) {
            for (uint32_t i = 0; i < rec.n_inventory; ++i) {
                auto & row = rec.inventory[i];
                if (row.viable() && row.is_chain() &&
                    !price_consequences(i, row)) {
                    reduction.all_points = false;
                }
            }
        }
        if (!reduction.all_points) {
            rec.clear_planner_outputs();
            rec.planner_status =
                common_cache_plan_planner_status::incomplete_evidence;
            local_refuse(rec,
                reduction.reason == common_cache_optimizer_fallback_reason::none
                    ? common_cache_optimizer_fallback_reason::incomplete_evidence
                    : reduction.reason,
                reduction.coverage == common_cache_optimizer_coverage_class::complete
                    ? common_cache_optimizer_coverage_class::point_estimate_incomplete
                    : reduction.coverage,
                reduction.state,
                reduction.disposition ==
                        common_cache_optimizer_disposition::certified_improvement
                    ? common_cache_optimizer_disposition::learning
                    : reduction.disposition,
                latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }
        rec.optimizer.local_authority.coefficient_source =
            common_cache_optimizer_profile_source::local_fit;

        rec.planner_status = common_cache_plan_choose_preestimated(rec);
        if (rec.planner_status != common_cache_plan_planner_status::ok ||
            !server_cache_plan_shadow_choice_valid(rec)) {
            rec.clear_planner_outputs();
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::incomplete_evidence,
                common_cache_optimizer_coverage_class::point_estimate_incomplete,
                common_cache_optimizer_profile_state::learning,
                common_cache_optimizer_disposition::learning, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }
        const int32_t challenger = rec.shadow_choice;
        rec.optimizer.economic_plan_candidate = challenger;
        rec.optimizer.profile_state = reduction.state;
        rec.optimizer.coverage_class = reduction.coverage;
        rec.optimizer.benefit_estimate_known = true;
        rec.optimizer.benefit_estimate_us =
            double(rec.inventory[size_t(legacy_plan_candidate)].predicted_total_us.value) -
            double(rec.inventory[size_t(challenger)].predicted_total_us.value);

        if (reduction.reason != common_cache_optimizer_fallback_reason::none) {
            local_refuse(rec,
                reduction.reason == common_cache_optimizer_fallback_reason::none
                    ? common_cache_optimizer_fallback_reason::insufficient_confidence
                    : reduction.reason,
                reduction.coverage, reduction.state,
                reduction.disposition ==
                        common_cache_optimizer_disposition::certified_improvement
                    ? common_cache_optimizer_disposition::learning
                    : reduction.disposition,
                latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }

        if (challenger == legacy_plan_candidate) {
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::insufficient_confidence,
                rec.optimizer.coverage_class, reduction.state,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::destruction_authority_required);
            return;
        }

        // Standalone restore fits are end-to-end and each includes its replay
        // tail. Summing the components is a conservative upper bound when a
        // chain is the challenger, but it is not a sound baseline estimate:
        // the shared replay tail would be counted twice and could fabricate a
        // benefit. Until a composite chain response is observed, a legacy
        // chain therefore executes unchanged.
        if (rec.inventory[size_t(legacy_plan_candidate)].is_chain()) {
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::incomplete_evidence,
                common_cache_optimizer_coverage_class::confidence_inactive,
                reduction.state,
                common_cache_optimizer_disposition::learning, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }

        std::array<server_cache_calibration_contribution, 16> contributions = {};
        size_t contribution_count = 0;
        const auto append_candidate = [&](
                int32_t candidate,
                server_cache_calibration_contribution_side side) {
            const auto append_one = [&](int32_t component) {
                if (component < 0 || uint32_t(component) >= rec.n_inventory ||
                    contribution_count == contributions.size()) return false;
                const auto & lookup = lookups[size_t(component)];
                if (!lookup.instance) return false;
                auto & term = contributions[contribution_count++];
                term.instance = lookup.instance;
                term.claim = lookup.claim;
                term.feature = evidence.candidates[size_t(component)].feature;
                term.side = side;
                term.now_unix_ms = now_unix_ms;
                term.authority_admission_allowed = true;
                return true;
            };
            const auto append_consequences = [&](int32_t owner) {
                if (owner < 0 || uint32_t(owner) >= rec.n_inventory) {
                    return false;
                }
                const auto & candidate_evidence =
                    evidence.candidates[size_t(owner)];
                for (uint8_t d = 0;
                     d < candidate_evidence.consequence_count; ++d) {
                    if (contribution_count == contributions.size()) return false;
                    const auto & consequence =
                        candidate_evidence.consequences[d];
                    const auto & consequence_lookup =
                        consequence_lookups[size_t(owner)][d];
                    if (!consequence_lookup.instance) return false;
                    auto & extra = contributions[contribution_count++];
                    extra.instance = consequence_lookup.instance;
                    extra.claim = consequence_lookup.claim;
                    extra.feature = consequence.feature;
                    extra.weight_milli = consequence.weight_milli;
                    extra.side = side;
                    extra.now_unix_ms = now_unix_ms;
                    extra.authority_admission_allowed = true;
                }
                return true;
            };
            const auto & row = rec.inventory[size_t(candidate)];
            if (!row.is_chain()) {
                return append_one(candidate) &&
                       append_consequences(candidate);
            }
            bool any = false;
            for (const int32_t component : row.component_ids) {
                if (component < 0) continue;
                if (!append_one(component)) return false;
                any = true;
            }
            return any && append_consequences(candidate);
        };
        if (!append_candidate(
                legacy_plan_candidate,
                server_cache_calibration_contribution_side::baseline) ||
            !append_candidate(
                challenger,
                server_cache_calibration_contribution_side::challenger)) {
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::internal_fault,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::quarantined,
                common_cache_optimizer_disposition::refused, latch);
            return;
        }
        // Preserve the exact typed terminal of a required contribution. The
        // mathematical bound intentionally has a smaller status vocabulary;
        // doing this at the selected-contribution boundary prevents drift from
        // being flattened into generic learning while keeping unrelated rows
        // diagnostic-only.
        for (size_t i = 0; i < contribution_count; ++i) {
            const auto terminal = contributions[i].instance->authority_terminal;
            if (terminal == server_cache_calibration_authority_terminal::none) {
                continue;
            }
            common_cache_optimizer_fallback_reason reason =
                common_cache_optimizer_fallback_reason::internal_fault;
            common_cache_optimizer_coverage_class coverage =
                common_cache_optimizer_coverage_class::unavailable;
            common_cache_optimizer_profile_state state =
                common_cache_optimizer_profile_state::quarantined;
            switch (terminal) {
                case server_cache_calibration_authority_terminal::tail_exceeded:
                    reason = common_cache_optimizer_fallback_reason::out_of_coverage;
                    coverage = common_cache_optimizer_coverage_class::out_of_coverage;
                    state = common_cache_optimizer_profile_state::provisional;
                    break;
                case server_cache_calibration_authority_terminal::confidence_budget_exhausted:
                case server_cache_calibration_authority_terminal::ordinal_exhausted:
                    reason = common_cache_optimizer_fallback_reason::insufficient_confidence;
                    coverage = common_cache_optimizer_coverage_class::confidence_inactive;
                    state = common_cache_optimizer_profile_state::provisional;
                    break;
                case server_cache_calibration_authority_terminal::drifted:
                    reason = common_cache_optimizer_fallback_reason::drifted;
                    coverage = common_cache_optimizer_coverage_class::confidence_inactive;
                    state = common_cache_optimizer_profile_state::drifted;
                    break;
                case server_cache_calibration_authority_terminal::numeric_fault:
                case server_cache_calibration_authority_terminal::_count:
                case server_cache_calibration_authority_terminal::none:
                    break;
            }
            local_refuse(rec, reason, coverage, state,
                common_cache_optimizer_disposition::refused, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }
        server_cache_calibration_direct_bound bound;
        if (!server_cache_calibration_bound_direct_difference(
                contributions.data(), contribution_count, bound) ||
            bound.status != server_cache_calibration_prediction_status::ok ||
            !(bound.benefit_lower_us > 0.0)) {
            rec.optimizer.benefit_lower_known =
                bound.status == server_cache_calibration_prediction_status::ok;
            rec.optimizer.benefit_lower_us = bound.benefit_lower_us;
            local_refuse(rec,
                bound.status == server_cache_calibration_prediction_status::ok
                    ? common_cache_optimizer_fallback_reason::insufficient_confidence
                    : local_prediction_reason(bound.status),
                bound.status == server_cache_calibration_prediction_status::ok
                    ? common_cache_optimizer_coverage_class::confidence_inactive
                    : local_prediction_coverage(bound.status),
                reduction.state,
                common_cache_optimizer_disposition::learning, latch);
            common_cache_plan_derive_shadow_authority(
                rec, configured_level,
                common_cache_plan_authority_fallback::incomplete_evidence);
            return;
        }

        rec.optimizer.benefit_lower_known = true;
        rec.optimizer.benefit_lower_us = bound.benefit_lower_us;
        rec.optimizer.coverage_class =
            common_cache_optimizer_coverage_class::complete;
        rec.optimizer.local_authority.candidate = challenger;
        const auto & authority_claim = contributions[0].claim;
        rec.optimizer.local_authority.boot_claim_ordinal =
            llama_cache_acct_value::measured(
                authority_claim.boot_claim_ordinal);
        rec.optimizer.local_authority.profile_generation =
            llama_cache_acct_value::measured(
                authority_claim.profile_generation_ordinal);
        for (size_t i = 1; i < contribution_count; ++i) {
            if (contributions[i].claim.boot_claim_ordinal !=
                    authority_claim.boot_claim_ordinal ||
                contributions[i].claim.profile_generation_ordinal !=
                    authority_claim.profile_generation_ordinal) {
                local_refuse(rec,
                    common_cache_optimizer_fallback_reason::internal_fault,
                    common_cache_optimizer_coverage_class::unavailable,
                    common_cache_optimizer_profile_state::quarantined,
                    common_cache_optimizer_disposition::refused, latch);
                return;
            }
        }
        struct required_generation {
            uint32_t slot = 0;
            std::array<uint8_t, 32> key_digest = {};
            uint64_t generation = 0;
        };
        std::array<required_generation, 16> required = {};
        size_t required_count = 0;
        for (size_t index = 0; index < contribution_count; ++index) {
            const auto & contribution = contributions[index];
            const auto duplicate = std::find_if(
                required.begin(), required.begin() + required_count,
                [&](const required_generation & item) {
                    return item.slot == contribution.claim.estimator_slot &&
                           item.generation == contribution.claim.fit_generation;
                });
            if (duplicate != required.begin() + required_count) {
                continue;
            }
            auto & item = required[required_count++];
            item.slot = contribution.claim.estimator_slot;
            item.generation = contribution.claim.fit_generation;
            if (!server_cache_calibration_instance_key_digest_v1(
                    contribution.instance->key, item.key_digest)) {
                local_refuse(rec,
                    common_cache_optimizer_fallback_reason::internal_fault,
                    common_cache_optimizer_coverage_class::unavailable,
                    common_cache_optimizer_profile_state::quarantined,
                    common_cache_optimizer_disposition::refused, latch);
                return;
            }
        }
        std::sort(required.begin(), required.begin() + required_count,
            [](const auto & a, const auto & b) {
                return std::tie(a.slot, a.key_digest) <
                       std::tie(b.slot, b.key_digest);
            });
        llama_sha256 generations;
        static constexpr char generation_domain[] =
            "buun-zc-authority-generation-v1";
        generations.update(generation_domain, sizeof(generation_domain));
        uint8_t procedure[4];
        uint8_t boot[8];
        uint8_t profile[8];
        uint8_t count[4];
        llama_store_le_u32(procedure,
            rec.optimizer.local_authority.procedure_version);
        llama_store_le_u64(boot,
            authority_claim.boot_claim_ordinal);
        llama_store_le_u64(profile,
            authority_claim.profile_generation_ordinal);
        llama_store_le_u32(count, uint32_t(required_count));
        generations.update(procedure, sizeof(procedure));
        generations.update(boot, sizeof(boot));
        generations.update(profile, sizeof(profile));
        generations.update(count, sizeof(count));
        for (size_t i = 0; i < required_count; ++i) {
            uint8_t entry_size[4];
            uint8_t slot_bytes[4];
            uint8_t generation_bytes[8];
            llama_store_le_u32(entry_size, 4 + 32 + 8);
            llama_store_le_u32(slot_bytes, required[i].slot);
            llama_store_le_u64(generation_bytes, required[i].generation);
            generations.update(entry_size, sizeof(entry_size));
            generations.update(slot_bytes, sizeof(slot_bytes));
            generations.update(required[i].key_digest.data(),
                               required[i].key_digest.size());
            generations.update(generation_bytes, sizeof(generation_bytes));
        }
        rec.optimizer.local_authority.instance_generation_digest =
            common_cache_plan_sha256_hex_digest(generations.finish());
        rec.optimizer.local_authority.state =
            common_cache_optimizer_authority_state::prequalified;
        if (latch &&
            !latch->prequalify(rec.optimizer.local_authority)) {
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::internal_fault,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::quarantined,
                common_cache_optimizer_disposition::refused, latch);
            return;
        }
        rec.optimizer.economic_disposition =
            common_cache_optimizer_disposition::certified_improvement;
        rec.optimizer.local_fallback_reason =
            common_cache_optimizer_fallback_reason::none;
        rec.optimizer.local_authority.state =
            common_cache_optimizer_authority_state::certified;
        rec.optimizer.local_authority.certified_once = true;
        rec.optimizer.local_authority.reason =
            common_cache_optimizer_fallback_reason::none;
        if (latch && !latch->certify(rec.optimizer.local_authority)) {
            local_refuse(rec,
                common_cache_optimizer_fallback_reason::internal_fault,
                common_cache_optimizer_coverage_class::unavailable,
                common_cache_optimizer_profile_state::quarantined,
                common_cache_optimizer_disposition::refused, latch);
            return;
        }
        common_cache_plan_derive_shadow_authority(
            rec, configured_level,
            common_cache_plan_authority_fallback::none);
        rec.authority_prequalified = true;
    } catch (...) {
        rec.clear_planner_outputs();
        rec.planner_status = common_cache_plan_planner_status::internal_fault;
        local_refuse(rec,
            common_cache_optimizer_fallback_reason::internal_fault,
            common_cache_optimizer_coverage_class::unavailable,
            common_cache_optimizer_profile_state::quarantined,
            common_cache_optimizer_disposition::refused, latch);
        common_cache_plan_derive_shadow_authority(
            rec, configured_level,
            common_cache_plan_authority_fallback::internal_fault);
    }
}

#endif // SERVER_CACHE_LOCAL_AUTHORITY

void server_cache_plan_authority::fail_closed(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason,
        server_cache_plan_local_authority_latch * latch) noexcept {
    rec.clear_planner_outputs();
    rec.planner_status = common_cache_plan_planner_status::internal_fault;
    common_cache_plan_derive_shadow_authority(rec, configured_level, reason);
    const auto decision_level = server_cache_plan_level_of(rec.selection);
    if (decision_level != common_cache_plan_authority_level::off &&
        decision_level != common_cache_plan_authority_level::_count &&
        server_cache_plan_level_enabled(configured_level, decision_level)) {
        fallback_legacy(rec, reason, latch);
    }
    rec.authority_prequalified = false;
    rec.planner_precomputed = true;
}

int32_t server_cache_plan_legacy_candidate(
        const common_cache_plan_record & rec,
        int32_t target_slot_id,
        bool host_lookup_enabled) noexcept {
    int32_t live = -1;
    int32_t host = -1;
    double f_keep = -1.0;
    double sim = 0.0;

    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain()) {
            continue;
        }
        if (candidate.provider == common_cache_plan_provider::live_slot) {
            live = int32_t(i);
            if (candidate.f_keep_known) {
                f_keep = candidate.f_keep;
            }
            if (candidate.sim_known) {
                sim = candidate.sim;
            }
            break;
        }
    }

    // Reproduce the legacy host selector's strict two-axis improvement and
    // insertion-order tie behavior. Invalid rows never enter that selector.
    for (uint32_t i = 0; host_lookup_enabled && i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::host_cache_entry ||
            !candidate.viable() || !candidate.f_keep_known ||
            !candidate.sim_known) {
            continue;
        }
        if (f_keep < candidate.f_keep && sim < candidate.sim) {
            f_keep = candidate.f_keep;
            sim = candidate.sim;
            host = int32_t(i);
        }
    }

    const int32_t host_source = host >= 0
        ? rec.inventory[size_t(host)].source_id : -1;
    int32_t checkpoint = -1;
    int32_t checkpoint_ordinal = -1;
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id != target_slot_id || candidate.is_chain() ||
            candidate.provider != common_cache_plan_provider::live_context_checkpoint ||
            !candidate.viable()) {
            continue;
        }
        const int32_t ordinal =
            server_cache_plan_checkpoint_ordinal_from_source_id(
                candidate.source_id, host >= 0 ? host_source : -1);
        if (host >= 0) {
            if (!candidate.component_only ||
                candidate.dependent_host_source_id != host_source) {
                continue;
            }
        } else if (candidate.component_only) {
            continue;
        }
        // The shipped selector scans newest-to-oldest, so the greatest forward
        // ordinal is its first viable checkpoint.
        if (ordinal >= 0 && ordinal > checkpoint_ordinal) {
            checkpoint_ordinal = ordinal;
            checkpoint = int32_t(i);
        }
    }

    if (checkpoint >= 0) {
        if (host < 0) {
            return checkpoint;
        }
        const auto * chain = rec.find_chain(
            common_cache_plan_provider::host_cache_entry, host, checkpoint);
        const int32_t chain_id = chain
            ? int32_t(chain - rec.inventory.data()) : -1;
        return chain_id >= 0 ? chain_id : host;
    }
    if (host >= 0) {
        return host;
    }
    if (live >= 0 && rec.inventory[size_t(live)].viable()) {
        return live;
    }
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.target_slot_id == target_slot_id &&
            candidate.provider == common_cache_plan_provider::cold_replay &&
            !candidate.is_chain() && candidate.viable()) {
            return int32_t(i);
        }
    }
    return -1;
}

bool server_cache_plan_execution_from_candidate(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t target_slot_id,
        server_cache_plan_execution & out) noexcept {
    out = {};
    if (candidate < 0 || uint32_t(candidate) >= rec.n_inventory) {
        return false;
    }
    const auto & selected = rec.inventory[size_t(candidate)];
    if (selected.target_slot_id != target_slot_id || !selected.viable()) {
        return false;
    }
    out.target = target_slot_id;
    if (selected.is_chain()) {
        const int32_t host = selected.component_ids[0];
        const int32_t checkpoint = selected.component_ids[1];
        if (host < 0 || checkpoint < 0 || uint32_t(host) >= rec.n_inventory ||
            uint32_t(checkpoint) >= rec.n_inventory) {
            return false;
        }
        const auto & h = rec.inventory[size_t(host)];
        const auto & c = rec.inventory[size_t(checkpoint)];
        if (h.target_slot_id != target_slot_id || c.target_slot_id != target_slot_id ||
            h.provider != common_cache_plan_provider::host_cache_entry ||
            c.provider != common_cache_plan_provider::live_context_checkpoint ||
            !c.component_only || c.dependent_host_source_id != h.source_id ||
            !h.viable() || !c.viable()) {
            return false;
        }
        out.kind = server_cache_plan_execution_kind::host_checkpoint_restore;
        out.host_source_id = h.source_id;
        out.checkpoint_source_id = c.source_id;
        return true;
    }
    switch (selected.provider) {
        case common_cache_plan_provider::live_slot:
            out.kind = server_cache_plan_execution_kind::live_replay;
            return true;
        case common_cache_plan_provider::host_cache_entry:
            out.kind = server_cache_plan_execution_kind::host_restore;
            out.host_source_id = selected.source_id;
            return true;
        case common_cache_plan_provider::live_context_checkpoint:
            if (selected.component_only) {
                return false;
            }
            out.kind = server_cache_plan_execution_kind::checkpoint_restore;
            out.checkpoint_source_id = selected.source_id;
            return true;
        case common_cache_plan_provider::cold_replay:
            out.kind = server_cache_plan_execution_kind::cold_replay;
            return true;
        case common_cache_plan_provider::_count:
            break;
    }
    return false;
}

static bool inside_pre_da_safety_envelope(
        const common_cache_plan_record & rec,
        int32_t planned_candidate,
        int32_t legacy_candidate,
        common_cache_plan_destruction_effect_set permitted_effects) noexcept {
    return server_cache_destruction_effects_for(
        rec, planned_candidate, legacy_candidate, permitted_effects) == 0;
}

static common_cache_plan_authority_fallback pre_da_envelope_refusal_reason(
        const common_cache_plan_record & rec,
        const server_cache_plan_execution & planned,
        const server_cache_plan_execution & legacy) noexcept {
    // Schema 5 has no eviction_evidence_unavailable spelling. At LRU, use its
    // existing budget/lease availability reason only for the D-A fence: a
    // target change, or a cold replacement of retained same-target state.
    // Consuming a different host source remains destruction authority, just as
    // it does at every earlier ratchet.
    if (rec.selection == common_cache_plan_selection::lru &&
        (planned.target != legacy.target ||
         (planned.kind == server_cache_plan_execution_kind::cold_replay &&
          legacy.kind != server_cache_plan_execution_kind::cold_replay))) {
        return common_cache_plan_authority_fallback::budget_or_lease_unavailable;
    }
    return common_cache_plan_authority_fallback::
        destruction_authority_required;
}

server_cache_plan_execution server_cache_plan_authority::authorize(
        common_cache_plan_record & rec,
        int32_t legacy_target_slot_id,
        bool host_lookup_enabled,
        bool target_identity_matches,
        common_cache_plan_destruction_effect_set permitted_effects,
        server_cache_plan_local_authority_latch latch) noexcept {
    server_cache_plan_execution execution;
    const bool local = rec.optimizer.local_authority.certified_once;
    const auto refuse = [&](common_cache_plan_authority_fallback reason) {
        fallback_legacy(rec, reason, local ? &latch : nullptr);
    };
    if (local && !latch.certified_for(rec.optimizer.local_authority)) {
        // A copied diagnostic receipt is not an authority capability. Do not
        // let a fresh latch transition through the ordinary fallback door and
        // thereby make the missing-capability failure look safety-certified.
        fallback_legacy(
            rec, common_cache_plan_authority_fallback::internal_fault, nullptr);
        return execution;
    }
    if (!local_currency_current(rec)) {
        refuse(common_cache_plan_authority_fallback::stale_capability);
        return execution;
    }
    const auto decision_level = server_cache_plan_level_of(rec.selection);
    if (decision_level == common_cache_plan_authority_level::off ||
        decision_level == common_cache_plan_authority_level::_count) {
        return execution;
    }
    if (!server_cache_plan_level_enabled(configured_level, decision_level)) {
        // Preserve a planner refusal (no profile, incomplete evidence, ...).
        // tier_not_enabled describes only an otherwise-qualified plan whose
        // decision ratchet has not landed yet.
        if (rec.authority.fallback_reason ==
                common_cache_plan_authority_fallback::none &&
            rec.authority_prequalified &&
            rec.planner_status == common_cache_plan_planner_status::ok) {
            rec.authority.fallback_reason =
                common_cache_plan_authority_fallback::tier_not_enabled;
        }
        return execution;
    }
    const int32_t legacy_plan_candidate = server_cache_plan_legacy_candidate(
        rec, legacy_target_slot_id, host_lookup_enabled);
    rec.authority.legacy_plan_candidate = legacy_plan_candidate;
    if (!server_cache_plan_candidate_prequalified(rec)) {
        refuse(
            rec.authority.fallback_reason !=
                    common_cache_plan_authority_fallback::none
                ? rec.authority.fallback_reason
                : common_cache_plan_authority_fallback::internal_fault);
        return execution;
    }
    const int32_t planned_target_slot_id = server_cache_plan_planned_target(
        rec, configured_level, legacy_target_slot_id);
    if (planned_target_slot_id < 0) {
        refuse(
            common_cache_plan_authority_fallback::incomplete_evidence);
        return {};
    }
    if (!server_cache_plan_execution_from_candidate(
            rec, rec.shadow_choice, planned_target_slot_id, execution)) {
        refuse(
            common_cache_plan_authority_fallback::internal_fault);
        return {};
    }
    if (!target_identity_matches &&
        execution.kind != server_cache_plan_execution_kind::cold_replay) {
        // Identity feasibility is known before mutation; this is incomplete
        // planner evidence, not capability drift discovered at execution.
        refuse(
            common_cache_plan_authority_fallback::incomplete_evidence);
        return {};
    }
    server_cache_plan_execution legacy_execution;
    if (!server_cache_plan_execution_from_candidate(
            rec, legacy_plan_candidate, legacy_target_slot_id,
            legacy_execution)) {
        refuse(
            common_cache_plan_authority_fallback::internal_fault);
        return {};
    }
    if (!inside_pre_da_safety_envelope(
            rec, rec.shadow_choice, legacy_plan_candidate,
            permitted_effects)) {
        refuse(
            pre_da_envelope_refusal_reason(
                rec, execution, legacy_execution));
        return {};
    }
    rec.authority.state = common_cache_plan_authority_state::authoritative;
    rec.authority.fallback_reason = common_cache_plan_authority_fallback::none;
    execution.local_authority = std::move(latch);
    return execution;
}

void server_cache_plan_authority::fallback_legacy(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason,
        server_cache_plan_local_authority_latch * latch) noexcept {
    rec.authority.state = common_cache_plan_authority_state::fallback_legacy;
    rec.authority.fallback_reason = reason;
    if (rec.optimizer.local_authority.certified_once) {
        const auto local_reason =
            reason == common_cache_plan_authority_fallback::stale_capability
                ? common_cache_optimizer_fallback_reason::currency_changed
                : common_cache_optimizer_fallback_reason::safety_refusal;
        if (!latch ||
            !latch->fallback(rec.optimizer.local_authority, local_reason)) {
            rec.optimizer.local_authority.state =
                common_cache_optimizer_authority_state::fallback;
            rec.optimizer.local_authority.reason =
                common_cache_optimizer_fallback_reason::internal_fault;
        }
        rec.optimizer.local_fallback_reason =
            rec.optimizer.local_authority.reason;
    }
}

bool server_cache_plan_authority::local_currency_current(
        const common_cache_plan_record & rec) const noexcept {
    if (!rec.optimizer.local_authority.certified_once) return true;
    return local_observations &&
        rec.optimizer.local_authority.authority_currency_serial.state ==
            llama_cache_acct_known::known &&
        local_observations->authority_currency_serial() ==
            rec.optimizer.local_authority.authority_currency_serial.value;
}

bool server_cache_plan_demote_for_coverage_recovery(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        int64_t pos_min,
        int64_t pos_min_threshold) noexcept {
    if (!server_cache_plan_requires_coverage_recovery(
            execution, pos_min, pos_min_threshold)) {
        return false;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability,
        &execution.local_authority);
    execution.clear();
    return true;
}

bool server_cache_plan_demote_for_vbr_low_lcp_reset(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        bool reset_applied) noexcept {
    if (!reset_applied || !execution.authoritative()) {
        return false;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability,
        &execution.local_authority);
    execution.clear();
    return true;
}

bool server_cache_plan_revalidate_checkpoint_execution(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        server_cache_plan_execution & execution,
        size_t checkpoint_count,
        bool eligible,
        int32_t & ordinal) noexcept {
    if (authority.local_currency_current(rec) &&
        server_cache_plan_checkpoint_override_ordinal(
            execution, checkpoint_count, eligible, ordinal)) {
        return true;
    }
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability,
        &execution.local_authority);
    execution.clear();
    return false;
}

void server_cache_plan_authority::finalize_execution(
        common_cache_plan_record & rec,
        server_cache_plan_local_authority_latch * latch) noexcept {
    common_cache_plan_finalize_shadow_authority(rec);
    if (rec.authority.state == common_cache_plan_authority_state::authoritative &&
        rec.authority.executed_plan_candidate !=
            rec.authority.planner_plan_candidate) {
        fallback_legacy(rec,
            common_cache_plan_authority_fallback::internal_fault, latch);
    }
    if (rec.optimizer.local_authority.certified_once &&
        rec.authority.state == common_cache_plan_authority_state::authoritative) {
        if (!latch || !latch->execute(rec.optimizer.local_authority)) {
            fallback_legacy(rec,
                common_cache_plan_authority_fallback::internal_fault, latch);
        }
    }
    counters.observe(rec.authority, rec.authority_prequalified);
}

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
        bool busy,
        bool has_payload,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens) noexcept {
    server_cache_plan_live_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : -1.0f;
    out.reason = busy ? COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY :
                 !has_payload ? COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE :
                 lcp_tokens == 0 ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_live(
        common_cache_plan_candidate * row,
        const server_cache_plan_live_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = evaluation.f_keep >= 0.0f;
    row->note_reject(evaluation.reason);
}

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
        bool payload_present,
        bool identity_matches,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_host_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.payload_bytes = payload_bytes;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : 0.0f;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !identity_matches ? COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH :
                 out.f_keep < 0.25f ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_host(
        common_cache_plan_candidate * row,
        const server_cache_plan_host_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = true;
    row->note_reject(evaluation.reason);
}

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
        bool payload_present,
        bool frontier_current,
        bool recurrent,
        bool representation_matches,
        int64_t pos_min,
        int64_t pos_max,
        int64_t next_position,
        int64_t min_position_threshold,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_checkpoint_evaluation out;
    out.lcp_tokens = pos_max >= 0 ? uint64_t(pos_max) : 0;
    out.payload_bytes = payload_bytes;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !frontier_current ? COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID :
                 !representation_matches ? COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED :
                 recurrent
                    ? (pos_max < next_position
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT)
                    : (pos_max <= next_position &&
                       (pos_min < min_position_threshold || pos_min == 0)
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
    return out;
}

void server_cache_plan_apply_checkpoint(
        common_cache_plan_candidate * row,
        const server_cache_plan_checkpoint_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->note_reject(evaluation.reason);
}
