#include "server-cache-destruction-quote.h"

#include "../../src/llama-sha256.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace {

int32_t host_source(const common_cache_plan_record & rec, int32_t candidate) noexcept {
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

common_cache_plan_destruction_manifest_digest manifest_digest(
        const std::vector<const server_cache_destruction_artifact *> & manifest) {
    llama_sha256_writer hash;
    static constexpr char tag[] = "cache-destruction-manifest-v1";
    hash.string(tag, sizeof(tag) - 1);
    hash.u64(manifest.size());
    for (const auto * artifact : manifest) {
        hash.u64(artifact->candidate.artifact_id.v);
        hash.u32(uint32_t(artifact->kind));
        hash.u32(uint32_t(artifact->pool));
    }
    return common_cache_plan_destruction_manifest_digest::from_sha256(
        hash.finish());
}

common_cache_plan_destruction_effect_digest effect_digest(
        const std::vector<llama_cache_acct_op_id> & ops,
        const llama_cache_acct_release_set_preview & release) {
    llama_sha256_writer hash;
    static constexpr char tag[] = "cache-destruction-union-effect-v1";
    hash.string(tag, sizeof(tag) - 1);
    hash.u64(ops.size());
    for (const auto op : ops) {
        hash.u64(op.v);
    }
    hash.u64(release.rows.size());
    for (const auto & row : release.rows) {
        hash.u32(uint32_t(row.domain.residency));
        hash.u32(uint32_t(row.domain.kind));
        hash.u32(row.domain.topology.v);
        hash.u32(row.domain.device_ordinal.v);
        hash.u64(row.logical_payload);
        hash.u64(row.resident_allocated);
    }
    return common_cache_plan_destruction_effect_digest::from_sha256(
        hash.finish());
}

std::string digest_key(
        const common_cache_plan_destruction_manifest_digest & digest) {
    return std::string(reinterpret_cast<const char *>(digest.bytes().data()),
                       digest.bytes().size());
}

common_cache_plan_destruction_lease_verdict lease_verdict(
        const server_cache_destruction_artifact & artifact) noexcept {
    if (artifact.mandatory_anchor) {
        return common_cache_plan_destruction_lease_verdict::mandatory_recovery;
    }
    if (artifact.candidate.lease.state != server_cache_lease_eval_state::known) {
        return common_cache_plan_destruction_lease_verdict::unavailable;
    }
    switch (artifact.candidate.lease.cls) {
        case server_cache_lease_class::none:
            return common_cache_plan_destruction_lease_verdict::unleased;
        case server_cache_lease_class::soft:
            return common_cache_plan_destruction_lease_verdict::soft_leased;
        case server_cache_lease_class::hard:
            return common_cache_plan_destruction_lease_verdict::hard_leased;
        case server_cache_lease_class::_count:
            return common_cache_plan_destruction_lease_verdict::unavailable;
    }
    return common_cache_plan_destruction_lease_verdict::unavailable;
}

void refuse(common_cache_plan_destruction_receipt & out,
            common_cache_plan_destruction_reason reason) noexcept {
    out.state = common_cache_plan_destruction_state::refused;
    out.reason = reason;
    out.selected_attention.clear();
    out.selected_recurrent.clear();
}

void bind_recovery(
        common_cache_plan_destruction_receipt & quote,
        common_cache_plan_destruction_effect_set effects,
        common_cache_plan_recovery_citation citation) noexcept {
    const auto displacement =
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::destructive_similarity_retarget);
    const bool prospective_cross_target =
        citation == common_cache_plan_recovery_citation::prospective &&
        effects != 0 && (effects & ~displacement) == 0;
    if (citation == common_cache_plan_recovery_citation::resolved ||
        prospective_cross_target) {
        quote.state = common_cache_plan_destruction_state::quoted;
        quote.reason = common_cache_plan_destruction_reason::none;
        quote.displaced_fate = common_cache_plan_displaced_fate::retained_host;
        quote.recovery_citation = citation;
    } else {
        quote.displaced_fate = common_cache_plan_displaced_fate::unavailable;
        quote.recovery_citation = common_cache_plan_recovery_citation::unavailable;
        refuse(quote, common_cache_plan_destruction_reason::recovery_unavailable);
    }
}

bool domain_row_equal(const common_cache_plan_yield_domain & a,
                      const llama_cache_budget_row & b) noexcept {
    const auto value_equal = [](const llama_cache_acct_value & lhs,
                                const llama_cache_acct_value & rhs) {
        return lhs.state == rhs.state && lhs.value == rhs.value;
    };
    return b.resource.kind ==
               llama_cache_budget_resource_kind::accounting_domain &&
           a.domain == b.resource.domain &&
           value_equal(a.current_resident_bytes, b.current_resident) &&
           value_equal(a.fit_before_bytes, b.before) &&
           value_equal(a.projected_release_bytes, b.released) &&
           value_equal(a.projected_reserve_bytes, b.reserved) &&
           value_equal(a.projected_after_bytes, b.after);
}

} // namespace

common_cache_plan_destruction_effect_set server_cache_destruction_effects_for(
        const common_cache_plan_record & rec,
        int32_t candidate,
        int32_t legacy_candidate) noexcept {
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
            // B-A's zero-destruction similarity retarget is already inside the
            // pre-D-A envelope; other independent effects still apply below.
        } else {
            effects |= common_cache_plan_destruction_effect_bit(
                rec.selection == common_cache_plan_selection::similarity
                    ? common_cache_plan_destruction_effect::
                          destructive_similarity_retarget
                    : common_cache_plan_destruction_effect::
                          cross_target_displacement);
        }
    }
    if (planned.target_slot_id == legacy.target_slot_id &&
        planned.provider == common_cache_plan_provider::cold_replay &&
        legacy.provider != common_cache_plan_provider::cold_replay) {
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::same_target_cold_replacement);
    }
    const int32_t planned_host = host_source(rec, candidate);
    const int32_t legacy_host = host_source(rec, legacy_candidate);
    if (planned_host >= 0 && planned_host != legacy_host) {
        effects |= common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
    }
    return effects;
}

bool server_cache_destruction_has_effect(
        const common_cache_plan_record & rec,
        int32_t legacy_candidate) noexcept {
    for (uint32_t i = 0; i < rec.n_inventory; ++i) {
        const auto & candidate = rec.inventory[i];
        if (candidate.viable() && !candidate.component_only &&
            (!candidate.is_chain() || candidate.component_ids[0] >= 0) &&
            common_cache_plan_origin_in_domain(
                candidate.origin_tier, rec.selection) &&
            server_cache_destruction_effects_for(
                rec, int32_t(i), legacy_candidate) != 0) {
            return true;
        }
    }
    return false;
}

bool server_cache_destruction_quote_all(
        common_cache_plan_record & rec,
        int32_t legacy_plan_candidate,
        const std::vector<server_cache_destruction_artifact> & artifacts,
        uint64_t accounting_serial,
        const server_cache_destruction_preview_callback & preview,
        const server_cache_destruction_projection_callback & project,
        const server_cache_destruction_quote_options & options,
        common_cache_plan_destruction_counters & counters) noexcept {
    rec.destruction_quotes.clear();
    rec.destruction = {};
    rec.destruction_legacy_plan_candidate = legacy_plan_candidate;
    rec.destruction.admission_sequence = options.admission_sequence;
    const auto fail_whole_pass = [&](
            common_cache_plan_destruction_state state,
            common_cache_plan_destruction_reason reason) {
        rec.destruction.state = state;
        rec.destruction.reason = reason;
        counters.observe(rec.selection, rec.destruction);
        return false;
    };
    if (!options.lifecycle_available) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::refused,
            common_cache_plan_destruction_reason::lifecycle_disabled);
    }
    if (legacy_plan_candidate < 0 ||
        uint32_t(legacy_plan_candidate) >= rec.n_inventory ||
        rec.inventory_saturated() ||
        artifacts.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::refused,
            common_cache_plan_destruction_reason::manifest_incomplete);
    }
    if (!preview || !project || options.admission_sequence == 0) {
        return fail_whole_pass(
            common_cache_plan_destruction_state::failed,
            common_cache_plan_destruction_reason::internal_fault);
    }
    try {
        std::map<int32_t, std::vector<const server_cache_destruction_artifact *>> by_slot;
        std::map<int32_t, std::vector<const server_cache_destruction_artifact *>> by_host;
        std::array<std::vector<const server_cache_destruction_artifact *>,
                   size_t(common_retention_pool::_count)> pools;
        for (const auto & artifact : artifacts) {
            if (artifact.pool >= common_retention_pool::_count) {
                return fail_whole_pass(
                    common_cache_plan_destruction_state::refused,
                    common_cache_plan_destruction_reason::manifest_incomplete);
            }
            pools[size_t(artifact.pool)].push_back(&artifact);
        }
        const auto artifact_less = [](const auto * a, const auto * b) {
            return a->candidate.artifact_id.v < b->candidate.artifact_id.v;
        };
        for (auto & pool : pools) {
            std::sort(pool.begin(), pool.end(), artifact_less);
        }
        for (const auto & pool : pools) {
            for (const auto * artifact : pool) {
                if (artifact->kind == common_retention_artifact_kind::host_entry &&
                    artifact->host_source_id >= 0) {
                    by_host[artifact->host_source_id].push_back(artifact);
                } else if (artifact->owner_slot >= 0) {
                    by_slot[artifact->owner_slot].push_back(artifact);
                }
            }
        }
        std::unordered_map<std::string, common_cache_plan_destruction_quote> memo;
        memo.reserve(rec.n_inventory);
        rec.destruction_quotes.reserve(rec.n_inventory);
        for (uint32_t i = 0; i < rec.n_inventory; ++i) {
            const auto & candidate = rec.inventory[i];
            if (!candidate.viable() || candidate.component_only ||
                !common_cache_plan_origin_in_domain(
                    candidate.origin_tier, rec.selection) ||
                (candidate.is_chain() && candidate.component_ids[0] < 0)) {
                continue;
            }
            const auto effects = server_cache_destruction_effects_for(
                rec, int32_t(i), legacy_plan_candidate);
            if (effects == 0) {
                continue;
            }

            std::vector<const server_cache_destruction_artifact *> manifest;
            std::unordered_set<uint64_t> manifest_ids;
            const auto add_manifest = [&](const auto & rows) {
                for (const auto * artifact : rows) {
                    if (manifest_ids.insert(
                            artifact->candidate.artifact_id.v).second) {
                        manifest.push_back(artifact);
                    }
                }
            };
            const auto displacement_effects =
                common_cache_plan_destruction_effect_bit(
                    common_cache_plan_destruction_effect::
                        cross_target_displacement) |
                common_cache_plan_destruction_effect_bit(
                    common_cache_plan_destruction_effect::
                        destructive_similarity_retarget) |
                common_cache_plan_destruction_effect_bit(
                    common_cache_plan_destruction_effect::
                        same_target_cold_replacement);
            if ((effects & displacement_effects) != 0) {
                const auto it = by_slot.find(candidate.target_slot_id);
                if (it != by_slot.end()) {
                    add_manifest(it->second);
                }
            }
            if (common_cache_plan_destruction_effect_has(
                    effects,
                    common_cache_plan_destruction_effect::
                        different_host_source_consumption)) {
                const int32_t source = host_source(rec, int32_t(i));
                const auto it = by_host.find(source);
                if (it != by_host.end()) {
                    add_manifest(it->second);
                }
            }
            std::sort(manifest.begin(), manifest.end(), artifact_less);

            common_cache_plan_destruction_quote staged;
            auto & quote = staged.receipt;
            quote.plan_candidate = int32_t(i);
            quote.admission_sequence = options.admission_sequence;
            quote.effects = effects;
            quote.quote_accounting_serial = accounting_serial;
            if (manifest.empty()) {
                refuse(quote, common_cache_plan_destruction_reason::manifest_incomplete);
                counters.observe(rec.selection, quote);
                rec.destruction_quotes.push_back(std::move(staged));
                continue;
            }
            quote.manifest_digest = manifest_digest(manifest);
            const auto key = digest_key(quote.manifest_digest);
            const auto cached = memo.find(key);
            if (cached != memo.end()) {
                const int32_t plan_candidate = quote.plan_candidate;
                const auto candidate_effects = quote.effects;
                staged = cached->second;
                auto & cached_receipt = staged.receipt;
                cached_receipt.plan_candidate = plan_candidate;
                cached_receipt.effects = candidate_effects;
                if (cached_receipt.reason ==
                        common_cache_plan_destruction_reason::none ||
                    cached_receipt.reason ==
                        common_cache_plan_destruction_reason::recovery_unavailable) {
                    bind_recovery(
                        cached_receipt, candidate_effects,
                        options.recovery_citation);
                }
                counters.quote_memo_hits++;
                counters.observe(rec.selection, cached_receipt);
                rec.destruction_quotes.push_back(std::move(staged));
                continue;
            }
            counters.quote_memo_misses++;

            std::vector<llama_cache_acct_op_id> ops;
            std::unordered_set<uint64_t> op_ids;
            bool unavailable = false;
            quote.lease_verdict = common_cache_plan_destruction_lease_verdict::unleased;
            for (const auto * artifact : manifest) {
                if (!artifact->candidate.identity_known) {
                    refuse(quote, common_cache_plan_destruction_reason::identity_unavailable);
                    unavailable = true;
                    break;
                }
                if (artifact->candidate.availability !=
                        server_retention_candidate_availability::available) {
                    refuse(quote, common_cache_plan_destruction_reason::manifest_incomplete);
                    unavailable = true;
                    break;
                }
                const auto verdict = lease_verdict(*artifact);
                if (verdict == common_cache_plan_destruction_lease_verdict::mandatory_recovery) {
                    quote.lease_verdict = verdict;
                    refuse(quote, common_cache_plan_destruction_reason::mandatory_anchor);
                    unavailable = true;
                    break;
                }
                if (verdict == common_cache_plan_destruction_lease_verdict::unavailable) {
                    quote.lease_verdict = verdict;
                    refuse(quote, common_cache_plan_destruction_reason::lease_unavailable);
                    unavailable = true;
                    break;
                }
                if (verdict == common_cache_plan_destruction_lease_verdict::hard_leased) {
                    quote.lease_verdict = verdict;
                    refuse(quote, common_cache_plan_destruction_reason::hard_lease_blocked);
                    unavailable = true;
                    break;
                }
                if (verdict == common_cache_plan_destruction_lease_verdict::soft_leased) {
                    quote.lease_verdict = verdict;
                }
                (artifact->pool == common_retention_pool::attention
                    ? quote.selected_attention : quote.selected_recurrent)
                    .push_back(artifact->candidate.artifact_id);
                for (const auto op : artifact->candidate.release_ops) {
                    if (!op) {
                        refuse(quote, common_cache_plan_destruction_reason::release_evidence_unavailable);
                        unavailable = true;
                        break;
                    }
                    if (!op_ids.insert(op.v).second) {
                        continue;
                    }
                    ops.push_back(op);
                }
                if (unavailable) {
                    break;
                }
            }
            if (!unavailable && ops.empty()) {
                refuse(quote, common_cache_plan_destruction_reason::release_evidence_unavailable);
                unavailable = true;
            }
            std::sort(ops.begin(), ops.end(), [](auto a, auto b) { return a.v < b.v; });
            llama_cache_acct_release_set_preview released;
            if (!unavailable && !preview(ops, accounting_serial, released)) {
                refuse(quote, common_cache_plan_destruction_reason::accounting_unavailable);
                unavailable = true;
            }
            if (!unavailable) {
                quote.union_effect_digest = effect_digest(ops, released);
                if (!project(released, staged.projected_domains)) {
                    refuse(quote, common_cache_plan_destruction_reason::capacity_refused);
                    unavailable = true;
                }
            }
            if (!unavailable) {
                bind_recovery(quote, effects, options.recovery_citation);
            }
            memo.emplace(key, staged);
            counters.observe(rec.selection, quote);
            rec.destruction_quotes.push_back(std::move(staged));
        }
        return true;
    } catch (...) {
        rec.destruction_quotes.clear();
        return fail_whole_pass(
            common_cache_plan_destruction_state::failed,
            common_cache_plan_destruction_reason::internal_fault);
    }
}

void server_cache_destruction_select_quote(
        common_cache_plan_record & rec,
        common_cache_plan_destruction_counters & counters) noexcept {
    if (rec.destruction_quotes.empty()) {
        return;
    }
    if (rec.shadow_choice < 0) {
        rec.destruction.state = common_cache_plan_destruction_state::failed;
        rec.destruction.reason =
            common_cache_plan_destruction_reason::internal_fault;
        rec.destruction.selected_attention.clear();
        rec.destruction.selected_recurrent.clear();
        counters.observe(rec.selection, rec.destruction);
        return;
    }
    const auto it = std::find_if(
        rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
        [&](const auto & quote) {
            return quote.receipt.plan_candidate == rec.shadow_choice;
        });
    if (it != rec.destruction_quotes.end()) {
        const uint64_t duration = rec.destruction.quote_duration_us;
        rec.destruction = it->receipt;
        rec.destruction.quote_duration_us = duration;
        return;
    }
    if (server_cache_destruction_effects_for(
            rec, rec.shadow_choice,
            rec.destruction_legacy_plan_candidate) == 0) {
        const uint64_t duration = rec.destruction.quote_duration_us;
        const uint64_t sequence = rec.destruction.admission_sequence;
        rec.destruction = {};
        rec.destruction.quote_duration_us = duration;
        rec.destruction.admission_sequence = sequence;
        return;
    }
    rec.destruction.state = common_cache_plan_destruction_state::failed;
    rec.destruction.reason = common_cache_plan_destruction_reason::internal_fault;
    rec.destruction.selected_attention.clear();
    rec.destruction.selected_recurrent.clear();
    counters.observe(rec.selection, rec.destruction);
}

void server_cache_destruction_finalize_projection(
        common_cache_plan_record & rec,
        const server_cache_yield_result & yield) noexcept {
    auto & receipt = rec.destruction;
    const auto quoted = std::find_if(
        rec.destruction_quotes.begin(), rec.destruction_quotes.end(),
        [&](const auto & item) {
            return item.receipt.plan_candidate == receipt.plan_candidate;
        });
    if (receipt.state != common_cache_plan_destruction_state::quoted ||
        !receipt.union_effect_digest.valid() ||
        quoted == rec.destruction_quotes.end() ||
        quoted->projected_domains.empty()) {
        return;
    }
    if (yield.status == server_cache_yield_status::fits &&
        yield.projected_fit.state == llama_cache_budget_fit_state::fits) {
        std::vector<llama_cache_acct_artifact_id> quote_ids = receipt.selected_attention;
        quote_ids.insert(quote_ids.end(), receipt.selected_recurrent.begin(), receipt.selected_recurrent.end());
        std::vector<llama_cache_acct_artifact_id> yield_ids = yield.selected[size_t(common_retention_pool::attention)];
        yield_ids.insert(yield_ids.end(), yield.selected[size_t(common_retention_pool::recurrent)].begin(),
                         yield.selected[size_t(common_retention_pool::recurrent)].end());
        const auto by_id = [](auto a, auto b) { return a.v < b.v; };
        std::sort(quote_ids.begin(), quote_ids.end(), by_id);
        std::sort(yield_ids.begin(), yield_ids.end(), by_id);
        bool same = quote_ids == yield_ids &&
                    quoted->projected_domains.size() == yield.projected_fit.domains.size();
        for (const auto & row : quoted->projected_domains) {
            const auto it = std::find_if(
                yield.projected_fit.domains.begin(), yield.projected_fit.domains.end(),
                [&](const auto & candidate) { return domain_row_equal(row, candidate); });
            same = same && it != yield.projected_fit.domains.end();
        }
        receipt.post_finalize_comparison = same
            ? common_cache_plan_destruction_comparison::matched
            : common_cache_plan_destruction_comparison::differed;
    } else if (yield.status ==
                   server_cache_yield_status::insufficient_yield) {
        receipt.post_finalize_comparison = common_cache_plan_destruction_comparison::
            ds6_insufficient_yield;
    } else if (yield.status ==
                   server_cache_yield_status::unsupported_required) {
        receipt.post_finalize_comparison = common_cache_plan_destruction_comparison::
            ds6_unsupported_required;
    } else {
        receipt.post_finalize_comparison =
            common_cache_plan_destruction_comparison::ds6_unavailable;
    }

    // Schema v6 defines the selected D-A quote as the sole projected-byte
    // source once its exact union is available. D-S6 remains an independent
    // comparator: its complete result becomes matched/differed above and any
    // incomplete verdict becomes comparison=unavailable. We intentionally do
    // not serialize a second D-S6 status/byte table. Actual remains explicitly
    // not_observed because D-A0a never mutates.
    rec.yield.status = common_cache_plan_yield_status::fits;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::planned;
    rec.yield.actual_state = common_cache_plan_yield_actual_state::not_observed;
    rec.yield.yield_policy_version = yield.yield_policy_version;
    rec.yield.accounting_serial = rec.acct.serial;
    rec.yield.selected_attention = receipt.selected_attention;
    rec.yield.selected_recurrent = receipt.selected_recurrent;
    rec.yield.projected_domains = quoted->projected_domains;
    rec.yield.actual_domains.clear();
}

bool server_cache_destruction_effect_matches(
        const common_cache_plan_destruction_receipt & quote,
        const common_cache_plan_destruction_effect_digest & current_effect,
        const std::vector<common_cache_plan_yield_domain> & quoted_domains,
        const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept {
    if (!quote.union_effect_digest.valid() || !current_effect.valid() ||
        quote.union_effect_digest != current_effect ||
        quoted_domains.size() != current_domains.size()) {
        return false;
    }
    for (const auto & row : quoted_domains) {
        const auto it = std::find_if(
            current_domains.begin(), current_domains.end(),
            [&](const auto & current) {
                return current.domain == row.domain &&
                       current.current_resident_bytes.state ==
                           row.current_resident_bytes.state &&
                       current.current_resident_bytes.value ==
                           row.current_resident_bytes.value &&
                       current.fit_before_bytes.state ==
                           row.fit_before_bytes.state &&
                       current.fit_before_bytes.value ==
                           row.fit_before_bytes.value &&
                       current.projected_release_bytes.state ==
                           row.projected_release_bytes.state &&
                       current.projected_release_bytes.value ==
                           row.projected_release_bytes.value &&
                       current.projected_reserve_bytes.state ==
                           row.projected_reserve_bytes.state &&
                       current.projected_reserve_bytes.value ==
                           row.projected_reserve_bytes.value &&
                       current.projected_after_bytes.state ==
                           row.projected_after_bytes.state &&
                       current.projected_after_bytes.value ==
                           row.projected_after_bytes.value;
            });
        if (it == current_domains.end()) {
            return false;
        }
    }
    return true;
}

common_cache_plan_destruction_reason server_cache_destruction_effect_recheck(
        const common_cache_plan_destruction_receipt & quote,
        const common_cache_plan_destruction_effect_digest & current_effect,
        const std::vector<common_cache_plan_yield_domain> & quoted_domains,
        const std::vector<common_cache_plan_yield_domain> & current_domains) noexcept {
    return server_cache_destruction_effect_matches(
               quote, current_effect, quoted_domains, current_domains)
        ? common_cache_plan_destruction_reason::none
        : common_cache_plan_destruction_reason::effect_drift;
}
