#include "server-cache-yield.h"

#include <algorithm>
#include <limits>
#include <tuple>

namespace {

bool op_union_add(
        std::vector<llama_cache_acct_op_id> & selected,
        const std::vector<llama_cache_acct_op_id> & added) {
    for (const auto op : added) {
        if (!op) {
            return false;
        }
        if (std::find(selected.begin(), selected.end(), op) ==
                selected.end()) {
            if (selected.size() == selected.max_size()) {
                return false;
            }
            selected.push_back(op);
        }
    }
    return true;
}

bool release_plan(
        const llama_cache_acct_release_set_preview & release,
        uint64_t accounting_serial,
        llama_cache_budget_plan & plan) {
    if (release.accounting_serial != accounting_serial) {
        return false;
    }
    plan = {};
    plan.accounting_serial = accounting_serial;
    try {
        plan.entries.reserve(release.rows.size());
        for (const auto & row : release.rows) {
            plan.entries.push_back({
                row.domain, 0, row.resident_allocated,
            });
        }
        return true;
    } catch (...) {
        plan = {};
        return false;
    }
}

}

const char * server_cache_yield_status_name(
        server_cache_yield_status status) noexcept {
    switch (status) {
        case server_cache_yield_status::fits:
            return "fits";
        case server_cache_yield_status::insufficient_yield:
            return "insufficient_yield";
        case server_cache_yield_status::unsupported_required:
            return "unsupported_required";
        case server_cache_yield_status::unavailable:
            return "unavailable";
        case server_cache_yield_status::_count:
            return "invalid";
    }
    return "invalid";
}

bool server_cache_yield_assemble(
        const std::vector<server_retention_candidate> & catalog,
        server_cache_lease_table & leases,
        const server_cache_yield_candidate_resolver & resolver,
        std::vector<server_cache_yield_candidate> & out) noexcept {
    out.clear();
    if (!resolver || catalog.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES) {
        return false;
    }
    try {
        out.reserve(catalog.size());
        for (const auto & source : catalog) {
            server_cache_yield_candidate candidate;
            candidate.artifact_id = source.artifact_id;
            candidate.record = source.record;
            candidate.availability = source.avail;
            candidate.release_ops = source.release_ops;
            server_cache_lease_identity identity;
            bool identity_known = false;
            resolver(source, candidate, identity, identity_known);
            candidate.identity_known = identity_known;
            candidate.lease = leases.inspect(
                candidate.artifact_id, identity);
            out.push_back(std::move(candidate));
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

server_cache_yield_result server_cache_yield_plan(
        const std::vector<server_cache_yield_candidate> & candidates,
        uint64_t accounting_serial,
        const server_cache_yield_preview_callback & preview,
        const server_cache_yield_fits_callback & fits,
        uint32_t policy_version) noexcept {
    server_cache_yield_result result;
    result.accounting_serial = accounting_serial;
    result.yield_policy_version = policy_version;
    const auto mark_unavailable = [&]() {
        result.status = server_cache_yield_status::unavailable;
        result.selected = {};
        result.plan.clear();
        result.unsupported.clear();
    };
    if (policy_version != SERVER_CACHE_YIELD_POLICY_VERSION ||
        candidates.size() > SERVER_CACHE_YIELD_MAX_CANDIDATES ||
        !preview || !fits) {
        mark_unavailable();
        return result;
    }

    try {
        std::array<std::vector<const server_cache_yield_candidate *>,
                   size_t(common_retention_pool::_count)> pools;
        bool unavailable_evidence = false;
        for (const auto & candidate : candidates) {
            if (candidate.has_unsupported_host_spill) {
                result.unsupported.push_back(candidate.artifact_id);
            }
            if (!candidate.artifact_id.v ||
                candidate.availability >=
                    server_retention_candidate_availability::_count ||
                candidate.availability !=
                    server_retention_candidate_availability::available ||
                !candidate.identity_known ||
                candidate.lease.state != server_cache_lease_eval_state::known ||
                candidate.lease.cls >= server_cache_lease_class::_count ||
                candidate.lease.eligibility >=
                    server_cache_lease_eligibility::_count ||
                !candidate.record.valid() ||
                candidate.record.stamp.state !=
                    common_retention_score_state::known ||
                candidate.record.stamp.soft_leased ||
                candidate.record.stamp.pool >= common_retention_pool::_count) {
                unavailable_evidence = true;
                continue;
            }
            if (candidate.record.stamp.mandatory_anchor ||
                candidate.lease.cls == server_cache_lease_class::hard ||
                candidate.lease.eligibility ==
                    server_cache_lease_eligibility::hard_blocked) {
                continue;
            }
            if (candidate.release_ops.empty()) {
                unavailable_evidence = true;
                continue;
            }

            // Validate the operation citations before they enter the order. The byte
            // result is deliberately discarded: only the selected UNION is priced.
            llama_cache_acct_release_set_preview validation;
            if (!preview(
                    candidate.release_ops, accounting_serial, validation)) {
                unavailable_evidence = true;
                continue;
            }
            pools[size_t(candidate.record.stamp.pool)].push_back(&candidate);
        }

        for (auto & pool : pools) {
            std::sort(pool.begin(), pool.end(),
                [](const auto & a, const auto & b) {
                    const auto & as = a->record.stamp;
                    const auto & bs = b->record.stamp;
                    const bool a_soft =
                        a->lease.cls == server_cache_lease_class::soft;
                    const bool b_soft =
                        b->lease.cls == server_cache_lease_class::soft;
                    return std::tie(
                               a_soft, as.anchor_rank,
                               as.recency_ordinal, as.coverage_tokens,
                               as.stable_id) <
                           std::tie(
                               b_soft, bs.anchor_rank,
                               bs.recency_ordinal, bs.coverage_tokens,
                               bs.stable_id);
                });
        }

        llama_cache_budget_plan budget_plan;
        budget_plan.accounting_serial = accounting_serial;
        auto fit = fits(budget_plan);
        if (fit.accounting_serial != accounting_serial ||
            fit.state == llama_cache_budget_fit_state::unavailable) {
            mark_unavailable();
            return result;
        }
        if (fit.state == llama_cache_budget_fit_state::fits) {
            result.status = server_cache_yield_status::fits;
            return result;
        }

        std::vector<llama_cache_acct_op_id> selected_ops;
        for (const auto pool_kind : {
                common_retention_pool::attention,
                common_retention_pool::recurrent }) {
            for (const auto * candidate : pools[size_t(pool_kind)]) {
                const size_t n_ops_before = selected_ops.size();
                if (!op_union_add(
                        selected_ops, candidate->release_ops)) {
                    mark_unavailable();
                    return result;
                }
                if (selected_ops.size() == n_ops_before) {
                    continue;
                }
                llama_cache_acct_release_set_preview released;
                if (!preview(selected_ops, accounting_serial, released) ||
                    !release_plan(
                        released, accounting_serial, budget_plan)) {
                    mark_unavailable();
                    return result;
                }
                fit = fits(budget_plan);
                if (fit.accounting_serial != accounting_serial ||
                    fit.state == llama_cache_budget_fit_state::unavailable) {
                    mark_unavailable();
                    return result;
                }
                result.selected[size_t(pool_kind)].push_back(
                    candidate->artifact_id);
                result.plan = budget_plan.entries;
                if (fit.state == llama_cache_budget_fit_state::fits) {
                    result.status = server_cache_yield_status::fits;
                    return result;
                }
            }
        }

        // These known terminals require an exclusively priceable/eligible
        // catalog. A host-entry-only catalog can reach insufficient_yield;
        // common mixed live catalogs remain unavailable while slots/checkpoints
        // lack exact operation ownership. unsupported_required additionally
        // awaits an available spill producer (milestone F).
        if (unavailable_evidence) {
            mark_unavailable();
            return result;
        } else if (!result.unsupported.empty()) {
            result.status =
                server_cache_yield_status::unsupported_required;
        } else {
            result.status =
                server_cache_yield_status::insufficient_yield;
        }
        return result;
    } catch (...) {
        result.status = server_cache_yield_status::unavailable;
        result.selected = {};
        result.plan.clear();
        result.unsupported.clear();
        return result;
    }
}
