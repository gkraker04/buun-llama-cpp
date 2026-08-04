#include "server-cache-destruction-quote.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);

static common_cache_plan_record record_with_cold_candidates() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * live = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    live->accept();
    live->f_keep = 0.25;
    live->f_keep_known = true;
    auto * cold_a = rec.find_or_add(
        common_cache_plan_provider::cold_replay, 10,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    cold_a->accept();
    auto * cold_b = rec.find_or_add(
        common_cache_plan_provider::cold_replay, 11,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    cold_b->accept();
    return rec;
}

static server_cache_destruction_artifact artifact(uint64_t id, uint64_t op) {
    server_cache_destruction_artifact out;
    out.candidate.artifact_id = { id };
    out.kind = common_retention_artifact_kind::live_slot;
    out.owner_slot = 0;
    out.pool = common_retention_pool::attention;
    out.candidate.identity_known = true;
    out.candidate.availability =
        server_retention_candidate_availability::available;
    out.candidate.lease.state = server_cache_lease_eval_state::known;
    out.candidate.lease.cls = server_cache_lease_class::none;
    out.candidate.lease.eligibility =
        server_cache_lease_eligibility::eligible;
    out.candidate.release_ops = { llama_cache_acct_op_id{op} };
    return out;
}

static server_cache_destruction_preview_callback preview(uint64_t serial) {
    return [serial](const auto & ops, uint64_t expected, auto & out) {
        out = {};
        if (expected != serial || ops.empty()) {
            return false;
        }
        out.accounting_serial = serial;
        out.rows.push_back({ HOST, 64, 64 });
        return true;
    };
}

static server_cache_destruction_projection_callback project() {
    return [](const auto & released, auto & out) {
        out.clear();
        for (const auto & row : released.rows) {
            llama_cache_budget_row fit;
            fit.resource.kind =
                llama_cache_budget_resource_kind::accounting_domain;
            fit.resource.domain = row.domain;
            fit.current_resident = llama_cache_acct_value::measured(128);
            fit.before = llama_cache_acct_value::measured(128);
            fit.released =
                llama_cache_acct_value::measured(row.resident_allocated);
            fit.reserved = llama_cache_acct_value::measured(0);
            fit.after = llama_cache_acct_value::measured(64);
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(fit, lowered)) {
                return false;
            }
            out.push_back(std::move(lowered));
        }
        return true;
    };
}

static void test_complete_memoized_and_permutation() {
    auto rec = record_with_cold_candidates();
    std::vector<server_cache_destruction_artifact> artifacts = {
        artifact(2, 22), artifact(1, 11),
    };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, artifacts, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() == 2);
    CHECK(rec.destruction_quotes[0].receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::none);
    CHECK(rec.destruction_quotes[0].receipt.selected_attention.size() == 2);
    CHECK(counters.quote_memo_misses == 1);
    CHECK(counters.quote_memo_hits == 1);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    const size_t cls = size_t(common_cache_plan_destruction_class::slot_drop);
    CHECK(counters.quoted[tier][cls] == 2);
    CHECK(counters.lease_verdict[tier][size_t(
              common_cache_plan_destruction_lease_verdict::unleased)] == 2);
    CHECK(!counters.has_receipt);
    const auto digest = rec.destruction_quotes[0].receipt.manifest_digest;
    const auto effect = rec.destruction_quotes[0].receipt.union_effect_digest;
    const auto domains = rec.destruction_quotes[0].projected_domains;
    CHECK(server_cache_destruction_effect_matches(
        rec.destruction_quotes[0].receipt, effect, domains, domains));

    std::reverse(artifacts.begin(), artifacts.end());
    common_cache_plan_destruction_counters permuted_counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, artifacts, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 2 }, permuted_counters));
    CHECK(rec.destruction_quotes[0].receipt.manifest_digest == digest);

    auto changed = common_cache_plan_destruction_effect_digest::from_sha256(
        std::array<uint8_t, 32>{ 1 });
    CHECK(!server_cache_destruction_effect_matches(
        rec.destruction_quotes[0].receipt, changed, domains, domains));
    CHECK(server_cache_destruction_effect_recheck(
        rec.destruction_quotes[0].receipt, changed, domains, domains) ==
        common_cache_plan_destruction_reason::effect_drift);
    auto later = rec.destruction_quotes[0].receipt;
    later.quote_accounting_serial = 99;
    CHECK(server_cache_destruction_effect_matches(
        later, effect, domains, domains));
    CHECK(server_cache_destruction_effect_recheck(
        later, effect, domains, domains) ==
        common_cache_plan_destruction_reason::none);
}

static void test_fail_closed_matrix() {
    struct cell {
        common_cache_plan_destruction_reason reason;
        void (*mutate)(server_cache_destruction_artifact &);
    };
    const cell cells[] = {
        { common_cache_plan_destruction_reason::identity_unavailable,
          [](auto & a) { a.candidate.identity_known = false; } },
        { common_cache_plan_destruction_reason::manifest_incomplete,
          [](auto & a) { a.candidate.availability = server_retention_candidate_availability::backing_missing_or_stale; } },
        { common_cache_plan_destruction_reason::mandatory_anchor,
          [](auto & a) { a.mandatory_anchor = true; } },
        { common_cache_plan_destruction_reason::lease_unavailable,
          [](auto & a) { a.candidate.lease.state = server_cache_lease_eval_state::unavailable; } },
        { common_cache_plan_destruction_reason::hard_lease_blocked,
          [](auto & a) { a.candidate.lease.cls = server_cache_lease_class::hard; a.candidate.lease.eligibility = server_cache_lease_eligibility::hard_blocked; } },
        { common_cache_plan_destruction_reason::release_evidence_unavailable,
          [](auto & a) { a.candidate.release_ops.clear(); } },
    };
    for (const auto & cell : cells) {
        auto rec = record_with_cold_candidates();
        auto a = artifact(1, 11);
        cell.mutate(a);
        common_cache_plan_destruction_counters counters;
        CHECK(server_cache_destruction_quote_all(
            rec, 0, { a }, 17, preview(17), project(),
            { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
        CHECK(!rec.destruction_quotes.empty());
        CHECK(rec.destruction_quotes[0].receipt.state ==
              common_cache_plan_destruction_state::refused);
        CHECK(rec.destruction_quotes[0].receipt.reason == cell.reason);
    }

    auto rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17,
        [](const auto &, uint64_t, auto &) { return false; },
        project(), { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::accounting_unavailable);

    rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17),
        [](const auto &, auto &) { return false; },
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::capacity_refused);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    const size_t slot_drop = size_t(
        common_cache_plan_destruction_class::slot_drop);
    CHECK(counters.quoted[tier][slot_drop] == 0);
    CHECK(counters.refused[tier][size_t(
              common_cache_plan_destruction_reason::capacity_refused)] == 2);
    CHECK(rec.destruction_quotes[0].receipt.selected_attention.empty());
    CHECK(rec.destruction_quotes[0].receipt.selected_recurrent.empty());

    rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters whole_counters;
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, {}, 17, preview(17), project(),
        { false, common_cache_plan_recovery_citation::unavailable, 1 },
        whole_counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::lifecycle_disabled);
    CHECK(whole_counters.refused[size_t(common_cache_plan_selection::lru)]
          [size_t(common_cache_plan_destruction_reason::lifecycle_disabled)] == 1);

    std::vector<server_cache_destruction_artifact> overflow(
        SERVER_CACHE_YIELD_MAX_CANDIDATES + 1, artifact(1, 11));
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, overflow, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 },
        whole_counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
    CHECK(whole_counters.refused[size_t(common_cache_plan_selection::lru)]
          [size_t(common_cache_plan_destruction_reason::manifest_incomplete)] == 1);
    rec = record_with_cold_candidates();
    rec.derived_plans_incomplete = true;
    CHECK(!server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
}

static void test_refusal_mapping_and_selection() {
    auto rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_has_effect(rec, 0));
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::same_target_cold_replacement));
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::unavailable, 1 }, counters));
    CHECK(rec.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    auto prospective_same = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        prospective_same, 0, { artifact(1, 11) }, 17,
        preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 2 },
        counters));
    CHECK(prospective_same.destruction_quotes[0].receipt.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 3 }, counters));
    rec.shadow_choice = 1;
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.plan_candidate == 1);
    CHECK(common_cache_plan_destruction_effect_has(
        rec.destruction.effects,
        common_cache_plan_destruction_effect::same_target_cold_replacement));

    server_cache_yield_result yield;
    yield.status = server_cache_yield_status::fits;
    yield.yield_policy_version = 1;
    yield.accounting_serial = 17;
    yield.selected[size_t(common_retention_pool::attention)] = { { 1 } };
    yield.projected_fit.state = llama_cache_budget_fit_state::fits;
    yield.projected_fit.accounting_serial = 17;
    llama_cache_budget_row projected;
    projected.resource.kind =
        llama_cache_budget_resource_kind::accounting_domain;
    projected.resource.domain = HOST;
    projected.current_resident = llama_cache_acct_value::measured(128);
    projected.before = llama_cache_acct_value::measured(128);
    projected.released = llama_cache_acct_value::measured(64);
    projected.reserved = llama_cache_acct_value::measured(0);
    projected.after = llama_cache_acct_value::measured(64);
    yield.projected_fit.domains.push_back(projected);
    rec.acct.serial = 23;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::matched);
    CHECK(rec.yield.actual_state ==
          common_cache_plan_yield_actual_state::not_observed);
    CHECK(rec.yield.projected_domains.size() == 1);
    CHECK(rec.yield.accounting_serial == rec.acct.serial);

    yield.selected[size_t(common_retention_pool::attention)] = { { 99 } };
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::differed);

    yield.selected[size_t(common_retention_pool::attention)] = { { 1 } };
    yield.projected_fit.domains[0].reserved =
        llama_cache_acct_value::measured(1);
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::differed);
    yield.projected_fit.domains[0].reserved =
        llama_cache_acct_value::measured(0);

    yield.status = server_cache_yield_status::insufficient_yield;
    rec.yield.unsupported = { { 77 } };
    rec.yield.projected_domains.clear();
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::ds6_insufficient_yield);
    CHECK(rec.yield.projected_domains.size() == 1);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 77 } }));
    CHECK(rec.yield.actual_state ==
          common_cache_plan_yield_actual_state::not_observed);

    yield.status = server_cache_yield_status::unsupported_required;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::
              ds6_unsupported_required);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 77 } }));
    yield.status = server_cache_yield_status::unavailable;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::ds6_unavailable);

    rec.selection = common_cache_plan_selection::similarity;
    rec.inventory[1].target_slot_id = 1;
    rec.inventory[1].provider = common_cache_plan_provider::live_slot;
    rec.inventory[1].origin_tier = common_cache_plan_selection::similarity;
    rec.inventory[1].f_keep_known = true;
    rec.inventory[1].f_keep = 0.5;
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::destructive_similarity_retarget));
    auto cross_artifact = artifact(3, 33);
    cross_artifact.owner_slot = 1;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { cross_artifact }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::prospective, 3 },
        counters));
    CHECK(rec.destruction_quotes[0].receipt.state ==
          common_cache_plan_destruction_state::quoted);
    CHECK(rec.destruction_quotes[0].receipt.recovery_citation ==
          common_cache_plan_recovery_citation::prospective);
    rec.selection = common_cache_plan_selection::route_home;
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(rec, 1, 0),
        common_cache_plan_destruction_effect::cross_target_displacement));

    common_cache_plan_record host_rec;
    host_rec.selection = common_cache_plan_selection::lru;
    auto * legacy_host = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 10,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    auto * planned_host = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 20,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(legacy_host != nullptr);
    CHECK(planned_host != nullptr);
    legacy_host->accept();
    planned_host->accept();
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(host_rec, 1, 0),
        common_cache_plan_destruction_effect::different_host_source_consumption));

    common_cache_plan_record nondestructive;
    nondestructive.selection = common_cache_plan_selection::lru;
    auto * only = nondestructive.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(only != nullptr);
    only->accept();
    CHECK(!server_cache_destruction_has_effect(nondestructive, 0));
}

static void test_plural_effect_union_and_counters() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * legacy = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    auto * planned = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 20,
        COMMON_CACHE_PLAN_PHASE_LRU, 1,
        common_cache_plan_selection::lru);
    CHECK(legacy != nullptr);
    CHECK(planned != nullptr);
    legacy->accept();
    planned->accept();

    const auto effects = server_cache_destruction_effects_for(rec, 1, 0);
    CHECK(common_cache_plan_destruction_effect_has(
        effects,
        common_cache_plan_destruction_effect::cross_target_displacement));
    CHECK(common_cache_plan_destruction_effect_has(
        effects,
        common_cache_plan_destruction_effect::different_host_source_consumption));

    auto live = artifact(1, 11);
    live.owner_slot = 1;
    auto host = artifact(2, 22);
    host.kind = common_retention_artifact_kind::host_entry;
    host.owner_slot = -1;
    host.host_source_id = 20;
    uint64_t preview_ops = 0;
    const auto preview_union = [&](const auto & ops, uint64_t serial, auto & out) {
        preview_ops = ops.size();
        return preview(17)(ops, serial, out);
    };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { live, host }, 17, preview_union, project(),
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() == 1);
    const auto & receipt = rec.destruction_quotes[0].receipt;
    CHECK(receipt.state == common_cache_plan_destruction_state::quoted);
    CHECK(preview_ops == 2);
    CHECK(receipt.selected_attention.size() == 2);
    const size_t tier = size_t(common_cache_plan_selection::lru);
    CHECK(counters.quoted[tier][size_t(
              common_cache_plan_destruction_class::slot_drop)] == 1);
    CHECK(counters.quoted[tier][size_t(
              common_cache_plan_destruction_class::host_artifact_drop)] == 1);
}

static void test_refused_projection_and_selection_failure() {
    auto rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17),
        [](const auto &, auto &) { return false; },
        { true, common_cache_plan_recovery_citation::resolved, 9 }, counters));
    rec.shadow_choice = 1;
    rec.destruction.quote_duration_us = 41;
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.state == common_cache_plan_destruction_state::refused);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::capacity_refused);
    CHECK(rec.destruction.admission_sequence == 9);
    CHECK(rec.destruction.quote_duration_us == 41);

    rec.yield.status = common_cache_plan_yield_status::insufficient_yield;
    rec.yield.plan_state = common_cache_plan_yield_plan_state::unavailable;
    rec.yield.accounting_serial = 55;
    rec.yield.unsupported = { { 7 } };
    server_cache_yield_result yield;
    yield.status = server_cache_yield_status::fits;
    yield.projected_fit.state = llama_cache_budget_fit_state::fits;
    server_cache_destruction_finalize_projection(rec, yield);
    CHECK(rec.destruction.post_finalize_comparison ==
          common_cache_plan_destruction_comparison::not_compared);
    CHECK(rec.yield.status ==
          common_cache_plan_yield_status::insufficient_yield);
    CHECK(rec.yield.plan_state ==
          common_cache_plan_yield_plan_state::unavailable);
    CHECK(rec.yield.accounting_serial == 55);
    CHECK(rec.yield.unsupported ==
          std::vector<llama_cache_acct_artifact_id>({ { 7 } }));

    rec = record_with_cold_candidates();
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 10 }, counters));
    rec.shadow_choice = -1;
    rec.destruction.quote_duration_us = 42;
    const auto before = counters.refused[size_t(rec.selection)][size_t(
        common_cache_plan_destruction_reason::internal_fault)];
    server_cache_destruction_select_quote(rec, counters);
    CHECK(rec.destruction.state == common_cache_plan_destruction_state::failed);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::internal_fault);
    CHECK(rec.destruction.admission_sequence == 10);
    CHECK(rec.destruction.quote_duration_us == 42);
    CHECK(counters.refused[size_t(rec.selection)][size_t(
              common_cache_plan_destruction_reason::internal_fault)] == before + 1);

    rec = record_with_cold_candidates();
    common_cache_plan_destruction_counters selected_counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17, preview(17), project(),
        { true, common_cache_plan_recovery_citation::resolved, 11 },
        selected_counters));
    CHECK(!rec.destruction_quotes.empty());
    rec.shadow_choice = 0;
    rec.destruction.quote_duration_us = 43;
    server_cache_destruction_select_quote(rec, selected_counters);
    CHECK(rec.destruction.state ==
          common_cache_plan_destruction_state::not_required);
    CHECK(rec.destruction.reason ==
          common_cache_plan_destruction_reason::none);
    CHECK(rec.destruction.quote_duration_us == 43);
    CHECK(rec.destruction.admission_sequence == 11);
    CHECK(selected_counters.refused[size_t(rec.selection)][size_t(
              common_cache_plan_destruction_reason::internal_fault)] == 0);
}

static void test_max_inventory_memoizes_one_manifest() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::lru;
    auto * legacy = rec.find_or_add(
        common_cache_plan_provider::live_slot, 0,
        COMMON_CACHE_PLAN_PHASE_LRU, 0,
        common_cache_plan_selection::lru);
    CHECK(legacy != nullptr);
    legacy->accept();
    for (size_t i = 1; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * candidate = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i),
            COMMON_CACHE_PLAN_PHASE_LRU, 0,
            common_cache_plan_selection::lru);
        CHECK(candidate != nullptr);
        candidate->accept();
    }
    CHECK(rec.n_inventory == COMMON_CACHE_PLAN_MAX_CANDIDATES);
    CHECK(!rec.inventory_saturated());

    uint64_t preview_calls = 0;
    uint64_t project_calls = 0;
    const server_cache_destruction_preview_callback preview_once =
        [&](const auto & ops, uint64_t serial, auto & out) {
            preview_calls++;
            out = {};
            out.accounting_serial = serial;
            CHECK(ops.size() == 1);
            out.rows.push_back({ HOST, 64, 64 });
            return true;
        };
    const server_cache_destruction_projection_callback project_once =
        [&](const auto & released, auto & out) {
            project_calls++;
            return project()(released, out);
        };
    common_cache_plan_destruction_counters counters;
    CHECK(server_cache_destruction_quote_all(
        rec, 0, { artifact(1, 11) }, 17,
        preview_once, project_once,
        { true, common_cache_plan_recovery_citation::resolved, 1 }, counters));
    CHECK(rec.destruction_quotes.size() ==
          COMMON_CACHE_PLAN_MAX_CANDIDATES - 1);
    CHECK(preview_calls == 1);
    CHECK(project_calls == 1);
    CHECK(counters.quote_memo_misses == 1);
    CHECK(counters.quote_memo_hits ==
          COMMON_CACHE_PLAN_MAX_CANDIDATES - 2);
}

int main() {
    test_complete_memoized_and_permutation();
    test_fail_closed_matrix();
    test_refusal_mapping_and_selection();
    test_plural_effect_union_and_counters();
    test_refused_projection_and_selection_failure();
    test_max_inventory_memoizes_one_manifest();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("cache destruction quote tests passed");
    return 0;
}
