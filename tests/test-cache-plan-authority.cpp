#include "server-cache-plan-authority.h"
#include "common-cache-plan-estimate.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static common_cache_plan_candidate * add_viable(
        common_cache_plan_record & rec,
        common_cache_plan_provider provider,
        int32_t source,
        int32_t target,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    auto * row = rec.find_or_add(provider, source, 0, target, origin);
    CHECK(row != nullptr);
    row->accept();
    return row;
}

struct host_checkpoint_chain_fixture {
    common_cache_plan_candidate * host = nullptr;
    common_cache_plan_candidate * checkpoint = nullptr;
    common_cache_plan_candidate * chain = nullptr;
};

static host_checkpoint_chain_fixture add_host_checkpoint_chain(
        common_cache_plan_record & rec,
        int32_t target,
        int32_t host_source,
        int32_t checkpoint_ordinal,
        common_cache_plan_selection origin =
            common_cache_plan_selection::by_id) {
    host_checkpoint_chain_fixture out;
    out.host = add_viable(rec, common_cache_plan_provider::host_cache_entry,
        host_source, target, origin);
    out.checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(
            host_source, checkpoint_ordinal), target, origin);
    out.checkpoint->component_only = true;
    out.checkpoint->dependent_host_source_id = host_source;
    out.chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(out.host - rec.inventory.data()),
        int32_t(out.checkpoint - rec.inventory.data()));
    CHECK(out.chain != nullptr);
    out.chain->accept();
    return out;
}

static void test_candidate_classifiers() {
    CHECK(common_cache_plan_strict_similarity(0.75, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.50, 0.50));
    CHECK(!common_cache_plan_strict_similarity(0.75, 0.0));
    CHECK(common_cache_plan_origin_in_domain(
        common_cache_plan_selection::similarity,
        common_cache_plan_selection::similarity));
    CHECK(!common_cache_plan_origin_in_domain(
        common_cache_plan_selection::route_home,
        common_cache_plan_selection::similarity));

    CHECK(common_cache_plan_authority_level_parse("off") ==
          common_cache_plan_authority_level::off);
    CHECK(common_cache_plan_authority_level_parse("route_home") ==
          common_cache_plan_authority_level::route_home);
    bool rejected = false;
    try {
        (void) common_cache_plan_authority_level_parse("future");
    } catch (...) {
        rejected = true;
    }
    CHECK(rejected);

    const auto busy = server_cache_plan_evaluate_live(true, true, 8, 16);
    CHECK(busy.reason == COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);
    const auto live = server_cache_plan_evaluate_live(false, true, 8, 16);
    CHECK(live.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(live.lcp_tokens == 8);
    common_cache_plan_candidate live_row;
    server_cache_plan_apply_live(&live_row, live);
    CHECK(!live_row.f_keep_known);
    CHECK(live_row.f_keep == -1.0f);

    const auto host = server_cache_plan_evaluate_host(
        true, true, 20, 40, 80, 1024);
    CHECK(host.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(host.f_keep == 0.25f);
    CHECK(server_cache_plan_evaluate_host(
        true, false, 20, 40, 80, 1024).reason ==
        COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);

    const auto ckpt = server_cache_plan_evaluate_checkpoint(
        true, true, true, true, 30, 39, 40, 0, 512);
    CHECK(ckpt.reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(ckpt.lcp_tokens == 39);
    CHECK(server_cache_plan_evaluate_checkpoint(
        true, true, true, false, 30, 39, 40, 0, 512).reason ==
        COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
    CHECK(server_cache_plan_viable(ckpt.reason));
    CHECK(!server_cache_plan_viable(
        COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT));
    CHECK(server_cache_plan_host_checkpoint_source_id(2, 3) == 1020003);
}

static void test_checkpoint_orientation_and_host_identity() {
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0) == 2);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 1) == 1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 2) == 0);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 3) == -1);
    CHECK(server_cache_plan_checkpoint_source_id_from_reverse(3, 0, 7) ==
          server_cache_plan_host_checkpoint_source_id(7, 2));
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(2) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 2);
    CHECK(server_cache_plan_checkpoint_ordinal_from_source_id(
              server_cache_plan_host_checkpoint_source_id(8, 2), 7) == -1);
    CHECK(server_cache_plan_checkpoint_reverse_position_from_source_id(
              3, server_cache_plan_host_checkpoint_source_id(7, 2), 7) == 0);

    common_cache_plan_record checkpoint_rec;
    auto * oldest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 5);
    auto * middle = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 1, 0, 5);
    auto * newest = checkpoint_rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 2, 0, 5);
    CHECK(oldest && middle && newest);
    CHECK(checkpoint_rec.find_or_add(
              common_cache_plan_provider::live_context_checkpoint,
              server_cache_plan_checkpoint_source_id_from_reverse(3, 0),
              COMMON_CACHE_PLAN_PHASE_CKPT_SCAN, 5) == newest);

    int32_t next_source = 0;
    int32_t old_prefix_instance = -1;
    int32_t survivor_instance = -1;
    int32_t replacement_instance = -1;
    int32_t source = -1;
    CHECK(server_cache_plan_assign_source_id(
        old_prefix_instance, next_source, source));
    const int32_t old_prefix = source;
    CHECK(source == 0);
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    const int32_t survivor = source;
    CHECK(source == 1);
    // Save-time prefix dedup removes old_prefix. The surviving physical node
    // must keep its identity after its list ordinal shifts from 1 to 0.
    CHECK(server_cache_plan_assign_source_id(
        survivor_instance, next_source, source));
    CHECK(source == 1);
    // The freshly saved replacement cannot inherit either removed ordinal.
    CHECK(server_cache_plan_assign_source_id(
        replacement_instance, next_source, source));
    const int32_t replacement = source;
    CHECK(source == 2);

    common_cache_plan_record host_rec;
    auto * removed_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, old_prefix, 0, 6);
    auto * survivor_row = host_rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, survivor, 0, 6);
    CHECK(removed_row && survivor_row && removed_row != survivor_row);
    // After dedup shifts the survivor to list ordinal zero, its immutable id
    // still rejoins its own pre-mutation row, never removed_row.
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, survivor,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) == survivor_row);
    CHECK(host_rec.find_or_add(
              common_cache_plan_provider::host_cache_entry, replacement,
              COMMON_CACHE_PLAN_PHASE_HOST_SCAN, 6) != survivor_row);
}

static void test_compose_excludes_destroyed_live_checkpoint() {
    common_cache_plan_record rec;
    auto * host = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 4, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host != nullptr);
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);

    auto * destroyed_live = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint, 0, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(destroyed_live != nullptr);
    destroyed_live->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);

    auto * host_checkpoint = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host_checkpoint != nullptr);
    host_checkpoint->accept();
    host_checkpoint->delivered = true;
    host_checkpoint->component_only = true;
    host_checkpoint->dependent_host_source_id = host->source_id;
    rec.select(common_cache_plan_provider::live_context_checkpoint,
               host_checkpoint);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    const uint32_t before = rec.n_inventory;
    common_cache_plan_compose_chains(rec);
    CHECK(rec.n_inventory == before + 1);
    CHECK(!destroyed_live->component_only);
    CHECK(rec.inventory[before].component_ids[0] ==
          int32_t(host - rec.inventory.data()));
    CHECK(rec.inventory[before].component_ids[1] ==
          int32_t(host_checkpoint - rec.inventory.data()));
}

static void test_composed_chain_reuses_inventory_identity() {
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;

    const auto fixture = add_host_checkpoint_chain(
        rec, 2, 4, 0, common_cache_plan_selection::similarity);
    auto * host = fixture.host;
    auto * selected = fixture.checkpoint;
    auto * selected_chain = fixture.chain;
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);
    auto * sibling = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 1), 0, 2,
        common_cache_plan_selection::similarity);
    auto * foreign = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(5, 0), 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(selected && sibling && foreign);
    selected->accept();
    selected->delivered = true;
    sibling->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling->component_only = true;
    sibling->dependent_host_source_id = 4;
    foreign->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    foreign->component_only = true;
    foreign->dependent_host_source_id = 5;
    rec.select(common_cache_plan_provider::live_context_checkpoint, selected);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    auto * sibling_chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(host - rec.inventory.data()),
        int32_t(sibling - rec.inventory.data()));
    CHECK(sibling_chain);
    selected_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling_chain->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    const int32_t selected_chain_id =
        int32_t(selected_chain - rec.inventory.data());
    const uint32_t inventory_before = rec.n_inventory;

    // The counterfactual planner chose the inventory-time physical chain. Finalize must
    // promote that same ordinal, not append a duplicate, and must not combine the delivered
    // host with a checkpoint from host source 5.
    rec.planner_status = common_cache_plan_planner_status::ok;
    rec.shadow_choice = selected_chain_id;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, common_cache_plan_authority_level::lru,
        common_cache_plan_authority_fallback::none);
    common_cache_plan_compose_chains(rec);

    CHECK(rec.n_inventory == inventory_before);
    CHECK(rec.shipped_plan_candidate == selected_chain_id);
    CHECK(selected_chain->delivered);
    CHECK(selected_chain->disposition ==
          common_cache_plan_disposition::accepted);
    CHECK(!sibling_chain->delivered);
    CHECK(sibling_chain->disposition ==
          common_cache_plan_disposition::valid_not_chosen_cost);

    common_cache_plan_finalize_shadow_authority(rec);
    CHECK(rec.authority.legacy_plan_candidate == selected_chain_id);
    CHECK(rec.authority.planner_plan_candidate == selected_chain_id);
    CHECK(!rec.authority.disagreed);
}

static void test_inventory_saturation_refuses_qualification() {
    common_cache_plan_record rec;
    rec.calibration_profile = "test";
    rec.n_prompt_tokens = llama_cache_acct_value::measured(32);
    for (size_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * row = rec.find_or_add(
            common_cache_plan_provider::cold_replay, int32_t(i), 0,
            int32_t(i), common_cache_plan_selection::lru);
        CHECK(row != nullptr);
        row->accept();
    }
    CHECK(!rec.inventory_saturated());
    CHECK(rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 1000, 0, 1000,
        common_cache_plan_selection::lru) == nullptr);
    CHECK(rec.inventory_saturated());
    CHECK(rec.inventory_states[size_t(
              common_cache_plan_provider::host_cache_entry)] ==
          common_cache_plan_inventory_state::overflowed);

    const common_cache_plan_calib calib = {
        "test", 1, 1.0, 1.0, 1.0,
    };
    CHECK(common_cache_plan_estimate_and_choose(rec, calib) ==
          common_cache_plan_planner_status::incomplete_evidence);
    CHECK(rec.shadow_choice == -1);
}

static void test_stale_capability_refuses_without_throwing() {
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::lru);
    common_cache_plan_record rec;
    authority.plan_before_mutation(rec, 10, 11);
    CHECK(rec.planner_precomputed);
    CHECK(!rec.authority_prequalified);
    CHECK(rec.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(rec.authority.configured_level ==
          common_cache_plan_authority_level::lru);
}

static server_cache_plan_execution authorize_choice(
        server_cache_plan_authority & authority,
        common_cache_plan_record & rec,
        int32_t choice,
        int32_t target,
        bool host_lookup_enabled = true,
        common_cache_plan_authority_level level =
            common_cache_plan_authority_level::by_id,
        common_cache_plan_selection selection =
            common_cache_plan_selection::by_id,
        bool target_identity_matches = true,
        common_cache_plan_destruction_effect_set permitted_effects = 0) {
    rec.selection = selection;
    rec.planner_status = common_cache_plan_planner_status::ok;
    rec.shadow_choice = choice;
    rec.authority_prequalified = true;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, level,
        common_cache_plan_authority_fallback::none);
    return authority.authorize(
        rec, target, host_lookup_enabled, target_identity_matches,
        permitted_effects);
}

static void test_execution_seam_fallbacks() {
    constexpr int32_t target = 4;
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);

    common_cache_plan_record coverage;
    auto * live = add_viable(
        coverage, common_cache_plan_provider::live_slot, target, target);
    auto execution = authorize_choice(
        authority, coverage,
        int32_t(live - coverage.inventory.data()), target);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_coverage_recovery(
        authority, coverage, execution, 64, 64));
    CHECK(!execution.authoritative());
    CHECK(coverage.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(coverage.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(coverage.authority_prequalified);

    // A routine dynamic-VBR low-LCP reset supersedes the armed plan after its
    // post-reclaim ownership sample. It is capability drift, not an internal
    // execution fault, and remains in the qualified denominator.
    common_cache_plan_record low_lcp;
    auto * low_lcp_live = add_viable(
        low_lcp, common_cache_plan_provider::live_slot, target, target,
        common_cache_plan_selection::route_home);
    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    execution = authorize_choice(
        route_home, low_lcp,
        int32_t(low_lcp_live - low_lcp.inventory.data()), target, true,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_vbr_low_lcp_reset(
        route_home, low_lcp, execution, true));
    CHECK(!execution.authoritative());
    CHECK(low_lcp.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(low_lcp.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(low_lcp.authority_prequalified);
    low_lcp.shipped_plan_candidate =
        int32_t(low_lcp_live - low_lcp.inventory.data());
    route_home.finalize_execution(low_lcp);
    CHECK(route_home.counters.authority_eligible[size_t(
              common_cache_plan_selection::route_home)] == 1);
    CHECK(route_home.counters.fallback_reason[size_t(
              common_cache_plan_authority_fallback::stale_capability)] == 1);

    server_cache_plan_execution checkpoint_execution;
    checkpoint_execution.kind =
        server_cache_plan_execution_kind::checkpoint_restore;
    CHECK(server_cache_plan_checkpoint_superseded_by_window(
        checkpoint_execution, true));
    CHECK(!server_cache_plan_checkpoint_superseded_by_window(
        checkpoint_execution, false));
    server_cache_plan_execution live_execution;
    live_execution.kind =
        server_cache_plan_execution_kind::live_replay;
    CHECK(!server_cache_plan_checkpoint_superseded_by_window(
        live_execution, true));
    CHECK(server_cache_plan_live_replay_lost_to_logits(
        live_execution, 0));
    CHECK(!server_cache_plan_live_replay_lost_to_logits(
        live_execution, 1));

    // A post-retarget coverage demotion executes legacy recovery on the
    // planner-selected slot, not on the legacy-selected slot. Preserve the
    // pre-mutation slot-A counterfactual so the receipt records disagreement.
    constexpr int32_t legacy_target = 8;
    constexpr int32_t planned_target = 9;
    common_cache_plan_record retarget;
    auto * legacy_live = add_viable(
        retarget, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target,
        common_cache_plan_selection::similarity);
    legacy_live->f_keep = 0.8;
    legacy_live->f_keep_known = true;
    auto * planned_live = add_viable(
        retarget, common_cache_plan_provider::live_slot,
        planned_target, planned_target,
        common_cache_plan_selection::similarity);
    planned_live->f_keep = 1.0;
    planned_live->f_keep_known = true;
    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    execution = authorize_choice(
        similarity, retarget,
        int32_t(planned_live - retarget.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(execution.authoritative());
    CHECK(server_cache_plan_demote_for_coverage_recovery(
        similarity, retarget, execution, 64, 64));
    retarget.shipped_plan_candidate =
        int32_t(planned_live - retarget.inventory.data());
    similarity.finalize_execution(retarget);
    CHECK(retarget.authority.legacy_plan_candidate ==
          int32_t(legacy_live - retarget.inventory.data()));
    CHECK(retarget.authority.planner_plan_candidate ==
          int32_t(planned_live - retarget.inventory.data()));
    CHECK(retarget.authority.executed_plan_candidate ==
          int32_t(planned_live - retarget.inventory.data()));
    CHECK(retarget.authority.disagreed);
    CHECK(similarity.counters.agree[size_t(
              common_cache_plan_selection::similarity)] == 0);
    CHECK(similarity.counters.disagree[size_t(
              common_cache_plan_selection::similarity)] == 1);

    common_cache_plan_record checkpoint;
    auto * row = add_viable(
        checkpoint,
        common_cache_plan_provider::live_context_checkpoint,
        2, target);
    execution = authorize_choice(
        authority, checkpoint,
        int32_t(row - checkpoint.inventory.data()), target);
    int32_t ordinal = -1;
    CHECK(!server_cache_plan_revalidate_checkpoint_execution(
        authority, checkpoint, execution, 3, false, ordinal));
    CHECK(!execution.authoritative());
    CHECK(ordinal == -1);
    // The seam retains its legacy iterator on refusal; it never translates a
    // missing/invalid planner checkpoint into a synthetic rend/cold choice.
    CHECK(checkpoint.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);

    common_cache_plan_record armed_record;
    auto armed_plan = std::make_unique<common_cache_plan_record>(armed_record);
    auto armed_recovery = std::make_unique<int>(1);
    server_cache_plan_execution armed;
    armed.kind = server_cache_plan_execution_kind::live_replay;
    armed.target = target;
    server_cache_plan_disarm_unlaunched(
        armed, armed_plan, armed_recovery);
    CHECK(!armed.authoritative());
    CHECK(!armed_plan);
    CHECK(!armed_recovery);
}

static void test_qualified_fallback_remains_eligible() {
    constexpr int32_t target = 1;
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    const int32_t candidate = int32_t(live - rec.inventory.data());
    auto execution = authorize_choice(
        authority, rec, candidate, target);
    CHECK(execution.authoritative());
    authority.fallback_legacy(
        rec, common_cache_plan_authority_fallback::stale_capability);
    CHECK(rec.authority_prequalified);
    rec.shipped_plan_candidate = candidate;
    authority.finalize_execution(rec);
    CHECK(authority.counters.authority_eligible[size_t(
              common_cache_plan_selection::by_id)] == 1);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 0);
}

static void test_by_id_execution_shapes_and_target_binding() {
    constexpr int32_t target = 7;
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    const auto fixture = add_host_checkpoint_chain(rec, target, 11, 2);
    auto * host = fixture.host;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.8; host->sim_known = true;
    auto * checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint, 3, target);
    auto * cold = add_viable(
        rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);
    auto * chain = fixture.chain;

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    struct expectation {
        common_cache_plan_candidate * candidate;
        server_cache_plan_execution_kind kind;
        int32_t host_source;
        int32_t checkpoint_source;
    } cases[] = {
        { live, server_cache_plan_execution_kind::live_replay, -1, -1 },
        { host, server_cache_plan_execution_kind::host_restore, 11, -1 },
        { checkpoint, server_cache_plan_execution_kind::checkpoint_restore, -1, 3 },
        { chain, server_cache_plan_execution_kind::host_checkpoint_restore,
          11, server_cache_plan_host_checkpoint_source_id(11, 2) },
        { cold, server_cache_plan_execution_kind::cold_replay, -1, -1 },
    };
    for (const auto & expected : cases) {
        const int32_t choice = int32_t(expected.candidate - rec.inventory.data());
        server_cache_plan_execution execution;
        CHECK(server_cache_plan_execution_from_candidate(
            rec, choice, target, execution));
        CHECK(execution.kind == expected.kind);
        CHECK(execution.target == target);
        CHECK(execution.host_source_id == expected.host_source);
        CHECK(execution.checkpoint_source_id == expected.checkpoint_source);
        if (expected.kind != server_cache_plan_execution_kind::cold_replay) {
            auto copy = rec;
            CHECK(authorize_choice(
                authority, copy, choice, target).authoritative());
        }
    }

    common_cache_plan_record cold_only;
    auto * isolated_cold = add_viable(
        cold_only, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);
    CHECK(authorize_choice(authority, cold_only,
              int32_t(isolated_cold - cold_only.inventory.data()), target).
          authoritative());

    // The full fixture's legacy counterfactual is the host+checkpoint chain;
    // a non-destructive live disagreement remains inside the pre-D-A envelope.
    auto copy = rec;
    CHECK(authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target).authoritative());

    // A planner row for another slot can never become an execution capability.
    copy = rec;
    const auto wrong_target = authorize_choice(
        authority, copy, int32_t(live - rec.inventory.data()), target + 1);
    CHECK(!wrong_target.authoritative());
    CHECK(copy.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);

    // Before D-A, an authority disagreement may avoid destruction but may not
    // consume a different retained host entry or turn reuse into cold.
    auto * foreign_host = add_viable(
        rec, common_cache_plan_provider::host_cache_entry, 12, target);
    foreign_host->f_keep = 0.1; foreign_host->f_keep_known = true;
    foreign_host->sim = 0.1; foreign_host->sim_known = true;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.8; host->sim_known = true;
    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(foreign_host - rec.inventory.data()), target).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);
    copy = rec;
    CHECK(!authorize_choice(
              authority, copy, int32_t(cold - rec.inventory.data()), target).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);

    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::by_id,
              common_cache_plan_selection::similarity).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    copy = rec;
    CHECK(authorize_choice(similarity, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::similarity,
              common_cache_plan_selection::similarity).authoritative());

    // The final landed level keeps every earlier ratchet cumulative and now
    // admits its own LRU tier as well.
    server_cache_plan_authority future(
        common_cache_plan_authority_level::lru);
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::route_home).authoritative());
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::similarity).authoritative());
    copy = rec;
    CHECK(authorize_choice(future, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::lru,
              common_cache_plan_selection::lru).authoritative());

    // A disabled future tier must not blur an existing planner refusal into
    // tier_not_enabled. The latter names only an otherwise-qualified plan.
    common_cache_plan_record unfitted = rec;
    unfitted.selection = common_cache_plan_selection::route_home;
    unfitted.planner_status =
        common_cache_plan_planner_status::profile_unfitted;
    unfitted.planner_precomputed = true;
    unfitted.authority_prequalified = false;
    common_cache_plan_derive_shadow_authority(
        unfitted, common_cache_plan_authority_level::lru,
        common_cache_plan_authority_fallback::none);
    CHECK(!future.authorize(unfitted, target).authoritative());
    CHECK(unfitted.authority.fallback_reason ==
          common_cache_plan_authority_fallback::profile_unfitted);

    copy = rec;
    CHECK(!authorize_choice(authority, copy,
              int32_t(live - rec.inventory.data()), target, true,
              common_cache_plan_authority_level::by_id,
              common_cache_plan_selection::by_id,
              false).authoritative());
    CHECK(copy.authority.fallback_reason ==
          common_cache_plan_authority_fallback::incomplete_evidence);
}

static void test_legacy_counterfactual_and_authoritative_receipt() {
    constexpr int32_t target = 2;
    common_cache_plan_record rec;
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, target, target);
    live->f_keep = 0.4; live->f_keep_known = true;
    live->sim = 0.4; live->sim_known = true;
    const auto fixture = add_host_checkpoint_chain(rec, target, 5, 1);
    auto * host = fixture.host;
    host->f_keep = 0.8; host->f_keep_known = true;
    host->sim = 0.7; host->sim_known = true;
    auto * chain = fixture.chain;
    add_viable(
        rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, target);

    const int32_t legacy = server_cache_plan_legacy_candidate(rec, target);
    CHECK(legacy == int32_t(chain - rec.inventory.data()));
    CHECK(server_cache_plan_legacy_candidate(rec, target, false) ==
          int32_t(live - rec.inventory.data()));

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    const int32_t planner = int32_t(live - rec.inventory.data());
    const auto execution = authorize_choice(authority, rec, planner, target);
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);
    rec.shipped_plan_candidate = planner;
    authority.finalize_execution(rec);
    CHECK(rec.authority.state ==
          common_cache_plan_authority_state::authoritative);
    CHECK(rec.authority.legacy_plan_candidate == legacy);
    CHECK(rec.authority.planner_plan_candidate == planner);
    CHECK(rec.authority.executed_plan_candidate == planner);
    CHECK(rec.authority.disagreed);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 1);
}

static void test_similarity_crossover_and_safety_envelope() {
    constexpr int32_t legacy_target = 2;
    constexpr int32_t other_target = 3;
    const common_cache_plan_calib calib = {
        "similarity-crossover", 1,
        /* replay_us_per_token */ 100.0,
        /* restore_us_per_byte */ 0.001,
        /* workspace_setup_us */  500.0,
    };

    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::similarity;
    rec.calibration_profile = calib.profile;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * live = add_viable(rec, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::similarity);
    live->lcp_tokens = llama_cache_acct_value::measured(90);
    live->f_keep = 0.9; live->f_keep_known = true;
    const auto host_checkpoint = add_host_checkpoint_chain(
        rec, legacy_target, 11, 0,
        common_cache_plan_selection::similarity);
    host_checkpoint.host->lcp_tokens = llama_cache_acct_value::measured(80);
    host_checkpoint.host->payload_bytes =
        llama_cache_acct_value::measured(5'000'000);
    host_checkpoint.host->f_keep = 0.95;
    host_checkpoint.host->f_keep_known = true;
    host_checkpoint.host->sim = 0.95;
    host_checkpoint.host->sim_known = true;
    host_checkpoint.checkpoint->lcp_tokens =
        llama_cache_acct_value::measured(99);
    host_checkpoint.checkpoint->payload_bytes =
        llama_cache_acct_value::measured(45'000'000);
    auto * cold = add_viable(rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::similarity);
    (void) cold;

    // A cheaper row below the strict threshold is route-home evidence, not a
    // similarity-tier authority candidate.
    auto * below_threshold = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::route_home);
    below_threshold->lcp_tokens = llama_cache_acct_value::measured(100);
    below_threshold->f_keep = 1.0; below_threshold->f_keep_known = true;

    CHECK(server_cache_plan_legacy_candidate(rec, legacy_target) ==
          int32_t(host_checkpoint.chain - rec.inventory.data()));
    rec.planner_status = common_cache_plan_estimate_and_choose(rec, calib);
    CHECK(rec.planner_status == common_cache_plan_planner_status::ok);
    CHECK(rec.shadow_choice == int32_t(live - rec.inventory.data()));

    server_cache_plan_authority authority(
        common_cache_plan_authority_level::similarity);
    rec.authority_prequalified = true;
    rec.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        rec, common_cache_plan_authority_level::similarity,
        common_cache_plan_authority_fallback::none);
    auto execution = authority.authorize(rec, legacy_target);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);
    CHECK(execution.target == legacy_target); // crossover changes provider, not slot
    rec.shipped_plan_candidate = int32_t(live - rec.inventory.data());
    authority.finalize_execution(rec);
    CHECK(rec.authority.executed_plan_candidate ==
          int32_t(live - rec.inventory.data()));
    CHECK(!host_checkpoint.checkpoint->delivered);
    CHECK(!rec.restore_attempt_failed);

    // A replay-wins crossover against a LIVE checkpoint is structurally
    // impossible: below the coverage threshold legacy replays, while at/above
    // it replay is invalid and the coverage seam demotes. The real replay win
    // above is the host-side prompt-load/host-checkpoint chain.

    common_cache_plan_record long_suffix;
    long_suffix.selection = common_cache_plan_selection::similarity;
    long_suffix.calibration_profile = calib.profile;
    long_suffix.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * long_live = add_viable(
        long_suffix, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target,
        common_cache_plan_selection::similarity);
    long_live->lcp_tokens = llama_cache_acct_value::measured(20);
    long_live->f_keep = 0.2; long_live->f_keep_known = true;
    auto * cheap_checkpoint = add_viable(
        long_suffix, common_cache_plan_provider::live_context_checkpoint,
        0, legacy_target, common_cache_plan_selection::similarity);
    cheap_checkpoint->lcp_tokens = llama_cache_acct_value::measured(90);
    cheap_checkpoint->payload_bytes = llama_cache_acct_value::measured(1'000);
    add_viable(long_suffix, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::similarity);
    long_suffix.planner_status =
        common_cache_plan_estimate_and_choose(long_suffix, calib);
    CHECK(long_suffix.planner_status == common_cache_plan_planner_status::ok);
    CHECK(long_suffix.shadow_choice ==
          int32_t(cheap_checkpoint - long_suffix.inventory.data()));
    long_suffix.authority_prequalified = true;
    long_suffix.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        long_suffix, common_cache_plan_authority_level::similarity,
        common_cache_plan_authority_fallback::none);
    execution = authority.authorize(long_suffix, legacy_target);
    CHECK(execution.authoritative());
    CHECK(execution.kind ==
          server_cache_plan_execution_kind::checkpoint_restore);
    int32_t restore_ordinal = -1;
    CHECK(server_cache_plan_revalidate_checkpoint_execution(
        authority, long_suffix, execution, 1, true, restore_ordinal));
    CHECK(execution.authoritative());
    CHECK(restore_ordinal == 0);

    // Cross-target similarity execution remains pre-D-A fail-closed unless it
    // retains the target's complete live prefix (the zero-destruction case).
    common_cache_plan_record cross;
    auto * legacy_live = add_viable(
        cross, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::similarity);
    legacy_live->f_keep = 0.8; legacy_live->f_keep_known = true;
    auto * cross_live = add_viable(
        cross, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::similarity);
    cross_live->f_keep = 0.5; cross_live->f_keep_known = true;
    execution = authorize_choice(
        authority, cross, int32_t(cross_live - cross.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(!execution.authoritative());
    CHECK(cross.authority.fallback_reason ==
          common_cache_plan_authority_fallback::destruction_authority_required);

    cross_live->f_keep = 1.0;
    execution = authorize_choice(
        authority, cross, int32_t(cross_live - cross.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(execution.authoritative());
    CHECK(execution.target == other_target);

    // A route-home row cannot be smuggled into a similarity receipt even if a
    // malformed caller installs it as the planner choice.
    execution = authorize_choice(
        authority, rec,
        int32_t(below_threshold - rec.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::similarity);
    CHECK(!execution.authoritative());
    CHECK(rec.authority.fallback_reason ==
          common_cache_plan_authority_fallback::incomplete_evidence);
}

static void test_route_home_authority_domain() {
    constexpr int32_t home_target = 4;
    constexpr int32_t other_target = 5;
    const common_cache_plan_calib calib = {
        "route-home", 1,
        /* replay_us_per_token */ 100.0,
        /* restore_us_per_byte */ 0.001,
        /* workspace_setup_us */  500.0,
    };

    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::route_home;
    rec.calibration_profile = calib.profile;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * home = add_viable(
        rec, common_cache_plan_provider::live_slot,
        home_target, home_target, common_cache_plan_selection::route_home);
    home->lcp_tokens = llama_cache_acct_value::measured(10);
    home->sim = 0.1;
    home->sim_known = true;
    home->f_keep = 1.0;
    home->f_keep_known = true;
    auto * checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint,
        0, home_target, common_cache_plan_selection::route_home);
    checkpoint->lcp_tokens = llama_cache_acct_value::measured(90);
    checkpoint->payload_bytes = llama_cache_acct_value::measured(1'000);
    add_viable(rec, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, home_target,
        common_cache_plan_selection::route_home);

    // LRU-origin and busy rows are each cheaper than the accepted checkpoint;
    // only their independent domain/provider exclusions keep them out.
    auto * lru_only = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::lru);
    lru_only->lcp_tokens = llama_cache_acct_value::measured(100);
    lru_only->sim = 1.0;
    lru_only->sim_known = true;
    lru_only->f_keep = 1.0;
    lru_only->f_keep_known = true;
    auto * busy = add_viable(
        rec, common_cache_plan_provider::live_slot,
        other_target + 1, other_target + 1,
        common_cache_plan_selection::route_home);
    busy->lcp_tokens = llama_cache_acct_value::measured(99);
    busy->note_reject(COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY);

    rec.planner_status = common_cache_plan_estimate_and_choose(rec, calib);
    CHECK(rec.planner_status == common_cache_plan_planner_status::ok);
    CHECK(rec.shadow_choice ==
          int32_t(checkpoint - rec.inventory.data()));
    CHECK(rec.inventory[size_t(rec.shadow_choice)].target_slot_id ==
          home_target);

    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    auto execution = authorize_choice(
        route_home, rec, rec.shadow_choice, home_target, false,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(execution.authoritative());
    CHECK(execution.kind ==
          server_cache_plan_execution_kind::checkpoint_restore);
    CHECK(execution.target == home_target);
    CHECK(rec.authority.decision_tier ==
          common_cache_plan_selection::route_home);

    // The previous ceiling remains a hard boundary for route-home records.
    auto lower = rec;
    server_cache_plan_authority similarity(
        common_cache_plan_authority_level::similarity);
    CHECK(!authorize_choice(
        similarity, lower, lower.shadow_choice, home_target, false,
        common_cache_plan_authority_level::similarity,
        common_cache_plan_selection::route_home).authoritative());
    CHECK(lower.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    // A BOS-only cross-target choice needs retention/destruction economics the
    // current fit does not carry. Dynamic VBR cannot durabilize the displaced
    // target in a host cache, so this shape remains typed legacy until D-A.
    common_cache_plan_record trivial;
    auto * legacy_bos = add_viable(
        trivial, common_cache_plan_provider::live_slot,
        home_target, home_target,
        common_cache_plan_selection::route_home);
    legacy_bos->lcp_tokens = llama_cache_acct_value::measured(1);
    legacy_bos->f_keep = 0.01;
    legacy_bos->f_keep_known = true;
    auto * alternate_bos = add_viable(
        trivial, common_cache_plan_provider::live_slot,
        other_target, other_target,
        common_cache_plan_selection::route_home);
    alternate_bos->lcp_tokens = llama_cache_acct_value::measured(1);
    alternate_bos->f_keep = 1.0;
    alternate_bos->f_keep_known = true;
    execution = authorize_choice(
        route_home, trivial,
        int32_t(alternate_bos - trivial.inventory.data()), home_target,
        false, common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::route_home);
    CHECK(!execution.authoritative());
    CHECK(trivial.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);

}

static void test_lru_authority_domain_and_eviction_fence() {
    constexpr int32_t legacy_target = 6;
    constexpr int32_t other_target = 2;
    server_cache_plan_authority lru(
        common_cache_plan_authority_level::lru);

    // Empty-slot spread: a target-qualified cold plan on the shipped LRU slot
    // has exactly the same destructive effect and is authoritative.
    common_cache_plan_record empty;
    auto * cold = add_viable(
        empty, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    auto execution = authorize_choice(
        lru, empty, int32_t(cold - empty.inventory.data()), legacy_target,
        true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::cold_replay);
    CHECK(execution.target == legacy_target);
    // Exercise the two server seam predicates that follow authorize(): an LRU
    // same-target cold plan must skip cross-target currency lookup (there is no
    // viable live row on an empty target) and pass construction-empty recheck.
    CHECK(!server_cache_plan_retarget_currency_required(
        server_cache_plan_selection_admits_retarget(
            common_cache_plan_authority_level::lru,
            common_cache_plan_selection::lru),
        execution.target, legacy_target));
    // and the positive direction: a genuine cross-target plan still requires
    // the currency lookup (a predicate degraded to `return false` must fail)
    CHECK(server_cache_plan_retarget_currency_required(
        server_cache_plan_selection_admits_retarget(
            common_cache_plan_authority_level::lru,
            common_cache_plan_selection::lru),
        execution.target + 1, legacy_target));
    CHECK(server_cache_plan_cold_target_current(execution, true, true));
    CHECK(!server_cache_plan_cold_target_current(execution, false, true));

    // Occupied same-target reuse remains inside the pre-D-A envelope.
    common_cache_plan_record occupied;
    auto * live = add_viable(
        occupied, common_cache_plan_provider::live_slot,
        legacy_target, legacy_target, common_cache_plan_selection::lru);
    live->f_keep = 1.0;
    live->f_keep_known = true;
    execution = authorize_choice(
        lru, occupied, int32_t(live - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);

    // Cross-target replacement has no certified eviction/retention evidence
    // before D-A. Schema 5 uses its existing availability spelling rather than
    // adding a new wire enum value.
    auto * foreign = add_viable(
        occupied, common_cache_plan_provider::live_slot,
        other_target, other_target, common_cache_plan_selection::lru);
    foreign->f_keep = 1.0;
    foreign->f_keep_known = true;
    execution = authorize_choice(
        lru, occupied, int32_t(foreign - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            occupied, int32_t(foreign - occupied.inventory.data()),
            occupied.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::cross_target_displacement));

    // Same-target cold replacement is the frozen B-A4 eviction-evidence
    // shape and remains closed without a D-A certificate.
    auto * occupied_cold = add_viable(
        occupied, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    execution = authorize_choice(
        lru, occupied, int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);
    const auto lifecycle_effects =
        server_cache_plan_nonconsuming_host_effects(true);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
              occupied,
              int32_t(occupied_cold - occupied.inventory.data()),
              occupied.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::same_target_cold_replacement));
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            occupied,
            int32_t(occupied_cold - occupied.inventory.data()),
            occupied.authority.legacy_plan_candidate,
            lifecycle_effects),
        common_cache_plan_destruction_effect::same_target_cold_replacement));

    // Lifecycle-on introduces that D-A5 class but keeps it closed until the
    // selected effect is certified.
    execution = authorize_choice(
        lru, occupied, int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, lifecycle_effects);
    CHECK(!execution.authoritative());
    CHECK(occupied.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              budget_or_lease_unavailable);

    // D-A5 opens exactly the certified effect bit. The same occupied cold
    // replacement remains refused above with no capability evidence.
    const auto certified_cold = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::same_target_cold_replacement);
    execution = authorize_choice(
        lru, occupied,
        int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true,
        certified_cold | lifecycle_effects);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::cold_replay);

    // A partial certificate never opens a plural destructive plan.
    auto plural = occupied;
    plural.inventory[size_t(occupied_cold - occupied.inventory.data())]
        .provider = common_cache_plan_provider::host_cache_entry;
    plural.inventory[size_t(occupied_cold - occupied.inventory.data())]
        .source_id = 77;
    execution = authorize_choice(
        lru, plural,
        int32_t(occupied_cold - occupied.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, certified_cold);
    CHECK(!execution.authoritative());

    // Host-consumption authority is not eviction evidence: choosing another
    // retained host source on the same target keeps the established reason.
    common_cache_plan_record hosts;
    auto * legacy_host = add_viable(
        hosts, common_cache_plan_provider::host_cache_entry,
        41, legacy_target, common_cache_plan_selection::lru);
    legacy_host->f_keep = 0.8; legacy_host->f_keep_known = true;
    legacy_host->sim = 0.8; legacy_host->sim_known = true;
    auto * other_host = add_viable(
        hosts, common_cache_plan_provider::host_cache_entry,
        42, legacy_target, common_cache_plan_selection::lru);
    other_host->f_keep = 0.7; other_host->f_keep_known = true;
    other_host->sim = 0.9; other_host->sim_known = true;
    execution = authorize_choice(
        lru, hosts, int32_t(other_host - hosts.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru);
    CHECK(!execution.authoritative());
    CHECK(hosts.authority.fallback_reason ==
          common_cache_plan_authority_fallback::
              destruction_authority_required);
    CHECK(common_cache_plan_destruction_effect_has(
        server_cache_destruction_effects_for(
            hosts, int32_t(other_host - hosts.inventory.data()),
            hosts.authority.legacy_plan_candidate),
        common_cache_plan_destruction_effect::
            different_host_source_consumption));
    const auto nonconsuming_host =
        server_cache_plan_nonconsuming_host_effects(true);
    CHECK(server_cache_destruction_effects_for(
              hosts, int32_t(other_host - hosts.inventory.data()),
              hosts.authority.legacy_plan_candidate,
              nonconsuming_host) == 0);
    execution = authorize_choice(
        lru, hosts, int32_t(other_host - hosts.inventory.data()),
        legacy_target, true, common_cache_plan_authority_level::lru,
        common_cache_plan_selection::lru, true, nonconsuming_host);
    CHECK(execution.authoritative());
    CHECK(execution.kind == server_cache_plan_execution_kind::host_restore);

    // The previous ceiling still cannot flip an LRU selection.
    common_cache_plan_record lower;
    auto * lower_cold = add_viable(
        lower, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, legacy_target,
        common_cache_plan_selection::lru);
    server_cache_plan_authority route_home(
        common_cache_plan_authority_level::route_home);
    CHECK(!authorize_choice(
        route_home, lower,
        int32_t(lower_cold - lower.inventory.data()), legacy_target, true,
        common_cache_plan_authority_level::route_home,
        common_cache_plan_selection::lru).authoritative());
    CHECK(lower.authority.fallback_reason ==
          common_cache_plan_authority_fallback::tier_not_enabled);

    // Raising the configured ceiling leaves all prior ratchets cumulative.
    for (const auto selection : {
            common_cache_plan_selection::by_id,
            common_cache_plan_selection::similarity,
            common_cache_plan_selection::route_home }) {
        common_cache_plan_record prior;
        auto * prior_live = add_viable(
            prior, common_cache_plan_provider::live_slot,
            legacy_target, legacy_target, selection);
        CHECK(authorize_choice(
            lru, prior, int32_t(prior_live - prior.inventory.data()),
            legacy_target, true, common_cache_plan_authority_level::lru,
            selection).authoritative());
    }
}

static void test_typed_planner_fallbacks() {
    const common_cache_plan_planner_status statuses[] = {
        common_cache_plan_planner_status::no_profile,
        common_cache_plan_planner_status::profile_unfitted,
        common_cache_plan_planner_status::incomplete_evidence,
        common_cache_plan_planner_status::internal_fault,
    };
    const common_cache_plan_authority_fallback reasons[] = {
        common_cache_plan_authority_fallback::no_profile,
        common_cache_plan_authority_fallback::profile_unfitted,
        common_cache_plan_authority_fallback::incomplete_evidence,
        common_cache_plan_authority_fallback::internal_fault,
    };
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id);
    for (size_t i = 0; i < std::size(statuses); ++i) {
        common_cache_plan_record rec;
        rec.selection = common_cache_plan_selection::by_id;
        rec.planner_status = statuses[i];
        rec.planner_precomputed = true;
        common_cache_plan_derive_shadow_authority(
            rec, common_cache_plan_authority_level::by_id,
            common_cache_plan_authority_fallback::none);
        const auto execution = authority.authorize(rec, 0);
        CHECK(!execution.authoritative());
        CHECK(rec.authority.state ==
              common_cache_plan_authority_state::fallback_legacy);
        CHECK(rec.authority.fallback_reason == reasons[i]);
    }
}

static void test_off_stays_shadow_and_failed_delivery_not_counted() {
    common_cache_plan_record base;
    auto * live = add_viable(
        base, common_cache_plan_provider::live_slot, 0, 0);
    const int32_t live_id = int32_t(live - base.inventory.data());

    server_cache_plan_authority off(common_cache_plan_authority_level::off);
    auto shadow = base;
    shadow.selection = common_cache_plan_selection::by_id;
    shadow.planner_status = common_cache_plan_planner_status::ok;
    shadow.shadow_choice = live_id;
    shadow.authority_prequalified = true;
    shadow.planner_precomputed = true;
    common_cache_plan_derive_shadow_authority(
        shadow, common_cache_plan_authority_level::off,
        common_cache_plan_authority_fallback::none);
    CHECK(!off.authorize(shadow, 0).authoritative());
    CHECK(shadow.authority.state == common_cache_plan_authority_state::shadow);

    common_cache_plan_record exception;
    exception.selection = common_cache_plan_selection::by_id;
    server_cache_plan_authority flipped(
        common_cache_plan_authority_level::by_id);
    flipped.fail_closed(exception);
    CHECK(exception.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(exception.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);

    server_cache_plan_authority on(
        common_cache_plan_authority_level::by_id);
    auto failed = base;
    CHECK(authorize_choice(on, failed, live_id, 0).authoritative());
    auto * cold = add_viable(
        failed, common_cache_plan_provider::cold_replay,
        COMMON_CACHE_PLAN_SOURCE_AGGREGATE, 0);
    failed.shipped_plan_candidate = int32_t(cold - failed.inventory.data());
    on.finalize_execution(failed);
    CHECK(failed.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(failed.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);
    CHECK(on.counters.authority_executed[size_t(
              common_cache_plan_selection::by_id)] == 0);
}

static void test_eligible_and_executed_index_different_tiers() {
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::lru);
    common_cache_plan_record rec;
    rec.planner_precomputed = true;
    rec.authority_prequalified = true;
    rec.selection = common_cache_plan_selection::similarity;
    rec.authority.configured_level = common_cache_plan_authority_level::lru;
    rec.authority.decision_tier = common_cache_plan_selection::route_home;
    rec.authority.planner_plan_candidate = 2;
    rec.shipped_plan_candidate = 1;
    authority.finalize_execution(rec);

    CHECK(authority.counters.observed[size_t(
              common_cache_plan_selection::similarity)] == 1);
    CHECK(authority.counters.disagree[size_t(
              common_cache_plan_selection::similarity)] == 1);
    CHECK(authority.counters.authority_eligible[size_t(
              common_cache_plan_selection::route_home)] == 1);
    CHECK(authority.counters.authority_executed[size_t(
              common_cache_plan_selection::route_home)] == 0);
    CHECK(rec.authority.executed_plan_candidate == 1);
}

static server_cache_observation_key local_key(
        common_cache_plan_provider provider,
        server_cache_observation_operation operation,
        const std::array<uint8_t, 32> & execution_root) {
    auto out = server_cache_observation_cpu_key(operation, provider, 0);
    if (operation == server_cache_observation_operation::replay) {
        out.feature_dim = 4;
    }
    out.profile_execution_digest = execution_root;
    out.adapter_application_digest[0] = uint8_t(provider) + 1;
    out.adapter_application_complete = true;
    out.representation_digest[0] = 0x44;
    CHECK(server_cache_calibration_single_participant_digest_v1(
        out.adapter_application_digest, out.representation_digest, 0,
        out.participant_execution_digest));
    CHECK(server_cache_calibration_effect_action_digest_v1(
        nullptr, 0, out.effect_action_shape_digest));
    out.identity_complete = true;
    out.identity_exact = true;
    return out;
}

static server_cache_observation_instance mature_local_instance(
        uint32_t slot,
        server_cache_observation_key key,
        const std::array<double, 4> & feature,
        double point_us,
        uint64_t now_ms) {
    server_cache_observation_instance out;
    out.used = true;
    out.estimator_slot = slot;
    out.key = key;
    out.n_success = 64;
    out.n_validation = 8;
    out.fit_region_count = key.operation ==
            server_cache_observation_operation::replay ? 5 : 4;
    out.validation_region_count = 3;
    out.fit_region_minutes = { 1, 2, 3, 4 };
    out.validation_region_minutes = { 1, 2, 3 };
    out.last_validation_unix_ms = now_ms;
    out.safe_measurable_opportunities = 8;
    out.opportunity_at_last_validation = 8;
    for (uint8_t i = 0; i < key.feature_dim; ++i) {
        out.v[i][i] = 1.0e12;
        out.b[i] = point_us * 1.0e12;
        out.feature_min[i] = feature[i];
        out.feature_max[i] = feature[i];
    }
    out.response_reservoir.fill(10);
    out.reservoir_seen = out.response_reservoir.size();
    return out;
}

static void test_local_by_id_certification_and_currency() {
    constexpr uint64_t now_ms = 600000;
    server_cache_execution_fingerprint fingerprint;
    fingerprint.complete = true;
    fingerprint.exact = true;
    fingerprint.execution_root[0] = 0x71;

    server_cache_observation_store observations;
    observations.set_execution_fingerprint(fingerprint);
    observations.set_calibration_claim_identity(true, 3, 5);

    server_cache_plan_local_inventory evidence;
    common_cache_plan_record rec;
    rec.selection = common_cache_plan_selection::by_id;
    rec.id_slot = 0;
    rec.n_prompt_tokens = llama_cache_acct_value::measured(100);
    auto * live = add_viable(
        rec, common_cache_plan_provider::live_slot, 0, 0);
    live->lcp_tokens = llama_cache_acct_value::measured(50);
    auto * checkpoint = add_viable(
        rec, common_cache_plan_provider::live_context_checkpoint, 0, 0);
    checkpoint->lcp_tokens = llama_cache_acct_value::measured(90);
    checkpoint->payload_bytes = llama_cache_acct_value::measured(1024);
    auto * host = add_viable(
        rec, common_cache_plan_provider::host_cache_entry, 7, 0);
    host->lcp_tokens = llama_cache_acct_value::measured(80);
    host->payload_bytes = llama_cache_acct_value::measured(2048);
    const uint32_t live_id = uint32_t(live - rec.inventory.data());
    const uint32_t checkpoint_id = uint32_t(checkpoint - rec.inventory.data());
    const uint32_t host_id = uint32_t(host - rec.inventory.data());

    uint8_t family = 0;
    uint8_t batch_bucket = 0;
    std::array<double, 4> live_feature = {};
    CHECK(server_cache_observation_replay_chain_geometry(
        50, 50, family, batch_bucket, live_feature));
    auto live_key = local_key(
        common_cache_plan_provider::live_slot,
        server_cache_observation_operation::replay,
        fingerprint.execution_root);
    live_key.size_family = family;
    live_key.batch_bucket = batch_bucket;
    live_key.ubatch_bucket = 0;
    live_key.start_bucket = 0;
    std::array<double, 4> checkpoint_feature = {};
    CHECK(server_cache_observation_byte_feature(
        1024, family, checkpoint_feature));
    auto checkpoint_key = local_key(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_observation_operation::restore,
        fingerprint.execution_root);
    checkpoint_key.size_family = family;
    checkpoint_key.batch_bucket =
        server_cache_observation_batch_bucket(10);
    checkpoint_key.ubatch_bucket = 0;
    checkpoint_key.start_bucket = 0;
    std::array<double, 4> host_feature = {};
    CHECK(server_cache_observation_byte_feature(
        2048, family, host_feature));
    auto host_key = local_key(
        common_cache_plan_provider::host_cache_entry,
        server_cache_observation_operation::restore,
        fingerprint.execution_root);
    host_key.size_family = family;
    CHECK(server_cache_observation_apply_restore_geometry(
        host_key, 80, 10, 1));
    std::array<double, 4> prepare_feature = {};
    CHECK(server_cache_observation_byte_feature(
        1024, family, prepare_feature));
    auto prepare_key = local_key(
        common_cache_plan_provider::live_slot,
        server_cache_observation_operation::durability_prepare,
        fingerprint.execution_root);
    prepare_key.prepare_shape = 2;
    prepare_key.size_family = family;

    std::array<server_cache_observation_instance,
               server_cache_observation_store::instance_capacity> instances = {};
    instances[0] = mature_local_instance(
        0, live_key, live_feature, 1000, now_ms);
    instances[1] = mature_local_instance(
        1, checkpoint_key, checkpoint_feature, 100000, now_ms);
    instances[2] = mature_local_instance(
        2, prepare_key, prepare_feature, 100, now_ms);
    instances[3] = mature_local_instance(
        3, host_key, host_feature, 50000, now_ms);
    // Point-complete but deliberately provisional. V1 requires confidence
    // only for the selected baseline/challenger, not every third candidate.
    instances[3].n_validation = 1;
    instances[3].validation_region_count = 1;
    CHECK(observations.restore_persisted_instances(instances, 1));
    server_cache_resume_validation_flags resume_pending = {};
    resume_pending[0] = true;
    resume_pending[1] = true;
    resume_pending[2] = true;
    resume_pending[3] = true;
    observations.set_resume_state(resume_pending, resume_pending, 0);
    observations.set_calibration_claim_identity(true, 3, 5);
    observations.set_committed_profile_mutation_generation(UINT64_MAX);

    evidence.candidates[live_id] = { live_key, live_feature, true };
    evidence.candidates[checkpoint_id] = {
        checkpoint_key, checkpoint_feature, true };
    evidence.candidates[host_id] = { host_key, host_feature, true };
    const auto original_rec = rec;
    server_cache_calibration_authority_snapshot snapshot;
    CHECK(server_cache_calibration_capture_snapshot(
        observations, now_ms, snapshot));
    server_cache_calibration_snapshot_lookup live_lookup;
    server_cache_calibration_snapshot_lookup checkpoint_lookup;
    server_cache_calibration_snapshot_lookup prepare_lookup;
    CHECK(server_cache_calibration_snapshot_lookup_exact(
        snapshot, live_key, live_feature, live_lookup));
    CHECK(server_cache_calibration_snapshot_lookup_exact(
        snapshot, checkpoint_key, checkpoint_feature, checkpoint_lookup));
    CHECK(server_cache_calibration_snapshot_lookup_exact(
        snapshot, prepare_key, prepare_feature, prepare_lookup));
    CHECK(live_lookup.state == server_cache_calibration_instance_state::active);
    CHECK(checkpoint_lookup.state == server_cache_calibration_instance_state::active);
    CHECK(prepare_lookup.state == server_cache_calibration_instance_state::active);
    server_cache_calibration_contribution direct_terms[2];
    direct_terms[0] = { checkpoint_lookup.instance, checkpoint_lookup.claim,
        checkpoint_feature, 1000,
        server_cache_calibration_contribution_side::baseline, now_ms, true };
    direct_terms[1] = { live_lookup.instance, live_lookup.claim,
        live_feature, 1000,
        server_cache_calibration_contribution_side::challenger, now_ms, true };
    server_cache_calibration_direct_bound direct;
    CHECK(server_cache_calibration_bound_direct_difference(
        direct_terms, 2, direct));
    CHECK(direct.status == server_cache_calibration_prediction_status::ok);
    CHECK(direct.benefit_lower_us > 0.0);
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id, &observations);
    std::array<uint8_t, 32> display_salt = {};
    display_salt[0] = 0xa5;
    CHECK(authority.set_profile_display_salt(display_salt));
    std::string display_a;
    std::string display_a_repeat;
    CHECK(authority.profile_display_label(
        fingerprint.execution_root, display_a));
    CHECK(authority.profile_display_label(
        fingerprint.execution_root, display_a_repeat));
    CHECK(display_a == display_a_repeat);
    server_cache_plan_authority other_process(
        common_cache_plan_authority_level::by_id, &observations);
    display_salt[1] = 0x5a;
    CHECK(other_process.set_profile_display_salt(display_salt));
    std::string display_b;
    CHECK(other_process.profile_display_label(
        fingerprint.execution_root, display_b));
    CHECK(display_a != display_b);
    const uint64_t planning_serial = observations.authority_currency_serial();
    server_cache_plan_local_authority_latch local_latch;
    authority.plan_local_before_mutation(
        rec, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &local_latch);
    CHECK(observations.authority_currency_serial() == planning_serial);
    CHECK(rec.planner_status == common_cache_plan_planner_status::ok);
    CHECK(rec.shadow_choice == int32_t(live_id));
    CHECK(rec.optimizer.economic_disposition ==
          common_cache_optimizer_disposition::certified_improvement);
    CHECK(rec.optimizer.local_authority.certified_once);
    CHECK(rec.optimizer.benefit_lower_known &&
          rec.optimizer.benefit_lower_us > 0.0);
    CHECK(rec.optimizer.profile_identity.rfind("local-", 0) == 0);
    CHECK(rec.optimizer.profile_identity.find("71") == std::string::npos);
    auto receipt_only = rec;
    auto execution = authority.authorize(rec, 0, true, true, 0,
        std::move(local_latch));
    CHECK(execution.kind == server_cache_plan_execution_kind::live_replay);
    auto copied_execution = authority.authorize(
        receipt_only, 0, true, true, 0, std::move(local_latch));
    CHECK(!copied_execution.authoritative());
    CHECK(receipt_only.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(receipt_only.authority.fallback_reason ==
          common_cache_plan_authority_fallback::internal_fault);
    rec.shipped_plan_candidate = rec.shadow_choice;
    authority.finalize_execution(rec, &execution.local_authority);
    CHECK(rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::executed);

    // A certified latch is bound to the exact legacy baseline ordinal used by
    // its economic proof. A changed execution-time baseline must consume the
    // capability through the stale-currency fallback, never reuse the proof.
    auto baseline_mismatch = original_rec;
    server_cache_plan_local_authority_latch baseline_mismatch_latch;
    authority.plan_local_before_mutation(
        baseline_mismatch, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &baseline_mismatch_latch);
    CHECK(baseline_mismatch.authority_prequalified);
    baseline_mismatch.optimizer.baseline_plan_candidate = int32_t(host_id);
    auto baseline_mismatch_execution = authority.authorize(
        baseline_mismatch, 0, true, true, 0,
        std::move(baseline_mismatch_latch));
    CHECK(!baseline_mismatch_execution.authoritative());
    CHECK(baseline_mismatch.authority.fallback_reason ==
          common_cache_plan_authority_fallback::stale_capability);
    CHECK(baseline_mismatch.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::fallback);
    CHECK(baseline_mismatch.optimizer.local_authority.reason ==
          common_cache_optimizer_fallback_reason::currency_changed);

    // The strict zero policy margin never promotes equality. Identical
    // baseline/challenger features cancel before the confidence radius.
    server_cache_calibration_contribution equal_terms[2];
    equal_terms[0] = { live_lookup.instance, live_lookup.claim, live_feature,
        1000, server_cache_calibration_contribution_side::baseline,
        now_ms, true };
    equal_terms[1] = equal_terms[0];
    equal_terms[1].side =
        server_cache_calibration_contribution_side::challenger;
    server_cache_calibration_direct_bound equal_bound;
    CHECK(server_cache_calibration_bound_direct_difference(
        equal_terms, 2, equal_bound));
    CHECK(equal_bound.benefit_lower_us == 0.0);

    auto immature = original_rec;
    auto * cold = add_viable(
        immature, common_cache_plan_provider::cold_replay, -1, 0);
    CHECK(cold != nullptr);
    authority.plan_local_before_mutation(
        immature, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms);
    CHECK(!immature.authority_prequalified);
    CHECK(immature.optimizer.local_fallback_reason ==
          common_cache_optimizer_fallback_reason::incomplete_evidence);

    auto chain = original_rec;
    auto * composed = chain.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(live_id), int32_t(checkpoint_id));
    CHECK(composed != nullptr);
    composed->disposition =
        common_cache_plan_disposition::valid_not_chosen_cost;
    authority.plan_local_before_mutation(
        chain, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms);
    CHECK(chain.authority_prequalified);
    CHECK(chain.planner_status == common_cache_plan_planner_status::ok);
    CHECK(chain.shadow_choice == int32_t(live_id));
    const int32_t chain_id = int32_t(composed - chain.inventory.data());
    auto chain_baseline = chain;
    authority.plan_local_before_mutation(
        chain_baseline, evidence, observations, chain_id,
        9, 9, now_ms);
    CHECK(!chain_baseline.authority_prequalified);
    CHECK(chain_baseline.optimizer.local_fallback_reason ==
          common_cache_optimizer_fallback_reason::incomplete_evidence);

    auto exception_rec = original_rec;
    server_cache_plan_local_authority_latch exception_latch;
    authority.plan_local_before_mutation(
        exception_rec, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &exception_latch);
    auto exception_execution = authority.authorize(
        exception_rec, 0, true, true, 0, std::move(exception_latch));
    CHECK(exception_execution.authoritative());
    authority.fail_closed(
        exception_rec,
        common_cache_plan_authority_fallback::internal_fault,
        &exception_execution.local_authority);
    CHECK(exception_rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::fallback);
    CHECK(exception_rec.optimizer.local_authority.certified_once);
    CHECK(!exception_execution.local_authority.execute(
        exception_rec.optimizer.local_authority));

    auto mismatch_rec = original_rec;
    server_cache_plan_local_authority_latch mismatch_latch;
    authority.plan_local_before_mutation(
        mismatch_rec, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &mismatch_latch);
    auto mismatch_execution = authority.authorize(
        mismatch_rec, 0, true, true, 0, std::move(mismatch_latch));
    CHECK(mismatch_execution.authoritative());
    mismatch_rec.shipped_plan_candidate = int32_t(checkpoint_id);
    authority.finalize_execution(
        mismatch_rec, &mismatch_execution.local_authority);
    CHECK(mismatch_rec.authority.state ==
          common_cache_plan_authority_state::fallback_legacy);
    CHECK(mismatch_rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::fallback);

    auto priced_d = original_rec;
    auto priced_evidence = evidence;
    auto & priced_live = priced_evidence.candidates[live_id];
    priced_live.requires_d_consequences = true;
    priced_live.consequence_count = 1;
    priced_live.consequences[0] = {
        prepare_key, prepare_feature,
        llama_cache_acct_cost_kind::transfer, 1000, true };
    server_cache_plan_local_authority_latch priced_latch;
    authority.plan_local_before_mutation(
        priced_d, priced_evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &priced_latch);
    CHECK(priced_d.authority_prequalified);
    CHECK(priced_d.inventory[live_id].predicted_total_us.value >
          rec.inventory[live_id].predicted_total_us.value);
    CHECK(priced_d.inventory[live_id]
              .cost_terms[size_t(llama_cache_acct_cost_kind::transfer)]
              .estimated_us.state == llama_cache_acct_known::known);

    auto missing_d = original_rec;
    auto missing_evidence = evidence;
    missing_evidence.candidates[live_id].requires_d_consequences = true;
    authority.plan_local_before_mutation(
        missing_d, missing_evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms);
    CHECK(!missing_d.authority_prequalified);
    CHECK(missing_d.optimizer.local_fallback_reason ==
          common_cache_optimizer_fallback_reason::incomplete_evidence);

    auto higher_tier = original_rec;
    higher_tier.selection = common_cache_plan_selection::similarity;
    authority.plan_local_before_mutation(
        higher_tier, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms);
    CHECK(!higher_tier.authority_prequalified);
    CHECK(higher_tier.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::not_attempted);

    // ZC5b raises only the graduated ceiling. The same estimator, receipt,
    // and move-only latch can certify a strict-similarity same-target replay
    // over the historical host restore; a by-id authority above already
    // refused this identical decision tier.
    server_cache_observation_store similarity_observations;
    similarity_observations.set_execution_fingerprint(fingerprint);
    similarity_observations.set_calibration_claim_identity(true, 3, 5);
    auto similarity_instances = instances;
    similarity_instances[3].n_validation = 8;
    similarity_instances[3].validation_region_count = 3;
    CHECK(similarity_observations.restore_persisted_instances(
        similarity_instances, 1));
    similarity_observations.set_calibration_claim_identity(true, 3, 5);
    similarity_observations.set_committed_profile_mutation_generation(
        UINT64_MAX);
    server_cache_plan_authority similarity_authority(
        common_cache_plan_authority_level::similarity,
        &similarity_observations);
    CHECK(similarity_authority.set_profile_display_salt(display_salt));
    const auto set_host_legacy = [&](common_cache_plan_record & value) {
        value.inventory[live_id].f_keep = 0.4;
        value.inventory[live_id].f_keep_known = true;
        value.inventory[live_id].sim = 0.8;
        value.inventory[live_id].sim_known = true;
        value.inventory[host_id].f_keep = 0.8;
        value.inventory[host_id].f_keep_known = true;
        value.inventory[host_id].sim = 0.9;
        value.inventory[host_id].sim_known = true;
    };
    auto similarity_rec = original_rec;
    set_host_legacy(similarity_rec);
    similarity_rec.selection = common_cache_plan_selection::similarity;
    similarity_rec.inventory[live_id].origin_tier =
        common_cache_plan_selection::similarity;
    similarity_rec.inventory[host_id].origin_tier =
        common_cache_plan_selection::similarity;
    auto similarity_evidence = evidence;
    // Production always retains a cold control. Replacing the live slot with
    // that control has a nonnegative D cost whose local preparation class may
    // be immature. Its optimistic zero-cost total is still far above the
    // mature live replay, so it cannot block the safe challenger. If this
    // missing-D candidate itself wins, the existing missing_d case above
    // proves that execution still refuses incomplete evidence.
    similarity_evidence.candidates[checkpoint_id].
        requires_d_consequences = true;
    server_cache_plan_local_authority_latch similarity_latch;
    similarity_authority.plan_local_before_mutation(
        similarity_rec, similarity_evidence, similarity_observations,
        int32_t(host_id),
        9, 9, now_ms, 0, &similarity_latch);
    CHECK(similarity_rec.authority_prequalified);
    CHECK(similarity_rec.shadow_choice == int32_t(live_id));
    CHECK(similarity_rec.optimizer.economic_disposition ==
          common_cache_optimizer_disposition::certified_improvement);
    auto similarity_execution = similarity_authority.authorize(
        similarity_rec, 0, true, true, 0, std::move(similarity_latch));
    CHECK(similarity_execution.kind ==
          server_cache_plan_execution_kind::live_replay);
    CHECK(similarity_execution.target == 0);
    similarity_rec.shipped_plan_candidate = int32_t(live_id);
    similarity_authority.finalize_execution(
        similarity_rec, &similarity_execution.local_authority);
    CHECK(similarity_rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::executed);

    // ZC5c graduates the same local capability one tier further. A nontrivial
    // home prefix may use the mature same-target live replay instead of the
    // historical host restore, while the prior similarity ceiling refuses
    // the identical route-home record before planning.
    auto route_rec = original_rec;
    set_host_legacy(route_rec);
    route_rec.selection = common_cache_plan_selection::route_home;
    route_rec.inventory[live_id].origin_tier =
        common_cache_plan_selection::route_home;
    route_rec.inventory[host_id].origin_tier =
        common_cache_plan_selection::route_home;
    auto route_lower = route_rec;
    similarity_authority.plan_local_before_mutation(
        route_lower, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms);
    CHECK(!route_lower.authority_prequalified);
    CHECK(route_lower.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::not_attempted);

    server_cache_plan_authority route_authority(
        common_cache_plan_authority_level::route_home,
        &similarity_observations);
    CHECK(route_authority.set_profile_display_salt(display_salt));
    server_cache_plan_local_authority_latch route_latch;
    route_authority.plan_local_before_mutation(
        route_rec, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms, 0, &route_latch);
    CHECK(route_rec.authority_prequalified);
    CHECK(route_rec.shadow_choice == int32_t(live_id));
    auto route_execution = route_authority.authorize(
        route_rec, 0, true, true, 0, std::move(route_latch));
    CHECK(route_execution.kind ==
          server_cache_plan_execution_kind::live_replay);
    CHECK(route_execution.target == 0);
    route_rec.shipped_plan_candidate = int32_t(live_id);
    route_authority.finalize_execution(
        route_rec, &route_execution.local_authority);
    CHECK(route_rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::executed);

    // ZC5d completes the graduated local ceiling. The same mature capability
    // may act on an LRU record, while the completed ZC5c route-home ceiling
    // remains a hard boundary for the identical evidence and choice.
    auto lru_rec = original_rec;
    set_host_legacy(lru_rec);
    lru_rec.selection = common_cache_plan_selection::lru;
    lru_rec.inventory[live_id].origin_tier =
        common_cache_plan_selection::lru;
    lru_rec.inventory[host_id].origin_tier =
        common_cache_plan_selection::lru;
    lru_rec.inventory[checkpoint_id].origin_tier =
        common_cache_plan_selection::lru;
    lru_rec.inventory[live_id].spec_capable_known = true;
    lru_rec.inventory[live_id].spec_capable = false;
    auto lru_lower = lru_rec;
    route_authority.plan_local_before_mutation(
        lru_lower, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms);
    CHECK(!lru_lower.authority_prequalified);
    CHECK(lru_lower.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::not_attempted);

    server_cache_plan_authority lru_authority(
        common_cache_plan_authority_level::lru,
        &similarity_observations);
    CHECK(lru_authority.set_profile_display_salt(display_salt));
    server_cache_plan_local_authority_latch lru_latch;
    lru_authority.plan_local_before_mutation(
        lru_rec, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms, 0, &lru_latch);
    CHECK(lru_rec.authority_prequalified);
    CHECK(lru_rec.shadow_choice == int32_t(live_id));
    auto lru_execution = lru_authority.authorize(
        lru_rec, 0, true, true, 0, std::move(lru_latch));
    CHECK(lru_execution.kind ==
          server_cache_plan_execution_kind::live_replay);
    CHECK(lru_execution.target == 0);
    lru_rec.shipped_plan_candidate = int32_t(live_id);
    lru_authority.finalize_execution(
        lru_rec, &lru_execution.local_authority);
    CHECK(lru_rec.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::executed);

    // Local fits share the exact LRU hard-stratum completeness rule with the
    // checked-in estimator. Unknown speculation on the selected target, or a
    // viable target with no live/spec carrier, refuses before an optimum.
    auto lru_unknown_spec = lru_rec;
    lru_unknown_spec.clear_planner_outputs();
    lru_unknown_spec.inventory[live_id].spec_capable_known = false;
    lru_authority.plan_local_before_mutation(
        lru_unknown_spec, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms);
    CHECK(!lru_unknown_spec.authority_prequalified);
    CHECK(lru_unknown_spec.optimizer.local_fallback_reason ==
          common_cache_optimizer_fallback_reason::incomplete_evidence);

    auto lru_missing_target = lru_rec;
    lru_missing_target.clear_planner_outputs();
    lru_missing_target.inventory[host_id].target_slot_id = 1;
    lru_authority.plan_local_before_mutation(
        lru_missing_target, similarity_evidence, similarity_observations,
        int32_t(host_id), 9, 9, now_ms);
    CHECK(!lru_missing_target.authority_prequalified);
    CHECK(lru_missing_target.optimizer.local_fallback_reason ==
          common_cache_optimizer_fallback_reason::incomplete_evidence);

    // A copyable receipt is diagnostic only. Without the move-only capability
    // produced by this exact planning invocation it cannot authorize.
    auto stale = receipt_only;
    CHECK(observations.note_safe_measurable_opportunity(live_key, 7));
    CHECK(!authority.authorize(stale, 0).authoritative());
    CHECK(stale.optimizer.local_authority.state ==
          common_cache_optimizer_authority_state::fallback);
    CHECK(stale.optimizer.local_authority.reason ==
          common_cache_optimizer_fallback_reason::internal_fault);

    auto stale_currency = original_rec;
    server_cache_plan_local_authority_latch stale_latch;
    authority.plan_local_before_mutation(
        stale_currency, evidence, observations, int32_t(checkpoint_id),
        9, 9, now_ms, 0, &stale_latch);
    CHECK(stale_currency.authority_prequalified);
    CHECK(observations.note_safe_measurable_opportunity(live_key, 8));
    CHECK(!authority.authorize(stale_currency, 0, true, true, 0,
        std::move(stale_latch)).authoritative());
    CHECK(stale_currency.optimizer.local_authority.reason ==
          common_cache_optimizer_fallback_reason::currency_changed);

    uint64_t terminal_mutation = 10;
    const auto check_terminal = [&] (
            server_cache_calibration_authority_terminal terminal,
            common_cache_optimizer_fallback_reason reason,
            common_cache_optimizer_profile_state state,
            common_cache_optimizer_coverage_class coverage,
            bool economic_candidate_available) {
        auto terminal_instances = instances;
        terminal_instances[live_id].authority_terminal = terminal;
        CHECK(observations.restore_persisted_instances(
            terminal_instances, terminal_mutation++));
        observations.set_calibration_claim_identity(true, 3, 5);
        observations.set_committed_profile_mutation_generation(UINT64_MAX);
        auto terminal_rec = original_rec;
        authority.plan_local_before_mutation(
            terminal_rec, evidence, observations, int32_t(checkpoint_id),
            9, 9, now_ms);
        CHECK(!terminal_rec.authority_prequalified);
        CHECK(terminal_rec.optimizer.local_fallback_reason == reason);
        CHECK(terminal_rec.optimizer.profile_state == state);
        CHECK(terminal_rec.optimizer.coverage_class == coverage);
        CHECK((terminal_rec.optimizer.economic_plan_candidate >= 0) ==
              economic_candidate_available);
    };
    check_terminal(
        server_cache_calibration_authority_terminal::tail_exceeded,
        common_cache_optimizer_fallback_reason::out_of_coverage,
        common_cache_optimizer_profile_state::provisional,
        common_cache_optimizer_coverage_class::out_of_coverage, true);
    check_terminal(
        server_cache_calibration_authority_terminal::confidence_budget_exhausted,
        common_cache_optimizer_fallback_reason::insufficient_confidence,
        common_cache_optimizer_profile_state::provisional,
        common_cache_optimizer_coverage_class::confidence_inactive, true);
    check_terminal(
        server_cache_calibration_authority_terminal::drifted,
        common_cache_optimizer_fallback_reason::drifted,
        common_cache_optimizer_profile_state::drifted,
        common_cache_optimizer_coverage_class::confidence_inactive, true);
    check_terminal(
        server_cache_calibration_authority_terminal::numeric_fault,
        common_cache_optimizer_fallback_reason::internal_fault,
        common_cache_optimizer_profile_state::quarantined,
        common_cache_optimizer_coverage_class::unavailable, false);

    // Confidence, coverage, and drift are contribution-local. The unselected
    // host row keeps its point estimate and diagnostic state, but cannot veto
    // the active checkpoint-vs-live comparison. The selected-row cases above
    // prove that the same terminals still fail closed with their typed reason.
    for (const auto terminal : {
            server_cache_calibration_authority_terminal::tail_exceeded,
            server_cache_calibration_authority_terminal::drifted,
            server_cache_calibration_authority_terminal::confidence_budget_exhausted }) {
        auto terminal_instances = instances;
        terminal_instances[host_id].authority_terminal = terminal;
        CHECK(observations.restore_persisted_instances(
            terminal_instances, terminal_mutation++));
        observations.set_calibration_claim_identity(true, 3, 5);
        observations.set_committed_profile_mutation_generation(UINT64_MAX);
        auto unselected_terminal = original_rec;
        server_cache_plan_local_authority_latch unselected_latch;
        authority.plan_local_before_mutation(
            unselected_terminal, evidence, observations,
            int32_t(checkpoint_id), 9, 9, now_ms, 0, &unselected_latch);
        CHECK(unselected_terminal.authority_prequalified);
        CHECK(unselected_terminal.optimizer.local_fallback_reason ==
              common_cache_optimizer_fallback_reason::none);
        CHECK(unselected_terminal.optimizer.coverage_class ==
              common_cache_optimizer_coverage_class::complete);
    }
}

static void test_local_by_id_max_cardinality_budget() {
    constexpr uint64_t now_ms = 600000;
    server_cache_execution_fingerprint fingerprint;
    fingerprint.complete = true;
    fingerprint.exact = true;
    fingerprint.execution_root[0] = 0x72;

    server_cache_observation_store observations;
    observations.set_execution_fingerprint(fingerprint);
    observations.set_calibration_claim_identity(true, 3, 5);

    common_cache_plan_record base;
    base.selection = common_cache_plan_selection::by_id;
    base.id_slot = 0;
    base.n_prompt_tokens = llama_cache_acct_value::measured(100);
    server_cache_plan_local_inventory evidence;
    std::array<server_cache_observation_instance,
               server_cache_observation_store::instance_capacity> instances = {};
    uint8_t family = 0;
    uint8_t batch_bucket = 0;
    std::array<double, 4> feature = {};
    CHECK(server_cache_observation_replay_chain_geometry(
        50, 50, family, batch_bucket, feature));
    for (uint32_t i = 0; i < COMMON_CACHE_PLAN_MAX_CANDIDATES; ++i) {
        auto * row = add_viable(
            base, common_cache_plan_provider::live_slot, int32_t(i), 0);
        row->lcp_tokens = llama_cache_acct_value::measured(50);
        const uint32_t candidate = uint32_t(row - base.inventory.data());
        auto key = local_key(
            common_cache_plan_provider::live_slot,
            server_cache_observation_operation::replay,
            fingerprint.execution_root);
        key.adapter_application_digest[0] = uint8_t(i + 1);
        CHECK(server_cache_calibration_single_participant_digest_v1(
            key.adapter_application_digest, key.representation_digest, 0,
            key.participant_execution_digest));
        key.size_family = family;
        key.batch_bucket = batch_bucket;
        key.ubatch_bucket = 0;
        key.start_bucket = 0;
        instances[i] = mature_local_instance(
            i, key, feature, i == 0 ? 100.0 : 100000.0 + i, now_ms);
        evidence.candidates[candidate] = { key, feature, true };
    }
    CHECK(observations.restore_persisted_instances(instances, 1));
    observations.set_calibration_claim_identity(true, 3, 5);
    observations.set_committed_profile_mutation_generation(UINT64_MAX);
    server_cache_plan_authority authority(
        common_cache_plan_authority_level::by_id, &observations);
    std::array<uint8_t, 32> salt = {};
    salt[0] = 0x5a;
    CHECK(authority.set_profile_display_salt(salt));

    std::array<uint64_t, 101> elapsed_us = {};
    for (size_t sample = 0; sample < elapsed_us.size(); ++sample) {
        auto rec = base;
        server_cache_plan_local_authority_latch latch;
        const auto begin = std::chrono::steady_clock::now();
        authority.plan_local_before_mutation(
            rec, evidence, observations,
            int32_t(COMMON_CACHE_PLAN_MAX_CANDIDATES - 1),
            9, 9, now_ms, 0, &latch);
        const auto end = std::chrono::steady_clock::now();
        elapsed_us[sample] = uint64_t(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - begin).count());
        CHECK(rec.authority_prequalified);
        CHECK(rec.shadow_choice == 0);
    }
    std::sort(elapsed_us.begin(), elapsed_us.end());
    std::fprintf(stderr,
        "ZC5_MAX_CARDINALITY candidates=96 instances=128 p50_us=%llu p95_us=%llu max_us=%llu\n",
        (unsigned long long) elapsed_us[50],
        (unsigned long long) elapsed_us[95],
        (unsigned long long) elapsed_us.back());
    // CI is not the hardware acceptance gate, but a gross algorithmic
    // regression must still fail here. Dorei separately pins the 2 ms policy
    // budget on the production CPU/compiler.
    CHECK(elapsed_us[95] < 10000);
}

int main() {
    test_candidate_classifiers();
    test_checkpoint_orientation_and_host_identity();
    test_compose_excludes_destroyed_live_checkpoint();
    test_composed_chain_reuses_inventory_identity();
    test_inventory_saturation_refuses_qualification();
    test_stale_capability_refuses_without_throwing();
    test_execution_seam_fallbacks();
    test_qualified_fallback_remains_eligible();
    test_by_id_execution_shapes_and_target_binding();
    test_legacy_counterfactual_and_authoritative_receipt();
    test_similarity_crossover_and_safety_envelope();
    test_route_home_authority_domain();
    test_lru_authority_domain_and_eviction_fence();
    test_typed_planner_fallbacks();
    test_off_stays_shadow_and_failed_delivery_not_counted();
    test_eligible_and_executed_index_different_tiers();
    test_local_by_id_certification_and_currency();
    test_local_by_id_max_cardinality_budget();
    return 0;
}
