#include "server-cache-plan-authority.h"
#include "common-cache-plan-estimate.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static void test_candidate_classifiers() {
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

    std::array<const void *, COMMON_CACHE_PLAN_MAX_CANDIDATES> instances{};
    uint32_t n_instances = 0;
    int old_prefix_instance = 0;
    int survivor_instance = 0;
    int replacement_instance = 0;
    int32_t source = -1;
    CHECK(server_cache_plan_find_or_assign_source_id(
        &old_prefix_instance, instances, n_instances, source));
    const int32_t old_prefix = source;
    CHECK(source == 0);
    CHECK(server_cache_plan_find_or_assign_source_id(
        &survivor_instance, instances, n_instances, source));
    const int32_t survivor = source;
    CHECK(source == 1);
    // Save-time prefix dedup removes old_prefix. The surviving physical node
    // must keep its identity after its list ordinal shifts from 1 to 0.
    CHECK(server_cache_plan_find_or_assign_source_id(
        &survivor_instance, instances, n_instances, source));
    CHECK(source == 1);
    // The freshly saved replacement cannot inherit either removed ordinal.
    CHECK(server_cache_plan_find_or_assign_source_id(
        &replacement_instance, instances, n_instances, source));
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

    auto * host = rec.find_or_add(
        common_cache_plan_provider::host_cache_entry, 4, 0, 2,
        common_cache_plan_selection::similarity);
    CHECK(host != nullptr);
    host->accept();
    host->delivered = true;
    rec.select(common_cache_plan_provider::host_cache_entry, host);

    auto * selected = rec.find_or_add(
        common_cache_plan_provider::live_context_checkpoint,
        server_cache_plan_host_checkpoint_source_id(4, 0), 0, 2,
        common_cache_plan_selection::similarity);
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
    selected->component_only = true;
    selected->dependent_host_source_id = 4;
    sibling->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    sibling->component_only = true;
    sibling->dependent_host_source_id = 4;
    foreign->note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    foreign->component_only = true;
    foreign->dependent_host_source_id = 5;
    rec.select(common_cache_plan_provider::live_context_checkpoint, selected);
    rec.chosen = common_cache_plan_provider::live_context_checkpoint;

    auto * selected_chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(host - rec.inventory.data()),
        int32_t(selected - rec.inventory.data()));
    auto * sibling_chain = rec.add_chain(
        common_cache_plan_provider::host_cache_entry,
        int32_t(host - rec.inventory.data()),
        int32_t(sibling - rec.inventory.data()));
    CHECK(selected_chain && sibling_chain);
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
    authority.finalize_legacy_execution(rec);

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

int main() {
    test_candidate_classifiers();
    test_checkpoint_orientation_and_host_identity();
    test_compose_excludes_destroyed_live_checkpoint();
    test_composed_chain_reuses_inventory_identity();
    test_inventory_saturation_refuses_qualification();
    test_stale_capability_refuses_without_throwing();
    test_eligible_and_executed_index_different_tiers();
    return 0;
}
