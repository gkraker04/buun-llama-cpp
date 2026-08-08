#include "server-cache-calibration-model.h"
#include "server-cache-calibration-store.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#define CHECK(value) do { \
    if (!(value)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #value); \
        std::abort(); \
    } \
} while (0)

namespace {

std::string digest_hex(const std::array<uint8_t, 32> & digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[2 * i] = digits[digest[i] >> 4];
        out[2 * i + 1] = digits[digest[i] & 0x0f];
    }
    return out;
}

server_cache_observation_key key(uint8_t dim = 2) {
    server_cache_observation_key out;
    out.operation = server_cache_observation_operation::restore;
    out.provider = common_cache_plan_provider::host_cache_entry;
    out.restore_kind = 1;
    out.model_kind = server_cache_calibration_model_kind::restore_scaled;
    out.size_family = 0;
    out.feature_dim = dim;
    out.identity_complete = true;
    out.identity_exact = true;
    out.adapter_application_complete = true;
    out.profile_execution_digest[0] = 1;
    out.participant_execution_digest[0] = 2;
    out.adapter_application_digest[0] = 3;
    out.representation_digest[0] = 4;
    CHECK(server_cache_calibration_effect_action_digest_v1(
        nullptr, 0, out.effect_action_shape_digest));
    return out;
}

server_cache_observation_instance instance(uint8_t dim = 2) {
    server_cache_observation_instance out;
    out.used = true;
    out.estimator_slot = 0;
    out.key = key(dim);
    for (uint8_t i = 0; i < dim; ++i) out.v[i][i] = 1.0;
    return out;
}

server_cache_observation_record record(
        const server_cache_observation_instance & value,
        std::array<double, 4> feature,
        uint64_t service_us) {
    server_cache_observation_record out;
    out.key = value.key;
    out.feature = feature;
    out.owned_cpu_us = service_us;
    out.owned_service_us = service_us;
    const uint64_t cap = server_cache_observation_response_cap_us(
        out.key.operation, out.key.size_family);
    out.capped_service_us = std::min(service_us, cap);
    out.tail_exceeded = service_us > cap;
    out.terminal = server_cache_observation_terminal::accepted;
    return out;
}

server_cache_calibration_update_context context(uint64_t ordinal) {
    server_cache_calibration_update_context out;
    out.steady_second = ordinal + 1;
    out.unix_minute = 100 + ordinal / 4;
    out.unix_ms = (100 + ordinal) * 60000;
    out.claim.boot_claim_ordinal = 0;
    out.claim.available = true;
    out.claim.profile_generation_ordinal = 0;
    out.claim.estimator_slot = 0;
    out.claim.fit_generation = 0;
    return out;
}

void test_known_coefficients_and_assignment() {
    auto value = instance();
    uint64_t fit = 0;
    uint64_t validation = 0;
    for (uint64_t i = 0; i < 64; ++i) {
        const bool high = (i & 1) != 0;
        const auto row = record(value,
            high ? std::array<double, 4>{0, 1, 0, 0}
                 : std::array<double, 4>{1, 0, 0, 0},
            high ? 3000 : 1000);
        auto clock = context(i);
        server_cache_calibration_update_result result;
        CHECK(server_cache_calibration_update(value, row, clock, result));
        fit += result.assignment == server_cache_calibration_assignment::fit;
        validation += result.assignment ==
            server_cache_calibration_assignment::validation;
    }
    CHECK(fit == 56);
    CHECK(validation == 8);
    CHECK(value.n_success == fit);
    CHECK(value.n_validation == validation);
    CHECK(value.qualified_execution_ordinal == 64);
    CHECK(value.fit_region_count > 1);
    CHECK(value.validation_region_count > 1);

    server_cache_calibration_prediction low;
    auto claim = context(0).claim;
    CHECK(server_cache_calibration_predict(value, claim, {1, 0, 0, 0}, low));
    CHECK(low.status == server_cache_calibration_prediction_status::ok);
    CHECK(low.point_us > 900 && low.point_us < 1100);
    CHECK(std::isfinite(low.radius_us));
    server_cache_calibration_prediction high;
    CHECK(server_cache_calibration_predict(value, claim, {0, 1, 0, 0}, high));
    CHECK(high.point_us > 2800 && high.point_us < 3100);
    CHECK(high.condition_number <= 1e8);
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0}, value.last_validation_unix_ms,
        nullptr) == server_cache_calibration_instance_state::active);
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0}, value.last_validation_unix_ms,
        nullptr, false) == server_cache_calibration_instance_state::provisional);
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0},
        value.last_validation_unix_ms + 10 * 60 * 1000 + 1,
        nullptr) == server_cache_calibration_instance_state::provisional);
}

void test_preassignment_and_rate_limit() {
    auto value = instance();
    auto row = record(value, {1, 0, 0, 0}, 1000);
    auto clock = context(0);
    server_cache_calibration_update_result first;
    CHECK(server_cache_calibration_update(value, row, clock, first));
    CHECK(first.assignment == server_cache_calibration_assignment::fit);

    server_cache_calibration_update_result second;
    CHECK(server_cache_calibration_update(value, row, clock, second));
    CHECK(second.assignment ==
          server_cache_calibration_assignment::fit_rate_limited);
    CHECK(value.n_success == 1);
    CHECK(value.qualified_execution_ordinal == 2);

    value.qualified_execution_ordinal = 7;
    auto forced = context(1);
    server_cache_calibration_update_result validation;
    CHECK(server_cache_calibration_update(value, row, forced, validation));
    CHECK(validation.assignment ==
          server_cache_calibration_assignment::validation_unavailable);
    CHECK(value.n_success == 1);
    CHECK(value.n_validation == 0);
    CHECK(value.qualified_execution_ordinal == 8);

    auto principal_limited = context(2);
    principal_limited.principal_admission_allowed = false;
    server_cache_calibration_update_result limited;
    CHECK(server_cache_calibration_update(
        value, row, principal_limited, limited));
    CHECK(limited.assignment ==
          server_cache_calibration_assignment::fit_rate_limited);

    auto failed = instance();
    failed.qualified_execution_ordinal = 7;
    server_cache_calibration_preassignment unavailable;
    CHECK(server_cache_calibration_preassign(
        failed, row.feature, context(10), unavailable));
    CHECK(unavailable.qualified_execution_ordinal == 7);
    CHECK(unavailable.assignment ==
          server_cache_calibration_assignment::validation_unavailable);
    CHECK(server_cache_calibration_abandon(failed, unavailable));
    CHECK(failed.qualified_execution_ordinal == 8);
    CHECK(failed.n_success == 0 && failed.n_validation == 0);
    server_cache_calibration_preassignment next;
    CHECK(server_cache_calibration_preassign(
        failed, row.feature, context(11), next));
    CHECK(next.qualified_execution_ordinal == 8);
    CHECK(next.assignment == server_cache_calibration_assignment::fit);
}

void test_validation_schedule_breaks_periodic_aliasing() {
    constexpr std::array<uint8_t, 64> expected = {
        7, 1, 6, 5, 2, 2, 0, 7, 6, 4, 2, 5, 3, 7, 6, 5,
        7, 3, 2, 4, 4, 7, 2, 6, 4, 1, 2, 2, 4, 0, 6, 2,
        1, 0, 5, 3, 3, 7, 0, 4, 2, 1, 5, 0, 3, 6, 7, 5,
        3, 0, 3, 0, 2, 6, 5, 4, 7, 1, 2, 4, 7, 1, 2, 5,
    };
    std::array<uint32_t, 8> within_block = {};
    std::array<uint32_t, 2> alternating_shape = {};
    for (uint64_t block = 0; block < 64; ++block) {
        uint32_t selected = 0;
        for (uint64_t offset = 0; offset < 8; ++offset) {
            const uint64_t ordinal = block * 8 + offset;
            if (!server_cache_calibration_validation_assignment(ordinal)) {
                continue;
            }
            ++selected;
            CHECK(offset == expected[block]);
            ++within_block[offset];
            ++alternating_shape[ordinal % 2];
        }
        CHECK(selected == 1);
    }
    CHECK(std::all_of(within_block.begin(), within_block.end(),
                      [](uint32_t count) { return count != 0; }));
    CHECK(alternating_shape[0] != 0 && alternating_shape[1] != 0);
}

void test_validation_carries_preupdate_prediction() {
    auto value = instance();
    for (uint64_t i = 0; i < 7; ++i) {
        auto row = record(value, {1, 0, 0, 0}, 1000);
        server_cache_calibration_update_result result;
        CHECK(server_cache_calibration_update(value, row, context(i), result));
    }
    CHECK(value.qualified_execution_ordinal == 7);
    server_cache_calibration_prediction before;
    CHECK(server_cache_calibration_predict(
        value, context(7).claim, {1, 0, 0, 0}, before));
    auto validation_row = record(value, {1, 0, 0, 0}, 1200);
    server_cache_calibration_update_result result;
    CHECK(server_cache_calibration_update(
        value, validation_row, context(7), result));
    CHECK(result.assignment == server_cache_calibration_assignment::validation);
    CHECK(result.validation_prediction_available);
    CHECK(result.validation_prediction.point_us == before.point_us);
    CHECK(result.validation_prediction.radius_us == before.radius_us);
}

void test_tail_is_capped_once_then_terminal() {
    auto value = instance();
    const uint64_t cap = server_cache_observation_response_cap_us(
        value.key.operation, value.key.size_family);
    const auto row = record(value, {1, 0, 0, 0}, cap + 123);
    server_cache_calibration_update_result result;
    CHECK(server_cache_calibration_update(value, row, context(0), result));
    CHECK(result.assignment == server_cache_calibration_assignment::fit);
    CHECK(result.moments_changed);
    CHECK(result.tail_latched);
    CHECK(value.n_success == 1);
    CHECK(value.b[0] == double(cap));
    CHECK(value.authority_terminal ==
          server_cache_calibration_authority_terminal::tail_exceeded);
    CHECK(value.tail_actual_max_us == cap + 123);
    server_cache_calibration_prediction prediction;
    CHECK(!server_cache_calibration_predict(
        value, context(0).claim, {1, 0, 0, 0}, prediction));
}

void test_validation_drift() {
    auto value = instance();
    for (uint64_t i = 0; i < 40; ++i) {
        const auto row = record(value, {1, 0, 0, 0}, 1000);
        auto clock = context(i);
        server_cache_calibration_update_result result;
        CHECK(server_cache_calibration_update(value, row, clock, result));
    }
    CHECK(value.n_success >= 4);
    for (uint64_t i = 0; i < 100 &&
            value.authority_terminal ==
                server_cache_calibration_authority_terminal::none; ++i) {
        const auto row = record(value, {1, 0, 0, 0}, 1900000);
        auto clock = context(100 + i);
        clock.force_validation = true;
        server_cache_calibration_update_result result;
        CHECK(server_cache_calibration_update(value, row, clock, result));
    }
    CHECK(value.authority_terminal ==
          server_cache_calibration_authority_terminal::drifted);
}

void test_arena_layout() {
    CHECK(server_cache_calibration_arena::layout_valid());
    server_cache_calibration_arena arena;
    CHECK(arena.allocate());
    CHECK(arena.ready());
    CHECK(arena.region(
        server_cache_calibration_arena_layout::profile_slots_begin,
        server_cache_calibration_arena_layout::profile_slots_size, 64) != nullptr);
    CHECK(arena.region(
        server_cache_calibration_arena_layout::reserve_begin,
        server_cache_calibration_arena_layout::reserve_size, 64) != nullptr);
    CHECK(arena.region(
        server_cache_calibration_arena_layout::total_size - 32, 64, 32) == nullptr);
    CHECK(arena.region(1, 1, 64) == nullptr);
    server_cache_calibration_arena_ptr<server_cache_observation_store> observer(
        arena.construct<server_cache_observation_store>(
            server_cache_calibration_arena_layout::global_tables_begin,
            server_cache_calibration_arena_layout::global_tables_size));
    server_cache_calibration_arena_ptr<
        server_cache_calibration_snapshot_workspace> snapshots(
            arena.construct<server_cache_calibration_snapshot_workspace>(
                server_cache_calibration_arena_layout::snapshots_begin,
                server_cache_calibration_arena_layout::snapshots_size));
    server_cache_calibration_arena_ptr<server_cache_calibration_coordinator> coordinator(
        arena.construct<server_cache_calibration_coordinator>(
            server_cache_calibration_arena_layout::profile_slots_begin,
            server_cache_calibration_arena_layout::profile_slots_size,
            snapshots.get(),
            arena.region(server_cache_calibration_arena_layout::codec_scratch_begin,
                         server_cache_calibration_arena_layout::codec_scratch_size,
                         64),
            server_cache_calibration_arena_layout::codec_scratch_size));
    CHECK(observer != nullptr);
    CHECK(snapshots != nullptr);
    CHECK(coordinator != nullptr);
    coordinator.reset();
    snapshots.reset();
    observer.reset();
    arena.reset();
    CHECK(!arena.ready());
}

void test_operation_identity_codec() {
    std::array<uint8_t, 32> representation_a = {};
    std::array<uint8_t, 32> representation_b = {};
    CHECK(server_cache_calibration_representation_digest_v1(
        "kf16-vf16", 10, representation_a));
    CHECK(server_cache_calibration_representation_digest_v1(
        "kf16-vf16", 10, representation_b));
    CHECK(representation_a == representation_b);
    CHECK(digest_hex(representation_a) ==
          "a7858688132b3e941dbab32411778c4e239d7ad5c8622332a8c6c28cf171dd25");
    CHECK(server_cache_calibration_representation_digest_v1(
        "kq8-vq8", 7, representation_b));
    CHECK(representation_a != representation_b);

    std::array<uint8_t, 32> adapter_a = {};
    std::array<uint8_t, 32> adapter_b = {};
    adapter_a[0] = 1;
    adapter_b[0] = 2;
    std::array<uint8_t, 32> participant_a = {};
    std::array<uint8_t, 32> participant_b = {};
    CHECK(server_cache_calibration_single_participant_digest_v1(
        adapter_a, representation_a, 0, participant_a));
    CHECK(digest_hex(participant_a) ==
          "eac03d10de5122acd96d57c86777b4ffdf88880a8ae7abfda49f9d6767d129f7");
    CHECK(server_cache_calibration_single_participant_digest_v1(
        adapter_b, representation_a, 0, participant_b));
    CHECK(participant_a != participant_b);
    CHECK(server_cache_calibration_single_participant_digest_v1(
        adapter_a, representation_a, 1, participant_b));
    CHECK(participant_a != participant_b);

    server_cache_calibration_participant_v1 participants[2] = {
        { adapter_a, 0, representation_a, 0 },
        { adapter_b, 0, representation_b, 0 },
    };
    std::array<uint8_t, 32> ordered = {};
    std::array<uint8_t, 32> reversed = {};
    CHECK(server_cache_calibration_participant_digest_v1(
        participants, 2, ordered));
    std::swap(participants[0], participants[1]);
    CHECK(server_cache_calibration_participant_digest_v1(
        participants, 2, reversed));
    CHECK(ordered != reversed);
    participants[0].media_runtime_class = 1;
    CHECK(!server_cache_calibration_participant_digest_v1(
        participants, 2, reversed));

    const server_cache_calibration_effect_action_v1 actions[2] = {
        { common_cache_plan_destruction_effect::cross_target_displacement,
          server_cache_destruction_class::slot_drop,
          server_cache_destruction_release_owner::legacy_wrapper_or_capability },
        { common_cache_plan_destruction_effect::different_host_source_consumption,
          server_cache_destruction_class::host_artifact_drop,
          server_cache_destruction_release_owner::legacy_wrapper_or_capability },
    };
    std::array<uint8_t, 32> action_digest = {};
    CHECK(server_cache_calibration_effect_action_digest_v1(
        actions, 2, action_digest));
    CHECK(digest_hex(action_digest) ==
          "a2a1f01936864a2c37c5babe993d21a8398ad38a35c6066227062ba30aef8b5a");
    const server_cache_calibration_effect_action_v1 reversed_actions[2] = {
        actions[1], actions[0],
    };
    CHECK(server_cache_calibration_effect_action_digest_v1(
        reversed_actions, 2, reversed));
    CHECK(action_digest != reversed);
    const server_cache_calibration_effect_action_v1 duplicate_actions[2] = {
        actions[0], actions[0],
    };
    CHECK(!server_cache_calibration_effect_action_digest_v1(
        duplicate_actions, 2, reversed));
    const server_cache_calibration_effect_action_v1 impossible_action = {
        common_cache_plan_destruction_effect::cross_target_displacement,
        server_cache_destruction_class::live_range_drop,
        server_cache_destruction_release_owner::legacy_wrapper_or_capability,
    };
    CHECK(!server_cache_calibration_effect_action_digest_v1(
        &impossible_action, 1, reversed));
    CHECK(server_cache_calibration_effect_action_digest_v1(
        nullptr, 0, reversed));
    CHECK((reversed != std::array<uint8_t, 32>{}));
    CHECK(server_cache_calibration_apply_shape_digest_v1(
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement) |
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop),
        server_cache_destruction_class::slot_drop,
        server_cache_destruction_release_owner::legacy_wrapper_or_capability,
        reversed));
    CHECK(!server_cache_calibration_apply_shape_digest_v1(
        0, server_cache_destruction_class::slot_drop,
        server_cache_destruction_release_owner::legacy_wrapper_or_capability,
        reversed));
    CHECK(!server_cache_calibration_apply_shape_digest_v1(
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::cross_target_displacement),
        server_cache_destruction_class::live_range_drop,
        server_cache_destruction_release_owner::legacy_wrapper_or_capability,
        reversed));
}

void test_exact_model_key_contract() {
    auto scaled = key();
    CHECK(server_cache_observation_key_valid(scaled));
    scaled.operation_extent_bytes = 1;
    CHECK(!server_cache_observation_key_valid(scaled));

    auto flat = key(1);
    flat.model_kind = server_cache_calibration_model_kind::restore_flat;
    flat.operation_extent_bytes = 1234;
    CHECK(server_cache_observation_key_valid(flat));
    flat.feature_dim = 2;
    CHECK(!server_cache_observation_key_valid(flat));

    auto prepare = key();
    prepare.operation =
        server_cache_observation_operation::durability_prepare;
    prepare.restore_kind = 0;
    prepare.prepare_shape = 2;
    prepare.model_kind =
        server_cache_calibration_model_kind::durability_prepare_scaled;
    CHECK(server_cache_observation_key_valid(prepare));
    prepare.prepare_shape = 0;
    CHECK(!server_cache_observation_key_valid(prepare));
}

server_cache_observation_instance trained_two_point_instance() {
    auto value = instance();
    for (uint64_t i = 0; i < 64; ++i) {
        const bool high = (i & 1) != 0;
        const auto row = record(value,
            high ? std::array<double, 4>{0, 1, 0, 0}
                 : std::array<double, 4>{1, 0, 0, 0},
            high ? 3000 : 1000);
        server_cache_calibration_update_result result;
        CHECK(server_cache_calibration_update(value, row, context(i), result));
    }
    return value;
}

void test_direct_difference_bound() {
    auto value = trained_two_point_instance();
    auto claim = context(0).claim;
    server_cache_calibration_contribution cancel[2] = {
        { &value, claim, { 1, 0, 0, 0 }, 1000,
          server_cache_calibration_contribution_side::baseline,
          value.last_validation_unix_ms },
        { &value, claim, { 1, 0, 0, 0 }, 1000,
          server_cache_calibration_contribution_side::challenger,
          value.last_validation_unix_ms },
    };
    server_cache_calibration_direct_bound bound;
    CHECK(server_cache_calibration_bound_direct_difference(
        cancel, 2, bound));
    CHECK(std::abs(bound.benefit_us) < 1e-12);
    CHECK(std::abs(bound.radius_us) < 1e-12);

    server_cache_calibration_contribution faster[2] = {
        { &value, claim, { 0, 1, 0, 0 }, 1000,
          server_cache_calibration_contribution_side::baseline,
          value.last_validation_unix_ms },
        { &value, claim, { 1, 0, 0, 0 }, 1000,
          server_cache_calibration_contribution_side::challenger,
          value.last_validation_unix_ms },
    };
    CHECK(server_cache_calibration_bound_direct_difference(
        faster, 2, bound));
    CHECK(bound.benefit_us > 1700 && bound.benefit_us < 2200);
    CHECK(bound.radius_us > 0);

    // A policy retention weight scales the same-key challenger feature and
    // can reverse the point estimate; it does not create timing evidence.
    faster[1].weight_milli = 4000;
    CHECK(server_cache_calibration_bound_direct_difference(
        faster, 2, bound));
    CHECK(bound.benefit_us < 0);

    faster[0].claim.fit_generation = value.fit_generation + 1;
    CHECK(!server_cache_calibration_bound_direct_difference(
        faster, 2, bound));
    CHECK(bound.status == server_cache_calibration_prediction_status::learning);
    faster[0].claim = claim;

    auto immature = instance();
    faster[1] = { &immature, claim, { 1, 0, 0, 0 }, 1000,
                  server_cache_calibration_contribution_side::challenger,
                  value.last_validation_unix_ms };
    CHECK(!server_cache_calibration_bound_direct_difference(
        faster, 2, bound));
    CHECK(bound.status == server_cache_calibration_prediction_status::learning);
}

void test_numeric_and_claim_fail_closed() {
    auto value = trained_two_point_instance();
    server_cache_calibration_prediction prediction;
    auto unavailable_claim = context(0).claim;
    unavailable_claim.available = false;
    CHECK(!server_cache_calibration_predict(
        value, unavailable_claim, { 1, 0, 0, 0 }, prediction));
    CHECK(prediction.status ==
          server_cache_calibration_prediction_status::confidence_budget_exhausted);

    value.v = {};
    value.v[0][0] = 1e12;
    value.v[1][1] = 1.0;
    value.n_success = 10;
    value.feature_min = { 0, 0, 0, 0 };
    value.feature_max = { 1, 1, 0, 0 };
    CHECK(!server_cache_calibration_predict(
        value, context(0).claim, { 1, 0, 0, 0 }, prediction));
    CHECK(prediction.status ==
          server_cache_calibration_prediction_status::numeric_fault);

    auto overflow = instance();
    overflow.n_success = UINT64_MAX;
    const auto overflow_row = record(overflow, {1, 0, 0, 0}, 1000);
    server_cache_calibration_update_result update;
    CHECK(!server_cache_calibration_update(
        overflow, overflow_row, context(1), update));
    CHECK(overflow.authority_terminal ==
          server_cache_calibration_authority_terminal::numeric_fault);
    CHECK(!server_cache_calibration_update(
        overflow, overflow_row, context(2), update));
    CHECK(overflow.authority_terminal ==
          server_cache_calibration_authority_terminal::numeric_fault);
}

void test_exactness_and_opportunity_boundary() {
    auto value = trained_two_point_instance();
    auto claim = context(0).claim;
    server_cache_calibration_prediction prediction;
    value.key.identity_exact = false;
    CHECK(server_cache_calibration_predict(
        value, claim, {1, 0, 0, 0}, prediction));
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0}, value.last_validation_unix_ms,
        nullptr) == server_cache_calibration_instance_state::provisional);
    value.key.identity_exact = true;
    value.opportunity_at_last_validation = 0;
    value.safe_measurable_opportunities = 255;
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0}, value.last_validation_unix_ms,
        nullptr) == server_cache_calibration_instance_state::active);
    value.safe_measurable_opportunities = 256;
    CHECK(server_cache_calibration_state(
        value, claim, {1, 0, 0, 0}, value.last_validation_unix_ms,
        nullptr) == server_cache_calibration_instance_state::provisional);
}

} // namespace

int main() {
    test_known_coefficients_and_assignment();
    test_preassignment_and_rate_limit();
    test_validation_schedule_breaks_periodic_aliasing();
    test_validation_carries_preupdate_prediction();
    test_tail_is_capped_once_then_terminal();
    test_validation_drift();
    test_arena_layout();
    test_operation_identity_codec();
    test_exact_model_key_contract();
    test_direct_difference_bound();
    test_numeric_and_claim_fail_closed();
    test_exactness_and_opportunity_boundary();
    std::puts("server cache calibration model tests passed");
    return 0;
}
