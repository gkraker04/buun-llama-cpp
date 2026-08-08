#include "server-cache-observer.h"
#include "server-cache-calibration-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #x); \
        std::abort(); \
    } \
} while (0)

static server_cache_observation_key key(uint8_t identity) {
    server_cache_observation_key out;
    out.operation = server_cache_observation_operation::replay;
    out.provider = common_cache_plan_provider::live_slot;
    out.feature_dim = 4;
    out.adapter_application_complete = true;
    out.identity_complete = true;
    const uint8_t nonzero_identity = uint8_t(identity + 1);
    out.profile_execution_digest[0] = nonzero_identity;
    out.participant_execution_digest[0] = nonzero_identity;
    out.adapter_application_digest[0] = nonzero_identity;
    out.representation_digest[0] = nonzero_identity;
    CHECK(server_cache_calibration_effect_action_digest_v1(
        nullptr, 0, out.effect_action_shape_digest));
    return out;
}

static server_cache_observation_record accepted(uint8_t identity) {
    server_cache_observation_record out;
    out.key = key(identity);
    out.admission_clock = { true, 1000000, 60000 };
    CHECK(server_cache_observation_replay_feature(
        512, out.key.size_family, out.feature));
    out.owned_cpu_us = 20;
    out.backend_service_us = 80;
    out.owned_service_us = 100;
    out.capped_service_us = 100;
    out.terminal = server_cache_observation_terminal::accepted;
    return out;
}

static server_cache_observation_cpu_start cpu_start(
        int64_t owned_start_us,
        int64_t steady_us,
        uint64_t unix_ms) {
    return { owned_start_us, { true, steady_us, unix_ms } };
}

static server_cache_observation_admission_clock admission_clock() {
    return { true, 1000000, 60000 };
}

static server_cache_observation_submission submission(
        int64_t at,
        uint64_t fence_serial,
        int64_t fence_us,
        uint32_t tokens,
        uint8_t family,
        const std::array<double, 4> & feature) {
    server_cache_observation_submission out;
    out.submission_us = at;
    out.fence_before = { fence_serial, fence_us };
    out.prompt_slots = 1;
    out.active_slots = 1;
    out.tokens = tokens;
    out.start_position = 0;
    out.effective_batch = tokens;
    out.effective_ubatch = tokens;
    out.target_participates = true;
    out.size_family = family;
    out.feature = feature;
    return out;
}

int main() {
    static_assert(sizeof(server_cache_observation_store) <= 256 * 1024,
                  "ZC2 process-local store exceeded its fixed host bound");
    {
        server_cache_observation_store store;
        CHECK(store.prepare_slot_scratch(
            server_cache_observation_store::slot_scratch_capacity));
        CHECK(!store.prepare_slot_scratch(
            server_cache_observation_store::slot_scratch_capacity + 1));
        CHECK(store.note_slot_submission(0, 2, 17));
        CHECK(store.note_slot_submission(0, 3, 11));
        CHECK(store.slot_batch_tokens(0) == 5);
        CHECK(store.slot_first_position(0) == 11);
        store.reset_slot_scratch();
        CHECK(store.slot_batch_tokens(0) == 0);
        CHECK(store.slot_first_position(0) ==
              std::numeric_limits<int32_t>::max());
    }
    {
        auto shadow_key = key(9);
        shadow_key.adapter_application_complete = false;
        shadow_key.identity_complete = false;
        shadow_key.identity_exact = false;
        auto exact_key = key(9);
        exact_key.identity_exact = true;
        CHECK(!(shadow_key == exact_key));
    }
    {
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.exact = true;
        fingerprint.execution_root[0] = 0xe3;
        fingerprint.config_root[0] = 0xc3;
        store.set_execution_fingerprint(fingerprint);

        auto operation_key = key(8);
        operation_key.identity_complete = true;
        store.apply_execution_fingerprint(operation_key);
        CHECK(operation_key.profile_execution_digest[0] == 0xe3);
        CHECK(operation_key.participant_execution_digest[0] == 9);
        CHECK(operation_key.representation_digest[0] == 9);
        CHECK(!operation_key.identity_complete);
        CHECK(operation_key.identity_exact);
    }
    {
        // ZC4 completes operation-local identity only after the profile root,
        // representation recipe, and effective adapter application all join.
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.execution_root[0] = 0xa1;
        store.set_execution_fingerprint(fingerprint);
        std::array<uint8_t, 32> representation = {};
        CHECK(server_cache_calibration_representation_digest_v1(
            "kf16-vf16", 10, representation));
        store.set_operation_identity(true, representation, 0);
        store.set_calibration_claim_identity(true, 0, 0);
        auto operation_key = server_cache_observation_cpu_key(
            server_cache_observation_operation::restore,
            common_cache_plan_provider::host_cache_entry, 0);
        operation_key.adapter_application_complete = true;
        operation_key.adapter_application_digest[0] = 0x44;
        store.apply_execution_fingerprint(operation_key);
        CHECK(operation_key.identity_complete);
        CHECK(operation_key.representation_digest == representation);
        CHECK((operation_key.participant_execution_digest !=
               std::array<uint8_t, 32>{}));
        CHECK(operation_key.restore_kind == 1);
    }
    {
        std::array<double, 4> phi;
        uint8_t family = 99;
        CHECK(!server_cache_observation_replay_feature(0, family, phi));
        constexpr uint64_t replay_caps[] = { 512, 2048, 8192, 65536 };
        for (uint8_t i = 0; i < 4; ++i) {
            CHECK(server_cache_observation_replay_feature(
                replay_caps[i], family, phi));
            CHECK(family == i);
            CHECK(std::abs(phi[3] - 1.0) < 1e-12);
            if (i != 3) {
                CHECK(server_cache_observation_replay_feature(
                    replay_caps[i] + 1, family, phi));
                CHECK(family == i + 1);
            }
        }
        CHECK(!server_cache_observation_replay_feature(
            replay_caps[3] + 1, family, phi));

        CHECK(server_cache_observation_byte_feature(0, family, phi));
        CHECK(family == 0);
        CHECK(std::abs(phi[0] - 1.0) < 1e-12);
        CHECK(std::abs(phi[1]) < 1e-12);
        constexpr uint64_t byte_caps[] = {
            64ULL * 1024 * 1024,
            256ULL * 1024 * 1024,
            1024ULL * 1024 * 1024,
            4096ULL * 1024 * 1024,
        };
        for (uint8_t i = 0; i < 4; ++i) {
            CHECK(server_cache_observation_byte_feature(
                byte_caps[i], family, phi));
            CHECK(family == i);
            CHECK(std::abs(phi[1] - 1.0) < 1e-12);
            if (i != 3) {
                CHECK(server_cache_observation_byte_feature(
                    byte_caps[i] + 1, family, phi));
                CHECK(family == i + 1);
            }
        }
        CHECK(!server_cache_observation_byte_feature(
            byte_caps[3] + 1, family, phi));
    }

    {
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(3, key(1), 20, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        epoch.note_submission(submission(120, 7, 90, 512, family, phi));
        server_cache_observation_record record;
        CHECK(!epoch.finish(store, &record));
        CHECK(!epoch.latch_fence({ 7, 200 }));
        CHECK(epoch.latch_fence({ 8, 200 }));
        // A completed fence is necessary but not sufficient: only the
        // provider-operation terminal may publish a successful sample.
        CHECK(!epoch.finish(store, &record));
        epoch.mark_operation_terminal();
        CHECK(epoch.finish(store, &record));
        CHECK(record.terminal == server_cache_observation_terminal::accepted);
        CHECK(record.reason == server_cache_observation_reason::none);
        CHECK(record.owned_cpu_us == 20);
        CHECK(record.backend_service_us == 80);
        CHECK(record.owned_service_us == 100);
        CHECK(record.capped_service_us == 100);
        CHECK(!record.tail_exceeded);
        CHECK(store.counters().accepted == 1);
        CHECK(store.instances()[0].n_success == 1);
        CHECK(store.instances()[0].b[3] == 100.0);
        CHECK(store.instances()[0].v[3][3] == 2.0);
    }

    {
        // A failed clock capture at provider ownership may not be retried when
        // the epoch is armed after provider work. The missing pre-outcome tuple
        // remains diagnostic and cannot allocate an estimator instance.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(3, key(1), 20, {});
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        epoch.note_submission(submission(120, 7, 90, 512, family, phi));
        CHECK(epoch.latch_fence({ 8, 200 }));
        epoch.mark_operation_terminal();
        server_cache_observation_record record;
        CHECK(epoch.finish(store, &record));
        CHECK(record.terminal == server_cache_observation_terminal::diagnostic);
        CHECK(record.reason == server_cache_observation_reason::invalid_geometry);
        CHECK(store.counters().accepted == 0);
        CHECK(store.instances()[0].used == false);
    }

    {
        auto first = submission(20, 2, 15, 16, 0, {});
        auto changed = first;
        changed.effective_batch = 512;
        changed.tokens = 512;
        changed.start_position = 16;
        CHECK(server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.prompt_slots;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.active_slots;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.payload_bytes;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.effective_ubatch;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; changed.target_participates = !changed.target_participates;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; changed.draft_participates = !changed.draft_participates;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; changed.speculative_participates = !changed.speculative_participates;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.contention_bucket;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
        changed = first; ++changed.ubatch_bucket;
        CHECK(!server_cache_observation_same_chain_geometry(first, changed));
    }

    {
        // Owned CPU is a sum of explicitly-owned spans. Lookup, scheduler,
        // observer-attribution, and sampler gaps between those spans are not
        // part of the response learned by the optimizer.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(14, key(14), 5, admission_clock());
        epoch.begin_owned_cpu(100);
        epoch.pause_owned_cpu(125);
        epoch.begin_owned_cpu(1000);
        epoch.pause_owned_cpu(1010);
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(8, family, phi));
        epoch.note_submission(submission(2000, 20, 1900, 8, family, phi));
        CHECK(epoch.latch_fence({ 21, 2040 }));
        epoch.mark_operation_terminal();
        server_cache_observation_record record;
        CHECK(epoch.finish(store, &record));
        CHECK(record.owned_cpu_us == 40);
        CHECK(record.backend_service_us == 40);
        CHECK(record.owned_service_us == 80);
    }

    {
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        auto incomplete = key(2);
        incomplete.identity_complete = false;
        epoch.arm(1, incomplete, 10, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(32, family, phi));
        epoch.note_submission(submission(20, 1, 5, 32, family, phi));
        server_cache_observation_record record;
        CHECK(epoch.latch_fence({ 2, 30 }));
        epoch.mark_operation_terminal();
        CHECK(epoch.finish(store, &record));
        CHECK(record.terminal == server_cache_observation_terminal::diagnostic);
        CHECK(record.reason ==
              server_cache_observation_reason::identity_unavailable);
        CHECK(store.counters().accepted == 0);
        CHECK(store.counters().diagnostic == 1);
    }

    {
        // Multi-submission replay sums separately fenced groups. The 9.9-ms
        // inter-group gap is outside both spans, and the total replay extent
        // selects the final feature family.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(3), 10, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        epoch.note_submission(submission(20, 2, 15, 512, family, phi));
        CHECK(epoch.latch_fence({ 3, 100 }));
        CHECK(server_cache_observation_replay_feature(1, family, phi));
        auto second = submission(10020, 3, 100, 1, family, phi);
        second.start_position = 512;
        second.start_bucket = 0;
        second.effective_ubatch = 512;
        epoch.note_submission(second);
        server_cache_observation_record record;
        CHECK(epoch.latch_fence({ 4, 10030 }));
        epoch.mark_operation_terminal();
        CHECK(epoch.finish(store, &record));
        CHECK(record.reason == server_cache_observation_reason::none);
        CHECK(record.terminal == server_cache_observation_terminal::accepted);
        CHECK(record.submissions == 2);
        CHECK(record.tokens == 513);
        CHECK(record.key.size_family == 1);
        CHECK(record.backend_service_us == 90);
        CHECK(record.owned_service_us == 100);
        CHECK(store.counters().accepted == 1);
    }

    {
        // A fence completed passively after the prior close hook. Consuming
        // it before the next provider CPU span excludes the idle gap and
        // counts post-fence CPU exactly once.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(30), 0, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(16, family, phi));
        epoch.note_submission(submission(20, 2, 15, 16, family, phi));
        CHECK(epoch.latch_fence({ 3, 50 }));
        epoch.begin_owned_cpu(100);
        epoch.pause_owned_cpu(110);
        auto second = submission(120, 3, 50, 16, family, phi);
        second.start_position = 16;
        epoch.note_submission(second);
        CHECK(epoch.latch_fence({ 4, 150 }));
        epoch.mark_operation_terminal();
        server_cache_observation_record record;
        CHECK(epoch.finish(store, &record));
        CHECK(record.backend_service_us == 60);
        CHECK(record.owned_cpu_us == 10);
        CHECK(record.owned_service_us == 70);
    }

    {
        // A failed decode attempt consumes its observation stream but cannot
        // be combined with the smaller successful retry.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(31), 0, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        epoch.note_submission(submission(20, 2, 15, 512, family, phi));
        server_cache_observation_record record;
        CHECK(epoch.abandon(
            server_cache_observation_reason::operation_failed,
            store, &record));
        CHECK(record.terminal ==
              server_cache_observation_terminal::operation_unavailable);
        epoch.note_submission(submission(40, 2, 15, 256, family, phi));
        CHECK(!epoch.latch_fence({ 3, 60 }));
        epoch.mark_operation_terminal();
        CHECK(!epoch.finish(store, &record));
        CHECK(store.counters().accepted == 0);
    }

    {
        // Retry failure is scoped to the attempted sub-batch. An unrelated
        // active epoch must survive and may later publish its own valid row.
        server_cache_observation_store store;
        CHECK(store.prepare_slot_scratch(2));
        CHECK(store.note_slot_submission(1, 16, 0));
        server_cache_calibration_epoch unaffected;
        server_cache_calibration_epoch attempted;
        unaffected.arm(0, key(32), 0, admission_clock());
        attempted.arm(1, key(33), 0, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(16, family, phi));
        unaffected.note_submission(
            submission(20, 2, 15, 16, family, phi));
        attempted.note_submission(
            submission(20, 2, 15, 16, family, phi));
        server_cache_observation_record record;
        if (store.slot_batch_tokens(size_t(attempted.slot_id())) > 0) {
            CHECK(attempted.abandon(
                server_cache_observation_reason::operation_failed,
                store, &record));
        }
        CHECK(unaffected.active());
        CHECK(!attempted.active());
        CHECK(unaffected.latch_fence({ 3, 50 }));
        unaffected.mark_operation_terminal();
        CHECK(unaffected.finish(store, &record));
        CHECK(record.terminal == server_cache_observation_terminal::accepted);
        CHECK(store.counters().accepted == 1);
        CHECK(store.counters().operation_unavailable == 1);
    }

    {
        // Several async prompt chunks may naturally share the final sampler
        // fence. The continuous 20->50 critical path is admitted once; the
        // CPU segment prepared while the first chunk was unresolved is not
        // stacked on top of it.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(4), 0, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(16, family, phi));
        epoch.note_submission(submission(20, 2, 15, 16, family, phi));
        epoch.begin_owned_cpu(30);
        epoch.pause_owned_cpu(35);
        auto second = submission(40, 2, 15, 16, family, phi);
        second.start_position = 16;
        epoch.note_submission(second);
        CHECK(epoch.latch_fence({ 3, 50 }));
        epoch.mark_operation_terminal();
        server_cache_observation_record record;
        CHECK(epoch.finish(store, &record));
        CHECK(record.terminal == server_cache_observation_terminal::accepted);
        CHECK(record.submissions == 2);
        CHECK(record.owned_cpu_us == 0);
        CHECK(record.backend_service_us == 30);
        CHECK(record.owned_service_us == 30);
    }

    {
        // An operation with no completion fence is still fail-closed.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(5), 0, admission_clock());
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(16, family, phi));
        epoch.note_submission(submission(20, 2, 15, 16, family, phi));
        epoch.mark_operation_terminal();
        server_cache_observation_record record;
        CHECK(!epoch.finish(store, &record));
        CHECK(epoch.abandon(
            server_cache_observation_reason::no_completion_fence,
            store, &record));
        CHECK(record.terminal ==
              server_cache_observation_terminal::operation_unavailable);
        CHECK(store.counters().accepted == 0);
    }

    {
        server_cache_observation_store store;
        for (size_t i = 0;
             i < server_cache_observation_store::instance_capacity; ++i) {
            auto row = accepted(uint8_t(i));
            CHECK(store.observe(row));
        }
        auto overflow = accepted(0);
        overflow.key.participant_execution_digest[1] = 1;
        CHECK(!store.observe(overflow));
        CHECK(store.counters().instance_capacity == 1);
        CHECK(overflow.reason == server_cache_observation_reason::instance_capacity);
        CHECK(store.records_seen() ==
              server_cache_observation_store::instance_capacity + 1);
    }

    {
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(4, key(4), 10, admission_clock());
        server_cache_observation_record record;
        CHECK(epoch.abandon(
            server_cache_observation_reason::operation_failed,
            store, &record));
        CHECK(record.terminal ==
              server_cache_observation_terminal::operation_unavailable);
        CHECK(store.counters().operation_unavailable == 1);
    }

    {
        server_cache_observation_store store;
        auto prepare = key(9);
        prepare.operation =
            server_cache_observation_operation::durability_prepare;
        prepare.model_kind =
            server_cache_calibration_model_kind::durability_prepare_scaled;
        prepare.prepare_shape = 1;
        prepare.feature_dim = 2;
        server_cache_observation_record record;
        CHECK(server_cache_observe_cpu_operation(
            &store, prepare, 1, 0, cpu_start(100, 1000000, 60000),
            175, true, true, &record));
        CHECK(record.feature[0] == 1.0);
        CHECK(record.feature[1] == 0.0);
        CHECK(record.owned_service_us == 75);
        CHECK(store.instances()[0].b[0] == 75.0);

        CHECK(!server_cache_observe_cpu_operation(
            &store, prepare, 1, 1, cpu_start(200, 2000000, 61000),
            205, false, true, &record));
        CHECK(record.terminal ==
              server_cache_observation_terminal::operation_unavailable);
        CHECK(store.instances()[0].n_success == 1);

        CHECK(!server_cache_observe_cpu_operation(
            &store, prepare, 1, 1, cpu_start(300, 3000000, 62000),
            305, true, false, &record));
        CHECK(record.reason == server_cache_observation_reason::mixed_slots);
        CHECK(store.instances()[0].n_success == 1);
    }

    {
        // Fit admission is frozen at provider ownership. A response that runs
        // across the next one-second boundary cannot admit itself by being
        // slower; only a different pre-outcome start second opens a new row.
        const auto second_assignment = [](uint64_t response_us) {
            server_cache_observation_store store;
            auto first = accepted(34);
            first.admission_clock = { true, 10100000, 60000 };
            CHECK(store.observe(first));
            auto second = accepted(34);
            second.admission_clock = { true, 10900000, 60000 };
            second.owned_cpu_us = response_us;
            second.backend_service_us = 0;
            second.owned_service_us = response_us;
            second.capped_service_us = response_us;
            CHECK(store.observe(second));
            CHECK(store.instances()[0].n_success == 1);
            return second.calibration_assignment;
        };
        const uint8_t short_assignment = second_assignment(100);
        const uint8_t long_assignment = second_assignment(900000);
        CHECK(short_assignment == long_assignment);
        CHECK(short_assignment ==
              uint8_t(server_cache_calibration_assignment::fit_rate_limited) + 1);

        server_cache_observation_store store;
        auto first = accepted(35);
        first.admission_clock = { true, 10100000, 60000 };
        CHECK(store.observe(first));
        auto next_second = accepted(35);
        next_second.admission_clock = { true, 11000000, 60000 };
        CHECK(store.observe(next_second));
        CHECK(next_second.calibration_assignment ==
              uint8_t(server_cache_calibration_assignment::fit) + 1);
        CHECK(store.instances()[0].n_success == 2);
    }

    {
        server_cache_observation_store store;
        auto missing_clock = accepted(36);
        missing_clock.admission_clock = {};
        CHECK(!store.observe(missing_clock));
        CHECK(missing_clock.reason ==
              server_cache_observation_reason::invalid_geometry);
        CHECK(std::none_of(store.instances().begin(), store.instances().end(),
                           [](const auto & value) { return value.used; }));
    }

    {
        server_cache_observation_store store;
        auto over_tail = accepted(10);
        over_tail.owned_cpu_us = 1000000;
        over_tail.backend_service_us = 1500000;
        over_tail.owned_service_us = 2500000;
        over_tail.capped_service_us = 2000000;
        over_tail.tail_exceeded = true;
        CHECK(store.observe(over_tail));
        CHECK(store.instances()[0].b[3] == 2000000.0);
        CHECK(store.instances()[0].response_reservoir[0] == 2000000);
        CHECK(store.instances()[0].tail_exceeded);

        auto inconsistent = accepted(11);
        inconsistent.owned_service_us = 101;
        inconsistent.capped_service_us = 101;
        CHECK(!store.observe(inconsistent));
        CHECK(store.counters().numeric_fault == 1);
        CHECK(inconsistent.reason ==
              server_cache_observation_reason::numeric_overflow);
        CHECK(store.recent_records()[1].reason ==
              server_cache_observation_reason::numeric_overflow);
    }

    {
        server_cache_observation_store store;
        auto invalid = accepted(12);
        invalid.terminal = server_cache_observation_terminal::_count;
        CHECK(!store.observe(invalid));
        CHECK(invalid.terminal == server_cache_observation_terminal::diagnostic);
        CHECK(invalid.reason == server_cache_observation_reason::numeric_overflow);

        auto warmup = accepted(13);
        warmup.warmup = true;
        CHECK(!store.observe(warmup));
        CHECK(warmup.reason ==
              server_cache_observation_reason::warmup_unsettled);
        CHECK(store.counters().accepted == 0);
    }

    {
        // A drifted stream is frozen; the next row allocates a fresh g in the
        // same bounded physical slot before inspecting the response.
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.execution_root = key(11).profile_execution_digest;
        store.set_execution_fingerprint(fingerprint);
        std::array<server_cache_observation_instance,
                   server_cache_observation_store::instance_capacity> instances = {};
        instances[0].used = true;
        instances[0].key = key(11);
        instances[0].fit_generation = 7;
        instances[0].authority_terminal =
            server_cache_calibration_authority_terminal::drifted;
        CHECK(store.restore_persisted_instances(instances, 1));
        auto next = accepted(11);
        CHECK(store.observe(next));
        CHECK(store.instances()[0].fit_generation == 8);
        CHECK(store.instances()[0].authority_terminal ==
              server_cache_calibration_authority_terminal::none);
        CHECK(store.instances()[0].n_success == 1);
    }

    {
        // An unrelated new key cannot discharge restart validation for a
        // persisted instance. The persisted key consumes the one-shot row.
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.execution_root = key(21).profile_execution_digest;
        store.set_execution_fingerprint(fingerprint);
        std::array<server_cache_observation_instance,
                   server_cache_observation_store::instance_capacity> instances = {};
        instances[0].used = true;
        instances[0].key = key(21);
        instances[0].n_success = 1;
        instances[0].v[0][0] = 2.0;
        instances[0].v[1][1] = 1.0;
        instances[0].v[2][2] = 1.0;
        instances[0].v[3][3] = 1.0;
        CHECK(store.restore_persisted_instances(instances, 1));
        server_cache_resume_validation_flags pending = {};
        pending[0] = true;
        store.set_resume_state(
            pending, pending, std::numeric_limits<int64_t>::max());
        auto unrelated = accepted(21);
        unrelated.key.participant_execution_digest[1] = 1;
        CHECK(store.observe(unrelated));
        CHECK(unrelated.calibration_assignment ==
              uint8_t(server_cache_calibration_assignment::fit_rate_limited) + 1);
        server_cache_resume_validation_outcome no_outcome;
        CHECK(store.take_resume_validation_outcomes(&no_outcome, 1) == 0);
        auto persisted = accepted(21);
        CHECK(store.observe(persisted));
        CHECK(persisted.calibration_assignment ==
              uint8_t(server_cache_calibration_assignment::validation_unavailable) + 1);
        server_cache_resume_validation_outcome unavailable;
        CHECK(store.take_resume_validation_outcomes(&unavailable, 1) == 1);
        CHECK(unavailable.estimator_slot == 0);
        CHECK(unavailable.kind ==
              server_cache_resume_validation_outcome_kind::unavailable);
    }

    {
        // Outcome failure at ordinal seven consumes the validation assignment;
        // the next success owns ordinal eight and cannot reuse it.
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.execution_root = key(31).profile_execution_digest;
        store.set_execution_fingerprint(fingerprint);
        std::array<server_cache_observation_instance,
                   server_cache_observation_store::instance_capacity> instances = {};
        instances[0].used = true;
        instances[0].key = key(31);
        instances[0].qualified_execution_ordinal = 7;
        for (uint8_t i = 0; i < 4; ++i) instances[0].v[i][i] = 1.0;
        CHECK(store.restore_persisted_instances(instances, 1));
        store.set_calibration_claim_identity(true, 0, 0);
        auto failed = accepted(31);
        failed.terminal = server_cache_observation_terminal::operation_unavailable;
        failed.reason = server_cache_observation_reason::operation_failed;
        CHECK(!store.observe(failed));
        CHECK(failed.calibration_assignment ==
              uint8_t(server_cache_calibration_assignment::validation_unavailable) + 1);
        CHECK(store.instances()[0].qualified_execution_ordinal == 8);
        CHECK(store.instances()[0].n_success == 0);
        auto next = accepted(31);
        CHECK(store.observe(next));
        CHECK(next.calibration_assignment ==
              uint8_t(server_cache_calibration_assignment::fit) + 1);
        CHECK(store.instances()[0].qualified_execution_ordinal == 9);
    }

    {
        // Restart obligations are independently armed and drained per slot.
        server_cache_observation_store store;
        server_cache_execution_fingerprint fingerprint;
        fingerprint.complete = true;
        fingerprint.execution_root = key(40).profile_execution_digest;
        store.set_execution_fingerprint(fingerprint);
        std::array<server_cache_observation_instance,
                   server_cache_observation_store::instance_capacity> instances = {};
        for (uint32_t slot = 0; slot < 2; ++slot) {
            instances[slot].used = true;
            instances[slot].key = key(uint8_t(40 + slot));
            for (uint8_t i = 0; i < 4; ++i) instances[slot].v[i][i] = 1.0;
        }
        // Both fixtures share the persisted profile root while retaining
        // distinct participant identities.
        instances[1].key.profile_execution_digest =
            instances[0].key.profile_execution_digest;
        CHECK(store.restore_persisted_instances(instances, 1));
        store.set_calibration_claim_identity(true, 0, 0);
        server_cache_resume_validation_flags pending = {};
        pending[0] = true;
        pending[1] = true;
        store.set_resume_state(pending, pending, 0);
        auto first = accepted(40);
        auto second = accepted(41);
        second.key.profile_execution_digest = first.key.profile_execution_digest;
        CHECK(store.observe(first));
        CHECK(store.observe(second));
        server_cache_resume_validation_outcome outcomes[2];
        CHECK(store.take_resume_validation_outcomes(outcomes, 2) == 2);
        CHECK(outcomes[0].estimator_slot == 0);
        CHECK(outcomes[1].estimator_slot == 1);
        CHECK(outcomes[0].kind ==
              server_cache_resume_validation_outcome_kind::unavailable);
        CHECK(outcomes[1].kind ==
              server_cache_resume_validation_outcome_kind::unavailable);
    }

    {
        // Inventory opportunities de-duplicate by exact key and inventory.
        server_cache_observation_store store;
        auto row = accepted(50);
        row.key.identity_exact = true;
        CHECK(store.observe(row));
        CHECK(store.note_safe_measurable_opportunity(row.key, 9));
        CHECK(store.note_safe_measurable_opportunity(row.key, 9));
        CHECK(store.instances()[0].safe_measurable_opportunities == 1);
        CHECK(store.note_safe_measurable_opportunity(row.key, 10));
        CHECK(store.instances()[0].safe_measurable_opportunities == 2);
    }

    std::puts("PASS");
    return 0;
}
