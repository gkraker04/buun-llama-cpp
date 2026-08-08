#include "server-cache-observer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    out.participant_execution_digest[0] = identity;
    return out;
}

static server_cache_observation_record accepted(uint8_t identity) {
    server_cache_observation_record out;
    out.key = key(identity);
    CHECK(server_cache_observation_replay_feature(
        512, out.key.size_family, out.feature));
    out.owned_cpu_us = 20;
    out.backend_service_us = 80;
    out.owned_service_us = 100;
    out.capped_service_us = 100;
    out.terminal = server_cache_observation_terminal::accepted;
    return out;
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
        auto shadow_key = key(9);
        shadow_key.adapter_application_complete = false;
        shadow_key.identity_complete = false;
        shadow_key.identity_exact = false;
        auto exact_key = key(9);
        exact_key.identity_exact = true;
        CHECK(shadow_key == exact_key);
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
        CHECK(operation_key.participant_execution_digest[0] == 8);
        CHECK(operation_key.representation_digest[0] == 0);
        CHECK(!operation_key.identity_complete);
        CHECK(operation_key.identity_exact);
    }
    {
        std::array<double, 4> phi;
        uint8_t family = 99;
        CHECK(!server_cache_observation_replay_feature(0, family, phi));
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        CHECK(family == 0);
        CHECK(std::abs(phi[3] - 1.0) < 1e-12);
        CHECK(server_cache_observation_replay_feature(513, family, phi));
        CHECK(family == 1);
        CHECK(!server_cache_observation_replay_feature(65537, family, phi));

        CHECK(server_cache_observation_byte_feature(0, family, phi));
        CHECK(family == 0);
        CHECK(std::abs(phi[0] - 1.0) < 1e-12);
        CHECK(std::abs(phi[1]) < 1e-12);
        CHECK(server_cache_observation_byte_feature(
            64ULL * 1024 * 1024 + 1, family, phi));
        CHECK(family == 1);
        CHECK(!server_cache_observation_byte_feature(
            4096ULL * 1024 * 1024 + 1, family, phi));
    }

    {
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(3, key(1), 20);
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(512, family, phi));
        epoch.note_submission(submission(120, 7, 90, 512, family, phi));
        server_cache_observation_record record;
        CHECK(!epoch.finish(store, &record));
        CHECK(!epoch.latch_fence({ 7, 200 }));
        CHECK(epoch.latch_fence({ 8, 200 }));
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
        // Owned CPU is a sum of explicitly-owned spans. Lookup, scheduler,
        // observer-attribution, and sampler gaps between those spans are not
        // part of the response learned by the optimizer.
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(14, key(14), 5);
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
        epoch.arm(1, incomplete, 10);
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
        server_cache_observation_store store;
        server_cache_calibration_epoch epoch;
        epoch.arm(2, key(3), 10);
        std::array<double, 4> phi;
        uint8_t family = 0;
        CHECK(server_cache_observation_replay_feature(16, family, phi));
        epoch.note_submission(submission(20, 2, 15, 16, family, phi));
        CHECK(epoch.latch_fence({ 3, 100 }));
        epoch.note_submission(submission(10020, 3, 100, 16, family, phi));
        server_cache_observation_record record;
        CHECK(!epoch.latch_fence({ 4, 10030 }));
        epoch.mark_operation_terminal();
        CHECK(epoch.finish(store, &record));
        CHECK(record.reason ==
              server_cache_observation_reason::multiple_submissions);
        CHECK(record.terminal == server_cache_observation_terminal::diagnostic);
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
        epoch.arm(4, key(4), 10);
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
        prepare.feature_dim = 2;
        server_cache_observation_record record;
        CHECK(server_cache_observe_cpu_operation(
            &store, prepare, 1, 0, 100, 175, true, &record));
        CHECK(record.feature[0] == 1.0);
        CHECK(record.feature[1] == 0.0);
        CHECK(record.owned_service_us == 75);
        CHECK(store.instances()[0].b[0] == 75.0);

        CHECK(!server_cache_observe_cpu_operation(
            &store, prepare, 1, 1, 200, 205, false, &record));
        CHECK(record.terminal ==
              server_cache_observation_terminal::operation_unavailable);
        CHECK(store.instances()[0].n_success == 1);
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

    std::puts("PASS");
    return 0;
}
