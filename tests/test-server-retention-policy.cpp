#include "server-cache-retention-policy.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <vector>

#define CHECK(condition)                                                                         \
    do {                                                                                         \
        if (!(condition)) {                                                                      \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                        \
        }                                                                                        \
    } while (0)

namespace {

server_cache_retention_member member(uint32_t ordinal, uint64_t frontier, bool incoming = false) {
    server_cache_retention_member out;
    out.ordinal   = ordinal;
    out.stable_id = uint64_t(ordinal) + 1;
    out.frontier  = frontier;
    out.incoming  = incoming;
    return out;
}

server_cache_retention_policy_config config(uint32_t capacity = 4, uint32_t recent = 2, uint32_t historical = 2) {
    server_cache_retention_policy_config out;
    out.capacity           = capacity;
    out.recent_floor       = recent;
    out.minimum_historical = historical;
    out.bucket_base        = 100;
    out.bucket_growth      = 2;
    return out;
}

server_cache_host_retention_member host_member(
        uint32_t ordinal,
        uint64_t last_use_epoch,
        uint64_t release_bytes = 100,
        uint64_t release_tokens = 10) {
    server_cache_host_retention_member out;
    out.ordinal        = ordinal;
    out.stable_id      = uint64_t(ordinal) + 1;
    out.last_use_epoch = last_use_epoch;
    out.release_bytes  = release_bytes;
    out.release_tokens = release_tokens;
    out.eligible       = true;
    return out;
}

void test_config_and_lane_quota() {
    CHECK(!server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status::ok));
    CHECK(!server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status::invalid_config));
    CHECK(server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status::incomplete_evidence));
    CHECK(!server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status::protected_over_capacity));
    CHECK(server_cache_retention_status_is_evidence_unavailable(
        server_cache_retention_policy_status::capacity_unavailable));

    auto invalid               = config();
    invalid.minimum_historical = 0;
    CHECK(server_cache_plan_retention_set(invalid, 1000, {}).status ==
          server_cache_retention_policy_status::invalid_config);

    auto result = server_cache_plan_retention_set(config(4, 4, 1), 1000, {});
    CHECK(result.status == server_cache_retention_policy_status::ok);
    CHECK(result.recent_quota == 3);
    CHECK(result.historical_quota == 1);

    result = server_cache_plan_retention_set(config(4, 4, 2), 1000, {});
    CHECK(result.recent_quota == 2);
    CHECK(result.historical_quota == 2);

    result = server_cache_plan_retention_set(config(1, 4, 2), 1000, {});
    CHECK(result.recent_quota == 1);
    CHECK(result.historical_quota == 0);
}

void test_recent_and_historical_selection() {
    std::vector<server_cache_retention_member> members = {
        member(0, 990),  // recent, newest
        member(1, 950),  // recent
        member(2, 890),  // historical bucket 1, newest
        member(3, 810),  // historical bucket 1, representative (oldest)
        member(4, 650),  // historical bucket 2
        member(5, 250),  // historical bucket 4
    };
    const auto result = server_cache_plan_retention_set(config(4, 2, 2), 1000, members);
    CHECK(result.status == server_cache_retention_policy_status::ok);
    CHECK(result.desired.size() == 4);
    CHECK(result.desired[0] == 0);  // first recent label
    CHECK(result.desired[1] == 3);  // newest occupied historical bucket rep
    CHECK(result.desired[2] == 1);  // second recent label
    CHECK(result.desired[3] == 5);  // oldest occupied historical bucket rep
    CHECK(std::find(result.desired.begin(), result.desired.end(), 2) == result.desired.end());
}

void test_protection_and_evidence() {
    auto protected_member       = member(0, 100);
    protected_member.protection = server_cache_retention_protection::hard_lease;
    auto ordinary               = member(1, 900);

    auto result = server_cache_plan_retention_set(config(1, 1, 1), 1000, { protected_member, ordinary });
    CHECK(result.status == server_cache_retention_policy_status::ok);
    CHECK(result.desired == std::vector<uint32_t>({ 0 }));
    CHECK(result.exclusion_order == std::vector<uint32_t>({ 1 }));

    auto second_protected       = ordinary;
    second_protected.protection = server_cache_retention_protection::recovery_pin;
    result = server_cache_plan_retention_set(config(1, 1, 1), 1000, { protected_member, second_protected });
    CHECK(result.status == server_cache_retention_policy_status::protected_over_capacity);
    CHECK(result.desired.empty());
    CHECK(result.exclusion_order.empty());

    auto zero_id      = ordinary;
    zero_id.stable_id = 0;
    result            = server_cache_plan_retention_set(config(), 1000, { zero_id });
    CHECK(result.status == server_cache_retention_policy_status::incomplete_evidence);

    auto duplicate_id       = ordinary;
    duplicate_id.ordinal    = 2;
    duplicate_id.stable_id  = protected_member.stable_id;
    result = server_cache_plan_retention_set(config(), 1000, { protected_member, duplicate_id });
    CHECK(result.status == server_cache_retention_policy_status::incomplete_evidence);

    auto duplicate_ordinal      = ordinary;
    duplicate_ordinal.stable_id = 99;
    duplicate_ordinal.ordinal   = protected_member.ordinal;
    result = server_cache_plan_retention_set(config(), 1000, { protected_member, duplicate_ordinal });
    CHECK(result.status == server_cache_retention_policy_status::incomplete_evidence);

    auto future     = ordinary;
    future.frontier = 1001;
    result = server_cache_plan_retention_set(config(), 1000, { future });
    CHECK(result.status == server_cache_retention_policy_status::incomplete_evidence);

    auto incoming_a     = member(3, 1000, true);
    auto incoming_b     = member(4, 1000, true);
    result = server_cache_plan_retention_set(config(), 1000, { incoming_a, incoming_b });
    CHECK(result.status == server_cache_retention_policy_status::incomplete_evidence);
}

void test_bucket_boundaries_and_stale_order() {
    // Age 99 is recent. The exact B and B*g boundaries belong to the next
    // older historical bucket. With (R,H)=(1,2), all three representatives
    // must survive in label order.
    std::vector<server_cache_retention_member> boundaries = {
        member(0, 901),  // age 99: recent
        member(1, 900),  // age 100: historical bucket 1
        member(2, 801),  // age 199: bucket 1 representative (oldest)
        member(3, 800),  // age 200: historical bucket 2
    };
    const auto boundary_result =
        server_cache_plan_retention_set(config(3, 1, 1), 1000, boundaries);
    CHECK(boundary_result.status == server_cache_retention_policy_status::ok);
    CHECK(boundary_result.desired == std::vector<uint32_t>({ 0, 2, 3 }));

    auto stale    = member(0, 100);
    stale.identity = server_cache_retention_identity::stale_conclusive_refused;
    auto unknown   = member(1, 200);
    unknown.identity = server_cache_retention_identity::identity_unknown;
    std::vector<server_cache_retention_member> occupants = {
        stale,
        unknown,
        member(2, 990),
        member(3, 700),
    };
    const auto ordered = server_cache_plan_retention_set(config(2, 1, 1), 1000, occupants);
    CHECK(ordered.status == server_cache_retention_policy_status::ok);
    CHECK(ordered.desired == std::vector<uint32_t>({ 2, 3 }));
    CHECK(ordered.exclusion_order.size() == 2);
    CHECK(ordered.exclusion_order[0] == 0);
    CHECK(ordered.exclusion_order[1] == 1);

    // Invalid protected occupants have no lane but remain deterministic and
    // consume capacity. Their class order is stale-refused before unknown.
    stale.protection   = server_cache_retention_protection::hard_lease;
    unknown.protection = server_cache_retention_protection::recovery_pin;
    std::vector<server_cache_retention_member> protected_invalid = { unknown, stale };
    const auto protected_result =
        server_cache_plan_retention_set(config(2, 1, 1), 1000, protected_invalid);
    CHECK(protected_result.status == server_cache_retention_policy_status::ok);
    CHECK(protected_result.desired == std::vector<uint32_t>({ 0, 1 }));
    CHECK(protected_result.exclusion_order.empty());
}

void test_incoming_and_determinism() {
    std::vector<server_cache_retention_member> members = {
        member(0, 300), member(1, 500), member(2, 700), member(3, 900), member(4, 1000, true),
    };
    const auto reference = server_cache_plan_retention_set(config(4, 2, 2), 1000, members);
    CHECK(reference.status == server_cache_retention_policy_status::ok);
    CHECK(reference.incoming_selected);
    CHECK(reference.desired.size() == 4);
    CHECK(reference.exclusion_order.size() == 1);

    std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) { return a.ordinal < b.ordinal; });
    do {
        const auto permuted = server_cache_plan_retention_set(config(4, 2, 2), 1000, members);
        CHECK(permuted.status == reference.status);
        CHECK(permuted.desired == reference.desired);
        CHECK(permuted.exclusion_order == reference.exclusion_order);
        CHECK(permuted.incoming_selected == reference.incoming_selected);
    } while (std::next_permutation(members.begin(), members.end(),
                                   [](const auto & a, const auto & b) { return a.ordinal < b.ordinal; }));
}

void test_exhaustive_protection_preservation() {
    for (uint32_t capacity = 0; capacity <= 8; ++capacity) {
        const uint32_t count   = std::min<uint32_t>(capacity + 1, 8);
        const uint32_t subsets = uint32_t(1) << count;
        for (uint32_t mask = 0; mask < subsets; ++mask) {
            std::vector<server_cache_retention_member> members;
            for (uint32_t i = 0; i < count; ++i) {
                auto next = member(i, 1000 - i * 101);
                if ((mask & (uint32_t(1) << i)) != 0) {
                    next.protection = server_cache_retention_protection::current_task;
                }
                members.push_back(next);
            }
            const uint32_t protected_count = uint32_t(__builtin_popcount(mask));
            const auto     result          = server_cache_plan_retention_set(config(capacity, 4, 2), 1000, members);
            if (protected_count > capacity) {
                CHECK(result.status == server_cache_retention_policy_status::protected_over_capacity);
                continue;
            }
            CHECK(result.status == server_cache_retention_policy_status::ok);
            if (result.desired.size() > capacity) {
                std::fprintf(stderr, "capacity=%u mask=%u protected=%u desired=%zu\n", capacity, mask, protected_count,
                             result.desired.size());
            }
            CHECK(result.desired.size() <= capacity);
            std::set<uint32_t> desired(result.desired.begin(), result.desired.end());
            CHECK(desired.size() == result.desired.size());
            for (uint32_t i = 0; i < count; ++i) {
                if ((mask & (uint32_t(1) << i)) != 0) {
                    CHECK(desired.count(i) == 1);
                    CHECK(std::find(result.exclusion_order.begin(), result.exclusion_order.end(), i) ==
                          result.exclusion_order.end());
                }
            }
        }
    }
}

void test_counterfactual_simulator() {
    std::vector<server_cache_retention_sim_event> events;
    events.push_back({ 0, UINT32_MAX, 100, false });
    for (uint32_t node = 1; node < 10; ++node) {
        events.push_back({ node, node - 1, uint64_t(node + 1) * 100, false });
    }
    events.push_back({ 10, 2, 350, true });
    events.push_back({ 11, 10, 450, false });

    server_cache_retention_sim_config recent_heavy;
    recent_heavy.policy                = config(4, 3, 1);
    recent_heavy.recent_replay_cap     = 200;
    recent_heavy.historical_replay_cap = 800;
    const auto recent_score            = server_cache_simulate_retention(recent_heavy, events);
    CHECK(recent_score.valid);

    auto historical_heavy       = recent_heavy;
    historical_heavy.policy     = config(4, 2, 2);
    const auto historical_score = server_cache_simulate_retention(historical_heavy, events);
    CHECK(historical_score.valid);
    CHECK(historical_score.total_replay_tokens <= recent_score.total_replay_tokens);
    CHECK(historical_score.zero_coverage_deep_rewinds <= recent_score.zero_coverage_deep_rewinds);

    const auto fifo_score = server_cache_simulate_fifo(4, events);
    CHECK(fifo_score.valid);
    CHECK(fifo_score.replay_samples.size() == events.size());
    CHECK(fifo_score.total_replay_tokens >= historical_score.total_replay_tokens);

    auto invalid_events                = events;
    invalid_events.front().parent_node = 999;
    CHECK(!server_cache_simulate_retention(historical_heavy, invalid_events).valid);
}

void test_tiny_ring_and_over_capacity_totality() {
    server_cache_retention_sim_config one;
    one.policy                = config(1, 4, 2);
    one.recent_replay_cap     = 100;
    one.historical_replay_cap = 400;
    const std::vector<server_cache_retention_sim_event> events = {
        { 0, UINT32_MAX, 100, false },
        { 1, 0, 200, false },
        { 2, 1, 300, false },
    };
    const auto frozen = server_cache_simulate_retention(one, events);
    CHECK(frozen.valid);
    CHECK(frozen.publication_skips == 2);
    CHECK(frozen.checkpoint_mutations == 1);
    CHECK(frozen.replay_samples == std::vector<uint64_t>({ 100, 100, 200 }));

    // The pure desired-set function remains total over an over-capacity input.
    // The production adapter, not this kernel, owns the normative
    // shrink-before-publication terminal.
    std::vector<server_cache_retention_member> over = {
        member(0, 400), member(1, 600), member(2, 800), member(3, 1000, true),
    };
    const auto result = server_cache_plan_retention_set(config(2, 1, 1), 1000, over);
    CHECK(result.status == server_cache_retention_policy_status::ok);
    CHECK(result.desired.size() == 2);
    CHECK(result.exclusion_order.size() == 2);
    CHECK(result.incoming_selected);
}

void test_host_retention_ordering() {
    auto oldest = host_member(0, 10);
    auto newer  = host_member(1, 20);
    auto result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { newer, oldest });
    CHECK(result.status == server_cache_host_retention_status::selected);
    CHECK(result.ordinal == oldest.ordinal);
    CHECK(result.projected_pressure_steps == 1);

    // Soft lease and main-family flags form priority strata, not vetoes.
    oldest.soft_leased = true;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { oldest, newer });
    CHECK(result.ordinal == newer.ordinal);
    oldest.soft_leased = false;
    oldest.main_family = true;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { oldest, newer });
    CHECK(result.ordinal == newer.ordinal);

    // Exact redundancy is preferred before recency. The server adapter still
    // has to certify the exact release before it mutates storage.
    oldest.main_family      = false;
    newer.exactly_redundant = true;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { oldest, newer });
    CHECK(result.ordinal == newer.ordinal);
}

void test_host_retention_pressure_resource_and_ties() {
    auto byte_progress = host_member(0, 10, 1000, 1);
    auto token_progress = host_member(1, 10, 100, 100);

    auto result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 500, { token_progress, byte_progress });
    CHECK(result.ordinal == byte_progress.ordinal);
    CHECK(result.projected_pressure_steps == 1);

    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::tokens, 50, { token_progress, byte_progress });
    CHECK(result.ordinal == token_progress.ordinal);
    CHECK(result.projected_pressure_steps == 1);

    // If all earlier keys tie, larger release wins, then stable identity.
    auto small = host_member(2, 30, 100, 10);
    auto large = host_member(3, 30, 200, 20);
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { small, large });
    CHECK(result.ordinal == large.ordinal);

    small.release_bytes = large.release_bytes;
    small.release_tokens = large.release_tokens;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 50, { large, small });
    CHECK(result.ordinal == small.ordinal);
}

void test_host_retention_exclusions_and_evidence() {
    auto ineligible = host_member(0, 10);
    auto incoming   = host_member(1, 20);
    auto zero       = host_member(2, 30, 0, 0);
    ineligible.eligible = false;
    incoming.incoming   = true;

    auto result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 1, { ineligible, incoming, zero });
    CHECK(result.status == server_cache_host_retention_status::no_eligible_progress);
    CHECK(result.ordinal == UINT32_MAX);

    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 0, { host_member(3, 40) });
    CHECK(result.status == server_cache_host_retention_status::no_eligible_progress);

    auto invalid = host_member(4, 0);
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 1, { invalid });
    CHECK(result.status == server_cache_host_retention_status::incomplete_evidence);

    auto first  = host_member(5, 50);
    auto second = host_member(6, 60);
    second.ordinal = first.ordinal;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 1, { first, second });
    CHECK(result.status == server_cache_host_retention_status::incomplete_evidence);

    second           = host_member(6, 60);
    second.stable_id = first.stable_id;
    result = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 1, { first, second });
    CHECK(result.status == server_cache_host_retention_status::incomplete_evidence);
}

void test_host_retention_permutation_determinism() {
    std::vector<server_cache_host_retention_member> members = {
        host_member(0, 40, 100, 10),
        host_member(1, 30, 200, 20),
        host_member(2, 20, 300, 30),
        host_member(3, 10, 400, 40),
    };
    members[0].soft_leased       = true;
    members[1].main_family       = true;
    members[2].exactly_redundant = true;
    const auto reference = server_cache_plan_host_retention_victim(
        server_cache_host_pressure_resource::bytes, 500, members);
    CHECK(reference.status == server_cache_host_retention_status::selected);

    std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) {
        return a.ordinal < b.ordinal;
    });
    do {
        const auto result = server_cache_plan_host_retention_victim(
            server_cache_host_pressure_resource::bytes, 500, members);
        CHECK(result.status == reference.status);
        CHECK(result.ordinal == reference.ordinal);
        CHECK(result.projected_pressure_steps == reference.projected_pressure_steps);
    } while (std::next_permutation(members.begin(), members.end(), [](const auto & a, const auto & b) {
        return a.ordinal < b.ordinal;
    }));
}

void test_lane_replay_cap_boundaries() {
    server_cache_retention_sim_config cfg;
    cfg.policy = config(4, 3, 1);
    cfg.policy.bucket_growth = 4;
    cfg.recent_replay_cap = 100;
    cfg.historical_replay_cap = 400;
    CHECK(server_cache_retention_replay_cap(cfg, 1000, 950) == 100);
    CHECK(server_cache_retention_replay_cap(cfg, 1000, 900) == 300);
    CHECK(server_cache_retention_replay_cap(cfg, 1000, 600) == 400);
    CHECK(server_cache_retention_replay_cap(cfg, 999, 1000) == 0);
}

void test_frozen_zc1_config() {
    const auto cfg = server_cache_zc1_retention_config(7, 100);
    CHECK(cfg.policy.capacity == 7);
    CHECK(cfg.policy.recent_floor == 3);
    CHECK(cfg.policy.minimum_historical == 1);
    CHECK(cfg.policy.bucket_base == 100);
    CHECK(cfg.policy.bucket_growth == 4);
    CHECK(cfg.recent_replay_cap == 100);
    CHECK(cfg.historical_replay_cap == 400);

    const auto normalized = server_cache_zc1_retention_config(0, 0);
    CHECK(normalized.policy.capacity == 0);
    CHECK(normalized.policy.bucket_base == 1);
    CHECK(normalized.recent_replay_cap == 1);
    CHECK(normalized.historical_replay_cap == 4);

    const auto saturated = server_cache_zc1_retention_config(
        4, UINT64_MAX);
    CHECK(saturated.historical_replay_cap == UINT64_MAX);
}

}  // namespace

int main() {
    test_config_and_lane_quota();
    test_recent_and_historical_selection();
    test_protection_and_evidence();
    test_bucket_boundaries_and_stale_order();
    test_incoming_and_determinism();
    test_exhaustive_protection_preservation();
    test_counterfactual_simulator();
    test_tiny_ring_and_over_capacity_totality();
    test_host_retention_ordering();
    test_host_retention_pressure_resource_and_ties();
    test_host_retention_exclusions_and_evidence();
    test_host_retention_permutation_determinism();
    test_lane_replay_cap_boundaries();
    test_frozen_zc1_config();
    std::puts("ZC1_RETENTION_POLICY_TEST PASS");
    return 0;
}
