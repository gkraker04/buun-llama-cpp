#include "common-retention-sidecar.h"
#include "server-retention-sidecar.h"

#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static std::string to_hex(const std::vector<uint8_t> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static common_chat_msg_spans make_spans() {
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER,      0,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT, 4,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     10,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,14,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     20,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,24,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     30,  4);
    spans.add(COMMON_CHAT_ROLE_ASSISTANT,34,  6);
    spans.add(COMMON_CHAT_ROLE_USER,     40,  4);
    return spans;
}

static void test_turn_table_and_geometry() {
    common_retention_turn_table turns;
    CHECK(common_retention_build_turn_table(make_spans(), true, 44, turns));
    CHECK(turns.valid());
    CHECK(turns.boundaries.size() == 5);
    CHECK(turns.boundaries.front().token_pos == 0);
    CHECK(turns.boundaries.front().token_end == 4);

    common_retention_stamp stamp;
    stamp.stable_id = 1;
    stamp.recency_ordinal = 1;
    stamp.coverage_tokens = 30;
    CHECK(common_retention_score(turns, 30, stamp));
    CHECK(stamp.state == common_retention_score_state::known);
    CHECK(stamp.mapped_turn_ordinal == 3);
    CHECK(stamp.anchor_rank == 3);
    CHECK(!stamp.mandatory_anchor);

    stamp.coverage_tokens = 40;
    CHECK(common_retention_score(turns, 40, stamp));
    CHECK(stamp.mapped_turn_ordinal == 4);
    CHECK(stamp.mandatory_anchor);

    common_retention_turn_table unavailable;
    common_chat_msg_spans missing;
    CHECK(common_retention_build_turn_table(missing, false, 44, unavailable));
    CHECK(unavailable.valid());
    CHECK(unavailable.source == common_retention_source_state::unavailable);

    auto malformed = make_spans();
    malformed.spans[2].pos = 3;
    CHECK(!common_retention_build_turn_table(malformed, true, 44, unavailable));
    malformed = make_spans();
    malformed.spans[2].role = COMMON_CHAT_ROLE_UNKNOWN;
    CHECK(!common_retention_build_turn_table(malformed, true, 44, unavailable));

    // Degenerate geometry is closed and deterministic: head-only and newest are
    // mandatory, while n=2 has exactly one optional geometric anchor.
    for (size_t n_user : { size_t(1), size_t(2), size_t(3) }) {
        common_chat_msg_spans small;
        for (size_t i = 0; i < n_user; ++i) {
            small.add(COMMON_CHAT_ROLE_USER, i*10, 2);
        }
        common_retention_turn_table table;
        CHECK(common_retention_build_turn_table(
            small, true, n_user*10, table));
        common_retention_stamp cur;
        cur.stable_id = 1;
        cur.recency_ordinal = 1;
        cur.coverage_tokens = n_user > 1 ? (n_user - 1)*10 : 0;
        CHECK(common_retention_score(table, cur.coverage_tokens, cur));
        CHECK(cur.mandatory_anchor);
        if (n_user == 3) {
            cur.coverage_tokens = 10;
            CHECK(common_retention_score(table, 10, cur));
            CHECK(!cur.mandatory_anchor);
            CHECK(cur.anchor_rank == 2);
        }
    }
}

static common_retention_sidecar_snapshot make_snapshot() {
    common_retention_sidecar_snapshot snapshot;
    common_retention_artifact_record record;
    record.kind = common_retention_artifact_kind::checkpoint;
    CHECK(common_retention_build_turn_table(make_spans(), true, 44, record.turns));
    record.stamp.pool = common_retention_pool::recurrent;
    record.stamp.stable_id = 7;
    record.stamp.recency_ordinal = 9;
    record.stamp.coverage_tokens = 30;
    CHECK(common_retention_score(record.turns, 30, record.stamp));
    snapshot.stable_high_water[1] = 7;
    snapshot.recency_high_water[1] = 9;
    snapshot.artifacts.push_back(record);
    return snapshot;
}

static void test_codec() {
    const auto snapshot = make_snapshot();
    CHECK(snapshot.valid());

    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    CHECK(common_retention_sidecar_encode(snapshot, a));
    CHECK(common_retention_sidecar_encode(snapshot, b));
    CHECK(a == b);
    uint64_t arithmetic_size = 0;
    CHECK(common_retention_sidecar_artifact_encoded_size(
        snapshot.artifacts.front(), arithmetic_size));
    CHECK(arithmetic_size == a.size());

    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(a.data(), a.size(), decoded));
    CHECK(decoded.valid());
    CHECK(decoded.artifacts.size() == 1);
    CHECK(decoded.artifacts[0].stamp.stable_id == 7);
    CHECK(decoded.artifacts[0].turns.boundaries.size() == 5);

    auto corrupt = a;
    corrupt.back() ^= 1;
    CHECK(!common_retention_sidecar_decode(corrupt.data(), corrupt.size(), decoded));
    CHECK(decoded.version == 0 && decoded.artifacts.empty());
    auto bad_version = a;
    bad_version[4] = 2;
    CHECK(!common_retention_sidecar_decode(
        bad_version.data(), bad_version.size(), decoded));
    auto bad_length = a;
    bad_length[8] ^= 1;
    CHECK(!common_retention_sidecar_decode(
        bad_length.data(), bad_length.size(), decoded));
    CHECK(!common_retention_sidecar_decode(a.data(), a.size() - 1, decoded));
    auto trailing = a;
    trailing.push_back(0);
    CHECK(!common_retention_sidecar_decode(
        trailing.data(), trailing.size(), decoded));
    auto too_many = snapshot;
    too_many.artifacts.resize(8193, snapshot.artifacts.front());
    CHECK(!common_retention_sidecar_encode(too_many, corrupt));

    // Golden locks the complete canonical envelope, not just a decoded field.
    static const char * golden =
        "52335344010000000a010000000000003d1c785a8295c1c0b105fec6b695df26"
        "6e7dd9e3946f758acf3441dc210672c800000000000000000900000000000000"
        "0000000000000000070000000000000001000000020000010000010000002c00"
        "0000000000000700000000000000090000000000000003000000000000000300"
        "0000000000001e00000000000000050000000000000000000000000000000000"
        "0000040000000000000001000000000000000a000000000000000e0000000000"
        "0000020000000000000014000000000000001800000000000000030000000000"
        "00001e0000000000000022000000000000000400000000000000280000000000"
        "00002c00000000000000";
    CHECK(to_hex(a) == golden);
}

static void test_store_and_allocator_import() {
    common_retention_allocator source;
    common_retention_stamp first;
    CHECK(source.issue(common_retention_pool::recurrent, first));
    CHECK(first.stable_id == 1 && first.recency_ordinal == 1);

    auto exported = make_snapshot();
    exported.stable_high_water[1] = 17;
    exported.recency_high_water[1] = 29;
    common_retention_allocator resumed;
    common_retention_allocator resumed_again;
    CHECK(resumed.import_snapshot(exported));
    CHECK(resumed_again.import_snapshot(exported));
    common_retention_stamp next;
    common_retention_stamp next_again;
    CHECK(resumed.issue(common_retention_pool::recurrent, next));
    CHECK(resumed_again.issue(common_retention_pool::recurrent, next_again));
    CHECK(next.stable_id == 18);
    CHECK(next.recency_ordinal == 30);
    CHECK(next.stable_id == next_again.stable_id);
    CHECK(next.recency_ordinal == next_again.recency_ordinal);
    CHECK(resumed.stable_high_water(common_retention_pool::recurrent) == 18);
    CHECK(resumed.recency_high_water(common_retention_pool::recurrent) == 30);
    common_retention_stamp attention;
    CHECK(resumed.issue(common_retention_pool::attention, attention));
    CHECK(attention.stable_id == 1 && attention.recency_ordinal == 1);
}

static void test_observer_store_accounting() {
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    llama_cache_acct_ledger ledger;
    const llama_cache_acct_completeness_requirement requirement = {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated }) {
        ledger.gauge_set(
            llama_cache_acct_category::artifact_descriptor_metadata,
            domain, measure, 0);
    }
    CHECK(ledger.certify_complete(
        domain, llama_cache_acct_producer::retention_sidecar));

    server_retention_sidecar_store store;
    store.configure(&ledger, domain);
    const auto live = server_retention_instance_key::for_slot(3);
    CHECK(store.publish(
        live, common_retention_pool::recurrent, make_spans(), true,
        44, 30, true));
    CHECK(store.artifact_id(live).v != 0);
    CHECK(store.live_bytes() > 0);
    CHECK(store.unavailable() == 0);
    const auto before_clone = store.live_bytes();

    const auto * checkpoint_ptr =
        reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(99));
    const auto host_checkpoint =
        server_retention_instance_key::for_checkpoint(-1, checkpoint_ptr);
    CHECK(store.clone(live, host_checkpoint));
    CHECK(store.live_bytes() > before_clone);
    const auto host_id = store.artifact_id(host_checkpoint);
    const auto rebound =
        server_retention_instance_key::for_checkpoint(7, checkpoint_ptr);
    CHECK(store.rebind(host_checkpoint, rebound));
    CHECK(store.artifact_id(rebound) == host_id);
    CHECK(store.artifact_id(host_checkpoint).v == 0);

    std::vector<uint8_t> exported;
    CHECK(store.export_bytes(exported));
    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(
        exported.data(), exported.size(), decoded));
    CHECK(decoded.artifacts.size() == 2);

    store.retire(rebound);
    store.retire(live);
    CHECK(store.live_bytes() == 0);
    CHECK(ledger.snapshot().allocations.empty());
    CHECK(store.publish_ok() == 2);
    CHECK(store.unavailable() == 0);
}

int main() {
    test_turn_table_and_geometry();
    test_codec();
    test_store_and_allocator_import();
    test_observer_store_accounting();
    if (failures != 0) {
        fprintf(stderr, "%d retention-sidecar test(s) failed\n", failures);
        return 1;
    }
    printf("retention sidecar: PASS\n");
    return 0;
}
