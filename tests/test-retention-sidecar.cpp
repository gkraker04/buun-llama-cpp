#include "common-retention-sidecar.h"
#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"
#include "server-retention-sidecar.h"
#include "llama-cache-authority.h"

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

class retention_test_clock final : public server_cache_lease_clock {
public:
    uint64_t now_ns() noexcept override {
        return now++;
    }

private:
    uint64_t now = 1;
};

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

static void test_checkpoint_desired_set_anchor_policy() {
    retention_test_clock clock;
    server_cache_lease_table leases(&clock);
    server_retention_sidecar_store store;
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    store.configure(nullptr, domain, &leases);

    const server_cache_lease_identity lineage_a = {
        "execution-a", "adapter", "media",
    };
    const server_cache_lease_identity lineage_b = {
        "execution-b", "adapter", "media",
    };
    const auto publish_checkpoint = [&](uintptr_t address, int32_t owner,
                                        const server_cache_lease_identity & identity,
                                        server_retention_anchor_policy anchor_policy) {
        common_retention_stamp stamp;
        CHECK(store.reserve_stamp(common_retention_pool::recurrent, stamp));
        const auto key = server_retention_instance_key::for_checkpoint(
            owner, reinterpret_cast<const common_prompt_checkpoint *>(address));
        CHECK(store.publish_reserved(
            key, stamp, make_spans(), true, 44, 40, true,
            &identity, nullptr, anchor_policy));
        return key;
    };

    const auto first = publish_checkpoint(
        501, 5, lineage_a,
        server_retention_anchor_policy::checkpoint_desired_set);
    const auto duplicate = publish_checkpoint(
        502, 5, lineage_a,
        server_retention_anchor_policy::checkpoint_desired_set);
    const auto other_lineage = publish_checkpoint(
        503, 5, lineage_b,
        server_retention_anchor_policy::checkpoint_desired_set);
    server_retention_checkpoint_inventory inventory;
    CHECK(store.checkpoint_inventory(first, inventory));
    CHECK(!inventory.mandatory_anchor);
    CHECK(store.checkpoint_inventory(duplicate, inventory));
    CHECK(!inventory.mandatory_anchor);
    CHECK(store.checkpoint_inventory(other_lineage, inventory));
    CHECK(!inventory.mandatory_anchor);

    // The landed/non-ZC publication door does not coalesce anchors.
    const auto legacy_first = publish_checkpoint(
        601, 6, lineage_a, server_retention_anchor_policy::scored);
    const auto legacy_second = publish_checkpoint(
        602, 6, lineage_a, server_retention_anchor_policy::scored);
    CHECK(store.checkpoint_inventory(legacy_first, inventory));
    CHECK(inventory.mandatory_anchor);
    CHECK(store.checkpoint_inventory(legacy_second, inventory));
    CHECK(inventory.mandatory_anchor);

    // Desired-set ownership is checkpoint-specific. A future caller cannot
    // accidentally suppress score-derived anchors on another artifact kind.
    common_retention_stamp invalid_stamp;
    CHECK(store.reserve_stamp(common_retention_pool::recurrent, invalid_stamp));
    const server_retention_instance_key host_key {
        common_retention_artifact_kind::host_entry, -1, 701,
    };
    CHECK(!store.publish_reserved(
        host_key, invalid_stamp, make_spans(), true, 44, 40, true,
        nullptr, nullptr,
        server_retention_anchor_policy::checkpoint_desired_set));
    server_retention_candidate ignored;
    CHECK(!store.candidate_for_instance(host_key, ignored));
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
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::artifact_descriptor_metadata,
            domain, measure, 0);
        ledger.gauge_set(
            llama_cache_acct_category::checkpoint_state_payload,
            domain, measure, 0);
    }
    CHECK(ledger.certify_complete(
        domain, llama_cache_acct_producer::retention_sidecar));

    retention_test_clock clock;
    server_cache_lease_table leases(&clock);
    server_retention_sidecar_store store;
    store.configure(&ledger, domain, &leases);
    const server_cache_lease_identity lease_identity = {
        "execution", "adapter", "media",
    };
    const auto live = server_retention_instance_key::for_slot(3);
    CHECK(store.publish(
        live, common_retention_pool::recurrent, make_spans(), true,
        44, 30, true, &lease_identity));
    CHECK(store.artifact_id(live).v != 0);
    const auto live_artifact = store.artifact_id(live);
    const auto live_payload_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, live_artifact },
        32, 32);
    CHECK(live_payload_op);
    CHECK(store.attach_release_ops(live, { live_payload_op }));
    const auto lease = leases.grant_soft(
        {
            live_artifact,
            common_retention_artifact_kind::live_slot,
            3,
        },
        server_cache_lease_scope::from(
            server_cache_context_scope_id { 1 }),
        lease_identity,
        100);
    CHECK(lease);
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
    server_retention_candidate aggregate_clone;
    CHECK(store.candidate_for_instance(rebound, aggregate_clone));
    CHECK(aggregate_clone.release_ops.empty());
    const auto payload_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, host_id },
        64, 64);
    CHECK(payload_op);
    CHECK(store.attach_release_ops(rebound, { payload_op }));
    server_retention_checkpoint_inventory inventory;
    CHECK(store.checkpoint_inventory(rebound, inventory));
    CHECK(inventory.artifact_id == host_id);
    CHECK(inventory.identity_known);
    CHECK(inventory.release_owned);
    CHECK(!inventory.recovery_pinned);
    {
        auto pin = store.acquire_recovery_pin(rebound);
        CHECK(pin.valid());
        CHECK(pin.binds_exact(host_id, { payload_op }));
        CHECK(store.checkpoint_inventory(rebound, inventory));
        CHECK(inventory.recovery_pinned);
    }
    CHECK(store.checkpoint_inventory(rebound, inventory));
    CHECK(!inventory.recovery_pinned);

    std::vector<uint8_t> exported;
    CHECK(store.export_bytes(exported));
    common_retention_sidecar_snapshot decoded;
    CHECK(common_retention_sidecar_decode(
        exported.data(), exported.size(), decoded));
    CHECK(decoded.artifacts.size() == 2);

    const auto candidates = store.candidate_snapshot();
    CHECK(candidates.size() == 2);
    for (const auto & candidate : candidates) {
        CHECK(candidate.artifact_id.v != 0);
        CHECK(candidate.record.valid());
        CHECK(candidate.provenance_op);
        if (candidate.artifact_id == live_artifact) {
            CHECK(candidate.release_ops ==
                  std::vector<llama_cache_acct_op_id> { live_payload_op });
        } else {
            CHECK(candidate.release_ops ==
                  std::vector<llama_cache_acct_op_id> { payload_op });
        }
        CHECK(candidate.avail ==
              server_retention_candidate_availability::available);
        CHECK(store.artifact_id(candidate.instance_key) ==
              candidate.artifact_id);
    }

    server_retention_candidate victim_candidate;
    CHECK(store.candidate_for_instance(rebound, victim_candidate));
    server_cache_destruction_artifact victim;
    victim.candidate.artifact_id = victim_candidate.artifact_id;
    victim.candidate.record = victim_candidate.record;
    victim.candidate.availability = victim_candidate.avail;
    victim.candidate.release_ops = victim_candidate.release_ops;
    victim.candidate.identity_known = true;
    victim.candidate.lease = {
        server_cache_lease_eval_state::known,
        server_cache_lease_class::none,
        server_cache_lease_eligibility::eligible,
    };
    victim.kind = common_retention_artifact_kind::checkpoint;
    victim.pool = victim_candidate.record.stamp.pool;
    const auto preview = [&](const auto & cited, uint64_t serial, auto & out) {
        return ledger.preview_release_set(cited, serial, out);
    };
    const auto project = [](const auto & released, auto & out) {
        out.clear();
        for (const auto & row : released.rows) {
            common_cache_plan_yield_domain domain_row;
            domain_row.domain = row.domain;
            domain_row.current_resident_bytes =
                llama_cache_acct_value::measured(row.resident_allocated);
            domain_row.fit_before_bytes =
                domain_row.current_resident_bytes;
            domain_row.projected_release_bytes =
                llama_cache_acct_value::measured(row.resident_allocated);
            domain_row.projected_reserve_bytes =
                llama_cache_acct_value::measured(0);
            domain_row.projected_after_bytes =
                llama_cache_acct_value::measured(0);
            out.push_back(domain_row);
        }
        return !out.empty();
    };
    auto quote = server_cache_destruction_quote_single_artifact(
        victim, 0, ledger.snapshot().serial, 1, preview, project);
    CHECK(quote.receipt.state ==
          common_cache_plan_destruction_state::quoted);
    auto same_member_pin = store.acquire_recovery_pin(rebound);
    CHECK(same_member_pin.valid());
    auto disjoint_refusal = server_cache_prepare_release_set(
        quote, { victim }, ledger, ledger.snapshot().serial,
        project, std::move(same_member_pin));
    CHECK(disjoint_refusal.status ==
          server_cache_prepare_release_status::recovery_unavailable);
    same_member_pin = {};
    auto recovery_pin = store.acquire_recovery_pin(live);
    CHECK(recovery_pin.valid());
    auto prepared = server_cache_prepare_release_set(
        quote, { victim }, ledger, ledger.snapshot().serial,
        project, std::move(recovery_pin));
    CHECK(prepared.status == server_cache_prepare_release_status::prepared);
    server_cache_recovery_pin retained_pin;
    CHECK(prepared.capability.commit(retained_pin) ==
          common_cache_plan_destruction_reason::none);
    store.retire_after_committed_release(rebound);
    retained_pin = {};

    // A latent legacy drop racing a recovery pin must fail soft, not abort or
    // invalidate the pin callback. Retirement is deferred until the pin
    // closes, at which point both descriptor and payload ops are discharged.
    const auto * pinned_ptr =
        reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(100));
    const auto pinned_key =
        server_retention_instance_key::for_checkpoint(9, pinned_ptr);
    CHECK(store.clone(live, pinned_key));
    const auto pinned_artifact = store.artifact_id(pinned_key);
    const auto pinned_op = server_cache_acct_charge_shadow(
        ledger,
        llama_cache_acct_category::checkpoint_state_payload,
        domain,
        llama_cache_acct_producer::retention_sidecar,
        { llama_cache_acct_attr_kind::artifact, -1, pinned_artifact },
        16, 16);
    CHECK(pinned_op);
    CHECK(store.attach_release_ops(pinned_key, { pinned_op }));
    auto latent_pin = store.acquire_recovery_pin(pinned_key);
    CHECK(latent_pin.valid());
    const auto live_ops_before_pinned_drop = ledger.snapshot().live_ops;
    store.retire(pinned_key);
    CHECK(store.artifact_id(pinned_key).v == 0);
    CHECK(store.unavailable() == 1);
    CHECK(ledger.snapshot().live_ops == live_ops_before_pinned_drop);
    latent_pin = {};
    CHECK(ledger.snapshot().live_ops + 2 == live_ops_before_pinned_drop);

    store.retire(live);
    CHECK(leases.evaluate(live_artifact, lease_identity).cls ==
          server_cache_lease_class::none);
    CHECK(store.live_bytes() == 0);
    CHECK(ledger.snapshot().allocations.empty());
    CHECK(store.publish_ok() == 3);
    CHECK(store.unavailable() == 1);
}

int main() {
    test_turn_table_and_geometry();
    test_codec();
    test_store_and_allocator_import();
    test_checkpoint_desired_set_anchor_policy();
    test_observer_store_accounting();
    if (failures != 0) {
        fprintf(stderr, "%d retention-sidecar test(s) failed\n", failures);
        return 1;
    }
    printf("retention sidecar: PASS\n");
    return 0;
}
