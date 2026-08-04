#include "server-cache-authority.h"
#include "server-cache-destruction-quote.h"
#include "server-cache-plan-authority.h"
#include "server-task.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <list>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(expr) do {                                                        \
    if (!(expr)) {                                                              \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",                   \
                     __FILE__, __LINE__, #expr);                                \
        failures++;                                                             \
    }                                                                           \
} while (0)

void configure_host_accounting(
        server_cache_authority & authority,
        bool with_sidecar = false) {
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const llama_cache_acct_completeness_requirement required[] = {
        { host, llama_cache_acct_producer::host_cache },
        { host, llama_cache_acct_producer::retention_sidecar },
    };
    const size_t n_required = with_sidecar ? std::size(required) : 1;
    CHECK(authority.ledger.configure_required_producers(
        required, n_required));
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload,
            llama_cache_acct_category::artifact_descriptor_metadata }) {
        if (!with_sidecar && category ==
                llama_cache_acct_category::artifact_descriptor_metadata) {
            continue;
        }
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            authority.ledger.gauge_set(category, host, measure, 0);
        }
    }
    CHECK(authority.ledger.certify_complete(
        host, llama_cache_acct_producer::host_cache));
    if (with_sidecar) {
        CHECK(authority.ledger.certify_complete(
            host, llama_cache_acct_producer::retention_sidecar));
        authority.retention.configure(
            &authority.ledger, host, &authority.leases);
    }
}

std::list<server_prompt_cache_state> make_entry(
        const char * identity,
        size_t bytes) {
    std::list<server_prompt_cache_state> entry;
    entry.emplace_back();
    entry.front().adapter_config_key = identity;
    entry.front().data.main.resize(bytes);
    return entry;
}

std::list<server_prompt_cache_state> make_prompt_entry(
        const char * identity,
        std::initializer_list<llama_token> tokens) {
    auto entry = make_entry(identity, 1);
    entry.front().prompt.tokens = server_tokens(
        llama_tokens(tokens), false);
    return entry;
}

std::list<server_prompt_cache_state> make_redundant_entry() {
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    entry.front().data.main.assign(16, 7);
    entry.front().data.drft.assign(4, 8);
    entry.front().prompt.checkpoints.emplace_back();
    auto & checkpoint = entry.front().prompt.checkpoints.back();
    checkpoint.n_tokens = 2;
    checkpoint.pos_min = 0;
    checkpoint.pos_max = 1;
    checkpoint.data_tgt.assign(8, 9);
    checkpoint.data_dft.assign(3, 10);
    checkpoint.accel.ring.assign(5, 11);
    checkpoint.accel.spec.assign(2, 12);
    return entry;
}

constexpr const char * HOST_TRADE_TEST_PROFILE =
    "qwen35-2b-q4-k---medium/nvidia-geforce-rtx-3090-ngl99/b512/kf16-vf16";

class available_host_fallback final : public server_cache_lease_fallback_provider {
public:
    server_cache_lease_fallback_state preflight(
            const server_cache_lease_subject &,
            const server_cache_lease_identity &) noexcept override {
        return server_cache_lease_fallback_state::available;
    }
};

void configure_host_trade(
        server_cache_authority & authority,
        server_prompt_cache & cache,
        const std::string & execution_identity,
        server_cache_lease_table * leases = nullptr) {
    configure_host_accounting(authority, true);
    authority.calibration_profile = HOST_TRADE_TEST_PROFILE;
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;
    cache.lease_obs = leases ? leases : &authority.leases;
    cache.lease_execution_identity = &execution_identity;
}

server_prompt_cache::iterator install_host_trade_entry(
        server_prompt_cache & cache,
        server_cache_authority & authority,
        const char * unique_adapter,
        size_t bytes) {
    static llama_token next_token = 100;
    const llama_token first = next_token;
    next_token += 3;
    auto entry = make_prompt_entry(
        unique_adapter, { first, first + 1, first + 2 });
    entry.front().data.main.assign(bytes, uint8_t(next_token));
    CHECK(cache.publish(std::move(entry)));
    auto installed = std::prev(cache.states.end());
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 1, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 2, 1);
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_host_entry(&*installed),
        common_retention_pool::attention,
        spans,
        true,
        3,
        1,
        true));
    return installed;
}

void make_host_trade_pair(
        server_prompt_cache::iterator victim,
        server_prompt_cache::iterator recovery,
        const char * adapter,
        llama_token token,
        int32_t source_id,
        bool main_family = false) {
    victim->adapter_config_key = adapter;
    recovery->adapter_config_key = adapter;
    victim->prompt.tokens = server_tokens(
        llama_tokens { token, token + 1, token + 2 }, false);
    recovery->prompt.tokens = server_tokens(
        llama_tokens { token, token + 1, token + 2 }, false);
    victim->prompt.sequence_epoch = uint64_t(token);
    recovery->prompt.sequence_epoch = uint64_t(token);
    victim->data.main = recovery->data.main;
    victim->cache_plan_source_id = source_id;
    recovery->cache_plan_source_id = source_id + 100;
    victim->main_family = main_family;
    // Keep the proof source outside the victim candidate set while still
    // allowing D-A's short-lived pin to nest over it.
    recovery->recovery_pins = 1;
    CHECK(server_prompt_cache::exactly_redundant(*victim, *recovery));
}

server_cache_lease_id grant_host_lease(
        server_prompt_cache & cache,
        server_cache_lease_table & leases,
        server_prompt_cache::iterator victim,
        server_cache_lease_class cls) {
    const auto artifact = cache.retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*victim));
    server_cache_lease_identity identity;
    CHECK(server_cache_lease_build_identity(
        *cache.lease_execution_identity,
        victim->adapter_config_key,
        victim->prompt.tokens,
        victim->prompt.n_tokens(),
        identity));
    const server_cache_lease_subject subject {
        artifact,
        common_retention_artifact_kind::host_entry,
        -1,
    };
    const auto scope = server_cache_lease_scope::from(
        leases.new_context_scope());
    return cls == server_cache_lease_class::hard
        ? leases.grant_hard(subject, scope, identity,
              server_cache_lease_table::IMPLICIT_SOFT_TTL_NS)
        : leases.grant_soft(subject, scope, identity,
              server_cache_lease_table::IMPLICIT_SOFT_TTL_NS);
}

bool host_source_present(
        const server_prompt_cache & cache,
        int32_t source_id) {
    return std::any_of(cache.states.begin(), cache.states.end(),
        [&](const auto & state) {
            return state.cache_plan_source_id == source_id;
        });
}

// Regression for F0b review MUST-1: lifecycle accounting may prove/transact publication, but the
// prompt cache's configured limit remains a FIFO rotation policy—not an admission ceiling. A full
// 1 MiB cache must accept a second 700 KiB entry and evict the oldest, rather than become fill-once.
void test_lifecycle_full_cache_rotates() {
    server_cache_authority authority;
    configure_host_accounting(authority);

    server_prompt_cache cache(/* limit_size_mib */ 1, /* limit_tokens */ 1024);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;

    CHECK(cache.publish(make_entry("oldest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "oldest");

    CHECK(cache.publish(make_entry("newest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "newest");
    CHECK(cache.size() == 700 * 1024);
    CHECK(authority.admission_commits == 2);
    CHECK(authority.admission_refusals == 0);
    CHECK(authority.destruction.prepared_release_commits == 1);
    CHECK(authority.destruction.prepared_release_fallbacks == 0);
    CHECK(authority.destruction.n_events == 1);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::prepared_release);
}

void test_lifecycle_restore_retains_immutable_source() {
    server_cache_authority authority;
    configure_host_accounting(authority);

    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;

    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    entry.front().data.main.assign(32, 7);
    entry.front().prompt.checkpoints.emplace_back();
    entry.front().prompt.checkpoints.back().n_tokens = 2;
    entry.front().prompt.checkpoints.back().data_tgt.assign(8, 9);
    CHECK(cache.publish(std::move(entry)));
    CHECK(cache.states.size() == 1);
    const auto live_ops_before = authority.ledger.snapshot().live_ops;
    const auto host_size_before = cache.states.front().size();
    const auto * source_checkpoint =
        &cache.states.front().prompt.checkpoints.front();

    server_prompt_cache_restore_delivery first;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), first));
    CHECK(first.retains_source);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);

    server_prompt live_first;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(first), live_first, 4);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);
    CHECK(live_first.n_tokens() == 3);
    CHECK(live_first.checkpoints.size() == 1);
    CHECK(&live_first.checkpoints.front() != source_checkpoint);
    CHECK(live_first.checkpoints.front().n_tokens == 2);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction.host_restores_retained == 1);
    CHECK(authority.destruction.host_restores_consumed == 0);

    server_prompt_cache_restore_delivery second;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), second));
    server_prompt live_second;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(second), live_second, 5);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == host_size_before);
    CHECK(live_second.n_tokens() == 3);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction.host_restores_retained == 2);

    cache.destroy_entry(
        cache.states.begin(), server_cache_destruction_reason::host_capacity);
    CHECK(cache.states.empty());
    CHECK(authority.ledger.snapshot().live_ops == 0);
    CHECK(authority.destruction.prepared_release_commits == 1);
}

void test_lifecycle_off_restore_consumes() {
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    server_cache_destruction_observer observer;
    cache.destruction_obs = &observer;
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));

    server_prompt_cache_restore_delivery delivery;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
    CHECK(!delivery.retains_source);
    server_prompt live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), live, 0);
    CHECK(cache.states.empty());
    CHECK(live.n_tokens() == 3);
    CHECK(observer.host_restores_retained == 0);
    CHECK(observer.host_restores_consumed == 1);
    CHECK(observer.n_events == 1);
    CHECK(observer.events[0].request.reason ==
          server_cache_destruction_reason::host_consumed_restore);
    CHECK(observer.events[0].execution ==
          server_cache_destruction_execution::pass_through);
}

void test_lifecycle_release_prepare_failure_keeps_legacy_bound() {
    server_cache_authority authority;
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;

    // A pre-authority/unaccounted node has no releasable op union. D-A1 may
    // not certify it, but must retain the legacy explicit-eviction bound.
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    cache.states.splice(cache.states.end(), entry);
    cache.destroy_entry(
        cache.states.begin(), server_cache_destruction_reason::host_capacity);
    CHECK(cache.states.empty());
    CHECK(authority.destruction.prepared_release_commits == 0);
    CHECK(authority.destruction.prepared_release_fallbacks == 1);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::pass_through);
}

void test_lifecycle_restore_clone_fault() {
    server_cache_authority authority;
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.publish_authority = &authority;
    auto entry = make_prompt_entry("same", { 1, 2, 3 });
    cache.states.splice(cache.states.end(), entry);
    const auto source_size = cache.states.front().size();

    // The injected tag exercises the explicit fail-closed seam. Deliberately
    // does not attempt to make the allocator throw std::bad_alloc.
    server_prompt_cache_restore_delivery delivery;
    CHECK(!cache.prepare_restore_delivery(cache.states.begin(), delivery));
    CHECK(!delivery.retains_source);
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().size() == source_size);
    CHECK(cache.states.front().prompt.n_tokens() == 3);
}

void test_lifecycle_authority_without_debug_is_silent() {
    server_cache_authority authority;
    configure_host_accounting(authority);
    server_cache_plan_authority plan_authority(
        common_cache_plan_authority_level::lru);
    CHECK(plan_authority.configured_level ==
          common_cache_plan_authority_level::lru);
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    CHECK(!cache.debug_observability);
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));

    server_prompt_cache_restore_delivery delivery;
    CHECK(cache.prepare_restore_delivery(cache.states.begin(), delivery));
    server_prompt live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(delivery), live, 0, 7);
    CHECK(cache.states.size() == 1);
    CHECK(cache.debug_lifecycle_emissions == 0);

    // Positive control: the same retained restore emits exactly once when the
    // explicit debug view is enabled, proving the zero above is a real gate.
    cache.debug_observability = true;
    server_prompt_cache_restore_delivery debug_delivery;
    CHECK(cache.prepare_restore_delivery(
        cache.states.begin(), debug_delivery));
    server_prompt debug_live;
    cache.commit_restore_delivery(
        cache.states.begin(), std::move(debug_delivery), debug_live, 1, 7);
    CHECK(cache.debug_lifecycle_emissions == 1);
}

void test_authority_source_ids_survive_save_dedup() {
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    CHECK(cache.publish(make_prompt_entry("same", { 1 })));
    CHECK(cache.publish(make_prompt_entry("same", { 9 })));
    CHECK(cache.states.size() == 2);

    cache.cache_plan_begin_inventory();
    auto old = cache.states.begin();
    auto survivor = std::next(old);
    int32_t old_source = -1;
    int32_t survivor_source = -1;
    CHECK(cache.cache_plan_get_source_id(*old, old_source));
    CHECK(cache.cache_plan_get_source_id(*survivor, survivor_source));
    CHECK(old_source == 0);
    CHECK(survivor_source == 1);

    // Publishing the larger {1,2} prompt removes {1}. The surviving {9}
    // node keeps source 1, while the new node gets 2 even if the allocator
    // recycles the erased node's address.
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2 })));
    CHECK(cache.states.size() == 2);
    CHECK(cache.cache_plan_get_source_id(
        cache.states.front(), old_source));
    CHECK(old_source == survivor_source);
    CHECK(cache.cache_plan_get_source_id(
        cache.states.back(), old_source));
    CHECK(old_source == 2);
}

void test_exact_redundant_host_eviction() {
    server_cache_authority authority;
    const std::string execution_identity = "test-execution";
    configure_host_accounting(authority, true);

    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    cache.retention_obs = &authority.retention;
    cache.lease_obs = &authority.leases;
    cache.lease_execution_identity = &execution_identity;

    server_prompt source;
    source.tokens = server_tokens(llama_tokens { 1, 2, 3 }, false);
    common_chat_msg_spans spans;
    spans.add(COMMON_CHAT_ROLE_USER, 0, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 1, 1);
    spans.add(COMMON_CHAT_ROLE_USER, 2, 1);
    CHECK(authority.retention.publish(
        server_retention_instance_key::for_slot(0),
        common_retention_pool::attention,
        spans,
        true,
        3,
        1,
        true));

    auto first = make_redundant_entry();
    CHECK(cache.publish(std::move(first), &source, 0));
    CHECK(cache.states.size() == 1);
    const auto live_ops_before = authority.ledger.snapshot().live_ops;

    auto duplicate = make_redundant_entry();
    CHECK(server_prompt_cache::exactly_redundant(
        cache.states.front(), duplicate.front()));
    CHECK(cache.publish(std::move(duplicate), &source, 0));

    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().recovery_pins == 0);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction.redundant_host_certified == 1);
    CHECK(authority.destruction.redundant_host_executed == 1);
    CHECK(authority.destruction.redundant_host_refused == 0);
    CHECK(authority.destruction.redundant_host_release_bytes == 38);
    CHECK(authority.destruction.events[0].execution ==
          server_cache_destruction_execution::redundant_host_eviction);
    CHECK(authority.destruction_counters.has_receipt);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::executed);
    CHECK(authority.destruction_counters.last_receipt.displaced_fate ==
          common_cache_plan_displaced_fate::exact_duplicate);
    CHECK(authority.destruction_counters.last_receipt.recovery_citation ==
          common_cache_plan_recovery_citation::resolved);
    const auto & recovery_receipt =
        authority.destruction_counters.last_receipt;
    CHECK(recovery_receipt.recovery_source_artifact_id.v != 0);
    CHECK(recovery_receipt.recovery_source_artifact_id.v !=
          recovery_receipt.selected_attention.front().v);
    CHECK(recovery_receipt.recovery_source_manifest_digest.valid());
    const auto survivor_ops = cache.states.front().release_ops();
    const std::vector<llama_cache_acct_op_id> survivor_op_vector(
        survivor_ops.begin(), survivor_ops.end());
    CHECK(recovery_receipt.recovery_source_manifest_digest ==
          server_cache_destruction_recovery_source_digest(
              recovery_receipt.recovery_source_artifact_id,
              survivor_op_vector));
    CHECK(authority.destruction_counters.quoted
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.certified
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.executed
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_class::host_artifact_drop)] == 1);
    CHECK(authority.destruction_counters.lease_verdict
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_lease_verdict::unleased)] == 1);
    CHECK(authority.destruction_counters.recovery_outcome
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_displaced_fate::exact_duplicate)] == 1);
    // Lifecycle + authority without --cache-debug must not emit maintenance
    // evidence, even though the certified execution and process counters run.
    CHECK(cache.debug_destruction_emissions == 0);

    // Positive control for the same seam: explicit debug emits quoted,
    // certified, and executed receipts exactly once each.
    cache.debug_observability = true;
    auto debug_duplicate = make_redundant_entry();
    CHECK(cache.publish(std::move(debug_duplicate), &source, 0));
    CHECK(cache.debug_destruction_emissions == 3);
}

void test_redundancy_payload_mismatch_and_missing_catalog() {
    auto victim = make_prompt_entry("same", { 1, 2, 3 });
    victim.front().data.main.assign(4, 1);
    victim.front().prompt.checkpoints.emplace_back();
    victim.front().prompt.checkpoints.back().n_tokens = 2;
    victim.front().prompt.checkpoints.back().data_tgt.assign(2, 3);
    auto survivor = make_prompt_entry("same", { 1, 2, 3 });
    survivor.front().data.main.assign(4, 1);
    survivor.front().prompt.checkpoints.emplace_back();
    survivor.front().prompt.checkpoints.back().n_tokens = 2;
    survivor.front().prompt.checkpoints.back().data_tgt.assign(2, 3);
    survivor.front().prompt.tokens = server_tokens(
        llama_tokens { 1, 2, 3, 4 }, false);
    // Coverage superset is accepted only because all three physical payload
    // planes are still byte-identical.
    CHECK(server_prompt_cache::exactly_redundant(
        victim.front(), survivor.front()));
    survivor.front().prompt.checkpoints.back().data_tgt[1] = 4;
    CHECK(!server_prompt_cache::exactly_redundant(
        victim.front(), survivor.front()));

    server_cache_authority authority;
    configure_host_accounting(authority);
    server_prompt_cache cache(/* limit_size_mib */ 0, /* limit_tokens */ 0);
    cache.acct = &authority.ledger;
    cache.publish_authority = &authority;
    cache.destruction_obs = &authority.destruction;
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));
    CHECK(cache.publish(make_prompt_entry("same", { 1, 2, 3 })));
    CHECK(cache.states.size() == 1);
    CHECK(authority.destruction.redundant_host_executed == 0);
    CHECK(authority.destruction.redundant_host_refused == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::manifest_incomplete);
    CHECK(authority.destruction.prepared_release_commits == 1);
}

void test_host_trade_soft_lease_weight_flips_victim() {
    server_cache_authority authority;
    const std::string execution = "trade-soft";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto a = install_host_trade_entry(cache, authority, "a-v", 64);
    auto ar = install_host_trade_entry(cache, authority, "a-r", 64);
    auto b = install_host_trade_entry(cache, authority, "b-v", 64);
    auto br = install_host_trade_entry(cache, authority, "b-r", 64);
    make_host_trade_pair(a, ar, "pair-a", 10, 10);
    make_host_trade_pair(b, br, "pair-b", 20, 20);
    CHECK(grant_host_lease(
        cache, authority.leases, a, server_cache_lease_class::soft));

    cache.limit_size = cache.size() - b->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 10));
    CHECK(!host_source_present(cache, 20));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_executed == 1);
    CHECK(authority.destruction.host_trade_soft_lease_evictions == 0);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::executed);
    CHECK(authority.destruction_counters.last_receipt.lease_verdict ==
          common_cache_plan_destruction_lease_verdict::unleased);

    // Soft protection is a price, never a veto: once it is the only
    // certifiable victim, the same lease must still permit eviction.
    cache.limit_size = cache.size() - a->size() + 1;
    cache.update();
    CHECK(!host_source_present(cache, 10));
    CHECK(authority.destruction.host_trade_executed == 2);
    CHECK(authority.destruction.host_trade_soft_lease_evictions == 1);
    CHECK(cache.debug_destruction_emissions == 0);
}

void test_host_trade_main_family_weight_flips_victim() {
    server_cache_authority authority;
    const std::string execution = "trade-main";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto main = install_host_trade_entry(cache, authority, "m-v", 64);
    auto main_r = install_host_trade_entry(cache, authority, "m-r", 64);
    auto child = install_host_trade_entry(cache, authority, "c-v", 64);
    auto child_r = install_host_trade_entry(cache, authority, "c-r", 64);
    make_host_trade_pair(main, main_r, "pair-main", 30, 30, true);
    make_host_trade_pair(child, child_r, "pair-child", 40, 40, false);

    cache.limit_size = cache.size() - child->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 30));
    CHECK(!host_source_present(cache, 40));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_main_family_evictions == 0);

    // The automatic family signal is likewise a finite pricing weight.
    cache.limit_size = cache.size() - main->size() + 1;
    cache.update();
    CHECK(!host_source_present(cache, 30));
    CHECK(authority.destruction.host_trade_executed == 2);
    CHECK(authority.destruction.host_trade_main_family_evictions == 1);
}

void test_host_trade_zero_destruction_tie_break() {
    server_cache_authority authority;
    const std::string execution = "trade-tie";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);
    cache.debug_observability = true;

    auto destructive = install_host_trade_entry(cache, authority, "d-v", 64);
    destructive->cache_plan_source_id = 1;
    auto duplicate = install_host_trade_entry(cache, authority, "z-v", 64);
    auto duplicate_r = install_host_trade_entry(cache, authority, "z-r", 64);
    make_host_trade_pair(
        duplicate, duplicate_r, "pair-zero", 50, 2, false);

    cache.limit_size = cache.size() - duplicate->size() + 1;
    cache.update();
    CHECK(host_source_present(cache, 1));
    CHECK(!host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 1);
    CHECK(authority.destruction.host_trade_refused == 0);
    CHECK(authority.destruction.host_trade_zero_destruction_ties == 1);
    CHECK(cache.debug_destruction_emissions == 3);
}

void test_host_trade_all_refuse_falls_back_to_legacy() {
    server_cache_authority authority;
    const std::string execution = "trade-fallback";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto oldest = install_host_trade_entry(cache, authority, "old", 64);
    auto newer = install_host_trade_entry(cache, authority, "new", 64);
    oldest->cache_plan_source_id = 1;
    newer->cache_plan_source_id = 2;
    cache.limit_tokens = 3;
    cache.update();
    CHECK(!host_source_present(cache, 1));
    CHECK(host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 2);
    CHECK(authority.destruction.host_trade_refused == 2);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
    CHECK(authority.destruction.prepared_release_commits == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::recovery_unavailable);
    const auto * event = authority.destruction.event_for_sequence(
        authority.destruction.n_events);
    CHECK(event != nullptr);
    CHECK(event->execution !=
          server_cache_destruction_execution::priced_host_eviction);
    CHECK(cache.debug_destruction_emissions == 0);
}

void test_host_trade_hard_lease_veto() {
    server_cache_authority authority;
    available_host_fallback fallback;
    server_cache_lease_table hard_leases(nullptr, &fallback);
    const std::string execution = "trade-hard";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution, &hard_leases);

    auto hard = install_host_trade_entry(cache, authority, "h-v", 64);
    auto open = install_host_trade_entry(cache, authority, "o-v", 64);
    hard->cache_plan_source_id = 1;
    open->cache_plan_source_id = 2;
    CHECK(grant_host_lease(
        cache, hard_leases, hard, server_cache_lease_class::hard));

    // Neither victim has durable recovery evidence, so the ranked ladder
    // refuses. The legacy floor must still honor the hard veto and evict the
    // next-oldest known-nonhard entry.
    cache.limit_tokens = 3;
    cache.update();
    CHECK(host_source_present(cache, 1));
    CHECK(!host_source_present(cache, 2));
    CHECK(authority.destruction.host_trade_attempted == 2);
    CHECK(authority.destruction.host_trade_hard_lease_vetoes == 1);
    CHECK(authority.destruction.host_trade_refused == 1);
    CHECK(authority.destruction.host_trade_executed == 0);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
    CHECK(authority.destruction_counters.refused
              [size_t(common_cache_plan_selection::none)]
              [size_t(common_cache_plan_destruction_reason::
                  hard_lease_blocked)] == 1);
}

void test_host_trade_all_hard_skips_publication() {
    server_cache_authority authority;
    available_host_fallback fallback;
    server_cache_lease_table hard_leases(nullptr, &fallback);
    const std::string execution = "trade-all-hard";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution, &hard_leases);
    cache.debug_observability = true;

    auto first = install_host_trade_entry(cache, authority, "hard-a", 64);
    auto second = install_host_trade_entry(cache, authority, "hard-b", 64);
    first->cache_plan_source_id = 11;
    second->cache_plan_source_id = 12;
    CHECK(grant_host_lease(
        cache, hard_leases, first, server_cache_lease_class::hard));
    CHECK(grant_host_lease(
        cache, hard_leases, second, server_cache_lease_class::hard));

    const auto live_ops_before = authority.ledger.snapshot().live_ops;
    cache.limit_tokens = cache.n_tokens();
    CHECK(!cache.publish(make_prompt_entry("incoming", { 90, 91, 92 })));
    CHECK(cache.states.size() == 2);
    CHECK(host_source_present(cache, 11));
    CHECK(host_source_present(cache, 12));
    CHECK(authority.destruction.host_trade_hard_lease_vetoes == 2);
    CHECK(authority.destruction.host_trade_refused == 0);
    CHECK(authority.destruction.host_trade_publication_skips == 1);
    CHECK(authority.ledger.snapshot().live_ops == live_ops_before);
    CHECK(authority.destruction_counters.last_receipt.state ==
          common_cache_plan_destruction_state::refused);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::hard_lease_blocked);
    CHECK(cache.debug_destruction_emissions == 3);
}

void test_host_trade_floor_skips_recovery_pin() {
    server_cache_authority authority;
    const std::string execution = "trade-pinned-floor";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);

    auto pinned = install_host_trade_entry(cache, authority, "pinned", 64);
    auto open = install_host_trade_entry(cache, authority, "open", 64);
    pinned->cache_plan_source_id = 21;
    open->cache_plan_source_id = 22;
    pinned->recovery_pins = 1;

    cache.limit_tokens = 3;
    cache.update();
    CHECK(host_source_present(cache, 21));
    CHECK(!host_source_present(cache, 22));
    CHECK(authority.destruction.host_trade_refused == 1);
    CHECK(authority.destruction.host_trade_legacy_fallbacks == 1);
}

void test_host_trade_partial_substrate_is_typed() {
    server_cache_authority authority;
    const std::string execution = "trade-partial-substrate";
    server_prompt_cache cache(0, 0);
    configure_host_trade(authority, cache, execution);
    auto first = install_host_trade_entry(cache, authority, "first", 64);
    auto second = install_host_trade_entry(cache, authority, "second", 64);
    first->cache_plan_source_id = 31;
    second->cache_plan_source_id = 32;

    cache.lease_obs = nullptr;
    cache.limit_tokens = 3;
    cache.update();
    CHECK(cache.states.size() == 1);
    CHECK(!host_source_present(cache, 31));
    CHECK(host_source_present(cache, 32));
    CHECK(cache.host_trade_substrate_warned);
    CHECK(authority.destruction.host_trade_substrate_unavailable == 1);
    CHECK(authority.destruction_counters.last_receipt.reason ==
          common_cache_plan_destruction_reason::lease_unavailable);
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();
    if (argc == 2 && std::string(argv[1]) == "--clone-fault") {
        test_lifecycle_restore_clone_fault();
        llama_backend_free();
        if (failures == 0) {
            std::puts("test-server-prompt-cache: CLONE_FAULT_PASS");
        }
        return failures == 0 ? 0 : 1;
    }
    test_lifecycle_full_cache_rotates();
    test_lifecycle_restore_retains_immutable_source();
    test_lifecycle_off_restore_consumes();
    test_lifecycle_release_prepare_failure_keeps_legacy_bound();
    test_lifecycle_authority_without_debug_is_silent();
    test_authority_source_ids_survive_save_dedup();
    test_exact_redundant_host_eviction();
    test_redundancy_payload_mismatch_and_missing_catalog();
    test_host_trade_soft_lease_weight_flips_victim();
    test_host_trade_main_family_weight_flips_victim();
    test_host_trade_zero_destruction_tie_break();
    test_host_trade_all_refuse_falls_back_to_legacy();
    test_host_trade_hard_lease_veto();
    test_host_trade_all_hard_skips_publication();
    test_host_trade_floor_skips_recovery_pin();
    test_host_trade_partial_substrate_is_typed();
    llama_backend_free();

    if (failures != 0) {
        std::fprintf(stderr, "test-server-prompt-cache: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test-server-prompt-cache: PASS");
    return 0;
}
