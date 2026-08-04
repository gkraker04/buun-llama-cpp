#include "server-cache-authority.h"
#include "server-cache-plan-authority.h"
#include "server-task.h"

#include "llama.h"

#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <list>
#include <string>

namespace {

int failures = 0;

#define CHECK(expr) do {                                                        \
    if (!(expr)) {                                                              \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",                   \
                     __FILE__, __LINE__, #expr);                                \
        failures++;                                                             \
    }                                                                           \
} while (0)

void configure_host_accounting(server_cache_authority & authority) {
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const llama_cache_acct_completeness_requirement required = {
        host, llama_cache_acct_producer::host_cache,
    };
    CHECK(authority.ledger.configure_required_producers(&required, 1));
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            authority.ledger.gauge_set(category, host, measure, 0);
        }
    }
    CHECK(authority.ledger.certify_complete(
        host, llama_cache_acct_producer::host_cache));
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
    llama_backend_free();

    if (failures != 0) {
        std::fprintf(stderr, "test-server-prompt-cache: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test-server-prompt-cache: PASS");
    return 0;
}
