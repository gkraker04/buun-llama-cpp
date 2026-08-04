#include "server-cache-authority.h"
#include "server-task.h"

#include "llama.h"

#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <list>

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

    CHECK(cache.publish(make_entry("oldest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "oldest");

    CHECK(cache.publish(make_entry("newest", 700 * 1024)));
    CHECK(cache.states.size() == 1);
    CHECK(cache.states.front().adapter_config_key == "newest");
    CHECK(cache.size() == 700 * 1024);
    CHECK(authority.admission_commits == 2);
    CHECK(authority.admission_refusals == 0);
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

int main() {
    llama_backend_init();
    test_lifecycle_full_cache_rotates();
    test_authority_source_ids_survive_save_dedup();
    llama_backend_free();

    if (failures != 0) {
        std::fprintf(stderr, "test-server-prompt-cache: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("test-server-prompt-cache: PASS");
    return 0;
}
