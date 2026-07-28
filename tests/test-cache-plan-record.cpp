// B0 decision-record contract tests: band monotonicity (compile-time), multi-failure
// first-reason precedence including out-of-order arrival, valid-loser disposition, upsert
// identity, unknown-vs-zero on measured fields, and exhaustive name tables (every member of
// every closed enum must produce a non-"invalid" name).

#include "common-cache-plan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

// multi-failure: the earliest precedence band is THE reason regardless of arrival order
static void test_precedence() {
    common_cache_plan_candidate c;
    c.note_reject(COMMON_CACHE_PLAN_REASON_PERSISTENT_BUDGET_EXCEEDED);   // 500 first
    c.note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);                // 206 arrives later
    c.note_reject(COMMON_CACHE_PLAN_REASON_KV_TYPE_MISMATCH);             // 402 later still
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
    CHECK(c.disposition == common_cache_plan_disposition::rejected_invalid);

    // identity always dominates
    c.note_reject(COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);      // 102
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH);
}

// a valid loser is not an invalid candidate — and an invalidity ever observed dominates cost
static void test_valid_loser() {
    common_cache_plan_candidate c;
    c.note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(c.disposition == common_cache_plan_disposition::valid_not_chosen_cost);

    c.note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    CHECK(c.reason == COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    CHECK(c.disposition == common_cache_plan_disposition::rejected_invalid);

    // and the reverse order: cost after invalidity never resurrects the candidate
    common_cache_plan_candidate d;
    d.note_reject(COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID);
    d.note_reject(COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL);
    CHECK(d.disposition == common_cache_plan_disposition::rejected_invalid);
}

// one row per provider across stages; none observed means typed-unknown fields, never zeros
static void test_record_stages() {
    common_cache_plan_record rec;
    // outcome `unknown` IS the typed not-finalized state
    CHECK(rec.outcome == common_cache_plan_outcome::unknown);
    CHECK(rec.n_reused_tokens.state == llama_cache_acct_known::unknown);
    CHECK(rec.ttft_us.state == llama_cache_acct_known::unknown);
    for (const auto & term : rec.cost_terms) {
        CHECK(term.estimated_us.state == llama_cache_acct_known::unknown);
    }
    // no provider observed yet — absence is not a vacuous verdict
    for (const auto & c : rec.candidates) {
        CHECK(!c.present);
        CHECK(!c.delivered);
        CHECK(!c.gen_eval.evaluated);
    }

    auto & slot_row = rec.row(common_cache_plan_provider::live_slot);
    slot_row.sim = 0.75; slot_row.sim_known = true;
    slot_row.disposition = common_cache_plan_disposition::accepted;
    CHECK(slot_row.present);

    // row() is idempotent identity, noexcept by construction (fixed array)
    auto & again = rec.row(common_cache_plan_provider::live_slot);
    CHECK(&again == &slot_row);

    auto & ckpt_row = rec.row(common_cache_plan_provider::live_context_checkpoint);
    ckpt_row.note_reject(COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED);
    CHECK(ckpt_row.present);
    CHECK(!rec.candidates[size_t(common_cache_plan_provider::host_cache_entry)].present);
    // a rejection is never a delivery
    CHECK(!ckpt_row.delivered);
}

// exhaustive name tables: every member names itself, no member is "invalid"
static void test_name_tables() {
    for (size_t i = 0; i < COMMON_CACHE_PLAN_REASON_MEMBER_COUNT; i++) {
        CHECK(strcmp(common_cache_plan_reason_name(common_cache_plan_reason_all[i]), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_disposition::_count); i++) {
        CHECK(strcmp(common_cache_plan_disposition_name(common_cache_plan_disposition(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_provider::_count); i++) {
        CHECK(strcmp(common_cache_plan_provider_name(common_cache_plan_provider(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_outcome::_count); i++) {
        CHECK(strcmp(common_cache_plan_outcome_name(common_cache_plan_outcome(i)), "invalid") != 0);
    }
    for (uint8_t i = 0; i < uint8_t(common_cache_plan_selection::_count); i++) {
        CHECK(strcmp(common_cache_plan_selection_name(common_cache_plan_selection(i)), "invalid") != 0);
    }
    // and the closed inventory really is closed: exactly today's four providers
    CHECK(uint8_t(common_cache_plan_provider::_count) == 4);
    // schema v1 member census + sentinel (compile-time pinned; echoed here as a wire check)
    CHECK(COMMON_CACHE_PLAN_REASON_MEMBER_COUNT == 30);
    CHECK(uint16_t(COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL) == 601);
}

// the cost array carries five DISTINCT kinds with their canonical raw units — a default
// array would collapse to five "restore"/bytes slots (Sol verify-r1 finding 9)
static void test_cost_term_defaults() {
    common_cache_plan_record rec;
    bool seen[size_t(llama_cache_acct_cost_kind::_count)] = {};
    for (const auto & term : rec.cost_terms) {
        CHECK(!seen[size_t(term.kind)]);
        seen[size_t(term.kind)] = true;
        CHECK(term.raw_unit == llama_cache_acct_cost_kind_unit(term.kind));
        CHECK(term.raw.state == llama_cache_acct_known::unknown);
        CHECK(term.estimated_us.state == llama_cache_acct_known::unknown);
    }
    CHECK(rec.cost_terms[size_t(llama_cache_acct_cost_kind::replay)].raw_unit ==
          llama_cache_acct_unit::tokens);
    // identity evidence starts typed-unknown across the board — never fabricated digests
    CHECK(rec.identity.model_digest.state == llama_cache_acct_known::unknown);
    CHECK(rec.identity.prefix_token_digest.state == llama_cache_acct_known::unknown);
}

int main() {
    test_precedence();
    test_valid_loser();
    test_record_stages();
    test_name_tables();
    test_cost_term_defaults();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-plan-record tests passed\n");
    return EXIT_SUCCESS;
}
