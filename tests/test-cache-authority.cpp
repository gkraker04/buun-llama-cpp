// F0a admission-composer + reservation-claim contract tests. The composer is the single place
// that runs snapshot -> fits(reserve-only) -> reserve_if_serial; these tests pin its honest status
// taxonomy (fail-closed on incomplete evidence; no fabricated precision) and the move-only claim's
// auto-abort so an admitted-but-uncommitted reservation can never leak. The reserve_if_serial and
// fits primitives themselves are covered by test-cache-accounting / test-cache-budget.

#include "llama-cache-authority.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static const auto HOST = llama_cache_acct_resource_domain::non_device(
    llama_cache_acct_residency::pageable_host);
static const auto PAYLOAD = llama_cache_acct_category::full_snapshot_payload;

// Build a ledger with one certified, available durable host cell — mirrors the server's host-cache
// setup order (configure required producers -> create the transactional leaf -> certify). A host
// leaf has an unbounded budget ceiling, so a small reservation against it fits under default caps.
static void configure_fitting_host(llama_cache_acct_ledger & ledger) {
    const llama_cache_acct_completeness_requirement req = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&req, 1));
    // fits() marks a domain unavailable unless every durable+host cell has known resident AND
    // (being transactional) known reserved bytes, so give all three durable+host payload leaves a
    // measured zero on each transactional measure (exactly as the server's host init does).
    for (const auto cat : { llama_cache_acct_category::full_snapshot_payload,
                            llama_cache_acct_category::checkpoint_state_payload,
                            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : { llama_cache_acct_measure::logical_payload,
                                    llama_cache_acct_measure::resident_allocated,
                                    llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(cat, HOST, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
}

static llama_cache_authority_request host_request(uint64_t resident) {
    llama_cache_authority_request req;
    req.category          = PAYLOAD;
    req.domain            = HOST;
    req.expected_logical  = resident;
    req.expected_resident = resident;
    return req;
}

// Configure a fresh host ledger and run one admission against it. The ledger is caller-owned so the
// returned claim's lifetime (and the live_ops assertions) stay in the test's own scope.
static llama_cache_admission_result admit_fresh_host(
        llama_cache_acct_ledger & ledger, uint64_t resident) {
    configure_fitting_host(ledger);
    const llama_cache_budget_config config; // empty devices + default unbounded host caps
    return llama_cache_admit_reservation(ledger, config, host_request(resident));
}

// An unconfigured ledger has a non-known completeness manifest: the composer must refuse before it
// prices anything (fail-closed), never admit on private counters.
static void test_incomplete_evidence() {
    llama_cache_acct_ledger ledger;
    llama_cache_budget_config config;
    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::incomplete_evidence);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

// Manifest is known but the priced domain has no budget-visible cell (no gauge): fits() collapses
// to unavailable and the composer reports budget_unavailable — NOT incomplete_evidence.
static void test_budget_unavailable() {
    llama_cache_acct_ledger ledger;
    const llama_cache_acct_completeness_requirement req = {
        HOST, llama_cache_acct_producer::host_cache,
    };
    CHECK(ledger.configure_required_producers(&req, 1));
    CHECK(ledger.certify_complete(HOST, llama_cache_acct_producer::host_cache));
    // No gauge_set: the manifest is known, but HOST owns no durable cell to price against.
    CHECK(ledger.snapshot().completeness_manifest == llama_cache_acct_known::known);

    llama_cache_budget_config config;
    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::budget_unavailable);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

static uint64_t host_reserved(llama_cache_acct_ledger & ledger) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category == PAYLOAD && row.domain == HOST) {
            return row.cell.measures[size_t(llama_cache_acct_measure::reserved)].value;
        }
    }
    return 0;
}

// Happy path: the reservation fits and is admitted, handing back a claim that owns the reserved op,
// and the reserved aggregate actually moved by the requested amount (not just live_ops).
static void test_admitted() {
    llama_cache_acct_ledger ledger;
    const auto res = admit_fresh_host(ledger, 128);
    CHECK(res.status == llama_cache_admission_status::admitted);
    CHECK(res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(host_reserved(ledger) == 128);
}

// Manifest known + host cell present, but a host total cap below the request: fits() reports exceeds
// and the composer maps it to exceeds_budget (distinct from budget_unavailable), reserving nothing.
static void test_exceeds_budget() {
    llama_cache_acct_ledger ledger;
    configure_fitting_host(ledger);
    llama_cache_budget_config config;
    config.host.total_state = llama_cache_budget_capacity_state::known;
    config.host.total_cap   = 100;

    const auto res = llama_cache_admit_reservation(ledger, config, host_request(128));
    CHECK(res.status == llama_cache_admission_status::exceeds_budget);
    CHECK(!res.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 0);
}

// A dropped claim (the only F0a terminal — there is no bare discharge) aborts its reserved op, so an
// admitted-but-abandoned reservation leaves the ledger with zero live ops — the leak guard for F0b.
static void test_claim_auto_abort() {
    llama_cache_acct_ledger ledger;
    {
        const auto res = admit_fresh_host(ledger, 64);
        CHECK(res.status == llama_cache_admission_status::admitted);
        CHECK(ledger.snapshot().live_ops == 1);
    } // res.claim destroyed here -> op aborted
    CHECK(ledger.snapshot().live_ops == 0);
}

// Move-construction transfers ownership; the moved-from claim is inert; exactly one abort fires.
static void test_claim_move_ctor() {
    llama_cache_acct_ledger ledger;
    auto res = admit_fresh_host(ledger, 64);
    CHECK(res.status == llama_cache_admission_status::admitted);
    {
        llama_cache_reservation_claim b(std::move(res.claim));
        CHECK(!res.claim.has_op());
        CHECK(b.has_op());
        CHECK(ledger.snapshot().live_ops == 1);
    }
    CHECK(ledger.snapshot().live_ops == 0); // only b aborted
}

// Move-assignment aborts the destination's own op before taking the source's; self-move is a no-op.
static void test_claim_move_assign() {
    llama_cache_acct_ledger ledger;
    auto dst = admit_fresh_host(ledger, 64);
    llama_cache_budget_config config; // ledger already configured by admit_fresh_host
    auto src = llama_cache_admit_reservation(ledger, config, host_request(64));
    CHECK(dst.status == llama_cache_admission_status::admitted);
    CHECK(src.status == llama_cache_admission_status::admitted);
    CHECK(ledger.snapshot().live_ops == 2);

    dst.claim = std::move(src.claim);
    CHECK(!src.claim.has_op());
    CHECK(dst.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1); // dst's original op was aborted

    // self-move (through a pointer so the compiler cannot warn): keeps its op.
    llama_cache_reservation_claim * self = &dst.claim;
    *self = std::move(*self);
    CHECK(dst.claim.has_op());
    CHECK(ledger.snapshot().live_ops == 1);
}

static void test_status_names() {
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::admitted)) == "admitted");
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::incomplete_evidence)) == "incomplete_evidence");
    CHECK(std::string(llama_cache_admission_status_name(
        llama_cache_admission_status::internal_fault)) == "internal_fault");
}

int main() {
    test_incomplete_evidence();
    test_budget_unavailable();
    test_admitted();
    test_exceeds_budget();
    test_claim_auto_abort();
    test_claim_move_ctor();
    test_claim_move_assign();
    test_status_names();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-authority tests passed\n");
    return EXIT_SUCCESS;
}
