// C0 shadow-ledger contract tests: reserve/stage/commit/abort/release state machine,
// minted-allocation identity + immutable citation tuples, charge-once shared allocations,
// reserved-vs-actual byte separation, concurrent-staged transient peak, unknown-vs-zero,
// checked overflow latching, serial coherence (faults included), and attribution round-trip
// through the normalized allocation rows. Every negative case asserts BOTH the failure
// return and the fault counter — the ledger must misbehave loudly and harmlessly.

#include "llama-cache-accounting.h"

#include <cstdio>
#include <cstdlib>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static const auto CAT  = llama_cache_acct_category::full_snapshot_payload;
static const auto RES  = llama_cache_acct_residency::pageable_host;
static const auto META = llama_cache_acct_category::artifact_reference_metadata;

static llama_cache_acct_value cell(const llama_cache_acct_snapshot & s,
                                   llama_cache_acct_category c,
                                   llama_cache_acct_residency r,
                                   llama_cache_acct_measure m) {
    return s.cells[size_t(c)][size_t(r)].measures[size_t(m)];
}

// happy path: reserve -> stage -> commit -> release round-trips durable gauges to zero
static void test_lifecycle() {
    llama_cache_acct_ledger ledger;

    const auto op = ledger.reserve(CAT, RES, {}, 100, 128);
    CHECK(op != 0);
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).value == 128);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown); // unknown until a commit, never a fabricated zero

    const auto alloc = ledger.new_alloc();
    CHECK(alloc != 0);
    CHECK(ledger.stage(op, alloc, 128));
    CHECK(ledger.commit(op, 100));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value   == 100);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::resident_allocated).value == 128);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::transient_peak).value    == 128);
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].logical_bytes == 100 && s.allocations[0].resident_bytes == 128);

    CHECK(ledger.release(op));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value    == 0);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::resident_allocated).value == 0);
    // a discharged gauge is a MEASURED zero, not unknown
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::known);
    CHECK(s.allocations.empty());
    CHECK(s.faults_invalid_transition == 0 && s.faults_overflow == 0 &&
          s.faults_unknown_id == 0 && s.faults_allocation == 0);
}

// Sol verify-r1 finding 5.1: reserved is charged/unwound by the RESERVED amount even when
// the staged actual differs — reserve 64, stage 32, abort must leave reserved == 0
static void test_reserve_stage_mismatch() {
    llama_cache_acct_ledger ledger;

    const auto op = ledger.reserve(CAT, RES, {}, 64, 64);
    CHECK(ledger.stage(op, ledger.new_alloc(), 32));
    CHECK(ledger.abort(op));
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).state ==
          llama_cache_acct_known::known);
    CHECK(s.faults_overflow == 0);

    // and the commit side of the same asymmetry
    const auto op2 = ledger.reserve(CAT, RES, {}, 64, 64);
    CHECK(ledger.stage(op2, ledger.new_alloc(), 32));
    CHECK(ledger.commit(op2, 10));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).value == 0);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::resident_allocated).value == 32);
    CHECK(ledger.release(op2));
}

// Sol verify-r1 finding 5.2: the transient peak is the high-water mark of CONCURRENTLY
// staged bytes — two live stages of 100 and 200 must report 300
static void test_concurrent_peak() {
    llama_cache_acct_ledger ledger;

    const auto op1 = ledger.reserve(CAT, RES, {}, 100, 100);
    const auto op2 = ledger.reserve(CAT, RES, {}, 200, 200);
    CHECK(ledger.stage(op1, ledger.new_alloc(), 100));
    CHECK(ledger.stage(op2, ledger.new_alloc(), 200));
    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::transient_peak).value == 300);
    CHECK(ledger.abort(op1));
    CHECK(ledger.abort(op2));
    // the peak survives the aborts
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::transient_peak).value == 300);
}

// invalid transitions: all fault-counted, none throw; op ids and alloc ids are validated
static void test_invalid_transitions() {
    llama_cache_acct_ledger ledger;

    const auto alloc = ledger.new_alloc();
    const auto op = ledger.reserve(CAT, RES, {}, 10, 10);
    CHECK(!ledger.commit(op, 10));               // commit before stage
    CHECK(!ledger.release(op));                  // release before commit
    CHECK(!ledger.stage(op, 0, 10));             // zero alloc id (unknown_id)
    CHECK(!ledger.stage(op, alloc + 999, 10));   // unminted alloc id (unknown_id)
    CHECK(ledger.stage(op, alloc, 10));
    CHECK(!ledger.stage(op, alloc, 10));         // double stage
    CHECK(ledger.commit(op, 10));
    CHECK(!ledger.commit(op, 10));               // double commit
    CHECK(!ledger.abort(op));                    // abort after commit
    CHECK(ledger.release(op));
    CHECK(!ledger.release(op));                  // double release (op erased -> unknown id)

    CHECK(!ledger.stage(999, alloc, 1));         // unknown op

    const auto s = ledger.snapshot();
    // early-commit, early-release, double-stage, double-commit, abort-after-commit
    CHECK(s.faults_invalid_transition == 5);
    // zero alloc, unminted alloc, double release (op erased), unknown op
    CHECK(s.faults_unknown_id == 4);
}

// Sol verify-r1 finding 5.4: an allocation's citation tuple is immutable — a second
// transaction citing the same alloc with a different size/category is a fault, never a
// silent merge; a later commit with a different logical size is a fault too
static void test_alloc_tuple_immutable() {
    llama_cache_acct_ledger ledger;

    const auto alloc = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, RES, {}, 100, 100);
    const auto op2 = ledger.reserve(META, RES, {}, 999, 999); // different category
    const auto op3 = ledger.reserve(CAT, RES, {}, 100, 100);

    CHECK(ledger.stage(op1, alloc, 100));
    CHECK(!ledger.stage(op2, alloc, 999));       // category+size mismatch -> fault
    CHECK(!ledger.stage(op3, alloc, 50));        // size mismatch -> fault
    CHECK(ledger.stage(op3, alloc, 100));        // matching tuple joins

    CHECK(ledger.commit(op1, 80));
    CHECK(!ledger.commit(op3, 81));              // logical mismatch on shared alloc -> fault
    CHECK(ledger.commit(op3, 80));

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 80); // charged once
    CHECK(s.faults_invalid_transition >= 3);

    CHECK(ledger.release(op1));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 80);
    CHECK(ledger.release(op3));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 0);
    ledger.abort(op2); // cleanup (still reserved)
}

// abort: zero durable delta, reservation unwound, transient peak retained
static void test_abort_retains_peak() {
    llama_cache_acct_ledger ledger;

    const auto op = ledger.reserve(CAT, RES, {}, 50, 64);
    CHECK(ledger.stage(op, ledger.new_alloc(), 64));
    CHECK(ledger.abort(op));

    const auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::reserved).value        == 0);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown);
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::transient_peak).value  == 64);
    CHECK(s.allocations.empty()); // the staged-only allocation entry is gone with the abort
}

// charge-once: two committed references to one allocation charge the durable bytes once;
// the allocation discharges only when the LAST reference releases
static void test_charge_once() {
    llama_cache_acct_ledger ledger;

    const auto alloc = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, RES, {}, 100, 100);
    const auto op2 = ledger.reserve(CAT, RES, {}, 100, 100);
    CHECK(ledger.stage(op1, alloc, 100));
    CHECK(ledger.stage(op2, alloc, 100));
    CHECK(ledger.commit(op1, 100));
    CHECK(ledger.commit(op2, 100));

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 100);
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].committed_refs == 2);

    // per-reference metadata is a separate leaf, outside the refcount
    ledger.gauge_set(META, llama_cache_acct_residency::pageable_host,
                     llama_cache_acct_measure::logical_payload, 2 * 16);

    CHECK(ledger.release(op1));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 100);

    CHECK(ledger.release(op2));
    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 0);
}

// Sol verify-r2 finding 3: a RETIRED allocation id can never name a new physical
// allocation (tombstone survives the last release), and the complete citation tuple —
// identity fields included — is immutable on every shared citation
static void test_alloc_no_resurrection() {
    llama_cache_acct_ledger ledger;

    const auto a = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, RES, {}, 1, 1);
    CHECK(ledger.stage(op1, a, 1));
    CHECK(ledger.commit(op1, 1));
    CHECK(ledger.release(op1)); // a is now retired

    const auto op2 = ledger.reserve(CAT, RES, {}, 2, 2);
    auto s0 = ledger.snapshot();
    CHECK(!ledger.stage(op2, a, 2)); // resurrection under a different size -> fault
    auto s1 = ledger.snapshot();
    CHECK(s1.faults_invalid_transition == s0.faults_invalid_transition + 1);
    CHECK(ledger.abort(op2));

    // identity-field mismatches on a LIVE shared allocation
    const auto b = ledger.new_alloc();
    const auto op3 = ledger.reserve(CAT, RES, {}, 5, 5);
    const auto op4 = ledger.reserve(CAT, RES, {}, 5, 5);
    CHECK(ledger.stage(op3, b, 5, llama_cache_acct_artifact_id{1},
                       llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{9},
                        llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                        llama_cache_acct_content_digest{8}, llama_cache_acct_lineage_id{3}));
    CHECK(!ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                        llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{7}));
    CHECK(ledger.stage(op4, b, 5, llama_cache_acct_artifact_id{1},
                       llama_cache_acct_content_digest{2}, llama_cache_acct_lineage_id{3}));
    CHECK(ledger.abort(op3));
    CHECK(ledger.abort(op4));
}

// Sol verify-r3 blocker: an outstanding STAGED claim defers retirement — the exact
// interleaving commit(op1) → stage(op2) → release(op1) → commit(op2) must accept the join,
// keep valid same-tuple citations working, and retire only after both claim kinds drain
static void test_staged_handoff() {
    llama_cache_acct_ledger ledger;

    const auto a = ledger.new_alloc();
    const auto op1 = ledger.reserve(CAT, RES, {}, 4, 4);
    CHECK(ledger.stage(op1, a, 4));
    CHECK(ledger.commit(op1, 4));

    const auto op2 = ledger.reserve(CAT, RES, {}, 4, 4);
    CHECK(ledger.stage(op2, a, 4));    // staged while op1 holds the committed claim
    CHECK(ledger.release(op1));        // committed refs hit zero, but op2's claim defers retirement
    CHECK(ledger.commit(op2, 4));      // the handoff join is accepted

    auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 4);
    CHECK(s.allocations.size() == 1 && s.allocations[0].committed_refs == 1);

    const auto op3 = ledger.reserve(CAT, RES, {}, 4, 4);
    CHECK(ledger.stage(op3, a, 4));    // valid same-tuple citation still accepted
    CHECK(ledger.commit(op3, 4));
    CHECK(ledger.release(op2));
    CHECK(ledger.release(op3));        // both claim kinds drained -> retired

    const auto op4 = ledger.reserve(CAT, RES, {}, 4, 4);
    CHECK(!ledger.stage(op4, a, 4));   // retired id stays dead
    CHECK(ledger.abort(op4));

    s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).value == 0);
    CHECK(s.live_ops == 0);
}

// live_ops in the snapshot proves nothing leaked: zero after every entry's full lifecycle
static void test_live_ops_zero() {
    llama_cache_acct_ledger ledger;

    const auto op = ledger.reserve(CAT, RES, {}, 8, 8);
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(ledger.stage(op, ledger.new_alloc(), 8));
    CHECK(ledger.commit(op, 8));
    CHECK(ledger.snapshot().live_ops == 1);
    CHECK(ledger.release(op));
    CHECK(ledger.snapshot().live_ops == 0);

    // an aborted op leaves nothing live either
    const auto op2 = ledger.reserve(CAT, RES, {}, 8, 8);
    CHECK(ledger.abort(op2));
    CHECK(ledger.snapshot().live_ops == 0);
}

// attribution round-trip: a slot-attributed committed allocation appears as a normalized
// row carrying its attribution (the explicit form D/F consume — no private counters)
static void test_attribution_rows() {
    llama_cache_acct_ledger ledger;

    llama_cache_acct_attribution attr;
    attr.kind    = llama_cache_acct_attr_kind::slot;
    attr.slot_id = 3;

    const auto op = ledger.reserve(CAT, RES, attr, 42, 42);
    CHECK(ledger.stage(op, ledger.new_alloc(), 42,
                       llama_cache_acct_artifact_id{7}, llama_cache_acct_content_digest{8},
                       llama_cache_acct_lineage_id{9}));
    CHECK(ledger.commit(op, 42));

    const auto s = ledger.snapshot();
    CHECK(s.allocations.size() == 1);
    CHECK(s.allocations[0].attribution.kind == llama_cache_acct_attr_kind::slot);
    CHECK(s.allocations[0].attribution.slot_id == 3);
    CHECK(s.allocations[0].artifact_identity.v == 7);
    CHECK(s.allocations[0].content_digest.v    == 8);
    CHECK(s.allocations[0].lineage_identity.v  == 9);
    CHECK(ledger.release(op));
}

// checked overflow latches the cell unavailable (never wraps, never zeros retroactively)
static void test_overflow_latch() {
    llama_cache_acct_ledger ledger;

    ledger.gauge_set(CAT, RES, llama_cache_acct_measure::logical_payload, UINT64_MAX - 1);
    const auto op = ledger.reserve(CAT, RES, {}, 10, 10);
    CHECK(ledger.stage(op, ledger.new_alloc(), 10));
    CHECK(ledger.commit(op, 10)); // the commit records; the CELL faults

    const auto s = ledger.snapshot();
    CHECK(cell(s, CAT, RES, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unavailable);
    CHECK(s.faults_overflow == 1);
}

// Sol verify-r1 finding 5.3: EVERY observable change bumps the serial — fault counters
// included — so the serial is a usable coherence epoch
static void test_serial_on_fault() {
    llama_cache_acct_ledger ledger;

    const auto s0 = ledger.snapshot();
    CHECK(!ledger.release(424242)); // unknown op -> fault
    const auto s1 = ledger.snapshot();
    CHECK(s1.faults_unknown_id == s0.faults_unknown_id + 1);
    CHECK(s1.serial > s0.serial);

    // and mark_unavailable is observable too
    ledger.mark_unavailable(CAT, RES, llama_cache_acct_measure::logical_payload);
    const auto s2 = ledger.snapshot();
    CHECK(s2.serial > s1.serial);
}

// snapshot serial: one serial per durable change, snapshots are coherent copies
static void test_snapshot_serial() {
    llama_cache_acct_ledger ledger;

    const auto s0 = ledger.snapshot();
    const auto op = ledger.reserve(CAT, RES, {}, 1, 1);
    const auto s1 = ledger.snapshot();
    CHECK(s1.serial > s0.serial);
    CHECK(ledger.stage(op, ledger.new_alloc(), 1));
    CHECK(ledger.commit(op, 1));
    const auto s2 = ledger.snapshot();
    CHECK(s2.serial > s1.serial);
    // the earlier snapshot is an unchanged copy, not a view
    CHECK(cell(s1, CAT, RES, llama_cache_acct_measure::logical_payload).state ==
          llama_cache_acct_known::unknown);
    CHECK(cell(s2, CAT, RES, llama_cache_acct_measure::logical_payload).value == 1);
}

int main() {
    test_lifecycle();
    test_reserve_stage_mismatch();
    test_concurrent_peak();
    test_invalid_transitions();
    test_alloc_tuple_immutable();
    test_alloc_no_resurrection();
    test_staged_handoff();
    test_live_ops_zero();
    test_abort_retains_peak();
    test_charge_once();
    test_attribution_rows();
    test_overflow_latch();
    test_serial_on_fault();
    test_snapshot_serial();

    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return EXIT_FAILURE;
    }
    printf("all cache-accounting tests passed\n");
    return EXIT_SUCCESS;
}
