#include "llama-cache-accounting.h"

#include <limits>

// Shadow accounting ledger (C0). Every entry point is fault-tolerant AND non-throwing by
// contract: an invalid transition, unknown id, tuple mismatch, overflow, or internal
// allocation failure increments a fault counter and returns failure — it never throws and
// never influences the shipped mutation it observes. bump_serial() runs on every observable
// change, fault counters included, so the serial is a usable coherence epoch.

llama_cache_acct_ledger::llama_cache_acct_ledger() {
    for (auto & row : state.cells) {
        for (auto & cell : row) {
            for (auto & m : cell.measures) {
                m = { 0, llama_cache_acct_known::unknown };
            }
        }
    }
    // The ledger reports exactly what producers report into it; completeness stays unknown
    // until the C aggregator can certify coverage.
    state.completeness = llama_cache_acct_known::unknown;
}

void llama_cache_acct_ledger::bump_serial() {
    if (state.serial == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        return;
    }
    state.serial++;
}

void llama_cache_acct_ledger::cell_add(llama_cache_acct_category c, llama_cache_acct_residency r,
                                       llama_cache_acct_measure m, uint64_t v) {
    auto & cell = state.cells[size_t(c)][size_t(r)].measures[size_t(m)];
    if (cell.state == llama_cache_acct_known::unavailable) {
        return; // latched by a prior overflow; only a schema reset clears it
    }
    if (cell.value > std::numeric_limits<uint64_t>::max() - v) {
        cell.state = llama_cache_acct_known::unavailable;
        state.faults_overflow++;
        return;
    }
    cell.value += v;
    cell.state  = llama_cache_acct_known::known;
}

void llama_cache_acct_ledger::cell_sub(llama_cache_acct_category c, llama_cache_acct_residency r,
                                       llama_cache_acct_measure m, uint64_t v) {
    auto & cell = state.cells[size_t(c)][size_t(r)].measures[size_t(m)];
    if (cell.state == llama_cache_acct_known::unavailable) {
        return;
    }
    if (cell.value < v) {
        // an underflow is an accounting bug, not a shipped-path condition: latch, count
        cell.state = llama_cache_acct_known::unavailable;
        state.faults_overflow++;
        return;
    }
    cell.value -= v;
    cell.state  = llama_cache_acct_known::known;
}

void llama_cache_acct_ledger::cell_latch_unavailable(llama_cache_acct_category c,
                                                     llama_cache_acct_residency r,
                                                     llama_cache_acct_measure m) {
    state.cells[size_t(c)][size_t(r)].measures[size_t(m)].state =
        llama_cache_acct_known::unavailable;
}

void llama_cache_acct_ledger::staged_add(llama_cache_acct_category c, llama_cache_acct_residency r,
                                         uint64_t v) {
    auto & now = staged_now[size_t(c)][size_t(r)];
    if (now > std::numeric_limits<uint64_t>::max() - v) {
        state.faults_overflow++;
        cell_latch_unavailable(c, r, llama_cache_acct_measure::transient_peak);
        return;
    }
    now += v;
    // the peak is the high-water mark of CONCURRENTLY staged bytes
    auto & peak = state.cells[size_t(c)][size_t(r)].measures[size_t(llama_cache_acct_measure::transient_peak)];
    if (peak.state != llama_cache_acct_known::unavailable && now > peak.value) {
        peak.value = now;
        peak.state = llama_cache_acct_known::known;
    }
}

void llama_cache_acct_ledger::staged_sub(llama_cache_acct_category c, llama_cache_acct_residency r,
                                         uint64_t v) {
    auto & now = staged_now[size_t(c)][size_t(r)];
    if (now < v) {
        state.faults_overflow++;
        cell_latch_unavailable(c, r, llama_cache_acct_measure::transient_peak);
        return;
    }
    now -= v;
}

void llama_cache_acct_ledger::maybe_retire(alloc_entry & entry) {
    if (entry.ever_committed && entry.staged_refs == 0 && entry.committed_refs == 0) {
        // the id is DEAD: the tombstone survives so it can never name a new allocation
        entry.retired = true;
    }
}

llama_cache_acct_alloc_id llama_cache_acct_ledger::new_alloc() {
    std::lock_guard<std::mutex> lock(mtx);
    if (next_alloc_id.v == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return {};
    }
    try {
        // the registry entry IS the mint: identity exists before any citation, and it
        // survives retirement as a tombstone (no id resurrection)
        const llama_cache_acct_alloc_id id = next_alloc_id;
        allocs.emplace(id, alloc_entry{});
        next_alloc_id.v++;
        return id;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return {};
    }
}

llama_cache_acct_op_id llama_cache_acct_ledger::reserve(
        llama_cache_acct_category      category,
        llama_cache_acct_residency     residency,
        llama_cache_acct_attribution   attribution,
        uint64_t                       expected_logical,
        uint64_t                       expected_resident) {
    (void) expected_logical; // expectation is observational; only commit charges logical

    std::lock_guard<std::mutex> lock(mtx);

    if (next_op.v == std::numeric_limits<uint64_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return {};
    }

    try {
        const llama_cache_acct_op_id op = next_op;
        next_op.v++;

        txn t;
        t.state          = llama_cache_acct_txn_state::reserved;
        t.category       = category;
        t.residency      = residency;
        t.attribution    = attribution;
        t.reserved_bytes = expected_resident;
        ops.emplace(op, t);

        cell_add(category, residency, llama_cache_acct_measure::reserved, expected_resident);
        bump_serial();
        return op;
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        return {};
    }
}

bool llama_cache_acct_ledger::stage(llama_cache_acct_op_id op, llama_cache_acct_alloc_id alloc,
                                    uint64_t resident_bytes,
                                    llama_cache_acct_artifact_id    artifact,
                                    llama_cache_acct_content_digest digest,
                                    llama_cache_acct_lineage_id     lineage) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::reserved) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    // allocation ids must come from new_alloc(): the registry entry is the mint proof
    auto ait = allocs.find(alloc);
    if (!alloc || ait == allocs.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    // a retired id names a DEAD physical allocation: citing it again is resurrection
    if (ait->second.retired) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }

    if (ait->second.tuple_set) {
        // the citation tuple is immutable — identity fields included: a mismatched
        // re-citation is a fault, never a silent merge (false charge-once would undercount)
        if (ait->second.category != it->second.category ||
            ait->second.residency != it->second.residency ||
            ait->second.resident_bytes != resident_bytes ||
            ait->second.artifact != artifact ||
            ait->second.digest   != digest ||
            ait->second.lineage  != lineage) {
            state.faults_invalid_transition++;
            bump_serial();
            return false;
        }
    } else {
        ait->second.tuple_set      = true;
        ait->second.category       = it->second.category;
        ait->second.residency      = it->second.residency;
        ait->second.resident_bytes = resident_bytes;
        ait->second.artifact       = artifact;
        ait->second.digest         = digest;
        ait->second.lineage        = lineage;
    }
    if (ait->second.staged_refs == std::numeric_limits<uint32_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return false;
    }
    ait->second.staged_refs++;

    it->second.state          = llama_cache_acct_txn_state::staged;
    it->second.alloc          = alloc;
    it->second.resident_bytes = resident_bytes;
    it->second.artifact       = artifact;
    it->second.digest         = digest;
    it->second.lineage        = lineage;

    staged_add(it->second.category, it->second.residency, resident_bytes);
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::commit(llama_cache_acct_op_id op, uint64_t logical_bytes) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::staged) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }

    auto ait = allocs.find(it->second.alloc);
    if (ait == allocs.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }

    auto & entry = ait->second;
    // defensive: a retired allocation accepts no commit (unreachable while retirement
    // accounts for staged claims, but fail loudly if that invariant ever breaks)
    if (entry.retired) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    if (entry.committed_refs > 0 && entry.charged_logical != logical_bytes) {
        // a shared immutable allocation has ONE logical size; a mismatched later commit is
        // a fault and does not join the refcount
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }
    if (entry.committed_refs == std::numeric_limits<uint32_t>::max()) {
        state.faults_overflow++;
        bump_serial();
        return false;
    }

    it->second.state = llama_cache_acct_txn_state::committed;

    // the staged and reserved observations both resolve at the publication boundary
    staged_sub(it->second.category, it->second.residency, it->second.resident_bytes);
    if (entry.staged_refs > 0) {
        entry.staged_refs--;
    }
    cell_sub(it->second.category, it->second.residency,
             llama_cache_acct_measure::reserved, it->second.reserved_bytes);

    entry.committed_refs++;
    entry.ever_committed = true;
    if (entry.committed_refs == 1) {
        // charge-once: the first committed reference charges the allocation's durable bytes
        entry.charged_logical = logical_bytes;
        entry.attribution     = it->second.attribution;
        cell_add(entry.category, entry.residency,
                 llama_cache_acct_measure::logical_payload,    entry.charged_logical);
        cell_add(entry.category, entry.residency,
                 llama_cache_acct_measure::resident_allocated, entry.resident_bytes);
    }
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::abort(llama_cache_acct_op_id op) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::reserved &&
        it->second.state != llama_cache_acct_txn_state::staged) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }

    // zero durable delta: the reservation unwinds by the RESERVED amount, the concurrent
    // staged tracking by the ACTUAL amount; the transient peak stays observed. The op is
    // erased — an aborted transaction holds nothing. The allocation entry persists (mint
    // registry), keeping its tuple immutable for any other citation.
    cell_sub(it->second.category, it->second.residency,
             llama_cache_acct_measure::reserved, it->second.reserved_bytes);
    if (it->second.state == llama_cache_acct_txn_state::staged) {
        staged_sub(it->second.category, it->second.residency, it->second.resident_bytes);
        auto ait = allocs.find(it->second.alloc);
        if (ait != allocs.end() && ait->second.staged_refs > 0) {
            ait->second.staged_refs--;
            maybe_retire(ait->second);
        }
    }
    ops.erase(it);
    bump_serial();
    return true;
}

bool llama_cache_acct_ledger::release(llama_cache_acct_op_id op) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = ops.find(op);
    if (it == ops.end()) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }
    if (it->second.state != llama_cache_acct_txn_state::committed) {
        state.faults_invalid_transition++;
        bump_serial();
        return false;
    }

    auto ait = allocs.find(it->second.alloc);
    if (ait == allocs.end() || ait->second.committed_refs == 0) {
        state.faults_unknown_id++;
        bump_serial();
        return false;
    }

    ait->second.committed_refs--;
    if (ait->second.committed_refs == 0) {
        cell_sub(ait->second.category, ait->second.residency,
                 llama_cache_acct_measure::logical_payload,    ait->second.charged_logical);
        cell_sub(ait->second.category, ait->second.residency,
                 llama_cache_acct_measure::resident_allocated, ait->second.resident_bytes);
    }
    // a staged claim may still commit: retirement waits for BOTH claim kinds to drain
    maybe_retire(ait->second);
    ops.erase(it);
    bump_serial();
    return true;
}

void llama_cache_acct_ledger::gauge_set(llama_cache_acct_category category,
                                        llama_cache_acct_residency residency,
                                        llama_cache_acct_measure measure,
                                        uint64_t value) {
    std::lock_guard<std::mutex> lock(mtx);

    auto & cell = state.cells[size_t(category)][size_t(residency)].measures[size_t(measure)];
    if (cell.state == llama_cache_acct_known::unavailable) {
        return;
    }
    cell.value = value;
    cell.state = llama_cache_acct_known::known;
    bump_serial();
}

void llama_cache_acct_ledger::mark_unavailable(llama_cache_acct_category category,
                                               llama_cache_acct_residency residency,
                                               llama_cache_acct_measure measure) {
    std::lock_guard<std::mutex> lock(mtx);
    cell_latch_unavailable(category, residency, measure);
    bump_serial();
}

llama_cache_acct_snapshot llama_cache_acct_ledger::snapshot() {
    std::lock_guard<std::mutex> lock(mtx);

    llama_cache_acct_snapshot out = state;
    out.live_ops = (uint64_t) ops.size();
    try {
        out.allocations.reserve(allocs.size());
        for (const auto & [alloc, entry] : allocs) {
            if (entry.committed_refs == 0) {
                continue; // staged-only and retired allocations are not durable rows
            }
            llama_cache_acct_allocation_row row;
            row.alloc             = alloc;
            row.attribution       = entry.attribution;
            row.category          = entry.category;
            row.residency         = entry.residency;
            row.logical_bytes     = entry.charged_logical;
            row.resident_bytes    = entry.resident_bytes;
            row.committed_refs    = entry.committed_refs;
            row.artifact_identity = entry.artifact;
            row.content_digest    = entry.digest;
            row.lineage_identity  = entry.lineage;
            out.allocations.push_back(row);
        }
    } catch (...) {
        state.faults_allocation++;
        bump_serial();
        out.allocations.clear();
        out.completeness = llama_cache_acct_known::unavailable;
        out.faults_allocation = state.faults_allocation;
        out.serial            = state.serial;
    }
    return out;
}
