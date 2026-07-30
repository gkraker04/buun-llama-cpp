#include "llama-cache-authority.h"

const char * llama_cache_admission_status_name(llama_cache_admission_status status) noexcept {
    switch (status) {
        case llama_cache_admission_status::admitted:            return "admitted";
        case llama_cache_admission_status::incomplete_evidence: return "incomplete_evidence";
        case llama_cache_admission_status::budget_unavailable:  return "budget_unavailable";
        case llama_cache_admission_status::exceeds_budget:      return "exceeds_budget";
        case llama_cache_admission_status::serial_conflict:     return "serial_conflict";
        case llama_cache_admission_status::ledger_fault:        return "ledger_fault";
        case llama_cache_admission_status::internal_fault:      return "internal_fault";
        case llama_cache_admission_status::_count:              break;
    }
    return "unknown";
}

llama_cache_reservation_claim::llama_cache_reservation_claim(
        llama_cache_acct_ledger * ledger, llama_cache_acct_op_id op) noexcept
    : ledger_(ledger), op_(op) {}

void llama_cache_reservation_claim::abort_if_live() noexcept {
    if (has_op()) {
        ledger_->abort(op_);
    }
    release();
}

llama_cache_reservation_claim::~llama_cache_reservation_claim() {
    abort_if_live();
}

llama_cache_reservation_claim::llama_cache_reservation_claim(
        llama_cache_reservation_claim && other) noexcept
    : ledger_(other.ledger_), op_(other.op_) {
    other.release();
}

llama_cache_reservation_claim & llama_cache_reservation_claim::operator=(
        llama_cache_reservation_claim && other) noexcept {
    if (this != &other) {
        abort_if_live();
        ledger_ = other.ledger_;
        op_     = other.op_;
        other.release();
    }
    return *this;
}

bool llama_cache_reservation_claim::commit(
        uint64_t logical_bytes,
        llama_cache_acct_op_id & committed_op) noexcept {
    committed_op = {};
    if (!has_op() || !ledger_->commit(op_, logical_bytes)) {
        return false;
    }
    committed_op = op_;
    release();
    return true;
}

llama_cache_admission_result llama_cache_admit_reservation(
        llama_cache_acct_ledger          & ledger,
        const llama_cache_budget_config  & budget_config,
        const llama_cache_authority_request & request) noexcept try {
    // 1. Coherent snapshot under one serial.
    llama_cache_acct_snapshot snap = ledger.snapshot();

    // 2. Fail-closed on incomplete evidence: an explicitly non-known manifest means the ledger
    //    cannot vouch for completeness, so refuse before pricing (never admit on private counters).
    if (snap.completeness_manifest != llama_cache_acct_known::known) {
        return { llama_cache_admission_status::incomplete_evidence, {} };
    }

    // 3. Local coordinator (reset() mutates it; a shared instance is unsafe across admissions).
    //    Move the potentially-large accounting snapshot into the one-shot coordinator now that
    //    this composer is on F0b's authority path.
    llama_cache_budget_coordinator coordinator;
    const uint64_t accounting_serial = snap.serial;
    if (!coordinator.reset(std::move(snap), budget_config)) {
        return { llama_cache_admission_status::budget_unavailable, {} };
    }

    // 4. Reserve-only plan priced at the snapshot's serial: one domain, no release credits.
    llama_cache_budget_plan plan;
    plan.accounting_serial = accounting_serial;
    plan.entries.push_back({ request.domain, request.expected_resident, /* release_bytes */ 0 });

    // 5. Price it.
    const llama_cache_budget_result fit = coordinator.fits(plan);
    switch (fit.state) {
        case llama_cache_budget_fit_state::fits:
            break;
        case llama_cache_budget_fit_state::exceeds:
            return { llama_cache_admission_status::exceeds_budget, {} };
        case llama_cache_budget_fit_state::unavailable:
        case llama_cache_budget_fit_state::_count:
        default:
            return { llama_cache_admission_status::budget_unavailable, {} };
    }

    // 6. Conditional reserve at the priced serial (single-shot: drift refuses, F0b re-drives).
    const llama_cache_conditional_reserve_result cr = ledger.reserve_if_serial(
        accounting_serial, request.category, request.domain, request.attribution,
        request.expected_logical, request.expected_resident);
    switch (cr.status) {
        case llama_cache_conditional_reserve_status::admitted:
            return { llama_cache_admission_status::admitted,
                     llama_cache_reservation_claim(&ledger, cr.op) };
        case llama_cache_conditional_reserve_status::serial_conflict:
            return { llama_cache_admission_status::serial_conflict, {} };
        case llama_cache_conditional_reserve_status::ledger_fault:
        case llama_cache_conditional_reserve_status::_count:
        default:
            return { llama_cache_admission_status::ledger_fault, {} };
    }
} catch (...) {
    // The only throwing step is plan.entries.push_back (the ledger/coordinator calls are noexcept);
    // a function-try-block turns any allocation failure into a typed fail-closed verdict, so no
    // exception ever crosses the authority boundary into F0b.
    return { llama_cache_admission_status::internal_fault, {} };
}
