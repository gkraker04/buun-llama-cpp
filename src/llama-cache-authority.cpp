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

const char * llama_cache_transaction_status_name(
        llama_cache_transaction_status status) noexcept {
    switch (status) {
        case llama_cache_transaction_status::committed:          return "committed";
        case llama_cache_transaction_status::invalid_argument:   return "invalid_argument";
        case llama_cache_transaction_status::admission_refused:  return "admission_refused";
        case llama_cache_transaction_status::after_admit_failed: return "after_admit_failed";
        case llama_cache_transaction_status::stage_failed:       return "stage_failed";
        case llama_cache_transaction_status::commit_failed:      return "commit_failed";
        case llama_cache_transaction_status::post_commit_fault:  return "post_commit_fault";
        case llama_cache_transaction_status::internal_fault:     return "internal_fault";
        case llama_cache_transaction_status::_count:             break;
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

llama_cache_transaction_result llama_cache_execute_reservation_transaction(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget_config,
        const std::vector<llama_cache_transaction_leaf> & leaves,
        const llama_cache_transaction_fault & fault,
        const llama_cache_transaction_after_admit & after_admit) noexcept {
    llama_cache_transaction_result out;
    try {
        if (leaves.empty() ||
            (after_admit.context != nullptr) !=
                (after_admit.run != nullptr)) {
            out.status = llama_cache_transaction_status::invalid_argument;
            return out;
        }
        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto & leaf = leaves[i];
            if (!leaf.committed_op ||
                (leaf.existing_allocation &&
                 leaf.reserve_resident != 0) ||
                (!leaf.existing_allocation &&
                 leaf.stage_resident != leaf.reserve_resident)) {
                out.status = llama_cache_transaction_status::invalid_argument;
                out.failed_leaf = i;
                return out;
            }
            for (size_t j = 0; j < i; ++j) {
                if (leaves[j].committed_op == leaf.committed_op ||
                    (leaf.allocation_out &&
                     leaves[j].allocation_out == leaf.allocation_out)) {
                    out.status =
                        llama_cache_transaction_status::invalid_argument;
                    out.failed_leaf = i;
                    return out;
                }
            }
        }

        std::vector<llama_cache_admission_result> admissions(
            leaves.size());
        static constexpr uint32_t MAX_ATTEMPTS = 3;
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_admission_status last =
                llama_cache_admission_status::serial_conflict;
            uint32_t attempts = 0;
            for (; attempts < MAX_ATTEMPTS; ++attempts) {
                const auto & leaf = leaves[i];
                llama_cache_authority_request request;
                request.category = leaf.category;
                request.domain = leaf.domain;
                request.attribution = leaf.attribution;
                request.expected_logical = leaf.expected_logical;
                request.expected_resident = leaf.reserve_resident;
                admissions[i] = llama_cache_admit_reservation(
                    ledger, budget_config, request);
                last = admissions[i].status;
                if (last !=
                        llama_cache_admission_status::serial_conflict) {
                    attempts++;
                    break;
                }
                out.serial_retries++;
            }
            if (last != llama_cache_admission_status::admitted) {
                out.status =
                    llama_cache_transaction_status::admission_refused;
                out.admission_status = last;
                out.failed_leaf = i;
                out.attempts = attempts;
                return out;
            }
        }

        if (after_admit.run &&
            !after_admit.run(after_admit.context)) {
            out.status =
                llama_cache_transaction_status::after_admit_failed;
            return out;
        }

        std::vector<llama_cache_acct_alloc_id> allocations(
            leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            if (fault.fail_stage_at == i) {
                out.status =
                    llama_cache_transaction_status::stage_failed;
                out.failed_leaf = i;
                return out;
            }
            allocations[i] = leaves[i].existing_allocation
                ? leaves[i].existing_allocation
                : ledger.new_alloc();
            // The ledger has no join-without-stage primitive. Re-staging an
            // existing immutable allocation therefore raises transient_peak
            // briefly even though no payload bytes move and durable charge
            // remains governed by first/last reference.
            if (!allocations[i] ||
                !ledger.stage(
                    admissions[i].claim.op(),
                    allocations[i],
                    leaves[i].stage_resident,
                    leaves[i].artifact,
                    leaves[i].content,
                    leaves[i].lineage)) {
                out.status =
                    llama_cache_transaction_status::stage_failed;
                out.failed_leaf = i;
                return out;
            }
        }

        std::vector<llama_cache_acct_op_id> committed;
        committed.reserve(leaves.size());
        struct rollback_guard {
            llama_cache_acct_ledger * ledger = nullptr;
            std::vector<llama_cache_acct_op_id> * operations = nullptr;
            uint64_t * rolled_back = nullptr;
            bool keep = false;

            void rollback() noexcept {
                if (keep || !ledger || !operations || !rolled_back) {
                    return;
                }
                for (const auto op : *operations) {
                    if (ledger->release(op)) {
                        (*rolled_back)++;
                    }
                }
                keep = true;
            }

            ~rollback_guard() {
                rollback();
            }
        } rollback {
            &ledger, &committed, &out.rolled_back, false,
        };

        for (size_t i = 0; i < leaves.size(); ++i) {
            if (fault.fail_commit_at == i) {
                out.status =
                    llama_cache_transaction_status::commit_failed;
                out.failed_leaf = i;
                rollback.rollback();
                return out;
            }
            llama_cache_acct_op_id operation;
            if (!admissions[i].claim.commit(
                    leaves[i].expected_logical, operation)) {
                out.status =
                    llama_cache_transaction_status::commit_failed;
                out.failed_leaf = i;
                rollback.rollback();
                return out;
            }
            committed.push_back(operation);
        }

        if (fault.fail_after_commit) {
            out.status =
                llama_cache_transaction_status::post_commit_fault;
            rollback.rollback();
            return out;
        }

        for (size_t i = 0; i < leaves.size(); ++i) {
            *leaves[i].committed_op = committed[i];
            if (leaves[i].allocation_out) {
                *leaves[i].allocation_out = allocations[i];
            }
        }
        rollback.keep = true;
        out.status = llama_cache_transaction_status::committed;
        out.admission_status = llama_cache_admission_status::admitted;
        return out;
    } catch (...) {
        out.status = llama_cache_transaction_status::internal_fault;
        return out;
    }
}
