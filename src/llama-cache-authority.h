#pragma once

// llama-cache-authority.h — P2 F0a admission composer (library side).
//
// The policy-free ledger (llama-cache-accounting) records; the budget coordinator
// (llama-cache-budget) prices. This unit is the single place that COMPOSES them into the
// authoritative admission sequence F flips on:
//
//     snapshot() → fits(reserve-only plan) → reserve_if_serial(snapshot.serial)
//
// It is reserve-only and single-shot by contract:
//   - reserve-only: the plan credits NO planned release. Crediting a release before F0b owns an
//     atomic release/mutate transaction would let a reservation be admitted against bytes that are
//     not yet actually free (false admission).
//   - single-shot: on serial drift it returns serial_conflict immediately rather than retrying with
//     a possibly-stale capacity config; F0b owns the bounded resample→resnapshot→fits→reserve retry.
//
// F0a wires this into NO production mutation path. It is exercised by unit tests only; F0b converts
// the destructive call sites and owns the mutation + stage/commit/abort that follows an admission.

#include "llama-cache-accounting.h"
#include "llama-cache-budget.h"

// One typed reservation request. Exactly one target domain; no release credits.
struct llama_cache_authority_request {
    llama_cache_acct_category        category = llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution     attribution;
    uint64_t                         expected_logical  = 0;
    uint64_t                         expected_resident = 0;
};

// Closed admission taxonomy. incomplete_evidence and budget_unavailable are kept distinct because
// fits() collapses every non-fit into `unavailable`; incomplete_evidence is claimed ONLY when the
// snapshot manifest is explicitly non-known (the fail-closed rule), never fabricated from an opaque
// coordinator refusal.
enum class llama_cache_admission_status : uint8_t {
    admitted,
    incomplete_evidence,
    budget_unavailable,
    exceeds_budget,
    serial_conflict,
    ledger_fault,
    internal_fault,     // the composer caught its own allocation/exception — fail-closed, never throws
    _count,
};

const char * llama_cache_admission_status_name(llama_cache_admission_status status) noexcept;

struct llama_cache_admission_result;

// Move-only handle to an admitted-but-not-yet-committed reservation. If it is destroyed while still
// holding a live op (an exception or early return before F0b's stage/commit), it aborts that op so a
// reserved operation can never leak (snapshot.live_ops would otherwise never return to zero). Only
// the composer mints an armed claim (constructor is private + friended), so no two claims can ever
// own the same op. F0a deliberately has NO disarm-without-abort path: an admitted claim ALWAYS
// auto-aborts. F0b adds a commit-through-claim terminal that disarms ONLY on a verified commit —
// exposing a bare "discharge" here would let a still-reserved (or committed-then-failed) op leak.
class llama_cache_reservation_claim {
public:
    llama_cache_reservation_claim() = default;
    ~llama_cache_reservation_claim();

    llama_cache_reservation_claim(const llama_cache_reservation_claim &)             = delete;
    llama_cache_reservation_claim & operator=(const llama_cache_reservation_claim &) = delete;
    llama_cache_reservation_claim(llama_cache_reservation_claim && other) noexcept;
    llama_cache_reservation_claim & operator=(llama_cache_reservation_claim && other) noexcept;

    bool                   has_op() const noexcept { return ledger_ != nullptr && op_.v != 0; }
    llama_cache_acct_op_id op()     const noexcept { return op_; }

private:
    llama_cache_reservation_claim(llama_cache_acct_ledger * ledger, llama_cache_acct_op_id op) noexcept;

    // Drop ownership without aborting; the single definition of "no longer holds an op".
    void release() noexcept { ledger_ = nullptr; op_ = {}; }
    void abort_if_live() noexcept;

    llama_cache_acct_ledger * ledger_ = nullptr;
    llama_cache_acct_op_id    op_     = {};

    friend llama_cache_admission_result llama_cache_admit_reservation(
            llama_cache_acct_ledger &, const llama_cache_budget_config &,
            const llama_cache_authority_request &) noexcept;
};

// The composer's result: a status plus, when admitted, the move-only claim that owns the reserved
// op until F0b's commit-through-claim terminal takes it over. Move-only (the claim is).
struct llama_cache_admission_result {
    llama_cache_admission_status  status = llama_cache_admission_status::ledger_fault;
    llama_cache_reservation_claim claim;
};

// Run the authoritative admission sequence once. `budget_config` is the caller's point-in-time
// capacity/config (the server samples physical capacity immediately before calling). A local
// coordinator is used because reset() mutates coordinator state — a shared mutable coordinator is
// not safe across concurrent admissions. noexcept: every failure — including its own allocation —
// becomes a typed status, so no exception ever crosses the authority boundary into F0b.
llama_cache_admission_result llama_cache_admit_reservation(
        llama_cache_acct_ledger          & ledger,
        const llama_cache_budget_config  & budget_config,
        const llama_cache_authority_request & request) noexcept;
