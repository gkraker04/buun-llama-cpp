#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

// llama-cache-accounting.h — P2 C0 accounting contract, schema version 1.
//
// Policy-free, library-neutral accounting types shared by the server observer (B0), the D
// lease/lifecycle work, and the F artifact-transaction work. This header is the SOLE byte
// accounting interface: no consumer grows private byte counters (C/F freeze requirement 9).
// Presentation (name strings, JSON) lives above, in common/ and server adapters — never here.
//
// P2 SEMANTICS (C-a containment): the ledger is a SHADOW OBSERVER. `commit` records that the
// shipped publication boundary happened; `reserve` is observational, not an admission
// reservation. No ledger fault, allocation failure, overflow, or incomplete snapshot may
// alter a shipped mutation — every entry point is genuinely non-throwing (internal
// allocation failure latches a fault and returns failure) and faults become counters plus
// typed-unavailable cells. F's later enforcement commit deliberately flips this authority
// (reservation precedes mutation; publication waits on accounting) — that flip is an F
// decision, not made here.

constexpr uint32_t LLAMA_CACHE_ACCT_SCHEMA_VERSION = 1;

// Mutually-exclusive semantic LEAF categories. host_cache / snapshot_blob are DERIVED
// provider/artifact groupings over these leaves, never additive leaves themselves: a
// host-cache entry's aggregate size already contains its state data and checkpoint payloads
// (server_prompt_cache_state::size()), so summing aggregate and parts would double-count.
enum class llama_cache_acct_category : uint8_t {
    live_attention_state = 0,
    live_recurrent_state,
    recurrent_rollback_planes,
    full_snapshot_payload,          // serialized main/draft state data of a host-cache entry
    checkpoint_state_payload,       // context-checkpoint target+draft payload bytes
    typed_accelerator_payload,
    checkpoint_generation_page_metadata,
    checkpoint_generation_unit_metadata,
    live_generation_metadata,
    ownership_index_metadata,
    unit_version_payload,
    clean_stash_payload,
    artifact_descriptor_metadata,
    artifact_reference_metadata,
    transfer_staging,
    codec_workspace,
    pinned_preimage_ring,
    // The parked rolling-window tape: present and typed so its zero is OBSERVED, not implied.
    // This names the parked mechanism specifically — the live fixed speculative tape is NOT
    // this category and is never asserted zero through it.
    rolling_window_tape,
    container_overhead,             // only where overhead cannot be attributed more precisely
    _count,
};

enum class llama_cache_acct_residency : uint8_t {
    device = 0,
    pinned_host,
    pageable_host,
    disk,
    remote,
    not_applicable,
    _count,
};

// Four measures per (category, residency) cell. An observation that cannot be made stays a
// typed unknown/unavailable — zero always means a measured zero. `transient_peak` is the
// high-water mark of CONCURRENTLY staged bytes for the cell, not the largest single stage.
enum class llama_cache_acct_measure : uint8_t {
    logical_payload = 0,
    resident_allocated,
    reserved,
    transient_peak,
    _count,
};

enum class llama_cache_acct_known : uint8_t {
    known = 0,
    unknown,        // not yet observed / producer absent
    unavailable,    // observation attempted and failed (fault, overflow)
    _count,
};

struct llama_cache_acct_value {
    uint64_t               value = 0;
    llama_cache_acct_known state = llama_cache_acct_known::unknown;

    static llama_cache_acct_value measured(uint64_t v) {
        return { v, llama_cache_acct_known::known };
    }
};

// Raw work quantities are never silently converted between units; a cost term carries its
// raw quantity for auditability plus an optional comparable time estimate.
enum class llama_cache_acct_unit : uint8_t {
    bytes = 0,
    tokens,
    operations,
    _count,
};

enum class llama_cache_acct_cost_kind : uint8_t {
    restore = 0,
    replay,
    transfer,
    eviction,
    workspace,
    _count,
};

// Canonical raw unit per cost kind (Q-C2 ruling): replay is counted in tokens, everything
// else in bytes. The unit is schema metadata, not a measurement — it is valid even while the
// term itself is unavailable.
constexpr llama_cache_acct_unit llama_cache_acct_cost_kind_unit(llama_cache_acct_cost_kind k) {
    return k == llama_cache_acct_cost_kind::replay ? llama_cache_acct_unit::tokens
                                                   : llama_cache_acct_unit::bytes;
}

// §7.5 cost-term shape consumed by the B planner. `estimated_us` is a versioned estimate;
// measured time is a separate actual-outcome field on the consumer's record and is never
// substituted into the estimate. `estimator_version` is meaningful only while `estimated_us`
// is known.
struct llama_cache_acct_cost_term {
    llama_cache_acct_cost_kind kind = llama_cache_acct_cost_kind::restore;
    llama_cache_acct_value     raw;
    llama_cache_acct_unit      raw_unit = llama_cache_acct_unit::bytes;
    llama_cache_acct_value     estimated_us;    // unknown until B lands an estimator
    uint32_t                   estimator_version = 0;
};

// Attribution axes (C/F freeze requirement 8): a closed kind tag; server-wide rows use the
// defaults. The tenant axis is deliberately ABSENT from this schema version, not an empty
// field: the server has no tenant identity until E1 (adding it is a schema-version bump).
enum class llama_cache_acct_attr_kind : uint8_t {
    server = 0,
    slot,
    artifact,
    _count,
};

// Identity discipline (C/F freeze requirement 3): five DISTINCT identities that must never
// be interchanged. The operation and allocation ids are process-local accounting identities;
// the artifact identity, content digest, and eligibility lineage identity are opaque
// contract fields carried and retained by the transaction (F populates and validates them).
// Distinct wrapper types make interchange a compile error (matrix asserted below). The
// zero id is the "none" sentinel (vbr_operation_id idiom): `explicit operator bool` tests
// it; only op/alloc ids get std::hash — they alone key the ledger maps.
struct llama_cache_acct_op_id {
    uint64_t v = 0;
    explicit operator bool() const { return v != 0; }
};
struct llama_cache_acct_alloc_id {
    uint64_t v = 0;
    explicit operator bool() const { return v != 0; }
};
struct llama_cache_acct_artifact_id    { uint64_t v = 0; };
struct llama_cache_acct_content_digest { uint64_t v = 0; };
struct llama_cache_acct_lineage_id     { uint64_t v = 0; };

inline bool operator==(llama_cache_acct_op_id          a, llama_cache_acct_op_id          b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_alloc_id       a, llama_cache_acct_alloc_id       b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_artifact_id    a, llama_cache_acct_artifact_id    b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_content_digest a, llama_cache_acct_content_digest b) { return a.v == b.v; }
inline bool operator==(llama_cache_acct_lineage_id     a, llama_cache_acct_lineage_id     b) { return a.v == b.v; }
inline bool operator!=(llama_cache_acct_op_id          a, llama_cache_acct_op_id          b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_alloc_id       a, llama_cache_acct_alloc_id       b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_artifact_id    a, llama_cache_acct_artifact_id    b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_content_digest a, llama_cache_acct_content_digest b) { return !(a == b); }
inline bool operator!=(llama_cache_acct_lineage_id     a, llama_cache_acct_lineage_id     b) { return !(a == b); }

template <> struct std::hash<llama_cache_acct_op_id> {
    size_t operator()(const llama_cache_acct_op_id & id) const { return std::hash<uint64_t>{}(id.v); }
};
template <> struct std::hash<llama_cache_acct_alloc_id> {
    size_t operator()(const llama_cache_acct_alloc_id & id) const { return std::hash<uint64_t>{}(id.v); }
};

// The non-interchange proof, in the header so every consumer TU enforces it: no pair among
// the five identities (or a raw integer) converts either way. Aggregate `{n}` init stays
// legal — mints construct ids on purpose; only IMPLICIT interchange is banned.
template <typename A, typename B>
constexpr bool llama_cache_acct_ids_distinct =
    !std::is_convertible_v<A, B> && !std::is_convertible_v<B, A>;
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_op_id,          llama_cache_acct_alloc_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_op_id,          llama_cache_acct_artifact_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_op_id,          llama_cache_acct_content_digest>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_op_id,          llama_cache_acct_lineage_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_op_id,          uint64_t>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_alloc_id,       llama_cache_acct_artifact_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_alloc_id,       llama_cache_acct_content_digest>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_alloc_id,       llama_cache_acct_lineage_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_alloc_id,       uint64_t>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_artifact_id,    llama_cache_acct_content_digest>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_artifact_id,    llama_cache_acct_lineage_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_artifact_id,    uint64_t>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_content_digest, llama_cache_acct_lineage_id>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_content_digest, uint64_t>);
static_assert(llama_cache_acct_ids_distinct<llama_cache_acct_lineage_id,     uint64_t>);

struct llama_cache_acct_attribution {
    llama_cache_acct_attr_kind   kind    = llama_cache_acct_attr_kind::server;
    int32_t                      slot_id = -1;   // meaningful when kind == slot
    llama_cache_acct_artifact_id artifact;       // meaningful when kind == artifact
};

enum class llama_cache_acct_txn_state : uint8_t {
    reserved = 0,
    staged,
    committed,
    aborted,
    released,
    _count,
};

// Point-in-time gauge cell (durable byte state). Counters below are monotone; every field is
// one or the other, never both.
struct llama_cache_acct_cell {
    std::array<llama_cache_acct_value, size_t(llama_cache_acct_measure::_count)> measures;
};

// Normalized attributed row: one per live committed physical allocation. Server aggregates
// live in `cells`; slot/artifact attribution is read from these rows (an explicit normalized
// form — no private per-consumer counters).
struct llama_cache_acct_allocation_row {
    llama_cache_acct_alloc_id      alloc;
    llama_cache_acct_attribution   attribution;
    llama_cache_acct_category      category  = llama_cache_acct_category::container_overhead;
    llama_cache_acct_residency     residency = llama_cache_acct_residency::not_applicable;
    uint64_t                       logical_bytes  = 0;
    uint64_t                       resident_bytes = 0;
    uint32_t                       committed_refs = 0;
    llama_cache_acct_artifact_id    artifact_identity;
    llama_cache_acct_content_digest content_digest;
    llama_cache_acct_lineage_id     lineage_identity;
};

struct llama_cache_acct_snapshot {
    uint32_t               schema_version = LLAMA_CACHE_ACCT_SCHEMA_VERSION;
    uint64_t               serial         = 0;    // bumped on EVERY observable change, faults included
    llama_cache_acct_known completeness   = llama_cache_acct_known::unknown;
    std::array<std::array<llama_cache_acct_cell,
                          size_t(llama_cache_acct_residency::_count)>,
               size_t(llama_cache_acct_category::_count)> cells;
    std::vector<llama_cache_acct_allocation_row> allocations;
    // in-flight transaction count (reserved + staged + committed-unreleased): zero after a
    // producer's entries are fully destroyed — a leaked op is an accounting bug
    uint64_t live_ops = 0;
    // monotone fault counters
    uint64_t faults_invalid_transition = 0;
    uint64_t faults_overflow           = 0;
    uint64_t faults_unknown_id         = 0;
    uint64_t faults_allocation         = 0;   // internal ledger allocation failure (non-throwing contract)
};

// Shadow accounting ledger: reserve → stage → commit | abort → release, observational in P2
// (header preamble). Charge-once for shared immutable allocations: the durable bytes of a
// physical allocation are charged when its FIRST reference commits and discharged when its
// LAST reference releases; per-reference metadata is reported by the referrer under its own
// leaf (artifact_reference_metadata), outside this refcount. Allocation ids must come from
// new_alloc() (zero and unminted ids are faults) and an allocation's (category, residency,
// resident-size) tuple is immutable — a mismatched citation is a fault, never a silent
// merge. NON-THROWING: no method throws; internal failure latches faults_allocation.
struct llama_cache_acct_ledger {
    llama_cache_acct_ledger();

    // Mint a fresh physical-allocation id (one owner for the whole accounting id space).
    llama_cache_acct_alloc_id new_alloc();

    // Observational reservation: records the expected resident bytes under `reserved`,
    // returns the op id (zero id on internal failure). Never blocks or admits anything.
    llama_cache_acct_op_id reserve(
            llama_cache_acct_category      category,
            llama_cache_acct_residency     residency,
            llama_cache_acct_attribution   attribution,
            uint64_t                       expected_logical,
            uint64_t                       expected_resident);

    // Associate the op with a minted physical allocation and its actual resident size.
    // Validates the allocation tuple against any existing citation. Updates the concurrent
    // staged high-water mark. The three opaque identities are retained on the transaction
    // (F populates them; empty is valid in P2). False on any fault.
    bool stage(llama_cache_acct_op_id op, llama_cache_acct_alloc_id alloc, uint64_t resident_bytes,
               llama_cache_acct_artifact_id    artifact = {},
               llama_cache_acct_content_digest digest   = {},
               llama_cache_acct_lineage_id     lineage  = {});

    // Record the shipped publication boundary. First committed reference of an allocation
    // charges its durable bytes; later references must cite the same logical size and only
    // join the refcount.
    bool commit(llama_cache_acct_op_id op, uint64_t logical_bytes);

    // Zero durable gauge delta; the observed transient peak is retained. The op is erased.
    bool abort(llama_cache_acct_op_id op);

    // Drop the op's reference; discharges durable bytes when the allocation loses its last
    // reference. Exactly-once per reference — a second release is a fault.
    bool release(llama_cache_acct_op_id op);

    // Direct gauge reporting for non-transactional producers (live state, metadata gauges).
    // Checked: overflow latches the cell unavailable and counts a fault.
    void gauge_set(llama_cache_acct_category category,
                   llama_cache_acct_residency residency,
                   llama_cache_acct_measure measure,
                   uint64_t value);

    // A producer whose own observation failed (e.g. checked-sum overflow) latches the cell
    // unavailable instead of reporting a fabricated value.
    void mark_unavailable(llama_cache_acct_category category,
                          llama_cache_acct_residency residency,
                          llama_cache_acct_measure measure);

    // Coherent copy of the observable state under one serial (gauges + normalized
    // allocation rows + fault counters). On internal copy failure the returned snapshot has
    // completeness == unavailable and no rows.
    llama_cache_acct_snapshot snapshot();

private:
    struct txn {
        llama_cache_acct_txn_state   state = llama_cache_acct_txn_state::reserved;
        llama_cache_acct_category    category  = llama_cache_acct_category::container_overhead;
        llama_cache_acct_residency   residency = llama_cache_acct_residency::not_applicable;
        llama_cache_acct_attribution attribution;
        llama_cache_acct_alloc_id    alloc;
        uint64_t                     reserved_bytes = 0; // charged at reserve, unwound by commit/abort
        uint64_t                     resident_bytes = 0; // actual, set at stage
        llama_cache_acct_artifact_id    artifact;
        llama_cache_acct_content_digest digest;
        llama_cache_acct_lineage_id     lineage;
    };

    // Allocation lifecycle: MINTED (registry entry created by new_alloc) → LIVE (first stage
    // fixes the complete immutable citation tuple) → RETIRED (last committed reference
    // released; the entry survives as a tombstone so a retired id can never resurrect as a
    // different physical allocation). One entry per id ever minted — bounded by allocation
    // churn, acceptable for the shadow ledger; F revisits lifecycle compaction.
    struct alloc_entry {
        bool     tuple_set      = false;
        bool     ever_committed = false;
        bool     retired        = false;
        uint32_t staged_refs    = 0;
        uint32_t committed_refs = 0;
        // immutable citation tuple, fixed by the first stage — ALL fields compared on every
        // shared citation (identity fields included)
        llama_cache_acct_category    category  = llama_cache_acct_category::container_overhead;
        llama_cache_acct_residency   residency = llama_cache_acct_residency::not_applicable;
        uint64_t                     resident_bytes = 0;
        uint64_t                     charged_logical = 0; // set by the first commit
        llama_cache_acct_attribution attribution;         // first committer's
        llama_cache_acct_artifact_id    artifact;
        llama_cache_acct_content_digest digest;
        llama_cache_acct_lineage_id     lineage;
    };

    // unlocked internal latch (callers hold the mutex)
    void cell_latch_unavailable(llama_cache_acct_category c, llama_cache_acct_residency r,
                                llama_cache_acct_measure m);
    // checked-add/sub on a cell measure; latches unavailable + fault on overflow/underflow
    void cell_add(llama_cache_acct_category c, llama_cache_acct_residency r,
                  llama_cache_acct_measure m, uint64_t v);
    void cell_sub(llama_cache_acct_category c, llama_cache_acct_residency r,
                  llama_cache_acct_measure m, uint64_t v);
    // concurrent-staged tracking: staged_now +=/-= v, peak = max(peak, staged_now)
    void staged_add(llama_cache_acct_category c, llama_cache_acct_residency r, uint64_t v);
    void staged_sub(llama_cache_acct_category c, llama_cache_acct_residency r, uint64_t v);
    void bump_serial();
    // retirement accounts for BOTH claim kinds: an allocation that ever committed retires
    // only when its last committed AND last staged reference are gone (a staged op holds a
    // valid claim that may still commit)
    void maybe_retire(alloc_entry & entry);

    mutable std::mutex mtx;
    llama_cache_acct_snapshot state;    // durable gauges + serial + faults live here (rows built on demand)
    std::array<std::array<uint64_t, size_t(llama_cache_acct_residency::_count)>,
               size_t(llama_cache_acct_category::_count)> staged_now = {};
    llama_cache_acct_op_id    next_op       = {1};
    llama_cache_acct_alloc_id next_alloc_id = {1};
    std::unordered_map<llama_cache_acct_op_id, txn>            ops;
    std::unordered_map<llama_cache_acct_alloc_id, alloc_entry> allocs;
};
