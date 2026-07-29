#pragma once

#include "common-checkpoint-shadow.h"      // A2 evaluation authority types (gen_eval subfields)
#include "../src/llama-cache-accounting.h" // staging API precedent: fit.cpp / speculative.cpp

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstdint>
#include <string>

// common-cache-plan.h — P2 B0/B shadow decision record, schema version 3.
//
// §7.7 decision records + §7.5 shadow-planner inventory: the ONE closed plan-reason enum
// shared by server and tests, the orthogonal candidate disposition, the closed provider
// inventory (today's real candidates only), and the multi-stage per-request record.
// SHADOW-ONLY: the record observes the shipped selection path; `slot.cache_status` and the
// live four-tier logic remain authoritative and untouched. Everything here is inert unless
// the --cache-debug observer is enabled, and the disabled branch performs strictly zero
// observer work (B-a).
//
// v1 → v2 (B pins, CONCUR-TERMINAL r4): the four per-provider summary rows are replaced by
// a bounded per-entry CANDIDATE INVENTORY — one row per candidate instance the shipped
// selectors actually visited (every evaluated live slot across the three slot loops, every
// scanned host entry, every visited checkpoint sibling), merged across selector phases by
// (provider, request-local source id). The declared candidate domain is exactly this
// shipped-visited set (B-a forbids observer re-scans); per-provider inventory-state markers
// record truncation (shipped short-circuit) and overflow. Cost terms move from the record
// to each row; shadow choice / tie set are planner outputs, typed-unavailable until the
// B chooser fills them. Candidate observation transport is noexcept by construction: fixed
// capacity in the record, append-or-mark-overflowed, no allocation in selector hooks.

constexpr uint32_t COMMON_CACHE_PLAN_SCHEMA_VERSION = 3;

// Explicit record→embedded-accounting compatibility table. A C schema bump cannot compile
// under the current record version until this table and the record version move together.
constexpr uint32_t common_cache_plan_accounting_schema(uint32_t record_schema) {
    return record_schema == 3 ? 2 :
           (record_schema == 1 || record_schema == 2 ? 1 : 0);
}
static_assert(common_cache_plan_accounting_schema(COMMON_CACHE_PLAN_SCHEMA_VERSION) ==
              LLAMA_CACHE_ACCT_SCHEMA_VERSION);

// Bounded inventory capacity (fixed-in-record option of the A2 transport contract). Sized
// for worst realistic scan breadth (parallel slots + host-cache entries + checkpoint
// siblings + derived chain rows); exhaustion is NOT an error: the overflow marker latches,
// planner completeness goes unavailable, the B0 record and shipped path are untouched.
constexpr size_t COMMON_CACHE_PLAN_MAX_CANDIDATES = 96;

// Bounded component references for composed candidate plans (host entry + checkpoint
// continuation today). A chain row references its components by inventory ordinal.
constexpr size_t COMMON_CACHE_PLAN_MAX_COMPONENTS = 2;

// Reserved source ids (the merge key is (provider, source_id); real sources are >= 0):
// an AGGREGATE row carries a provider-level classification when no entries were scanned;
// a CHAIN row is the derived composed plan over selected component rows.
constexpr int32_t COMMON_CACHE_PLAN_SOURCE_AGGREGATE = -1;
constexpr int32_t COMMON_CACHE_PLAN_SOURCE_CHAIN     = -2;

// Fixed §7.7 precedence encoded in the VALUES: validity before economics, band order
// identity(100) → structural(200) → generation/lineage(300) → domain/tier(400) → budget(500)
// → cost(600). The first failing check in that order IS the reason; out-of-order observation
// keeps the lowest value (see note_reject). Values are stable and append-only within a band
// for this schema version.
//
// ONE X-macro list mechanically binds membership, wire value, and name-table spelling:
// adding a member is one line, and an omission anywhere is a compile failure (the name
// switch stays a real switch, so -Wswitch exhaustiveness is preserved). Each name string
// below is its ONLY spelling in the tree — CI extracts this list and bans replicas.
#define COMMON_CACHE_PLAN_REASON_LIST(X) \
    X(NONE,                          "none",                          0)   \
    /* identity */ \
    X(MODEL_IDENTITY_MISMATCH,       "model_identity_mismatch",       100) \
    X(EXECUTION_IDENTITY_MISMATCH,   "execution_identity_mismatch",   101) \
    X(ADAPTER_CONFIG_MISMATCH,       "adapter_config_mismatch",       102) \
    X(MEDIA_CONTENT_MISMATCH,        "media_content_mismatch",        103) \
    X(TOKENIZER_TEMPLATE_MISMATCH,   "tokenizer_template_mismatch",   104) \
    X(PREFIX_TOKEN_DIGEST_MISMATCH,  "prefix_token_digest_mismatch",  105) \
    /* structural */ \
    X(PROVIDER_UNAVAILABLE,          "provider_unavailable",          200) \
    X(PROVIDER_BUSY,                 "provider_busy",                 201) \
    X(FRONTIER_INVALID,              "frontier_invalid",              202) \
    X(COVERAGE_INSUFFICIENT,         "coverage_insufficient",         203) \
    X(PAYLOAD_EMPTY,                 "payload_empty",                 204) \
    X(PAYLOAD_INCOMPLETE,            "payload_incomplete",            205) \
    X(PAYLOAD_SHORT,                 "payload_short",                 206) \
    X(CHECKSUM_MISMATCH,             "checksum_mismatch",             207) \
    X(PAYLOAD_VERSION_UNSUPPORTED,   "payload_version_unsupported",   208) \
    X(COMPONENT_SHAPE_MISMATCH,      "component_shape_mismatch",      209) \
    X(ACCELERATOR_UNRESTORABLE,      "accelerator_unrestorable",      210) \
    /* generation / lineage — the A-track evaluator stays the ONE authority; its closed
       category/reason/tombstone/refinement ride as subfields, never as mirrored values */ \
    X(REPRESENTATION_EPOCH_CHANGED,  "representation_epoch_changed",  300) \
    X(SEQUENCE_EPOCH_CHANGED,        "sequence_epoch_changed",        301) \
    X(GENERATION_NOT_ELIGIBLE,       "generation_not_eligible",       302) \
    /* domain / tier */ \
    X(STORAGE_DOMAIN_MISMATCH,       "storage_domain_mismatch",       400) \
    X(REPRESENTATION_TIER_UNSUPPORTED, "representation_tier_unsupported", 401) \
    X(KV_TYPE_MISMATCH,              "kv_type_mismatch",              402) \
    X(SHARD_TOPOLOGY_MISMATCH,       "shard_topology_mismatch",       403) \
    X(RECOVERABILITY_UNSUPPORTED,    "recoverability_unsupported",    404) \
    /* budget */ \
    X(PERSISTENT_BUDGET_EXCEEDED,    "persistent_budget_exceeded",    500) \
    X(WORKSPACE_BUDGET_EXCEEDED,     "workspace_budget_exceeded",     501) \
    X(MANDATORY_ANCHOR_OVERFLOW,     "mandatory_anchor_overflow",     502) \
    /* cost — the only rejection of a VALID candidate (it lost the economics) */ \
    X(COST_NOT_MINIMAL,              "cost_not_minimal",              600)

enum common_cache_plan_reason : uint16_t {
#define COMMON_CACHE_PLAN_REASON_ENUM_MEMBER(sym, name, val) COMMON_CACHE_PLAN_REASON_##sym = val,
    COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_ENUM_MEMBER)
#undef COMMON_CACHE_PLAN_REASON_ENUM_MEMBER
    // closed-set sentinel (v2 pin): one past the last member
    COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL,
};

// Every member, generated from the same list — cannot drift from the enum.
constexpr common_cache_plan_reason common_cache_plan_reason_all[] = {
#define COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER(sym, name, val) COMMON_CACHE_PLAN_REASON_##sym,
    COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER)
#undef COMMON_CACHE_PLAN_REASON_ARRAY_MEMBER
};

constexpr size_t COMMON_CACHE_PLAN_REASON_MEMBER_COUNT =
    sizeof(common_cache_plan_reason_all) / sizeof(common_cache_plan_reason_all[0]);
static_assert(uint16_t(COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL) == 601,
              "sentinel is one past cost_not_minimal for schema version 1");

// band = value / 100
constexpr uint16_t common_cache_plan_reason_band(common_cache_plan_reason r) {
    return uint16_t(r) / 100;
}

constexpr bool common_cache_plan_reasons_monotone() {
    for (size_t i = 1; i < COMMON_CACHE_PLAN_REASON_MEMBER_COUNT; i++) {
        if (uint16_t(common_cache_plan_reason_all[i]) <=
            uint16_t(common_cache_plan_reason_all[i - 1])) {
            return false;
        }
    }
    return true;
}
static_assert(common_cache_plan_reasons_monotone(),
              "plan-reason values must be strictly increasing (bands encode precedence)");
// exact band starts are schema-stable wire values; a move is a breaking change
static_assert(COMMON_CACHE_PLAN_REASON_MODEL_IDENTITY_MISMATCH      == 100 &&
              COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE         == 200 &&
              COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED == 300 &&
              COMMON_CACHE_PLAN_REASON_STORAGE_DOMAIN_MISMATCH      == 400 &&
              COMMON_CACHE_PLAN_REASON_PERSISTENT_BUDGET_EXCEEDED   == 500 &&
              COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL             == 600,
              "band starts pinned at 100/200/300/400/500/600");

// Orthogonal candidate disposition: a valid loser is not an invalid candidate.
enum class common_cache_plan_disposition : uint8_t {
    accepted = 0,
    rejected_invalid,
    valid_not_chosen_cost,
    unavailable,
    _count,
};

// Closed provider inventory — today's real request-level candidates ONLY. `seq_cp` is a
// capability (copy_state_to's internal primitive), reported as a constant on the emitted
// record, not a candidate; device/disk/remote tiers and the parked rolling tape enter only
// when their providers exist.
enum class common_cache_plan_provider : uint8_t {
    live_slot = 0,
    live_context_checkpoint,
    host_cache_entry,
    cold_replay,
    _count,
};

enum class common_cache_plan_outcome : uint8_t {
    unknown = 0,        // the typed not-finalized state
    restored,
    restore_failed_fell_back_cold,
    cold,
    _count,
};

// How the shipped path picked the slot (stage 1).
enum class common_cache_plan_selection : uint8_t {
    none = 0,
    by_id,
    similarity,
    route_home,
    lru,
    _count,
};

// Which authoritative shipped scan observed a candidate (bitmask on the row). A physical
// candidate visited by several phases keeps ONE row (merge key = provider + source id);
// each phase ORs its bit and adds only the scalars that phase computed (r3 reading 1).
enum common_cache_plan_phase : uint8_t {
    COMMON_CACHE_PLAN_PHASE_BY_ID      = 1 << 0,
    COMMON_CACHE_PLAN_PHASE_SIMILARITY = 1 << 1,
    COMMON_CACHE_PLAN_PHASE_ROUTE_HOME = 1 << 2,
    COMMON_CACHE_PLAN_PHASE_LRU        = 1 << 3,
    COMMON_CACHE_PLAN_PHASE_HOST_SCAN  = 1 << 4,
    COMMON_CACHE_PLAN_PHASE_CKPT_SCAN  = 1 << 5,
    COMMON_CACHE_PLAN_PHASE_CHAIN      = 1 << 6,   // derived composed-plan row
};

// Per-provider observed-inventory completeness over the DECLARED domain (= the shipped-
// visited set). `truncated_by_shipped_short_circuit` marks scans the shipped path cut off
// (checkpoint reverse find_if): entries beyond it are outside the domain, and such a record
// is scoped evidence only — never full-inventory absorption evidence (r3 reading 2).
// `overflowed` = the fixed inventory filled; shadow choice is then unavailable, never an
// optimum over a partial set.
enum class common_cache_plan_inventory_state : uint8_t {
    unobserved = 0,
    complete,
    truncated_by_shipped_short_circuit,
    overflowed,
    _count,
};

// Closed planner-attempt status (verify-r1 finding 8): every finalized record says exactly
// what the planner did, and an ordinary refusal is countable without conflating "no fitted
// profile exists" with an internal fault.
enum class common_cache_plan_planner_status : uint8_t {
    not_attempted = 0,      // record finalized before the planner stage (should not emit)
    ok,                     // matched profile, complete evidence, shadow choice computed
    no_profile,             // the server composed no calibration profile
    profile_unfitted,       // profile composed but no fitted table entry exists
    invalid_calibration,    // profile mismatch, unreviewed version, or non-finite/negative coefficients
    incomplete_evidence,    // overflow / unresolved candidate / missing scalars — never a partial optimum
    internal_fault,         // exception inside the planner boundary
    _count,
};

// Opaque identity evidence (§7.7 redaction: digests of already-computed keys/strings, never
// raw values). An identity the server has not computed stays typed unknown — never a
// fabricated digest.
struct common_cache_plan_identity_evidence {
    llama_cache_acct_value model_digest;
    llama_cache_acct_value execution_digest;
    llama_cache_acct_value adapter_config_digest;
    llama_cache_acct_value media_content_digest;
    llama_cache_acct_value tokenizer_template_digest;
    llama_cache_acct_value prefix_token_digest;
};

// §7.5 cost-term slots: one per kind with its canonical raw unit, unavailable until an
// estimator fills them (a default array would collapse to five "restore" slots).
constexpr std::array<llama_cache_acct_cost_term, size_t(llama_cache_acct_cost_kind::_count)>
common_cache_plan_default_cost_terms() {
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> terms{};
    for (size_t i = 0; i < terms.size(); i++) {
        terms[i].kind     = llama_cache_acct_cost_kind(i);
        terms[i].raw_unit = llama_cache_acct_cost_kind_unit(terms[i].kind);
    }
    return terms;
}

// One candidate-plan row: a candidate instance the shipped path actually visited (or a
// derived chain over such instances). Membership in the inventory IS presence — no row, no
// observation, never a vacuous verdict. `delivered` = this candidate actually applied state
// to the slot — recorded as data at the delivery site, never inferred. gen_eval transports
// the A2 evaluator's own typed result (`evaluated == false` = evaluator did not run).
// Trivially copyable and written only through noexcept record methods (A2 transport).
struct common_cache_plan_candidate {
    common_cache_plan_provider provider = common_cache_plan_provider::cold_replay;
    // request-local source identity (merge key with provider): live slot id, host-entry scan
    // ordinal, checkpoint scan ordinal; -1 for cold_replay and derived rows
    int32_t source_id   = -1;
    uint8_t phases_seen = 0;    // OR of common_cache_plan_phase bits

    bool delivered = false;

    // root-feasibility (verify-r1 finding 1): a row whose state was only reachable through
    // a delivered base component (e.g. a checkpoint exposed by a host restore) is EVIDENCE
    // and a chain component, but not a standalone plan — it never enters the root optimum.
    bool component_only = false;

    common_cache_plan_disposition disposition = common_cache_plan_disposition::unavailable;
    common_cache_plan_reason      reason      = COMMON_CACHE_PLAN_REASON_NONE;

    // diagnostics transported from the shipped path's own computation (B-a: never re-derived;
    // a scalar the visiting loop did not compute stays typed unknown)
    llama_cache_acct_value lcp_tokens;
    llama_cache_acct_value payload_bytes;
    llama_cache_acct_value t_last_used_us;                  // LRU loop recency
    double sim        = 0; bool sim_known          = false;
    double f_keep     = 0; bool f_keep_known       = false;
    bool spec_capable = false; bool spec_capable_known = false;
    // live_context_checkpoint: rows the short-circuiting shipped scan actually VISITED
    // (not the container size) + how many it rejected for a changed representation epoch
    uint32_t siblings_rejected_epoch = 0;
    uint32_t siblings_scanned        = 0;

    common_checkpoint_shadow_evaluation gen_eval;

    // composed plan: ordinals of this chain's component rows (-1 = unused slot); only rows
    // with the CHAIN phase bit use these
    std::array<int32_t, COMMON_CACHE_PLAN_MAX_COMPONENTS> component_ids = {-1, -1};

    // per-candidate §7.5 economics: filled by the B estimator inside the planner boundary;
    // typed-unavailable in transport
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> cost_terms =
        common_cache_plan_default_cost_terms();
    llama_cache_acct_value predicted_total_us;

    bool is_chain() const noexcept { return (phases_seen & COMMON_CACHE_PLAN_PHASE_CHAIN) != 0; }

    // the shipped winner's promotion over the scan-time cost-loser default — one invariant
    // pair, so no site can reset the disposition but forget the reason
    void accept() noexcept {
        disposition = common_cache_plan_disposition::accepted;
        reason      = COMMON_CACHE_PLAN_REASON_NONE;
    }

    // First-failing-check discipline for out-of-order observation: the lowest value (earliest
    // precedence band) is THE reason; later/higher failures do not overwrite it.
    void note_reject(common_cache_plan_reason r) noexcept {
        if (r == COMMON_CACHE_PLAN_REASON_NONE) {
            return;
        }
        if (reason == COMMON_CACHE_PLAN_REASON_NONE || uint16_t(r) < uint16_t(reason)) {
            reason = r;
        }
        // disposition follows the retained (earliest-band) reason, not the arrival order
        disposition = (reason == COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL)
            ? common_cache_plan_disposition::valid_not_chosen_cost
            : common_cache_plan_disposition::rejected_invalid;
    }
};

// Multi-stage per-request record. The shipped path selects and mutates across three stages
// (slot routing → host-cache load → context-checkpoint selection); candidate rows
// accumulate at each stage and the record is finalized exactly once (outcome flips off
// `unknown`), after the actual restore/cold path and measured TTFT are known. Fields no
// shipped computation produced remain typed unknown/unavailable — budget deltas and
// would-evict lists are D work and stay unavailable in B.
//
// A2 transport contract (pins v4, verbatim contract): this base record exists independently
// of planner outputs; the inventory is FIXED-CAPACITY in the record, so every selector hook
// is allocation-free and noexcept — append either succeeds into reserved storage or latches
// the provider's inventory state `overflowed` without touching the shipped loop. The
// planner (estimation, composed-plan construction, tie set, shadow choice) runs later,
// inside its own boundary in finalize, and its failure clears planner outputs only: the B0
// record is always emitted.
struct common_cache_plan_record {
    uint32_t schema_version = COMMON_CACHE_PLAN_SCHEMA_VERSION;

    int64_t id_task = -1;   // request/decision id
    int32_t id_slot = -1;

    common_cache_plan_selection selection = common_cache_plan_selection::none;
    double sim_best_any = 0; bool sim_best_any_known = false;

    common_cache_plan_identity_evidence identity;

    // B-2: stable calibration-profile id ({model class}/{hardware class}/{batch regime});
    // empty = no matching profile (typed unknown on the wire) — estimators then refuse.
    // Set once at record creation (inside the creation try), never from selector hooks.
    std::string calibration_profile;

    // ---- candidate inventory (declared domain = shipped-visited set) ----
    std::array<common_cache_plan_candidate, COMMON_CACHE_PLAN_MAX_CANDIDATES> inventory{};
    uint32_t n_inventory = 0;
    std::array<common_cache_plan_inventory_state,
               size_t(common_cache_plan_provider::_count)> inventory_states{};   // unobserved

    // the shipped path's selected row per provider (inventory ordinal, -1 = none) — delivery
    // marking, revocation, and the delivered chain operate on selected rows
    std::array<int32_t, size_t(common_cache_plan_provider::_count)> selected = {-1, -1, -1, -1};

    common_cache_plan_provider chosen  = common_cache_plan_provider::cold_replay;
    common_cache_plan_outcome  outcome = common_cache_plan_outcome::unknown; // != unknown ⇔ finalized

    // the COMPLETE shipped plan as a candidate ordinal (verify-r1 finding 1): the chain row
    // when the delivery was composed, else the terminal provider's selected row. `chosen`
    // stays the outcome summary; offline agreement runs against THIS ordinal.
    int32_t shipped_plan_candidate = -1;

    // a derived plan (chain) could not be recorded at capacity (verify-r1 finding 4): the
    // plan set is incomplete even though every provider inventory looks intact — the
    // planner must refuse.
    bool derived_plans_incomplete = false;

    // closed planner-attempt outcome, set at finalize (verify-r1 finding 8)
    common_cache_plan_planner_status planner_status = common_cache_plan_planner_status::not_attempted;

    // measured actuals (never estimates)
    llama_cache_acct_value n_prompt_tokens;
    llama_cache_acct_value n_reused_tokens;
    llama_cache_acct_value n_replayed_tokens;
    llama_cache_acct_value ttft_us;

    // a restore was attempted and failed (drives restore_failed_fell_back_cold at finalize
    // when nothing else delivered)
    bool restore_attempt_failed = false;

    // ---- planner outputs (B chooser; unavailable until it runs, cleared on planner fault) ----
    int32_t  shadow_choice = -1;                       // inventory ordinal; -1 = unavailable
    std::array<int32_t, COMMON_CACHE_PLAN_MAX_CANDIDATES> shadow_tie_set = {};   // valid [0, n_shadow_ties)
    uint32_t n_shadow_ties = 0;

    // C0 accounting snapshot; meaningful once outcome != unknown
    llama_cache_acct_snapshot acct;

    // Find the row for (provider, source_id) or append one — the cross-phase merge point:
    // one row per physical candidate, each visiting phase ORs its bit and adds its scalars.
    // noexcept by construction: fixed storage, linear scan over n_inventory (O(visited)).
    // nullptr = capacity exhausted; the provider's inventory latches `overflowed`, planner
    // completeness dies, the caller (a shipped-path hook) just skips.
    common_cache_plan_candidate * find_or_add(common_cache_plan_provider provider,
                                              int32_t source_id, uint8_t phase_bit) noexcept {
        for (uint32_t i = 0; i < n_inventory; i++) {
            if (inventory[i].provider == provider && inventory[i].source_id == source_id) {
                inventory[i].phases_seen |= phase_bit;
                return &inventory[i];
            }
        }
        if (n_inventory >= COMMON_CACHE_PLAN_MAX_CANDIDATES) {
            inventory_states[size_t(provider)] = common_cache_plan_inventory_state::overflowed;
            return nullptr;
        }
        auto & c = inventory[n_inventory++];
        c.provider    = provider;
        c.source_id   = source_id;
        c.phases_seen = phase_bit;
        // first observation of this provider upgrades unobserved → complete; truncation and
        // overflow are latched explicitly by the hooks that detect them and never downgrade
        auto & st = inventory_states[size_t(provider)];
        if (st == common_cache_plan_inventory_state::unobserved) {
            st = common_cache_plan_inventory_state::complete;
        }
        return &c;
    }

    // provider-level inventory verdicts from the hooks that KNOW the scan's shape. A
    // completed scan with zero rows is still an observation (complete, empty domain);
    // truncation records a shipped short-circuit; overflow (set by find_or_add) never
    // downgrades. noexcept.
    void note_inventory_complete(common_cache_plan_provider provider) noexcept {
        auto & st = inventory_states[size_t(provider)];
        if (st == common_cache_plan_inventory_state::unobserved) {
            st = common_cache_plan_inventory_state::complete;
        }
    }
    void note_inventory_truncated(common_cache_plan_provider provider) noexcept {
        auto & st = inventory_states[size_t(provider)];
        if (st != common_cache_plan_inventory_state::overflowed) {
            st = common_cache_plan_inventory_state::truncated_by_shipped_short_circuit;
        }
    }

    // the shipped path's selected candidate for a provider (nullptr = none selected)
    common_cache_plan_candidate * selected_row(common_cache_plan_provider provider) noexcept {
        const int32_t i = selected[size_t(provider)];
        return i >= 0 && uint32_t(i) < n_inventory ? &inventory[size_t(i)] : nullptr;
    }

    // Append a derived CHAIN row (composed plan over component ordinals). Distinct from
    // find_or_add: a failed append is silently skipped WITHOUT latching any provider's
    // inventory state — a derived-row capacity miss must never poison a real provider's
    // completeness (that would make the estimator refuse the whole record). noexcept.
    common_cache_plan_candidate * add_chain(common_cache_plan_provider base_provider,
                                            int32_t comp0, int32_t comp1) noexcept {
        if (n_inventory >= COMMON_CACHE_PLAN_MAX_CANDIDATES) {
            derived_plans_incomplete = true; // plan set incomplete; planner refuses [F4]
            return nullptr;
        }
        auto & c = inventory[n_inventory++];
        c.provider          = base_provider;
        c.source_id         = COMMON_CACHE_PLAN_SOURCE_CHAIN;
        c.phases_seen       = COMMON_CACHE_PLAN_PHASE_CHAIN;
        c.component_ids[0]  = comp0;
        c.component_ids[1]  = comp1;
        return &c;
    }
    void select(common_cache_plan_provider provider, const common_cache_plan_candidate * c) noexcept {
        selected[size_t(provider)] = c ? int32_t(c - inventory.data()) : -1;
    }

    // A fallback that invalidates already-installed provider state revokes its deliveries:
    // failure/fallback dominates historical delivery — cold is a FINAL-STATE fact, and a
    // restore whose bytes were later discarded did not deliver. noexcept.
    void revoke_deliveries() noexcept {
        for (uint32_t i = 0; i < n_inventory; i++) {
            if (inventory[i].provider != common_cache_plan_provider::cold_replay) {
                inventory[i].delivered = false;
            }
        }
    }

    // planner-fault cleanup (A2): clear every planner output, leave the B0 evidence intact
    void clear_planner_outputs() noexcept {
        shadow_choice = -1;
        n_shadow_ties = 0;
        for (uint32_t i = 0; i < n_inventory; i++) {
            inventory[i].cost_terms         = common_cache_plan_default_cost_terms();
            inventory[i].predicted_total_us = {};
        }
    }
};

// Exhaustive name tables (presentation layer; switch-based so -Wswitch enforces coverage).
// The C0 accounting enums are named here too: src/llama-cache-accounting.h stays policy- and
// string-free, and these are the only spellings — CI bans replicas.
const char * common_cache_plan_reason_name(common_cache_plan_reason r);
const char * common_cache_plan_disposition_name(common_cache_plan_disposition d);
const char * common_cache_plan_provider_name(common_cache_plan_provider p);
const char * common_cache_plan_outcome_name(common_cache_plan_outcome o);
const char * common_cache_plan_selection_name(common_cache_plan_selection s);
// Finalize-time chain composition (one testable implementation; the server calls this
// after `chosen`/`selected`/deliveries settle). Sets shipped_plan_candidate to the
// complete shipped plan: selected[chosen], upgraded to the delivered host→checkpoint
// chain on composed deliveries. When the host entry delivered, EVERY checkpoint sibling
// becomes component-only (the scanned list arrived with the host restore) and each valid
// sibling gains its true complete plan as a cost-loser chain; a chain dropped at capacity
// latches derived_plans_incomplete, and a composed delivery whose own chain could not be
// recorded reports NO shipped-plan ordinal (-1) rather than the infeasible bare row.
void common_cache_plan_compose_chains(common_cache_plan_record & rec);

const char * common_cache_plan_inventory_state_name(common_cache_plan_inventory_state s);
const char * common_cache_plan_planner_status_name(common_cache_plan_planner_status s);

const char * common_cache_acct_category_name(llama_cache_acct_category c);
const char * common_cache_acct_residency_name(llama_cache_acct_residency r);
const char * common_cache_acct_domain_kind_name(llama_cache_acct_domain_kind k);
const char * common_cache_acct_producer_name(llama_cache_acct_producer p);
const char * common_cache_acct_measure_name(llama_cache_acct_measure m);
const char * common_cache_acct_known_name(llama_cache_acct_known k);
const char * common_cache_acct_unit_name(llama_cache_acct_unit u);
const char * common_cache_acct_cost_kind_name(llama_cache_acct_cost_kind k);

// One typed-known-value JSON shape shared by CACHE_PLAN and process-local
// observer siblings such as CACHE_BUDGET.
nlohmann::ordered_json common_cache_plan_value_json(
        const llama_cache_acct_value & value);

// One JSON shape for both B0 surfaces (the --cache-debug log line and /slots.cache_plan).
// Identities stay opaque by construction: no prompt bytes, no raw adapter/media identities —
// only closed-enum names, counts, and sizes. Only present rows and non-unknown accounting
// cells are emitted.
nlohmann::ordered_json common_cache_plan_record_json(const common_cache_plan_record & rec);
