#pragma once

#include "common-checkpoint-shadow.h"      // A2 evaluation authority types (gen_eval subfields)
#include "../src/llama-cache-accounting.h" // staging API precedent: fit.cpp / speculative.cpp

#include <nlohmann/json_fwd.hpp>

#include <array>
#include <cstdint>

// common-cache-plan.h — P2 B0 shadow decision record, schema version 1.
//
// §7.7 decision records: the ONE closed plan-reason enum shared by server and tests, the
// orthogonal candidate disposition, the closed provider inventory (today's real candidates
// only), and the multi-stage per-request record. SHADOW-ONLY: the record observes the shipped
// selection path; `slot.cache_status` and the live four-tier logic remain authoritative and
// untouched. Everything here is inert unless the --cache-debug observer is enabled, and the
// disabled branch performs strictly zero observer work (B-a).

constexpr uint32_t COMMON_CACHE_PLAN_SCHEMA_VERSION = 1;

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

// One provider's observed row. `present` = a shipped stage observed this provider (no row =
// no observation, never a vacuous verdict). `delivered` = the provider actually applied state
// to the slot — recorded as data at the delivery site, never inferred. gen_eval transports
// the A2 evaluator's own typed result (`evaluated == false` = evaluator did not run).
struct common_cache_plan_candidate {
    bool present   = false;
    bool delivered = false;

    common_cache_plan_disposition disposition = common_cache_plan_disposition::unavailable;
    common_cache_plan_reason      reason      = COMMON_CACHE_PLAN_REASON_NONE;

    // diagnostics transported from the shipped path's own computation (B-a: never re-derived)
    llama_cache_acct_value lcp_tokens;
    llama_cache_acct_value payload_bytes;
    double sim        = 0; bool sim_known    = false;
    double f_keep     = 0; bool f_keep_known = false;
    // live_context_checkpoint: rows the short-circuiting shipped scan actually VISITED
    // (not the container size) + how many it rejected for a changed representation epoch
    uint32_t siblings_rejected_epoch = 0;
    uint32_t siblings_scanned        = 0;

    common_checkpoint_shadow_evaluation gen_eval;

    // First-failing-check discipline for out-of-order observation: the lowest value (earliest
    // precedence band) is THE reason; later/higher failures do not overwrite it.
    void note_reject(common_cache_plan_reason r) {
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
// (slot routing → host-cache load → context-checkpoint selection); rows accumulate at each
// stage and the record is finalized exactly once (outcome flips off `unknown`), after the
// actual restore/cold path and measured TTFT are known. Fields no shipped computation
// produced remain typed unknown/unavailable — estimates, budget deltas, would-evict lists,
// and a shadow-selected optimum are B/D work and stay unavailable in B0.
//
// Rows live in a fixed array indexed by provider so row() is noexcept: stage hooks inside
// the shipped selection path can never throw out of the observer.
struct common_cache_plan_record {
    uint32_t schema_version = COMMON_CACHE_PLAN_SCHEMA_VERSION;

    int64_t id_task = -1;   // request/decision id
    int32_t id_slot = -1;

    common_cache_plan_selection selection = common_cache_plan_selection::none;
    double sim_best_any = 0; bool sim_best_any_known = false;

    common_cache_plan_identity_evidence identity;

    std::array<common_cache_plan_candidate,
               size_t(common_cache_plan_provider::_count)> candidates;

    common_cache_plan_provider chosen  = common_cache_plan_provider::cold_replay;
    common_cache_plan_outcome  outcome = common_cache_plan_outcome::unknown; // != unknown ⇔ finalized

    // measured actuals (never estimates)
    llama_cache_acct_value n_prompt_tokens;
    llama_cache_acct_value n_reused_tokens;
    llama_cache_acct_value n_replayed_tokens;
    llama_cache_acct_value ttft_us;

    // a restore was attempted and failed (drives restore_failed_fell_back_cold at finalize
    // when nothing else delivered)
    bool restore_attempt_failed = false;

    // §7.5 cost-term slots: one per kind with its canonical raw unit, unavailable until B
    // lands an estimator (a default array would collapse to five "restore" slots)
    static constexpr std::array<llama_cache_acct_cost_term,
                                size_t(llama_cache_acct_cost_kind::_count)>
    default_cost_terms() {
        std::array<llama_cache_acct_cost_term,
                   size_t(llama_cache_acct_cost_kind::_count)> terms{};
        for (size_t i = 0; i < terms.size(); i++) {
            terms[i].kind     = llama_cache_acct_cost_kind(i);
            terms[i].raw_unit = llama_cache_acct_cost_kind_unit(terms[i].kind);
        }
        return terms;
    }
    std::array<llama_cache_acct_cost_term,
               size_t(llama_cache_acct_cost_kind::_count)> cost_terms = default_cost_terms();

    // C0 accounting snapshot; meaningful once outcome != unknown
    llama_cache_acct_snapshot acct;

    // observe-and-return: marks the provider's row present. noexcept by construction.
    common_cache_plan_candidate & row(common_cache_plan_provider provider) {
        auto & c = candidates[size_t(provider)];
        c.present = true;
        return c;
    }

    // A fallback that invalidates already-installed provider state revokes its deliveries:
    // failure/fallback dominates historical delivery — cold is a FINAL-STATE fact, and a
    // restore whose bytes were later discarded did not deliver. noexcept.
    void revoke_deliveries() {
        for (const auto prov : { common_cache_plan_provider::live_slot,
                                 common_cache_plan_provider::host_cache_entry,
                                 common_cache_plan_provider::live_context_checkpoint }) {
            candidates[size_t(prov)].delivered = false;
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

const char * common_cache_acct_category_name(llama_cache_acct_category c);
const char * common_cache_acct_residency_name(llama_cache_acct_residency r);
const char * common_cache_acct_measure_name(llama_cache_acct_measure m);
const char * common_cache_acct_known_name(llama_cache_acct_known k);
const char * common_cache_acct_unit_name(llama_cache_acct_unit u);
const char * common_cache_acct_cost_kind_name(llama_cache_acct_cost_kind k);

// One JSON shape for both B0 surfaces (the --cache-debug log line and /slots.cache_plan).
// Identities stay opaque by construction: no prompt bytes, no raw adapter/media identities —
// only closed-enum names, counts, and sizes. Only present rows and non-unknown accounting
// cells are emitted.
nlohmann::ordered_json common_cache_plan_record_json(const common_cache_plan_record & rec);
