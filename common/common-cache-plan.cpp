#include "common-cache-plan.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

// Exhaustive name tables for the B0 closed enums. Switch-based with no default case so a new
// member without a name is a compile-time -Wswitch error, and the single unreachable return
// keeps release builds defined. These are the ONLY spellings of these names — CI bans replicas.

const char * common_cache_plan_reason_name(common_cache_plan_reason r) {
    switch (r) {
#define COMMON_CACHE_PLAN_REASON_NAME_CASE(sym, name, val) case COMMON_CACHE_PLAN_REASON_##sym: return name;
        COMMON_CACHE_PLAN_REASON_LIST(COMMON_CACHE_PLAN_REASON_NAME_CASE)
#undef COMMON_CACHE_PLAN_REASON_NAME_CASE
        case COMMON_CACHE_PLAN_REASON_COUNT_SENTINEL: break;
    }
    return "invalid";
}

const char * common_cache_plan_disposition_name(common_cache_plan_disposition d) {
    switch (d) {
        case common_cache_plan_disposition::accepted:              return "accepted";
        case common_cache_plan_disposition::rejected_invalid:      return "rejected_invalid";
        case common_cache_plan_disposition::valid_not_chosen_cost: return "valid_not_chosen_cost";
        case common_cache_plan_disposition::unavailable:           return "unavailable";
        case common_cache_plan_disposition::_count:                break;
    }
    return "invalid";
}

const char * common_cache_plan_provider_name(common_cache_plan_provider p) {
    switch (p) {
        case common_cache_plan_provider::live_slot:               return "live_slot";
        case common_cache_plan_provider::live_context_checkpoint: return "live_context_checkpoint";
        case common_cache_plan_provider::host_cache_entry:        return "host_cache_entry";
        case common_cache_plan_provider::cold_replay:             return "cold_replay";
        case common_cache_plan_provider::_count:                  break;
    }
    return "invalid";
}

const char * common_cache_plan_outcome_name(common_cache_plan_outcome o) {
    switch (o) {
        case common_cache_plan_outcome::unknown:                       return "unknown";
        case common_cache_plan_outcome::restored:                      return "restored";
        case common_cache_plan_outcome::restore_failed_fell_back_cold: return "restore_failed_fell_back_cold";
        case common_cache_plan_outcome::cold:                          return "cold";
        case common_cache_plan_outcome::_count:                        break;
    }
    return "invalid";
}

const char * common_cache_plan_selection_name(common_cache_plan_selection s) {
    switch (s) {
        case common_cache_plan_selection::none:       return "none";
        case common_cache_plan_selection::by_id:      return "by_id";
        case common_cache_plan_selection::similarity: return "similarity";
        case common_cache_plan_selection::route_home: return "route_home";
        case common_cache_plan_selection::lru:        return "lru";
        case common_cache_plan_selection::_count:     break;
    }
    return "invalid";
}

const char * common_cache_acct_category_name(llama_cache_acct_category c) {
    switch (c) {
        case llama_cache_acct_category::live_attention_state:                 return "live_attention_state";
        case llama_cache_acct_category::live_recurrent_state:                 return "live_recurrent_state";
        case llama_cache_acct_category::recurrent_rollback_planes:            return "recurrent_rollback_planes";
        case llama_cache_acct_category::full_snapshot_payload:                return "full_snapshot_payload";
        case llama_cache_acct_category::checkpoint_state_payload:             return "checkpoint_state_payload";
        case llama_cache_acct_category::typed_accelerator_payload:            return "typed_accelerator_payload";
        case llama_cache_acct_category::checkpoint_generation_page_metadata:  return "checkpoint_generation_page_metadata";
        case llama_cache_acct_category::checkpoint_generation_unit_metadata:  return "checkpoint_generation_unit_metadata";
        case llama_cache_acct_category::live_generation_metadata:             return "live_generation_metadata";
        case llama_cache_acct_category::ownership_index_metadata:             return "ownership_index_metadata";
        case llama_cache_acct_category::unit_version_payload:                 return "unit_version_payload";
        case llama_cache_acct_category::clean_stash_payload:                  return "clean_stash_payload";
        case llama_cache_acct_category::artifact_descriptor_metadata:         return "artifact_descriptor_metadata";
        case llama_cache_acct_category::artifact_reference_metadata:          return "artifact_reference_metadata";
        case llama_cache_acct_category::transfer_staging:                     return "transfer_staging";
        case llama_cache_acct_category::codec_workspace:                      return "codec_workspace";
        case llama_cache_acct_category::pinned_preimage_ring:                 return "pinned_preimage_ring";
        case llama_cache_acct_category::rolling_window_tape:                  return "rolling_window_tape";
        case llama_cache_acct_category::container_overhead:                   return "container_overhead";
        case llama_cache_acct_category::_count:                               break;
    }
    return "invalid";
}

const char * common_cache_acct_residency_name(llama_cache_acct_residency r) {
    switch (r) {
        case llama_cache_acct_residency::device:         return "device";
        case llama_cache_acct_residency::pinned_host:    return "pinned_host";
        case llama_cache_acct_residency::pageable_host:  return "pageable_host";
        case llama_cache_acct_residency::disk:           return "disk";
        case llama_cache_acct_residency::remote:         return "remote";
        case llama_cache_acct_residency::not_applicable: return "not_applicable";
        case llama_cache_acct_residency::_count:         break;
    }
    return "invalid";
}

const char * common_cache_acct_measure_name(llama_cache_acct_measure m) {
    switch (m) {
        case llama_cache_acct_measure::logical_payload:    return "logical_payload";
        case llama_cache_acct_measure::resident_allocated: return "resident_allocated";
        case llama_cache_acct_measure::reserved:           return "reserved";
        case llama_cache_acct_measure::transient_peak:     return "transient_peak";
        case llama_cache_acct_measure::_count:             break;
    }
    return "invalid";
}

const char * common_cache_acct_known_name(llama_cache_acct_known k) {
    switch (k) {
        case llama_cache_acct_known::known:       return "known";
        case llama_cache_acct_known::unknown:     return "unknown";
        case llama_cache_acct_known::unavailable: return "unavailable";
        case llama_cache_acct_known::_count:      break;
    }
    return "invalid";
}

const char * common_cache_acct_unit_name(llama_cache_acct_unit u) {
    switch (u) {
        case llama_cache_acct_unit::bytes:      return "bytes";
        case llama_cache_acct_unit::tokens:     return "tokens";
        case llama_cache_acct_unit::operations: return "operations";
        case llama_cache_acct_unit::_count:     break;
    }
    return "invalid";
}

const char * common_cache_acct_cost_kind_name(llama_cache_acct_cost_kind k) {
    switch (k) {
        case llama_cache_acct_cost_kind::restore:   return "restore";
        case llama_cache_acct_cost_kind::replay:    return "replay";
        case llama_cache_acct_cost_kind::transfer:  return "transfer";
        case llama_cache_acct_cost_kind::eviction:  return "eviction";
        case llama_cache_acct_cost_kind::workspace: return "workspace";
        case llama_cache_acct_cost_kind::_count:    break;
    }
    return "invalid";
}

static json cache_plan_value_json(const llama_cache_acct_value & v) {
    if (v.state == llama_cache_acct_known::known) {
        return json(v.value);
    }
    return json(common_cache_acct_known_name(v.state));
}

json common_cache_plan_record_json(const common_cache_plan_record & rec) {
    const bool finalized = rec.outcome != common_cache_plan_outcome::unknown;

    json cands = json::array();
    for (size_t p = 0; p < size_t(common_cache_plan_provider::_count); p++) {
        const auto & c = rec.candidates[p];
        if (!c.present) {
            continue; // an unobserved provider has no row — absence is not a verdict
        }
        json jc = {
            { "provider",      common_cache_plan_provider_name(common_cache_plan_provider(p)) },
            { "disposition",   common_cache_plan_disposition_name(c.disposition) },
            { "reason",        common_cache_plan_reason_name(c.reason) },
            { "delivered",     c.delivered },
            { "lcp_tokens",    cache_plan_value_json(c.lcp_tokens) },
            { "payload_bytes", cache_plan_value_json(c.payload_bytes) },
        };
        if (c.sim_known)    { jc["sim"]    = c.sim; }
        if (c.f_keep_known) { jc["f_keep"] = c.f_keep; }
        if (c.siblings_scanned > 0) {
            jc["siblings_scanned"]        = c.siblings_scanned;
            jc["siblings_rejected_epoch"] = c.siblings_rejected_epoch;
        }
        if (c.gen_eval.evaluated) {
            jc["generation_eval"] = json {
                { "category",   common_checkpoint_shadow_category_name(c.gen_eval.category) },
                { "reason",     common_checkpoint_shadow_eval_reason_name(c.gen_eval.reason) },
                { "tombstone",  common_checkpoint_shadow_tombstone_name(c.gen_eval.tombstone_class) },
                { "refinement", c.gen_eval.refinement_used },
            };
        }
        cands.push_back(std::move(jc));
    }

    json terms = json::object();
    for (const auto & term : rec.cost_terms) {
        json jt = {
            { "raw",          cache_plan_value_json(term.raw) },
            { "unit",         common_cache_acct_unit_name(term.raw_unit) }, // schema metadata, valid while unavailable
            { "estimated_us", cache_plan_value_json(term.estimated_us) },
        };
        // the estimator version is metadata OF an estimate: emitting it while the estimate
        // is unavailable would fabricate evidence
        if (term.estimated_us.state == llama_cache_acct_known::known) {
            jt["estimator_version"] = term.estimator_version;
        }
        terms[common_cache_acct_cost_kind_name(term.kind)] = std::move(jt);
    }

    // causal delivery chain: which providers actually applied state, in shipped order
    // (live slot prefix → host snapshot → context checkpoint). `chosen` is the terminal one.
    json chain = json::array();
    for (const auto prov : { common_cache_plan_provider::live_slot,
                             common_cache_plan_provider::host_cache_entry,
                             common_cache_plan_provider::live_context_checkpoint }) {
        const auto & c = rec.candidates[size_t(prov)];
        if (c.present && c.delivered) {
            chain.push_back(common_cache_plan_provider_name(prov));
        }
    }

    json out = {
        { "schema_version",    rec.schema_version },
        { "id_task",           rec.id_task },
        { "id_slot",           rec.id_slot },
        { "selection",         common_cache_plan_selection_name(rec.selection) },
        { "identity", json {
            { "model",              cache_plan_value_json(rec.identity.model_digest) },
            { "execution",          cache_plan_value_json(rec.identity.execution_digest) },
            { "adapter_config",     cache_plan_value_json(rec.identity.adapter_config_digest) },
            { "media_content",      cache_plan_value_json(rec.identity.media_content_digest) },
            { "tokenizer_template", cache_plan_value_json(rec.identity.tokenizer_template_digest) },
            { "prefix_tokens",      cache_plan_value_json(rec.identity.prefix_token_digest) },
        } },
        { "candidates",        std::move(cands) },
        { "delivered_chain",   std::move(chain) },
        { "seq_cp_capability", true }, // copy_state_to's primitive exists on every build
        { "chosen",            finalized ? common_cache_plan_provider_name(rec.chosen) : "unknown" },
        { "outcome",           common_cache_plan_outcome_name(rec.outcome) },
        { "n_prompt_tokens",   cache_plan_value_json(rec.n_prompt_tokens) },
        { "n_reused_tokens",   cache_plan_value_json(rec.n_reused_tokens) },
        { "n_replayed_tokens", cache_plan_value_json(rec.n_replayed_tokens) },
        { "ttft_us",           cache_plan_value_json(rec.ttft_us) },
        { "cost_terms",        std::move(terms) },
    };
    if (rec.sim_best_any_known) {
        out["sim_best_any"] = rec.sim_best_any;
    }

    if (finalized) {
        // touched cells only (state != unknown) — a known ZERO is an observation and is
        // emitted; an unknown cell is silence, never a zero
        json cells = json::array();
        for (size_t c = 0; c < size_t(llama_cache_acct_category::_count); c++) {
            for (size_t r = 0; r < size_t(llama_cache_acct_residency::_count); r++) {
                for (size_t m = 0; m < size_t(llama_cache_acct_measure::_count); m++) {
                    const auto & cell = rec.acct.cells[c][r].measures[m];
                    if (cell.state == llama_cache_acct_known::unknown) {
                        continue;
                    }
                    cells.push_back(json {
                        { "category",  common_cache_acct_category_name(llama_cache_acct_category(c)) },
                        { "residency", common_cache_acct_residency_name(llama_cache_acct_residency(r)) },
                        { "measure",   common_cache_acct_measure_name(llama_cache_acct_measure(m)) },
                        { "value",     cache_plan_value_json(cell) },
                    });
                }
            }
        }
        out["accounting"] = json {
            { "schema_version", rec.acct.schema_version },
            { "serial",         rec.acct.serial },
            { "completeness",   common_cache_acct_known_name(rec.acct.completeness) },
            { "live_ops",       rec.acct.live_ops },
            { "cells",          std::move(cells) },
            { "faults", json {
                { "invalid_transition", rec.acct.faults_invalid_transition },
                { "overflow",           rec.acct.faults_overflow },
                { "unknown_id",         rec.acct.faults_unknown_id },
                { "allocation",         rec.acct.faults_allocation },
            } },
        };
    }

    return out;
}
