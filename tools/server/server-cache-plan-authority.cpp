#include "server-cache-plan-authority.h"

#include "common-cache-plan-estimate.h"

uint64_t server_cache_plan_capability_fold(
        uint64_t hash,
        uint64_t value) noexcept {
    // FNV-1a with an explicit value delimiter. This is a process-local drift
    // detector, not a durable/content identity.
    for (unsigned i = 0; i < 8; ++i) {
        hash = (hash ^ uint8_t(value >> (8*i))) * 1099511628211ull;
    }
    return (hash ^ 0xffu) * 1099511628211ull;
}

bool server_cache_plan_find_or_assign_source_id(
        const void * instance,
        std::array<const void *, COMMON_CACHE_PLAN_MAX_CANDIDATES> & instances,
        uint32_t & n_instances,
        int32_t & source_id) noexcept {
    if (!instance) {
        source_id = -1;
        return false;
    }
    for (uint32_t i = 0; i < n_instances; ++i) {
        if (instances[i] == instance) {
            source_id = int32_t(i);
            return true;
        }
    }
    if (n_instances >= instances.size() ||
        n_instances > uint32_t(SERVER_CACHE_PLAN_MAX_HOST_SOURCE_ID)) {
        source_id = -1;
        return false;
    }
    source_id = int32_t(n_instances);
    instances[n_instances++] = instance;
    return true;
}

int32_t server_cache_plan_checkpoint_source_id_from_reverse(
        size_t checkpoint_count,
        uint32_t reverse_ordinal,
        int32_t host_source_id) noexcept {
    if (checkpoint_count == 0 || reverse_ordinal >= checkpoint_count) {
        return -1;
    }
    const size_t forward = checkpoint_count - 1 - reverse_ordinal;
    if (forward >= size_t(SERVER_CACHE_PLAN_HOST_CHECKPOINT_STRIDE)) {
        return -1;
    }
    return host_source_id >= 0
        ? server_cache_plan_host_checkpoint_source_id(
              host_source_id, int32_t(forward))
        : int32_t(forward);
}

void server_cache_plan_authority::plan_before_mutation(
        common_cache_plan_record & rec,
        uint64_t capability_before,
        uint64_t capability_after) noexcept {
    rec.authority = {};
    common_cache_plan_authority_fallback fallback =
        common_cache_plan_authority_fallback::none;
    if (capability_before != capability_after) {
        rec.clear_planner_outputs();
        rec.planner_status = common_cache_plan_planner_status::incomplete_evidence;
        fallback = common_cache_plan_authority_fallback::stale_capability;
    } else {
        common_cache_plan_run_planner(rec);
    }
    common_cache_plan_derive_shadow_authority(rec, configured_level, fallback);
    rec.authority_prequalified =
        rec.planner_status == common_cache_plan_planner_status::ok &&
        capability_before == capability_after;
    rec.planner_precomputed = true;
}

void server_cache_plan_authority::fail_closed(
        common_cache_plan_record & rec,
        common_cache_plan_authority_fallback reason) noexcept {
    rec.clear_planner_outputs();
    rec.planner_status = common_cache_plan_planner_status::internal_fault;
    common_cache_plan_derive_shadow_authority(rec, configured_level, reason);
    rec.authority_prequalified = false;
    rec.planner_precomputed = true;
}

void server_cache_plan_authority::finalize_legacy_execution(
        common_cache_plan_record & rec) noexcept {
    common_cache_plan_finalize_shadow_authority(rec);
    counters.observe(rec.authority, rec.authority_prequalified);
}

server_cache_plan_live_evaluation server_cache_plan_evaluate_live(
        bool busy,
        bool has_payload,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens) noexcept {
    server_cache_plan_live_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.reason = busy ? COMMON_CACHE_PLAN_REASON_PROVIDER_BUSY :
                 !has_payload ? COMMON_CACHE_PLAN_REASON_PROVIDER_UNAVAILABLE :
                 lcp_tokens == 0 ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_live(
        common_cache_plan_candidate * row,
        const server_cache_plan_live_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->note_reject(evaluation.reason);
}

server_cache_plan_host_evaluation server_cache_plan_evaluate_host(
        bool payload_present,
        bool identity_matches,
        uint64_t lcp_tokens,
        uint64_t prompt_tokens,
        uint64_t source_tokens,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_host_evaluation out;
    out.lcp_tokens = lcp_tokens;
    out.payload_bytes = payload_bytes;
    out.sim = prompt_tokens ? float(lcp_tokens) / float(prompt_tokens) : 0.0f;
    out.f_keep = source_tokens ? float(lcp_tokens) / float(source_tokens) : 0.0f;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !identity_matches ? COMMON_CACHE_PLAN_REASON_ADAPTER_CONFIG_MISMATCH :
                 out.f_keep < 0.25f ? COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT :
                 COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL;
    return out;
}

void server_cache_plan_apply_host(
        common_cache_plan_candidate * row,
        const server_cache_plan_host_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->sim = evaluation.sim;
    row->sim_known = true;
    row->f_keep = evaluation.f_keep;
    row->f_keep_known = true;
    row->note_reject(evaluation.reason);
}

server_cache_plan_checkpoint_evaluation server_cache_plan_evaluate_checkpoint(
        bool payload_present,
        bool frontier_current,
        bool recurrent,
        bool representation_matches,
        int64_t pos_min,
        int64_t pos_max,
        int64_t next_position,
        int64_t min_position_threshold,
        uint64_t payload_bytes) noexcept {
    server_cache_plan_checkpoint_evaluation out;
    out.lcp_tokens = pos_max >= 0 ? uint64_t(pos_max) : 0;
    out.payload_bytes = payload_bytes;
    out.reason = !payload_present ? COMMON_CACHE_PLAN_REASON_PAYLOAD_EMPTY :
                 !frontier_current ? COMMON_CACHE_PLAN_REASON_FRONTIER_INVALID :
                 !representation_matches ? COMMON_CACHE_PLAN_REASON_REPRESENTATION_EPOCH_CHANGED :
                 recurrent
                    ? (pos_max < next_position
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT)
                    : (pos_max <= next_position &&
                       (pos_min < min_position_threshold || pos_min == 0)
                        ? COMMON_CACHE_PLAN_REASON_COST_NOT_MINIMAL
                        : COMMON_CACHE_PLAN_REASON_COVERAGE_INSUFFICIENT);
    return out;
}

void server_cache_plan_apply_checkpoint(
        common_cache_plan_candidate * row,
        const server_cache_plan_checkpoint_evaluation & evaluation) noexcept {
    if (!row) {
        return;
    }
    row->lcp_tokens = llama_cache_acct_value::measured(evaluation.lcp_tokens);
    row->payload_bytes = llama_cache_acct_value::measured(evaluation.payload_bytes);
    row->note_reject(evaluation.reason);
}
