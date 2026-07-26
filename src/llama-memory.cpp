#include "llama-memory.h"

#include <limits>

llama_memory_resume_plan llama_memory_i::plan_resume(
        llama_seq_id seq_id,
        llama_pos    target_pos) const {
    llama_memory_resume_plan plan = {};
    plan.full_replay = true;

    if (seq_id < 0 || target_pos < 0) {
        plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_INVALID_ARGUMENT;
        return plan;
    }

    const llama_pos pos_min = seq_pos_min(seq_id);
    const llama_pos pos_max = seq_pos_max(seq_id);
    if (pos_min < 0 || pos_max < pos_min) {
        plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_EMPTY_SEQUENCE;
        return plan;
    }
    if (pos_max == std::numeric_limits<llama_pos>::max()) {
        plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_INVALID_ARGUMENT;
        return plan;
    }

    const int64_t live_next = (int64_t) pos_max + 1;
    plan.replay_tokens = live_next;

    if (target_pos < pos_min) {
        plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_TARGET_BEFORE_COVERAGE;
        return plan;
    }
    if ((int64_t) target_pos > live_next) {
        plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_TARGET_AFTER_FRONTIER;
        return plan;
    }

    plan.resumable    = true;
    plan.full_replay  = false;
    plan.components   = LLAMA_MEMORY_RESUME_COMPONENT_ATTN;
    plan.reuse_tokens = (int64_t) target_pos - pos_min;
    plan.replay_tokens = live_next - target_pos;
    plan.reject_reason = LLAMA_MEMORY_RESUME_REJECT_NONE;
    return plan;
}

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
