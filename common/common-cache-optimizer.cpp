#include "common.h"

#include "common-cache-plan.h"

#include <algorithm>
#include <stdexcept>

const char * common_cache_optimizer_mode_name(common_cache_optimizer_mode mode) {
    switch (mode) {
        case common_cache_optimizer_mode::off:       return "off";
        case common_cache_optimizer_mode::baseline:  return "baseline";
        case common_cache_optimizer_mode::learn:     return "learn";
        case common_cache_optimizer_mode::auto_mode: return "auto";
        case common_cache_optimizer_mode::_count:    break;
    }
    return "invalid";
}

common_cache_optimizer_mode common_cache_optimizer_mode_parse(const std::string & value) {
    if (value == "off") {
        return common_cache_optimizer_mode::off;
    }
    if (value == "baseline") {
        return common_cache_optimizer_mode::baseline;
    }
    if (value == "learn") {
        return common_cache_optimizer_mode::learn;
    }
    if (value == "auto") {
        return common_cache_optimizer_mode::auto_mode;
    }
    throw std::invalid_argument("invalid cache optimizer mode: " + value);
}

const char * common_cache_optimizer_config_error_name(
        common_cache_optimizer_config_error error) {
    switch (error) {
        case common_cache_optimizer_config_error::none:
            return "none";
        case common_cache_optimizer_config_error::landed_authority_conflict:
            return "landed_authority_conflict";
        case common_cache_optimizer_config_error::_count:
            break;
    }
    return "invalid";
}

common_cache_optimizer_effective_config common_cache_optimizer_resolve(
        const common_cache_optimizer_raw_config & raw) noexcept {
    common_cache_optimizer_effective_config out;
    // Learned authority remains opt-in. Every explicit mode remains exact;
    // omission preserves the historical policy and its zero-overhead path.
    const auto mode = raw.mode_explicit
        ? raw.mode
        : common_cache_optimizer_mode::off;
    out.mode = mode;
    out.cache_debug = raw.cache_debug;
    out.cache_plan_preflight = raw.cache_plan_preflight;
    out.cache_control_api = raw.cache_control_api;

    if (mode == common_cache_optimizer_mode::off) {
        // Explicit off preserves every landed expert flag exactly.
        out.cache_lifecycle = raw.cache_lifecycle;
        out.landed_authority_level = raw.cache_plan_authority;
        out.local_authority_ceiling = common_cache_plan_authority_level::off;
        return out;
    }

    // Every non-off mode requests the lifecycle substrate for the ZC
    // retention policy. The policy/observer outputs are descriptive in ZC0b;
    // later ratchets attach their behavior to these already-frozen fields.
    out.cache_lifecycle = true;
    out.landed_authority_level = common_cache_plan_authority_level::off;
    out.retention_policy = common_cache_optimizer_retention_policy::intentional_baseline;
    out.observer_store_enabled = mode == common_cache_optimizer_mode::learn ||
                                 mode == common_cache_optimizer_mode::auto_mode;

    if (mode == common_cache_optimizer_mode::baseline ||
        mode == common_cache_optimizer_mode::learn) {
        if (raw.cache_plan_authority_explicit &&
            raw.cache_plan_authority != common_cache_plan_authority_level::off) {
            out.error = common_cache_optimizer_config_error::landed_authority_conflict;
        }
        return out;
    }

    // ZC5d lands the LRU local ratchet. An explicit higher ceiling is
    // a ceiling, not a request to enable unlanded tiers; explicit off remains
    // a complete local-authority kill switch.
    out.local_authority_ceiling = raw.cache_plan_authority_explicit
        ? std::min(raw.cache_plan_authority,
                   common_cache_plan_authority_level::lru)
        : common_cache_plan_authority_level::lru;
    return out;
}

bool common_cache_optimizer_resolve_params(
        common_params & params,
        std::string * error) noexcept {
    common_cache_optimizer_raw_config raw;
    raw.mode = params.cache_optimizer_mode;
    raw.mode_explicit = params.cache_optimizer_mode_explicit;
    raw.cache_lifecycle = params.cache_lifecycle;
    raw.cache_lifecycle_explicit = params.cache_lifecycle_explicit;
    raw.cache_plan_authority = params.cache_plan_authority;
    raw.cache_plan_authority_explicit = params.cache_plan_authority_explicit;
    raw.cache_debug = params.cache_debug;
    raw.cache_debug_explicit = params.cache_debug_explicit;
    raw.cache_plan_preflight = params.cache_plan_preflight;
    raw.cache_plan_preflight_explicit = params.cache_plan_preflight_explicit;
    raw.cache_control_api = params.cache_control_api;
    raw.cache_control_api_explicit = params.cache_control_api_explicit;
    params.cache_optimizer = common_cache_optimizer_resolve(raw);
    if (params.cache_optimizer.error == common_cache_optimizer_config_error::none) {
        if (error) {
            error->clear();
        }
        return true;
    }
    if (error) {
        *error = common_cache_optimizer_config_error_name(params.cache_optimizer.error);
    }
    return false;
}
