#include "arg.h"
#include "common-cache-plan.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(COND) do { if (!(COND)) { \
    std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #COND); \
    std::abort(); \
} } while (0)

static void check_same_effective(
        const common_cache_optimizer_effective_config & a,
        const common_cache_optimizer_effective_config & b) {
    CHECK(a.mode == b.mode);
    CHECK(a.cache_lifecycle == b.cache_lifecycle);
    CHECK(a.landed_authority_level == b.landed_authority_level);
    CHECK(a.cache_debug == b.cache_debug);
    CHECK(a.cache_plan_preflight == b.cache_plan_preflight);
    CHECK(a.cache_control_api == b.cache_control_api);
    CHECK(a.retention_policy == b.retention_policy);
    CHECK(a.observer_store_enabled == b.observer_store_enabled);
    CHECK(a.local_authority_ceiling == b.local_authority_ceiling);
    CHECK(a.error == b.error);
}

static common_cache_optimizer_raw_config raw_off(
        bool lifecycle,
        common_cache_plan_authority_level authority,
        bool debug,
        bool preflight,
        bool control,
        bool explicit_mode) {
    common_cache_optimizer_raw_config raw;
    raw.mode = common_cache_optimizer_mode::off;
    raw.mode_explicit = explicit_mode;
    raw.cache_lifecycle = lifecycle;
    raw.cache_lifecycle_explicit = lifecycle;
    raw.cache_plan_authority = authority;
    raw.cache_plan_authority_explicit = authority !=
        common_cache_plan_authority_level::off;
    raw.cache_debug = debug;
    raw.cache_debug_explicit = debug;
    raw.cache_plan_preflight = preflight;
    raw.cache_plan_preflight_explicit = preflight;
    raw.cache_control_api = control;
    raw.cache_control_api_explicit = control;
    return raw;
}

static void test_off_is_total_identity() {
    for (uint8_t level = uint8_t(common_cache_plan_authority_level::off);
         level < uint8_t(common_cache_plan_authority_level::_count); ++level) {
        for (uint8_t bits = 0; bits < 16; ++bits) {
            const auto authority =
                static_cast<common_cache_plan_authority_level>(level);
            const auto absent = common_cache_optimizer_resolve(raw_off(
                bits & 1, authority, bits & 2, bits & 4, bits & 8, false));
            const auto explicit_off = common_cache_optimizer_resolve(raw_off(
                bits & 1, authority, bits & 2, bits & 4, bits & 8, true));
            check_same_effective(absent, explicit_off);
            CHECK(absent.mode == common_cache_optimizer_mode::off);
            CHECK(absent.cache_lifecycle == bool(bits & 1));
            CHECK(absent.landed_authority_level == authority);
            CHECK(absent.cache_debug == bool(bits & 2));
            CHECK(absent.cache_plan_preflight == bool(bits & 4));
            CHECK(absent.cache_control_api == bool(bits & 8));
            CHECK(absent.retention_policy ==
                  common_cache_optimizer_retention_policy::historical_legacy);
            CHECK(!absent.observer_store_enabled);
            CHECK(absent.local_authority_ceiling ==
                  common_cache_plan_authority_level::off);
            CHECK(absent.error == common_cache_optimizer_config_error::none);
        }
    }
}

static void test_nonoff_modes() {
    common_cache_optimizer_raw_config raw;
    raw.cache_debug = true;
    raw.cache_plan_preflight = true;
    raw.cache_control_api = true;

    raw.mode = common_cache_optimizer_mode::baseline;
    auto out = common_cache_optimizer_resolve(raw);
    CHECK(out.cache_lifecycle);
    CHECK(out.retention_policy ==
          common_cache_optimizer_retention_policy::intentional_baseline);
    CHECK(!out.observer_store_enabled);
    CHECK(out.landed_authority_level == common_cache_plan_authority_level::off);
    CHECK(out.cache_debug && out.cache_plan_preflight && out.cache_control_api);

    raw.mode = common_cache_optimizer_mode::learn;
    out = common_cache_optimizer_resolve(raw);
    CHECK(out.cache_lifecycle);
    CHECK(out.observer_store_enabled);
    CHECK(out.local_authority_ceiling == common_cache_plan_authority_level::off);

    raw.cache_plan_authority = common_cache_plan_authority_level::similarity;
    raw.cache_plan_authority_explicit = true;
    out = common_cache_optimizer_resolve(raw);
    CHECK(out.error ==
          common_cache_optimizer_config_error::landed_authority_conflict);

    raw.cache_plan_authority = common_cache_plan_authority_level::off;
    out = common_cache_optimizer_resolve(raw);
    CHECK(out.error == common_cache_optimizer_config_error::none);

    raw.mode = common_cache_optimizer_mode::auto_mode;
    for (uint8_t level = uint8_t(common_cache_plan_authority_level::off);
         level < uint8_t(common_cache_plan_authority_level::_count); ++level) {
        raw.cache_plan_authority =
            static_cast<common_cache_plan_authority_level>(level);
        out = common_cache_optimizer_resolve(raw);
        CHECK(out.error == common_cache_optimizer_config_error::none);
        CHECK(out.landed_authority_level == common_cache_plan_authority_level::off);
        CHECK(out.local_authority_ceiling ==
              (raw.cache_plan_authority == common_cache_plan_authority_level::off
                  ? common_cache_plan_authority_level::off
                  : std::min(raw.cache_plan_authority,
                             common_cache_plan_authority_level::lru)));
    }
    raw.cache_plan_authority_explicit = false;
    raw.cache_plan_authority = common_cache_plan_authority_level::lru;
    out = common_cache_optimizer_resolve(raw);
    CHECK(out.local_authority_ceiling ==
          common_cache_plan_authority_level::lru);
}

static std::vector<char *> argv_for(std::vector<std::string> & args) {
    std::vector<char *> out;
    out.reserve(args.size());
    for (auto & arg : args) {
        out.push_back(arg.data());
    }
    return out;
}

static void test_parser_explicitness() {
    std::vector<std::string> args {
        "test-cache-optimizer-config",
        "--cache-optimizer", "auto",
        "--cache-plan-authority", "similarity",
        "--cache-lifecycle",
        "--cache-debug",
        "--cache-plan-preflight",
        "--cache-control-api",
    };
    auto argv = argv_for(args);
    common_params params;
    CHECK(common_params_parse(
        int(argv.size()), argv.data(), params, LLAMA_EXAMPLE_SERVER));
    CHECK(params.cache_optimizer_mode_explicit);
    CHECK(params.cache_plan_authority_explicit);
    CHECK(params.cache_lifecycle_explicit);
    CHECK(params.cache_debug_explicit);
    CHECK(params.cache_plan_preflight_explicit);
    CHECK(params.cache_control_api_explicit);
    CHECK(params.cache_optimizer.mode == common_cache_optimizer_mode::auto_mode);
    CHECK(params.cache_optimizer.cache_lifecycle);
    CHECK(params.cache_optimizer.landed_authority_level ==
          common_cache_plan_authority_level::off);
    CHECK(params.cache_optimizer.local_authority_ceiling ==
          common_cache_plan_authority_level::similarity);

    std::vector<std::string> absent_args { "test-cache-optimizer-config" };
    auto absent_argv = argv_for(absent_args);
    common_params absent;
    CHECK(common_params_parse(int(absent_argv.size()), absent_argv.data(),
                              absent, LLAMA_EXAMPLE_SERVER));
    CHECK(!absent.cache_optimizer_mode_explicit);
    CHECK(absent.cache_optimizer.mode == common_cache_optimizer_mode::off);

    std::vector<std::string> explicit_args {
        "test-cache-optimizer-config", "--cache-optimizer", "off"
    };
    auto explicit_argv = argv_for(explicit_args);
    common_params explicit_off;
    CHECK(common_params_parse(int(explicit_argv.size()), explicit_argv.data(),
                              explicit_off, LLAMA_EXAMPLE_SERVER));
    CHECK(explicit_off.cache_optimizer_mode_explicit);
    check_same_effective(absent.cache_optimizer, explicit_off.cache_optimizer);

    std::vector<std::string> conflict_args {
        "test-cache-optimizer-config", "--cache-optimizer", "learn",
        "--cache-plan-authority", "lru"
    };
    auto conflict_argv = argv_for(conflict_args);
    common_params conflict;
    CHECK(!common_params_parse(int(conflict_argv.size()), conflict_argv.data(),
                               conflict, LLAMA_EXAMPLE_SERVER));
}

int main() {
    common_init();
    CHECK(common_cache_optimizer_mode_parse("off") ==
          common_cache_optimizer_mode::off);
    CHECK(common_cache_optimizer_mode_parse("auto") ==
          common_cache_optimizer_mode::auto_mode);
    bool rejected = false;
    try {
        (void) common_cache_optimizer_mode_parse("future");
    } catch (...) {
        rejected = true;
    }
    CHECK(rejected);

    test_off_is_total_identity();
    test_nonoff_modes();
    test_parser_explicitness();
    std::puts("test-cache-optimizer-config: PASS");
    return 0;
}
