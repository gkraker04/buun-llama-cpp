#include "server-cache-calibration-store.h"

#include "../../src/llama-sha256.h"
#include "server-common.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <tuple>

#if defined(_WIN32)
#  include <bcrypt.h>
#  include <fcntl.h>
#  include <io.h>
#  include <process.h>
#  include <share.h>
#  include <windows.h>
#else
#  include <dirent.h>
#  include <fcntl.h>
#  include <sys/file.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <sys/random.h>
#  endif
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

constexpr uint32_t STORE_SCHEMA = 1;
constexpr uint32_t ESTIMATOR_VERSION = 2;
constexpr size_t ROOT_PAYLOAD_LIMIT = 64 * 1024;
constexpr size_t PROFILE_PAYLOAD_LIMIT = 1024 * 1024;
constexpr size_t MAX_PROFILES = 16;
constexpr size_t ENVELOPE_FIXED = 8 + 4 + 4 + 32;
constexpr size_t JSON_EVENT_LIMIT = 32768;
constexpr int JSON_DEPTH_LIMIT = 12;
constexpr size_t JSON_KEY_LIMIT = 64;
constexpr size_t JSON_STRING_LIMIT = 128;
constexpr size_t STORE_ENTRY_LIMIT = 64;
constexpr uint64_t STORE_BYTE_LIMIT = 32 * 1024 * 1024;
// A cleaned store can contain at most sixteen referenced payloads. During a
// commit it can additionally contain the new immutable payload and both the
// old and temporary root. This is the true pre-GC high-water mark and remains
// far below the public store cap.
constexpr uint64_t STORE_COMMIT_HIGH_WATER =
    (MAX_PROFILES + 1) * (PROFILE_PAYLOAD_LIMIT + ENVELOPE_FIXED) +
    2 * (ROOT_PAYLOAD_LIMIT + ENVELOPE_FIXED);
static_assert(STORE_COMMIT_HIGH_WATER < STORE_BYTE_LIMIT);
constexpr char MAGIC[] = "BUUNCAL1";
constexpr char DIGEST_DOMAIN[] = "buun-cache-calibration-v1";
std::atomic<server_cache_calibration_test_fault> test_fault =
    server_cache_calibration_test_fault::none;
std::atomic<uint64_t> test_fault_hits = 0;

bool take_test_fault(server_cache_calibration_test_fault expected) {
    auto value = expected;
    if (!test_fault.compare_exchange_strong(
            value, server_cache_calibration_test_fault::none,
            std::memory_order_acq_rel)) return false;
    test_fault_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out.push_back(uint8_t(value >> (8 * i)));
}

uint32_t read_u32(const uint8_t * data) {
    uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) out |= uint32_t(data[i]) << (8 * i);
    return out;
}

std::string hex_digest(const std::array<uint8_t, 32> & value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (size_t i = 0; i < value.size(); ++i) {
        out[2 * i] = digits[value[i] >> 4];
        out[2 * i + 1] = digits[value[i] & 15];
    }
    return out;
}

bool parse_digest(const json & value, std::array<uint8_t, 32> & out) {
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    if (text.size() != 64) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int hi = nibble(text[2 * i]);
        const int lo = nibble(text[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}

std::string double_bits(double value) {
    if (value == 0) value = 0;
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (size_t i = 0; i < 16; ++i) {
        out[15 - i] = digits[bits & 15];
        bits >>= 4;
    }
    return out;
}

bool parse_double_bits(const json & value, double & out) {
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    if (text.size() != 16) return false;
    uint64_t bits = 0;
    for (char c : text) {
        int digit = c >= '0' && c <= '9' ? c - '0' :
                    c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (digit < 0) return false;
        bits = (bits << 4) | uint64_t(digit);
    }
    if (bits == UINT64_C(0x8000000000000000)) return false;
    std::memcpy(&out, &bits, sizeof(out));
    return std::isfinite(out);
}

template <size_t N>
json doubles_json(const std::array<double, N> & values) {
    json out = json::array();
    for (double value : values) out.push_back(double_bits(value));
    return out;
}

template <size_t N>
bool parse_doubles(const json & value, std::array<double, N> & out) {
    if (!value.is_array() || value.size() != N) return false;
    for (size_t i = 0; i < N; ++i) {
        if (!parse_double_bits(value[i], out[i])) return false;
    }
    return true;
}

template <size_t Capacity, typename Size>
json integers_json(const server_cache_calibration_bounded_array<
                       uint64_t, Capacity, Size> & values) {
    json out = json::array();
    for (uint64_t value : values) out.push_back(value);
    return out;
}

template <size_t Capacity, typename Size>
bool parse_unsigned_array(const json & value,
                          server_cache_calibration_bounded_array<
                              uint64_t, Capacity, Size> & out) {
    out.clear();
    if (!value.is_array() || value.size() > Capacity) return false;
    for (const auto & item : value) {
        if (!item.is_number_unsigned() || !out.push_back(item.get<uint64_t>())) {
            out.clear();
            return false;
        }
    }
    return true;
}

bool exact_keys(const json & value, std::initializer_list<const char *> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    for (const char * key : keys) if (!value.contains(key)) return false;
    return true;
}

template <typename T>
bool exact_unsigned(const json & value, T & out) {
    static_assert(std::is_unsigned_v<T>);
    if (!value.is_number_unsigned()) return false;
    const uint64_t raw = value.get<uint64_t>();
    if (raw > uint64_t(std::numeric_limits<T>::max())) return false;
    out = T(raw);
    return true;
}

const char * authority_terminal_name(
        server_cache_calibration_authority_terminal value) {
    switch (value) {
        case server_cache_calibration_authority_terminal::none: return "none";
        case server_cache_calibration_authority_terminal::tail_exceeded: return "tail_exceeded";
        case server_cache_calibration_authority_terminal::confidence_budget_exhausted: return "confidence_budget_exhausted";
        case server_cache_calibration_authority_terminal::ordinal_exhausted: return "ordinal_exhausted";
        case server_cache_calibration_authority_terminal::numeric_fault: return "numeric_fault";
        case server_cache_calibration_authority_terminal::_count: break;
    }
    return "invalid";
}

bool parse_authority_terminal(
        const json & value,
        server_cache_calibration_authority_terminal & out) {
    if (!value.is_string()) return false;
    const std::string text = value.get<std::string>();
    for (uint8_t i = 0;
         i < uint8_t(server_cache_calibration_authority_terminal::_count); ++i) {
        const auto candidate = server_cache_calibration_authority_terminal(i);
        if (text == authority_terminal_name(candidate)) {
            out = candidate;
            return true;
        }
    }
    return false;
}

json key_json(const server_cache_observation_key & key) {
    return {
        { "operation", server_cache_observation_operation_name(key.operation) },
        { "provider", common_cache_plan_provider_name(key.provider) },
        { "restore_kind", key.restore_kind },
        { "prepare_shape", key.prepare_shape },
        { "contention_bucket", key.contention_bucket },
        { "start_bucket", key.start_bucket },
        { "batch_bucket", key.batch_bucket },
        { "ubatch_bucket", key.ubatch_bucket },
        { "size_family", key.size_family },
        { "feature_dim", key.feature_dim },
        { "profile_execution_digest", hex_digest(key.profile_execution_digest) },
        { "participant_execution_digest", hex_digest(key.participant_execution_digest) },
        { "adapter_application_digest", hex_digest(key.adapter_application_digest) },
        { "representation_digest", hex_digest(key.representation_digest) },
        { "effect_action_shape_digest", hex_digest(key.effect_action_shape_digest) },
        { "adapter_application_complete", key.adapter_application_complete },
        { "identity_complete", key.identity_complete },
        { "identity_exact", key.identity_exact },
    };
}

bool parse_key(const json & value, server_cache_observation_key & out) {
    if (!exact_keys(value, {
            "operation", "provider", "restore_kind", "prepare_shape",
            "contention_bucket", "start_bucket", "batch_bucket",
            "ubatch_bucket", "size_family", "feature_dim",
            "profile_execution_digest", "participant_execution_digest",
            "adapter_application_digest", "representation_digest",
            "effect_action_shape_digest", "adapter_application_complete",
            "identity_complete", "identity_exact" })) return false;
    try {
        if (!value.at("operation").is_string() ||
            !value.at("provider").is_string()) return false;
        const std::string operation = value.at("operation").get<std::string>();
        const std::string provider = value.at("provider").get<std::string>();
        bool operation_found = false;
        bool provider_found = false;
        for (uint8_t i = 0;
             i < uint8_t(server_cache_observation_operation::_count); ++i) {
            const auto candidate = server_cache_observation_operation(i);
            if (operation == server_cache_observation_operation_name(candidate)) {
                out.operation = candidate;
                operation_found = true;
                break;
            }
        }
        for (uint8_t i = 0; i < uint8_t(common_cache_plan_provider::_count); ++i) {
            const auto candidate = common_cache_plan_provider(i);
            if (provider == common_cache_plan_provider_name(candidate)) {
                out.provider = candidate;
                provider_found = true;
                break;
            }
        }
        if (!operation_found || !provider_found) return false;
        if (!exact_unsigned(value.at("restore_kind"), out.restore_kind) ||
            !exact_unsigned(value.at("prepare_shape"), out.prepare_shape) ||
            !exact_unsigned(value.at("contention_bucket"), out.contention_bucket) ||
            !exact_unsigned(value.at("start_bucket"), out.start_bucket) ||
            !exact_unsigned(value.at("batch_bucket"), out.batch_bucket) ||
            !exact_unsigned(value.at("ubatch_bucket"), out.ubatch_bucket) ||
            !exact_unsigned(value.at("size_family"), out.size_family) ||
            !exact_unsigned(value.at("feature_dim"), out.feature_dim)) return false;
        if (out.size_family >= 4 || out.feature_dim == 0 || out.feature_dim > 4 ||
            !parse_digest(value.at("profile_execution_digest"), out.profile_execution_digest) ||
            !parse_digest(value.at("participant_execution_digest"), out.participant_execution_digest) ||
            !parse_digest(value.at("adapter_application_digest"), out.adapter_application_digest) ||
            !parse_digest(value.at("representation_digest"), out.representation_digest) ||
            !parse_digest(value.at("effect_action_shape_digest"), out.effect_action_shape_digest)) return false;
        out.adapter_application_complete = value.at("adapter_application_complete").get<bool>();
        out.identity_complete = value.at("identity_complete").get<bool>();
        out.identity_exact = value.at("identity_exact").get<bool>();
        return true;
    } catch (...) {
        return false;
    }
}

json instance_json(const server_cache_calibration_instance_snapshot & value) {
    json v = json::array();
    for (const auto & row : value.v) v.push_back(doubles_json(row));
    return {
        { "slot", value.slot },
        { "key", key_json(value.key) },
        { "fit_generation", value.fit_generation },
        { "authority_terminal", authority_terminal_name(value.authority_terminal) },
        { "tail_actual_max_us", value.tail_actual_max_us },
        { "V_bits", std::move(v) },
        { "b_bits", doubles_json(value.b) },
        { "n_fit", value.n_fit },
        { "feature_min_bits", doubles_json(value.feature_min) },
        { "feature_max_bits", doubles_json(value.feature_max) },
        { "qualified_execution_ordinal", value.qualified_execution_ordinal },
        { "log_wealth_bits", doubles_json(value.log_wealth) },
        { "n_validation", value.n_validation },
        { "fit_region_minutes", integers_json(value.fit_region_minutes) },
        { "validation_region_minutes", integers_json(value.validation_region_minutes) },
        { "safe_measurable_opportunities", value.safe_measurable_opportunities },
        { "opportunity_at_last_validation", value.opportunity_at_last_validation },
        { "last_fit_unix_ms", value.last_fit_unix_ms },
        { "last_validation_unix_ms", value.last_validation_unix_ms },
        { "response_reservoir", value.response_reservoir },
        { "reservoir_seen", value.reservoir_seen },
    };
}

template <size_t Capacity, typename Size>
bool monotonic_regions(const server_cache_calibration_bounded_array<
                           uint64_t, Capacity, Size> & regions) {
    return std::adjacent_find(regions.begin(), regions.end(),
            [](uint64_t a, uint64_t b) { return a >= b; }) == regions.end();
}

bool positive_definite(
        const std::array<std::array<double, 4>, 4> & value,
        uint8_t dimension) {
    std::array<std::array<double, 4>, 4> lower = {};
    for (uint8_t i = 0; i < dimension; ++i) {
        for (uint8_t j = 0; j <= i; ++j) {
            double sum = value[i][j];
            for (uint8_t k = 0; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!std::isfinite(sum) || sum <= 0) return false;
                lower[i][j] = std::sqrt(sum);
            } else {
                lower[i][j] = sum / lower[j][j];
                if (!std::isfinite(lower[i][j])) return false;
            }
        }
    }
    return true;
}

bool nonzero_digest(const std::array<uint8_t, 32> & value) {
    return std::any_of(value.begin(), value.end(), [](uint8_t byte) {
        return byte != 0;
    });
}

bool snapshot_semantically_valid(
        const server_cache_calibration_profile_snapshot & value) {
    if (value.mutation_generation == 0 ||
        value.mutation_generation == UINT64_MAX ||
        !nonzero_digest(value.profile_identity_digest) ||
        value.instances.size() > value.instances.capacity()) return false;
    std::array<bool, server_cache_observation_store::instance_capacity> slots = {};
    for (const auto & instance : value.instances) {
        if (instance.slot >= slots.size() || slots[instance.slot] ||
            !server_cache_observation_key_valid(instance.key) ||
            instance.key.contention_bucket != 0 ||
            instance.key.profile_execution_digest !=
                value.profile_identity_digest ||
            !monotonic_regions(instance.fit_region_minutes) ||
            !monotonic_regions(instance.validation_region_minutes) ||
            instance.opportunity_at_last_validation >
                instance.safe_measurable_opportunities ||
            instance.authority_terminal ==
                server_cache_calibration_authority_terminal::_count ||
            (instance.authority_terminal ==
                 server_cache_calibration_authority_terminal::tail_exceeded &&
             instance.tail_actual_max_us == 0)) return false;
        slots[instance.slot] = true;
        for (size_t i = 0; i < 4; ++i) {
            if (!std::isfinite(instance.b[i]) ||
                !std::isfinite(instance.feature_min[i]) ||
                !std::isfinite(instance.feature_max[i]) ||
                instance.feature_min[i] > instance.feature_max[i]) return false;
            for (size_t j = 0; j < 4; ++j) {
                if (!std::isfinite(instance.v[i][j])) return false;
            }
        }
        for (double wealth : instance.log_wealth) {
            if (!std::isfinite(wealth)) return false;
        }
        for (uint8_t i = 0; i < instance.key.feature_dim; ++i) {
            if (instance.v[i][i] < 1.0) return false;
            for (uint8_t j = 0; j < instance.key.feature_dim; ++j) {
                if (instance.v[i][j] != instance.v[j][i]) return false;
            }
        }
        if (!positive_definite(instance.v, instance.key.feature_dim)) return false;
    }
    return true;
}

bool manifest_semantically_valid(
        const server_cache_calibration_manifest & value) {
    if (!nonzero_digest(value.store_lineage_id) ||
        value.profiles.size() > MAX_PROFILES) return false;
    std::set<uint64_t> q_values;
    std::set<uint64_t> file_values;
    std::set<uint64_t> prune_values;
    std::set<std::array<uint8_t, 32>> identities;
    for (const auto & ref : value.profiles) {
        if (!nonzero_digest(ref.profile_identity_digest) ||
            ref.profile_generation_ordinal >=
                value.next_profile_generation_ordinal ||
            ref.profile_file_generation >= value.next_immutable_file_ordinal ||
            ref.persisted_prune_recency >= value.next_persisted_prune_epoch ||
            !q_values.insert(ref.profile_generation_ordinal).second ||
            !file_values.insert(ref.profile_file_generation).second ||
            !prune_values.insert(ref.persisted_prune_recency).second ||
            !identities.insert(ref.profile_identity_digest).second) return false;
    }
    return true;
}

bool parse_instance(const json & value,
                    server_cache_calibration_instance_snapshot & out) {
    if (!exact_keys(value, {
            "slot", "key", "fit_generation", "authority_terminal",
            "tail_actual_max_us", "V_bits", "b_bits", "n_fit",
            "feature_min_bits", "feature_max_bits",
            "qualified_execution_ordinal", "log_wealth_bits",
            "n_validation", "fit_region_minutes", "validation_region_minutes",
            "safe_measurable_opportunities", "opportunity_at_last_validation",
            "last_fit_unix_ms", "last_validation_unix_ms",
            "response_reservoir", "reservoir_seen" })) return false;
    try {
        if (!exact_unsigned(value.at("slot"), out.slot) ||
            out.slot >= server_cache_observation_store::instance_capacity ||
            !parse_key(value.at("key"), out.key) ||
            !parse_authority_terminal(value.at("authority_terminal"), out.authority_terminal)) return false;
        if (!exact_unsigned(value.at("fit_generation"), out.fit_generation) ||
            !exact_unsigned(value.at("tail_actual_max_us"), out.tail_actual_max_us)) return false;
        const auto & v = value.at("V_bits");
        if (!v.is_array() || v.size() != out.v.size()) return false;
        for (size_t i = 0; i < out.v.size(); ++i) {
            if (!parse_doubles(v[i], out.v[i])) return false;
        }
        if (!parse_doubles(value.at("b_bits"), out.b) ||
            !parse_doubles(value.at("feature_min_bits"), out.feature_min) ||
            !parse_doubles(value.at("feature_max_bits"), out.feature_max) ||
            !parse_doubles(value.at("log_wealth_bits"), out.log_wealth)) return false;
        if (!exact_unsigned(value.at("n_fit"), out.n_fit) ||
            !exact_unsigned(value.at("qualified_execution_ordinal"),
                            out.qualified_execution_ordinal) ||
            !exact_unsigned(value.at("n_validation"), out.n_validation)) return false;
        if (!parse_unsigned_array(value.at("fit_region_minutes"),
                                  out.fit_region_minutes) ||
            !parse_unsigned_array(value.at("validation_region_minutes"),
                                  out.validation_region_minutes) ||
            !monotonic_regions(out.fit_region_minutes) ||
            !monotonic_regions(out.validation_region_minutes)) return false;
        if (!exact_unsigned(value.at("safe_measurable_opportunities"),
                            out.safe_measurable_opportunities) ||
            !exact_unsigned(value.at("opportunity_at_last_validation"),
                            out.opportunity_at_last_validation)) return false;
        if (out.opportunity_at_last_validation > out.safe_measurable_opportunities) return false;
        if (!exact_unsigned(value.at("last_fit_unix_ms"), out.last_fit_unix_ms) ||
            !exact_unsigned(value.at("last_validation_unix_ms"),
                            out.last_validation_unix_ms)) return false;
        const auto & reservoir = value.at("response_reservoir");
        if (!reservoir.is_array() || reservoir.size() != out.response_reservoir.size()) return false;
        for (size_t i = 0; i < out.response_reservoir.size(); ++i) {
            if (!exact_unsigned(reservoir[i], out.response_reservoir[i])) return false;
        }
        if (!exact_unsigned(value.at("reservoir_seen"), out.reservoir_seen)) return false;
        for (uint8_t i = 0; i < out.key.feature_dim; ++i) {
            if (out.feature_min[i] > out.feature_max[i] ||
                out.v[i][i] < 1.0 || out.v[i][i] < 0.0) return false;
            for (uint8_t j = 0; j < out.key.feature_dim; ++j) {
                if (out.v[i][j] != out.v[j][i]) return false;
            }
        }
        if (!positive_definite(out.v, out.key.feature_dim)) return false;
        if (out.authority_terminal ==
                server_cache_calibration_authority_terminal::tail_exceeded &&
            out.tail_actual_max_us == 0) return false;
        return true;
    } catch (...) {
        return false;
    }
}

std::array<uint8_t, 32> envelope_digest(
        const uint8_t schema[4], const uint8_t length[4],
        const uint8_t * payload, size_t payload_size) {
    llama_sha256 hash;
    hash.update(DIGEST_DOMAIN, sizeof(DIGEST_DOMAIN));
    hash.update(schema, 4);
    hash.update(length, 4);
    hash.update(payload, payload_size);
    return hash.finish();
}

bool encode_envelope(const json & payload, size_t limit,
                     std::vector<uint8_t> & out) {
    const std::string text = payload.dump();
    if (text.size() > limit || text.size() > UINT32_MAX) return false;
    out.clear();
    out.reserve(ENVELOPE_FIXED + text.size());
    out.insert(out.end(), MAGIC, MAGIC + 8);
    append_u32(out, STORE_SCHEMA);
    append_u32(out, uint32_t(text.size()));
    out.insert(out.end(), text.begin(), text.end());
    const auto digest = envelope_digest(
        out.data() + 8, out.data() + 12,
        reinterpret_cast<const uint8_t *>(text.data()), text.size());
    out.insert(out.end(), digest.begin(), digest.end());
    return true;
}

bool decode_envelope(const uint8_t * data, size_t size, size_t limit,
                     const char * expected_object, json & out) {
    if (!data || size < ENVELOPE_FIXED ||
        std::memcmp(data, MAGIC, 8) != 0 || read_u32(data + 8) != STORE_SCHEMA) return false;
    const uint32_t length = read_u32(data + 12);
    if (length > limit || size != ENVELOPE_FIXED + size_t(length)) return false;
    const auto expected = envelope_digest(data + 8, data + 12, data + 16, length);
    if (!std::equal(expected.begin(), expected.end(), data + 16 + length)) return false;
    bool refused = false;
    size_t events = 0;
    std::vector<std::set<std::string>> object_keys;
    auto callback = [&](int depth, json::parse_event_t event, json & parsed) {
        if (++events > JSON_EVENT_LIMIT || depth < 0 ||
            depth > JSON_DEPTH_LIMIT) {
            refused = true;
            return false;
        }
        if (event == json::parse_event_t::object_start) {
            object_keys.emplace_back();
        } else if (event == json::parse_event_t::key) {
            if (object_keys.empty() || !parsed.is_string()) {
                refused = true;
                return false;
            }
            const std::string key = parsed.get<std::string>();
            if (key.size() > JSON_KEY_LIMIT ||
                !object_keys.back().insert(key).second) {
                refused = true;
                return false;
            }
        } else if (event == json::parse_event_t::object_end) {
            if (object_keys.empty()) {
                refused = true;
                return false;
            }
            object_keys.pop_back();
        } else if (event == json::parse_event_t::value && parsed.is_string() &&
                   parsed.get_ref<const std::string &>().size() >
                       JSON_STRING_LIMIT) {
            refused = true;
            return false;
        }
        return !refused;
    };
    try {
        out = json::parse(data + 16, data + 16 + length, callback, true, false);
        return !refused && object_keys.empty() && out.is_object() &&
            out.value("object", "") == expected_object &&
            out.value("schema_version", 0u) == STORE_SCHEMA &&
            out.value("estimator_version", 0u) == ESTIMATOR_VERSION;
    } catch (...) {
        return false;
    }
}

json manifest_json(const server_cache_calibration_manifest & value) {
    json profiles = json::array();
    for (const auto & profile : value.profiles) {
        profiles.push_back({
            { "profile_generation_ordinal", profile.profile_generation_ordinal },
            { "profile_identity_digest", hex_digest(profile.profile_identity_digest) },
            { "profile_file_generation", profile.profile_file_generation },
            { "profile_payload_digest", hex_digest(profile.profile_payload_digest) },
            { "persisted_prune_recency", profile.persisted_prune_recency },
        });
    }
    return {
        { "object", "cache_calibration_store" },
        { "schema_version", STORE_SCHEMA },
        { "estimator_version", ESTIMATOR_VERSION },
        { "store_lineage_id", hex_digest(value.store_lineage_id) },
        { "next_boot_claim_ordinal", value.next_boot_claim_ordinal },
        { "next_profile_generation_ordinal", value.next_profile_generation_ordinal },
        { "next_persisted_prune_epoch", value.next_persisted_prune_epoch },
        { "next_immutable_file_ordinal", value.next_immutable_file_ordinal },
        { "generation", value.generation },
        { "profiles", std::move(profiles) },
        { "last_update_unix_ms", value.last_update_unix_ms },
    };
}

json profile_json(const std::array<uint8_t, 32> & lineage,
                  const server_cache_calibration_profile_snapshot & value) {
    json instances = json::array();
    for (const auto & instance : value.instances) instances.push_back(instance_json(instance));
    return {
        { "object", "cache_calibration_profile" },
        { "schema_version", STORE_SCHEMA },
        { "estimator_version", ESTIMATOR_VERSION },
        { "store_lineage_id", hex_digest(lineage) },
        { "profile_generation_ordinal", value.profile_generation_ordinal },
        { "profile_file_generation", value.profile_file_generation },
        { "persisted_prune_recency", value.persisted_prune_recency },
        { "mutation_generation", value.mutation_generation },
        { "profile_identity_digest", hex_digest(value.profile_identity_digest) },
        { "identity_exact", value.identity_exact },
        { "instances", std::move(instances) },
        { "bounded_diagnostic_residual_reservoir", json::object() },
        { "profile_last_update_unix_ms", value.profile_last_update_unix_ms },
    };
}

#if !defined(_WIN32)
bool validated_regular_stat(const struct stat & status) {
    return S_ISREG(status.st_mode) && status.st_uid == geteuid() &&
        status.st_nlink == 1 && (status.st_mode & 0777) == 0600;
}
#endif

bool read_file_bounded(int directory_descriptor, const std::string & directory,
                       const std::string & name, size_t limit,
                       std::vector<uint8_t> & out) {
#if defined(_WIN32)
    const std::string path = (fs::path(directory) / name).string();
    std::error_code ec;
    const uintmax_t size = fs::file_size(path, ec);
    if (ec || size > limit || size > SIZE_MAX) return false;
    out.resize(size_t(size));
    std::ifstream file(path, std::ios::binary);
    return file && (out.empty() || bool(file.read(
        reinterpret_cast<char *>(out.data()), std::streamsize(out.size()))));
#else
    (void) directory;
    const int fd = openat(directory_descriptor, name.c_str(),
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat status = {};
    const bool valid = fstat(fd, &status) == 0 &&
        validated_regular_stat(status) && status.st_size >= 0 &&
        uintmax_t(status.st_size) <= limit &&
        uintmax_t(status.st_size) <= SIZE_MAX;
    if (!valid) {
        close(fd);
        return false;
    }
    out.resize(size_t(status.st_size));
    size_t offset = 0;
    while (offset < out.size()) {
        const ssize_t got = read(fd, out.data() + offset, out.size() - offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            close(fd);
            out.clear();
            return false;
        }
        offset += size_t(got);
    }
    struct stat named_status = {};
    const bool stable = fstat(fd, &status) == 0 &&
        validated_regular_stat(status) &&
        size_t(status.st_size) == out.size() &&
        fstatat(directory_descriptor, name.c_str(), &named_status,
                AT_SYMLINK_NOFOLLOW) == 0 &&
        named_status.st_dev == status.st_dev &&
        named_status.st_ino == status.st_ino;
    close(fd);
    if (!stable) out.clear();
    return stable;
#endif
}

bool sync_descriptor(int fd) {
#if defined(_WIN32)
    return _commit(fd) == 0;
#else
    return fsync(fd) == 0;
#endif
}

void close_descriptor(int fd) {
#if defined(_WIN32)
    if (fd >= 0) _close(fd);
#else
    if (fd >= 0) close(fd);
#endif
}

bool write_all(int fd, const uint8_t * data, size_t size) {
    while (size != 0) {
#if defined(_WIN32)
        const int chunk = int(std::min<size_t>(size, INT_MAX));
        const int wrote = _write(fd, data, chunk);
#else
        const ssize_t wrote = write(fd, data, size);
        if (wrote < 0 && errno == EINTR) continue;
#endif
        if (wrote <= 0) return false;
        data += size_t(wrote);
        size -= size_t(wrote);
    }
    return true;
}

bool write_file_exclusive(int directory_descriptor, const std::string & directory,
                          const std::string & name,
                          const std::vector<uint8_t> & bytes) {
#if defined(_WIN32)
    const std::string path = (fs::path(directory) / name).string();
    const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
                         _S_IREAD | _S_IWRITE);
#else
    (void) directory;
    const int fd = openat(directory_descriptor, name.c_str(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                          0600);
#endif
    if (fd < 0) return false;
#if !defined(_WIN32)
    struct stat created_status = {};
    const bool created_valid = fchmod(fd, 0600) == 0 &&
        fstat(fd, &created_status) == 0 &&
        validated_regular_stat(created_status);
#else
    const bool created_valid = true;
#endif
    const bool ok = created_valid &&
        write_all(fd, bytes.data(), bytes.size()) && sync_descriptor(fd);
    close_descriptor(fd);
    if (!ok) {
#if defined(_WIN32)
        std::error_code ec;
        fs::remove(path, ec);
#else
        unlinkat(directory_descriptor, name.c_str(), 0);
#endif
    }
    return ok;
}

bool replace_file(int directory_descriptor, const std::string & directory,
                  const std::string & name,
                  const std::vector<uint8_t> & bytes) {
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    const std::string temp = name + ".tmp." + std::to_string(pid);
#if defined(_WIN32)
    const std::string path = (fs::path(directory) / name).string();
    const std::string temp_path = (fs::path(directory) / temp).string();
    std::error_code ec;
    fs::remove(temp_path, ec);
    if (!write_file_exclusive(directory_descriptor, directory, temp, bytes)) return false;
    if (!MoveFileExA(temp_path.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        fs::remove(temp_path, ec);
        return false;
    }
#else
    struct stat status = {};
    if (fstatat(directory_descriptor, name.c_str(), &status,
                AT_SYMLINK_NOFOLLOW) == 0) {
        if (!validated_regular_stat(status)) return false;
    } else if (errno != ENOENT) {
        return false;
    }
    if (fstatat(directory_descriptor, temp.c_str(), &status,
                AT_SYMLINK_NOFOLLOW) == 0) {
        if (!validated_regular_stat(status) ||
            unlinkat(directory_descriptor, temp.c_str(), 0) != 0) return false;
    } else if (errno != ENOENT) {
        return false;
    }
    if (!write_file_exclusive(directory_descriptor, directory, temp, bytes) ||
        renameat(directory_descriptor, temp.c_str(),
                 directory_descriptor, name.c_str()) != 0 ||
        fsync(directory_descriptor) != 0) return false;
#endif
    return true;
}

std::string profile_name(uint64_t q, uint64_t f) {
    return "profile-" + std::to_string(q) + "-" + std::to_string(f) + ".bcal";
}

bool profile_filename(const std::string & name,
                      uint64_t * q_out = nullptr,
                      uint64_t * f_out = nullptr) {
    static constexpr char prefix[] = "profile-";
    static constexpr char suffix[] = ".bcal";
    if (name.size() <= sizeof(prefix) - 1 + sizeof(suffix) - 1 ||
        name.compare(0, sizeof(prefix) - 1, prefix) != 0 ||
        name.compare(name.size() - (sizeof(suffix) - 1),
                     sizeof(suffix) - 1, suffix) != 0) return false;
    const size_t body_end = name.size() - (sizeof(suffix) - 1);
    const size_t separator = name.find('-', sizeof(prefix) - 1);
    if (separator == std::string::npos || separator + 1 == body_end) return false;
    for (size_t i = sizeof(prefix) - 1; i < body_end; ++i) {
        if (i == separator) continue;
        if (name[i] < '0' || name[i] > '9') return false;
    }
    auto parse_component = [&](size_t begin, size_t end, uint64_t & out) {
        if (begin == end) return false;
        out = 0;
        for (size_t i = begin; i < end; ++i) {
            const uint64_t digit = uint64_t(name[i] - '0');
            if (out > (UINT64_MAX - digit) / 10) return false;
            out = out * 10 + digit;
        }
        return true;
    };
    uint64_t q = 0;
    uint64_t f = 0;
    if (!parse_component(sizeof(prefix) - 1, separator, q) ||
        !parse_component(separator + 1, body_end, f)) return false;
    if (q_out) *q_out = q;
    if (f_out) *f_out = f;
    return true;
}

bool directory_initializable(int directory_descriptor,
                             const std::string & directory) {
#if defined(_WIN32)
    std::error_code ec;
    size_t count = 0;
    for (const auto & entry : fs::directory_iterator(directory, ec)) {
        if (ec || ++count > 4) return false;
        if (entry.path().filename() != "writer.lock") return false;
    }
    return !ec;
#else
    (void) directory;
    const int scan_descriptor = openat(directory_descriptor, ".",
        O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    DIR * stream = scan_descriptor >= 0 ? fdopendir(scan_descriptor) : nullptr;
    if (!stream) return false;
    bool valid = true;
    size_t count = 0;
    while (const dirent * entry = readdir(stream)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        if (++count > 4 || name != "writer.lock") {
            valid = false;
            break;
        }
    }
    closedir(stream);
    return valid;
#endif
}

bool random_lineage(std::array<uint8_t, 32> & out) {
#if defined(__linux__)
    size_t offset = 0;
    while (offset < out.size()) {
        const ssize_t got = getrandom(out.data() + offset, out.size() - offset, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        offset += size_t(got);
    }
    return true;
#elif defined(_WIN32)
    return BCryptGenRandom(nullptr, out.data(), ULONG(out.size()),
                           BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
      defined(__NetBSD__)
    arc4random_buf(out.data(), out.size());
    return true;
#else
    return false;
#endif
}

uint64_t unix_ms() {
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

bool profile_nonregressed(
        const server_cache_calibration_profile_snapshot & older,
        const server_cache_calibration_profile_snapshot & newer) {
    if (older.profile_identity_digest != newer.profile_identity_digest ||
        newer.mutation_generation <= older.mutation_generation ||
        newer.instances.size() < older.instances.size()) return false;
    for (const auto & old_instance : older.instances) {
        const auto found = std::find_if(newer.instances.begin(), newer.instances.end(),
            [&](const auto & candidate) { return candidate.slot == old_instance.slot; });
        if (found == newer.instances.end() ||
            found->fit_generation < old_instance.fit_generation) return false;
        if (found->fit_generation > old_instance.fit_generation) {
            // A checked new fit generation may reuse the bounded physical
            // slot with a new key and reset moments. Sticky authority
            // terminals remain lineage-wide until an explicit store reset.
            if (old_instance.authority_terminal !=
                    server_cache_calibration_authority_terminal::none &&
                found->authority_terminal != old_instance.authority_terminal) {
                return false;
            }
            continue;
        }
        if (!(found->key == old_instance.key) ||
            found->n_fit < old_instance.n_fit ||
            found->qualified_execution_ordinal < old_instance.qualified_execution_ordinal ||
            found->n_validation < old_instance.n_validation ||
            found->safe_measurable_opportunities < old_instance.safe_measurable_opportunities ||
            found->opportunity_at_last_validation < old_instance.opportunity_at_last_validation ||
            found->reservoir_seen < old_instance.reservoir_seen ||
            (old_instance.authority_terminal !=
                 server_cache_calibration_authority_terminal::none &&
             found->authority_terminal != old_instance.authority_terminal)) return false;
        for (size_t i = 0; i < old_instance.v.size(); ++i) {
            if (found->b[i] < old_instance.b[i]) return false;
            for (size_t j = 0; j < old_instance.v[i].size(); ++j) {
                if (found->v[i][j] < old_instance.v[i][j]) return false;
            }
        }
    }
    return true;
}

} // namespace

bool server_cache_calibration_validate_profile(
        const server_cache_calibration_profile_snapshot & value,
        const server_cache_calibration_profile_snapshot * previous) noexcept {
    try {
        return snapshot_semantically_valid(value) &&
            (!previous || profile_nonregressed(*previous, value));
    } catch (...) {
        return false;
    }
}

void server_cache_calibration_set_test_fault(
        server_cache_calibration_test_fault value) noexcept {
    test_fault.store(value, std::memory_order_release);
}

uint64_t server_cache_calibration_test_fault_hits() noexcept {
    return test_fault_hits.load(std::memory_order_acquire);
}

bool server_cache_calibration_encode_manifest(
        const server_cache_calibration_manifest & value,
        std::vector<uint8_t> & out) noexcept {
    try {
        out.clear();
        if (!manifest_semantically_valid(value)) return false;
        return encode_envelope(manifest_json(value), ROOT_PAYLOAD_LIMIT, out);
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_calibration_decode_manifest(
        const uint8_t * data, size_t size,
        server_cache_calibration_manifest & out) noexcept {
    out = {};
    try {
        json value;
        if (!decode_envelope(data, size, ROOT_PAYLOAD_LIMIT,
                             "cache_calibration_store", value) ||
            !exact_keys(value, {
                "object", "schema_version", "estimator_version",
                "store_lineage_id", "next_boot_claim_ordinal",
                "next_profile_generation_ordinal", "next_persisted_prune_epoch",
                "next_immutable_file_ordinal", "generation", "profiles",
                "last_update_unix_ms" }) ||
            !parse_digest(value.at("store_lineage_id"), out.store_lineage_id) ||
            !nonzero_digest(out.store_lineage_id) ||
            !exact_unsigned(value.at("next_boot_claim_ordinal"),
                            out.next_boot_claim_ordinal) ||
            !exact_unsigned(value.at("next_profile_generation_ordinal"),
                            out.next_profile_generation_ordinal) ||
            !exact_unsigned(value.at("next_persisted_prune_epoch"),
                            out.next_persisted_prune_epoch) ||
            !exact_unsigned(value.at("next_immutable_file_ordinal"),
                            out.next_immutable_file_ordinal) ||
            !exact_unsigned(value.at("generation"), out.generation) ||
            !exact_unsigned(value.at("last_update_unix_ms"),
                            out.last_update_unix_ms)) return false;
        const auto & profiles = value.at("profiles");
        if (!profiles.is_array() || profiles.size() > MAX_PROFILES) return false;
        for (const auto & row : profiles) {
            if (!exact_keys(row, {
                    "profile_generation_ordinal", "profile_identity_digest",
                    "profile_file_generation", "profile_payload_digest",
                    "persisted_prune_recency" })) return false;
            server_cache_calibration_profile_reference ref;
            if (!exact_unsigned(row.at("profile_generation_ordinal"),
                                ref.profile_generation_ordinal) ||
                !exact_unsigned(row.at("profile_file_generation"),
                                ref.profile_file_generation) ||
                !exact_unsigned(row.at("persisted_prune_recency"),
                                ref.persisted_prune_recency) ||
                !parse_digest(row.at("profile_identity_digest"), ref.profile_identity_digest) ||
                !parse_digest(row.at("profile_payload_digest"), ref.profile_payload_digest) ||
                !nonzero_digest(ref.profile_identity_digest) ||
                ref.profile_generation_ordinal >= out.next_profile_generation_ordinal ||
                ref.profile_file_generation >= out.next_immutable_file_ordinal ||
                ref.persisted_prune_recency >= out.next_persisted_prune_epoch ||
                !out.profiles.push_back(ref)) return false;
        }
        std::sort(out.profiles.begin(), out.profiles.end(), [](const auto & a, const auto & b) {
            return a.profile_generation_ordinal < b.profile_generation_ordinal;
        });
        std::set<std::array<uint8_t, 32>> identities;
        std::set<uint64_t> files;
        std::set<uint64_t> prune_epochs;
        for (size_t i = 0; i < out.profiles.size(); ++i) {
            if ((i != 0 && out.profiles[i - 1].profile_generation_ordinal ==
                           out.profiles[i].profile_generation_ordinal) ||
                !identities.insert(out.profiles[i].profile_identity_digest).second ||
                !files.insert(out.profiles[i].profile_file_generation).second ||
                !prune_epochs.insert(out.profiles[i].persisted_prune_recency).second) {
                return false;
            }
        }
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_encode_profile(
        const std::array<uint8_t, 32> & store_lineage_id,
        const server_cache_calibration_profile_snapshot & value,
        std::vector<uint8_t> & out) noexcept {
    try {
        out.clear();
        if (!nonzero_digest(store_lineage_id) ||
            !server_cache_calibration_validate_profile(value)) return false;
        return encode_envelope(profile_json(store_lineage_id, value),
                               PROFILE_PAYLOAD_LIMIT, out);
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_calibration_decode_profile(
        const uint8_t * data, size_t size,
        const std::array<uint8_t, 32> & expected_lineage_id,
        server_cache_calibration_profile_snapshot & out) noexcept {
    out = {};
    try {
        json value;
        std::array<uint8_t, 32> lineage = {};
        if (!decode_envelope(data, size, PROFILE_PAYLOAD_LIMIT,
                             "cache_calibration_profile", value) ||
            !exact_keys(value, {
                "object", "schema_version", "estimator_version",
                "store_lineage_id", "profile_generation_ordinal",
                "profile_file_generation", "persisted_prune_recency",
                "mutation_generation", "profile_identity_digest",
                "identity_exact", "instances",
                "bounded_diagnostic_residual_reservoir",
                "profile_last_update_unix_ms" }) ||
            !parse_digest(value.at("store_lineage_id"), lineage) ||
            lineage != expected_lineage_id ||
            !parse_digest(value.at("profile_identity_digest"), out.profile_identity_digest) ||
            !exact_unsigned(value.at("profile_generation_ordinal"),
                            out.profile_generation_ordinal) ||
            !exact_unsigned(value.at("profile_file_generation"),
                            out.profile_file_generation) ||
            !exact_unsigned(value.at("persisted_prune_recency"),
                            out.persisted_prune_recency) ||
            !exact_unsigned(value.at("mutation_generation"),
                            out.mutation_generation) ||
            !nonzero_digest(out.profile_identity_digest)) return false;
        out.identity_exact = value.at("identity_exact").get<bool>();
        if (!exact_unsigned(value.at("profile_last_update_unix_ms"),
                            out.profile_last_update_unix_ms)) return false;
        const auto & instances = value.at("instances");
        if (!instances.is_array() ||
            instances.size() > server_cache_observation_store::instance_capacity ||
            !value.at("bounded_diagnostic_residual_reservoir").is_object() ||
            !value.at("bounded_diagnostic_residual_reservoir").empty()) return false;
        std::array<bool, server_cache_observation_store::instance_capacity> used = {};
        for (const auto & row : instances) {
            server_cache_calibration_instance_snapshot instance;
            if (!parse_instance(row, instance) || used[instance.slot]) return false;
            used[instance.slot] = true;
            if (!out.instances.push_back(std::move(instance))) return false;
        }
        return server_cache_calibration_validate_profile(out);
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_snapshot_observer(
        const server_cache_observation_store & store,
        server_cache_calibration_profile_snapshot & out) noexcept {
    out = {};
    try {
        const auto & fingerprint = store.execution_fingerprint();
        if (!fingerprint.complete || store.mutation_generation() == 0) return false;
        out.profile_identity_digest = fingerprint.execution_root;
        out.identity_exact = fingerprint.exact;
        out.mutation_generation = store.mutation_generation();
        const auto & source = store.instances();
        for (uint32_t slot = 0; slot < source.size(); ++slot) {
            if (!source[slot].used) continue;
            server_cache_calibration_instance_snapshot instance;
            instance.slot = slot;
            instance.key = source[slot].key;
            instance.fit_generation = source[slot].fit_generation;
            instance.authority_terminal = source[slot].authority_terminal;
            instance.tail_actual_max_us = source[slot].tail_actual_max_us;
            instance.v = source[slot].v;
            instance.b = source[slot].b;
            instance.n_fit = source[slot].n_success;
            instance.feature_min = source[slot].feature_min;
            instance.feature_max = source[slot].feature_max;
            instance.qualified_execution_ordinal =
                source[slot].qualified_execution_ordinal;
            instance.log_wealth = source[slot].log_wealth;
            instance.n_validation = source[slot].n_validation;
            for (uint8_t i = 0; i < source[slot].fit_region_count; ++i) {
                if (!instance.fit_region_minutes.push_back(
                        source[slot].fit_region_minutes[i])) return false;
            }
            for (uint8_t i = 0;
                 i < source[slot].validation_region_count; ++i) {
                if (!instance.validation_region_minutes.push_back(
                        source[slot].validation_region_minutes[i])) return false;
            }
            instance.safe_measurable_opportunities =
                source[slot].safe_measurable_opportunities;
            instance.opportunity_at_last_validation =
                source[slot].opportunity_at_last_validation;
            instance.last_fit_unix_ms = source[slot].last_fit_unix_ms;
            instance.last_validation_unix_ms =
                source[slot].last_validation_unix_ms;
            instance.response_reservoir = source[slot].response_reservoir;
            instance.reservoir_seen = source[slot].reservoir_seen;
            if (!out.instances.push_back(std::move(instance))) return false;
        }
        return server_cache_calibration_validate_profile(out);
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_restore_observer(
        const server_cache_calibration_profile_snapshot & value,
        server_cache_observation_store & store) noexcept {
    if (!server_cache_calibration_validate_profile(value) ||
        value.profile_identity_digest !=
            store.execution_fingerprint().execution_root ||
        value.mutation_generation == 0 ||
        value.instances.size() > server_cache_observation_store::instance_capacity) {
        return false;
    }
    std::array<server_cache_observation_instance,
               server_cache_observation_store::instance_capacity> instances = {};
    for (const auto & source : value.instances) {
        if (source.slot >= instances.size() || instances[source.slot].used ||
            source.key.profile_execution_digest != value.profile_identity_digest) {
            return false;
        }
        auto & target = instances[source.slot];
        target.used = true;
        target.key = source.key;
        target.key.identity_exact = store.execution_fingerprint().exact;
        target.fit_generation = source.fit_generation;
        target.authority_terminal = source.authority_terminal;
        target.v = source.v;
        target.b = source.b;
        target.n_success = source.n_fit;
        target.feature_min = source.feature_min;
        target.feature_max = source.feature_max;
        target.qualified_execution_ordinal =
            source.qualified_execution_ordinal;
        target.log_wealth = source.log_wealth;
        target.n_validation = source.n_validation;
        target.fit_region_count = uint8_t(source.fit_region_minutes.size());
        std::copy(source.fit_region_minutes.begin(),
                  source.fit_region_minutes.end(),
                  target.fit_region_minutes.begin());
        target.validation_region_count =
            uint8_t(source.validation_region_minutes.size());
        std::copy(source.validation_region_minutes.begin(),
                  source.validation_region_minutes.end(),
                  target.validation_region_minutes.begin());
        target.safe_measurable_opportunities =
            source.safe_measurable_opportunities;
        target.opportunity_at_last_validation =
            source.opportunity_at_last_validation;
        target.last_fit_unix_ms = source.last_fit_unix_ms;
        target.last_validation_unix_ms = source.last_validation_unix_ms;
        target.response_reservoir = source.response_reservoir;
        target.reservoir_seen = source.reservoir_seen;
        target.tail_exceeded = source.authority_terminal ==
            server_cache_calibration_authority_terminal::tail_exceeded;
        target.tail_actual_max_us = source.tail_actual_max_us;
    }
    return store.restore_persisted_instances(instances, value.mutation_generation);
}

server_cache_calibration_store::~server_cache_calibration_store() {
    close();
}

bool server_cache_calibration_store::create_lineage() noexcept {
    manifest_ = {};
    if (!random_lineage(manifest_.store_lineage_id)) return false;
    manifest_.next_boot_claim_ordinal = 0;
    manifest_.next_profile_generation_ordinal = 0;
    manifest_.next_persisted_prune_epoch = 0;
    manifest_.next_immutable_file_ordinal = 0;
    manifest_.last_update_unix_ms = unix_ms();
    return commit_manifest();
}

bool server_cache_calibration_store::commit_manifest() noexcept {
    if (failed_ || manifest_.generation == UINT64_MAX) return false;
    auto next = manifest_;
    ++next.generation;
    next.last_update_unix_ms = unix_ms();
    std::vector<uint8_t> bytes;
    if (!server_cache_calibration_encode_manifest(next, bytes) ||
        take_test_fault(
            server_cache_calibration_test_fault::manifest_replace_once) ||
        !replace_file(directory_descriptor_, directory_, "manifest.bcal", bytes)) {
        failed_ = true;
        return false;
    }
    manifest_ = std::move(next);
    return true;
}

bool server_cache_calibration_store::garbage_collect_profiles() noexcept {
    if (failed_) return false;
    try {
        std::set<std::string> live;
        for (const auto & profile : manifest_.profiles) {
            live.insert(profile_name(profile.profile_generation_ordinal,
                                     profile.profile_file_generation));
        }
        size_t entries = 0;
        uint64_t bytes_seen = 0;
        bool removed = false;
#if defined(_WIN32)
        std::error_code ec;
        for (const auto & entry : fs::directory_iterator(directory_, ec)) {
            if (ec || ++entries > STORE_ENTRY_LIMIT) return false;
            const std::string name = entry.path().filename().string();
            const auto status = entry.symlink_status(ec);
            if (ec || !fs::is_regular_file(status) ||
                fs::hard_link_count(entry.path(), ec) != 1 || ec) return false;
            const uintmax_t child_size = entry.file_size(ec);
            if (ec || child_size > STORE_BYTE_LIMIT - bytes_seen) return false;
            bytes_seen += uint64_t(child_size);
            if (name == "writer.lock" || name == "manifest.bcal") continue;
            if (name.rfind("manifest.bcal.tmp.", 0) == 0) {
                if (!fs::remove(entry.path(), ec) || ec) return false;
                removed = true;
                continue;
            }
            uint64_t q = 0;
            uint64_t f = 0;
            if (!profile_filename(name, &q, &f) || name != profile_name(q, f)) return false;
            std::vector<uint8_t> data;
            server_cache_calibration_profile_snapshot profile;
            if (!read_file_bounded(directory_descriptor_, directory_, name,
                                   PROFILE_PAYLOAD_LIMIT + ENVELOPE_FIXED, data) ||
                !server_cache_calibration_decode_profile(data.data(), data.size(),
                                                          manifest_.store_lineage_id,
                                                          profile) ||
                profile.profile_generation_ordinal != q ||
                profile.profile_file_generation != f) return false;
            if (live.find(name) == live.end()) {
                if (!fs::remove(entry.path(), ec) || ec) return false;
                removed = true;
            }
        }
        return !ec;
#else
        const int scan_descriptor = openat(directory_descriptor_, ".",
            O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        DIR * stream = scan_descriptor >= 0 ? fdopendir(scan_descriptor) : nullptr;
        if (!stream) return false;
        bool valid = true;
        while (const dirent * entry = readdir(stream)) {
            const std::string name = entry->d_name;
            if (name == "." || name == "..") continue;
            if (++entries > STORE_ENTRY_LIMIT) {
                valid = false;
                break;
            }
            struct stat status = {};
            if (fstatat(directory_descriptor_, name.c_str(), &status,
                        AT_SYMLINK_NOFOLLOW) != 0 ||
                !validated_regular_stat(status) || status.st_size < 0 ||
                uint64_t(status.st_size) > STORE_BYTE_LIMIT - bytes_seen) {
                valid = false;
                break;
            }
            bytes_seen += uint64_t(status.st_size);
            if (name == "writer.lock" || name == "manifest.bcal") continue;
            if (name.rfind("manifest.bcal.tmp.", 0) == 0) {
                if (unlinkat(directory_descriptor_, name.c_str(), 0) != 0) {
                    valid = false;
                    break;
                }
                removed = true;
                continue;
            }
            uint64_t q = 0;
            uint64_t f = 0;
            if (!profile_filename(name, &q, &f) || name != profile_name(q, f)) {
                valid = false;
                break;
            }
            std::vector<uint8_t> data;
            server_cache_calibration_profile_snapshot profile;
            if (!read_file_bounded(directory_descriptor_, directory_, name,
                                   PROFILE_PAYLOAD_LIMIT + ENVELOPE_FIXED, data) ||
                !server_cache_calibration_decode_profile(data.data(), data.size(),
                                                          manifest_.store_lineage_id,
                                                          profile) ||
                profile.profile_generation_ordinal != q ||
                profile.profile_file_generation != f) {
                valid = false;
                break;
            }
            if (live.find(name) == live.end()) {
                if (unlinkat(directory_descriptor_, name.c_str(), 0) != 0) {
                    valid = false;
                    break;
                }
                removed = true;
            }
        }
        closedir(stream);
        return valid && (!removed || fsync(directory_descriptor_) == 0);
#endif
    } catch (...) {
        return false;
    }
}

server_cache_calibration_load_status server_cache_calibration_store::open(
        const std::string & directory) noexcept {
    close();
#if defined(_WIN32)
    // v1's owner/mode/link and descriptor-relative contract has no reviewed
    // Windows HANDLE/reparse-point implementation. Persistence therefore
    // refuses as unsupported while inference remains available.
    (void) directory;
    return server_cache_calibration_load_status::unsupported;
#else
    try {
        if (directory.empty()) return server_cache_calibration_load_status::io_fault;
        std::error_code ec;
        directory_ = directory;
        failed_ = false;
        const fs::path store_path(directory);
        const fs::path parent_path = store_path.parent_path();
        const std::string leaf = store_path.filename().string();
        if (parent_path.empty() || leaf.empty() || leaf == "." || leaf == "..") {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        fs::create_directories(parent_path, ec);
        if (ec) {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        const int parent_descriptor = ::open(parent_path.c_str(),
            O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (parent_descriptor < 0) {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        bool created = false;
        if (mkdirat(parent_descriptor, leaf.c_str(), 0700) == 0) {
            created = true;
            if (fchmodat(parent_descriptor, leaf.c_str(), 0700, 0) != 0) {
                ::close(parent_descriptor);
                close();
                return server_cache_calibration_load_status::io_fault;
            }
        } else if (errno != EEXIST) {
            ::close(parent_descriptor);
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        directory_descriptor_ = openat(parent_descriptor, leaf.c_str(),
            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        struct stat directory_status = {};
        const bool directory_stat_valid = directory_descriptor_ >= 0 &&
            fstat(directory_descriptor_, &directory_status) == 0;
        const mode_t directory_mode = directory_status.st_mode & 0777;
        const bool directory_valid = directory_stat_valid &&
            S_ISDIR(directory_status.st_mode) &&
            directory_status.st_uid == geteuid() &&
            (created ? fchmod(directory_descriptor_, 0700) == 0
                     : directory_mode == 0700) &&
            (!created || fsync(parent_descriptor) == 0) &&
            fsync(directory_descriptor_) == 0;
        ::close(parent_descriptor);
        if (!directory_valid) {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        const std::string lock_path = (fs::path(directory_) / "writer.lock").string();
#if defined(_WIN32)
        const errno_t lock_error = _sopen_s(
            &lock_descriptor_, lock_path.c_str(),
            _O_RDWR | _O_CREAT | _O_BINARY, _SH_DENYRW,
            _S_IREAD | _S_IWRITE);
        if (lock_error != 0) {
            lock_descriptor_ = -1;
        }
#else
        bool lock_created = false;
        lock_descriptor_ = openat(directory_descriptor_, "writer.lock",
            O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (lock_descriptor_ >= 0) {
            lock_created = true;
        } else if (errno == EEXIST) {
            lock_descriptor_ = openat(directory_descriptor_, "writer.lock",
                O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        }
#endif
        if (lock_descriptor_ < 0) {
#if defined(_WIN32)
            return lock_error == EACCES || lock_error == EAGAIN
#else
            return errno == EACCES || errno == EAGAIN
#endif
                ? server_cache_calibration_load_status::busy
                : server_cache_calibration_load_status::io_fault;
        }
#if !defined(_WIN32)
        struct stat lock_status = {};
        if ((lock_created && fchmod(lock_descriptor_, 0600) != 0) ||
            fstat(lock_descriptor_, &lock_status) != 0 ||
            !validated_regular_stat(lock_status)) {
            close();
            return server_cache_calibration_load_status::corrupt;
        }
        if (flock(lock_descriptor_, LOCK_EX | LOCK_NB) != 0) {
            close();
            return server_cache_calibration_load_status::busy;
        }
#endif
#if defined(_WIN32)
        const bool root_exists = fs::exists(
            fs::path(directory_) / "manifest.bcal", ec);
        if (ec) {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
#else
        struct stat root_status = {};
        const bool root_exists = fstatat(directory_descriptor_, "manifest.bcal",
            &root_status, AT_SYMLINK_NOFOLLOW) == 0;
        if (!root_exists && errno != ENOENT) {
            close();
            return server_cache_calibration_load_status::corrupt;
        }
#endif
        if (!root_exists) {
            if (!directory_initializable(directory_descriptor_, directory_)) {
                close();
                return server_cache_calibration_load_status::corrupt;
            }
            if (!create_lineage()) {
                close();
                return server_cache_calibration_load_status::io_fault;
            }
        } else {
            std::vector<uint8_t> bytes;
            if (!read_file_bounded(directory_descriptor_, directory_, "manifest.bcal",
                                   ROOT_PAYLOAD_LIMIT + ENVELOPE_FIXED, bytes) ||
                !server_cache_calibration_decode_manifest(
                    bytes.data(), bytes.size(), manifest_)) {
                close();
                return server_cache_calibration_load_status::corrupt;
            }
        }
        if (manifest_.next_boot_claim_ordinal == UINT64_MAX ||
            manifest_.generation == UINT64_MAX) {
            close();
            return server_cache_calibration_load_status::ordinal_exhausted;
        }
        boot_claim_ordinal_ = manifest_.next_boot_claim_ordinal++;
        if (!commit_manifest()) {
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        if (!garbage_collect_profiles()) {
            failed_ = true;
            close();
            return server_cache_calibration_load_status::io_fault;
        }
        return server_cache_calibration_load_status::ok;
    } catch (...) {
        close();
        return server_cache_calibration_load_status::io_fault;
    }
#endif
}

void server_cache_calibration_store::close() noexcept {
    close_descriptor(lock_descriptor_);
    lock_descriptor_ = -1;
#if !defined(_WIN32)
    close_descriptor(directory_descriptor_);
#endif
    directory_descriptor_ = -1;
    boot_claim_ordinal_ = 0;
    directory_.clear();
    manifest_ = {};
    failed_ = false;
}

server_cache_calibration_load_status server_cache_calibration_store::load_profiles(
        server_cache_calibration_profile_set & out) noexcept {
    out.clear();
    if (!is_open()) return server_cache_calibration_load_status::io_fault;
    try {
        for (const auto & ref : manifest_.profiles) {
            server_cache_calibration_profile_snapshot profile;
            if (!load_referenced_profile(ref, profile)) {
                out.clear();
                return server_cache_calibration_load_status::corrupt;
            }
            if (!out.push_back(std::move(profile))) {
                out.clear();
                return server_cache_calibration_load_status::capacity;
            }
        }
        return server_cache_calibration_load_status::ok;
    } catch (...) {
        out.clear();
        return server_cache_calibration_load_status::io_fault;
    }
}

bool server_cache_calibration_store::load_referenced_profile(
        const server_cache_calibration_profile_reference & ref,
        server_cache_calibration_profile_snapshot & out) noexcept {
    out = {};
    try {
        std::vector<uint8_t> bytes;
        if (!read_file_bounded(directory_descriptor_, directory_,
                profile_name(ref.profile_generation_ordinal,
                             ref.profile_file_generation),
                PROFILE_PAYLOAD_LIMIT + ENVELOPE_FIXED, bytes) ||
            bytes.size() < 32 ||
            !std::equal(ref.profile_payload_digest.begin(),
                        ref.profile_payload_digest.end(), bytes.end() - 32) ||
            !server_cache_calibration_decode_profile(
                bytes.data(), bytes.size(), manifest_.store_lineage_id, out) ||
            out.profile_generation_ordinal != ref.profile_generation_ordinal ||
            out.profile_file_generation != ref.profile_file_generation ||
            out.profile_identity_digest != ref.profile_identity_digest ||
            out.persisted_prune_recency != ref.persisted_prune_recency) {
            out = {};
            return false;
        }
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

server_cache_calibration_load_status server_cache_calibration_store::commit_profile(
        server_cache_calibration_profile_snapshot value) noexcept {
    if (!is_open()) return server_cache_calibration_load_status::io_fault;
    try {
        if (!server_cache_calibration_validate_profile(value)) {
            return server_cache_calibration_load_status::corrupt;
        }
        // Remove and authenticate every crash orphan before reserving or
        // writing a new generation. Combined with STORE_COMMIT_HIGH_WATER,
        // this proves the 32 MiB cap cannot be crossed transiently.
        if (!garbage_collect_profiles()) {
            failed_ = true;
            return server_cache_calibration_load_status::io_fault;
        }
        auto existing = std::find_if(manifest_.profiles.begin(), manifest_.profiles.end(),
            [&](const auto & ref) {
                return ref.profile_identity_digest == value.profile_identity_digest;
            });
        const bool is_new = existing == manifest_.profiles.end();
        const size_t existing_index = is_new
            ? 0 : size_t(existing - manifest_.profiles.begin());
        if (!is_new) {
            server_cache_calibration_profile_snapshot prior;
            if (!load_referenced_profile(*existing, prior) ||
                !server_cache_calibration_validate_profile(value, &prior)) {
                return server_cache_calibration_load_status::corrupt;
            }
        }
        if ((is_new && manifest_.next_profile_generation_ordinal == UINT64_MAX) ||
            manifest_.next_persisted_prune_epoch == UINT64_MAX ||
            manifest_.next_immutable_file_ordinal == UINT64_MAX ||
            manifest_.generation > UINT64_MAX - 2 ||
            value.mutation_generation == UINT64_MAX) {
            return server_cache_calibration_load_status::ordinal_exhausted;
        }
        if (is_new) {
            value.profile_generation_ordinal = manifest_.next_profile_generation_ordinal++;
        } else {
            value.profile_generation_ordinal = existing->profile_generation_ordinal;
        }
        value.profile_file_generation = manifest_.next_immutable_file_ordinal++;
        value.persisted_prune_recency = manifest_.next_persisted_prune_epoch++;
        value.profile_last_update_unix_ms = unix_ms();

        // Reserve q/file/prune ordinals in the root before writing the
        // immutable payload. A crash can orphan the reserved file number but
        // can never reuse it on the next boot.
        if (!commit_manifest()) {
            return server_cache_calibration_load_status::io_fault;
        }
        std::vector<uint8_t> bytes;
        if (!server_cache_calibration_encode_profile(
                manifest_.store_lineage_id, value, bytes) ||
            take_test_fault(
                server_cache_calibration_test_fault::profile_write_once) ||
            !write_file_exclusive(directory_descriptor_, directory_,
                profile_name(value.profile_generation_ordinal,
                             value.profile_file_generation), bytes)) {
            return server_cache_calibration_load_status::io_fault;
        }
        server_cache_calibration_profile_reference ref;
        ref.profile_generation_ordinal = value.profile_generation_ordinal;
        ref.profile_identity_digest = value.profile_identity_digest;
        ref.profile_file_generation = value.profile_file_generation;
        ref.persisted_prune_recency = value.persisted_prune_recency;
        std::copy(bytes.end() - 32, bytes.end(), ref.profile_payload_digest.begin());
        if (is_new && manifest_.profiles.size() == MAX_PROFILES) {
            const auto victim = std::min_element(
                manifest_.profiles.begin(), manifest_.profiles.end(),
                [](const auto & a, const auto & b) {
                    return std::tie(a.persisted_prune_recency,
                                    a.profile_identity_digest,
                                    a.profile_generation_ordinal) <
                           std::tie(b.persisted_prune_recency,
                                    b.profile_identity_digest,
                                    b.profile_generation_ordinal);
                });
            manifest_.profiles.erase(victim);
        }
        if (is_new) {
            if (!manifest_.profiles.push_back(ref)) {
                failed_ = true;
                return server_cache_calibration_load_status::capacity;
            }
        } else {
            manifest_.profiles[existing_index] = ref;
        }
        if (take_test_fault(
                server_cache_calibration_test_fault::
                    referencing_manifest_replace_once) ||
            !commit_manifest()) {
            return server_cache_calibration_load_status::io_fault;
        }
        if (!garbage_collect_profiles()) {
            failed_ = true;
            return server_cache_calibration_load_status::io_fault;
        }
        return server_cache_calibration_load_status::ok;
    } catch (...) {
        return server_cache_calibration_load_status::io_fault;
    }
}

server_cache_calibration_writer::~server_cache_calibration_writer() {
    stop();
}

bool server_cache_calibration_writer::start(std::string directory) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ever_started_ || started_ || directory.empty()) return false;
    ever_started_ = true;
    started_ = true;
    stop_ = false;
    health_.store(server_cache_calibration_writer_health::starting,
                  std::memory_order_release);
    try {
        thread_ = std::thread(
            [this, directory = std::move(directory)]() mutable {
                run(std::move(directory));
            });
        return true;
    } catch (...) {
        started_ = false;
        health_.store(server_cache_calibration_writer_health::quarantined,
                      std::memory_order_release);
        return false;
    }
}

void server_cache_calibration_writer::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) return;
        stop_ = true;
        condition_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
    health_.store(server_cache_calibration_writer_health::stopped,
                  std::memory_order_release);
}

bool server_cache_calibration_writer::enqueue(
        const server_cache_calibration_profile_snapshot & value) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!started_ || stop_ ||
        health_.load(std::memory_order_acquire) !=
            server_cache_calibration_writer_health::healthy ||
        value.mutation_generation == 0) return false;
    try {
        if (last_enqueued_identity_ == value.profile_identity_digest &&
            value.mutation_generation <= last_enqueued_mutation_generation_) {
            return false;
        }
        if (pending_ &&
            pending_profile_.profile_identity_digest ==
                value.profile_identity_digest &&
            value.mutation_generation <= pending_profile_.mutation_generation) {
            return false;
        }
        if (pending_ && pending_profile_.profile_identity_digest !=
                            value.profile_identity_digest) return false;
        pending_profile_ = value;
        last_enqueued_identity_ = pending_profile_.profile_identity_digest;
        last_enqueued_mutation_generation_ =
            pending_profile_.mutation_generation;
        pending_ = true;
        condition_.notify_one();
        return true;
    } catch (...) {
        return false;
    }
}

bool server_cache_calibration_writer::poll_loaded(
        server_cache_calibration_load_status & status,
        server_cache_calibration_profile_set & profiles) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!load_ready_ || load_delivered_) return false;
    try {
        profiles.clear();
        if (loaded_profiles_) profiles = std::move(*loaded_profiles_);
        loaded_profiles_.reset();
        status = load_status_;
        load_delivered_ = true;
        return true;
    } catch (...) {
        return false;
    }
}

bool server_cache_calibration_writer::poll_committed(
        server_cache_calibration_commit_ack & out) noexcept {
    if (committed_ack_count_.load(std::memory_order_acquire) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (committed_acks_.empty()) return false;
    out = committed_acks_.front();
    committed_acks_.erase(committed_acks_.begin());
    committed_ack_count_.store(uint8_t(committed_acks_.size()),
                               std::memory_order_release);
    return true;
}

void server_cache_calibration_writer::run(std::string directory) noexcept {
    server_cache_calibration_store store;
    const auto open_status = store.open(directory);
    std::unique_ptr<server_cache_calibration_profile_set> loaded;
    try {
        loaded = std::make_unique<server_cache_calibration_profile_set>();
    } catch (...) {
        health_.store(server_cache_calibration_writer_health::quarantined,
                      std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex_);
        load_status_ = server_cache_calibration_load_status::capacity;
        load_ready_ = true;
        return;
    }
    const auto load_status = open_status == server_cache_calibration_load_status::ok
        ? store.load_profiles(*loaded) : open_status;
    if (open_status == server_cache_calibration_load_status::ok &&
        load_status == server_cache_calibration_load_status::ok) {
        health_.store(server_cache_calibration_writer_health::healthy,
                      std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        load_status_ = load_status;
        loaded_profiles_ = std::move(loaded);
        load_ready_ = true;
    }
    if (open_status != server_cache_calibration_load_status::ok ||
        load_status != server_cache_calibration_load_status::ok) {
        health_.store(server_cache_calibration_writer_health::quarantined,
                      std::memory_order_release);
        return;
    }
    for (;;) {
        server_cache_calibration_profile_snapshot value;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return stop_ || pending_; });
            if (stop_ && !pending_) break;
            value = std::move(pending_profile_);
            pending_ = false;
        }
        unsigned retry = 0;
        for (;;) {
            const auto status = store.commit_profile(value);
            if (status == server_cache_calibration_load_status::ok) {
                const auto found = std::find_if(
                    store.manifest().profiles.begin(),
                    store.manifest().profiles.end(), [&](const auto & ref) {
                        return ref.profile_identity_digest ==
                            value.profile_identity_digest;
                    });
                if (found == store.manifest().profiles.end()) {
                    health_.store(server_cache_calibration_writer_health::quarantined,
                                  std::memory_order_release);
                    return;
                }
                server_cache_calibration_commit_ack ack;
                ack.profile_identity_digest = found->profile_identity_digest;
                ack.profile_generation_ordinal =
                    found->profile_generation_ordinal;
                ack.mutation_generation = value.mutation_generation;
                ack.profile_file_generation = found->profile_file_generation;
                ack.root_generation = store.manifest().generation;
                std::lock_guard<std::mutex> lock(mutex_);
                if (!committed_acks_.push_back(ack)) {
                    health_.store(
                        server_cache_calibration_writer_health::quarantined,
                        std::memory_order_release);
                    return;
                }
                committed_ack_count_.store(uint8_t(committed_acks_.size()),
                                           std::memory_order_release);
                break;
            }
            if (status != server_cache_calibration_load_status::io_fault ||
                !store.is_open()) {
                health_.store(server_cache_calibration_writer_health::quarantined,
                              std::memory_order_release);
                return;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            const auto delay = std::chrono::milliseconds(
                std::min<unsigned>(30000, 100u << std::min<unsigned>(retry++, 8)));
            condition_.wait_for(lock, delay, [&] { return stop_ || pending_; });
            if (stop_) return;
            if (pending_ &&
                pending_profile_.profile_identity_digest ==
                    value.profile_identity_digest &&
                pending_profile_.mutation_generation > value.mutation_generation) {
                value = std::move(pending_profile_);
                pending_ = false;
            }
        }
    }
}

bool server_cache_calibration_coordinator::start(
        std::string directory) noexcept {
    return writer_.start(std::move(directory));
}

bool server_cache_calibration_coordinator::cache_snapshot(
        const server_cache_calibration_profile_snapshot & value) noexcept {
    const auto cached = std::find_if(
        loaded_profiles_.begin(), loaded_profiles_.end(),
        [&](const auto & profile) {
            return profile.profile_identity_digest ==
                value.profile_identity_digest;
        });
    if (cached != loaded_profiles_.end()) {
        const bool persisted_seed = cached->persisted_seed;
        *cached = value;
        cached->persisted_seed = persisted_seed;
        cached_dirty_ = has_cached_dirty();
        return true;
    }
    if (loaded_profiles_.push_back(value)) {
        cached_dirty_ = has_cached_dirty();
        return true;
    }

    // Slot reuse requires immutable-snapshot acceptance, not merely a disk
    // hope. last_enqueued is advanced only after writer_.enqueue() takes
    // ownership, so an in-flight accepted image is reusable without evidence
    // loss while a rejected dirty image is not.
    auto victim = loaded_profiles_.end();
    for (auto it = loaded_profiles_.begin(); it != loaded_profiles_.end(); ++it) {
        const auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & candidate) {
                return candidate.profile_identity_digest ==
                    it->profile_identity_digest;
            });
        if (currency == profile_currencies_.end() ||
            it->mutation_generation >
                currency->last_enqueued_mutation_generation) continue;
        if (victim == loaded_profiles_.end() ||
            std::tie(it->persisted_prune_recency,
                     it->profile_identity_digest) <
                std::tie(victim->persisted_prune_recency,
                         victim->profile_identity_digest)) {
            victim = it;
        }
    }
    if (victim == loaded_profiles_.end()) return false;
    const auto victim_identity = victim->profile_identity_digest;
    *victim = value;
    const auto victim_currency = std::find_if(
        profile_currencies_.begin(), profile_currencies_.end(),
        [&](const auto & candidate) {
            return candidate.profile_identity_digest == victim_identity;
        });
    if (victim_currency != profile_currencies_.end()) {
        profile_currencies_.erase(victim_currency);
    }
    cached_dirty_ = has_cached_dirty();
    return true;
}

bool server_cache_calibration_coordinator::resolve_load(
        const server_cache_execution_fingerprint & fingerprint,
        server_cache_observation_store & observer) noexcept {
    const auto select_resume_state = [&]() {
        const auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    profile_identity_digest_;
            });
        resume_pending_ = currency != profile_currencies_.end() &&
            currency->resume_validation_pending;
        resume_started_us_ = resume_pending_ ? currency->resume_started_us : 0;
    };
    const auto seed_persisted_currency = [&](const auto & profile) {
        auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    profile.profile_identity_digest;
            });
        if (currency != profile_currencies_.end()) return true;
        server_cache_calibration_profile_currency value;
        value.profile_identity_digest = profile.profile_identity_digest;
        value.last_enqueued_mutation_generation = profile.mutation_generation;
        value.committed_mutation_generation = profile.mutation_generation;
        value.committed_profile_generation_ordinal =
            profile.profile_generation_ordinal;
        value.committed_profile_file_generation =
            profile.profile_file_generation;
        value.committed_ack_seen = true;
        value.resume_validation_pending = true;
        value.resume_started_us = ggml_time_us();
        return profile_currencies_.push_back(value);
    };
    if (load_resolved_) {
        if (profile_identity_digest_ == fingerprint.execution_root) {
            auto effective_fingerprint = fingerprint;
            if (overflow_dirty_) {
                deferred_fingerprint_ = fingerprint;
                effective_fingerprint.complete = false;
            }
            observer.set_execution_fingerprint(effective_fingerprint);
            return true;
        }

        // Preserve the old profile in the bounded in-memory table before the
        // atomic root transition. A future A->B->A switch therefore restores
        // A without scheduler-thread file I/O or cross-profile row aliasing.
        if (observer.mutation_generation() != 0 &&
            server_cache_calibration_snapshot_observer(observer,
                                                        snapshot_buffer_)) {
            const bool enqueued = enqueue_latest(observer, ggml_time_us());
            if (!cache_snapshot(snapshot_buffer_) && !enqueued) {
                // One overflow image preserves the final dirty profile. The
                // next root runs identity-unavailable shadow mode until the
                // worker accepts it; inference and model lifecycle continue.
                overflow_snapshot_ = snapshot_buffer_;
                overflow_dirty_ = true;
                deferred_fingerprint_ = fingerprint;
            }
        }

        auto effective_fingerprint = fingerprint;
        if (overflow_dirty_) {
            deferred_fingerprint_ = fingerprint;
            effective_fingerprint.complete = false;
        }
        observer.set_execution_fingerprint(effective_fingerprint);
        profile_identity_digest_ = fingerprint.execution_root;
        const auto found = std::find_if(
            loaded_profiles_.begin(), loaded_profiles_.end(),
            [&](const auto & profile) {
                return profile.profile_identity_digest ==
                    fingerprint.execution_root;
            });
        if (found != loaded_profiles_.end() &&
            server_cache_calibration_restore_observer(*found, observer)) {
            if (found->persisted_seed) (void) seed_persisted_currency(*found);
        }
        select_resume_state();
        return true;
    }
    server_cache_calibration_load_status status;
    if (!writer_.poll_loaded(status, loaded_profiles_)) return false;
    for (auto & profile : loaded_profiles_) {
        profile.persisted_seed = true;
        if (!seed_persisted_currency(profile)) return false;
    }

    // The execution root and the matching persisted image become visible to
    // the observer in one scheduler-owned transition. No accepted live row can
    // be overwritten by a late store load.
    observer.set_execution_fingerprint(fingerprint);
    profile_identity_digest_ = fingerprint.execution_root;
    if (status == server_cache_calibration_load_status::ok) {
        const auto found = std::find_if(
            loaded_profiles_.begin(), loaded_profiles_.end(),
            [&](const auto & profile) {
                return profile.profile_identity_digest ==
                    fingerprint.execution_root;
            });
        if (found != loaded_profiles_.end() &&
            server_cache_calibration_restore_observer(*found, observer)) {
            // ZC3 restores only a typed shadow seed. ZC4 must consume this
            // state through its forced-validation and 60-second fit-admission
            // barriers; restored moments are never indistinguishable from
            // current-process evidence at the future authority boundary.
            select_resume_state();
        }
    }
    load_resolved_ = true;
    return true;
}

void server_cache_calibration_coordinator::complete_resume_validation() noexcept {
    const auto currency = std::find_if(
        profile_currencies_.begin(), profile_currencies_.end(),
        [&](const auto & value) {
            return value.profile_identity_digest == profile_identity_digest_;
        });
    if (currency != profile_currencies_.end()) {
        currency->resume_validation_pending = false;
        currency->resume_started_us = 0;
    }
    resume_pending_ = false;
    resume_started_us_ = 0;
}

bool server_cache_calibration_coordinator::consume_acks() noexcept {
    bool progressed = false;
    server_cache_calibration_commit_ack ack;
    while (writer_.poll_committed(ack)) {
        progressed = true;
        const auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    ack.profile_identity_digest;
            });
        if (currency == profile_currencies_.end() ||
            ack.mutation_generation < currency->committed_mutation_generation ||
            ack.root_generation < currency->committed_root_generation ||
            (currency->committed_ack_seen &&
             (ack.profile_generation_ordinal !=
                  currency->committed_profile_generation_ordinal ||
              ack.profile_file_generation <=
                  currency->committed_profile_file_generation))) {
            if (currency != profile_currencies_.end()) {
                currency->last_enqueued_mutation_generation =
                    currency->committed_mutation_generation;
            }
            continue;
        }
        currency->committed_mutation_generation = ack.mutation_generation;
        currency->committed_profile_generation_ordinal =
            ack.profile_generation_ordinal;
        currency->committed_profile_file_generation =
            ack.profile_file_generation;
        currency->committed_root_generation = ack.root_generation;
        currency->committed_ack_seen = true;
    }
    return progressed;
}

bool server_cache_calibration_coordinator::has_cached_dirty() const noexcept {
    if (overflow_dirty_) return true;
    for (const auto & profile : loaded_profiles_) {
        const auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    profile.profile_identity_digest;
            });
        const uint64_t enqueued = currency == profile_currencies_.end()
            ? 0 : currency->last_enqueued_mutation_generation;
        if (profile.mutation_generation > enqueued) return true;
    }
    return false;
}

bool server_cache_calibration_coordinator::enqueue_one_cached_dirty() noexcept {
    if (!cached_dirty_ && !overflow_dirty_) return false;
    if (overflow_dirty_) {
        auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    overflow_snapshot_.profile_identity_digest;
            });
        const bool currency_capacity =
            currency == profile_currencies_.end() &&
            profile_currencies_.size() == profile_currencies_.capacity();
        if (currency == profile_currencies_.end() && !currency_capacity) {
            server_cache_calibration_profile_currency value;
            value.profile_identity_digest =
                overflow_snapshot_.profile_identity_digest;
            if (!profile_currencies_.push_back(value)) return false;
            currency = profile_currencies_.end() - 1;
        }
        if (!currency_capacity) {
            if (!writer_.enqueue(overflow_snapshot_)) return false;
            currency->last_enqueued_mutation_generation =
                overflow_snapshot_.mutation_generation;
            overflow_dirty_ = false;
            overflow_snapshot_ = {};
            cached_dirty_ = has_cached_dirty();
            return true;
        }
        // All currency slots are occupied. First transfer ownership of one
        // cached dirty image; that newly reusable slot can then carry the
        // overflow without exceeding the fixed 16-profile table.
    }
    for (const auto & profile : loaded_profiles_) {
        const auto currency = std::find_if(
            profile_currencies_.begin(), profile_currencies_.end(),
            [&](const auto & value) {
                return value.profile_identity_digest ==
                    profile.profile_identity_digest;
            });
        const uint64_t enqueued = currency == profile_currencies_.end()
            ? 0 : currency->last_enqueued_mutation_generation;
        if (profile.mutation_generation <= enqueued) continue;
        if (!writer_.enqueue(profile)) return false;
        if (currency == profile_currencies_.end()) {
            server_cache_calibration_profile_currency value;
            value.profile_identity_digest = profile.profile_identity_digest;
            value.last_enqueued_mutation_generation =
                profile.mutation_generation;
            if (!profile_currencies_.push_back(value)) return false;
        } else {
            currency->last_enqueued_mutation_generation =
                profile.mutation_generation;
        }
        if (overflow_dirty_) {
            (void) cache_snapshot(overflow_snapshot_);
        }
        cached_dirty_ = has_cached_dirty();
        return true;
    }
    cached_dirty_ = false;
    return false;
}

bool server_cache_calibration_coordinator::enqueue_latest(
        server_cache_observation_store & observer,
        int64_t now_us) noexcept {
    const uint64_t mutation = observer.mutation_generation();
    auto currency = std::find_if(
        profile_currencies_.begin(), profile_currencies_.end(),
        [&](const auto & value) {
            return value.profile_identity_digest == profile_identity_digest_;
        });
    if (currency == profile_currencies_.end()) {
        server_cache_calibration_profile_currency value;
        value.profile_identity_digest = profile_identity_digest_;
        if (!profile_currencies_.push_back(value)) return false;
        currency = profile_currencies_.end() - 1;
    }
    if (!load_resolved_ || mutation == 0 ||
        mutation <= currency->last_enqueued_mutation_generation ||
        writer_.health() != server_cache_calibration_writer_health::healthy) {
        return false;
    }
    if (!server_cache_calibration_snapshot_observer(observer, snapshot_buffer_) ||
        !writer_.enqueue(snapshot_buffer_)) return false;
    profile_identity_digest_ = observer.execution_fingerprint().execution_root;
    currency->last_enqueued_mutation_generation = mutation;
    last_enqueue_us_ = now_us;
    return true;
}

void server_cache_calibration_coordinator::lifecycle(
        server_cache_observation_store & observer) noexcept {
    if (!load_resolved_ || writer_.health() !=
            server_cache_calibration_writer_health::healthy) return;
    const bool worker_progress = consume_acks();
    const auto current = std::find_if(
        profile_currencies_.begin(), profile_currencies_.end(),
        [&](const auto & value) {
            return value.profile_identity_digest == profile_identity_digest_;
        });
    const uint64_t mutation = observer.mutation_generation();
    const uint64_t last_enqueued = current == profile_currencies_.end()
        ? 0 : current->last_enqueued_mutation_generation;
    const bool current_dirty = mutation > last_enqueued;
    if (!cached_dirty_ && !overflow_dirty_ && !current_dirty) return;
    const int64_t now_us = ggml_time_us();
    if ((cached_dirty_ || overflow_dirty_) &&
        (worker_progress || cached_retry_us_ == 0 ||
         now_us - cached_retry_us_ >= 30000000)) {
        (void) enqueue_one_cached_dirty();
        cached_retry_us_ = now_us;
    }
    if (!overflow_dirty_ && deferred_fingerprint_) {
        observer.set_execution_fingerprint(*deferred_fingerprint_);
        profile_identity_digest_ = deferred_fingerprint_->execution_root;
        deferred_fingerprint_.reset();
    }
    if (!current_dirty) return;
    const bool sample_due = mutation >= last_enqueued &&
        mutation - last_enqueued >= 64;
    const bool time_due = last_enqueue_us_ == 0 ||
        now_us - last_enqueue_us_ >= 30000000;
    if (sample_due || time_due) {
        enqueue_latest(observer, now_us);
    }
}

void server_cache_calibration_coordinator::flush_latest(
        server_cache_observation_store & observer) noexcept {
    if (observer.mutation_generation() != 0 &&
        server_cache_calibration_snapshot_observer(observer, snapshot_buffer_)) {
        const bool enqueued = enqueue_latest(observer, ggml_time_us());
        if (!cache_snapshot(snapshot_buffer_) && !enqueued) {
            overflow_snapshot_ = snapshot_buffer_;
            overflow_dirty_ = true;
        }
    }
}

void server_cache_calibration_coordinator::drain_latest_for_shutdown(
        server_cache_observation_store & observer) noexcept {
    flush_latest(observer);
    // Final teardown is the only blocking persistence door. Drain the bounded
    // dirty-profile set one immutable image at a time.
    for (unsigned spin = 0; spin < 30000 &&
         writer_.health() == server_cache_calibration_writer_health::healthy;
         ++spin) {
        (void) consume_acks();
        (void) enqueue_one_cached_dirty();
        if (!has_cached_dirty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!overflow_dirty_ && deferred_fingerprint_) {
        observer.set_execution_fingerprint(*deferred_fingerprint_);
        profile_identity_digest_ = deferred_fingerprint_->execution_root;
        deferred_fingerprint_.reset();
    }
}

void server_cache_calibration_coordinator::stop() noexcept {
    writer_.stop();
}
