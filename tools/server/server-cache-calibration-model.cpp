#include "server-cache-calibration-model.h"
#include "../../src/llama-sha256.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace {

static_assert(uint8_t(common_cache_plan_destruction_effect::cross_target_displacement) == 1 &&
              uint8_t(common_cache_plan_destruction_effect::destructive_similarity_retarget) == 2 &&
              uint8_t(common_cache_plan_destruction_effect::same_target_cold_replacement) == 3 &&
              uint8_t(common_cache_plan_destruction_effect::different_host_source_consumption) == 4 &&
              uint8_t(common_cache_plan_destruction_effect::checkpoint_member_drop) == 5,
              "ZC4 effect-action codec must move atomically with schema effect codes");

constexpr double RIDGE_LAMBDA = 1.0;
constexpr double CONFIDENCE_ERROR_SYSTEM = 1e-3;
constexpr double DRIFT_FALSE_ALARM_SYSTEM = 1e-3;
constexpr double VALIDATION_TAU = 1e-3;
constexpr double LOG_WEALTH_LIMIT = 1e6;
constexpr double CONDITION_LIMIT = 1e8;
constexpr std::array<double, 3> VALIDATION_LAMBDAS = {
    1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0,
};

bool finite_feature(const std::array<double, 4> & feature, uint8_t dim) {
    if (dim == 0 || dim > feature.size()) return false;
    for (uint8_t i = 0; i < dim; ++i) {
        if (!std::isfinite(feature[i]) || feature[i] < 0.0 || feature[i] > 1.0) {
            return false;
        }
    }
    return true;
}

double log_weight(uint64_t ordinal) {
    const long double n = static_cast<long double>(ordinal);
    const long double value = -std::log(n + 1.0L) - std::log(n + 2.0L);
    return static_cast<double>(value);
}

bool log_budget(double system,
                const server_cache_calibration_claim_identity & claim,
                double & out) {
    if (!claim.available || !(system > 0.0) || !(system < 1.0) ||
        claim.estimator_slot >= 128) {
        return false;
    }
    out = std::log(system) + log_weight(claim.boot_claim_ordinal) +
          log_weight(claim.profile_generation_ordinal) - std::log(128.0) +
          log_weight(claim.fit_generation);
    return std::isfinite(out) && out < 0.0;
}

struct factorization {
    std::array<std::array<double, 4>, 4> l = {};
    std::array<std::array<double, 4>, 4> inverse = {};
    std::array<double, 4> theta = {};
    double log_det = 0.0;
    double condition = 0.0;
};

bool solve_lower(const std::array<std::array<double, 4>, 4> & l,
                 const std::array<double, 4> & rhs,
                 uint8_t dim,
                 std::array<double, 4> & out) {
    out = {};
    for (uint8_t i = 0; i < dim; ++i) {
        double value = rhs[i];
        for (uint8_t j = 0; j < i; ++j) value -= l[i][j] * out[j];
        if (!(l[i][i] > 0.0) || !std::isfinite(value)) return false;
        out[i] = value / l[i][i];
        if (!std::isfinite(out[i])) return false;
    }
    return true;
}

bool solve_upper(const std::array<std::array<double, 4>, 4> & l,
                 const std::array<double, 4> & rhs,
                 uint8_t dim,
                 std::array<double, 4> & out) {
    out = {};
    for (int i = int(dim) - 1; i >= 0; --i) {
        double value = rhs[size_t(i)];
        for (uint8_t j = uint8_t(i + 1); j < dim; ++j) {
            value -= l[j][size_t(i)] * out[j];
        }
        if (!(l[size_t(i)][size_t(i)] > 0.0) || !std::isfinite(value)) return false;
        out[size_t(i)] = value / l[size_t(i)][size_t(i)];
        if (!std::isfinite(out[size_t(i)])) return false;
    }
    return true;
}

bool factor(const server_cache_observation_instance & instance,
            factorization & out) {
    out = {};
    const uint8_t dim = instance.key.feature_dim;
    if (dim == 0 || dim > 4) return false;
    for (uint8_t i = 0; i < dim; ++i) {
        for (uint8_t j = 0; j <= i; ++j) {
            double sum = instance.v[i][j];
            if (!std::isfinite(sum)) return false;
            for (uint8_t k = 0; k < j; ++k) {
                sum -= out.l[i][k] * out.l[j][k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) return false;
                out.l[i][j] = std::sqrt(sum);
                out.log_det += 2.0 * std::log(out.l[i][j]);
            } else {
                out.l[i][j] = sum / out.l[j][j];
            }
            if (!std::isfinite(out.l[i][j])) return false;
        }
    }

    for (uint8_t column = 0; column < dim; ++column) {
        std::array<double, 4> basis = {};
        basis[column] = 1.0;
        std::array<double, 4> lower = {};
        std::array<double, 4> solution = {};
        if (!solve_lower(out.l, basis, dim, lower) ||
            !solve_upper(out.l, lower, dim, solution)) return false;
        for (uint8_t row = 0; row < dim; ++row) {
            out.inverse[row][column] = solution[row];
        }
    }
    std::array<double, 4> lower = {};
    if (!solve_lower(out.l, instance.b, dim, lower) ||
        !solve_upper(out.l, lower, dim, out.theta)) return false;

    double matrix_norm = 0.0;
    double inverse_norm = 0.0;
    for (uint8_t i = 0; i < dim; ++i) {
        double row = 0.0;
        double inverse_row = 0.0;
        for (uint8_t j = 0; j < dim; ++j) {
            row += std::fabs(instance.v[i][j]);
            inverse_row += std::fabs(out.inverse[i][j]);
        }
        matrix_norm = std::max(matrix_norm, row);
        inverse_norm = std::max(inverse_norm, inverse_row);
    }
    out.condition = matrix_norm * inverse_norm;
    return std::isfinite(out.log_det) && std::isfinite(out.condition) &&
           out.condition <= CONDITION_LIMIT;
}

bool covered(const server_cache_observation_instance & instance,
             const std::array<double, 4> & feature) {
    for (uint8_t i = 0; i < instance.key.feature_dim; ++i) {
        if (feature[i] == 0.0) continue;
        if (feature[i] < instance.feature_min[i] ||
            feature[i] > instance.feature_max[i]) return false;
    }
    return true;
}

double dot(const std::array<double, 4> & lhs,
           const std::array<double, 4> & rhs,
           uint8_t dim) {
    double out = 0.0;
    for (uint8_t i = 0; i < dim; ++i) out += lhs[i] * rhs[i];
    return out;
}

double logsumexp6(const std::array<double, 6> & values) {
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum)) return maximum;
    double sum = 0.0;
    for (double value : values) sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

void add_region(std::array<uint64_t, 8> & regions, uint8_t & count,
                uint64_t value) {
    for (uint8_t i = 0; i < count; ++i) {
        if (regions[i] == value) return;
    }
    if (count < regions.size()) {
        regions[count++] = value;
    }
}

bool update_moments(server_cache_observation_instance & instance,
                    const server_cache_observation_record & record) {
    const uint8_t dim = instance.key.feature_dim;
    auto next_v = instance.v;
    auto next_b = instance.b;
    for (uint8_t i = 0; i < dim; ++i) {
        for (uint8_t j = 0; j < dim; ++j) {
            next_v[i][j] += record.feature[i] * record.feature[j];
            if (!std::isfinite(next_v[i][j])) return false;
        }
        next_b[i] += record.feature[i] * double(record.capped_service_us);
        if (!std::isfinite(next_b[i])) return false;
    }
    instance.v = next_v;
    instance.b = next_b;
    return true;
}

} // namespace

bool server_cache_calibration_representation_digest_v1(
        const void * bytes,
        size_t size,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    if ((!bytes && size != 0) || size > 64 * 1024) return false;
    try {
        static constexpr char domain[] = "buun-zc-representation-v1\0";
        llama_sha256 hash;
        hash.update(domain, sizeof(domain) - 1);
        const uint64_t length = uint64_t(size);
        std::array<uint8_t, 8> le = {};
        for (size_t i = 0; i < le.size(); ++i) {
            le[i] = uint8_t(length >> (8 * i));
        }
        hash.update(le.data(), le.size());
        if (size != 0) hash.update(bytes, size);
        out = hash.finish();
        return std::any_of(out.begin(), out.end(),
                           [](uint8_t value) { return value != 0; });
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_participant_digest_v1(
        const server_cache_calibration_participant_v1 * participants,
        size_t count,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    if ((!participants && count != 0) || count == 0 || count > 4) return false;
    const auto nonzero = [](const std::array<uint8_t, 32> & digest) {
        return std::any_of(digest.begin(), digest.end(),
                           [](uint8_t value) { return value != 0; });
    };
    for (size_t i = 0; i < count; ++i) {
        if (participants[i].media_runtime_class != 0 ||
            participants[i].target_draft_spec_composition > 3 ||
            !nonzero(participants[i].adapter_application_digest) ||
            !nonzero(participants[i].representation_digest)) return false;
    }
    try {
        static constexpr char domain[] = "buun-zc-operation-participants-v1\0";
        llama_sha256 hash;
        hash.update(domain, sizeof(domain) - 1);
        const uint32_t count32 = uint32_t(count);
        const uint8_t count_le[4] = {
            uint8_t(count32), uint8_t(count32 >> 8),
            uint8_t(count32 >> 16), uint8_t(count32 >> 24),
        };
        hash.update(count_le, sizeof(count_le));
        for (size_t i = 0; i < count; ++i) {
            hash.update(participants[i].adapter_application_digest.data(),
                        participants[i].adapter_application_digest.size());
            const uint8_t media_le[2] = {
                uint8_t(participants[i].media_runtime_class),
                uint8_t(participants[i].media_runtime_class >> 8),
            };
            hash.update(media_le, sizeof(media_le));
            hash.update(participants[i].representation_digest.data(),
                        participants[i].representation_digest.size());
            const uint8_t composition_le[2] = {
                uint8_t(participants[i].target_draft_spec_composition),
                uint8_t(participants[i].target_draft_spec_composition >> 8),
            };
            hash.update(composition_le, sizeof(composition_le));
        }
        out = hash.finish();
        return std::any_of(out.begin(), out.end(),
                           [](uint8_t value) { return value != 0; });
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_single_participant_digest_v1(
        const std::array<uint8_t, 32> & adapter_application_digest,
        const std::array<uint8_t, 32> & representation_digest,
        uint8_t target_draft_spec_composition,
        std::array<uint8_t, 32> & out) noexcept {
    const server_cache_calibration_participant_v1 participant = {
        adapter_application_digest, 0, representation_digest,
        target_draft_spec_composition,
    };
    return server_cache_calibration_participant_digest_v1(
        &participant, 1, out);
}

bool server_cache_calibration_effect_action_digest_v1(
        const server_cache_calibration_effect_action_v1 * actions,
        size_t count,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    if ((!actions && count != 0) || count > 6) return false;
    for (size_t i = 0; i < count; ++i) {
        if (actions[i].effect == common_cache_plan_destruction_effect::none ||
            actions[i].effect >= common_cache_plan_destruction_effect::_count ||
            actions[i].destruction_class >=
                server_cache_destruction_class::_count ||
            actions[i].release_owner >
                server_cache_destruction_release_owner::
                    legacy_wrapper_or_capability ||
            server_cache_destruction_census[
                size_t(actions[i].destruction_class)].release_owner !=
                    actions[i].release_owner) return false;
        for (size_t j = 0; j < i; ++j) {
            if (actions[i].effect == actions[j].effect &&
                actions[i].destruction_class == actions[j].destruction_class &&
                actions[i].release_owner == actions[j].release_owner) {
                return false;
            }
        }
    }
    try {
        static constexpr char domain[] = "buun-zc-effect-action-shape-v1\0";
        llama_sha256 hash;
        hash.update(domain, sizeof(domain) - 1);
        const uint32_t count32 = uint32_t(count);
        const uint8_t count_le[4] = {
            uint8_t(count32), uint8_t(count32 >> 8),
            uint8_t(count32 >> 16), uint8_t(count32 >> 24),
        };
        hash.update(count_le, sizeof(count_le));
        for (size_t i = 0; i < count; ++i) {
            const uint16_t fields[] = {
                uint16_t(actions[i].effect),
                uint16_t(uint16_t(actions[i].destruction_class) + 1),
                uint16_t(actions[i].release_owner),
            };
            for (uint16_t field : fields) {
                const uint8_t le[2] = { uint8_t(field), uint8_t(field >> 8) };
                hash.update(le, sizeof(le));
            }
        }
        out = hash.finish();
        return std::any_of(out.begin(), out.end(),
                           [](uint8_t value) { return value != 0; });
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_calibration_apply_shape_digest_v1(
        common_cache_plan_destruction_effect_set effects,
        server_cache_destruction_class destruction_class,
        server_cache_destruction_release_owner release_owner,
        std::array<uint8_t, 32> & out) noexcept {
    if (destruction_class >= server_cache_destruction_class::_count ||
        server_cache_destruction_census[size_t(destruction_class)].release_owner !=
            release_owner) {
        out = {};
        return false;
    }
    std::array<server_cache_calibration_effect_action_v1,
               size_t(common_cache_plan_destruction_effect::_count) - 1>
        actions = {};
    size_t count = 0;
    common_cache_plan_destruction_effect_set known = 0;
    for (uint16_t code = 1;
         code < uint16_t(common_cache_plan_destruction_effect::_count);
         ++code) {
        const auto effect = common_cache_plan_destruction_effect(code);
        const auto bit = common_cache_plan_destruction_effect_bit(effect);
        known |= bit;
        if ((effects & bit) != 0) {
            actions[count++] = { effect, destruction_class, release_owner };
        }
    }
    if (count == 0 || (effects & ~known) != 0) {
        out = {};
        return false;
    }
    return server_cache_calibration_effect_action_digest_v1(
        actions.data(), count, out);
}

bool server_cache_calibration_predict(
        const server_cache_observation_instance & instance,
        const server_cache_calibration_claim_identity & claim,
        const std::array<double, 4> & feature,
        server_cache_calibration_prediction & out) noexcept {
    out = {};
    const uint8_t dim = instance.key.feature_dim;
    const uint64_t floor = dim == 1 ? 4 : uint64_t(dim) + 1;
    if (!instance.used || instance.authority_terminal !=
            server_cache_calibration_authority_terminal::none ||
        instance.estimator_slot == UINT32_MAX ||
        claim.estimator_slot != instance.estimator_slot ||
        claim.fit_generation != instance.fit_generation ||
        instance.n_success < floor) {
        out.status = server_cache_calibration_prediction_status::learning;
        return false;
    }
    if (!finite_feature(feature, dim) || !covered(instance, feature)) {
        out.status = server_cache_calibration_prediction_status::out_of_coverage;
        return false;
    }
    factorization solved;
    if (!factor(instance, solved)) {
        out.status = server_cache_calibration_prediction_status::numeric_fault;
        return false;
    }
    double log_delta = 0.0;
    if (!log_budget(CONFIDENCE_ERROR_SYSTEM, claim, log_delta)) {
        out.status = server_cache_calibration_prediction_status::confidence_budget_exhausted;
        return false;
    }
    std::array<double, 4> inverse_feature = {};
    for (uint8_t i = 0; i < dim; ++i) {
        for (uint8_t j = 0; j < dim; ++j) {
            inverse_feature[i] += solved.inverse[i][j] * feature[j];
        }
    }
    const double leverage = dot(feature, inverse_feature, dim);
    const double ymax = double(server_cache_observation_response_cap_us(
        instance.key.operation, instance.key.size_family));
    const double point = dot(feature, solved.theta, dim);
    const double r = ymax / 2.0;
    const double s = ymax * std::sqrt(double(dim));
    const double beta_argument = solved.log_det - 2.0 * log_delta;
    if (!(leverage >= 0.0) || !std::isfinite(leverage) ||
        !(ymax > 0.0) || !std::isfinite(point) || point < 0.0 ||
        point > ymax || beta_argument < 0.0 || !std::isfinite(beta_argument)) {
        out.status = server_cache_calibration_prediction_status::numeric_fault;
        return false;
    }
    const double beta = r * std::sqrt(beta_argument) + s;
    const double radius = beta * std::sqrt(leverage);
    if (!std::isfinite(radius)) {
        out.status = server_cache_calibration_prediction_status::numeric_fault;
        return false;
    }
    out.status = server_cache_calibration_prediction_status::ok;
    out.point_us = point;
    out.radius_us = radius;
    out.lower_us = std::max(0.0, point - radius);
    out.upper_us = std::min(ymax, point + radius);
    out.condition_number = solved.condition;
    out.log_determinant = solved.log_det;
    return true;
}

const char * server_cache_calibration_instance_state_name(
        server_cache_calibration_instance_state value) noexcept {
    switch (value) {
        case server_cache_calibration_instance_state::unseen: return "unseen";
        case server_cache_calibration_instance_state::learning: return "learning";
        case server_cache_calibration_instance_state::provisional: return "provisional";
        case server_cache_calibration_instance_state::active: return "active";
        case server_cache_calibration_instance_state::drifted: return "drifted";
        case server_cache_calibration_instance_state::quarantined: return "quarantined";
    }
    return "invalid";
}

server_cache_calibration_instance_state server_cache_calibration_state(
        const server_cache_observation_instance & instance,
        const server_cache_calibration_claim_identity & claim,
        const std::array<double, 4> & feature,
        uint64_t now_unix_ms,
        server_cache_calibration_prediction * prediction,
        bool authority_admission_allowed) noexcept {
    if (!instance.used) return server_cache_calibration_instance_state::unseen;
    if (instance.authority_terminal ==
            server_cache_calibration_authority_terminal::drifted) {
        return server_cache_calibration_instance_state::drifted;
    }
    if (instance.authority_terminal !=
            server_cache_calibration_authority_terminal::none) {
        return server_cache_calibration_instance_state::quarantined;
    }
    server_cache_calibration_prediction local;
    if (!server_cache_calibration_predict(instance, claim, feature, local)) {
        if (prediction) *prediction = local;
        return server_cache_calibration_instance_state::learning;
    }
    if (prediction) *prediction = local;
    const bool replay = instance.key.operation ==
        server_cache_observation_operation::replay;
    const uint64_t fit_floor = replay ? 20 : 8;
    const uint8_t fit_regions = replay ? 5 : 4;
    double log_alpha = 0.0;
    if (!log_budget(DRIFT_FALSE_ALARM_SYSTEM, claim, log_alpha)) {
        return server_cache_calibration_instance_state::provisional;
    }
    const double log_e = logsumexp6(instance.log_wealth) - std::log(6.0);
    const bool clock_fresh = instance.last_validation_unix_ms != 0 &&
        now_unix_ms >= instance.last_validation_unix_ms &&
        now_unix_ms - instance.last_validation_unix_ms <= 10 * 60 * 1000;
    const bool opportunity_fresh =
        instance.safe_measurable_opportunities >=
            instance.opportunity_at_last_validation &&
        instance.safe_measurable_opportunities -
            instance.opportunity_at_last_validation < 256;
    const bool active = authority_admission_allowed &&
        instance.key.identity_exact &&
        instance.n_success >= fit_floor &&
        instance.fit_region_count >= fit_regions &&
        instance.n_validation >= 4 &&
        instance.validation_region_count >= 3 &&
        clock_fresh && opportunity_fresh && std::isfinite(log_e) &&
        log_e < -log_alpha;
    return active ? server_cache_calibration_instance_state::active
                  : server_cache_calibration_instance_state::provisional;
}

bool server_cache_calibration_bound_direct_difference(
        const server_cache_calibration_contribution * contributions,
        size_t count,
        server_cache_calibration_direct_bound & out) noexcept {
    out = {};
    if ((!contributions && count != 0) || count == 0 || count > 32) {
        out.status = server_cache_calibration_prediction_status::numeric_fault;
        return false;
    }
    struct group {
        const server_cache_observation_instance * instance = nullptr;
        server_cache_calibration_claim_identity claim;
        std::array<double, 4> delta = {};
    };
    std::array<group, 32> groups = {};
    size_t group_count = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & term = contributions[i];
        if (!term.instance || term.weight_milli == 0 ||
            term.weight_milli > 1000000 ||
            term.side > server_cache_calibration_contribution_side::challenger ||
            !finite_feature(term.feature, term.instance->key.feature_dim)) {
            out.status = server_cache_calibration_prediction_status::out_of_coverage;
            return false;
        }
        if (term.claim.fit_generation != term.instance->fit_generation ||
            term.claim.estimator_slot != term.instance->estimator_slot ||
            server_cache_calibration_state(
                *term.instance, term.claim, term.feature, term.now_unix_ms,
                nullptr, term.authority_admission_allowed) !=
                    server_cache_calibration_instance_state::active) {
            out.status = server_cache_calibration_prediction_status::learning;
            return false;
        }
        server_cache_calibration_prediction available;
        if (!server_cache_calibration_predict(
                *term.instance, term.claim, term.feature, available)) {
            out.status = available.status;
            return false;
        }
        size_t group_index = group_count;
        for (size_t j = 0; j < group_count; ++j) {
            if (groups[j].instance == term.instance) {
                group_index = j;
                break;
            }
        }
        if (group_index == group_count) {
            groups[group_count].instance = term.instance;
            groups[group_count].claim = term.claim;
            ++group_count;
        } else {
            const auto & claim = groups[group_index].claim;
            if (claim.available != term.claim.available ||
                claim.boot_claim_ordinal != term.claim.boot_claim_ordinal ||
                claim.profile_generation_ordinal !=
                    term.claim.profile_generation_ordinal ||
                claim.estimator_slot != term.claim.estimator_slot ||
                claim.fit_generation != term.claim.fit_generation) {
                out.status =
                    server_cache_calibration_prediction_status::numeric_fault;
                return false;
            }
        }
        const double sign = term.side ==
                server_cache_calibration_contribution_side::baseline
            ? 1.0 : -1.0;
        const double scalar = sign * double(term.weight_milli) / 1000.0;
        for (uint8_t d = 0; d < term.instance->key.feature_dim; ++d) {
            groups[group_index].delta[d] += scalar * term.feature[d];
            if (!std::isfinite(groups[group_index].delta[d])) {
                out.status =
                    server_cache_calibration_prediction_status::numeric_fault;
                return false;
            }
        }
    }

    double benefit = 0.0;
    double radius = 0.0;
    for (size_t i = 0; i < group_count; ++i) {
        const auto & item = groups[i];
        const uint8_t dim = item.instance->key.feature_dim;
        factorization solved;
        double log_delta = 0.0;
        if (!factor(*item.instance, solved)) {
            out.status =
                server_cache_calibration_prediction_status::numeric_fault;
            return false;
        }
        if (!log_budget(CONFIDENCE_ERROR_SYSTEM, item.claim, log_delta)) {
            out.status = server_cache_calibration_prediction_status::
                confidence_budget_exhausted;
            return false;
        }
        std::array<double, 4> inverse_delta = {};
        for (uint8_t row = 0; row < dim; ++row) {
            for (uint8_t column = 0; column < dim; ++column) {
                inverse_delta[row] +=
                    solved.inverse[row][column] * item.delta[column];
            }
        }
        double leverage = dot(item.delta, inverse_delta, dim);
        if (leverage < 0.0 && leverage > -1e-12) leverage = 0.0;
        const double ymax = double(server_cache_observation_response_cap_us(
            item.instance->key.operation, item.instance->key.size_family));
        const double beta_argument = solved.log_det - 2.0 * log_delta;
        if (beta_argument < 0.0 || !std::isfinite(beta_argument) ||
            !(ymax > 0.0)) {
            out.status =
                server_cache_calibration_prediction_status::numeric_fault;
            return false;
        }
        const double beta = ymax / 2.0 * std::sqrt(beta_argument) +
                            ymax * std::sqrt(double(dim));
        const double item_benefit = dot(item.delta, solved.theta, dim);
        const double item_radius = beta * std::sqrt(leverage);
        if (!(leverage >= 0.0) ||
            !std::isfinite(item_benefit) || !std::isfinite(item_radius) ||
            radius > std::numeric_limits<double>::max() - item_radius) {
            out.status =
                server_cache_calibration_prediction_status::numeric_fault;
            return false;
        }
        benefit += item_benefit;
        radius += item_radius;
        if (!std::isfinite(benefit) || !std::isfinite(radius)) {
            out.status =
                server_cache_calibration_prediction_status::numeric_fault;
            return false;
        }
    }
    out.status = server_cache_calibration_prediction_status::ok;
    out.benefit_us = benefit;
    out.radius_us = radius;
    out.benefit_lower_us = benefit - radius;
    return std::isfinite(out.benefit_lower_us);
}

bool server_cache_calibration_validation_assignment(
        uint64_t qualified_execution_ordinal) noexcept {
    const uint64_t block = qualified_execution_ordinal / 8;
    uint64_t mixed = block + UINT64_C(0x9e3779b97f4a7c15);
    mixed = (mixed ^ (mixed >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    mixed = (mixed ^ (mixed >> 27)) * UINT64_C(0x94d049bb133111eb);
    mixed ^= mixed >> 31;
    return qualified_execution_ordinal % 8 == mixed % 8;
}

bool server_cache_calibration_preassign(
        server_cache_observation_instance & instance,
        const std::array<double, 4> & feature,
        const server_cache_calibration_update_context & context,
        server_cache_calibration_preassignment & out) noexcept {
    out = {};
    if (!instance.used || instance.authority_terminal !=
            server_cache_calibration_authority_terminal::none ||
        !finite_feature(feature, instance.key.feature_dim) ||
        instance.qualified_execution_ordinal == UINT64_MAX ||
        context.claim.estimator_slot != instance.estimator_slot ||
        context.claim.fit_generation != instance.fit_generation) return false;

    out.qualified_execution_ordinal = instance.qualified_execution_ordinal;
    out.fit_generation = instance.fit_generation;
    const bool validation = context.force_validation ||
        server_cache_calibration_validation_assignment(
            out.qualified_execution_ordinal);
    if (validation) {
        out.validation_prediction_available = server_cache_calibration_predict(
            instance, context.claim, feature, out.validation_prediction);
        out.assignment = out.validation_prediction_available
            ? server_cache_calibration_assignment::validation
            : server_cache_calibration_assignment::validation_unavailable;
    } else if (!context.fit_admission_allowed ||
               !context.principal_admission_allowed ||
               instance.last_fit_steady_second == context.steady_second) {
        out.assignment = server_cache_calibration_assignment::fit_rate_limited;
    } else {
        out.assignment = server_cache_calibration_assignment::fit;
    }

    ++instance.qualified_execution_ordinal;
    if (out.assignment == server_cache_calibration_assignment::fit) {
        // The one-per-block admission is consumed before the response exists;
        // a failed operation cannot donate its fit slot to the next outcome.
        instance.last_fit_steady_second = context.steady_second;
    }
    out.valid = true;
    return true;
}

bool server_cache_calibration_abandon(
        const server_cache_observation_instance & instance,
        const server_cache_calibration_preassignment & assignment) noexcept {
    return assignment.valid && instance.used &&
        instance.authority_terminal ==
            server_cache_calibration_authority_terminal::none &&
        instance.fit_generation == assignment.fit_generation &&
        instance.qualified_execution_ordinal ==
            assignment.qualified_execution_ordinal + 1;
}

bool server_cache_calibration_complete(
        server_cache_observation_instance & instance,
        const server_cache_observation_record & record,
        const server_cache_calibration_update_context & context,
        const server_cache_calibration_preassignment & assignment,
        server_cache_calibration_update_result & out) noexcept {
    out = {};
    out.assignment = assignment.assignment;
    out.validation_prediction_available =
        assignment.validation_prediction_available;
    out.validation_prediction = assignment.validation_prediction;
    if (!assignment.valid || !instance.used || !(instance.key == record.key) ||
        record.terminal != server_cache_observation_terminal::accepted ||
        !finite_feature(record.feature, instance.key.feature_dim) ||
        instance.fit_generation != assignment.fit_generation ||
        instance.qualified_execution_ordinal !=
            assignment.qualified_execution_ordinal + 1) {
        instance.authority_terminal =
            server_cache_calibration_authority_terminal::numeric_fault;
        return false;
    }

    auto next = instance;
    if (assignment.assignment == server_cache_calibration_assignment::validation) {
        const double ymax = double(server_cache_observation_response_cap_us(
            next.key.operation, next.key.size_family));
        const double z = (double(record.capped_service_us) -
                          assignment.validation_prediction.point_us) / ymax;
        if (!(ymax > 0.0) || !std::isfinite(z) || z < -1.0 || z > 1.0 ||
            next.n_validation == UINT64_MAX) {
            instance.authority_terminal =
                server_cache_calibration_authority_terminal::numeric_fault;
            return false;
        }
        auto next_wealth = next.log_wealth;
        for (size_t lambda_i = 0; lambda_i < VALIDATION_LAMBDAS.size(); ++lambda_i) {
            for (size_t sign_i = 0; sign_i < 2; ++sign_i) {
                const size_t arm = lambda_i * 2 + sign_i;
                const double lambda = VALIDATION_LAMBDAS[lambda_i];
                const double sign = sign_i == 0 ? -1.0 : 1.0;
                const double increment = lambda * (sign * z - VALIDATION_TAU) -
                                         lambda * lambda / 2.0;
                if (!std::isfinite(next_wealth[arm] + increment)) {
                    instance.authority_terminal =
                        server_cache_calibration_authority_terminal::numeric_fault;
                    return false;
                }
                next_wealth[arm] = std::clamp(next_wealth[arm] + increment,
                    -LOG_WEALTH_LIMIT, LOG_WEALTH_LIMIT);
            }
        }
        next.log_wealth = next_wealth;
        ++next.n_validation;
        add_region(next.validation_region_minutes,
                   next.validation_region_count, context.unix_minute);
        next.opportunity_at_last_validation =
            next.safe_measurable_opportunities;
        next.last_validation_unix_ms = context.unix_ms;
        out.validation_changed = true;

        double log_alpha = 0.0;
        if (!log_budget(DRIFT_FALSE_ALARM_SYSTEM, context.claim, log_alpha)) {
            next.authority_terminal =
                server_cache_calibration_authority_terminal::confidence_budget_exhausted;
            out.drifted = true;
        } else {
            const double log_e = logsumexp6(next.log_wealth) - std::log(6.0);
            if (!std::isfinite(log_e) || log_e >= -log_alpha) {
                next.authority_terminal =
                    server_cache_calibration_authority_terminal::drifted;
                out.drifted = true;
            }
        }
    } else if (assignment.assignment == server_cache_calibration_assignment::fit) {
        if (next.n_success == UINT64_MAX || !update_moments(next, record)) {
            instance.authority_terminal =
                server_cache_calibration_authority_terminal::numeric_fault;
            return false;
        }
        ++next.n_success;
        if (next.n_success == 1) {
            next.feature_min = record.feature;
            next.feature_max = record.feature;
        } else {
            for (uint8_t i = 0; i < next.key.feature_dim; ++i) {
                next.feature_min[i] =
                    std::min(next.feature_min[i], record.feature[i]);
                next.feature_max[i] =
                    std::max(next.feature_max[i], record.feature[i]);
            }
        }
        add_region(next.fit_region_minutes, next.fit_region_count,
                   context.unix_minute);
        next.last_fit_unix_ms = context.unix_ms;
        out.moments_changed = true;
    }

    next.response_reservoir[
        next.reservoir_seen % next.residual_capacity] =
        record.capped_service_us;
    if (next.reservoir_seen != UINT64_MAX) ++next.reservoir_seen;
    if (record.tail_exceeded) {
        next.tail_exceeded = true;
        next.tail_actual_max_us = std::max(
            next.tail_actual_max_us, record.owned_service_us);
        next.authority_terminal =
            server_cache_calibration_authority_terminal::tail_exceeded;
        out.tail_latched = true;
    }
    instance = next;
    return true;
}

bool server_cache_calibration_update(
        server_cache_observation_instance & instance,
        const server_cache_observation_record & record,
        const server_cache_calibration_update_context & context,
        server_cache_calibration_update_result & out) noexcept {
    server_cache_calibration_preassignment assignment;
    if (!server_cache_calibration_preassign(
            instance, record.feature, context, assignment)) return false;
    if (record.terminal ==
            server_cache_observation_terminal::operation_unavailable) {
        out = {};
        out.assignment = assignment.assignment;
        out.validation_prediction_available =
            assignment.validation_prediction_available;
        out.validation_prediction = assignment.validation_prediction;
        return server_cache_calibration_abandon(instance, assignment);
    }
    return server_cache_calibration_complete(
        instance, record, context, assignment, out);
}

server_cache_calibration_arena::~server_cache_calibration_arena() {
    reset();
}

bool server_cache_calibration_arena::allocate() noexcept {
    if (storage_) return true;
    if (!layout_valid()) return false;
    storage_ = static_cast<std::byte *>(::operator new(
        server_cache_calibration_arena_layout::total_size,
        std::align_val_t(server_cache_calibration_arena_layout::alignment),
        std::nothrow));
    return storage_ != nullptr;
}

void server_cache_calibration_arena::reset() noexcept {
    if (!storage_) return;
    ::operator delete(storage_,
        std::align_val_t(server_cache_calibration_arena_layout::alignment));
    storage_ = nullptr;
}

void * server_cache_calibration_arena::region(
        size_t offset, size_t size, size_t alignment) noexcept {
    if (!storage_ || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        offset % alignment != 0 || offset > server_cache_calibration_arena_layout::total_size ||
        size > server_cache_calibration_arena_layout::total_size - offset) return nullptr;
    return storage_ + offset;
}

const void * server_cache_calibration_arena::region(
        size_t offset, size_t size, size_t alignment) const noexcept {
    return const_cast<server_cache_calibration_arena *>(this)->region(
        offset, size, alignment);
}

bool server_cache_calibration_arena::layout_valid() noexcept {
    using layout = server_cache_calibration_arena_layout;
    return layout::profile_slots_begin == 0 &&
        layout::snapshots_begin == layout::profile_slots_begin + layout::profile_slots_size &&
        layout::global_tables_begin == layout::snapshots_begin + layout::snapshots_size &&
        layout::fingerprint_begin == layout::global_tables_begin + layout::global_tables_size &&
        layout::codec_scratch_begin == layout::fingerprint_begin + layout::fingerprint_size &&
        layout::reserve_begin == layout::codec_scratch_begin + layout::codec_scratch_size &&
        layout::total_size == layout::reserve_begin + layout::reserve_size;
}
