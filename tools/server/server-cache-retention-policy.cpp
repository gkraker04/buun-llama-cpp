#include "server-cache-retention-policy.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace {

enum class lane : uint8_t {
    recent = 0,
    historical,
};

struct classified_member {
    server_cache_retention_member member;
    lane                          retention_lane            = lane::recent;
    uint64_t                      age                       = 0;
    uint32_t                      bucket                    = 0;
    bool                          protected_member          = false;
    bool                          historical_representative = false;
    uint32_t                      historical_rank           = UINT32_MAX;
};

uint64_t saturated_multiply(uint64_t value, uint32_t factor) noexcept {
    if (factor == 0 || value > std::numeric_limits<uint64_t>::max() / factor) {
        return std::numeric_limits<uint64_t>::max();
    }
    return value * factor;
}

uint32_t historical_bucket(uint64_t age, uint64_t base, uint32_t growth) noexcept {
    uint32_t bucket = 1;
    uint64_t upper  = saturated_multiply(base, growth);
    while (age >= upper && upper != std::numeric_limits<uint64_t>::max()) {
        if (bucket == UINT32_MAX) {
            break;
        }
        bucket++;
        upper = saturated_multiply(upper, growth);
    }
    return bucket;
}

uint8_t protected_class(const classified_member & member) noexcept {
    if (member.member.identity == server_cache_retention_identity::stale_conclusive_refused) {
        return 2;
    }
    if (member.member.identity == server_cache_retention_identity::identity_unknown) {
        return 3;
    }
    return member.retention_lane == lane::recent ? 0 : 1;
}

bool erase_first_lane(std::vector<lane> & labels, lane wanted) {
    const auto it = std::find(labels.begin(), labels.end(), wanted);
    if (it == labels.end()) {
        return false;
    }
    labels.erase(it);
    return true;
}

bool contains_ordinal(const std::vector<uint32_t> & values, uint32_t ordinal) noexcept {
    return std::find(values.begin(), values.end(), ordinal) != values.end();
}

}  // namespace

server_cache_retention_policy_result server_cache_plan_retention_set(
    const server_cache_retention_policy_config &       config,
    uint64_t                                           current_frontier,
    const std::vector<server_cache_retention_member> & members) noexcept {
    server_cache_retention_policy_result out;
    if (config.recent_floor == 0 || config.minimum_historical == 0 || config.bucket_base == 0 ||
        config.bucket_growth < 2) {
        return out;
    }

    if (config.capacity == 0) {
        // No lane labels. Evidence and protection are still classified below
        // so protected incumbents fail closed instead of becoming victims.
    } else if (config.capacity == 1) {
        out.recent_quota = 1;
    } else {
        const uint32_t historical_min = std::min(config.minimum_historical, config.capacity - 1);
        out.recent_quota              = std::min(config.recent_floor, config.capacity - historical_min);
        out.recent_quota              = std::max(out.recent_quota, uint32_t(1));
        out.historical_quota          = config.capacity - out.recent_quota;
    }

    try {
        std::set<uint32_t>             ordinals;
        std::set<uint64_t>             stable_ids;
        size_t                         incoming_count = 0;
        std::vector<classified_member> classified;
        classified.reserve(members.size());
        for (const auto & member : members) {
            if (member.stable_id == 0 || !ordinals.insert(member.ordinal).second ||
                !stable_ids.insert(member.stable_id).second ||
                (member.identity == server_cache_retention_identity::current && member.frontier > current_frontier)) {
                out.status = server_cache_retention_policy_status::incomplete_evidence;
                return out;
            }
            incoming_count += member.incoming ? 1 : 0;
            classified_member next;
            next.member           = member;
            next.protected_member = member.protection != server_cache_retention_protection::none;
            if (member.identity == server_cache_retention_identity::current) {
                next.age = current_frontier - member.frontier;
                if (next.age >= config.bucket_base) {
                    next.retention_lane = lane::historical;
                    next.bucket         = historical_bucket(next.age, config.bucket_base, config.bucket_growth);
                }
            }
            classified.push_back(std::move(next));
        }
        if (incoming_count > 1) {
            out.status = server_cache_retention_policy_status::incomplete_evidence;
            return out;
        }

        std::vector<lane> labels;
        labels.reserve(config.capacity);
        uint32_t recent_left     = out.recent_quota;
        uint32_t historical_left = out.historical_quota;
        while (recent_left > 0 && historical_left > 0) {
            labels.push_back(lane::recent);
            recent_left--;
            labels.push_back(lane::historical);
            historical_left--;
        }
        labels.insert(labels.end(), recent_left, lane::recent);
        labels.insert(labels.end(), historical_left, lane::historical);

        std::vector<classified_member *> protected_members;
        for (auto & member : classified) {
            if (member.protected_member) {
                protected_members.push_back(&member);
            }
        }
        std::sort(protected_members.begin(), protected_members.end(),
                  [](const classified_member * a, const classified_member * b) {
                      return std::make_tuple(protected_class(*a),
                                             a->member.identity == server_cache_retention_identity::current ?
                                                 a->member.frontier :
                                                 UINT64_MAX,
                                             a->member.stable_id) <
                             std::make_tuple(protected_class(*b),
                                             b->member.identity == server_cache_retention_identity::current ?
                                                 b->member.frontier :
                                                 UINT64_MAX,
                                             b->member.stable_id);
                  });
        if (protected_members.size() > config.capacity) {
            out.status = server_cache_retention_policy_status::protected_over_capacity;
            return out;
        }
        for (const auto * member : protected_members) {
            out.desired.push_back(member->member.ordinal);
            const bool current = member->member.identity == server_cache_retention_identity::current;
            if (!current || !erase_first_lane(labels, member->retention_lane)) {
                if (!labels.empty()) {
                    labels.pop_back();
                }
            }
        }

        std::vector<classified_member *> recent;
        std::vector<classified_member *> historical;
        std::vector<classified_member *> stale;
        for (auto & member : classified) {
            if (member.protected_member) {
                continue;
            }
            if (member.member.identity != server_cache_retention_identity::current) {
                stale.push_back(&member);
            } else if (member.retention_lane == lane::recent) {
                recent.push_back(&member);
            } else {
                historical.push_back(&member);
            }
        }
        std::sort(recent.begin(), recent.end(), [](const classified_member * a, const classified_member * b) {
            return std::make_tuple(std::numeric_limits<uint64_t>::max() - a->member.frontier, a->member.stable_id) <
                   std::make_tuple(std::numeric_limits<uint64_t>::max() - b->member.frontier, b->member.stable_id);
        });

        std::vector<uint32_t> buckets;
        for (const auto * member : historical) {
            if (std::find(buckets.begin(), buckets.end(), member->bucket) == buckets.end()) {
                buckets.push_back(member->bucket);
            }
        }
        std::sort(buckets.begin(), buckets.end());
        std::vector<uint32_t> bucket_order;
        if (!buckets.empty()) {
            bucket_order.push_back(buckets.front());
        }
        if (buckets.size() > 1) {
            bucket_order.push_back(buckets.back());
        }
        while (bucket_order.size() < buckets.size()) {
            uint32_t selected          = 0;
            uint32_t selected_distance = 0;
            bool     have_selected     = false;
            for (const uint32_t candidate : buckets) {
                if (std::find(bucket_order.begin(), bucket_order.end(), candidate) != bucket_order.end()) {
                    continue;
                }
                uint32_t distance = UINT32_MAX;
                for (const uint32_t chosen : bucket_order) {
                    const uint32_t delta = candidate > chosen ? candidate - chosen : chosen - candidate;
                    distance             = std::min(distance, delta);
                }
                if (!have_selected || distance > selected_distance ||
                    (distance == selected_distance && candidate > selected)) {
                    selected          = candidate;
                    selected_distance = distance;
                    have_selected     = true;
                }
            }
            bucket_order.push_back(selected);
        }

        std::vector<classified_member *> historical_order;
        for (size_t rank = 0; rank < bucket_order.size(); ++rank) {
            std::vector<classified_member *> in_bucket;
            for (auto * member : historical) {
                if (member->bucket == bucket_order[rank]) {
                    in_bucket.push_back(member);
                }
            }
            std::sort(in_bucket.begin(), in_bucket.end(), [](const classified_member * a, const classified_member * b) {
                return std::make_tuple(a->member.frontier, a->member.stable_id) <
                       std::make_tuple(b->member.frontier, b->member.stable_id);
            });
            if (!in_bucket.empty()) {
                in_bucket.front()->historical_representative = true;
                for (auto * member : in_bucket) {
                    member->historical_rank = uint32_t(rank);
                }
                historical_order.push_back(in_bucket.front());
            }
        }
        for (const uint32_t bucket : bucket_order) {
            std::vector<classified_member *> extras;
            for (auto * member : historical) {
                if (member->bucket == bucket && !member->historical_representative) {
                    extras.push_back(member);
                }
            }
            std::sort(extras.begin(), extras.end(), [](const classified_member * a, const classified_member * b) {
                return std::make_tuple(std::numeric_limits<uint64_t>::max() - a->member.frontier, a->member.stable_id) <
                       std::make_tuple(std::numeric_limits<uint64_t>::max() - b->member.frontier, b->member.stable_id);
            });
            historical_order.insert(historical_order.end(), extras.begin(), extras.end());
        }

        size_t            recent_pos     = 0;
        size_t            historical_pos = 0;
        std::vector<lane> unfilled;
        for (const lane label : labels) {
            classified_member * selected = nullptr;
            if (label == lane::recent && recent_pos < recent.size()) {
                selected = recent[recent_pos++];
            } else if (label == lane::historical && historical_pos < historical_order.size()) {
                selected = historical_order[historical_pos++];
            }
            if (selected) {
                out.desired.push_back(selected->member.ordinal);
            } else {
                unfilled.push_back(label);
            }
        }

        std::vector<classified_member *> leftovers;
        leftovers.insert(leftovers.end(), recent.begin() + recent_pos, recent.end());
        leftovers.insert(leftovers.end(), historical_order.begin() + historical_pos, historical_order.end());
        std::sort(stale.begin(), stale.end(), [](const classified_member * a, const classified_member * b) {
            return std::make_tuple(a->member.frontier, a->member.stable_id) <
                   std::make_tuple(b->member.frontier, b->member.stable_id);
        });
        leftovers.insert(leftovers.end(), stale.begin(), stale.end());
        for (size_t i = 0; i < unfilled.size() && i < leftovers.size(); ++i) {
            out.desired.push_back(leftovers[i]->member.ordinal);
        }

        for (const auto & member : classified) {
            if (member.member.incoming && contains_ordinal(out.desired, member.member.ordinal)) {
                out.incoming_selected = true;
            }
        }

        std::vector<classified_member *> excluded;
        for (auto & member : classified) {
            if (!member.member.incoming && !member.protected_member &&
                !contains_ordinal(out.desired, member.member.ordinal)) {
                excluded.push_back(&member);
            }
        }
        std::sort(excluded.begin(), excluded.end(), [](const classified_member * a, const classified_member * b) {
            const auto category = [](const classified_member & member) {
                if (member.member.identity == server_cache_retention_identity::stale_conclusive_refused) {
                    return 0;
                }
                if (member.member.identity == server_cache_retention_identity::identity_unknown) {
                    return 1;
                }
                if (member.retention_lane == lane::historical && !member.historical_representative) {
                    return 2;
                }
                if (member.retention_lane == lane::recent) {
                    return 3;
                }
                return 4;
            };
            const int ca = category(*a);
            const int cb = category(*b);
            if (ca != cb) {
                return ca < cb;
            }
            if (ca <= 1) {
                return std::make_tuple(a->member.frontier, a->member.stable_id) <
                       std::make_tuple(b->member.frontier, b->member.stable_id);
            }
            if (ca == 3) {
                return std::make_tuple(a->member.frontier, std::numeric_limits<uint64_t>::max() - a->member.stable_id) <
                       std::make_tuple(b->member.frontier, std::numeric_limits<uint64_t>::max() - b->member.stable_id);
            }
            return std::make_tuple(std::numeric_limits<uint32_t>::max() - a->historical_rank,
                                   std::numeric_limits<uint64_t>::max() - a->member.frontier,
                                   std::numeric_limits<uint64_t>::max() - a->member.stable_id) <
                   std::make_tuple(std::numeric_limits<uint32_t>::max() - b->historical_rank,
                                   std::numeric_limits<uint64_t>::max() - b->member.frontier,
                                   std::numeric_limits<uint64_t>::max() - b->member.stable_id);
        });
        for (const auto * member : excluded) {
            out.exclusion_order.push_back(member->member.ordinal);
        }
        out.status = server_cache_retention_policy_status::ok;
        return out;
    } catch (...) {
        out.status = server_cache_retention_policy_status::capacity_unavailable;
        out.desired.clear();
        out.exclusion_order.clear();
        out.incoming_selected = false;
        return out;
    }
}

server_cache_retention_sim_score server_cache_simulate_retention(
    const server_cache_retention_sim_config &             config,
    const std::vector<server_cache_retention_sim_event> & events) noexcept {
    server_cache_retention_sim_score score;
    if (config.policy.capacity == 0 || config.recent_replay_cap == 0 || config.historical_replay_cap == 0) {
        return score;
    }

    struct node_state {
        uint32_t parent   = UINT32_MAX;
        uint64_t frontier = 0;
    };

    struct ring_state {
        uint32_t node      = 0;
        uint64_t stable_id = 0;
    };

    try {
        std::map<uint32_t, node_state> nodes;
        std::vector<ring_state>        ring;
        uint64_t                       next_stable_id = 1;

        const auto is_ancestor = [&](uint32_t ancestor, uint32_t node) {
            uint32_t cursor = node;
            for (size_t steps = 0; steps <= nodes.size(); ++steps) {
                if (cursor == ancestor) {
                    return true;
                }
                const auto it = nodes.find(cursor);
                if (it == nodes.end() || it->second.parent == UINT32_MAX) {
                    return false;
                }
                cursor = it->second.parent;
            }
            return false;
        };

        for (const auto & event : events) {
            if (nodes.count(event.node) != 0 || event.frontier == 0 ||
                (event.parent_node != UINT32_MAX && nodes.count(event.parent_node) == 0)) {
                return score;
            }
            if (event.parent_node != UINT32_MAX && nodes.at(event.parent_node).frontier > event.frontier) {
                return score;
            }
            nodes.emplace(event.node, node_state{ event.parent_node, event.frontier });

            uint64_t recovered_frontier = 0;
            for (const auto & retained : ring) {
                if (is_ancestor(retained.node, event.node)) {
                    recovered_frontier = std::max(recovered_frontier, nodes.at(retained.node).frontier);
                }
            }
            const uint64_t replay = event.frontier - recovered_frontier;
            if (replay > UINT64_MAX - score.total_replay_tokens) {
                return server_cache_retention_sim_score{};
            }
            score.total_replay_tokens += replay;
            score.replay_samples.push_back(replay);
            if (event.deep_rewind && recovered_frontier == 0) {
                score.zero_coverage_deep_rewinds++;
            }

            // The production pre-pass retires at most one conclusive stale
            // member per scheduler turn. The corpus abstracts the quiescent
            // interval between requests, so drain those deterministic passes
            // before presenting the next publication opportunity.
            for (auto it = ring.begin(); it != ring.end();) {
                if (!is_ancestor(it->node, event.node)) {
                    it = ring.erase(it);
                    score.checkpoint_mutations++;
                } else {
                    ++it;
                }
            }

            std::vector<server_cache_retention_member> candidates;
            candidates.reserve(ring.size() + 1);
            std::map<uint32_t, ring_state> by_ordinal;
            uint32_t                       ordinal = 0;
            for (const auto & retained : ring) {
                server_cache_retention_member candidate;
                candidate.ordinal   = ordinal;
                candidate.stable_id = retained.stable_id;
                candidate.frontier  = nodes.at(retained.node).frontier;
                candidates.push_back(candidate);
                by_ordinal.emplace(ordinal, retained);
                ordinal++;
            }
            server_cache_retention_member incoming;
            incoming.ordinal   = ordinal;
            incoming.stable_id = next_stable_id++;
            if (next_stable_id == 0) {
                return server_cache_retention_sim_score{};
            }
            incoming.frontier = event.frontier;
            incoming.incoming = true;
            candidates.push_back(incoming);
            by_ordinal.emplace(incoming.ordinal, ring_state{ event.node, incoming.stable_id });

            const auto plan = server_cache_plan_retention_set(config.policy, event.frontier, candidates);
            if (plan.status != server_cache_retention_policy_status::ok) {
                return server_cache_retention_sim_score{};
            }
            if (!plan.incoming_selected) {
                score.publication_skips++;
                continue;
            }

            bool certified = ring.size() < config.policy.capacity;
            if (!certified) {
                for (const uint32_t victim_ordinal : plan.exclusion_order) {
                    const auto victim_it = by_ordinal.find(victim_ordinal);
                    if (victim_it == by_ordinal.end()) {
                        continue;
                    }
                    const auto     victim_node     = victim_it->second.node;
                    const uint64_t victim_frontier = nodes.at(victim_node).frontier;
                    const uint64_t age             = event.frontier - victim_frontier;
                    uint64_t       replay_cap      = config.recent_replay_cap;
                    if (age >= config.policy.bucket_base) {
                        replay_cap = config.historical_replay_cap;
                        const uint32_t bucket =
                            historical_bucket(age, config.policy.bucket_base, config.policy.bucket_growth);
                        uint64_t lower = config.policy.bucket_base;
                        for (uint32_t i = 1; i < bucket; ++i) {
                            lower = saturated_multiply(lower, config.policy.bucket_growth);
                        }
                        const uint64_t upper = saturated_multiply(lower, config.policy.bucket_growth);
                        const uint64_t width = upper == UINT64_MAX ? UINT64_MAX : upper - lower;
                        replay_cap           = std::min(replay_cap, width);
                    } else {
                        replay_cap = std::min(replay_cap, config.policy.bucket_base);
                    }

                    for (const auto & recovery : ring) {
                        if (recovery.node == victim_node || !is_ancestor(recovery.node, victim_node)) {
                            continue;
                        }
                        const uint64_t recovery_frontier = nodes.at(recovery.node).frontier;
                        if (recovery_frontier <= victim_frontier && victim_frontier - recovery_frontier <= replay_cap) {
                            certified = true;
                            break;
                        }
                    }
                    if (certified) {
                        break;
                    }
                }
            }
            if (!certified) {
                score.publication_skips++;
                continue;
            }

            std::vector<ring_state> next_ring;
            next_ring.reserve(plan.desired.size());
            for (const uint32_t desired_ordinal : plan.desired) {
                const auto it = by_ordinal.find(desired_ordinal);
                if (it == by_ordinal.end()) {
                    return server_cache_retention_sim_score{};
                }
                next_ring.push_back(it->second);
            }
            if (ring.size() >= config.policy.capacity) {
                score.checkpoint_mutations++;
            }
            score.checkpoint_mutations++;
            ring = std::move(next_ring);
        }

        std::sort(score.replay_samples.begin(), score.replay_samples.end());
        if (!score.replay_samples.empty()) {
            const size_t p95_index  = (score.replay_samples.size() * 95 + 99) / 100 - 1;
            const size_t p99_index  = (score.replay_samples.size() * 99 + 99) / 100 - 1;
            score.p95_replay_tokens = score.replay_samples[p95_index];
            score.p99_replay_tokens = score.replay_samples[p99_index];
            score.max_replay_tokens = score.replay_samples.back();
        }
        score.valid = true;
        return score;
    } catch (...) {
        return server_cache_retention_sim_score{};
    }
}

server_cache_retention_sim_score server_cache_simulate_fifo(
    uint32_t                                              capacity,
    const std::vector<server_cache_retention_sim_event> & events) noexcept {
    server_cache_retention_sim_score score;
    if (capacity == 0) {
        return score;
    }

    struct node_state {
        uint32_t parent   = UINT32_MAX;
        uint64_t frontier = 0;
    };

    try {
        std::map<uint32_t, node_state> nodes;
        std::vector<uint32_t>          ring;
        const auto                     is_ancestor = [&](uint32_t ancestor, uint32_t node) {
            uint32_t cursor = node;
            for (size_t steps = 0; steps <= nodes.size(); ++steps) {
                if (cursor == ancestor) {
                    return true;
                }
                const auto it = nodes.find(cursor);
                if (it == nodes.end() || it->second.parent == UINT32_MAX) {
                    return false;
                }
                cursor = it->second.parent;
            }
            return false;
        };
        for (const auto & event : events) {
            if (nodes.count(event.node) != 0 || event.frontier == 0 ||
                (event.parent_node != UINT32_MAX && nodes.count(event.parent_node) == 0) ||
                (event.parent_node != UINT32_MAX && nodes.at(event.parent_node).frontier > event.frontier)) {
                return server_cache_retention_sim_score{};
            }
            nodes.emplace(event.node, node_state{ event.parent_node, event.frontier });
            uint64_t recovered = 0;
            for (const uint32_t retained : ring) {
                if (is_ancestor(retained, event.node)) {
                    recovered = std::max(recovered, nodes.at(retained).frontier);
                }
            }
            const uint64_t replay = event.frontier - recovered;
            if (replay > UINT64_MAX - score.total_replay_tokens) {
                return server_cache_retention_sim_score{};
            }
            score.total_replay_tokens += replay;
            score.replay_samples.push_back(replay);
            if (event.deep_rewind && recovered == 0) {
                score.zero_coverage_deep_rewinds++;
            }

            for (auto it = ring.begin(); it != ring.end();) {
                if (!is_ancestor(*it, event.node)) {
                    it = ring.erase(it);
                    score.checkpoint_mutations++;
                } else {
                    ++it;
                }
            }
            if (ring.size() >= capacity) {
                ring.erase(ring.begin());
                score.checkpoint_mutations++;
            }
            ring.push_back(event.node);
            score.checkpoint_mutations++;
        }
        std::sort(score.replay_samples.begin(), score.replay_samples.end());
        if (!score.replay_samples.empty()) {
            const size_t p95_index  = (score.replay_samples.size() * 95 + 99) / 100 - 1;
            const size_t p99_index  = (score.replay_samples.size() * 99 + 99) / 100 - 1;
            score.p95_replay_tokens = score.replay_samples[p95_index];
            score.p99_replay_tokens = score.replay_samples[p99_index];
            score.max_replay_tokens = score.replay_samples.back();
        }
        score.valid = true;
        return score;
    } catch (...) {
        return server_cache_retention_sim_score{};
    }
}
