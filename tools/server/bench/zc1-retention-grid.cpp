#include "../server-cache-retention-policy.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct candidate_result {
    uint32_t                         recent_floor              = 0;
    uint32_t                         bucket_multiplier         = 0;
    uint32_t                         growth                    = 0;
    uint32_t                         recent_cap_multiplier     = 0;
    uint32_t                         historical_cap_multiplier = 0;
    uint32_t                         minimum_historical        = 0;
    bool                             v7_eligible               = false;
    server_cache_retention_sim_score score;
};

bool add_checked(uint64_t value, uint64_t & total) {
    if (value > UINT64_MAX - total) {
        return false;
    }
    total += value;
    return true;
}

bool multiply_checked(uint64_t value, uint32_t multiplier, uint64_t & out) {
    if (multiplier != 0 && value > UINT64_MAX / multiplier) {
        out = 0;
        return false;
    }
    out = value * multiplier;
    return true;
}

json score_json(const server_cache_retention_sim_score & score) {
    return {
        { "valid",                      score.valid                      },
        { "total_replay_tokens",        score.total_replay_tokens        },
        { "p95_replay_tokens",          score.p95_replay_tokens          },
        { "p99_replay_tokens",          score.p99_replay_tokens          },
        { "max_replay_tokens",          score.max_replay_tokens          },
        { "zero_coverage_deep_rewinds", score.zero_coverage_deep_rewinds },
        { "checkpoint_mutations",       score.checkpoint_mutations       },
        { "publication_skips",          score.publication_skips          },
    };
}

json candidate_json(const candidate_result & candidate) {
    json out = {
        { "recent_floor",                     candidate.recent_floor              },
        { "bucket_base_multiplier",           candidate.bucket_multiplier         },
        { "bucket_growth",                    candidate.growth                    },
        { "recent_replay_cap_multiplier",     candidate.recent_cap_multiplier     },
        { "historical_replay_cap_multiplier", candidate.historical_cap_multiplier },
        { "minimum_historical",               candidate.minimum_historical        },
        { "v7_eligible",                      candidate.v7_eligible                },
    };
    out.update(score_json(candidate.score));
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string input_path;
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--events" || arg == "--out") && i + 1 < argc) {
            (arg == "--events" ? input_path : output_path) = argv[++i];
        } else {
            std::cerr << "usage: zc1-retention-grid --events FILE --out FILE\n";
            return 2;
        }
    }
    if (input_path.empty() || output_path.empty()) {
        std::cerr << "usage: zc1-retention-grid --events FILE --out FILE\n";
        return 2;
    }

    try {
        std::ifstream input(input_path);
        if (!input) {
            throw std::runtime_error("cannot open events input");
        }
        json corpus;
        input >> corpus;
        if (corpus.value("schema", std::string()) != "zc1_retention_events/v1" || !corpus.contains("capacity") ||
            !corpus.contains("checkpoint_min_step") || !corpus.contains("source_trace_sha256") ||
            !corpus.contains("chains") || !corpus.at("chains").is_array()) {
            throw std::runtime_error("invalid event corpus envelope");
        }
        const uint32_t capacity = corpus.at("capacity").get<uint32_t>();
        const uint64_t min_step = corpus.at("checkpoint_min_step").get<uint64_t>();
        if (capacity == 0 || min_step == 0 || corpus.at("chains").empty()) {
            throw std::runtime_error("invalid event corpus constants");
        }

        std::vector<std::vector<server_cache_retention_sim_event>> chains;
        for (const auto & chain_json : corpus.at("chains")) {
            if (!chain_json.is_object() || !chain_json.contains("events") || !chain_json.at("events").is_array() ||
                chain_json.at("events").empty()) {
                throw std::runtime_error("invalid chain");
            }
            std::vector<server_cache_retention_sim_event> events;
            for (const auto & event_json : chain_json.at("events")) {
                server_cache_retention_sim_event event;
                event.node = event_json.at("node").get<uint32_t>();
                if (event_json.at("parent_node").is_null()) {
                    event.parent_node = UINT32_MAX;
                } else {
                    event.parent_node = event_json.at("parent_node").get<uint32_t>();
                }
                event.frontier    = event_json.at("frontier_tokens").get<uint64_t>();
                event.deep_rewind = event_json.at("deep_rewind").get<bool>();
                events.push_back(event);
            }
            chains.push_back(std::move(events));
        }

        server_cache_retention_sim_score fifo;
        fifo.valid = true;
        std::vector<uint64_t> fifo_samples;
        for (const auto & chain : chains) {
            const auto chain_score = server_cache_simulate_fifo(capacity, chain);
            fifo.valid             = fifo.valid && chain_score.valid;
            if (!chain_score.valid || !add_checked(chain_score.total_replay_tokens, fifo.total_replay_tokens) ||
                !add_checked(chain_score.zero_coverage_deep_rewinds, fifo.zero_coverage_deep_rewinds) ||
                !add_checked(chain_score.checkpoint_mutations, fifo.checkpoint_mutations)) {
                fifo.valid = false;
                break;
            }
            fifo_samples.insert(fifo_samples.end(), chain_score.replay_samples.begin(),
                                chain_score.replay_samples.end());
        }
        if (!fifo.valid) {
            throw std::runtime_error("invalid FIFO counterfactual");
        }
        std::sort(fifo_samples.begin(), fifo_samples.end());
        fifo.p95_replay_tokens = fifo_samples[(fifo_samples.size() * 95 + 99) / 100 - 1];
        fifo.p99_replay_tokens = fifo_samples[(fifo_samples.size() * 99 + 99) / 100 - 1];
        fifo.max_replay_tokens = fifo_samples.back();
        fifo.replay_samples    = std::move(fifo_samples);

        std::vector<candidate_result> candidates;
        for (const uint32_t recent_floor : { 1U, 2U, 3U, 4U }) {
            for (const uint32_t bucket_multiplier : { 1U, 2U, 4U }) {
                for (const uint32_t growth : { 2U, 3U, 4U }) {
                    for (const uint32_t recent_cap_multiplier : { 1U, 2U }) {
                        for (const uint32_t historical_cap_multiplier : { 2U, 4U, 8U }) {
                            for (const uint32_t minimum_historical : { 1U, 2U }) {
                                candidate_result candidate;
                                candidate.recent_floor              = recent_floor;
                                candidate.bucket_multiplier         = bucket_multiplier;
                                candidate.growth                    = growth;
                                candidate.recent_cap_multiplier     = recent_cap_multiplier;
                                candidate.historical_cap_multiplier = historical_cap_multiplier;
                                candidate.minimum_historical        = minimum_historical;
                                candidate.score.valid               = true;
                                std::vector<uint64_t> samples;
                                for (const auto & chain : chains) {
                                    server_cache_retention_sim_config config;
                                    uint64_t                          bucket_base    = 0;
                                    uint64_t                          recent_cap     = 0;
                                    uint64_t                          historical_cap = 0;
                                    if (!multiply_checked(min_step, bucket_multiplier, bucket_base) ||
                                        !multiply_checked(min_step, recent_cap_multiplier, recent_cap) ||
                                        !multiply_checked(min_step, historical_cap_multiplier, historical_cap)) {
                                        candidate.score.valid = false;
                                        candidates.push_back(std::move(candidate));
                                        continue;
                                    }
                                    config.policy.capacity           = capacity;
                                    config.policy.recent_floor       = recent_floor;
                                    config.policy.minimum_historical = minimum_historical;
                                    config.policy.bucket_base        = bucket_base;
                                    config.policy.bucket_growth      = growth;
                                    config.recent_replay_cap         = recent_cap;
                                    config.historical_replay_cap     = historical_cap;
                                    const auto score                 = server_cache_simulate_retention(config, chain);
                                    candidate.score.valid            = candidate.score.valid && score.valid;
                                    if (!score.valid ||
                                        !add_checked(score.total_replay_tokens, candidate.score.total_replay_tokens) ||
                                        !add_checked(score.zero_coverage_deep_rewinds,
                                                     candidate.score.zero_coverage_deep_rewinds) ||
                                        !add_checked(score.checkpoint_mutations,
                                                     candidate.score.checkpoint_mutations) ||
                                        !add_checked(score.publication_skips, candidate.score.publication_skips)) {
                                        candidate.score.valid = false;
                                        break;
                                    }
                                    samples.insert(samples.end(), score.replay_samples.begin(),
                                                   score.replay_samples.end());
                                }
                                if (candidate.score.valid) {
                                    std::sort(samples.begin(), samples.end());
                                    const size_t p95_index = (samples.size() * 95 + 99) / 100 - 1;
                                    const size_t p99_index = (samples.size() * 99 + 99) / 100 - 1;
                                    candidate.score.p95_replay_tokens = samples[p95_index];
                                    candidate.score.p99_replay_tokens = samples[p99_index];
                                    candidate.score.max_replay_tokens = samples.back();
                                    candidate.score.replay_samples    = std::move(samples);
                                }
                                candidates.push_back(std::move(candidate));
                            }
                        }
                    }
                }
            }
        }

        uint64_t total_allowance = (fifo.total_replay_tokens + 999) / 1000;
        uint64_t total_limit = 0;
        uint64_t p95_limit = 0;
        if (!add_checked(fifo.total_replay_tokens, total_limit) ||
            !add_checked(total_allowance, total_limit) ||
            !add_checked(fifo.p95_replay_tokens, p95_limit) ||
            !add_checked(min_step, p95_limit)) {
            throw std::runtime_error("V7 acceptance-limit overflow");
        }
        for (auto & candidate : candidates) {
            const bool tail_better = fifo.zero_coverage_deep_rewinds == 0
                ? candidate.score.p99_replay_tokens <= fifo.p99_replay_tokens &&
                    candidate.score.max_replay_tokens <= fifo.max_replay_tokens
                : candidate.score.p99_replay_tokens < fifo.p99_replay_tokens &&
                    candidate.score.max_replay_tokens < fifo.max_replay_tokens;
            candidate.v7_eligible = candidate.score.valid &&
                candidate.score.zero_coverage_deep_rewinds == 0 &&
                candidate.score.total_replay_tokens <= total_limit &&
                candidate.score.p95_replay_tokens <= p95_limit &&
                tail_better &&
                candidate.score.checkpoint_mutations <= fifo.checkpoint_mutations &&
                candidate.score.publication_skips == 0;
        }

        std::sort(candidates.begin(), candidates.end(), [](const candidate_result & a, const candidate_result & b) {
            return std::make_tuple(!a.v7_eligible, !a.score.valid, a.score.total_replay_tokens,
                                   a.score.p95_replay_tokens, a.score.p99_replay_tokens,
                                   a.score.max_replay_tokens, a.score.checkpoint_mutations, a.recent_floor,
                                   a.bucket_multiplier, a.growth, a.recent_cap_multiplier, a.historical_cap_multiplier,
                                   a.minimum_historical) <
                   std::make_tuple(!b.v7_eligible, !b.score.valid, b.score.total_replay_tokens,
                                   b.score.p95_replay_tokens, b.score.p99_replay_tokens,
                                   b.score.max_replay_tokens, b.score.checkpoint_mutations, b.recent_floor,
                                   b.bucket_multiplier, b.growth, b.recent_cap_multiplier, b.historical_cap_multiplier,
                                   b.minimum_historical);
        });
        if (candidates.empty() || !candidates.front().v7_eligible) {
            throw std::runtime_error("no V7-eligible retention candidate");
        }

        json all = json::array();
        for (const auto & candidate : candidates) {
            all.push_back(candidate_json(candidate));
        }
        const json result = {
            { "schema",              "zc1_retention_grid_result/v2"     },
            { "source_trace_sha256", corpus.at("source_trace_sha256")   },
            { "candidate_count",     candidates.size()                  },
            { "v7_contract", {
                { "aggregate_allowance_per_mille", 1 },
                { "total_replay_limit", total_limit },
                { "p95_replay_limit", p95_limit },
                { "zero_coverage_deep_rewinds", 0 },
                { "tail_strict_when_fifo_has_deep_miss", true },
                { "publication_skips", 0 }
            } },
            { "fifo",                score_json(fifo)                   },
            { "winner",              candidate_json(candidates.front()) },
            { "candidates",          std::move(all)                     },
        };
        std::ofstream output(output_path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open output");
        }
        output << result.dump() << '\n';
        output.close();
        if (!output) {
            throw std::runtime_error("cannot write output");
        }
        std::cout << "ZC1_RETENTION_GRID PASS candidates=" << candidates.size() << " out=" << output_path << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "ZC1_RETENTION_GRID FAIL: " << error.what() << '\n';
        return 1;
    }
}
