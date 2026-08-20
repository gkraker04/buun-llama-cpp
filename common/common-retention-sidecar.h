#pragma once

#include "chat.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t COMMON_RETENTION_SIDECAR_VERSION = 1;
constexpr uint32_t COMMON_RETENTION_TURN_TABLE_VERSION = 1;
// Stable ids are shifted left by one to form the pool-qualified durable key.
// Keep the ceiling beside the codec/allocator contract so the server cannot drift.
constexpr uint64_t COMMON_RETENTION_MAX_POOL_COUNTER =
    (UINT64_MAX >> 1) - 1;

// Fidelity of the durable turn-table source.
enum class common_retention_source_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

enum class common_retention_pool : uint8_t {
    attention = 0,
    recurrent,
    _count,
};

enum class common_retention_artifact_kind : uint8_t {
    live_slot = 0,
    host_entry,
    checkpoint,
    _count,
};

// Availability of the derived retention score. This is intentionally distinct
// from source_state: future scoring policy may refuse an otherwise valid table.
enum class common_retention_score_state : uint8_t {
    known = 0,
    unavailable,
    _count,
};

struct common_retention_turn_boundary {
    uint64_t ordinal  = 0;
    uint64_t token_pos = 0;
    uint64_t token_end = 0;
};

struct common_retention_turn_table {
    uint32_t version = COMMON_RETENTION_TURN_TABLE_VERSION;
    common_retention_source_state source = common_retention_source_state::unavailable;
    uint64_t token_count = 0;
    std::vector<common_retention_turn_boundary> boundaries;

    bool valid() const noexcept;
};

struct common_retention_stamp {
    common_retention_score_state state = common_retention_score_state::unavailable;
    common_retention_pool pool = common_retention_pool::attention;
    bool soft_leased = false;
    // Serialized scoring output: consumers must not have to reimplement the
    // geometry policy to recover the non-evictable result.
    bool mandatory_anchor = false;
    uint64_t stable_id = 0;
    uint64_t recency_ordinal = 0;
    uint64_t mapped_turn_ordinal = 0;
    uint64_t anchor_rank = 0;
    uint64_t coverage_tokens = 0;

    bool valid() const noexcept;
};

struct common_retention_artifact_record {
    common_retention_artifact_kind kind = common_retention_artifact_kind::live_slot;
    common_retention_turn_table turns;
    common_retention_stamp stamp;

    bool valid() const noexcept;
};

struct common_retention_sidecar_snapshot {
    uint32_t version = COMMON_RETENTION_SIDECAR_VERSION;
    std::array<uint64_t, size_t(common_retention_pool::_count)> recency_high_water = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> stable_high_water = {};
    std::vector<common_retention_artifact_record> artifacts;

    bool valid() const noexcept;
};

bool common_retention_build_turn_table(
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t token_count,
        common_retention_turn_table & out) noexcept;

bool common_retention_score(
        const common_retention_turn_table & turns,
        uint64_t frontier,
        common_retention_stamp & stamp) noexcept;

bool common_retention_sidecar_encode(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & out) noexcept;

// Exact one-record envelope size used by the in-memory catalog's C charge.
// This is arithmetic only: no payload allocation and no checksum computation.
bool common_retention_sidecar_artifact_encoded_size(
        const common_retention_artifact_record & artifact,
        uint64_t & out) noexcept;

bool common_retention_sidecar_decode(
        const uint8_t * data,
        size_t size,
        common_retention_sidecar_snapshot & out) noexcept;

class common_retention_allocator {
public:
    bool issue(common_retention_pool pool, common_retention_stamp & stamp) noexcept;
    bool import_snapshot(const common_retention_sidecar_snapshot & snapshot) noexcept;
    uint64_t recency_high_water(common_retention_pool pool) const noexcept;
    uint64_t stable_high_water(common_retention_pool pool) const noexcept;

private:
    std::array<uint64_t, size_t(common_retention_pool::_count)> next_recency = { 1, 1 };
    std::array<uint64_t, size_t(common_retention_pool::_count)> next_stable = { 1, 1 };
};
