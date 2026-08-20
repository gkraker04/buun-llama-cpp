#include "common-retention-sidecar.h"

#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <utility>

namespace {

constexpr uint32_t SIDECAR_MAGIC = 0x44533352; // "DS3R", canonical little-endian
constexpr size_t SIDECAR_HEADER_SIZE = 4 + 4 + 8 + 32;
constexpr size_t SIDECAR_SNAPSHOT_PREFIX_SIZE = 8*4 + 4;
constexpr size_t SIDECAR_ARTIFACT_FIXED_SIZE = 6 + 4 + 8*6 + 4;
constexpr size_t SIDECAR_BOUNDARY_SIZE = 8*3;
constexpr uint64_t MAX_SIDECAR_BYTES = 64ull * 1024 * 1024;
constexpr uint32_t MAX_ARTIFACTS = 8192;
constexpr uint32_t MAX_TURNS_PER_ARTIFACT = 8192;

void put_u8(std::vector<uint8_t> & out, uint8_t value) {
    out.push_back(value);
}

void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    uint8_t data[4];
    llama_store_le_u32(data, value);
    out.insert(out.end(), std::begin(data), std::end(data));
}

void put_u64(std::vector<uint8_t> & out, uint64_t value) {
    uint8_t data[8];
    llama_store_le_u64(data, value);
    out.insert(out.end(), std::begin(data), std::end(data));
}

struct reader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t pos = 0;

    bool bytes(void * dst, size_t n) {
        if (n > size - pos) {
            return false;
        }
        memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }

    bool u8(uint8_t & value) {
        return bytes(&value, sizeof(value));
    }

    bool u32(uint32_t & value) {
        uint8_t raw[4];
        if (!bytes(raw, sizeof(raw))) {
            return false;
        }
        value = 0;
        for (size_t i = 0; i < sizeof(raw); ++i) {
            value |= uint32_t(raw[i]) << (8*i);
        }
        return true;
    }

    bool u64(uint64_t & value) {
        uint8_t raw[8];
        if (!bytes(raw, sizeof(raw))) {
            return false;
        }
        value = 0;
        for (size_t i = 0; i < sizeof(raw); ++i) {
            value |= uint64_t(raw[i]) << (8*i);
        }
        return true;
    }
};

uint64_t ceil_log2(uint64_t value) {
    uint64_t result = 0;
    uint64_t power = 1;
    while (power < value) {
        if (power > std::numeric_limits<uint64_t>::max()/2) {
            return 64;
        }
        power <<= 1;
        result++;
    }
    return result;
}

bool encode_payload(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & payload) {
    put_u64(payload, snapshot.recency_high_water[0]);
    put_u64(payload, snapshot.recency_high_water[1]);
    put_u64(payload, snapshot.stable_high_water[0]);
    put_u64(payload, snapshot.stable_high_water[1]);
    put_u32(payload, uint32_t(snapshot.artifacts.size()));

    for (const auto & artifact : snapshot.artifacts) {
        put_u8(payload, uint8_t(artifact.kind));
        put_u8(payload, uint8_t(artifact.turns.source));
        put_u8(payload, uint8_t(artifact.stamp.state));
        put_u8(payload, uint8_t(artifact.stamp.pool));
        put_u8(payload, artifact.stamp.soft_leased ? 1 : 0);
        put_u8(payload, artifact.stamp.mandatory_anchor ? 1 : 0);
        put_u32(payload, artifact.turns.version);
        put_u64(payload, artifact.turns.token_count);
        put_u64(payload, artifact.stamp.stable_id);
        put_u64(payload, artifact.stamp.recency_ordinal);
        put_u64(payload, artifact.stamp.mapped_turn_ordinal);
        put_u64(payload, artifact.stamp.anchor_rank);
        put_u64(payload, artifact.stamp.coverage_tokens);
        put_u32(payload, uint32_t(artifact.turns.boundaries.size()));
        for (const auto & boundary : artifact.turns.boundaries) {
            put_u64(payload, boundary.ordinal);
            put_u64(payload, boundary.token_pos);
            put_u64(payload, boundary.token_end);
        }
    }
    return payload.size() <= MAX_SIDECAR_BYTES - SIDECAR_HEADER_SIZE;
}

bool decode_artifact(reader & in, common_retention_artifact_record & artifact) {
    uint8_t kind;
    uint8_t source;
    uint8_t score_state;
    uint8_t pool;
    uint8_t soft;
    uint8_t mandatory;
    uint32_t n_turns;
    if (!in.u8(kind) ||
        !in.u8(source) ||
        !in.u8(score_state) ||
        !in.u8(pool) ||
        !in.u8(soft) ||
        !in.u8(mandatory) ||
        !in.u32(artifact.turns.version) ||
        !in.u64(artifact.turns.token_count) ||
        !in.u64(artifact.stamp.stable_id) ||
        !in.u64(artifact.stamp.recency_ordinal) ||
        !in.u64(artifact.stamp.mapped_turn_ordinal) ||
        !in.u64(artifact.stamp.anchor_rank) ||
        !in.u64(artifact.stamp.coverage_tokens) ||
        !in.u32(n_turns)) {
        return false;
    }
    if (kind >= uint8_t(common_retention_artifact_kind::_count) ||
        source >= uint8_t(common_retention_source_state::_count) ||
        score_state >= uint8_t(common_retention_score_state::_count) ||
        pool >= uint8_t(common_retention_pool::_count) ||
        soft > 1 || mandatory > 1 ||
        n_turns > MAX_TURNS_PER_ARTIFACT) {
        return false;
    }
    artifact.kind = common_retention_artifact_kind(kind);
    artifact.turns.source = common_retention_source_state(source);
    artifact.stamp.state = common_retention_score_state(score_state);
    artifact.stamp.pool = common_retention_pool(pool);
    artifact.stamp.soft_leased = soft != 0;
    artifact.stamp.mandatory_anchor = mandatory != 0;
    artifact.turns.boundaries.resize(n_turns);
    for (auto & boundary : artifact.turns.boundaries) {
        if (!in.u64(boundary.ordinal) ||
            !in.u64(boundary.token_pos) ||
            !in.u64(boundary.token_end)) {
            return false;
        }
    }
    return artifact.valid();
}

} // namespace

bool common_retention_turn_table::valid() const noexcept {
    if (version != COMMON_RETENTION_TURN_TABLE_VERSION ||
        source >= common_retention_source_state::_count) {
        return false;
    }
    if (source == common_retention_source_state::unavailable) {
        return boundaries.empty();
    }
    if (boundaries.empty() ||
        boundaries.front().ordinal != 0 ||
        boundaries.front().token_pos != 0) {
        return false;
    }
    for (size_t i = 0; i < boundaries.size(); ++i) {
        const auto & cur = boundaries[i];
        if (cur.ordinal != i ||
            cur.token_pos > cur.token_end ||
            cur.token_end > token_count) {
            return false;
        }
        if (i > 0 && boundaries[i - 1].token_pos >= cur.token_pos) {
            return false;
        }
    }
    return true;
}

bool common_retention_stamp::valid() const noexcept {
    if (state >= common_retention_score_state::_count ||
        pool >= common_retention_pool::_count ||
        stable_id == 0 ||
        recency_ordinal == 0 ||
        stable_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        recency_ordinal > COMMON_RETENTION_MAX_POOL_COUNTER) {
        return false;
    }
    if (state == common_retention_score_state::unavailable) {
        return !mandatory_anchor && mapped_turn_ordinal == 0 && anchor_rank == 0;
    }
    return true;
}

bool common_retention_artifact_record::valid() const noexcept {
    if (kind >= common_retention_artifact_kind::_count ||
        !turns.valid() ||
        !stamp.valid()) {
        return false;
    }
    return stamp.state == common_retention_score_state::unavailable ||
           (turns.source == common_retention_source_state::known &&
            stamp.coverage_tokens <= turns.token_count &&
            stamp.mapped_turn_ordinal < turns.boundaries.size());
}

bool common_retention_sidecar_snapshot::valid() const noexcept {
    if (version != COMMON_RETENTION_SIDECAR_VERSION ||
        artifacts.size() > MAX_ARTIFACTS) {
        return false;
    }
    std::array<uint64_t, size_t(common_retention_pool::_count)> max_recency = {};
    std::array<uint64_t, size_t(common_retention_pool::_count)> max_stable = {};
    std::array<std::vector<uint64_t>, size_t(common_retention_pool::_count)> ids;
    for (const auto & artifact : artifacts) {
        if (!artifact.valid()) {
            return false;
        }
        const size_t pool = size_t(artifact.stamp.pool);
        max_recency[pool] = std::max(max_recency[pool], artifact.stamp.recency_ordinal);
        max_stable[pool] = std::max(max_stable[pool], artifact.stamp.stable_id);
        ids[pool].push_back(artifact.stamp.stable_id);
    }
    for (size_t pool = 0; pool < max_recency.size(); ++pool) {
        if (recency_high_water[pool] < max_recency[pool] ||
            stable_high_water[pool] < max_stable[pool] ||
            recency_high_water[pool] > COMMON_RETENTION_MAX_POOL_COUNTER ||
            stable_high_water[pool] > COMMON_RETENTION_MAX_POOL_COUNTER) {
            return false;
        }
        std::sort(ids[pool].begin(), ids[pool].end());
        if (std::adjacent_find(ids[pool].begin(), ids[pool].end()) != ids[pool].end()) {
            return false;
        }
    }
    return true;
}

bool common_retention_build_turn_table(
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t token_count,
        common_retention_turn_table & out) noexcept {
    out = {};
    try {
        common_retention_turn_table built;
        built.token_count = token_count;
        if (!source_known || spans.spans.empty()) {
            out = std::move(built);
            return true;
        }

        built.source = common_retention_source_state::known;
        built.boundaries.reserve(spans.spans.size() + 1);
        built.boundaries.push_back({ 0, 0, 0 });
        uint64_t prior_end = 0;
        bool saw_user = false;
        for (const auto & span : spans.spans) {
            if (!span.valid() ||
                span.pos > token_count ||
                span.len > token_count - span.pos ||
                span.pos < prior_end) {
                return false;
            }
            const uint64_t end = uint64_t(span.pos) + uint64_t(span.len);
            prior_end = end;
            if (span.role != COMMON_CHAT_ROLE_USER) {
                continue;
            }
            saw_user = true;
            if (span.pos == 0) {
                built.boundaries.front().token_end = end;
                continue;
            }
            if (built.boundaries.back().token_pos >= span.pos) {
                return false;
            }
            built.boundaries.push_back({
                uint64_t(built.boundaries.size()),
                uint64_t(span.pos),
                end,
            });
        }
        if (!saw_user) {
            built.source = common_retention_source_state::unavailable;
            built.boundaries.clear();
        }
        if (!built.valid()) {
            return false;
        }
        out = std::move(built);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_score(
        const common_retention_turn_table & turns,
        uint64_t frontier,
        common_retention_stamp & stamp) noexcept {
    stamp.state = common_retention_score_state::unavailable;
    stamp.mandatory_anchor = false;
    stamp.mapped_turn_ordinal = 0;
    stamp.anchor_rank = 0;
    if (!turns.valid() ||
        turns.source != common_retention_source_state::known ||
        frontier > turns.token_count) {
        return false;
    }

    const auto upper = std::upper_bound(
        turns.boundaries.begin(), turns.boundaries.end(), frontier,
        [](uint64_t pos, const common_retention_turn_boundary & boundary) {
            return pos < boundary.token_pos;
        });
    const size_t mapped =
        size_t(std::distance(turns.boundaries.begin(), upper) - 1);
    const uint64_t n = turns.boundaries.size() - 1;
    stamp.state = common_retention_score_state::known;
    stamp.mapped_turn_ordinal = mapped;
    stamp.mandatory_anchor = mapped == 0 || mapped == n;
    if (stamp.mandatory_anchor) {
        return true;
    }

    const uint64_t k_max = ceil_log2(std::max<uint64_t>(n, 1));
    for (uint64_t k = 0; k <= k_max && k < 64; ++k) {
        const uint64_t distance = uint64_t(1) << k;
        const uint64_t index = distance >= n ? 0 : n - distance;
        if (index == mapped) {
            stamp.anchor_rank = k_max + 1 - k;
            break;
        }
        if (index == 0) {
            break;
        }
    }
    return true;
}

bool common_retention_sidecar_encode(
        const common_retention_sidecar_snapshot & snapshot,
        std::vector<uint8_t> & out) noexcept {
    try {
        if (!snapshot.valid()) {
            return false;
        }
        std::vector<uint8_t> payload;
        if (!encode_payload(snapshot, payload)) {
            return false;
        }
        llama_sha256 hash;
        hash.update(payload.data(), payload.size());
        const auto digest = hash.finish();

        std::vector<uint8_t> encoded;
        encoded.reserve(SIDECAR_HEADER_SIZE + payload.size());
        put_u32(encoded, SIDECAR_MAGIC);
        put_u32(encoded, COMMON_RETENTION_SIDECAR_VERSION);
        put_u64(encoded, SIDECAR_HEADER_SIZE + payload.size());
        encoded.insert(encoded.end(), digest.begin(), digest.end());
        encoded.insert(encoded.end(), payload.begin(), payload.end());
        out = std::move(encoded);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_sidecar_artifact_encoded_size(
        const common_retention_artifact_record & artifact,
        uint64_t & out) noexcept {
    out = 0;
    if (!artifact.valid() ||
        artifact.turns.boundaries.size() > MAX_TURNS_PER_ARTIFACT) {
        return false;
    }
    const uint64_t n_boundaries = artifact.turns.boundaries.size();
    const uint64_t fixed =
        SIDECAR_HEADER_SIZE +
        SIDECAR_SNAPSHOT_PREFIX_SIZE +
        SIDECAR_ARTIFACT_FIXED_SIZE;
    if (n_boundaries >
        (MAX_SIDECAR_BYTES - fixed)/SIDECAR_BOUNDARY_SIZE) {
        return false;
    }
    out = fixed + n_boundaries*SIDECAR_BOUNDARY_SIZE;
    return out <= MAX_SIDECAR_BYTES;
}

bool common_retention_sidecar_decode(
        const uint8_t * data,
        size_t size,
        common_retention_sidecar_snapshot & out) noexcept {
    // Fail closed even when the caller reuses an object that previously held valid
    // evidence: no decode failure may leave that prior record observable.
    out.version = 0;
    out.recency_high_water = {};
    out.stable_high_water = {};
    out.artifacts.clear();
    try {
        if (!data || size < SIDECAR_HEADER_SIZE || size > MAX_SIDECAR_BYTES) {
            return false;
        }
        reader header { data, size, 0 };
        uint32_t magic;
        uint32_t version;
        uint64_t declared_size;
        std::array<uint8_t, 32> expected;
        if (!header.u32(magic) ||
            !header.u32(version) ||
            !header.u64(declared_size) ||
            !header.bytes(expected.data(), expected.size()) ||
            magic != SIDECAR_MAGIC ||
            version != COMMON_RETENTION_SIDECAR_VERSION ||
            declared_size != size) {
            return false;
        }

        llama_sha256 hash;
        hash.update(data + SIDECAR_HEADER_SIZE, size - SIDECAR_HEADER_SIZE);
        if (hash.finish() != expected) {
            return false;
        }

        reader payload {
            data + SIDECAR_HEADER_SIZE,
            size - SIDECAR_HEADER_SIZE,
            0,
        };
        common_retention_sidecar_snapshot decoded;
        uint32_t n_artifacts;
        if (!payload.u64(decoded.recency_high_water[0]) ||
            !payload.u64(decoded.recency_high_water[1]) ||
            !payload.u64(decoded.stable_high_water[0]) ||
            !payload.u64(decoded.stable_high_water[1]) ||
            !payload.u32(n_artifacts) ||
            n_artifacts > MAX_ARTIFACTS) {
            return false;
        }
        decoded.artifacts.resize(n_artifacts);
        for (auto & artifact : decoded.artifacts) {
            if (!decode_artifact(payload, artifact)) {
                return false;
            }
        }
        if (payload.pos != payload.size || !decoded.valid()) {
            return false;
        }
        out = std::move(decoded);
        return true;
    } catch (...) {
        return false;
    }
}

bool common_retention_allocator::issue(
        common_retention_pool pool,
        common_retention_stamp & stamp) noexcept {
    const size_t index = size_t(pool);
    if (index >= next_recency.size() ||
        next_recency[index] == 0 ||
        next_stable[index] == 0 ||
        next_recency[index] > COMMON_RETENTION_MAX_POOL_COUNTER ||
        next_stable[index] > COMMON_RETENTION_MAX_POOL_COUNTER) {
        return false;
    }
    stamp.pool = pool;
    stamp.recency_ordinal = next_recency[index]++;
    stamp.stable_id = next_stable[index]++;
    return true;
}

bool common_retention_allocator::import_snapshot(
        const common_retention_sidecar_snapshot & imported) noexcept {
    if (!imported.valid()) {
        return false;
    }
    for (size_t i = 0; i < next_recency.size(); ++i) {
        if (imported.recency_high_water[i] >=
                COMMON_RETENTION_MAX_POOL_COUNTER ||
            imported.stable_high_water[i] >=
                COMMON_RETENTION_MAX_POOL_COUNTER) {
            return false;
        }
    }
    for (size_t i = 0; i < next_recency.size(); ++i) {
        next_recency[i] =
            std::max(next_recency[i], imported.recency_high_water[i] + 1);
        next_stable[i] =
            std::max(next_stable[i], imported.stable_high_water[i] + 1);
    }
    return true;
}

uint64_t common_retention_allocator::recency_high_water(
        common_retention_pool pool) const noexcept {
    const size_t index = size_t(pool);
    return index < next_recency.size() ? next_recency[index] - 1 : 0;
}

uint64_t common_retention_allocator::stable_high_water(
        common_retention_pool pool) const noexcept {
    const size_t index = size_t(pool);
    return index < next_stable.size() ? next_stable[index] - 1 : 0;
}
