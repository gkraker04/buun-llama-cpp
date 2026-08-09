#include "server-cache-fingerprint.h"
#include "server-cache-calibration-model.h"

#include "../../src/llama-sha256.h"
#include "../../common/build-info.h"
#include "../../common/common.h"
#include "../../common/common-cache-plan-estimate.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <tuple>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
#  include <cpuid.h>
#endif

namespace {

constexpr size_t MAX_CODEC_BYTES = 1024 * 1024;
// Frozen total fingerprint-worker region budget. The in-object codec buffer
// reserves a small suffix for the combiner state.
constexpr size_t FINGERPRINT_ARENA_BYTES = 1024 * 1024;
constexpr char ARTIFACT_DOMAIN[] = "buun-zc-cost-structures-v1";
constexpr char CONFIG_DOMAIN[] = "buun-zc-config-v1";
constexpr char EXEC_DOMAIN[] = "buun-zc-exec-v2";
constexpr char ADAPTER_APPLICATION_DOMAIN[] =
    "buun-zc-adapter-application-v1";

void append_u16(std::vector<uint8_t> & out, uint16_t value) {
    out.push_back(uint8_t(value));
    out.push_back(uint8_t(value >> 8));
}

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8) {
        out.push_back(uint8_t(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8) {
        out.push_back(uint8_t(value >> shift));
    }
}

std::vector<uint8_t> little_u16(uint16_t value) {
    std::vector<uint8_t> out;
    out.reserve(2);
    append_u16(out, value);
    return out;
}

std::vector<uint8_t> little_u32(uint32_t value) {
    std::vector<uint8_t> out;
    out.reserve(4);
    append_u32(out, value);
    return out;
}

std::vector<uint8_t> little_u64(uint64_t value) {
    std::vector<uint8_t> out;
    out.reserve(8);
    append_u64(out, value);
    return out;
}

bool valid_utf8(const uint8_t * data, size_t size) noexcept {
    size_t i = 0;
    while (i < size) {
        const uint8_t c = data[i++];
        if (c == 0) {
            return false;
        }
        if (c < 0x80) {
            continue;
        }
        uint32_t code = 0;
        size_t tails = 0;
        uint32_t minimum = 0;
        if ((c & 0xe0) == 0xc0) {
            code = c & 0x1f;
            tails = 1;
            minimum = 0x80;
        } else if ((c & 0xf0) == 0xe0) {
            code = c & 0x0f;
            tails = 2;
            minimum = 0x800;
        } else if ((c & 0xf8) == 0xf0) {
            code = c & 0x07;
            tails = 3;
            minimum = 0x10000;
        } else {
            return false;
        }
        if (tails > size - i) {
            return false;
        }
        while (tails-- != 0) {
            const uint8_t tail = data[i++];
            if ((tail & 0xc0) != 0x80) {
                return false;
            }
            code = (code << 6) | (tail & 0x3f);
        }
        if (code < minimum || code > 0x10ffff ||
            (code >= 0xd800 && code <= 0xdfff)) {
            return false;
        }
    }
    return true;
}

bool append_bounded(
        std::vector<uint8_t> & out, const void * data, size_t size) {
    if (size > MAX_CODEC_BYTES - out.size()) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    if (!data) {
        return false;
    }
    const auto * first = static_cast<const uint8_t *>(data);
    out.insert(out.end(), first, first + size);
    return true;
}

bool valid_artifact_role(server_cache_fingerprint_artifact_role role) {
    return role >= server_cache_fingerprint_artifact_role::target &&
           role <= server_cache_fingerprint_artifact_role::adapter;
}

constexpr std::array<server_cache_fingerprint_field_type, 32> FIELD_TYPES = {
    server_cache_fingerprint_field_type::u32,      // 1 estimator_version
    server_cache_fingerprint_field_type::u32,      // 2 formula_version
    server_cache_fingerprint_field_type::u32,      // 3 procedure_version
    server_cache_fingerprint_field_type::digest32, // 4 server_build_cost_abi
    server_cache_fingerprint_field_type::utf8,     // 5 backend_build_id
    server_cache_fingerprint_field_type::u32,      // 6 cuda_driver_version
    server_cache_fingerprint_field_type::u32,      // 7 cuda_runtime_version
    server_cache_fingerprint_field_type::utf8,     // 8 cpu_backend_versions
    server_cache_fingerprint_field_type::bytes,    // 9 device_records_v1
    server_cache_fingerprint_field_type::bytes,    // 10 device_topology_v1
    server_cache_fingerprint_field_type::bytes,    // 11 effective_placement_v1
    server_cache_fingerprint_field_type::u32,      // 12 n_batch
    server_cache_fingerprint_field_type::u32,      // 13 n_ubatch
    server_cache_fingerprint_field_type::u32,      // 14 prompt_threads
    server_cache_fingerprint_field_type::u32,      // 15 batch_threads
    server_cache_fingerprint_field_type::bytes,    // 16 prompt_thread_pinning_v1
    server_cache_fingerprint_field_type::bytes,    // 17 batch_thread_pinning_v1
    server_cache_fingerprint_field_type::enum_u16, // 18 numa_policy
    server_cache_fingerprint_field_type::bool_u8,  // 19 flash_attention
    server_cache_fingerprint_field_type::enum_u16, // 20 pipeline_mode
    server_cache_fingerprint_field_type::enum_u16, // 21 target_draft_spec_composition
    server_cache_fingerprint_field_type::bytes,    // 22 speculative_config_v1
    server_cache_fingerprint_field_type::enum_u16, // 23 kv_k_type
    server_cache_fingerprint_field_type::enum_u16, // 24 kv_v_type
    server_cache_fingerprint_field_type::bool_u8,  // 25 kv_unified
    server_cache_fingerprint_field_type::u32,      // 26 n_ctx
    server_cache_fingerprint_field_type::u32,      // 27 n_parallel
    server_cache_fingerprint_field_type::bytes,    // 28 vbr_resolved_regime_v1
    server_cache_fingerprint_field_type::digest32, // 29 override census
    server_cache_fingerprint_field_type::u32,      // 30 provider mask
    server_cache_fingerprint_field_type::bytes,    // 31 restore policy
    server_cache_fingerprint_field_type::enum_u16, // 32 allocator/VMM regime
};

uint16_t read_u16(const std::vector<uint8_t> & value, size_t offset) {
    return uint16_t(value[offset]) |
           (uint16_t(value[offset + 1]) << 8);
}

uint32_t read_u32(const std::vector<uint8_t> & value, size_t offset) {
    uint32_t out = 0;
    for (unsigned i = 0; i < 4; ++i) {
        out |= uint32_t(value[offset + i]) << (8 * i);
    }
    return out;
}

uint64_t read_u64(const std::vector<uint8_t> & value, size_t offset) {
    uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) {
        out |= uint64_t(value[offset + i]) << (8 * i);
    }
    return out;
}

bool canonical_binary64_at(
        const std::vector<uint8_t> & value, size_t offset) {
    const uint64_t bits = read_u64(value, offset);
    double number = 0;
    std::memcpy(&number, &bits, sizeof(number));
    return std::isfinite(number) && bits != UINT64_C(0x8000000000000000);
}

bool canonical_binary32_at(
        const std::vector<uint8_t> & value, size_t offset) {
    const uint32_t bits = read_u32(value, offset);
    float number = 0;
    std::memcpy(&number, &bits, sizeof(number));
    return std::isfinite(number) && bits != UINT32_C(0x80000000);
}

bool valid_enum_field(const server_cache_fingerprint_field & field) {
    const uint16_t value = read_u16(field.payload, 0);
    switch (field.id) {
        case 18: return value < GGML_NUMA_STRATEGY_COUNT;
        case 20: return value <= 1;
        case 21: return value <= 3;
        case 23:
        case 24: return value < GGML_TYPE_COUNT;
        case 32: return value <= 1;
        default: return true;
    }
}

bool valid_text_at(const std::vector<uint8_t> & bytes, size_t & offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        return false;
    }
    const uint32_t size = read_u32(bytes, offset);
    offset += 4;
    if (size > bytes.size() - offset ||
        !valid_utf8(bytes.data() + offset, size)) {
        return false;
    }
    offset += size;
    return true;
}

bool valid_structured_bytes(const server_cache_fingerprint_field & field) {
    const auto & value = field.payload;
    if (value.size() < 4) {
        return false;
    }
    const uint32_t count = read_u32(value, 0);
    switch (field.id) {
        case 9: {
            if (count > (MAX_CODEC_BYTES - 4) / 26 ||
                value.size() != 4 + size_t(count) * 26) {
                return false;
            }
            for (uint32_t i = 0; i < count; ++i) {
                const size_t offset = 4 + size_t(i) * 26;
                if (read_u32(value, offset) != i ||
                    read_u16(value, offset + 20) != 0) {
                    return false;
                }
            }
            return true;
        }
        case 10: {
            if (count > (MAX_CODEC_BYTES - 4) / 19 ||
                value.size() != 4 + size_t(count) * 19) {
                return false;
            }
            std::pair<uint32_t, uint32_t> previous = {};
            for (uint32_t i = 0; i < count; ++i) {
                const size_t offset = 4 + size_t(i) * 19;
                const std::pair<uint32_t, uint32_t> current = {
                    read_u32(value, offset), read_u32(value, offset + 4) };
                if (read_u16(value, offset + 8) != 0 ||
                    value[offset + 10] > 1 ||
                    (i != 0 && current <= previous)) {
                    return false;
                }
                previous = current;
            }
            return true;
        }
        case 11: {
            if (value.size() < 16 || read_u16(value, 0) > 2 ||
                value[value.size() - 2] > 1 ||
                value[value.size() - 1] > 1) {
                return false;
            }
            const uint32_t splits = read_u32(value, 10);
            if (splits > (MAX_CODEC_BYTES - 16) / 8 ||
                value.size() != 16 + size_t(splits) * 8) {
                return false;
            }
            for (uint32_t i = 0; i < splits; ++i) {
                if (!canonical_binary64_at(value, 14 + size_t(i) * 8)) {
                    return false;
                }
            }
            return true;
        }
        case 16:
        case 17: {
            if (count > (MAX_CODEC_BYTES - 4) / 4 ||
                value.size() != 4 + size_t(count) * 4) {
                return false;
            }
            uint32_t previous = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t cpu = read_u32(value, 4 + size_t(i) * 4);
                if (cpu >= GGML_MAX_N_THREADS || (i != 0 && cpu <= previous)) {
                    return false;
                }
                previous = cpu;
            }
            return true;
        }
        case 22:
            return value.size() == 64 &&
                   read_u16(value, 0) < COMMON_SPECULATIVE_TYPE_COUNT &&
                   canonical_binary64_at(value, 14) &&
                   canonical_binary64_at(value, 22) &&
                   value[30] <= 1 && value[31] <= 1;
        case 28: {
            if (value.size() < 3 || value[0] > 1 || value[1] > 1 ||
                value[2] > 1) {
                return false;
            }
            size_t offset = 3;
            if (!valid_text_at(value, offset) ||
                !valid_text_at(value, offset) ||
                !valid_text_at(value, offset) ||
                value.size() - offset != 65) {
                return false;
            }
            offset += 32;
            for (int field_index = 0; field_index < 2; ++field_index) {
                if (!canonical_binary64_at(value, offset)) {
                    return false;
                }
                offset += 8;
            }
            offset += 8; // vram_budget_bytes
            for (int field_index = 0; field_index < 2; ++field_index) {
                if (!canonical_binary32_at(value, offset)) {
                    return false;
                }
                offset += 4;
            }
            return value[offset] <= 1 && offset + 1 == value.size();
        }
        case 31: {
            if (count > (MAX_CODEC_BYTES - 4) / 36 ||
                value.size() != 4 + size_t(count) * 36) {
                return false;
            }
            uint16_t previous = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const size_t offset = 4 + size_t(i) * 36;
                const uint16_t provider = read_u16(value, offset);
                if ((provider != uint16_t(common_cache_plan_provider::host_cache_entry) &&
                     provider != uint16_t(common_cache_plan_provider::live_context_checkpoint)) ||
                    read_u16(value, offset + 2) != 0 ||
                    (i != 0 && provider <= previous)) {
                    return false;
                }
                previous = provider;
            }
            return true;
        }
        default:
            return true;
    }
}

bool valid_field(const server_cache_fingerprint_field & field) noexcept {
    if (field.id == 0 || field.id > FIELD_TYPES.size() ||
        field.type != FIELD_TYPES[field.id - 1] ||
        field.payload.size() > MAX_CODEC_BYTES) {
        return false;
    }
    switch (field.type) {
        case server_cache_fingerprint_field_type::bool_u8:
            return field.payload.size() == 1 && field.payload[0] <= 1;
        case server_cache_fingerprint_field_type::enum_u16:
            return field.payload.size() == 2 && valid_enum_field(field);
        case server_cache_fingerprint_field_type::u32:
            return field.payload.size() == 4;
        case server_cache_fingerprint_field_type::u64:
            return field.payload.size() == 8;
        case server_cache_fingerprint_field_type::binary64:
            return field.payload.size() == 8 &&
                   canonical_binary64_at(field.payload, 0);
        case server_cache_fingerprint_field_type::utf8:
            return valid_utf8(field.payload.data(), field.payload.size());
        case server_cache_fingerprint_field_type::bytes:
            return valid_structured_bytes(field);
        case server_cache_fingerprint_field_type::digest32:
            return field.payload.size() == 32;
    }
    return false;
}

std::array<uint8_t, 32> digest(const std::vector<uint8_t> & bytes) {
    llama_sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finish();
}

void hash_u16(llama_sha256 & hash, uint16_t value) {
    const uint8_t bytes[2] = { uint8_t(value), uint8_t(value >> 8) };
    hash.update(bytes, sizeof(bytes));
}

void hash_u32(llama_sha256 & hash, uint32_t value) {
    uint8_t bytes[4];
    llama_store_le_u32(bytes, value);
    hash.update(bytes, sizeof(bytes));
}

void hash_u64(llama_sha256 & hash, uint64_t value) {
    uint8_t bytes[8];
    llama_store_le_u64(bytes, value);
    hash.update(bytes, sizeof(bytes));
}

bool fingerprint_config_root_from_fields(
        const std::vector<server_cache_fingerprint_field> & fields,
        std::array<uint8_t, 32> & root, bool & exact) noexcept {
    try {
        if (fields.size() != FIELD_TYPES.size()) return false;
        llama_sha256 hash;
        hash.update(CONFIG_DOMAIN, sizeof(CONFIG_DOMAIN));
        hash_u32(hash, uint32_t(fields.size()));
        exact = true;
        for (size_t i = 0; i < fields.size(); ++i) {
            const auto & field = fields[i];
            if (field.id != i + 1 || !valid_field(field) ||
                field.payload.size() > UINT32_MAX) return false;
            hash_u16(hash, field.id);
            const uint8_t type = uint8_t(field.type);
            hash.update(&type, sizeof(type));
            hash_u32(hash, uint32_t(field.payload.size()));
            hash.update(field.payload.data(), field.payload.size());
            exact = exact && field.exact;
        }
        root = hash.finish();
        return true;
    } catch (...) {
        root = {};
        exact = false;
        return false;
    }
}

void fingerprint_artifact_hash_update(
        llama_sha256 & hash,
        const server_cache_fingerprint_artifact & artifact) {
    hash_u16(hash, uint16_t(artifact.role));
    hash_u32(hash, artifact.ordinal);
    hash_u64(hash, artifact.byte_length);
    hash.update(artifact.structure_sha256.data(), artifact.structure_sha256.size());
}

std::array<uint8_t, 32> digest_text(const std::string & value) {
    llama_sha256 hash;
    hash.update(value.data(), value.size());
    return hash.finish();
}

bool append_text32(std::vector<uint8_t> & out, const std::string & value) {
    if (value.size() > UINT32_MAX || out.size() > MAX_CODEC_BYTES - 4 ||
        value.size() > MAX_CODEC_BYTES - out.size() - 4) {
        return false;
    }
    append_u32(out, uint32_t(value.size()));
    return append_bounded(out, value.data(), value.size());
}

std::vector<uint8_t> cpu_mask_payload(const common_cpu_params & cpu) {
    std::vector<uint8_t> out;
    uint32_t count = 0;
    if (cpu.mask_valid) {
        for (bool selected : cpu.cpumask) {
            count += selected ? 1u : 0u;
        }
    }
    append_u32(out, count);
    if (cpu.mask_valid) {
        for (uint32_t i = 0; i < GGML_MAX_N_THREADS; ++i) {
            if (cpu.cpumask[i]) {
                append_u32(out, i);
            }
        }
    }
    return out;
}

bool decode_hex_digest(const std::string & value,
                       std::array<uint8_t, 32> & out) {
    if (value.size() != 64) {
        return false;
    }
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int high = nibble(value[2 * i]);
        const int low = nibble(value[2 * i + 1]);
        if (high < 0 || low < 0) {
            return false;
        }
        out[i] = uint8_t((high << 4) | low);
    }
    return true;
}

bool vbr_override_census_payload(
        const std::vector<common_cache_plan_vbr_override_row> & rows,
        std::vector<uint8_t> & out,
        std::array<uint8_t, 32> & schedule_digest) {
    out.assign(4, 0);
    schedule_digest = digest_text("");
    if (rows.size() > UINT32_MAX) {
        return false;
    }
    uint16_t previous_id = 0;
    for (const auto & row : rows) {
        if (row.census_id == 0 || row.census_id <= previous_id ||
            row.value.empty() || row.value.size() > UINT32_MAX ||
            row.grammar > common_cache_plan_vbr_value_grammar::inline_or_path) {
            return false;
        }
        previous_id = row.census_id;
        append_u16(out, row.census_id);
        out.push_back(uint8_t(row.grammar));
        append_u32(out, uint32_t(row.value.size()));
        if (!append_bounded(out, row.value.data(), row.value.size())) {
            return false;
        }
        if (row.grammar ==
                common_cache_plan_vbr_value_grammar::inline_or_path) {
            static constexpr char SHA_PREFIX[] = "sha256-";
            if (row.value.compare(
                    0, sizeof(SHA_PREFIX) - 1, SHA_PREFIX) != 0 ||
                !decode_hex_digest(
                    row.value.substr(sizeof(SHA_PREFIX) - 1),
                                   schedule_digest)) {
                return false;
            }
        }
    }
    const uint32_t count = uint32_t(rows.size());
    out[0] = uint8_t(count);
    out[1] = uint8_t(count >> 8);
    out[2] = uint8_t(count >> 16);
    out[3] = uint8_t(count >> 24);
    return true;
}

struct bounded_bytes {
    uint8_t * data = nullptr;
    size_t capacity = 0;
    size_t size = 0;

    void clear() noexcept { size = 0; }
    bool append(const void * source, size_t count) noexcept {
        if (count > capacity - size || (count != 0 && !source)) return false;
        if (count != 0) std::memcpy(data + size, source, count);
        size += count;
        return true;
    }
    bool u8(uint8_t value) noexcept { return append(&value, 1); }
    bool u16(uint16_t value) noexcept {
        const uint8_t bytes[2] = { uint8_t(value), uint8_t(value >> 8) };
        return append(bytes, sizeof(bytes));
    }
    bool u32(uint32_t value) noexcept {
        uint8_t bytes[4];
        llama_store_le_u32(bytes, value);
        return append(bytes, sizeof(bytes));
    }
    bool u64(uint64_t value) noexcept {
        uint8_t bytes[8];
        llama_store_le_u64(bytes, value);
        return append(bytes, sizeof(bytes));
    }
    bool binary64(double value) noexcept {
        if (!std::isfinite(value)) return false;
        if (value == 0.0) value = 0.0;
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return u64(bits);
    }
    bool binary32(float value) noexcept {
        if (!std::isfinite(value)) return false;
        if (value == 0.0f) value = 0.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return u32(bits);
    }
    bool text32(const std::string & value) noexcept {
        return value.size() <= UINT32_MAX && u32(uint32_t(value.size())) &&
               append(value.data(), value.size());
    }
};

struct config_root_builder {
    llama_sha256 hash;
    uint16_t next_id = 1;
    bool exact = true;
    uint32_t inexact_fields = 0;

    config_root_builder() {
        hash.update(CONFIG_DOMAIN, sizeof(CONFIG_DOMAIN));
        hash_u32(hash, uint32_t(FIELD_TYPES.size()));
    }

    bool field(uint16_t id, server_cache_fingerprint_field_type type,
               const void * data, size_t size, bool field_exact = true) noexcept {
        if (id != next_id || id > FIELD_TYPES.size() ||
            type != FIELD_TYPES[id - 1] || size > UINT32_MAX ||
            (size != 0 && !data)) return false;
        hash_u16(hash, id);
        const uint8_t type_byte = uint8_t(type);
        hash.update(&type_byte, 1);
        hash_u32(hash, uint32_t(size));
        hash.update(data, size);
        exact = exact && field_exact;
        if (!field_exact) inexact_fields |= uint32_t(1) << (id - 1);
        ++next_id;
        return true;
    }
    bool u32(uint16_t id, uint32_t value, bool field_exact = true) noexcept {
        uint8_t bytes[4];
        llama_store_le_u32(bytes, value);
        return field(id, server_cache_fingerprint_field_type::u32,
                     bytes, sizeof(bytes), field_exact);
    }
    bool enumeration(uint16_t id, uint16_t value) noexcept {
        const uint8_t bytes[2] = { uint8_t(value), uint8_t(value >> 8) };
        return field(id, server_cache_fingerprint_field_type::enum_u16,
                     bytes, sizeof(bytes));
    }
    bool boolean(uint16_t id, bool value) noexcept {
        const uint8_t byte = uint8_t(value);
        return field(id, server_cache_fingerprint_field_type::bool_u8,
                     &byte, sizeof(byte));
    }
};

std::array<uint8_t, 32> digest_parts(
        std::initializer_list<std::pair<const char *, size_t>> parts) {
    llama_sha256 hash;
    for (const auto & part : parts) hash.update(part.first, part.second);
    return hash.finish();
}

struct canonical_device_identity {
    ggml_backend_dev_t device = nullptr;
    ggml_backend_device_identity_v1 identity = {};
    uint32_t input_ordinal = 0;
};

bool cpu_backend_identity_v1(char * out, size_t capacity,
                             size_t & size) noexcept {
    size = 0;
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__GNUC__) || defined(__clang__))
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) ||
        __get_cpuid_max(0x80000000u, nullptr) < 0x80000004u) {
        return false;
    }
    char brand[49] = {};
    for (unsigned leaf = 0; leaf != 3; ++leaf) {
        unsigned words[4] = {};
        __cpuid(0x80000002u + leaf,
                words[0], words[1], words[2], words[3]);
        std::memcpy(brand + 16 * leaf, words, 16);
    }
    const char * features = llama_print_system_info();
    const int written = std::snprintf(
        out, capacity, "cpuid-v1|signature=%08x|brand=%s|features=%s",
        eax, brand, features ? features : "");
    if (written < 0 || size_t(written) >= capacity) return false;
    size = size_t(written);
    return true;
#else
    GGML_UNUSED(out);
    GGML_UNUSED(capacity);
    return false;
#endif
}

bool resolve_canonical_devices(
        const common_params & params,
        std::array<canonical_device_identity, 128> & devices,
        size_t & count,
        uint32_t & driver_version,
        uint32_t & runtime_version,
        bool & exact) noexcept {
    count = 0;
    driver_version = 0;
    runtime_version = 0;
    exact = true;
    auto admit = [&](ggml_backend_dev_t device) {
        if (!device || count == devices.size()) {
            exact = false;
            return;
        }
        auto & row = devices[count];
        row = {};
        row.device = device;
        row.input_ordinal = uint32_t(count);
        row.identity.struct_size = sizeof(row.identity);
        auto * reg = ggml_backend_dev_backend_reg(device);
        const auto query = reinterpret_cast<ggml_backend_device_identity_v1_t>(
            ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_DEVICE_IDENTITY_V1_PROC));
        bool nonzero_uuid = false;
        if (!query || !query(device, &row.identity) ||
            row.identity.struct_size != sizeof(row.identity) ||
            row.identity.driver_version == 0 ||
            row.identity.runtime_version == 0 ||
            row.identity.backend_kind == GGML_BACKEND_IDENTITY_KIND_UNKNOWN ||
            row.identity.arch_major == 0) {
            nonzero_uuid = false;
        } else {
            for (const uint8_t byte : row.identity.uuid) {
                nonzero_uuid |= byte != 0;
            }
        }
        if (!nonzero_uuid) {
            const char * name = ggml_backend_dev_name(device);
            const char * description = ggml_backend_dev_description(device);
            const char newline[] = "\n";
            const auto pseudo_uuid = digest_parts({
                { name, std::strlen(name) }, { newline, 1 },
                { description, std::strlen(description) },
            });
            row.identity = {};
            row.identity.struct_size = sizeof(row.identity);
            std::memcpy(row.identity.uuid, pseudo_uuid.data(), 16);
            exact = false;
        }
        ++count;
    };
    if (params.devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                admit(device);
            }
        }
    } else {
        for (auto * device : params.devices) {
            if (device) admit(device);
        }
    }
    if (count == 0) {
        exact = true;
        return true;
    }
    if (exact) {
        driver_version = devices[0].identity.driver_version;
        runtime_version = devices[0].identity.runtime_version;
        for (size_t i = 1; i < count; ++i) {
            exact = exact &&
                devices[i].identity.driver_version == driver_version &&
                devices[i].identity.runtime_version == runtime_version;
        }
    }
    std::sort(devices.begin(), devices.begin() + count,
        [](const auto & a, const auto & b) {
            const int uuid_order = std::memcmp(
                a.identity.uuid, b.identity.uuid, sizeof(a.identity.uuid));
            return uuid_order != 0
                ? uuid_order < 0
                : a.identity.pci_domain_bus_device_function <
                      b.identity.pci_domain_bus_device_function;
        });
    for (size_t i = 1; i < count; ++i) {
        if (std::memcmp(devices[i - 1].identity.uuid,
                        devices[i].identity.uuid, 16) == 0) {
            exact = false;
        }
    }
    if (!exact) {
        driver_version = 0;
        runtime_version = 0;
    }
    return true;
}

bool has_active_tensor_buft_override(const common_params & params) noexcept {
    for (const auto & row : params.tensor_buft_overrides) {
        if (row.pattern != nullptr || row.buft != nullptr) return true;
    }
    return false;
}

bool canonical_main_device(
        const std::array<canonical_device_identity, 128> & devices,
        size_t count,
        int32_t input_main,
        uint32_t & canonical) noexcept {
    if (input_main < 0) return false;
    for (uint32_t i = 0; i < count; ++i) {
        if (devices[i].input_ordinal == uint32_t(input_main)) {
            canonical = i;
            return true;
        }
    }
    return false;
}

bool decode_hex_digest(const char * value, size_t size,
                       std::array<uint8_t, 32> & out) noexcept {
    if (!value || size != 64) return false;
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        const int high = nibble(value[2 * i]);
        const int low = nibble(value[2 * i + 1]);
        if (high < 0 || low < 0) return false;
        out[i] = uint8_t((high << 4) | low);
    }
    return true;
}

bool fingerprint_config_root_from_params(
        const common_params & params,
        const float * effective_tensor_split,
        size_t effective_tensor_split_count,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime,
        uint8_t * scratch, size_t scratch_size,
        std::array<uint8_t, 32> & root, bool & exact,
        uint32_t & inexact_fields) noexcept {
    try {
        bounded_bytes bytes { scratch, scratch_size, 0 };
        config_root_builder out;
        if (!out.u32(1, 2) || !out.u32(2, 2) || !out.u32(3, 2)) return false;

        static constexpr char ABI_BACKEND[] = "|ggml-backend-abi=v2";
        const char * build = llama_build_info();
        const char * compiler = llama_compiler();
        const char * target = llama_build_target();
        const char separator[] = "|";
        const auto cost_abi = digest_parts({
            { SERVER_CACHE_CALIBRATION_ESTIMATOR_ABI_PREFIX,
              sizeof(SERVER_CACHE_CALIBRATION_ESTIMATOR_ABI_PREFIX) - 1 },
            { build, std::strlen(build) },
            { separator, 1 }, { compiler, std::strlen(compiler) },
            { separator, 1 }, { target, std::strlen(target) },
            { ABI_BACKEND, sizeof(ABI_BACKEND) - 1 },
        });
        if (!out.field(4, server_cache_fingerprint_field_type::digest32,
                       cost_abi.data(), cost_abi.size())) return false;

        bytes.clear();
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));
            if ((i != 0 && !bytes.u8('|')) ||
                !bytes.append(name, std::strlen(name))) return false;
        }
        if (!out.field(5, server_cache_fingerprint_field_type::utf8,
                       bytes.data, bytes.size)) return false;

        std::array<canonical_device_identity, 128> devices = {};
        size_t device_count = 0;
        uint32_t driver_version = 0;
        uint32_t runtime_version = 0;
        bool hardware_exact = false;
        if (!resolve_canonical_devices(
                params, devices, device_count, driver_version,
                runtime_version, hardware_exact)) return false;
        // The CLI pads this vector with null rows for the auto-fit workspace.
        // Capacity is not a user override and must not demote every ordinary
        // server launch to a shadow-only hardware identity.
        if (has_active_tensor_buft_override(params)) hardware_exact = false;
        if (device_count == 0) effective_n_gpu_layers = 0;
        if (!out.u32(6, driver_version, hardware_exact) ||
            !out.u32(7, runtime_version, hardware_exact)) return false;

        char cpu_identity[1024] = {};
        size_t cpu_identity_size = 0;
        const bool cpu_exact = cpu_backend_identity_v1(
            cpu_identity, sizeof(cpu_identity), cpu_identity_size);
        const char * cpu_value = cpu_exact
            ? cpu_identity : llama_print_system_info();
        const size_t cpu_value_size = cpu_exact
            ? cpu_identity_size : std::strlen(cpu_value);
        if (!out.field(8, server_cache_fingerprint_field_type::utf8,
                       cpu_value, cpu_value_size, cpu_exact)) return false;

        bytes.clear();
        if (device_count > UINT32_MAX || !bytes.u32(uint32_t(device_count))) return false;
        for (uint32_t i = 0; i < device_count; ++i) {
            const auto & identity = devices[i].identity;
            if (!bytes.u32(i) || !bytes.append(identity.uuid, 16) ||
                !bytes.u16(identity.backend_kind) ||
                !bytes.u16(identity.arch_major) ||
                !bytes.u16(identity.arch_minor)) return false;
        }
        if (!out.field(9, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size, hardware_exact)) return false;

        bytes.clear();
        if (device_count > UINT32_MAX ||
            (device_count != 0 && device_count > UINT32_MAX / device_count) ||
            !bytes.u32(uint32_t(device_count * device_count))) return false;
        for (uint32_t src = 0; src < device_count; ++src) {
            for (uint32_t dst = 0; dst < device_count; ++dst) {
                uint16_t link_class = 1;
                uint8_t p2p = 1;
                uint64_t bandwidth_class =
                    devices[src].identity.pci_domain_bus_device_function;
                if (src != dst) {
                    ggml_backend_device_link_v1 link = {};
                    link.struct_size = sizeof(link);
                    auto * reg = ggml_backend_dev_backend_reg(devices[src].device);
                    const auto query = reinterpret_cast<
                        ggml_backend_device_link_v1_t>(
                            ggml_backend_reg_get_proc_address(
                                reg, GGML_BACKEND_DEVICE_LINK_V1_PROC));
                    if (!query || !query(
                            devices[src].device, devices[dst].device, &link) ||
                        link.struct_size != sizeof(link) ||
                        link.link_class == 0 || link.p2p > 1) {
                        hardware_exact = false;
                        link = {};
                    }
                    link_class = link.link_class;
                    p2p = link.p2p;
                    bandwidth_class = link.bandwidth_class;
                }
                if (!bytes.u32(src) || !bytes.u32(dst) ||
                    !bytes.u16(link_class) || !bytes.u8(p2p) ||
                    !bytes.u64(bandwidth_class)) return false;
            }
        }
        if (!out.field(10, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size, hardware_exact)) return false;

        bytes.clear();
        uint32_t canonical_main = 0;
        if (device_count != 0 && !canonical_main_device(
                devices, device_count, params.main_gpu, canonical_main)) {
            hardware_exact = false;
        }
        if (!bytes.u16(uint16_t(params.split_mode)) ||
            !bytes.u32(canonical_main) ||
            !bytes.u32(effective_n_gpu_layers) ||
            !bytes.u32(uint32_t(device_count))) return false;
        for (size_t i = 0; i < device_count; ++i) {
            if (!effective_tensor_split ||
                devices[i].input_ordinal >= effective_tensor_split_count ||
                !bytes.binary64(
                    effective_tensor_split[devices[i].input_ordinal])) {
                return false;
            }
        }
        if (!bytes.u8(uint8_t(!params.no_kv_offload)) ||
            !bytes.u8(uint8_t(!params.no_op_offload)) ||
            !out.field(11, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size, hardware_exact)) return false;
        out.exact = out.exact && hardware_exact;
        if (!out.u32(12, uint32_t(std::max(0, params.n_batch))) ||
            !out.u32(13, uint32_t(std::max(0, params.n_ubatch))) ||
            !out.u32(14, uint32_t(std::max(0, params.cpuparams.n_threads))) ||
            !out.u32(15, uint32_t(std::max(0, params.cpuparams_batch.n_threads)))) return false;
        auto emit_cpu_mask = [&](uint16_t id, const common_cpu_params & cpu) {
            bytes.clear();
            uint32_t count = 0;
            if (cpu.mask_valid) {
                for (bool selected : cpu.cpumask) count += selected ? 1u : 0u;
            }
            if (!bytes.u32(count)) return false;
            if (cpu.mask_valid) {
                for (uint32_t i = 0; i < GGML_MAX_N_THREADS; ++i) {
                    if (cpu.cpumask[i] && !bytes.u32(i)) return false;
                }
            }
            return out.field(id, server_cache_fingerprint_field_type::bytes,
                             bytes.data, bytes.size);
        };
        if (!emit_cpu_mask(16, params.cpuparams) ||
            !emit_cpu_mask(17, params.cpuparams_batch) ||
            !out.enumeration(18, uint16_t(params.numa)) ||
            !out.boolean(19, params.flash_attn_type !=
                              LLAMA_FLASH_ATTN_TYPE_DISABLED) ||
            pipeline_mode > 1 || allocator_vmm_regime > 1 ||
            !out.enumeration(20, pipeline_mode)) return false;
        const bool has_draft = params.speculative.has_dft();
        const bool has_model_free = params.speculative.has_model_free_type();
        if (!out.enumeration(21, uint16_t(has_draft) |
                                 (uint16_t(has_model_free) << 1))) return false;

        bytes.clear();
        char policy[128];
        const int policy_size = std::snprintf(
            policy, sizeof(policy), "%d|%d|%f",
            params.speculative.tree_budget,
            params.speculative.draft_topk,
            double(params.speculative.sample_temp));
        if (policy_size < 0 || size_t(policy_size) >= sizeof(policy)) return false;
        const auto dflash_policy = digest_parts({ { policy, size_t(policy_size) } });
        if (!bytes.u16(uint16_t(params.speculative.type())) ||
            !bytes.u32(uint32_t(std::max(0, params.speculative.n_max))) ||
            !bytes.u32(uint32_t(std::max(0, params.speculative.n_min))) ||
            !bytes.u32(uint32_t(std::max(0, params.speculative.draft.n_max))) ||
            !bytes.binary64(params.speculative.p_min) ||
            !bytes.binary64(params.speculative.p_split) ||
            !bytes.u8(uint8_t(has_model_free)) ||
            !bytes.u8(uint8_t(params.speculative.type() ==
                               COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) ||
            !bytes.append(dflash_policy.data(), dflash_policy.size())) return false;
        const bool speculative_exact = params.speculative.types.empty() ||
            (params.speculative.types.size() == 1 &&
             params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE);
        if (!out.field(22, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size, speculative_exact) ||
            !out.enumeration(23, uint16_t(params.cache_type_k)) ||
            !out.enumeration(24, uint16_t(params.cache_type_v)) ||
            !out.boolean(25, params.kv_unified) ||
            !out.u32(26, uint32_t(std::max(0, params.n_ctx))) ||
            !out.u32(27, uint32_t(std::max(0, params.n_parallel)))) return false;

        std::array<uint8_t, 32> schedule_digest = digest_parts({ { "", 0 } });
        bytes.clear();
        if (vbr.override_rows.size() > UINT32_MAX ||
            !bytes.u32(uint32_t(vbr.override_rows.size()))) return false;
        uint16_t previous_id = 0;
        for (const auto & row : vbr.override_rows) {
            if (row.census_id == 0 || row.census_id <= previous_id ||
                row.value.empty() || row.value.size() > UINT32_MAX ||
                row.grammar > common_cache_plan_vbr_value_grammar::inline_or_path ||
                !bytes.u16(row.census_id) || !bytes.u8(uint8_t(row.grammar)) ||
                !bytes.u32(uint32_t(row.value.size())) ||
                !bytes.append(row.value.data(), row.value.size())) return false;
            previous_id = row.census_id;
            if (row.grammar == common_cache_plan_vbr_value_grammar::inline_or_path) {
                static constexpr char SHA_PREFIX[] = "sha256-";
                if (row.value.size() != sizeof(SHA_PREFIX) - 1 + 64 ||
                    row.value.compare(0, sizeof(SHA_PREFIX) - 1, SHA_PREFIX) != 0 ||
                    !decode_hex_digest(row.value.data() + sizeof(SHA_PREFIX) - 1,
                                       64, schedule_digest)) return false;
            }
        }
        llama_sha256 override_hash;
        override_hash.update(bytes.data, bytes.size);
        const auto override_digest = override_hash.finish();

        bytes.clear();
        if (!bytes.u8(uint8_t(vbr.armed)) || !bytes.u8(uint8_t(vbr.side_k)) ||
            !bytes.u8(uint8_t(vbr.side_v)) || !bytes.text32(vbr.budget_mode) ||
            !bytes.text32(vbr.family) || !bytes.text32(vbr.policy) ||
            !bytes.append(schedule_digest.data(), schedule_digest.size()) ||
            !bytes.binary64(vbr.capacity_bits) ||
            !bytes.binary64(vbr.selected_bpv) ||
            !bytes.u64(vbr.vram_budget_bytes) ||
            !bytes.binary32(vbr.reclaim_floor_bpv) ||
            !bytes.binary32(vbr.reset_keep_frac) ||
            !bytes.u8(uint8_t(vbr.unrepresented_override)) ||
            !out.field(28, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size, !vbr.unrepresented_override) ||
            !out.field(29, server_cache_fingerprint_field_type::digest32,
                       override_digest.data(), override_digest.size())) return false;

        uint32_t providers = 0;
        providers |= 1u << uint16_t(common_cache_plan_provider::live_slot);
        providers |= 1u << uint16_t(common_cache_plan_provider::cold_replay);
        if (params.cache_ram_mib != 0) providers |=
            1u << uint16_t(common_cache_plan_provider::host_cache_entry);
        if (params.n_ctx_checkpoints > 0) providers |=
            1u << uint16_t(common_cache_plan_provider::live_context_checkpoint);
        if (!out.u32(30, providers)) return false;

        bytes.clear();
        if (!bytes.u32(2)) return false;
        for (const auto provider : {
                 common_cache_plan_provider::host_cache_entry,
                 common_cache_plan_provider::live_context_checkpoint }) {
            const char * name = common_cache_plan_provider_name(provider);
            static constexpr char RECIPE_PREFIX[] = "restore-representation-v1|";
            const auto recipe = digest_parts({
                { RECIPE_PREFIX, sizeof(RECIPE_PREFIX) - 1 },
                { name, std::strlen(name) },
            });
            if (!bytes.u16(uint16_t(provider)) || !bytes.u16(0) ||
                !bytes.append(recipe.data(), recipe.size())) return false;
        }
        if (!out.field(31, server_cache_fingerprint_field_type::bytes,
                       bytes.data, bytes.size) ||
            !out.enumeration(32, allocator_vmm_regime) ||
            out.next_id != FIELD_TYPES.size() + 1) return false;
        root = out.hash.finish();
        exact = out.exact;
        inexact_fields = out.inexact_fields;
        return true;
    } catch (...) {
        root = {};
        exact = false;
        inexact_fields = 0;
        return false;
    }
}

} // namespace

static_assert(sizeof(server_cache_fingerprint_worker) <= FINGERPRINT_ARENA_BYTES,
              "fingerprint worker must fit its fixed 1-MiB arena region");

server_cache_fingerprint_field server_cache_fingerprint_bool(
        uint16_t id, bool value) {
    return { id, server_cache_fingerprint_field_type::bool_u8,
             { uint8_t(value ? 1 : 0) } };
}

server_cache_fingerprint_field server_cache_fingerprint_enum(
        uint16_t id, uint16_t value) {
    return { id, server_cache_fingerprint_field_type::enum_u16,
             little_u16(value) };
}

server_cache_fingerprint_field server_cache_fingerprint_u32(
        uint16_t id, uint32_t value) {
    return { id, server_cache_fingerprint_field_type::u32,
             little_u32(value) };
}

server_cache_fingerprint_field server_cache_fingerprint_u64(
        uint16_t id, uint64_t value) {
    return { id, server_cache_fingerprint_field_type::u64,
             little_u64(value) };
}

bool server_cache_fingerprint_binary64(
        uint16_t id, double value,
        server_cache_fingerprint_field & out) noexcept {
    if (!std::isfinite(value)) {
        return false;
    }
    if (value == 0.0) {
        value = 0.0;
    }
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "binary64 width");
    std::memcpy(&bits, &value, sizeof(bits));
    try {
        out = { id, server_cache_fingerprint_field_type::binary64,
                little_u64(bits) };
        return true;
    } catch (...) {
        return false;
    }
}

bool server_cache_fingerprint_utf8(
        uint16_t id, const char * data, size_t size,
        server_cache_fingerprint_field & out) noexcept {
    if ((size != 0 && data == nullptr) || size > UINT32_MAX ||
        !valid_utf8(reinterpret_cast<const uint8_t *>(data), size)) {
        return false;
    }
    try {
        out.id = id;
        out.type = server_cache_fingerprint_field_type::utf8;
        out.payload.clear();
        if (size != 0) {
            out.payload.assign(data, data + size);
        }
        return true;
    } catch (...) {
        return false;
    }
}

server_cache_fingerprint_field server_cache_fingerprint_bytes(
        uint16_t id, const void * data, size_t size) {
    server_cache_fingerprint_field out;
    out.id = id;
    out.type = server_cache_fingerprint_field_type::bytes;
    const auto * first = static_cast<const uint8_t *>(data);
    if (size != 0) {
        out.payload.assign(first, first + size);
    }
    return out;
}

server_cache_fingerprint_field server_cache_fingerprint_digest(
        uint16_t id, const std::array<uint8_t, 32> & value) {
    return { id, server_cache_fingerprint_field_type::digest32,
             std::vector<uint8_t>(value.begin(), value.end()) };
}

bool server_cache_execution_fingerprint_v1(
        std::vector<server_cache_fingerprint_artifact> artifacts,
        std::vector<server_cache_fingerprint_field> fields,
        server_cache_execution_fingerprint & out) noexcept {
    out = {};
    try {
        if (artifacts.empty() || artifacts.size() > UINT32_MAX ||
            fields.size() != FIELD_TYPES.size()) {
            return false;
        }
        bool exact = true;
        for (size_t i = 0; i < artifacts.size(); ++i) {
            if (!valid_artifact_role(artifacts[i].role) ||
                (i == 0 && (artifacts[i].role !=
                                server_cache_fingerprint_artifact_role::target ||
                            artifacts[i].ordinal != 0)) ||
                (i != 0 &&
                 std::tie(artifacts[i].role, artifacts[i].ordinal) <=
                     std::tie(artifacts[i - 1].role,
                              artifacts[i - 1].ordinal)) ||
                (i != 0 && artifacts[i].role == artifacts[i - 1].role &&
                 artifacts[i].ordinal != artifacts[i - 1].ordinal + 1) ||
                (i != 0 && artifacts[i].role != artifacts[i - 1].role &&
                 artifacts[i].ordinal != 0)) {
                return false;
            }
            exact = exact && artifacts[i].exact;
        }
        for (size_t i = 0; i < fields.size(); ++i) {
            if (fields[i].id != i + 1 || !valid_field(fields[i])) {
                return false;
            }
            exact = exact && fields[i].exact;
        }

        std::vector<uint8_t> artifact_bytes;
        if (!append_bounded(artifact_bytes, ARTIFACT_DOMAIN,
                            sizeof(ARTIFACT_DOMAIN)) ||
            artifact_bytes.size() > MAX_CODEC_BYTES - 4) {
            return false;
        }
        append_u32(artifact_bytes, uint32_t(artifacts.size()));
        for (const auto & artifact : artifacts) {
            if (artifact_bytes.size() > MAX_CODEC_BYTES - (2 + 4 + 8 + 32)) {
                return false;
            }
            append_u16(artifact_bytes, uint16_t(artifact.role));
            append_u32(artifact_bytes, artifact.ordinal);
            append_u64(artifact_bytes, artifact.byte_length);
            if (!append_bounded(artifact_bytes,
                                artifact.structure_sha256.data(), 32)) {
                return false;
            }
        }

        std::vector<uint8_t> config_bytes;
        if (!append_bounded(config_bytes, CONFIG_DOMAIN,
                            sizeof(CONFIG_DOMAIN)) ||
            config_bytes.size() > MAX_CODEC_BYTES - 4) {
            return false;
        }
        append_u32(config_bytes, uint32_t(fields.size()));
        for (const auto & field : fields) {
            if (field.payload.size() > UINT32_MAX ||
                config_bytes.size() > MAX_CODEC_BYTES - (2 + 1 + 4) ||
                field.payload.size() >
                    MAX_CODEC_BYTES - config_bytes.size() - (2 + 1 + 4)) {
                return false;
            }
            append_u16(config_bytes, field.id);
            config_bytes.push_back(uint8_t(field.type));
            append_u32(config_bytes, uint32_t(field.payload.size()));
            if (!append_bounded(config_bytes, field.payload.data(),
                                field.payload.size())) {
                return false;
            }
        }

        out.artifact_root = digest(artifact_bytes);
        out.config_root = digest(config_bytes);
        std::vector<uint8_t> execution_bytes;
        if (!append_bounded(execution_bytes, EXEC_DOMAIN,
                            sizeof(EXEC_DOMAIN)) ||
            !append_bounded(execution_bytes, out.artifact_root.data(), 32) ||
            !append_bounded(execution_bytes, out.config_root.data(), 32)) {
            return false;
        }
        out.execution_root = digest(execution_bytes);
        out.complete = true;
        out.exact = exact;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_cache_fingerprint_fields_v1(
        const common_params & params,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime,
        std::vector<server_cache_fingerprint_field> & out) noexcept {
    out.clear();
    struct clear_partial_output {
        std::vector<server_cache_fingerprint_field> & value;
        bool complete = false;
        ~clear_partial_output() {
            if (!complete) {
                value.clear();
            }
        }
    } output_guard { out };
    try {
        auto add_utf8 = [&](uint16_t id, const std::string & value,
                            bool exact = true) {
            server_cache_fingerprint_field field;
            if (!server_cache_fingerprint_utf8(
                    id, value.data(), value.size(), field)) {
                return false;
            }
            field.exact = exact;
            out.push_back(std::move(field));
            return true;
        };
        auto add_bytes = [&](uint16_t id, const std::vector<uint8_t> & value,
                             bool exact = true) {
            auto field = server_cache_fingerprint_bytes(
                id, value.data(), value.size());
            field.exact = exact;
            out.push_back(std::move(field));
        };
        auto append_binary64 = [](std::vector<uint8_t> & bytes, double value) {
            if (!std::isfinite(value)) {
                return false;
            }
            if (value == 0.0) {
                value = 0.0;
            }
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            append_u64(bytes, bits);
            return true;
        };
        auto append_binary32 = [](std::vector<uint8_t> & bytes, float value) {
            if (!std::isfinite(value)) {
                return false;
            }
            if (value == 0.0f) {
                value = 0.0f;
            }
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            append_u32(bytes, bits);
            return true;
        };

        out.reserve(32);
        out.push_back(server_cache_fingerprint_u32(1, 2));
        out.push_back(server_cache_fingerprint_u32(2, 2));
        out.push_back(server_cache_fingerprint_u32(3, 2));

        const std::string cost_abi =
            std::string(SERVER_CACHE_CALIBRATION_ESTIMATOR_ABI_PREFIX) +
            llama_build_info() + "|" + llama_compiler() + "|" +
            llama_build_target() + "|ggml-backend-abi=v2";
        out.push_back(server_cache_fingerprint_digest(
            4, digest_text(cost_abi)));

        std::string backends;
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            if (!backends.empty()) backends.push_back('|');
            backends += ggml_backend_reg_name(ggml_backend_reg_get(i));
        }
        if (!add_utf8(5, backends)) return false;
        std::array<canonical_device_identity, 128> effective_devices = {};
        size_t device_count = 0;
        uint32_t driver_version = 0;
        uint32_t runtime_version = 0;
        bool hardware_exact = false;
        if (!resolve_canonical_devices(
                params, effective_devices, device_count, driver_version,
                runtime_version, hardware_exact)) return false;
        if (has_active_tensor_buft_override(params)) hardware_exact = false;
        if (device_count == 0) effective_n_gpu_layers = 0;

        auto driver = server_cache_fingerprint_u32(6, driver_version);
        auto runtime = server_cache_fingerprint_u32(7, runtime_version);
        driver.exact = hardware_exact;
        runtime.exact = hardware_exact;
        out.push_back(std::move(driver));
        out.push_back(std::move(runtime));

        char cpu_identity[1024] = {};
        size_t cpu_identity_size = 0;
        const bool cpu_exact = cpu_backend_identity_v1(
            cpu_identity, sizeof(cpu_identity), cpu_identity_size);
        if (!add_utf8(8, cpu_exact
                ? std::string(cpu_identity, cpu_identity_size)
                : std::string(llama_print_system_info()), cpu_exact)) return false;

        std::vector<uint8_t> devices;
        append_u32(devices, uint32_t(device_count));
        for (uint32_t i = 0; i < device_count; ++i) {
            const auto & identity = effective_devices[i].identity;
            append_u32(devices, i);
            devices.insert(devices.end(), identity.uuid, identity.uuid + 16);
            append_u16(devices, identity.backend_kind);
            append_u16(devices, identity.arch_major);
            append_u16(devices, identity.arch_minor);
        }
        add_bytes(9, devices, hardware_exact);

        std::vector<uint8_t> topology;
        append_u32(topology, uint32_t(device_count * device_count));
        for (uint32_t src = 0; src < device_count; ++src) {
            for (uint32_t dst = 0; dst < device_count; ++dst) {
                uint16_t link_class = 1;
                uint8_t p2p = 1;
                uint64_t bandwidth_class = effective_devices[src].identity.
                    pci_domain_bus_device_function;
                if (src != dst) {
                    ggml_backend_device_link_v1 link = {};
                    link.struct_size = sizeof(link);
                    auto * reg = ggml_backend_dev_backend_reg(
                        effective_devices[src].device);
                    const auto query = reinterpret_cast<
                        ggml_backend_device_link_v1_t>(
                            ggml_backend_reg_get_proc_address(
                                reg, GGML_BACKEND_DEVICE_LINK_V1_PROC));
                    if (!query || !query(
                            effective_devices[src].device,
                            effective_devices[dst].device, &link) ||
                        link.struct_size != sizeof(link) ||
                        link.link_class == 0 || link.p2p > 1) {
                        hardware_exact = false;
                        link = {};
                    }
                    link_class = link.link_class;
                    p2p = link.p2p;
                    bandwidth_class = link.bandwidth_class;
                }
                append_u32(topology, src);
                append_u32(topology, dst);
                append_u16(topology, link_class);
                topology.push_back(p2p);
                append_u64(topology, bandwidth_class);
            }
        }
        add_bytes(10, topology, hardware_exact);

        std::vector<uint8_t> placement;
        append_u16(placement, uint16_t(params.split_mode));
        uint32_t canonical_main = 0;
        if (device_count != 0 && !canonical_main_device(
                effective_devices, device_count, params.main_gpu,
                canonical_main)) hardware_exact = false;
        append_u32(placement, canonical_main);
        append_u32(placement, effective_n_gpu_layers);
        append_u32(placement, uint32_t(device_count));
        for (size_t i = 0; i < device_count; ++i) {
            if (!append_binary64(
                    placement,
                    params.tensor_split[effective_devices[i].input_ordinal])) {
                return false;
            }
        }
        placement.push_back(uint8_t(!params.no_kv_offload));
        placement.push_back(uint8_t(!params.no_op_offload));
        add_bytes(11, placement, hardware_exact);

        out.push_back(server_cache_fingerprint_u32(
            12, uint32_t(std::max(0, params.n_batch))));
        out.push_back(server_cache_fingerprint_u32(
            13, uint32_t(std::max(0, params.n_ubatch))));
        out.push_back(server_cache_fingerprint_u32(
            14, uint32_t(std::max(0, params.cpuparams.n_threads))));
        out.push_back(server_cache_fingerprint_u32(
            15, uint32_t(std::max(0, params.cpuparams_batch.n_threads))));
        add_bytes(16, cpu_mask_payload(params.cpuparams));
        add_bytes(17, cpu_mask_payload(params.cpuparams_batch));
        out.push_back(server_cache_fingerprint_enum(
            18, uint16_t(params.numa)));
        out.push_back(server_cache_fingerprint_bool(
            19, params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED));
        if (pipeline_mode > 1 || allocator_vmm_regime > 1) {
            return false;
        }
        out.push_back(server_cache_fingerprint_enum(20, pipeline_mode));
        const bool has_draft = params.speculative.has_dft();
        const bool has_model_free =
            params.speculative.has_model_free_type();
        out.push_back(server_cache_fingerprint_enum(
            21, uint16_t(has_draft) | (uint16_t(has_model_free) << 1)));

        std::vector<uint8_t> speculative;
        append_u16(speculative, uint16_t(params.speculative.type()));
        append_u32(speculative, uint32_t(std::max(0, params.speculative.n_max)));
        append_u32(speculative, uint32_t(std::max(0, params.speculative.n_min)));
        append_u32(speculative, uint32_t(std::max(0, params.speculative.draft.n_max)));
        if (!append_binary64(speculative, params.speculative.p_min) ||
            !append_binary64(speculative, params.speculative.p_split)) {
            return false;
        }
        speculative.push_back(uint8_t(params.speculative.has_model_free_type()));
        speculative.push_back(uint8_t(params.speculative.type() ==
            COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH));
        const auto dflash_policy = digest_text(
            std::to_string(params.speculative.tree_budget) + "|" +
            std::to_string(params.speculative.draft_topk) + "|" +
            std::to_string(params.speculative.sample_temp));
        speculative.insert(speculative.end(), dflash_policy.begin(),
                           dflash_policy.end());
        // The current common params surface does not yet expose a closed
        // canonical encoding for every multi-strategy speculative knob.
        // Preserve a useful shadow seed, but never call an active speculative
        // profile exact until that table is complete.
        const bool speculative_exact =
            params.speculative.types.empty() ||
            (params.speculative.types.size() == 1 &&
             params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE);
        add_bytes(22, speculative, speculative_exact);

        out.push_back(server_cache_fingerprint_enum(
            23, uint16_t(params.cache_type_k)));
        out.push_back(server_cache_fingerprint_enum(
            24, uint16_t(params.cache_type_v)));
        out.push_back(server_cache_fingerprint_bool(25, params.kv_unified));
        out.push_back(server_cache_fingerprint_u32(
            26, uint32_t(std::max(0, params.n_ctx))));
        out.push_back(server_cache_fingerprint_u32(
            27, uint32_t(std::max(0, params.n_parallel))));

        std::vector<uint8_t> vbr_payload;
        vbr_payload.push_back(uint8_t(vbr.armed));
        vbr_payload.push_back(uint8_t(vbr.side_k));
        vbr_payload.push_back(uint8_t(vbr.side_v));
        if (!append_text32(vbr_payload, vbr.budget_mode) ||
            !append_text32(vbr_payload, vbr.family) ||
            !append_text32(vbr_payload, vbr.policy)) {
            return false;
        }
        std::vector<uint8_t> override_census;
        std::array<uint8_t, 32> schedule_digest = {};
        if (!vbr_override_census_payload(
                vbr.override_rows, override_census, schedule_digest)) {
            return false;
        }
        vbr_payload.insert(vbr_payload.end(), schedule_digest.begin(),
                           schedule_digest.end());
        if (!append_binary64(vbr_payload, vbr.capacity_bits) ||
            !append_binary64(vbr_payload, vbr.selected_bpv)) {
            return false;
        }
        append_u64(vbr_payload, vbr.vram_budget_bytes);
        if (!append_binary32(vbr_payload, vbr.reclaim_floor_bpv) ||
            !append_binary32(vbr_payload, vbr.reset_keep_frac)) {
            return false;
        }
        vbr_payload.push_back(uint8_t(vbr.unrepresented_override));
        add_bytes(28, vbr_payload, !vbr.unrepresented_override);
        out.push_back(server_cache_fingerprint_digest(
            29, digest(override_census)));

        uint32_t providers = 0;
        providers |= 1u << uint16_t(common_cache_plan_provider::live_slot);
        providers |= 1u << uint16_t(common_cache_plan_provider::cold_replay);
        if (params.cache_ram_mib != 0) {
            providers |= 1u << uint16_t(
                common_cache_plan_provider::host_cache_entry);
        }
        if (params.n_ctx_checkpoints > 0) {
            providers |= 1u << uint16_t(
                common_cache_plan_provider::live_context_checkpoint);
        }
        out.push_back(server_cache_fingerprint_u32(30, providers));

        std::vector<uint8_t> restore_policy;
        append_u32(restore_policy, 2);
        for (const auto provider : {
                 common_cache_plan_provider::host_cache_entry,
                 common_cache_plan_provider::live_context_checkpoint }) {
            append_u16(restore_policy, uint16_t(provider));
            append_u16(restore_policy, 0);
            const auto recipe = digest_text(
                std::string("restore-representation-v1|") +
                common_cache_plan_provider_name(provider));
            restore_policy.insert(restore_policy.end(), recipe.begin(),
                                  recipe.end());
        }
        add_bytes(31, restore_policy);
        out.push_back(server_cache_fingerprint_enum(
            32, allocator_vmm_regime));

        output_guard.complete = out.size() == 32;
        return output_guard.complete;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_adapter_application_digest_v1(
        const std::vector<common_adapter_lora_info> & adapters,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    try {
        if (adapters.size() > UINT32_MAX) {
            return false;
        }
        uint32_t active_count = 0;
        for (const auto & adapter : adapters) {
            if (!std::isfinite(adapter.scale) ||
                (!adapter.ptr && adapter.scale != 0.0f)) {
                return false;
            }
            active_count += adapter.scale != 0.0f;
        }
        llama_sha256 hash;
        hash.update(ADAPTER_APPLICATION_DOMAIN,
                    sizeof(ADAPTER_APPLICATION_DOMAIN));
        uint8_t count_bytes[4];
        llama_store_le_u32(count_bytes, active_count);
        hash.update(count_bytes, sizeof(count_bytes));
        for (uint32_t ordinal = 0; ordinal < adapters.size(); ++ordinal) {
            const auto & adapter = adapters[ordinal];
            if (adapter.scale == 0.0f) {
                continue;
            }
            uint8_t ordinal_bytes[4];
            llama_store_le_u32(ordinal_bytes, ordinal);
            hash.update(ordinal_bytes, sizeof(ordinal_bytes));
            double scale = double(adapter.scale);
            uint64_t scale_bits = 0;
            std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
            uint8_t scale_bytes[8];
            llama_store_le_u64(scale_bytes, scale_bits);
            hash.update(scale_bytes, sizeof(scale_bytes));
            const uint16_t mode = uint16_t(
                llama_adapter_get_alora_n_invocation_tokens(adapter.ptr) != 0);
            const uint8_t mode_bytes[2] = {
                uint8_t(mode), uint8_t(mode >> 8) };
            hash.update(mode_bytes, sizeof(mode_bytes));
        }
        out = hash.finish();
        return true;
    } catch (...) {
        return false;
    }
}

bool server_cache_adapter_application_entries_digest_v1(
        const std::vector<server_cache_adapter_application_entry> & entries,
        std::array<uint8_t, 32> & out) noexcept {
    out = {};
    try {
        if (entries.size() > UINT32_MAX) {
            return false;
        }
        llama_sha256 hash;
        hash.update(ADAPTER_APPLICATION_DOMAIN,
                    sizeof(ADAPTER_APPLICATION_DOMAIN));
        uint8_t count_bytes[4];
        llama_store_le_u32(count_bytes, uint32_t(entries.size()));
        hash.update(count_bytes, sizeof(count_bytes));
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto & row = entries[i];
            if (!std::isfinite(row.scale) || row.scale == 0.0 ||
                row.application_mode > 1) {
                return false;
            }
            for (size_t previous = 0; previous < i; ++previous) {
                if (entries[previous].ordinal == row.ordinal) {
                    return false;
                }
            }
            uint8_t ordinal_bytes[4];
            llama_store_le_u32(ordinal_bytes, row.ordinal);
            hash.update(ordinal_bytes, sizeof(ordinal_bytes));
            double scale = row.scale;
            if (scale == 0.0) {
                scale = 0.0;
            }
            uint64_t scale_bits = 0;
            std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
            uint8_t scale_bytes[8];
            llama_store_le_u64(scale_bytes, scale_bits);
            hash.update(scale_bytes, sizeof(scale_bytes));
            const uint8_t mode_bytes[2] = {
                uint8_t(row.application_mode),
                uint8_t(row.application_mode >> 8) };
            hash.update(mode_bytes, sizeof(mode_bytes));
        }
        out = hash.finish();
        return true;
    } catch (...) {
        return false;
    }
}

bool server_cache_fingerprint_worker::configure(
        const common_params & params,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime) noexcept {
    return configure(
        params, params.tensor_split,
        sizeof(params.tensor_split) / sizeof(params.tensor_split[0]),
        vbr, effective_n_gpu_layers, pipeline_mode, allocator_vmm_regime);
}

bool server_cache_fingerprint_worker::configure(
        const common_params & params,
        const float * effective_tensor_split,
        size_t effective_tensor_split_count,
        const common_cache_plan_vbr_regime & vbr,
        uint32_t effective_n_gpu_layers,
        uint16_t pipeline_mode,
        uint16_t allocator_vmm_regime) noexcept {
    if (started_ || config_ready_) return false;
    if (!fingerprint_config_root_from_params(
            params, effective_tensor_split, effective_tensor_split_count,
            vbr, effective_n_gpu_layers, pipeline_mode,
            allocator_vmm_regime, hash_buffer_.data(), hash_buffer_.size(),
            config_root_, config_exact_, config_inexact_fields_)) return false;
    config_ready_ = true;
    return true;
}

bool server_cache_fingerprint_worker::add_fixed_artifact(
        server_cache_fingerprint_artifact value) noexcept {
    if (started_ || !valid_artifact_role(value.role) ||
        fixed_artifact_count_ == fixed_artifacts_.size()) return false;
    fixed_artifacts_[fixed_artifact_count_++] = value;
    return true;
}

bool server_cache_fingerprint_worker::launch() noexcept {
    if (started_ || !config_ready_ || fixed_artifact_count_ == 0) return false;
    started_ = true;
    combine();
    return true;
}

bool server_cache_fingerprint_worker::poll(
        server_cache_execution_fingerprint & out) noexcept {
    if (!ready_ || delivered_) {
        return false;
    }
    out = result_;
    delivered_ = true;
    return true;
}

void server_cache_fingerprint_worker::combine() noexcept {
    try {
        bool failed = false;
        bool exact = config_exact_;
        std::sort(fixed_artifacts_.begin(),
                  fixed_artifacts_.begin() + fixed_artifact_count_,
            [](const auto & a, const auto & b) {
                return std::tie(a.role, a.ordinal) <
                       std::tie(b.role, b.ordinal);
            });
        server_cache_execution_fingerprint result;
        if (!failed) {
            llama_sha256 artifact_hash;
            artifact_hash.update(ARTIFACT_DOMAIN, sizeof(ARTIFACT_DOMAIN));
            hash_u32(artifact_hash, uint32_t(fixed_artifact_count_));
            server_cache_fingerprint_artifact previous;
            bool has_previous = false;
            for (size_t fixed_index = 0;
                 fixed_index < fixed_artifact_count_; ++fixed_index) {
                const auto & artifact = fixed_artifacts_[fixed_index];
                if (!valid_artifact_role(artifact.role) ||
                    (!has_previous && (artifact.role !=
                         server_cache_fingerprint_artifact_role::target ||
                         artifact.ordinal != 0)) ||
                    (has_previous &&
                     std::tie(artifact.role, artifact.ordinal) <=
                         std::tie(previous.role, previous.ordinal)) ||
                    (has_previous && artifact.role == previous.role &&
                     artifact.ordinal != previous.ordinal + 1) ||
                    (has_previous && artifact.role != previous.role &&
                     artifact.ordinal != 0)) {
                    failed = true;
                    break;
                }
                fingerprint_artifact_hash_update(artifact_hash, artifact);
                exact = exact && artifact.exact;
                previous = artifact;
                has_previous = true;
            }
            if (!failed) result.artifact_root = artifact_hash.finish();
        }
        if (!failed) {
            result.config_root = config_root_;
            llama_sha256 execution_hash;
            execution_hash.update(EXEC_DOMAIN, sizeof(EXEC_DOMAIN));
            execution_hash.update(result.artifact_root.data(),
                                  result.artifact_root.size());
            execution_hash.update(result.config_root.data(),
                                  result.config_root.size());
            result.execution_root = execution_hash.finish();
            result.complete = true;
            result.exact = exact;
        }
        result_ = result;
        ready_ = true;
    } catch (...) {
        result_ = {};
        ready_ = true;
    }
}
