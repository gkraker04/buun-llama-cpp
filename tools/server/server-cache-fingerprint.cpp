#include "server-cache-fingerprint.h"

#include "../../src/llama-sha256.h"
#include "../../common/build-info.h"
#include "../../common/common.h"
#include "../../common/common-cache-plan-estimate.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>

#if defined(_WIN32)
#  include <io.h>
#else
#  if defined(__linux__)
#    include <linux/ioprio.h>
#    include <sys/syscall.h>
#  endif
#  include <unistd.h>
#endif

namespace {

constexpr size_t MAX_CODEC_BYTES = 1024 * 1024;
constexpr size_t HASH_CHUNK_BYTES = 1024 * 1024;
constexpr uint64_t HASH_RATE_BYTES_PER_SECOND = 32ULL * 1024 * 1024;
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

void close_descriptor(int fd) noexcept {
    if (fd < 0) {
        return;
    }
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}

int64_t read_at(int fd, void * data, size_t size, uint64_t offset) noexcept {
#if defined(_WIN32)
    if (_lseeki64(fd, offset, SEEK_SET) < 0) {
        return -1;
    }
    return _read(fd, data, unsigned(std::min<size_t>(size, INT_MAX)));
#else
    return pread(fd, data, size, off_t(offset));
#endif
}

struct worker_input {
    std::vector<server_cache_fingerprint_descriptor> descriptors;
    std::vector<server_cache_fingerprint_field> fields;
    std::vector<server_cache_fingerprint_artifact> fixed_artifacts;

    ~worker_input() {
        for (auto & row : descriptors) {
            close_descriptor(row.descriptor);
        }
    }
};

} // namespace

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
        static constexpr char ARTIFACT_DOMAIN[] = "buun-zc-artifacts-v1";
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
                                artifact.content_sha256.data(), 32)) {
                return false;
            }
        }

        std::vector<uint8_t> config_bytes;
        static constexpr char CONFIG_DOMAIN[] = "buun-zc-config-v1";
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
        static constexpr char EXEC_DOMAIN[] = "buun-zc-exec-v1";
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

        const std::string cost_abi = std::string("zc-estimator-v2|") +
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
        std::vector<ggml_backend_dev_t> effective_devices;
        if (params.devices.empty()) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * device = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(device) ==
                        GGML_BACKEND_DEVICE_TYPE_GPU) {
                    effective_devices.push_back(device);
                }
            }
        } else {
            for (auto * device : params.devices) {
                if (device) {
                    effective_devices.push_back(device);
                }
            }
        }
        if (effective_devices.empty()) {
            effective_n_gpu_layers = 0;
        }

        auto driver = server_cache_fingerprint_u32(6, 0);
        auto runtime = server_cache_fingerprint_u32(7, 0);
        // GGML has no cross-backend driver/runtime query. A zero is an honest
        // stable shadow value, never exact on a device-backed profile.
        driver.exact = effective_devices.empty();
        runtime.exact = effective_devices.empty();
        out.push_back(std::move(driver));
        out.push_back(std::move(runtime));
        // Registry names alone do not distinguish runtime CPU ISA/model
        // dispatch. Preserve the complete backend feature string as the
        // compatibility seed, but keep it shadow-only until the backend
        // exposes a versioned, collision-resistant hardware identity.
        if (!add_utf8(8, llama_print_system_info(), false)) return false;

        std::vector<uint8_t> devices;
        append_u32(devices, uint32_t(effective_devices.size()));
        for (uint32_t i = 0; i < effective_devices.size(); ++i) {
            const auto dev = effective_devices[i];
            append_u32(devices, i);
            const auto pseudo_uuid = digest_text(
                std::string(ggml_backend_dev_name(dev)) + "\n" +
                ggml_backend_dev_description(dev));
            devices.insert(devices.end(), pseudo_uuid.begin(),
                           pseudo_uuid.begin() + 16);
            append_u16(devices, 0); // backend enum unavailable
            append_u16(devices, 0); // arch major unavailable
            append_u16(devices, 0); // arch minor unavailable
        }
        add_bytes(9, devices, effective_devices.empty());

        std::vector<uint8_t> topology;
        append_u32(topology, 0); // no portable link/UUID API in GGML v1
        add_bytes(10, topology, effective_devices.empty());

        std::vector<uint8_t> placement;
        append_u16(placement, uint16_t(params.split_mode));
        append_u32(placement, uint32_t(std::max(0, params.main_gpu)));
        append_u32(placement, effective_n_gpu_layers);
        append_u32(placement, uint32_t(effective_devices.size()));
        for (size_t i = 0; i < effective_devices.size(); ++i) {
            if (!append_binary64(placement, params.tensor_split[i])) {
                return false;
            }
        }
        placement.push_back(uint8_t(!params.no_kv_offload));
        placement.push_back(uint8_t(!params.no_op_offload));
        add_bytes(11, placement);

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

server_cache_fingerprint_worker::~server_cache_fingerprint_worker() {
    stop();
}

void server_cache_fingerprint_descriptors_close(
        std::vector<server_cache_fingerprint_descriptor> & descriptors) noexcept {
    for (auto & row : descriptors) {
        close_descriptor(row.descriptor);
        row.descriptor = -1;
    }
}

bool server_cache_fingerprint_worker::start(
        std::vector<server_cache_fingerprint_descriptor> descriptors,
        std::vector<server_cache_fingerprint_field> fields,
        std::vector<server_cache_fingerprint_artifact> fixed_artifacts) noexcept {
    if (started_ || thread_.joinable() ||
        (descriptors.empty() && fixed_artifacts.empty())) {
        for (auto & row : descriptors) {
            close_descriptor(row.descriptor);
            row.descriptor = -1;
        }
        return false;
    }
    cancel_.store(false, std::memory_order_relaxed);
    started_ = true;
    try {
        auto input = std::make_unique<worker_input>();
        input->descriptors.swap(descriptors);
        input->fields.swap(fields);
        input->fixed_artifacts.swap(fixed_artifacts);
        thread_ = std::thread(
            [this, input = std::move(input)]() mutable {
                run(std::move(input->descriptors),
                    std::move(input->fields),
                    std::move(input->fixed_artifacts));
            });
        return true;
    } catch (...) {
        started_ = false;
        for (auto & row : descriptors) {
            close_descriptor(row.descriptor);
        }
        return false;
    }
}

bool server_cache_fingerprint_worker::poll(
        server_cache_execution_fingerprint & out) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_ || delivered_) {
        return false;
    }
    out = result_;
    delivered_ = true;
    return true;
}

void server_cache_fingerprint_worker::stop() noexcept {
    cancel_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void server_cache_fingerprint_worker::run(
        std::vector<server_cache_fingerprint_descriptor> descriptors,
        std::vector<server_cache_fingerprint_field> fields,
        std::vector<server_cache_fingerprint_artifact> fixed_artifacts) noexcept {
    std::vector<server_cache_fingerprint_artifact> artifacts =
        std::move(fixed_artifacts);
    try {
#if defined(__linux__)
        // Best-effort background I/O class. Failure is safe: the explicit
        // scheduler pause and bounded rate limiter remain authoritative.
        (void) syscall(SYS_ioprio_set, IOPRIO_WHO_PROCESS, 0,
                       IOPRIO_PRIO_VALUE(IOPRIO_CLASS_IDLE, 0));
#endif
        artifacts.reserve(artifacts.size() + descriptors.size());
        std::vector<uint8_t> buffer(HASH_CHUNK_BYTES);
        auto rate_window_started = std::chrono::steady_clock::now();
        uint64_t rate_window_read = 0;
        bool failed = false;
        for (auto & source : descriptors) {
            llama_sha256 hash;
            uint64_t offset = 0;
            while (offset < source.byte_length) {
                if (cancel_.load(std::memory_order_relaxed)) {
                    failed = true;
                    break;
                }
                while (scheduler_demand_.load(std::memory_order_relaxed) &&
                       !cancel_.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    rate_window_started = std::chrono::steady_clock::now();
                    rate_window_read = 0;
                }
                if (cancel_.load(std::memory_order_relaxed)) {
                    failed = true;
                    break;
                }
                const size_t want = size_t(std::min<uint64_t>(
                    buffer.size(), source.byte_length - offset));
                const int64_t got = read_at(
                    source.descriptor, buffer.data(), want, offset);
                if (got <= 0 || size_t(got) != want) {
                    failed = true;
                    break;
                }
                hash.update(buffer.data(), size_t(got));
                offset += uint64_t(got);
                rate_window_read += uint64_t(got);
                const uint64_t whole_seconds =
                    rate_window_read / HASH_RATE_BYTES_PER_SECOND;
                const uint64_t remainder =
                    rate_window_read % HASH_RATE_BYTES_PER_SECOND;
                const uint64_t target_us = whole_seconds * 1000000ULL +
                    remainder * 1000000ULL /
                        HASH_RATE_BYTES_PER_SECOND;
                const auto target = rate_window_started +
                    std::chrono::microseconds(target_us);
                while (target > std::chrono::steady_clock::now() &&
                       !cancel_.load(std::memory_order_relaxed)) {
                    const auto remaining = target -
                        std::chrono::steady_clock::now();
                    std::this_thread::sleep_for(std::min(
                        remaining,
                        std::chrono::duration_cast<
                            std::chrono::steady_clock::duration>(
                            std::chrono::milliseconds(2))));
                }
            }
            if (!failed) {
                artifacts.push_back({ source.role, source.ordinal,
                    source.byte_length, hash.finish(),
                    source.integrity_exact });
            }
            close_descriptor(source.descriptor);
            source.descriptor = -1;
            if (failed) {
                break;
            }
        }
        for (auto & source : descriptors) {
            close_descriptor(source.descriptor);
            source.descriptor = -1;
        }
        server_cache_execution_fingerprint result;
        if (!failed && !cancel_.load(std::memory_order_relaxed)) {
            std::sort(artifacts.begin(), artifacts.end(),
                [](const auto & a, const auto & b) {
                    return std::tie(a.role, a.ordinal) <
                           std::tie(b.role, b.ordinal);
                });
            (void) server_cache_execution_fingerprint_v1(
                std::move(artifacts), std::move(fields), result);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = result;
        ready_ = true;
    } catch (...) {
        for (auto & source : descriptors) {
            close_descriptor(source.descriptor);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = {};
        ready_ = true;
    }
}
