#include "server-cache-fingerprint.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-ext.h"
#include "../src/llama-sha256.h"
#include "common.h"
#include "common-cache-plan-estimate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

static std::atomic<bool> reject_allocations { false };

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void * operator new(std::size_t size) {
    if (reject_allocations.load(std::memory_order_relaxed)) throw std::bad_alloc();
    if (void * value = std::malloc(size)) return value;
    throw std::bad_alloc();
}

void * operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void * value) noexcept { std::free(value); }
void operator delete[](void * value) noexcept { std::free(value); }
void operator delete(void * value, std::size_t) noexcept { std::free(value); }
void operator delete[](void * value, std::size_t) noexcept { std::free(value); }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define CHECK(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #x); \
        std::abort(); \
    } \
} while (0)

static_assert(sizeof(server_cache_fingerprint_worker) <= 1024 * 1024,
              "fingerprint worker and its buffer must fit the ZC4 arena");

static std::string hex(const std::array<uint8_t, 32> & value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint8_t byte : value) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 15]);
    }
    return out;
}

static uint32_t read_u32(const std::vector<uint8_t> & value, size_t offset) {
    return uint32_t(value[offset]) |
        (uint32_t(value[offset + 1]) << 8) |
        (uint32_t(value[offset + 2]) << 16) |
        (uint32_t(value[offset + 3]) << 24);
}

static std::array<uint8_t, 32> config_root(
        const std::vector<server_cache_fingerprint_field> & fields) {
    static constexpr char domain[] = "buun-zc-config-v1";
    llama_sha256 hash;
    hash.update(domain, sizeof(domain));
    uint8_t count[4];
    llama_store_le_u32(count, uint32_t(fields.size()));
    hash.update(count, sizeof(count));
    for (const auto & field : fields) {
        const uint8_t header[3] = {
            uint8_t(field.id), uint8_t(field.id >> 8), uint8_t(field.type) };
        hash.update(header, sizeof(header));
        uint8_t size[4];
        llama_store_le_u32(size, uint32_t(field.payload.size()));
        hash.update(size, sizeof(size));
        hash.update(field.payload.data(), field.payload.size());
    }
    return hash.finish();
}

static server_cache_fingerprint_field utf8(uint16_t id, const char * value) {
    server_cache_fingerprint_field out;
    CHECK(server_cache_fingerprint_utf8(id, value, std::strlen(value), out));
    return out;
}

static std::vector<server_cache_fingerprint_field> fields() {
    std::array<uint8_t, 32> build = {};
    build[0] = 0x42;
    const uint8_t empty_count[] = { 0, 0, 0, 0 };
    const uint8_t placement[] = {
        1, 0,             // split_mode
        0, 0, 0, 0,      // main_device
        7, 0, 0, 0,      // n_gpu_layers
        0, 0, 0, 0,      // split_count
        1, 1,             // offload_kqv, op_offload
    };
    const uint8_t speculative[] = {
        0, 0,             // strategy
        0, 0, 0, 0,      // n_draft
        0, 0, 0, 0,      // n_min
        0, 0, 0, 0,      // n_max
        0, 0, 0, 0, 0, 0, 0, 0, // p_min
        0, 0, 0, 0, 0, 0, 0, 0, // p_split
        0, 0,             // dynamic, dflash
        // dflash policy digest follows
    };
    std::vector<uint8_t> speculative_full(
        speculative, speculative + sizeof(speculative));
    speculative_full.resize(speculative_full.size() + 32, 0);
    std::vector<uint8_t> vbr(3, 0); // armed, side_k, side_v
    vbr.resize(vbr.size() + 3 * 4, 0); // three empty UTF-8 fields
    vbr.resize(vbr.size() + 32, 0); // schedule digest
    vbr.resize(vbr.size() + 8 + 8 + 8 + 4 + 4 + 1, 0);

    std::vector<server_cache_fingerprint_field> out;
    out.reserve(32);
    out.push_back(server_cache_fingerprint_u32(1, 2));
    out.push_back(server_cache_fingerprint_u32(2, 2));
    out.push_back(server_cache_fingerprint_u32(3, 2));
    out.push_back(server_cache_fingerprint_digest(4, build));
    out.push_back(utf8(5, "cpu-test/v1"));
    out.push_back(server_cache_fingerprint_u32(6, 0));
    out.push_back(server_cache_fingerprint_u32(7, 0));
    out.push_back(utf8(8, "x86-test"));
    out.push_back(server_cache_fingerprint_bytes(9, empty_count, 4));
    out.push_back(server_cache_fingerprint_bytes(10, empty_count, 4));
    out.push_back(server_cache_fingerprint_bytes(11, placement, sizeof(placement)));
    out.push_back(server_cache_fingerprint_u32(12, 512));
    out.push_back(server_cache_fingerprint_u32(13, 128));
    out.push_back(server_cache_fingerprint_u32(14, 4));
    out.push_back(server_cache_fingerprint_u32(15, 4));
    out.push_back(server_cache_fingerprint_bytes(16, empty_count, 4));
    out.push_back(server_cache_fingerprint_bytes(17, empty_count, 4));
    out.push_back(server_cache_fingerprint_enum(18, 0));
    out.push_back(server_cache_fingerprint_bool(19, true));
    out.push_back(server_cache_fingerprint_enum(20, 1));
    out.push_back(server_cache_fingerprint_enum(21, 0));
    out.push_back(server_cache_fingerprint_bytes(
        22, speculative_full.data(), speculative_full.size()));
    out.push_back(server_cache_fingerprint_enum(23, 1));
    out.push_back(server_cache_fingerprint_enum(24, 1));
    out.push_back(server_cache_fingerprint_bool(25, false));
    out.push_back(server_cache_fingerprint_u32(26, 32768));
    out.push_back(server_cache_fingerprint_u32(27, 4));
    out.push_back(server_cache_fingerprint_bytes(28, vbr.data(), vbr.size()));
    out.push_back(server_cache_fingerprint_digest(29, build));
    out.push_back(server_cache_fingerprint_u32(30, 7));
    out.push_back(server_cache_fingerprint_bytes(31, empty_count, 4));
    out.push_back(server_cache_fingerprint_enum(32, 0));
    return out;
}

static std::vector<server_cache_fingerprint_artifact> artifacts() {
    std::array<uint8_t, 32> target = {};
    std::array<uint8_t, 32> draft = {};
    target[0] = 0x11;
    draft[0] = 0x22;
    return {
        { server_cache_fingerprint_artifact_role::target, 0, 123, target, true },
        { server_cache_fingerprint_artifact_role::draft, 0, 456, draft, true },
    };
}

struct fake_gpu_identity {
    const char * name;
    const char * description;
    ggml_backend_device_identity_v1 identity;
};

static const char * fake_gpu_name(ggml_backend_dev_t device) {
    return static_cast<fake_gpu_identity *>(device->context)->name;
}

static const char * fake_gpu_description(ggml_backend_dev_t device) {
    return static_cast<fake_gpu_identity *>(device->context)->description;
}

static enum ggml_backend_dev_type fake_gpu_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static bool fake_gpu_identity_query(
        ggml_backend_dev_t device,
        ggml_backend_device_identity_v1 * identity) {
    if (!device || !identity || identity->struct_size != sizeof(*identity)) {
        return false;
    }
    *identity = static_cast<fake_gpu_identity *>(device->context)->identity;
    return true;
}

static bool fake_gpu_link_query(
        ggml_backend_dev_t source,
        ggml_backend_dev_t destination,
        ggml_backend_device_link_v1 * link) {
    if (!source || !destination || !link ||
        link->struct_size != sizeof(*link) || source == destination) {
        return false;
    }
    const auto * src = static_cast<fake_gpu_identity *>(source->context);
    const auto * dst = static_cast<fake_gpu_identity *>(destination->context);
    *link = {};
    link->struct_size = sizeof(*link);
    link->link_class = uint16_t(2 +
        ((src->identity.pci_domain_bus_device_function ^
          dst->identity.pci_domain_bus_device_function) & 3));
    link->p2p = 1;
    return true;
}

static void * fake_gpu_proc(ggml_backend_reg_t, const char * name) {
    if (std::strcmp(name, GGML_BACKEND_DEVICE_IDENTITY_V1_PROC) == 0) {
        return reinterpret_cast<void *>(fake_gpu_identity_query);
    }
    if (std::strcmp(name, GGML_BACKEND_DEVICE_LINK_V1_PROC) == 0) {
        return reinterpret_cast<void *>(fake_gpu_link_query);
    }
    return nullptr;
}

static ggml_backend_device make_fake_gpu(
        ggml_backend_reg_t reg, fake_gpu_identity * identity) {
    ggml_backend_device out = {};
    out.iface.get_name = fake_gpu_name;
    out.iface.get_description = fake_gpu_description;
    out.iface.get_type = fake_gpu_type;
    out.reg = reg;
    out.context = identity;
    return out;
}

static fake_gpu_identity make_fake_gpu_identity(
        const char * name, uint8_t uuid_byte, uint32_t pci_bdf) {
    fake_gpu_identity out = {};
    out.name = name;
    out.description = "fake CUDA GPU";
    out.identity.struct_size = sizeof(out.identity);
    out.identity.driver_version = 13010;
    out.identity.runtime_version = 13000;
    std::fill(std::begin(out.identity.uuid), std::end(out.identity.uuid), uuid_byte);
    out.identity.pci_domain_bus_device_function = pci_bdf;
    out.identity.backend_kind = GGML_BACKEND_IDENTITY_KIND_CUDA;
    out.identity.arch_major = 8;
    out.identity.arch_minor = 6;
    return out;
}

int main() {
    server_cache_execution_fingerprint first;
    CHECK(server_cache_execution_fingerprint_v1(
        artifacts(), fields(), first));
    CHECK(first.complete && first.exact);

    auto shuffled_fields = fields();
    std::reverse(shuffled_fields.begin(), shuffled_fields.end());
    auto shuffled_artifacts = artifacts();
    std::reverse(shuffled_artifacts.begin(), shuffled_artifacts.end());
    server_cache_execution_fingerprint reordered;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), shuffled_fields, reordered));
    CHECK(!server_cache_execution_fingerprint_v1(
        shuffled_artifacts, fields(), reordered));

    // Production-codec golden. Names and paths are intentionally absent: a
    // rename of the same loader object cannot alter any of these bytes.
    CHECK(hex(first.artifact_root) ==
          "3c6440ad78d136e44565da591e0171606d66fe1d561be4663c65dc605bed5ab6");
    CHECK(hex(first.config_root) ==
          "6ca7e20b5bedd77c62565a8853b959e6d85d709dd25655293fa203fad7e12aff");
    CHECK(hex(first.execution_root) ==
          "bad581506275f13c4118cf01d56ba31bb1f0141dd4371250daf8adc4f4b15084");

    auto changed = artifacts();
    changed[0].content_sha256[3] = 7;
    server_cache_execution_fingerprint different;
    CHECK(server_cache_execution_fingerprint_v1(changed, fields(), different));
    CHECK(different.execution_root != first.execution_root);

    auto with_mmproj = artifacts();
    std::array<uint8_t, 32> mmproj = {};
    mmproj[0] = 0x33;
    with_mmproj.push_back({
        server_cache_fingerprint_artifact_role::mmproj,
        0, 789, mmproj, false });
    CHECK(server_cache_execution_fingerprint_v1(
        with_mmproj, fields(), different));
    CHECK(different.execution_root != first.execution_root);
    CHECK(!different.exact);

    auto duplicate_artifact = artifacts();
    duplicate_artifact.push_back(duplicate_artifact.front());
    CHECK(!server_cache_execution_fingerprint_v1(
        duplicate_artifact, fields(), different));

    auto duplicate_field = fields();
    duplicate_field.back() = duplicate_field.front();
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), duplicate_field, different));

    auto bad_type = fields();
    bad_type[0].type = server_cache_fingerprint_field_type::u64;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), bad_type, different));

    auto unknown_enum = fields();
    unknown_enum[17] = server_cache_fingerprint_enum(18, UINT16_MAX);
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), unknown_enum, different));

    auto trailing_structured_bytes = fields();
    trailing_structured_bytes[10].payload.push_back(0);
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), trailing_structured_bytes, different));

    auto bad_device_ordinal = fields();
    bad_device_ordinal[8].payload.resize(4 + 26, 0);
    bad_device_ordinal[8].payload[0] = 1;
    bad_device_ordinal[8].payload[4] = 1;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), bad_device_ordinal, different));

    auto unknown_link_class = fields();
    unknown_link_class[9].payload.resize(4 + 19, 0);
    unknown_link_class[9].payload[0] = 1;
    unknown_link_class[9].payload[12] = 1;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), unknown_link_class, different));

    auto negative_zero_split = fields();
    negative_zero_split[10].payload.assign(24, 0);
    negative_zero_split[10].payload[0] = 1;
    negative_zero_split[10].payload[10] = 1;
    negative_zero_split[10].payload[21] = 0x80;
    negative_zero_split[10].payload[22] = 1;
    negative_zero_split[10].payload[23] = 1;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), negative_zero_split, different));

    auto nonfinite_spec = fields();
    nonfinite_spec[21].payload[20] = 0xf0;
    nonfinite_spec[21].payload[21] = 0x7f;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), nonfinite_spec, different));

    auto negative_zero_vbr = fields();
    negative_zero_vbr[27].payload[54] = 0x80;
    CHECK(!server_cache_execution_fingerprint_v1(
        artifacts(), negative_zero_vbr, different));

    server_cache_fingerprint_field invalid_utf8;
    const char overlong[] = { char(0xc0), char(0x80) };
    CHECK(!server_cache_fingerprint_utf8(
        5, overlong, sizeof(overlong), invalid_utf8));
    CHECK(!server_cache_fingerprint_binary64(
        1, std::numeric_limits<double>::infinity(), invalid_utf8));

    auto shadow = artifacts();
    shadow[0].exact = false;
    CHECK(server_cache_execution_fingerprint_v1(shadow, fields(), different));
    CHECK(different.complete && !different.exact);

    // Request-effective adapter identity is ordered and scale-sensitive. A
    // server-wide loaded catalog cannot stand in for this per-request key.
    server_cache_adapter_application_entry adapter_a;
    adapter_a.ordinal = 1;
    adapter_a.scale = 1.0f;
    server_cache_adapter_application_entry adapter_b;
    adapter_b.ordinal = 3;
    adapter_b.scale = 0.5f;
    std::array<uint8_t, 32> application_ab = {};
    std::array<uint8_t, 32> application_a = {};
    std::array<uint8_t, 32> application_scaled = {};
    CHECK(server_cache_adapter_application_entries_digest_v1(
        { adapter_a, adapter_b }, application_ab));
    CHECK(server_cache_adapter_application_entries_digest_v1(
        { adapter_a }, application_a));
    adapter_b.scale = 0.25f;
    CHECK(server_cache_adapter_application_entries_digest_v1(
        { adapter_a, adapter_b }, application_scaled));
    CHECK(application_ab != application_a);
    CHECK(application_ab != application_scaled);
    CHECK(hex(application_ab) ==
          "b3f15fa073cad9076b22cd15fae92ce16e48b7604e85bd84844d5910342dcdf4");
    adapter_b.scale = 0.5;
    adapter_b.application_mode = 1;
    CHECK(server_cache_adapter_application_entries_digest_v1(
        { adapter_a, adapter_b }, application_scaled));
    CHECK(application_ab != application_scaled);
    adapter_b.application_mode = 2;
    CHECK(!server_cache_adapter_application_entries_digest_v1(
        { adapter_a, adapter_b }, application_scaled));
    adapter_b.application_mode = 0;
    CHECK(server_cache_adapter_application_entries_digest_v1(
        { adapter_b, adapter_a }, application_scaled));
    CHECK(application_ab != application_scaled);
    CHECK(!server_cache_adapter_application_entries_digest_v1(
        { adapter_a, adapter_a }, application_scaled));

    // Production lowering uses resolved placement. An explicit CPU device
    // selection cannot retain the loader's negative/all-layers GPU sentinel.
    common_params production_params;
    production_params.devices = { nullptr };
    production_params.speculative.set_type(COMMON_SPECULATIVE_TYPE_NONE);
    common_cache_plan_vbr_regime production_vbr;
    std::vector<server_cache_fingerprint_field> production_fields;
    CHECK(server_cache_fingerprint_fields_v1(
        production_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields.size() == 32);
    CHECK(production_fields[10].id == 11);
    CHECK(read_u32(production_fields[10].payload, 6) == 0);
    const auto default_production_fields = production_fields;

    // Argument parsing reserves the auto-fit override workspace with null
    // entries on every normal CLI launch. Those rows are capacity, not an
    // active placement override, and must preserve the exact same profile.
    production_params.tensor_buft_overrides.resize(
        llama_max_tensor_buft_overrides(), { nullptr, nullptr });
    CHECK(server_cache_fingerprint_fields_v1(
        production_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields.size() == default_production_fields.size());
    for (size_t i = 0; i < production_fields.size(); ++i) {
        CHECK(production_fields[i].id == default_production_fields[i].id);
        CHECK(production_fields[i].type == default_production_fields[i].type);
        CHECK(production_fields[i].payload == default_production_fields[i].payload);
        CHECK(production_fields[i].exact == default_production_fields[i].exact);
    }
    production_params.tensor_buft_overrides[0] = { ".*", nullptr };
    CHECK(server_cache_fingerprint_fields_v1(
        production_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(!production_fields[5].exact);
    CHECK(!production_fields[6].exact);
    CHECK(!production_fields[8].exact);
    CHECK(!production_fields[9].exact);
    CHECK(!production_fields[10].exact);
    production_params.tensor_buft_overrides[0] = { nullptr, nullptr };
    production_params.tensor_buft_overrides.clear();

    // A CUDA-enabled gate exercises the real optional backend query rather
    // than counting only fake-interface tokens. CPU-only builds keep this
    // branch inert and prove the zero-device profile separately above.
    size_t real_gpu_count = 0;
    bool real_gpu_queries_complete = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ++real_gpu_count;
        auto * reg = ggml_backend_dev_backend_reg(device);
        const auto query = reinterpret_cast<
            ggml_backend_device_identity_v1_t>(
                ggml_backend_reg_get_proc_address(
                    reg, GGML_BACKEND_DEVICE_IDENTITY_V1_PROC));
        if (!query) {
            real_gpu_queries_complete = false;
            continue;
        }
        ggml_backend_device_identity_v1 identity = {};
        identity.struct_size = sizeof(identity);
        if (!query(device, &identity) || identity.driver_version == 0 ||
            identity.runtime_version == 0) {
            real_gpu_queries_complete = false;
        }
    }
    if (real_gpu_count != 0) {
        common_params real_gpu_params;
        real_gpu_params.speculative.set_type(COMMON_SPECULATIVE_TYPE_NONE);
        CHECK(server_cache_fingerprint_fields_v1(
            real_gpu_params, production_vbr, 99, 0, 0, production_fields));
        CHECK(read_u32(production_fields[8].payload, 0) == real_gpu_count);
        if (real_gpu_queries_complete) {
            for (size_t i = 5; i <= 10; ++i) CHECK(production_fields[i].exact);
        } else {
            CHECK(!production_fields[5].exact || !production_fields[8].exact ||
                  !production_fields[9].exact || !production_fields[10].exact);
        }
    }

    // Physical CUDA identity and placement are canonicalized independently of
    // loader enumeration. Reversing A/B while retaining the same main device
    // and physical tensor split must reuse the exact same persisted profile.
    ggml_backend_reg fake_reg = {};
    fake_reg.api_version = GGML_BACKEND_API_VERSION;
    fake_reg.iface.get_proc_address = fake_gpu_proc;
    auto gpu_a_identity = make_fake_gpu_identity("CUDA-A", 0x11, 0x00000100);
    auto gpu_b_identity = make_fake_gpu_identity("CUDA-B", 0x22, 0x00000200);
    auto gpu_a = make_fake_gpu(&fake_reg, &gpu_a_identity);
    auto gpu_b = make_fake_gpu(&fake_reg, &gpu_b_identity);

    common_params placement_ab = production_params;
    placement_ab.devices = { &gpu_a, &gpu_b, nullptr };
    placement_ab.main_gpu = 1;
    placement_ab.tensor_split[0] = 0.25f;
    placement_ab.tensor_split[1] = 0.75f;
    std::vector<server_cache_fingerprint_field> fields_ab;
    CHECK(server_cache_fingerprint_fields_v1(
        placement_ab, production_vbr, 99, 0, 0, fields_ab));
    CHECK(fields_ab[5].exact && fields_ab[6].exact &&
          fields_ab[8].exact && fields_ab[9].exact && fields_ab[10].exact);

    common_params placement_ba = production_params;
    placement_ba.devices = { &gpu_b, &gpu_a, nullptr };
    placement_ba.main_gpu = 0;
    placement_ba.tensor_split[0] = 0.75f;
    placement_ba.tensor_split[1] = 0.25f;
    std::vector<server_cache_fingerprint_field> fields_ba;
    CHECK(server_cache_fingerprint_fields_v1(
        placement_ba, production_vbr, 99, 0, 0, fields_ba));
    for (size_t i = 5; i <= 10; ++i) {
        CHECK(fields_ab[i].payload == fields_ba[i].payload);
        CHECK(fields_ab[i].exact == fields_ba[i].exact);
    }

    auto changed_placement = placement_ab;
    changed_placement.tensor_split[0] = 0.5f;
    changed_placement.tensor_split[1] = 0.5f;
    CHECK(server_cache_fingerprint_fields_v1(
        changed_placement, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields[10].payload != fields_ab[10].payload);

    auto single_gpu = placement_ab;
    single_gpu.devices = { &gpu_a, nullptr };
    single_gpu.main_gpu = 0;
    single_gpu.tensor_split[0] = 1.0f;
    CHECK(server_cache_fingerprint_fields_v1(
        single_gpu, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields[8].payload != fields_ab[8].payload);
    CHECK(production_fields[10].payload != fields_ab[10].payload);

    auto changed_driver_identity = gpu_a_identity;
    changed_driver_identity.identity.driver_version += 1;
    auto gpu_a_new_driver = make_fake_gpu(&fake_reg, &changed_driver_identity);
    auto changed_driver = single_gpu;
    changed_driver.devices = { &gpu_a_new_driver, nullptr };
    CHECK(server_cache_fingerprint_fields_v1(
        changed_driver, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields[5].exact);
    CHECK(production_fields[5].payload != fields_ab[5].payload);

    auto missing_uuid_identity = gpu_a_identity;
    std::fill(std::begin(missing_uuid_identity.identity.uuid),
              std::end(missing_uuid_identity.identity.uuid), 0);
    auto missing_uuid = make_fake_gpu(&fake_reg, &missing_uuid_identity);
    auto missing_uuid_params = single_gpu;
    missing_uuid_params.devices = { &missing_uuid, nullptr };
    CHECK(server_cache_fingerprint_fields_v1(
        missing_uuid_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(!production_fields[5].exact && !production_fields[8].exact &&
          !production_fields[10].exact);

    auto changed_runtime = gpu_b_identity;
    changed_runtime.identity.runtime_version += 1;
    auto gpu_b_new_runtime = make_fake_gpu(&fake_reg, &changed_runtime);
    auto runtime_params = placement_ab;
    runtime_params.devices = { &gpu_a, &gpu_b_new_runtime, nullptr };
    CHECK(server_cache_fingerprint_fields_v1(
        runtime_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(!production_fields[5].exact && !production_fields[6].exact &&
          !production_fields[8].exact && !production_fields[9].exact &&
          !production_fields[10].exact);

    auto active_spec = production_params;
    active_spec.speculative.set_type(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
    CHECK(server_cache_fingerprint_fields_v1(
        active_spec, production_vbr, 0, 0, 0, production_fields));
    CHECK(!production_fields[21].exact);

    // Descriptor hashing consumes the exact loader object, not a reopened
    // path, and publishes only a complete root. The synthetic file remains
    // mutable, so the resulting compatibility seed must stay shadow-only.
    FILE * file = std::tmpfile();
    CHECK(file != nullptr);
    CHECK(std::fwrite("abc", 1, 3, file) == 3);
    CHECK(std::fflush(file) == 0);
#if defined(_WIN32)
    const int duplicate = _dup(_fileno(file));
#else
    const int duplicate = dup(fileno(file));
#endif
    CHECK(duplicate >= 0);
    server_cache_fingerprint_worker worker;
    CHECK(worker.start({ {
        server_cache_fingerprint_artifact_role::target,
        0, duplicate, 3, false } }, fields()));
    server_cache_execution_fingerprint worker_result;
    bool worker_delivered = false;
    for (int i = 0; i < 200 &&
             !(worker_delivered = worker.poll(worker_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(worker_delivered);
    CHECK(worker_result.complete && !worker_result.exact);
    const std::array<uint8_t, 32> abc_sha = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    server_cache_execution_fingerprint expected_worker;
    CHECK(server_cache_execution_fingerprint_v1({ {
        server_cache_fingerprint_artifact_role::target,
        0, 3, abc_sha, false } }, fields(), expected_worker));
    CHECK(worker_result.execution_root == expected_worker.execution_root);
    worker.stop();

    // The production-only config path streams the identical frozen bytes into
    // the arena-owned worker without constructing the public vector codec.
#if defined(_WIN32)
    const int configured_duplicate = _dup(_fileno(file));
#else
    const int configured_duplicate = dup(fileno(file));
#endif
    CHECK(configured_duplicate >= 0);
    auto configured_worker = std::make_unique<server_cache_fingerprint_worker>();
    reject_allocations.store(true, std::memory_order_relaxed);
    const bool configured_without_allocation = configured_worker->configure(
        production_params, production_vbr, 99, 0, 0);
    reject_allocations.store(false, std::memory_order_relaxed);
    CHECK(configured_without_allocation);
    CHECK(configured_worker->add_descriptor({
        server_cache_fingerprint_artifact_role::target,
        0, configured_duplicate, 3, false }));
    CHECK(configured_worker->launch());
    server_cache_execution_fingerprint configured_result;
    bool configured_delivered = false;
    for (int i = 0; i < 200 &&
             !(configured_delivered = configured_worker->poll(configured_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(configured_delivered && configured_result.complete);
    CHECK(configured_result.config_root == config_root(default_production_fields));
    configured_worker->stop();

#if defined(_WIN32)
    const int configured_gpu_duplicate = _dup(_fileno(file));
#else
    const int configured_gpu_duplicate = dup(fileno(file));
#endif
    CHECK(configured_gpu_duplicate >= 0);
    auto configured_gpu_worker =
        std::make_unique<server_cache_fingerprint_worker>();
    auto placement_without_effective_split = placement_ab;
    std::fill(std::begin(placement_without_effective_split.tensor_split),
              std::end(placement_without_effective_split.tensor_split), 0.0f);
    const float effective_split[] = { 0.25f, 0.75f };
    CHECK(configured_gpu_worker->configure(
        placement_without_effective_split,
        effective_split, std::size(effective_split),
        production_vbr, 99, 0, 0));
    CHECK(configured_gpu_worker->add_descriptor({
        server_cache_fingerprint_artifact_role::target,
        0, configured_gpu_duplicate, 3, false }));
    CHECK(configured_gpu_worker->launch());
    server_cache_execution_fingerprint configured_gpu_result;
    bool configured_gpu_delivered = false;
    for (int i = 0; i < 200 &&
             !(configured_gpu_delivered = configured_gpu_worker->poll(
                   configured_gpu_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(configured_gpu_delivered && configured_gpu_result.complete);
    CHECK(configured_gpu_result.config_root == config_root(fields_ab));
    configured_gpu_worker->stop();

#if !defined(_WIN32)
    // Bounded admission closes the rejected descriptor immediately, and the
    // unlaunched worker destructor closes every descriptor it already owns.
    int first_bounded_descriptor = -1;
    {
        auto bounded_worker = std::make_unique<server_cache_fingerprint_worker>();
        CHECK(bounded_worker->configure(
            production_params, production_vbr, 99, 0, 0));
        for (size_t i = 0;
             i < server_cache_fingerprint_worker::descriptor_capacity; ++i) {
            const int descriptor = dup(fileno(file));
            CHECK(descriptor >= 0);
            if (i == 0) first_bounded_descriptor = descriptor;
            CHECK(bounded_worker->add_descriptor({
                server_cache_fingerprint_artifact_role::target,
                uint32_t(i), descriptor, 3, false }));
        }
        const int overflow_descriptor = dup(fileno(file));
        CHECK(overflow_descriptor >= 0);
        CHECK(!bounded_worker->add_descriptor({
            server_cache_fingerprint_artifact_role::target,
            uint32_t(server_cache_fingerprint_worker::descriptor_capacity),
            overflow_descriptor, 3, false }));
        CHECK(fcntl(overflow_descriptor, F_GETFD) == -1 && errno == EBADF);
    }
    CHECK(first_bounded_descriptor >= 0);
    CHECK(fcntl(first_bounded_descriptor, F_GETFD) == -1 && errno == EBADF);
#endif
    {
        auto bounded_artifacts = std::make_unique<server_cache_fingerprint_worker>();
        CHECK(bounded_artifacts->configure(
            production_params, production_vbr, 99, 0, 0));
        for (size_t i = 0;
             i < server_cache_fingerprint_worker::fixed_artifact_capacity; ++i) {
            CHECK(bounded_artifacts->add_fixed_artifact({
                server_cache_fingerprint_artifact_role::target,
                uint32_t(i), 0, {}, false }));
        }
        CHECK(!bounded_artifacts->add_fixed_artifact({
            server_cache_fingerprint_artifact_role::target,
            uint32_t(server_cache_fingerprint_worker::fixed_artifact_capacity),
            0, {}, false }));
    }

    // The worker is all-or-nothing: a short descriptor never publishes a
    // partial root, and artifact ordering is the canonical role/ordinal order.
#if defined(_WIN32)
    const int short_duplicate = _dup(_fileno(file));
#else
    const int short_duplicate = dup(fileno(file));
#endif
    CHECK(short_duplicate >= 0);
    server_cache_fingerprint_worker short_worker;
    CHECK(short_worker.start({ {
        server_cache_fingerprint_artifact_role::target,
        0, short_duplicate, 4, false } }, fields()));
    server_cache_execution_fingerprint short_result;
    bool short_delivered = false;
    for (int i = 0; i < 200 &&
             !(short_delivered = short_worker.poll(short_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(short_delivered);
    CHECK(!short_result.complete);
    short_worker.stop();

#if defined(_WIN32)
    const int target_duplicate = _dup(_fileno(file));
    const int draft_duplicate = _dup(_fileno(file));
#else
    const int target_duplicate = dup(fileno(file));
    const int draft_duplicate = dup(fileno(file));
#endif
    CHECK(target_duplicate >= 0 && draft_duplicate >= 0);
    server_cache_fingerprint_worker ordered_worker;
    CHECK(ordered_worker.start({
        { server_cache_fingerprint_artifact_role::target,
          0, target_duplicate, 3, false },
        { server_cache_fingerprint_artifact_role::draft,
          0, draft_duplicate, 3, false },
    }, fields()));
    server_cache_execution_fingerprint ordered_result;
    bool ordered_delivered = false;
    for (int i = 0; i < 200 &&
             !(ordered_delivered = ordered_worker.poll(ordered_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(ordered_delivered);
    CHECK(ordered_result.complete);
    server_cache_execution_fingerprint expected_ordered;
    CHECK(server_cache_execution_fingerprint_v1({
        { server_cache_fingerprint_artifact_role::target,
          0, 3, abc_sha, false },
        { server_cache_fingerprint_artifact_role::draft,
          0, 3, abc_sha, false },
    }, fields(), expected_ordered));
    CHECK(ordered_result.execution_root == expected_ordered.execution_root);
    ordered_worker.stop();

#if defined(_WIN32)
    const int paused_duplicate = _dup(_fileno(file));
#else
    const int paused_duplicate = dup(fileno(file));
#endif
    CHECK(paused_duplicate >= 0);
    server_cache_fingerprint_worker paused_worker;
    paused_worker.set_scheduler_demand(true);
    CHECK(paused_worker.start({ {
        server_cache_fingerprint_artifact_role::target,
        0, paused_duplicate, 3, false } }, fields()));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    server_cache_execution_fingerprint paused_result;
    CHECK(!paused_worker.poll(paused_result));
    paused_worker.set_scheduler_demand(false);
    bool paused_delivered = false;
    for (int i = 0; i < 200 &&
             !(paused_delivered = paused_worker.poll(paused_result)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(paused_delivered);
    CHECK(paused_result.complete);
    paused_worker.stop();

#if defined(_WIN32)
    const int cancelled_duplicate = _dup(_fileno(file));
#else
    const int cancelled_duplicate = dup(fileno(file));
#endif
    CHECK(cancelled_duplicate >= 0);
    server_cache_fingerprint_worker cancelled_worker;
    cancelled_worker.set_scheduler_demand(true);
    CHECK(cancelled_worker.start({ {
        server_cache_fingerprint_artifact_role::target,
        0, cancelled_duplicate, 3, false } }, fields()));
    cancelled_worker.stop();
    server_cache_execution_fingerprint cancelled_result;
    CHECK(cancelled_worker.poll(cancelled_result));
    CHECK(!cancelled_result.complete);
#if !defined(_WIN32)
    const int rejected_duplicate = dup(fileno(file));
    CHECK(rejected_duplicate >= 0);
    CHECK(!worker.start({ {
        server_cache_fingerprint_artifact_role::target,
        0, rejected_duplicate, 3, false } }, fields()));
    CHECK(fcntl(rejected_duplicate, F_GETFD) == -1 && errno == EBADF);
#endif

    // The arena-owned buffer is slightly smaller than its 1-MiB region so the
    // worker state fits beside it. A full-MiB descriptor therefore proves the
    // same fixed buffer is reused across reads without changing the digest.
    {
        FILE * chunked_file = std::tmpfile();
        CHECK(chunked_file != nullptr);
        std::array<uint8_t, 64 * 1024> block = {};
        for (size_t i = 0; i < block.size(); ++i) {
            block[i] = uint8_t(i * 17 + 3);
        }
        llama_sha256 chunked_hash;
        for (size_t written = 0; written < 1024 * 1024;
             written += block.size()) {
            CHECK(std::fwrite(block.data(), 1, block.size(), chunked_file) ==
                  block.size());
            chunked_hash.update(block.data(), block.size());
        }
        CHECK(std::fflush(chunked_file) == 0);
#if defined(_WIN32)
        const int chunked_duplicate = _dup(_fileno(chunked_file));
#else
        const int chunked_duplicate = dup(fileno(chunked_file));
#endif
        CHECK(chunked_duplicate >= 0);
        server_cache_fingerprint_worker chunked_worker;
        CHECK(chunked_worker.start({ {
            server_cache_fingerprint_artifact_role::target,
            0, chunked_duplicate, 1024 * 1024, false } }, fields()));
        server_cache_execution_fingerprint chunked_result;
        bool chunked_delivered = false;
        for (int i = 0; i < 500 &&
                 !(chunked_delivered = chunked_worker.poll(chunked_result)); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(chunked_delivered && chunked_result.complete);
        server_cache_execution_fingerprint expected_chunked;
        CHECK(server_cache_execution_fingerprint_v1({ {
            server_cache_fingerprint_artifact_role::target,
            0, 1024 * 1024, chunked_hash.finish(), false } },
            fields(), expected_chunked));
        CHECK(chunked_result.execution_root == expected_chunked.execution_root);
        chunked_worker.stop();
        std::fclose(chunked_file);
    }
    std::fclose(file);

    CHECK(!llama_model_artifact_capture_enabled());
    CHECK(!llama_model_artifact_capture_set(true));
    CHECK(llama_model_artifact_capture_enabled());
    CHECK(llama_model_artifact_capture_set(false));
    CHECK(!llama_model_artifact_capture_enabled());

    std::puts("PASS");
    return 0;
}
