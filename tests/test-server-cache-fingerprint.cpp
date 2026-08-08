#include "server-cache-fingerprint.h"
#include "../src/llama-ext.h"
#include "common.h"
#include "common-cache-plan-estimate.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

#define CHECK(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #x); \
        std::abort(); \
    } \
} while (0)

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
    common_cache_plan_vbr_regime production_vbr;
    std::vector<server_cache_fingerprint_field> production_fields;
    CHECK(server_cache_fingerprint_fields_v1(
        production_params, production_vbr, 99, 0, 0, production_fields));
    CHECK(production_fields.size() == 32);
    CHECK(production_fields[10].id == 11);
    CHECK(read_u32(production_fields[10].payload, 6) == 0);

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
    std::fclose(file);

    CHECK(!llama_model_artifact_capture_enabled());
    CHECK(!llama_model_artifact_capture_set(true));
    CHECK(llama_model_artifact_capture_enabled());
    CHECK(llama_model_artifact_capture_set(false));
    CHECK(!llama_model_artifact_capture_enabled());

    std::puts("PASS");
    return 0;
}
