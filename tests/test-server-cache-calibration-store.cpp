#include "server-cache-calibration-store.h"
#include "../src/llama-sha256.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#if defined(_WIN32)
#  include <process.h>
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#define CHECK(x) do { \
    if (!(x)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #x); \
        std::abort(); \
    } \
} while (0)

namespace fs = std::filesystem;

static void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out.push_back(uint8_t(value >> (8 * i)));
}

static std::vector<uint8_t> envelope(const std::string & payload) {
    static constexpr char magic[] = "BUUNCAL1";
    static constexpr char domain[] = "buun-cache-calibration-v1";
    std::vector<uint8_t> out(magic, magic + 8);
    append_u32(out, 1);
    append_u32(out, uint32_t(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    llama_sha256 hash;
    hash.update(domain, sizeof(domain));
    hash.update(out.data() + 8, 8);
    hash.update(payload.data(), payload.size());
    const auto digest = hash.finish();
    out.insert(out.end(), digest.begin(), digest.end());
    return out;
}

static bool sha256_is(const std::vector<uint8_t> & bytes,
                      const char * expected) {
    llama_sha256 hash;
    hash.update(bytes.data(), bytes.size());
    const auto digest = hash.finish();
    static constexpr char digits[] = "0123456789abcdef";
    std::string actual;
    actual.reserve(64);
    for (uint8_t byte : digest) {
        actual.push_back(digits[byte >> 4]);
        actual.push_back(digits[byte & 15]);
    }
    return actual == expected;
}

static server_cache_calibration_profile_snapshot profile(uint8_t identity) {
    server_cache_calibration_profile_snapshot out;
    out.profile_identity_digest[0] = identity;
    out.identity_exact = false;
    out.mutation_generation = 7;
    server_cache_calibration_instance_snapshot instance;
    instance.slot = 3;
    instance.key.operation = server_cache_observation_operation::replay;
    instance.key.provider = common_cache_plan_provider::live_slot;
    instance.key.size_family = 0;
    instance.key.feature_dim = 4;
    instance.key.profile_execution_digest[0] = identity;
    instance.key.participant_execution_digest[0] = 1;
    instance.key.adapter_application_digest[0] = 2;
    instance.key.representation_digest[0] = 3;
    instance.key.adapter_application_complete = true;
    instance.key.identity_complete = true;
    instance.v[0][0] = 2.0;
    instance.v[1][1] = 1.0;
    instance.v[2][2] = 1.0;
    instance.v[3][3] = 1.0;
    instance.b[0] = 50.0;
    instance.n_fit = 1;
    instance.feature_max[0] = 1.0;
    instance.qualified_execution_ordinal = 1;
    CHECK(instance.fit_region_minutes.push_back(10));
    CHECK(instance.fit_region_minutes.push_back(11));
    instance.safe_measurable_opportunities = 1;
    instance.response_reservoir[0] = 50;
    instance.reservoir_seen = 1;
    out.instances.push_back(instance);
    return out;
}

static void write_bytes(const fs::path & path,
                        const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    CHECK(output.good());
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 std::streamsize(bytes.size()));
    CHECK(output.good());
    output.close();
#if !defined(_WIN32)
    fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace);
#endif
}

static server_cache_observation_record accepted_record(
        const server_cache_execution_fingerprint & fingerprint) {
    server_cache_observation_record out;
    out.key.operation = server_cache_observation_operation::replay;
    out.key.provider = common_cache_plan_provider::live_slot;
    out.key.feature_dim = 4;
    out.key.adapter_application_complete = true;
    out.key.identity_complete = true;
    out.key.profile_execution_digest = fingerprint.execution_root;
    out.key.participant_execution_digest[0] = 0x31;
    out.key.adapter_application_digest[0] = 0x32;
    out.key.representation_digest[0] = 0x33;
    out.feature = { 0.0, 0.0, 0.0, 1.0 };
    out.owned_cpu_us = 10;
    out.backend_service_us = 90;
    out.owned_service_us = 100;
    out.capped_service_us = 100;
    out.terminal = server_cache_observation_terminal::accepted;
    return out;
}

int main() {
    server_cache_calibration_manifest manifest;
    manifest.store_lineage_id[0] = 0xab;
    manifest.next_boot_claim_ordinal = 0;
    manifest.next_profile_generation_ordinal = 0;
    manifest.next_persisted_prune_epoch = 0;
    manifest.next_immutable_file_ordinal = 0;
    std::vector<uint8_t> encoded;
    CHECK(server_cache_calibration_encode_manifest(manifest, encoded));
    CHECK(sha256_is(encoded,
        "0b2cdff19d7f99a6598fcea8a264e059ba58dfed63b534c939ed5f351dd8bb11"));
    server_cache_calibration_manifest decoded;
    CHECK(server_cache_calibration_decode_manifest(
        encoded.data(), encoded.size(), decoded));
    CHECK(decoded.store_lineage_id == manifest.store_lineage_id);
    const auto encoded_manifest = encoded;

    auto reused_manifest = manifest;
    reused_manifest.next_profile_generation_ordinal = 2;
    reused_manifest.next_immutable_file_ordinal = 2;
    reused_manifest.next_persisted_prune_epoch = 2;
    server_cache_calibration_profile_reference reused_a;
    reused_a.profile_generation_ordinal = 0;
    reused_a.profile_file_generation = 0;
    reused_a.persisted_prune_recency = 0;
    reused_a.profile_identity_digest[0] = 1;
    reused_a.profile_payload_digest[0] = 1;
    auto reused_b = reused_a;
    reused_b.profile_generation_ordinal = 1;
    reused_b.profile_identity_digest[0] = 2;
    CHECK(reused_manifest.profiles.push_back(reused_a));
    CHECK(reused_manifest.profiles.push_back(reused_b));
    CHECK(!server_cache_calibration_encode_manifest(reused_manifest, encoded));
    reused_b.profile_file_generation = 1;
    reused_b.persisted_prune_recency = 0;
    reused_manifest.profiles[1] = reused_b;
    CHECK(!server_cache_calibration_encode_manifest(reused_manifest, encoded));
    encoded = encoded_manifest;

    const uint32_t payload_size = uint32_t(encoded[12]) |
        uint32_t(encoded[13]) << 8 | uint32_t(encoded[14]) << 16 |
        uint32_t(encoded[15]) << 24;
    std::string payload(encoded.begin() + 16,
                        encoded.begin() + 16 + payload_size);
    payload.insert(1, "\"generation\":0,");
    auto duplicate_key = envelope(payload);
    CHECK(!server_cache_calibration_decode_manifest(
        duplicate_key.data(), duplicate_key.size(), decoded));

    auto corrupted = encoded;
    corrupted[20] ^= 1;
    CHECK(!server_cache_calibration_decode_manifest(
        corrupted.data(), corrupted.size(), decoded));
    CHECK(!server_cache_calibration_decode_manifest(
        encoded.data(), encoded.size() - 1, decoded));
    std::vector<uint8_t> oversized_root(
        "BUUNCAL1", "BUUNCAL1" + 8);
    append_u32(oversized_root, 1);
    append_u32(oversized_root, 64 * 1024 + 1);
    oversized_root.resize(8 + 4 + 4 + 64 * 1024 + 1 + 32);
    CHECK(!server_cache_calibration_decode_manifest(
        oversized_root.data(), oversized_root.size(), decoded));

    auto signed_payload = payload;
    const auto generation_at = signed_payload.find("\"generation\":0");
    CHECK(generation_at != std::string::npos);
    signed_payload.replace(generation_at, std::strlen("\"generation\":0"),
                           "\"generation\":-1");
    auto signed_counter = envelope(signed_payload);
    CHECK(!server_cache_calibration_decode_manifest(
        signed_counter.data(), signed_counter.size(), decoded));
    auto floating_payload = payload;
    floating_payload.replace(generation_at, std::strlen("\"generation\":0"),
                             "\"generation\":0.5");
    auto floating_counter = envelope(floating_payload);
    CHECK(!server_cache_calibration_decode_manifest(
        floating_counter.data(), floating_counter.size(), decoded));

    auto nested_duplicate_payload = payload;
    const auto update_at = nested_duplicate_payload.find(
        "\"last_update_unix_ms\"");
    CHECK(update_at != std::string::npos);
    nested_duplicate_payload.insert(update_at,
        "\"nested\":{\"x\":1},\"generation\":0,");
    auto nested_duplicate = envelope(nested_duplicate_payload);
    CHECK(!server_cache_calibration_decode_manifest(
        nested_duplicate.data(), nested_duplicate.size(), decoded));

    auto snapshot = profile(0x42);
    CHECK(server_cache_calibration_encode_profile(
        manifest.store_lineage_id, snapshot, encoded));
    CHECK(sha256_is(encoded,
        "a92f11418d31cb60c14353d65fdec12c9872fabba01f3944a997fd42f013dd83"));
    server_cache_calibration_profile_snapshot decoded_profile;
    CHECK(server_cache_calibration_decode_profile(
        encoded.data(), encoded.size(), manifest.store_lineage_id,
        decoded_profile));
    CHECK(decoded_profile.profile_identity_digest ==
          snapshot.profile_identity_digest);
    CHECK(decoded_profile.instances.size() == 1);
    CHECK(decoded_profile.instances[0].v[0][0] == 2.0);
    CHECK(decoded_profile.instances[0].qualified_execution_ordinal == 1);
    CHECK(decoded_profile.instances[0].fit_region_minutes.size() == 2);

    // Every persisted authority terminal and all future ZC4 sufficient-state
    // carriers survive the live observer seam; ZC3 never consumes them.
    for (auto terminal : {
             server_cache_calibration_authority_terminal::none,
             server_cache_calibration_authority_terminal::tail_exceeded,
             server_cache_calibration_authority_terminal::confidence_budget_exhausted,
             server_cache_calibration_authority_terminal::ordinal_exhausted,
             server_cache_calibration_authority_terminal::numeric_fault }) {
        auto terminal_profile = profile(0x43);
        auto & instance = terminal_profile.instances.front();
        instance.fit_generation = 9;
        instance.authority_terminal = terminal;
        instance.tail_actual_max_us = terminal ==
                server_cache_calibration_authority_terminal::tail_exceeded
            ? 2000001 : 0;
        instance.log_wealth[2] = -0.25;
        instance.n_validation = 3;
        CHECK(instance.validation_region_minutes.push_back(12));
        instance.safe_measurable_opportunities = 7;
        instance.opportunity_at_last_validation = 6;
        instance.last_fit_unix_ms = 100;
        instance.last_validation_unix_ms = 101;
        CHECK(server_cache_calibration_encode_profile(
            manifest.store_lineage_id, terminal_profile, encoded));
        CHECK(server_cache_calibration_decode_profile(
            encoded.data(), encoded.size(), manifest.store_lineage_id,
            decoded_profile));
        server_cache_observation_store terminal_store;
        server_cache_execution_fingerprint terminal_fingerprint;
        terminal_fingerprint.complete = true;
        terminal_fingerprint.execution_root =
            terminal_profile.profile_identity_digest;
        terminal_store.set_execution_fingerprint(terminal_fingerprint);
        CHECK(server_cache_calibration_restore_observer(
            decoded_profile, terminal_store));
        server_cache_calibration_profile_snapshot terminal_roundtrip;
        CHECK(server_cache_calibration_snapshot_observer(
            terminal_store, terminal_roundtrip));
        const auto & roundtrip = terminal_roundtrip.instances.front();
        CHECK(roundtrip.fit_generation == 9);
        CHECK(roundtrip.authority_terminal == terminal);
        CHECK(roundtrip.log_wealth[2] == -0.25);
        CHECK(roundtrip.n_validation == 3);
        CHECK(roundtrip.validation_region_minutes.size() == 1);
        CHECK(roundtrip.safe_measurable_opportunities == 7);
        CHECK(roundtrip.opportunity_at_last_validation == 6);
        CHECK(roundtrip.last_fit_unix_ms == 100);
        CHECK(roundtrip.last_validation_unix_ms == 101);
    }

    // Closed key semantics are shared by live admission and persistence.
    for (int mutation = 0; mutation < 8; ++mutation) {
        auto invalid_key = profile(0x44);
        auto & key = invalid_key.instances.front().key;
        switch (mutation) {
            case 0: key.restore_kind = 5; break;
            case 1: key.prepare_shape = 5; break;
            case 2: key.contention_bucket = 2; break;
            case 3: key.start_bucket = 4; break;
            case 4: key.batch_bucket = 4; break;
            case 5: key.ubatch_bucket = 4; break;
            case 6: key.participant_execution_digest = {}; break;
            case 7:
                key.operation =
                    server_cache_observation_operation::destruction_apply;
                key.effect_action_shape_digest = {};
                break;
        }
        CHECK(!server_cache_calibration_validate_profile(invalid_key));
    }

    auto maximal = profile(0x55);
    maximal.profile_generation_ordinal = UINT64_MAX;
    maximal.profile_file_generation = UINT64_MAX;
    maximal.persisted_prune_recency = UINT64_MAX;
    maximal.mutation_generation = UINT64_MAX - 1;
    maximal.profile_last_update_unix_ms = UINT64_MAX;
    maximal.instances.clear();
    for (uint32_t i = 0;
         i < server_cache_observation_store::instance_capacity; ++i) {
        auto instance = profile(uint8_t(i)).instances.front();
        instance.slot = i;
        instance.key.profile_execution_digest =
            maximal.profile_identity_digest;
        instance.key.participant_execution_digest[0] = uint8_t(i + 1);
        instance.fit_generation = UINT64_MAX;
        instance.n_fit = UINT64_MAX;
        instance.feature_min.fill(std::numeric_limits<double>::max());
        instance.feature_max.fill(std::numeric_limits<double>::max());
        instance.qualified_execution_ordinal = UINT64_MAX;
        instance.log_wealth.fill(std::numeric_limits<double>::max());
        instance.n_validation = UINT64_MAX;
        instance.safe_measurable_opportunities = UINT64_MAX;
        instance.opportunity_at_last_validation = UINT64_MAX;
        instance.last_fit_unix_ms = UINT64_MAX;
        instance.last_validation_unix_ms = UINT64_MAX;
        instance.response_reservoir.fill(UINT64_MAX);
        instance.reservoir_seen = UINT64_MAX;
        instance.fit_region_minutes.clear();
        instance.validation_region_minutes.clear();
        for (uint64_t minute = UINT64_MAX - 7;; ++minute) {
            CHECK(instance.fit_region_minutes.push_back(minute));
            CHECK(instance.validation_region_minutes.push_back(minute));
            if (minute == UINT64_MAX) break;
        }
        CHECK(maximal.instances.push_back(instance));
    }
    CHECK(server_cache_calibration_encode_profile(
        manifest.store_lineage_id, maximal, encoded));
    CHECK(encoded.size() <= 1024 * 1024 + 48);

    auto negative_zero = encoded;
    const std::string needle = "0000000000000000";
    auto found = std::search(negative_zero.begin(), negative_zero.end(),
        needle.begin(), needle.end());
    CHECK(found != negative_zero.end());
    *found = '8';
    CHECK(!server_cache_calibration_decode_profile(
        negative_zero.data(), negative_zero.size(), manifest.store_lineage_id,
        decoded_profile));
    std::vector<uint8_t> oversized_profile(
        "BUUNCAL1", "BUUNCAL1" + 8);
    append_u32(oversized_profile, 1);
    append_u32(oversized_profile, 1024 * 1024 + 1);
    oversized_profile.resize(8 + 4 + 4 + 1024 * 1024 + 1 + 32);
    CHECK(!server_cache_calibration_decode_profile(
        oversized_profile.data(), oversized_profile.size(),
        manifest.store_lineage_id, decoded_profile));
    auto nonfinite = snapshot;
    nonfinite.instances.front().b[0] =
        std::numeric_limits<double>::infinity();
    CHECK(!server_cache_calibration_encode_profile(
        manifest.store_lineage_id, nonfinite, encoded));
    auto saturated_mutation = snapshot;
    saturated_mutation.mutation_generation = UINT64_MAX;
    CHECK(!server_cache_calibration_validate_profile(saturated_mutation));

#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    const fs::path directory = fs::temp_directory_path() /
        ("buun-zc3b-" + std::to_string(pid));
    std::error_code ec;
    fs::remove_all(directory, ec);
    server_cache_calibration_store store;
    CHECK(store.open(directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(store.boot_claim_ordinal() == 0);
    server_cache_calibration_store contending;
    CHECK(contending.open(directory.string()) ==
          server_cache_calibration_load_status::busy);
    CHECK(store.commit_profile(profile(0x42)) ==
          server_cache_calibration_load_status::ok);
    auto loaded_owner = std::make_unique<server_cache_calibration_bounded_array<
        server_cache_calibration_profile_snapshot, 16, uint8_t>>();
    auto & loaded = *loaded_owner;
    CHECK(store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(loaded.size() == 1);
    const auto first_q = loaded[0].profile_generation_ordinal;
    const auto first_file = loaded[0].profile_file_generation;
    store.close();

    // Crash branch: the allocator root was durably advanced, then an
    // immutable profile file was fsynced before the referencing manifest.
    // Restart must skip that orphaned file ordinal rather than collide/reuse.
    const fs::path manifest_path = directory / "manifest.bcal";
    std::ifstream manifest_input(manifest_path, std::ios::binary);
    std::vector<uint8_t> manifest_bytes(
        std::istreambuf_iterator<char>(manifest_input), {});
    server_cache_calibration_manifest crash_manifest;
    CHECK(server_cache_calibration_decode_manifest(
        manifest_bytes.data(), manifest_bytes.size(), crash_manifest));
    const uint64_t orphan_file = crash_manifest.next_immutable_file_ordinal++;
    CHECK(server_cache_calibration_encode_manifest(
        crash_manifest, manifest_bytes));
    std::ofstream manifest_output(
        manifest_path, std::ios::binary | std::ios::trunc);
    manifest_output.write(reinterpret_cast<const char *>(manifest_bytes.data()),
                          std::streamsize(manifest_bytes.size()));
    manifest_output.close();
    auto orphan_profile = profile(0x42);
    orphan_profile.profile_generation_ordinal = first_q;
    orphan_profile.profile_file_generation = orphan_file;
    orphan_profile.persisted_prune_recency =
        crash_manifest.next_persisted_prune_epoch;
    std::vector<uint8_t> orphan_bytes;
    CHECK(server_cache_calibration_encode_profile(
        crash_manifest.store_lineage_id, orphan_profile, orphan_bytes));
    std::ofstream orphan(directory /
        ("profile-" + std::to_string(first_q) + "-" +
         std::to_string(orphan_file) + ".bcal"), std::ios::binary);
    orphan.write(reinterpret_cast<const char *>(orphan_bytes.data()),
                 std::streamsize(orphan_bytes.size()));
    orphan.close();
#if !defined(_WIN32)
    fs::permissions(directory /
        ("profile-" + std::to_string(first_q) + "-" +
         std::to_string(orphan_file) + ".bcal"),
        fs::perms::owner_read | fs::perms::owner_write,
        fs::perm_options::replace);
#endif

    CHECK(store.open(directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(store.boot_claim_ordinal() == 1);
    CHECK(!fs::exists(directory /
        ("profile-" + std::to_string(first_q) + "-" +
         std::to_string(orphan_file) + ".bcal")));
    CHECK(store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(loaded.size() == 1);
    CHECK(loaded[0].profile_generation_ordinal == first_q);
    CHECK(loaded[0].profile_file_generation == first_file);
    auto updated_profile = profile(0x42);
    updated_profile.mutation_generation = 8;
    CHECK(store.commit_profile(updated_profile) ==
          server_cache_calibration_load_status::ok);
    CHECK(store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(loaded[0].profile_generation_ordinal == first_q);
    CHECK(loaded[0].profile_file_generation > orphan_file);
    CHECK(!fs::exists(directory /
        ("profile-" + std::to_string(first_q) + "-" +
         std::to_string(first_file) + ".bcal")));
    auto restored_profile =
        std::make_unique<server_cache_calibration_profile_snapshot>(loaded[0]);
    store.close();

    // Same-writer regression refuses without replacing the referenced image.
    const fs::path regression_directory = fs::temp_directory_path() /
        ("buun-zc3b-regression-" + std::to_string(pid));
    fs::remove_all(regression_directory, ec);
    fs::copy(directory, regression_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    server_cache_calibration_store regression_store;
    CHECK(regression_store.open(regression_directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(regression_store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    auto regressed = profile(0x42);
    regressed.mutation_generation = loaded.front().mutation_generation + 1;
    regressed.instances.front().n_fit = 0;
    CHECK(regression_store.commit_profile(regressed) ==
          server_cache_calibration_load_status::corrupt);
    CHECK(regression_store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(loaded.front().instances.front().n_fit == 1);
    auto next_generation = loaded.front();
    ++next_generation.mutation_generation;
    ++next_generation.instances.front().fit_generation;
    next_generation.instances.front().key.participant_execution_digest[0] ^= 0x7f;
    next_generation.instances.front().n_fit = 0;
    next_generation.instances.front().qualified_execution_ordinal = 0;
    next_generation.instances.front().v = {};
    for (uint8_t i = 0;
         i < next_generation.instances.front().key.feature_dim; ++i) {
        next_generation.instances.front().v[i][i] = 1.0;
    }
    next_generation.instances.front().b = {};
    next_generation.instances.front().feature_min = {};
    next_generation.instances.front().feature_max = {};
    next_generation.instances.front().fit_region_minutes.clear();
    next_generation.instances.front().response_reservoir = {};
    next_generation.instances.front().reservoir_seen = 0;
    CHECK(regression_store.commit_profile(next_generation) ==
          server_cache_calibration_load_status::ok);
    auto reused_generation = next_generation;
    ++reused_generation.mutation_generation;
    --reused_generation.instances.front().fit_generation;
    CHECK(regression_store.commit_profile(reused_generation) ==
          server_cache_calibration_load_status::corrupt);
    regression_store.close();

    // The bounded disk set deterministically prunes the oldest persisted use.
    const fs::path prune_directory = fs::temp_directory_path() /
        ("buun-zc3b-prune-" + std::to_string(pid));
    fs::remove_all(prune_directory, ec);
    server_cache_calibration_store prune_store;
    CHECK(prune_store.open(prune_directory.string()) ==
          server_cache_calibration_load_status::ok);
    for (uint8_t identity = 1; identity <= 17; ++identity) {
        CHECK(prune_store.commit_profile(profile(identity)) ==
              server_cache_calibration_load_status::ok);
    }
    CHECK(prune_store.load_profiles(loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(loaded.size() == 16);
    CHECK(std::none_of(loaded.begin(), loaded.end(), [](const auto & value) {
        return value.profile_identity_digest[0] == 1;
    }));
    prune_store.close();

    const fs::path manifest_fault_directory = fs::temp_directory_path() /
        ("buun-zc3b-manifest-fault-" + std::to_string(pid));
    fs::remove_all(manifest_fault_directory, ec);
    server_cache_calibration_store manifest_fault_store;
    CHECK(manifest_fault_store.open(manifest_fault_directory.string()) ==
          server_cache_calibration_load_status::ok);
    std::ifstream before_fault_input(
        manifest_fault_directory / "manifest.bcal", std::ios::binary);
    const std::vector<uint8_t> before_fault(
        std::istreambuf_iterator<char>(before_fault_input), {});
    server_cache_calibration_set_test_fault(
        server_cache_calibration_test_fault::manifest_replace_once);
    CHECK(manifest_fault_store.commit_profile(profile(0xa1)) ==
          server_cache_calibration_load_status::io_fault);
    std::ifstream after_fault_input(
        manifest_fault_directory / "manifest.bcal", std::ios::binary);
    const std::vector<uint8_t> after_fault(
        std::istreambuf_iterator<char>(after_fault_input), {});
    CHECK(after_fault == before_fault);
    manifest_fault_store.close();

    // The second manifest seam is after the immutable payload fsync. Its
    // injected crash leaves a valid unreferenced orphan which restart must
    // authenticate/remove without ever loading or reusing its ordinal.
    const fs::path reference_fault_directory = fs::temp_directory_path() /
        ("buun-zc3b-reference-fault-" + std::to_string(pid));
    fs::remove_all(reference_fault_directory, ec);
    server_cache_calibration_store reference_fault_store;
    CHECK(reference_fault_store.open(reference_fault_directory.string()) ==
          server_cache_calibration_load_status::ok);
    server_cache_calibration_set_test_fault(
        server_cache_calibration_test_fault::referencing_manifest_replace_once);
    CHECK(reference_fault_store.commit_profile(profile(0xa2)) ==
          server_cache_calibration_load_status::io_fault);
    const uint64_t reserved_orphan =
        reference_fault_store.manifest().next_immutable_file_ordinal - 1;
    CHECK(fs::exists(reference_fault_directory /
        ("profile-0-" + std::to_string(reserved_orphan) + ".bcal")));
    reference_fault_store.close();
    CHECK(reference_fault_store.open(reference_fault_directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(!fs::exists(reference_fault_directory /
        ("profile-0-" + std::to_string(reserved_orphan) + ".bcal")));
    CHECK(reference_fault_store.commit_profile(profile(0xa2)) ==
          server_cache_calibration_load_status::ok);
    CHECK(reference_fault_store.manifest().profiles.front().
              profile_file_generation > reserved_orphan);
    reference_fault_store.close();

    // A valid but unreferenced replacement envelope cannot become the
    // non-regression baseline for an update.
    const fs::path swapped_directory = fs::temp_directory_path() /
        ("buun-zc3b-swapped-" + std::to_string(pid));
    fs::remove_all(swapped_directory, ec);
    fs::copy(directory, swapped_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    std::ifstream swapped_manifest_input(
        swapped_directory / "manifest.bcal", std::ios::binary);
    std::vector<uint8_t> swapped_manifest_bytes(
        std::istreambuf_iterator<char>(swapped_manifest_input), {});
    server_cache_calibration_manifest swapped_manifest;
    CHECK(server_cache_calibration_decode_manifest(
        swapped_manifest_bytes.data(), swapped_manifest_bytes.size(),
        swapped_manifest));
    CHECK(swapped_manifest.profiles.size() == 1);
    const auto swapped_ref = swapped_manifest.profiles.front();
    auto swapped_profile = profile(0x42);
    swapped_profile.profile_generation_ordinal =
        swapped_ref.profile_generation_ordinal;
    swapped_profile.profile_file_generation =
        swapped_ref.profile_file_generation;
    swapped_profile.persisted_prune_recency =
        swapped_ref.persisted_prune_recency;
    swapped_profile.mutation_generation = 99;
    std::vector<uint8_t> swapped_profile_bytes;
    CHECK(server_cache_calibration_encode_profile(
        swapped_manifest.store_lineage_id, swapped_profile,
        swapped_profile_bytes));
    write_bytes(swapped_directory /
        ("profile-" + std::to_string(swapped_ref.profile_generation_ordinal) +
         "-" + std::to_string(swapped_ref.profile_file_generation) + ".bcal"),
        swapped_profile_bytes);
    server_cache_calibration_store swapped_store;
    CHECK(swapped_store.open(swapped_directory.string()) ==
          server_cache_calibration_load_status::ok);
    auto swapped_update = profile(0x42);
    swapped_update.mutation_generation = 100;
    CHECK(swapped_store.commit_profile(swapped_update) ==
          server_cache_calibration_load_status::corrupt);
    swapped_store.close();

    const fs::path exhausted_directory = fs::temp_directory_path() /
        ("buun-zc3b-exhausted-" + std::to_string(pid));
    fs::remove_all(exhausted_directory, ec);
    fs::copy(directory, exhausted_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    std::ifstream exhausted_input(
        exhausted_directory / "manifest.bcal", std::ios::binary);
    std::vector<uint8_t> exhausted_bytes(
        std::istreambuf_iterator<char>(exhausted_input), {});
    server_cache_calibration_manifest exhausted_manifest;
    CHECK(server_cache_calibration_decode_manifest(
        exhausted_bytes.data(), exhausted_bytes.size(), exhausted_manifest));
    exhausted_manifest.next_immutable_file_ordinal = UINT64_MAX;
    CHECK(server_cache_calibration_encode_manifest(
        exhausted_manifest, exhausted_bytes));
    write_bytes(exhausted_directory / "manifest.bcal", exhausted_bytes);
    server_cache_calibration_store exhausted_store;
    CHECK(exhausted_store.open(exhausted_directory.string()) ==
          server_cache_calibration_load_status::ok);
    auto exhausted_update = profile(0x42);
    exhausted_update.mutation_generation = 100;
    CHECK(exhausted_store.commit_profile(exhausted_update) ==
          server_cache_calibration_load_status::ordinal_exhausted);
    exhausted_store.close();

    const fs::path root_exhausted_directory = fs::temp_directory_path() /
        ("buun-zc3b-root-exhausted-" + std::to_string(pid));
    fs::remove_all(root_exhausted_directory, ec);
    fs::copy(directory, root_exhausted_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    std::ifstream root_exhausted_input(
        root_exhausted_directory / "manifest.bcal", std::ios::binary);
    std::vector<uint8_t> root_exhausted_bytes(
        std::istreambuf_iterator<char>(root_exhausted_input), {});
    server_cache_calibration_manifest root_exhausted_manifest;
    CHECK(server_cache_calibration_decode_manifest(
        root_exhausted_bytes.data(), root_exhausted_bytes.size(),
        root_exhausted_manifest));
    root_exhausted_manifest.generation = UINT64_MAX;
    CHECK(server_cache_calibration_encode_manifest(
        root_exhausted_manifest, root_exhausted_bytes));
    write_bytes(root_exhausted_directory / "manifest.bcal",
                root_exhausted_bytes);
    server_cache_calibration_store root_exhausted_store;
    CHECK(root_exhausted_store.open(root_exhausted_directory.string()) ==
          server_cache_calibration_load_status::ordinal_exhausted);

    const fs::path root_budget_directory = fs::temp_directory_path() /
        ("buun-zc3b-root-budget-" + std::to_string(pid));
    fs::remove_all(root_budget_directory, ec);
    fs::copy(directory, root_budget_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    root_exhausted_manifest.generation = UINT64_MAX - 2;
    CHECK(server_cache_calibration_encode_manifest(
        root_exhausted_manifest, root_exhausted_bytes));
    write_bytes(root_budget_directory / "manifest.bcal",
                root_exhausted_bytes);
    server_cache_calibration_store root_budget_store;
    CHECK(root_budget_store.open(root_budget_directory.string()) ==
          server_cache_calibration_load_status::ok);
    auto root_budget_update = profile(0x42);
    root_budget_update.mutation_generation = 100;
    CHECK(root_budget_store.commit_profile(root_budget_update) ==
          server_cache_calibration_load_status::ordinal_exhausted);
    root_budget_store.close();

#if !defined(_WIN32)
    // The dedicated directory capability rejects aliases, links, and
    // preexisting state without a manifest rather than following/deleting it.
    const fs::path missing_root = fs::temp_directory_path() /
        ("buun-zc3b-missing-root-" + std::to_string(pid));
    fs::remove_all(missing_root, ec);
    fs::create_directory(missing_root);
    fs::permissions(missing_root, fs::perms::owner_all,
                    fs::perm_options::replace);
    write_bytes(missing_root / "profile-0-0.bcal", { 1, 2, 3 });
    server_cache_calibration_store missing_store;
    CHECK(missing_store.open(missing_root.string()) ==
          server_cache_calibration_load_status::corrupt);

    const fs::path malformed_orphan_directory = fs::temp_directory_path() /
        ("buun-zc3b-malformed-orphan-" + std::to_string(pid));
    fs::remove_all(malformed_orphan_directory, ec);
    fs::copy(directory, malformed_orphan_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    write_bytes(malformed_orphan_directory / "profile-999-999.bcal",
                { 1, 2, 3 });
    server_cache_calibration_store malformed_orphan_store;
    CHECK(malformed_orphan_store.open(malformed_orphan_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    const fs::path child_symlink_directory = fs::temp_directory_path() /
        ("buun-zc3b-child-symlink-" + std::to_string(pid));
    fs::remove_all(child_symlink_directory, ec);
    fs::copy(directory, child_symlink_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    fs::create_symlink(child_symlink_directory / "manifest.bcal",
                       child_symlink_directory / "profile-999-999.bcal", ec);
    CHECK(!ec);
    server_cache_calibration_store child_symlink_store;
    CHECK(child_symlink_store.open(child_symlink_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    const fs::path linked_directory = fs::temp_directory_path() /
        ("buun-zc3b-hardlink-" + std::to_string(pid));
    fs::remove_all(linked_directory, ec);
    fs::remove_all(malformed_orphan_directory, ec);
    fs::remove_all(child_symlink_directory, ec);
    fs::copy(directory, linked_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    const auto linked_profile_path = linked_directory /
        ("profile-" + std::to_string(swapped_ref.profile_generation_ordinal) +
         "-" + std::to_string(swapped_ref.profile_file_generation) + ".bcal");
    fs::create_hard_link(linked_profile_path,
                         linked_directory / "profile-hardlink-alias", ec);
    CHECK(!ec);
    server_cache_calibration_store linked_store;
    CHECK(linked_store.open(linked_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    const fs::path symlink_target = fs::temp_directory_path() /
        ("buun-zc3b-symlink-target-" + std::to_string(pid));
    const fs::path symlink_directory = fs::temp_directory_path() /
        ("buun-zc3b-symlink-" + std::to_string(pid));
    fs::remove_all(symlink_target, ec);
    fs::remove(symlink_directory, ec);
    fs::create_directory(symlink_target);
    fs::create_directory_symlink(symlink_target, symlink_directory, ec);
    CHECK(!ec);
    server_cache_calibration_store symlink_store;
    CHECK(symlink_store.open(symlink_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    const fs::path open_mode_directory = fs::temp_directory_path() /
        ("buun-zc3b-open-mode-" + std::to_string(pid));
    fs::remove_all(open_mode_directory, ec);
    fs::create_directory(open_mode_directory);
    fs::permissions(open_mode_directory, fs::perms::all,
                    fs::perm_options::replace);
    server_cache_calibration_store open_mode_store;
    CHECK(open_mode_store.open(open_mode_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    const fs::path readable_mode_directory = fs::temp_directory_path() /
        ("buun-zc3b-readable-mode-" + std::to_string(pid));
    fs::remove_all(readable_mode_directory, ec);
    fs::create_directory(readable_mode_directory);
    fs::permissions(readable_mode_directory,
                    fs::perms::owner_all | fs::perms::group_read |
                        fs::perms::others_read,
                    fs::perm_options::replace);
    server_cache_calibration_store readable_mode_store;
    CHECK(readable_mode_store.open(readable_mode_directory.string()) ==
          server_cache_calibration_load_status::io_fault);

    // Creation is independent of a restrictive caller umask: every created
    // child is explicitly fchmod'd and revalidated before it becomes durable.
    const fs::path umask_directory = fs::temp_directory_path() /
        ("buun-zc3b-umask-" + std::to_string(pid));
    fs::remove_all(umask_directory, ec);
    const mode_t prior_umask = umask(0777);
    server_cache_calibration_store umask_store;
    CHECK(umask_store.open(umask_directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(umask_store.commit_profile(profile(0x72)) ==
          server_cache_calibration_load_status::ok);
    umask_store.close();
    umask(prior_umask);
    CHECK(umask_store.open(umask_directory.string()) ==
          server_cache_calibration_load_status::ok);
    umask_store.close();
#endif

    server_cache_observation_store observer;
    server_cache_execution_fingerprint fingerprint;
    fingerprint.complete = true;
    fingerprint.execution_root[0] = 0x42;
    observer.set_execution_fingerprint(fingerprint);
    CHECK(server_cache_calibration_restore_observer(*restored_profile, observer));
    CHECK(observer.instances()[3].used);
    CHECK(observer.instances()[3].b[0] == 50.0);
    server_cache_calibration_profile_snapshot observer_snapshot;
    CHECK(server_cache_calibration_snapshot_observer(
        observer, observer_snapshot));
    CHECK(observer_snapshot.mutation_generation ==
          restored_profile->mutation_generation);

    const fs::path async_directory = fs::temp_directory_path() /
        ("buun-zc3b-async-" + std::to_string(pid));
    fs::remove_all(async_directory, ec);
    auto writer_owner = std::make_unique<server_cache_calibration_writer>();
    auto & writer = *writer_owner;
    CHECK(writer.start(async_directory.string()));
    server_cache_calibration_load_status writer_status;
    auto writer_loaded_owner = std::make_unique<server_cache_calibration_bounded_array<
        server_cache_calibration_profile_snapshot, 16, uint8_t>>();
    auto & writer_loaded = *writer_loaded_owner;
    bool load_delivered = false;
    for (int i = 0; i < 500 &&
             !(load_delivered = writer.poll_loaded(
                 writer_status, writer_loaded)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(load_delivered);
    CHECK(writer_status == server_cache_calibration_load_status::ok);
    auto async_profile = profile(0x77);
    CHECK(writer.enqueue(async_profile));
    server_cache_calibration_commit_ack async_ack;
    bool acked = false;
    for (int i = 0; i < 500 &&
             !(acked = writer.poll_committed(async_ack)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(acked);
    CHECK(async_ack.profile_identity_digest ==
          async_profile.profile_identity_digest);
    CHECK(async_ack.mutation_generation == async_profile.mutation_generation);
    auto async_newer = async_profile;
    async_newer.mutation_generation = async_profile.mutation_generation + 1;
    auto async_latest = async_newer;
    async_latest.mutation_generation = async_newer.mutation_generation + 1;
    CHECK(writer.enqueue(async_newer));
    CHECK(writer.enqueue(async_latest));
    CHECK(!writer.enqueue(async_newer));
    bool latest_acked = false;
    for (int i = 0; i < 1000 && !latest_acked; ++i) {
        server_cache_calibration_commit_ack candidate;
        if (writer.poll_committed(candidate) &&
            candidate.mutation_generation ==
                async_latest.mutation_generation) {
            latest_acked = true;
        }
        if (!latest_acked) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    CHECK(latest_acked);
    auto second_async_profile = profile(0x78);
    second_async_profile.mutation_generation = 1;
    CHECK(writer.enqueue(second_async_profile));
    bool second_acked = false;
    for (int i = 0; i < 1000 && !second_acked; ++i) {
        server_cache_calibration_commit_ack candidate;
        if (writer.poll_committed(candidate) &&
            candidate.profile_identity_digest ==
                second_async_profile.profile_identity_digest) {
            CHECK(candidate.mutation_generation == 1);
            second_acked = true;
        }
        if (!second_acked) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    CHECK(second_acked);
    writer.stop();
    server_cache_calibration_store async_store;
    CHECK(async_store.open(async_directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(async_store.load_profiles(writer_loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(writer_loaded.size() == 2);
    const auto async_loaded = std::find_if(
        writer_loaded.begin(), writer_loaded.end(), [&](const auto & value) {
            return value.profile_identity_digest ==
                async_latest.profile_identity_digest;
        });
    CHECK(async_loaded != writer_loaded.end());
    CHECK(async_loaded->mutation_generation ==
          async_latest.mutation_generation);
    async_store.close();

    // A failure after durable ordinal reservation but before immutable-file
    // creation retries the same dirty evidence without requiring a new row.
    auto retry_writer_owner =
        std::make_unique<server_cache_calibration_writer>();
    auto & retry_writer = *retry_writer_owner;
    CHECK(retry_writer.start(async_directory.string()));
    bool retry_loaded = false;
    for (int i = 0; i < 500 &&
             !(retry_loaded = retry_writer.poll_loaded(
                 writer_status, writer_loaded)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(retry_loaded);
    CHECK(writer_status == server_cache_calibration_load_status::ok);
    const auto retry_loaded_profile = std::find_if(
        writer_loaded.begin(), writer_loaded.end(), [&](const auto & value) {
            return value.profile_identity_digest ==
                async_latest.profile_identity_digest;
        });
    CHECK(retry_loaded_profile != writer_loaded.end());
    auto retry_profile = *retry_loaded_profile;
    ++retry_profile.mutation_generation;
    const uint64_t hits_before = server_cache_calibration_test_fault_hits();
    server_cache_calibration_set_test_fault(
        server_cache_calibration_test_fault::profile_write_once);
    CHECK(retry_writer.enqueue(retry_profile));
    bool retry_acked = false;
    for (int i = 0; i < 2000 &&
             !(retry_acked = retry_writer.poll_committed(async_ack)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(retry_acked);
    CHECK(async_ack.mutation_generation == retry_profile.mutation_generation);
    CHECK(server_cache_calibration_test_fault_hits() == hits_before + 1);
    retry_writer.stop();

    const fs::path corrupt_writer_directory = fs::temp_directory_path() /
        ("buun-zc3b-corrupt-writer-" + std::to_string(pid));
    fs::remove_all(corrupt_writer_directory, ec);
    fs::copy(async_directory, corrupt_writer_directory,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    std::fstream corrupt_root(corrupt_writer_directory / "manifest.bcal",
                              std::ios::binary | std::ios::in | std::ios::out);
    CHECK(corrupt_root.good());
    corrupt_root.seekp(20);
    const char corrupt_byte = '\xff';
    corrupt_root.write(&corrupt_byte, 1);
    corrupt_root.close();
    auto corrupt_writer_owner =
        std::make_unique<server_cache_calibration_writer>();
    auto & corrupt_writer = *corrupt_writer_owner;
    CHECK(corrupt_writer.start(corrupt_writer_directory.string()));
    bool corrupt_loaded = false;
    for (int i = 0; i < 500 &&
             !(corrupt_loaded = corrupt_writer.poll_loaded(
                 writer_status, writer_loaded)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(corrupt_loaded);
    CHECK(writer_status == server_cache_calibration_load_status::corrupt);
    CHECK(corrupt_writer.health() ==
          server_cache_calibration_writer_health::quarantined);
    CHECK(!corrupt_writer.enqueue(retry_profile));
    corrupt_writer.stop();

    // One sub-cadence row is flushed explicitly at the model-sleep/final
    // lifecycle door and reconstructs on the next process.
    const fs::path coordinator_directory = fs::temp_directory_path() /
        ("buun-zc3b-coordinator-" + std::to_string(pid));
    fs::remove_all(coordinator_directory, ec);
    fs::remove_all(corrupt_writer_directory, ec);
    auto coordinator_owner =
        std::make_unique<server_cache_calibration_coordinator>();
    auto & coordinator = *coordinator_owner;
    CHECK(coordinator.start(coordinator_directory.string()));
    server_cache_execution_fingerprint coordinator_fingerprint;
    coordinator_fingerprint.complete = true;
    coordinator_fingerprint.execution_root[0] = 0x91;
    server_cache_observation_store coordinator_observer;
    bool coordinator_loaded = false;
    for (int i = 0; i < 500 &&
             !(coordinator_loaded = coordinator.resolve_load(
                 coordinator_fingerprint, coordinator_observer)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(coordinator_loaded);
    auto one_row = accepted_record(coordinator_fingerprint);
    CHECK(coordinator_observer.observe(one_row));
    server_cache_execution_fingerprint second_fingerprint =
        coordinator_fingerprint;
    second_fingerprint.execution_root[0] = 0x92;
    CHECK(coordinator.resolve_load(second_fingerprint, coordinator_observer));
    CHECK(coordinator_observer.execution_fingerprint().execution_root ==
          second_fingerprint.execution_root);
    CHECK(std::none_of(coordinator_observer.instances().begin(),
                       coordinator_observer.instances().end(),
                       [](const auto & value) { return value.used; }));
    auto second_row = accepted_record(second_fingerprint);
    CHECK(coordinator_observer.observe(second_row));
    CHECK(coordinator.resolve_load(
        coordinator_fingerprint, coordinator_observer));
    CHECK(coordinator_observer.execution_fingerprint().execution_root ==
          coordinator_fingerprint.execution_root);
    CHECK(std::any_of(coordinator_observer.instances().begin(),
                      coordinator_observer.instances().end(),
                      [&](const auto & value) {
                          return value.used &&
                              value.key.profile_execution_digest ==
                                  coordinator_fingerprint.execution_root;
                      }));
    CHECK(!coordinator.resume_pending());
    const auto sleep_flush_start = std::chrono::steady_clock::now();
    coordinator.flush_latest(coordinator_observer);
    CHECK(std::chrono::steady_clock::now() - sleep_flush_start <
          std::chrono::milliseconds(100));
    coordinator.drain_latest_for_shutdown(coordinator_observer);
    coordinator.stop();
    server_cache_calibration_store coordinator_store;
    CHECK(coordinator_store.open(coordinator_directory.string()) ==
          server_cache_calibration_load_status::ok);
    CHECK(coordinator_store.load_profiles(writer_loaded) ==
          server_cache_calibration_load_status::ok);
    CHECK(writer_loaded.size() == 2);
    CHECK(std::all_of(writer_loaded.begin(), writer_loaded.end(),
                      [](const auto & value) {
                          return value.instances.size() == 1;
                      }));
    coordinator_store.close();
    auto resumed_coordinator_owner =
        std::make_unique<server_cache_calibration_coordinator>();
    auto & resumed_coordinator = *resumed_coordinator_owner;
    CHECK(resumed_coordinator.start(coordinator_directory.string()));
    server_cache_observation_store resumed_observer;
    bool resumed_loaded = false;
    for (int i = 0; i < 500 &&
             !(resumed_loaded = resumed_coordinator.resolve_load(
                 coordinator_fingerprint, resumed_observer)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(resumed_loaded);
    CHECK(resumed_coordinator.resume_pending());
    CHECK(resumed_coordinator.resume_started_us() > 0);
    CHECK(std::any_of(resumed_observer.instances().begin(),
                      resumed_observer.instances().end(),
                      [](const auto & value) { return value.used; }));
    CHECK(resumed_coordinator.resolve_load(second_fingerprint,
                                            resumed_observer));
    CHECK(resumed_coordinator.resume_pending());
    CHECK(resumed_coordinator.resolve_load(coordinator_fingerprint,
                                            resumed_observer));
    CHECK(resumed_coordinator.resume_pending());
    resumed_coordinator.complete_resume_validation();
    CHECK(!resumed_coordinator.resume_pending());
    resumed_coordinator.lifecycle(resumed_observer);
    CHECK(resumed_coordinator.health() ==
          server_cache_calibration_writer_health::healthy);
    resumed_coordinator.stop();

    // A full 16-profile shadow cache cannot stall model lifecycle. A dirty
    // novel root is retained by deterministic replacement, and a subsequent
    // novel root still publishes atomically instead of leaving the pending
    // fingerprint wedged forever.
    auto capacity_coordinator_owner =
        std::make_unique<server_cache_calibration_coordinator>();
    auto & capacity_coordinator = *capacity_coordinator_owner;
    CHECK(capacity_coordinator.start(prune_directory.string()));
    server_cache_observation_store capacity_observer;
    server_cache_execution_fingerprint capacity_fingerprint;
    capacity_fingerprint.complete = true;
    capacity_fingerprint.execution_root[0] = 2;
    bool capacity_loaded = false;
    for (int i = 0; i < 500 &&
             !(capacity_loaded = capacity_coordinator.resolve_load(
                 capacity_fingerprint, capacity_observer)); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(capacity_loaded);
    server_cache_execution_fingerprint novel_a = capacity_fingerprint;
    novel_a.execution_root[0] = 0xe0;
    CHECK(capacity_coordinator.resolve_load(novel_a, capacity_observer));
    auto novel_a_row = accepted_record(novel_a);
    CHECK(capacity_observer.observe(novel_a_row));
    server_cache_execution_fingerprint novel_b = capacity_fingerprint;
    novel_b.execution_root[0] = 0xe1;
    CHECK(capacity_coordinator.resolve_load(novel_b, capacity_observer));
    CHECK(capacity_observer.execution_fingerprint().execution_root ==
          novel_b.execution_root);
    CHECK(capacity_coordinator.health() ==
          server_cache_calibration_writer_health::healthy);
    capacity_coordinator.stop();
    fs::remove_all(directory, ec);
    fs::remove_all(async_directory, ec);
    fs::remove_all(regression_directory, ec);
    fs::remove_all(prune_directory, ec);
    fs::remove_all(manifest_fault_directory, ec);
    fs::remove_all(reference_fault_directory, ec);
    fs::remove_all(swapped_directory, ec);
    fs::remove_all(exhausted_directory, ec);
    fs::remove_all(root_exhausted_directory, ec);
    fs::remove_all(root_budget_directory, ec);
    fs::remove_all(coordinator_directory, ec);
#if !defined(_WIN32)
    fs::remove_all(missing_root, ec);
    fs::remove_all(linked_directory, ec);
    fs::remove(symlink_directory, ec);
    fs::remove_all(symlink_target, ec);
    fs::permissions(open_mode_directory, fs::perms::owner_all,
                    fs::perm_options::replace, ec);
    fs::remove_all(open_mode_directory, ec);
    fs::permissions(readable_mode_directory, fs::perms::owner_all,
                    fs::perm_options::replace, ec);
    fs::remove_all(readable_mode_directory, ec);
    fs::remove_all(umask_directory, ec);
#endif

    std::puts("PASS");
    return 0;
}
