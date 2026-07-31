#include "llama-vbr-artifact.h"
#include "llama-vbr-artifact-catalog.h"
#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

static std::string hex(const std::array<uint8_t, 32> & bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size()*2);
    for (uint8_t byte : bytes) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 0x0f]);
    }
    return out;
}

static std::array<uint8_t, 32> digest_of(const std::vector<uint8_t> & bytes) {
    llama_sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.finish();
}

struct memory_source {
    std::vector<uint8_t> bytes;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        const auto * self = static_cast<const memory_source *>(context);
        if (offset > self->bytes.size() ||
            size > self->bytes.size() - size_t(offset)) {
            return false;
        }
        memcpy(destination, self->bytes.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { bytes.size(), this, read };
    }
};

struct generated_source {
    uint64_t size = 0;
    uint8_t salt = 0;
    uint64_t calls = 0;
    size_t max_request = 0;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = const_cast<generated_source *>(
            static_cast<const generated_source *>(context));
        self->calls++;
        self->max_request = std::max(self->max_request, size);
        if (offset > self->size || size > self->size - offset) {
            return false;
        }
        for (size_t i = 0; i < size; ++i) {
            destination[i] =
                uint8_t(self->salt + uint8_t((offset + i)*131u));
        }
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { size, this, read };
    }
};

struct phased_source {
    std::array<uint8_t, 4> first  = { 0x10, 0x11, 0x12, 0x13 };
    std::array<uint8_t, 4> second = { 0xa0, 0xa1, 0xa2, 0xa3 };
    uint32_t calls = 0;
    uint32_t switch_after = 3;

    static bool read(
            const void * context,
            uint64_t offset,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = const_cast<phased_source *>(
            static_cast<const phased_source *>(context));
        self->calls++;
        const auto & selected =
            self->calls <= self->switch_after ?
                self->first : self->second;
        if (offset > selected.size() ||
            size > selected.size() - size_t(offset)) {
            return false;
        }
        memcpy(destination, selected.data() + offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const {
        return { first.size(), this, read };
    }
};

struct memory_reader {
    const std::vector<uint8_t> * bytes = nullptr;
    size_t position = 0;

    static bool read(
            void * context,
            uint8_t * destination,
            size_t size) noexcept {
        auto * self = static_cast<memory_reader *>(context);
        if (size > self->bytes->size() - self->position) {
            return false;
        }
        memcpy(destination, self->bytes->data() + self->position, size);
        self->position += size;
        return true;
    }
};

struct staged_consumer {
    uint64_t bytes = 0;
    uint32_t finishes = 0;
    bool verified = false;

    static bool consume(
            void * context,
            vbr_artifact_section_kind,
            uint32_t,
            uint32_t,
            bool,
            uint64_t,
            uint64_t,
            const uint8_t *,
            size_t size) noexcept {
        static_cast<staged_consumer *>(context)->bytes += size;
        return true;
    }

    static void finish(void * context, bool success) noexcept {
        auto * self = static_cast<staged_consumer *>(context);
        self->finishes++;
        self->verified = success;
        if (!success) {
            self->bytes = 0;
        }
    }
};

static std::array<uint8_t, 32> marker(uint8_t value) {
    std::array<uint8_t, 32> result;
    result.fill(value);
    return result;
}

struct fixture_storage {
    memory_source payload0 { { 0x10, 0x11, 0x12, 0x13 } };
    memory_source payload1 { { 0x20, 0x21, 0x22, 0x23 } };
    memory_source stash0   { { 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 } };
    memory_source stash1   { { 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47 } };
    memory_source recurrent { { 0x90, 0x91, 0x92 } };
};

static vbr_artifact_portable_topology make_topology() {
    llama_cache_acct_shard_topology source;
    const std::vector<std::string> devices = {
        "fixture-device-a",
        "fixture-device-b",
    };
    const float weights[] = { 0.6f, 0.4f };
    CHECK(llama_cache_acct_build_shard_topology(
        devices, LLAMA_SPLIT_MODE_TENSOR, 0, weights, source));

    return source;
}

static vbr_artifact_shard_descriptor make_shard(
        uint32_t index,
        const vbr_artifact_byte_source & source) {
    vbr_artifact_shard_descriptor shard;
    shard.shard_index = index;
    shard.topology_index = 0;
    shard.device_ordinal = uint16_t(index);
    shard.logical_offset = index;
    shard.row_count = 1;
    shard.column_count = source.size;
    shard.row_bytes = source.size;
    shard.payload_bytes = source.size;
    shard.payload = source;
    return shard;
}

static vbr_generation_page_ref make_page(uint64_t mask) {
    vbr_generation_page_ref page;
    page.page_index = 0;
    page.captured_page_gen = 7;
    page.covered_mask[0] = mask;
    return page;
}

static vbr_artifact_package make_package(fixture_storage & storage) {
    vbr_artifact_package package;
    package.topologies.push_back(make_topology());

    vbr_artifact_unit_blob blob;
    auto & descriptor = blob.descriptor;
    descriptor.child_id = 0;
    descriptor.logical_unit_id = 0;
    // This UUID is the validated SOURCE controller identity. The golden therefore proves the
    // buun OQ5 native-lineage branch is representable without rebasing it to a target UUID.
    descriptor.lineage_uuid = { 0x0102030405060708ull, 0x1112131415161718ull };
    descriptor.repr_gen = 17;
    descriptor.current_type = GGML_TYPE_TURBO4_0;
    descriptor.last_source_type = GGML_TYPE_TURBO8_0;
    descriptor.promote_hops = 1;
    descriptor.last_transition = vbr_repr_transition::degrade_other;
    descriptor.representation.kind =
        vbr_artifact_representation_kind::approximate;
    descriptor.representation.codec_id = 0x5438;
    descriptor.representation.codec_version = 1;
    descriptor.representation.reference_digest = marker(0x51);
    descriptor.representation.source_loss_history = 1;
    descriptor.representation.checkpoint_codec_hops = 0;
    descriptor.side = vbr_artifact_side::key;
    descriptor.n_stream = 1;
    descriptor.unified = true;
    descriptor.wm_cells = 1;
    descriptor.rank = 2;
    descriptor.dimensions = { 1, 4, 0, 0 };
    descriptor.row_alignment = 4;
    descriptor.row_codec_version = 1;
    descriptor.codebook_digest = marker(0x61);
    descriptor.rotation_digest = marker(0x62);
    descriptor.meansub_digest = marker(0x63);
    descriptor.shards = {
        make_shard(0, storage.payload0.source()),
        make_shard(1, storage.payload1.source()),
    };
    descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::present;
    descriptor.clean_stash.valid_rows = 1;
    descriptor.clean_stash.domain = vbr_repr_domain::tapped;
    descriptor.clean_stash.row_count = 1;
    descriptor.clean_stash.column_count = 4;
    descriptor.clean_stash.row_bytes = 8;
    descriptor.clean_stash.shards = {
        make_shard(0, storage.stash0.source()),
        make_shard(1, storage.stash1.source()),
    };
    package.unit_blobs.push_back(blob);

    auto & manifest = package.manifest;
    manifest.identity_policy_order_digest = marker(0x71);
    manifest.identity.execution_identity = "exec:qwen";
    manifest.identity.adapter_config_identity = "base:no-lora";
    manifest.identity.media_content_identity = "text-only";
    manifest.identity.sequence_epoch = 3;
    manifest.identity.token_count = 2;
    manifest.identity.next_position = 2;
    manifest.generation.version = 1;
    manifest.generation.status =
        vbr_checkpoint_generation_status::complete;
    manifest.generation.identity_policy_order_digest =
        manifest.identity_policy_order_digest;

    vbr_checkpoint_generation_controller controller;
    controller.child_id = 0;
    controller.dependency_mode =
        checkpoint_child_dependency_mode::live_guarded;
    controller.lineage_uuid = descriptor.lineage_uuid;
    controller.global_generation = 5;
    controller.units.push_back({
        descriptor.repr_gen,
        descriptor.current_type,
        descriptor.last_source_type,
        vbr_repr_domain::tapped,
        descriptor.promote_hops,
        descriptor.last_transition,
    });
    vbr_checkpoint_generation_stream stream;
    stream.stream_index = 0;
    stream.dependency_seq_id = 0;
    stream.computation_frontier = 2;
    stream.captured_dependency_count = 2;
    stream.pages.push_back(make_page(0x3));
    controller.streams.push_back(stream);
    manifest.generation.controllers.push_back(controller);
    manifest.consistency.kind =
        vbr_artifact_consistency_kind::capture_exact;

    vbr_artifact_controller_policy policy;
    policy.child_id = 0;
    policy.dependency_mode = controller.dependency_mode;
    policy.degrade_order_digest = marker(0x72);
    policy.policy_digest = marker(0x73);
    policy.cursor = 1;
    policy.floor_type = 4;
    policy.pressure_independent_settings = 0x55;
    policy.n_stream = 1;
    policy.unified = true;
    policy.wm_cells = 1;
    policy.current_type_vector_digest = marker(0x74);
    policy.completed_wave = true;
    manifest.controller_policy.push_back(policy);

    vbr_artifact_unit_reference reference;
    reference.lineage_uuid = descriptor.lineage_uuid;
    reference.logical_unit_id = descriptor.logical_unit_id;
    reference.repr_gen = descriptor.repr_gen;
    reference.authorized_stream_refs = { 0 };
    reference.has_stash_reference = true;
    reference.stash_reference.valid_rows = 1;
    reference.stash_reference.domain = vbr_repr_domain::tapped;
    reference.stash_reference.row_count = 1;
    reference.stash_reference.column_count = 4;
    reference.stash_reference.row_bytes = 8;
    reference.stash_reference.captured_sink_count = 2;
    reference.stash_reference.covered_sink_pages = { make_page(0x3) };
    manifest.unit_references.push_back(reference);

    const vbr_artifact_portable_domain device0 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        0,
        0,
    };
    const vbr_artifact_portable_domain device1 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        0,
        1,
    };
    const vbr_artifact_portable_domain host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    manifest.accounting = {
        {
            vbr_artifact_accounting_role::unit_payload,
            device0, 4, 4, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::unit_payload,
            device1, 4, 4, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::clean_stash_payload,
            device0, 8, 8, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::clean_stash_payload,
            device1, 8, 8, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::descriptor_metadata,
            host, 512, 512, llama_cache_acct_attr_kind::artifact,
        },
        {
            vbr_artifact_accounting_role::reference_metadata,
            host, 256, 256, llama_cache_acct_attr_kind::artifact,
        },
    };
    return package;
}

static vbr_artifact_decode_limits limits(uint64_t bytes) {
    vbr_artifact_decode_limits result;
    result.max_total_bytes = bytes;
    return result;
}

static void test_golden_and_native_lineage() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(!encoded.empty());
    CHECK(package.topologies[0].digest ==
          llama_cache_acct_compute_topology_digest(package.topologies[0]));
    CHECK(hex(package.unit_blobs[0].unit_version_id.bytes()) ==
          "44dcfd7b6235cae759e3570b2aed858103c08b9b3603c46cf62755686ab2401f");
    CHECK(hex(package.unit_blobs[0].payload_digest.bytes()) ==
          "8325d422f361b83e19f57dd0e6e566f330961cae4889c2f934bf94619316705f");
    CHECK(hex(package.unit_blobs[0].descriptor.clean_stash.payload_id.bytes()) ==
          "5c20925bd0120766c0a1db7995677754755488a70b3780dcd53e63ab1535cf18");
    CHECK(hex(package.manifest.capture_generation_id.bytes()) ==
          "7227e21ce3e7076d85d625a3e41baca0ebe3a8d078b44f16b97ca69f81a79407");
    CHECK(hex(package.manifest.manifest_digest.bytes()) ==
          "3d8267620a9c563a1e84c2a277fc58aabc6a081c794d947e8eee51e2a03fbf6e");
    CHECK(hex(digest_of(encoded)) ==
          "b8ba3cb1191ca5be00720d5ab77a2b8c49406b9c5957b39f52c932e2c6c68e8d");
    CHECK(encoded.size() == 2254);
    CHECK(encoded[0] == 0x56 && encoded[1] == 0x42 &&
          encoded[2] == 0x52 && encoded[3] == 0x32);
    CHECK(encoded[4] == 1 && encoded[5] == 0 &&
          encoded[6] == 0 && encoded[7] == 0);

    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) ==
          vbr_artifact_status::ok);
    CHECK(decoded.version == VBR_UNIT_ARTIFACT_FORMAT_VERSION);
    CHECK(decoded.unit_blobs.size() == 1);
    CHECK(decoded.manifest.generation.controllers.size() == 1);
    CHECK(decoded.manifest.generation.controllers[0].lineage_uuid ==
          package.manifest.generation.controllers[0].lineage_uuid);
    CHECK(decoded.manifest.generation.controllers[0].lineage_uuid ==
          package.unit_blobs[0].descriptor.lineage_uuid);
    CHECK(decoded.manifest.consistency.kind ==
          vbr_artifact_consistency_kind::capture_exact);
    CHECK(decoded.manifest.consistency.source_capture_generation_id ==
          decoded.manifest.capture_generation_id);
}

static void test_identity_and_reference_separation() {
    fixture_storage storage;
    auto first = make_package(storage);
    std::vector<uint8_t> bytes;
    CHECK(vbr_artifact_encode_vector(first, bytes, 1024*1024) ==
          vbr_artifact_status::ok);

    auto ownership_changed = make_package(storage);
    ownership_changed.manifest.generation.controllers[0].streams[0]
        .pages[0].covered_mask[0] = 0x5;
    ownership_changed.manifest.generation.controllers[0].streams[0]
        .captured_dependency_count = 2;
    ownership_changed.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    CHECK(vbr_artifact_encode_vector(
              ownership_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(ownership_changed.unit_blobs[0].unit_version_id ==
          first.unit_blobs[0].unit_version_id);
    CHECK(ownership_changed.manifest.manifest_digest !=
          first.manifest.manifest_digest);

    storage.payload0.bytes[0] ^= 1;
    auto payload_changed = make_package(storage);
    CHECK(vbr_artifact_encode_vector(
              payload_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(payload_changed.unit_blobs[0].descriptor.lineage_uuid ==
          first.unit_blobs[0].descriptor.lineage_uuid);
    CHECK(payload_changed.unit_blobs[0].descriptor.repr_gen ==
          first.unit_blobs[0].descriptor.repr_gen);
    CHECK(payload_changed.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(payload_changed.unit_blobs[0].descriptor.clean_stash.payload_id ==
          first.unit_blobs[0].descriptor.clean_stash.payload_id);

    storage.payload0.bytes[0] ^= 1;
    auto generation_changed = make_package(storage);
    generation_changed.unit_blobs[0].descriptor.repr_gen++;
    generation_changed.manifest.generation.controllers[0].units[0].repr_gen++;
    generation_changed.manifest.unit_references[0].repr_gen++;
    CHECK(vbr_artifact_encode_vector(
              generation_changed, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(generation_changed.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(generation_changed.unit_blobs[0].descriptor.repr_gen !=
          first.unit_blobs[0].descriptor.repr_gen);

    auto mismatched_tuple = first;
    CHECK(mismatched_tuple.manifest.unit_references[0].unit_version_id ==
          mismatched_tuple.unit_blobs[0].unit_version_id);
    mismatched_tuple.manifest.unit_references[0].lineage_uuid.lo ^= 1;
    CHECK(vbr_artifact_encode_vector(
              mismatched_tuple, bytes, 1024*1024) ==
          vbr_artifact_status::generation_mismatch);

    // A different logical unit may intern the same immutable clean-stash
    // subobject without sharing the unit address.
    auto shared_stash = make_package(storage);
    shared_stash.unit_blobs[0].descriptor.lineage_uuid.lo++;
    shared_stash.manifest.generation.controllers[0].lineage_uuid.lo++;
    shared_stash.manifest.unit_references[0].lineage_uuid.lo++;
    CHECK(vbr_artifact_encode_vector(
              shared_stash, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(shared_stash.unit_blobs[0].descriptor.clean_stash.payload_id ==
          first.unit_blobs[0].descriptor.clean_stash.payload_id);
    CHECK(shared_stash.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);

    auto absent = make_package(storage);
    absent.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    absent.unit_blobs[0].descriptor.clean_stash = {};
    absent.manifest.unit_references[0].has_stash_reference = false;
    absent.manifest.unit_references[0].stash_reference = {};
    CHECK(vbr_artifact_encode_vector(absent, bytes, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(absent.unit_blobs[0].unit_version_id !=
          first.unit_blobs[0].unit_version_id);
    CHECK(absent.manifest.consistency.kind ==
          vbr_artifact_consistency_kind::capture_exact);

    // A source-present stash omission is a different wire state. It cannot claim
    // capture_exact; only a target-side, transition-authorized live_rebased reference may
    // carry it.
    auto omitted = make_package(storage);
    CHECK(vbr_artifact_prepare(omitted) == vbr_artifact_status::ok);
    omitted.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::omitted_source_present;
    omitted.unit_blobs[0].descriptor.clean_stash = {};
    omitted.manifest.unit_references[0].has_stash_reference = false;
    omitted.manifest.unit_references[0].stash_reference = {};
    omitted.manifest.accounting.erase(
        std::remove_if(
            omitted.manifest.accounting.begin(),
            omitted.manifest.accounting.end(),
            [](const vbr_artifact_portable_accounting_row & row) {
                return row.role ==
                    vbr_artifact_accounting_role::clean_stash_payload;
            }),
        omitted.manifest.accounting.end());
    CHECK(vbr_artifact_encode_vector(
              omitted, bytes, 1024*1024) != vbr_artifact_status::ok);
    omitted.manifest.consistency.kind =
        vbr_artifact_consistency_kind::live_rebased;
    omitted.manifest.consistency.source_capture_generation_id =
        omitted.manifest.capture_generation_id;
    omitted.manifest.consistency.target_capture_generation_id =
        vbr_capture_generation_id::from_sha256(marker(0x81));
    omitted.manifest.consistency.transition_lineage_id =
        vbr_transition_lineage_id::from_sha256(marker(0x82));
    CHECK(vbr_artifact_encode_vector(
              omitted, bytes, 1024*1024) == vbr_artifact_status::ok);
}

static void test_fail_closed_decode() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);

    vbr_artifact_package decoded = package;
    auto corrupt = encoded;
    corrupt.back() ^= 1;
    CHECK(vbr_artifact_decode_vector(
              corrupt, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    CHECK(decoded.version == 0 && decoded.unit_blobs.empty());

    staged_consumer staged;
    vbr_artifact_payload_consumer consumer {
        &staged, staged_consumer::consume, staged_consumer::finish,
    };
    memory_reader corrupt_reader { &corrupt, 0 };
    vbr_artifact_stream_reader stream {
        &corrupt_reader, memory_reader::read,
    };
    CHECK(vbr_artifact_decode(
              stream, corrupt.size(), limits(1024*1024),
              &consumer, decoded) != vbr_artifact_status::ok);
    CHECK(staged.finishes == 1 && !staged.verified && staged.bytes == 0);

    staged = {};
    memory_reader good_reader { &encoded, 0 };
    stream = { &good_reader, memory_reader::read };
    CHECK(vbr_artifact_decode(
              stream, encoded.size(), limits(1024*1024),
              &consumer, decoded) == vbr_artifact_status::ok);
    CHECK(staged.finishes == 1 && staged.verified && staged.bytes == 24);

    // Integrity is independent of eligibility: corrupt a byte in the first
    // canonical payload while leaving the reference tuple untouched.
    const auto payload_pos = std::search(
        encoded.begin(), encoded.end(),
        storage.payload0.bytes.begin(), storage.payload0.bytes.end());
    CHECK(payload_pos != encoded.end());
    auto corrupt_payload = encoded;
    corrupt_payload[size_t(payload_pos - encoded.begin())] ^= 1;
    const auto tuple_before =
        decoded.manifest.unit_references[0];
    CHECK(std::equal(
        encoded.begin(), encoded.begin() + (payload_pos - encoded.begin()),
        corrupt_payload.begin()));
    CHECK(std::equal(
        encoded.begin() + (payload_pos - encoded.begin()) + 1,
        encoded.end(),
        corrupt_payload.begin() + (payload_pos - encoded.begin()) + 1));
    CHECK(vbr_artifact_decode_vector(
              corrupt_payload, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    CHECK(tuple_before.lineage_uuid ==
          package.manifest.unit_references[0].lineage_uuid);
    CHECK(tuple_before.logical_unit_id ==
          package.manifest.unit_references[0].logical_unit_id);
    CHECK(tuple_before.repr_gen ==
          package.manifest.unit_references[0].repr_gen);

    auto bad_version = encoded;
    bad_version[4] = 2;
    CHECK(vbr_artifact_decode_vector(
              bad_version, limits(1024*1024), decoded) ==
          vbr_artifact_status::unsupported_version);
    auto bad_length = encoded;
    bad_length[16] ^= 1;
    CHECK(vbr_artifact_decode_vector(
              bad_length, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    auto bad_order = encoded;
    bad_order[28] ^= 1;
    CHECK(vbr_artifact_decode_vector(
          bad_order, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    // The topology section begins after the 92-byte package header and
    // 48-byte section header. A count within the configured maximum but
    // impossible for the remaining section must reject before resize().
    auto impossible_count = encoded;
    impossible_count[140] = 16;
    impossible_count[141] = 0;
    impossible_count[142] = 0;
    impossible_count[143] = 0;
    CHECK(vbr_artifact_decode_vector(
              impossible_count, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    auto trailing = encoded;
    trailing.push_back(0);
    CHECK(vbr_artifact_decode_vector(
              trailing, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);
    encoded.pop_back();
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) !=
          vbr_artifact_status::ok);

    auto limited = limits(16);
    CHECK(vbr_artifact_decode_vector(
              corrupt, limited, decoded) ==
          vbr_artifact_status::out_of_bounds);
}

static void test_encoder_rejects_source_mutation() {
    fixture_storage storage;
    phased_source changing;
    auto package = make_package(storage);
    package.unit_blobs[0].descriptor.shards[0].payload =
        changing.source();
    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::content_id_mismatch);
    CHECK(encoded.empty());
    CHECK(changing.calls > changing.switch_after);
}

static void test_validation_and_ordering() {
    fixture_storage storage;
    auto package = make_package(storage);
    std::vector<uint8_t> encoded;
    std::vector<uint8_t> canonical;
    CHECK(vbr_artifact_encode_vector(
              package, canonical, 1024*1024) ==
          vbr_artifact_status::ok);

    auto bad_topology = package;
    bad_topology.topologies[0].shard_weights[0]++;
    CHECK(vbr_artifact_encode_vector(
              bad_topology, encoded, 1024*1024) ==
          vbr_artifact_status::topology_mismatch);

    auto bad_accounting = package;
    bad_accounting.manifest.accounting[0].role =
        vbr_artifact_accounting_role::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_accounting, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    bad_accounting = package;
    bad_accounting.manifest.accounting.push_back(
        bad_accounting.manifest.accounting.front());
    CHECK(vbr_artifact_encode_vector(
              bad_accounting, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    // Completion order is not wire order: stable shard indices canonicalize an
    // arbitrarily completed vector into identical bytes.
    auto reordered_shards = make_package(storage);
    std::swap(
        reordered_shards.unit_blobs[0].descriptor.shards[0],
        reordered_shards.unit_blobs[0].descriptor.shards[1]);
    std::swap(
        reordered_shards.unit_blobs[0].descriptor.clean_stash.shards[0],
        reordered_shards.unit_blobs[0].descriptor.clean_stash.shards[1]);
    CHECK(vbr_artifact_encode_vector(
              reordered_shards, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    CHECK(encoded == canonical);

    auto duplicate_shard = make_package(storage);
    duplicate_shard.unit_blobs[0].descriptor.shards[1].shard_index = 0;
    CHECK(vbr_artifact_encode_vector(
              duplicate_shard, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_enum = package;
    bad_enum.unit_blobs[0].descriptor.layout =
        vbr_artifact_layout::_count;
    CHECK(vbr_artifact_encode_vector(
          bad_enum, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_representation = make_package(storage);
    bad_representation.unit_blobs[0].descriptor.representation.kind =
        vbr_artifact_representation_kind::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_representation, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_recoverability = make_package(storage);
    bad_recoverability.unit_blobs[0].descriptor.recoverability =
        vbr_artifact_recoverability::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_recoverability, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_side = make_package(storage);
    bad_side.unit_blobs[0].descriptor.side = vbr_artifact_side::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_side, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_stash_state = make_package(storage);
    bad_stash_state.unit_blobs[0].descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_stash_state, encoded, 1024*1024) !=
          vbr_artifact_status::ok);
    auto bad_consistency = make_package(storage);
    bad_consistency.manifest.consistency.kind =
        vbr_artifact_consistency_kind::_count;
    CHECK(vbr_artifact_encode_vector(
              bad_consistency, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto bad_stash_authorization = package;
    bad_stash_authorization.manifest.unit_references[0].stash_reference
        .covered_sink_pages[0].covered_mask[0] = 0x5;
    CHECK(vbr_artifact_encode_vector(
              bad_stash_authorization, encoded, 1024*1024) !=
          vbr_artifact_status::ok);

    auto constrained = limits(canonical.size());
    constrained.max_devices_per_topology = 1;
    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
    constrained = limits(canonical.size());
    constrained.max_unit_blobs = 0;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
    constrained = limits(canonical.size());
    constrained.max_accounting_rows = 1;
    CHECK(vbr_artifact_decode_vector(
              canonical, constrained, decoded) !=
          vbr_artifact_status::ok);
}

static void test_companion_payload() {
    fixture_storage storage;
    auto package = make_package(storage);
    vbr_artifact_companion_payload companion;
    companion.kind = vbr_artifact_companion_kind::recurrent;
    companion.format_version = 3;
    companion.build_identity_digest = marker(0x91);
    companion.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    companion.payload_bytes = storage.recurrent.bytes.size();
    companion.payload = storage.recurrent.source();
    package.companions.push_back(companion);
    package.manifest.accounting.push_back({
        vbr_artifact_accounting_role::recurrent_payload,
        companion.domain,
        companion.payload_bytes,
        companion.payload_bytes,
        llama_cache_acct_attr_kind::artifact,
    });

    std::vector<uint8_t> encoded;
    CHECK(vbr_artifact_encode_vector(
              package, encoded, 1024*1024) ==
          vbr_artifact_status::ok);
    vbr_artifact_package decoded;
    CHECK(vbr_artifact_decode_vector(
              encoded, limits(1024*1024), decoded) ==
          vbr_artifact_status::ok);
    CHECK(decoded.companions.size() == 1);
    CHECK(decoded.manifest.companions.size() == 1);
    CHECK(decoded.companions[0].domain == companion.domain);
    CHECK(decoded.companions[0].payload_digest ==
          decoded.manifest.companions[0].payload_digest);
}

struct discard_writer {
    uint64_t bytes = 0;

    static bool write(
            void * context,
            const uint8_t *,
            size_t size) noexcept {
        auto * self = static_cast<discard_writer *>(context);
        self->bytes += size;
        return true;
    }
};

static void test_stream_larger_than_capture_ring() {
    constexpr uint64_t ring_bytes = 256ull*1024*1024;
    generated_source generated { ring_bytes + 1, 0x39 };
    fixture_storage storage;
    auto package = make_package(storage);
    auto & descriptor = package.unit_blobs[0].descriptor;
    descriptor.shards.resize(1);
    descriptor.shards[0] = make_shard(0, generated.source());
    descriptor.shards[0].device_ordinal = 0;
    descriptor.wm_cells = 1;
    descriptor.dimensions = { 1, generated.size, 0, 0 };
    descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    descriptor.clean_stash = {};
    package.manifest.unit_references[0].has_stash_reference = false;
    package.manifest.unit_references[0].stash_reference = {};
    const auto descriptor_row = package.manifest.accounting[
        package.manifest.accounting.size() - 2];
    const auto reference_row = package.manifest.accounting.back();
    package.manifest.accounting = {
        {
            vbr_artifact_accounting_role::unit_payload,
            {
                llama_cache_acct_residency::device,
                llama_cache_acct_domain_kind::device_topology,
                0,
                0,
            },
            generated.size,
            generated.size,
            llama_cache_acct_attr_kind::artifact,
        },
        descriptor_row,
        reference_row,
    };

    discard_writer discarded;
    const vbr_artifact_stream_writer sink {
        &discarded, discard_writer::write,
    };
    uint64_t encoded_size = 0;
    CHECK(vbr_artifact_encode(
              package, sink, ring_bytes + 1024*1024, &encoded_size) ==
          vbr_artifact_status::ok);
    CHECK(encoded_size > ring_bytes);
    CHECK(discarded.bytes == encoded_size);
    CHECK(generated.calls > 1);
    CHECK(generated.max_request <= 1024*1024);
}

static llama_cache_acct_value catalog_cell(
        const llama_cache_acct_snapshot & snapshot,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_measure measure) {
    const auto row = std::find_if(
        snapshot.cells.begin(), snapshot.cells.end(),
        [&](const llama_cache_acct_cell_row & candidate) {
            return candidate.category == category &&
                   candidate.domain == domain;
        });
    return row == snapshot.cells.end()
        ? llama_cache_acct_value {}
        : row->cell.measures[size_t(measure)];
}

struct catalog_fixture {
    fixture_storage storage;
    vbr_artifact_package package;
    llama_cache_acct_ledger ledger;
    std::unique_ptr<llama_vbr_artifact_catalog> catalog;
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    llama_cache_budget_config budget;
    llama_cache_acct_resource_domain host =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host);

    explicit catalog_fixture(bool configure_catalog = true)
        : package(make_package(storage)),
          catalog(new llama_vbr_artifact_catalog(ledger)) {
        CHECK(catalog->bind_topologies(package.topologies, bindings));
        std::vector<llama_cache_acct_completeness_requirement> required;
        required.push_back({
            host, llama_cache_acct_producer::retention_sidecar,
        });
        for (const auto & binding : bindings) {
            required.push_back({
                binding.domain, llama_cache_acct_producer::live_memory,
            });
        }
        CHECK(ledger.configure_required_producers(
            required.data(), required.size()));
        if (configure_catalog) {
            CHECK(catalog->configure_accounting(package));
        }
        const auto initialize = [&](llama_cache_acct_category category,
                                    const llama_cache_acct_resource_domain & domain,
                                    bool transactional) {
            ledger.gauge_set(
                category, domain,
                llama_cache_acct_measure::resident_allocated, 0);
            if (transactional) {
                ledger.gauge_set(
                    category, domain,
                    llama_cache_acct_measure::reserved, 0);
            }
        };
        initialize(
            llama_cache_acct_category::full_snapshot_payload,
            host, true);
        initialize(
            llama_cache_acct_category::checkpoint_state_payload,
            host, true);
        initialize(
            llama_cache_acct_category::typed_accelerator_payload,
            host, true);
        for (const auto & binding : bindings) {
            initialize(
                llama_cache_acct_category::live_attention_state,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::live_recurrent_state,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::recurrent_rollback_planes,
                binding.domain, false);
            initialize(
                llama_cache_acct_category::rolling_window_tape,
                binding.domain, false);
        }
        CHECK(ledger.certify_complete(
            host, llama_cache_acct_producer::retention_sidecar));
        for (const auto & binding : bindings) {
            CHECK(ledger.certify_complete(
                binding.domain, llama_cache_acct_producer::live_memory));
            llama_cache_budget_device_input input;
            input.backend_device =
                reinterpret_cast<const void *>(uintptr_t(1));
            input.domain = binding.domain;
            input.physical_total = 1ull << 30;
            // Fake-shard storage is already resident when publication begins,
            // so the injected point-in-time sample must include it in physical
            // used just as a backend sample would.
            input.physical_free = (1ull << 30) - 1024;
            input.phys_state =
                llama_cache_budget_capacity_state::known;
            input.current_compute_allocated = 0;
            input.configured_compute_reserve = 0;
            input.compute_state =
                llama_cache_budget_capacity_state::known;
            input.cache_cap_state =
                llama_cache_budget_capacity_state::unbounded;
            budget.devices.push_back(input);
        }
        budget.host.pageable_state =
            llama_cache_budget_capacity_state::unbounded;
    }

    std::vector<llama_vbr_artifact_fake_shard_completion>
    completions() const {
        return {
            { 0, 1, true,  true, storage.stash1.bytes },
            { 0, 0, false, true, storage.payload0.bytes },
            { 0, 0, true,  true, storage.stash0.bytes },
            { 0, 1, false, true, storage.payload1.bytes },
        };
    }
};

static vbr_verified_segment verified_segment(
        const llama_vbr_artifact_fake_shard_completion & completion,
        size_t split = SIZE_MAX) {
    auto chain = std::make_shared<artifact_segment_chain>();
    const size_t first = std::min(
        split, completion.bytes.size());
    CHECK(chain->append(completion.bytes.data(), first));
    if (first < completion.bytes.size()) {
        CHECK(chain->append(
            completion.bytes.data() + first,
            completion.bytes.size() - first));
    }
    vbr_verified_segment out;
    out.unit_index = completion.unit_index;
    out.shard_index = completion.shard_index;
    out.clean_stash = completion.clean_stash;
    out.bytes = std::move(chain);
    out.streaming_digest =
        vbr_capture_stream_digest(*out.bytes);
    return out;
}

static void test_catalog_streaming_protocol() {
    catalog_fixture f;
    vbr_capture_stream_status status;
    auto build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(build);
    CHECK(status == vbr_capture_stream_status::ok);
    if (!build) {
        return;
    }
    const uint64_t claims =
        f.ledger.snapshot().live_ops;
    CHECK(claims ==
          f.package.manifest.accounting.size() + 1);

    auto unit = build->begin_unit(0, status);
    CHECK(unit);
    const auto completions = f.completions();
    const size_t order[] = { 3, 0, 2, 1 };
    for (const size_t index : order) {
        auto segment = verified_segment(
            completions[index], 1);
        CHECK(unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(f.ledger.snapshot().live_ops == claims);
    CHECK(unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    auto late = verified_segment(completions[0]);
    CHECK(unit->accept_verified_segment(late) ==
          vbr_capture_stream_status::late_segment);
    const auto streamed = build->publish_reference();
    CHECK(streamed.status ==
          vbr_capture_stream_status::ok);
    CHECK(!streamed.adopted);
    CHECK(streamed.reference_artifact.v != 0);
    CHECK(streamed.unit_content.v != 0);
    unit.reset();
    build.reset();

    auto snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops ==
          f.package.manifest.accounting.size());
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::transfer_staging,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);

    auto adopted_build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(adopted_build);
    auto adopted_unit =
        adopted_build->begin_unit(0, status);
    CHECK(adopted_unit);
    for (const auto & completion : f.completions()) {
        auto value = verified_segment(completion, 2);
        CHECK(adopted_unit->accept_verified_segment(value) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(adopted_unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    const auto adopted =
        adopted_build->publish_reference();
    CHECK(adopted.status == vbr_capture_stream_status::ok);
    CHECK(adopted.adopted);
    CHECK(adopted.reference_artifact !=
          streamed.reference_artifact);
    CHECK(adopted.unit_content == streamed.unit_content);
    CHECK(adopted.reference_lineage ==
          streamed.reference_lineage);
    adopted_unit.reset();
    adopted_build.reset();

    snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops ==
          2*f.package.manifest.accounting.size());
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    const auto adopted_state = f.catalog->snapshot();
    CHECK(adopted_state.blobs == 1);
    CHECK(adopted_state.stashes == 1);
    CHECK(adopted_state.references == 2);
    CHECK(adopted_state.published == 1);
    CHECK(adopted_state.adopted == 1);

    catalog_fixture fake_equivalent;
    const auto fake_first = fake_equivalent.catalog->publish(
        fake_equivalent.package,
        fake_equivalent.completions(),
        fake_equivalent.budget);
    CHECK(fake_first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(fake_first.reference_artifact ==
          streamed.reference_artifact);

    catalog_fixture invalid;
    auto bad_build = invalid.catalog->begin_capture(
        invalid.package, invalid.budget, {}, status);
    CHECK(bad_build);
    auto bad_unit = bad_build->begin_unit(0, status);
    CHECK(bad_unit);
    auto segment = verified_segment(
        invalid.completions()[0]);
    CHECK(bad_unit->accept_verified_segment(segment) ==
          vbr_capture_stream_status::ok);
    CHECK(bad_unit->accept_verified_segment(segment) ==
          vbr_capture_stream_status::duplicate_segment);
    CHECK(bad_unit->seal_unit() ==
          vbr_capture_stream_status::duplicate_segment);
    CHECK(bad_build->publish_reference().status ==
          vbr_capture_stream_status::duplicate_segment);
    bad_unit.reset();
    bad_build.reset();
    CHECK(invalid.ledger.snapshot().live_ops == 0);
    CHECK(invalid.catalog->snapshot().references == 0);

    catalog_fixture missing;
    auto missing_build = missing.catalog->begin_capture(
        missing.package, missing.budget, {}, status);
    CHECK(missing_build);
    auto missing_unit =
        missing_build->begin_unit(0, status);
    CHECK(missing_unit);
    auto only = verified_segment(missing.completions()[0]);
    CHECK(missing_unit->accept_verified_segment(only) ==
          vbr_capture_stream_status::ok);
    CHECK(missing_unit->seal_unit() ==
          vbr_capture_stream_status::missing_segment);
    missing_unit.reset();
    missing_build.reset();
    CHECK(missing.ledger.snapshot().live_ops == 0);
    CHECK(missing.catalog->snapshot().references == 0);

    for (const bool commit_fault : { false, true }) {
        catalog_fixture faulted;
        llama_cache_transaction_fault fault;
        if (commit_fault) {
            fault.fail_commit_at = 0;
        } else {
            fault.fail_stage_at = 0;
        }
        auto fault_build = faulted.catalog->begin_capture(
            faulted.package, faulted.budget, fault, status);
        CHECK(fault_build);
        auto fault_unit =
            fault_build->begin_unit(0, status);
        CHECK(fault_unit);
        for (const auto & completion :
             faulted.completions()) {
            auto value = verified_segment(completion);
            CHECK(fault_unit->accept_verified_segment(value) ==
                  vbr_capture_stream_status::ok);
        }
        CHECK(fault_unit->seal_unit() ==
              vbr_capture_stream_status::ok);
        const auto failed = fault_build->publish_reference();
        CHECK(failed.status ==
              (commit_fault
                   ? vbr_capture_stream_status::commit_failed
                   : vbr_capture_stream_status::stage_failed));
        fault_unit.reset();
        fault_build.reset();
        CHECK(faulted.ledger.snapshot().live_ops == 0);
        CHECK(faulted.catalog->snapshot().references == 0);
    }

    catalog_fixture overlap;
    const auto generous_overlap_budget = overlap.budget;
    overlap.budget.host.pageable_cap = 1000;
    overlap.budget.host.pageable_state =
        llama_cache_budget_capacity_state::known;
    vbr_capture_begin_diagnostics refusal_diagnostics;
    auto refused = overlap.catalog->begin_capture(
        overlap.package, overlap.budget, {}, status,
        &refusal_diagnostics);
    CHECK(!refused);
    CHECK(status ==
          vbr_capture_stream_status::accounting_refused);
    CHECK(refusal_diagnostics.reservation_group ==
          vbr_capture_reservation_group::durable_artifact);
    CHECK(refusal_diagnostics.prepare_status ==
          llama_cache_prepare_status::admission_refused);
    CHECK(refusal_diagnostics.admission_status ==
          llama_cache_admission_status::exceeds_budget);
    CHECK(overlap.catalog->snapshot()
              .staging_overlap_refusals == 1);
    CHECK(overlap.ledger.snapshot().live_ops == 0);
    const auto fake_after_refusal = overlap.catalog->publish(
        overlap.package, overlap.completions(),
        generous_overlap_budget);
    CHECK(fake_after_refusal.status ==
          llama_vbr_artifact_publish_status::published);

    catalog_fixture unconfigured(false);
    const auto unavailable = unconfigured.catalog->publish(
        unconfigured.package, unconfigured.completions(),
        unconfigured.budget);
    CHECK(unavailable.status ==
          llama_vbr_artifact_publish_status::
              accounting_unavailable);
    CHECK(unconfigured.ledger.snapshot().live_ops == 0);
}

static void test_catalog_multi_unit_atomic_publish() {
    catalog_fixture f;
    auto & package = f.package;
    auto second = package.unit_blobs.front();
    second.descriptor.logical_unit_id = 1;
    second.descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state::absent_at_source;
    second.descriptor.clean_stash = {};
    second.unit_version_id = {};
    second.payload_digest = {};
    package.unit_blobs.push_back(second);
    package.manifest.generation.controllers[0].units.push_back(
        package.manifest.generation.controllers[0].units.front());
    auto second_reference =
        package.manifest.unit_references.front();
    second_reference.logical_unit_id = 1;
    second_reference.unit_version_id = {};
    second_reference.payload_digest = {};
    second_reference.has_stash_reference = false;
    second_reference.stash_reference = {};
    package.manifest.unit_references.push_back(second_reference);
    for (auto & row : package.manifest.accounting) {
        if (row.role ==
                vbr_artifact_accounting_role::unit_payload) {
            row.logical_bytes *= 2;
            row.resident_bytes *= 2;
        }
    }
    package.manifest.manifest_digest = {};
    package.manifest.capture_generation_id = {};
    package.manifest.consistency = {};

    CHECK(f.catalog->configure_accounting(package));
    vbr_capture_stream_status status;
    {
        auto aborted = f.catalog->begin_capture(
            package, f.budget, {}, status);
        CHECK(aborted);
        auto first_only = aborted
            ? aborted->begin_unit(0, status) : nullptr;
        CHECK(first_only);
        if (first_only) {
            for (const auto & completion : f.completions()) {
                auto segment = verified_segment(completion, 2);
                CHECK(first_only->accept_verified_segment(segment) ==
                      vbr_capture_stream_status::ok);
            }
            CHECK(first_only->seal_unit() ==
                  vbr_capture_stream_status::ok);
        }
        first_only.reset();
        aborted.reset();
        CHECK(f.catalog->snapshot().references == 0);
        CHECK(f.ledger.snapshot().live_ops == 0);
    }
    auto build = f.catalog->begin_capture(
        package, f.budget, {}, status);
    CHECK(build);
    if (!build) {
        return;
    }
    const auto completions = f.completions();
    auto first = build->begin_unit(0, status);
    CHECK(first);
    for (const auto & completion : completions) {
        auto segment = verified_segment(completion, 2);
        CHECK(first->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(first->seal_unit() ==
          vbr_capture_stream_status::ok);
    first.reset();

    auto second_unit = build->begin_unit(1, status);
    CHECK(second_unit);
    for (const auto & completion : completions) {
        if (completion.clean_stash) {
            continue;
        }
        auto copy = completion;
        copy.unit_index = 1;
        auto segment = verified_segment(copy, 3);
        CHECK(second_unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(second_unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    second_unit.reset();

    const auto published = build->publish_reference();
    CHECK(published.status == vbr_capture_stream_status::ok);
    CHECK(published.reference_artifact.v != 0);
    const auto snapshot = f.catalog->snapshot();
    CHECK(snapshot.blobs == 2);
    CHECK(snapshot.stashes == 1);
    CHECK(snapshot.references == 1);
    build.reset();
    CHECK(f.catalog->retire(published.reference_artifact));
    CHECK(f.catalog->snapshot().blobs == 0);
    CHECK(f.catalog->snapshot().stashes == 0);
    CHECK(f.ledger.snapshot().live_ops == 0);
}

static void test_catalog_streaming_companion_lifetime() {
    catalog_fixture f;
    vbr_artifact_companion_payload companion;
    companion.kind = vbr_artifact_companion_kind::recurrent;
    companion.format_version = 1;
    companion.build_identity_digest = marker(0xa1);
    companion.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX,
        UINT16_MAX,
    };
    companion.payload_bytes = f.storage.recurrent.bytes.size();
    f.package.companions.push_back(companion);
    f.package.manifest.accounting.push_back({
        vbr_artifact_accounting_role::recurrent_payload,
        companion.domain,
        companion.payload_bytes,
        companion.payload_bytes,
        llama_cache_acct_attr_kind::artifact,
    });
    CHECK(f.catalog->configure_accounting(f.package));

    vbr_capture_stream_status status;
    auto build = f.catalog->begin_capture(
        f.package, f.budget, {}, status);
    CHECK(build);
    if (!build) {
        return;
    }
    auto unit = build->begin_unit(0, status);
    CHECK(unit);
    for (const auto & completion : f.completions()) {
        const auto segment = verified_segment(completion, 2);
        CHECK(unit->accept_verified_segment(segment) ==
              vbr_capture_stream_status::ok);
    }
    CHECK(unit->seal_unit() ==
          vbr_capture_stream_status::ok);
    unit.reset();

    auto companion_bytes =
        std::make_shared<artifact_segment_chain>();
    CHECK(companion_bytes->append(
        f.storage.recurrent.bytes.data(),
        f.storage.recurrent.bytes.size()));
    vbr_verified_companion verified;
    verified.companion_index = 0;
    verified.bytes = companion_bytes;
    verified.streaming_digest =
        vbr_capture_stream_digest(*companion_bytes);
    CHECK(build->accept_verified_companion(verified) ==
          vbr_capture_stream_status::ok);
    const auto published = build->publish_reference();
    CHECK(published.status == vbr_capture_stream_status::ok);
    build.reset();
    companion_bytes.reset();

    const auto snapshot = f.ledger.snapshot();
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::full_snapshot_payload,
        f.host,
        llama_cache_acct_measure::resident_allocated).value ==
        f.storage.recurrent.bytes.size());
    CHECK(f.catalog->retire(published.reference_artifact));
    CHECK(f.ledger.snapshot().live_ops == 0);
}

static void test_catalog_charge_once_and_retire() {
    catalog_fixture f;
    const size_t alloc_baseline =
        f.ledger.allocation_registry_size();
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(first.reference_artifact.v != 0);
    CHECK(first.unit_content.v != 0);

    auto snapshot = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 256);

    const auto second =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::adopted);
    CHECK(second.reference_artifact.v != first.reference_artifact.v);
    CHECK(second.unit_content == first.unit_content);
    CHECK(second.reference_lineage == first.reference_lineage);
    snapshot = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::unit_version_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 4);
        CHECK(catalog_cell(
            snapshot,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 512);
    const auto catalog_state = f.catalog->snapshot();
    CHECK(catalog_state.blobs == 1);
    CHECK(catalog_state.stashes == 1);
    CHECK(catalog_state.references == 2);
    CHECK(f.ledger.allocation_registry_size() > alloc_baseline);

    CHECK(f.catalog->retire(first.reference_artifact));
    CHECK(f.ledger.allocation_registry_size() > alloc_baseline);
    snapshot = f.ledger.snapshot();
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 4);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 256);
    CHECK(f.catalog->retire(second.reference_artifact));
    CHECK(f.ledger.allocation_registry_size() == alloc_baseline);
    snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::clean_stash_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_descriptor_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
    const auto empty = f.catalog->snapshot();
    CHECK(empty.blobs == 0 && empty.stashes == 0 &&
          empty.references == 0);
}

static void test_catalog_all_shard_failures_and_rollback() {
    for (uint32_t mode = 0; mode < 6; ++mode) {
        catalog_fixture f;
        auto completions = f.completions();
        llama_vbr_artifact_publish_fault fault;
        llama_vbr_artifact_publish_status expected;
        switch (mode) {
            case 0:
                completions.pop_back();
                expected =
                    llama_vbr_artifact_publish_status::missing_completion;
                break;
            case 1:
                completions.back() = completions.front();
                expected =
                    llama_vbr_artifact_publish_status::duplicate_completion;
                break;
            case 2:
                completions[1].success = false;
                expected =
                    llama_vbr_artifact_publish_status::shard_failed;
                break;
            case 3:
                fault.fail_stage_at = 1;
                expected =
                    llama_vbr_artifact_publish_status::stage_failed;
                break;
            case 4:
                fault.fail_commit_at = 2;
                expected =
                    llama_vbr_artifact_publish_status::commit_failed;
                break;
            default:
                fault.fail_after_commit = true;
                expected =
                    llama_vbr_artifact_publish_status::publication_failed;
                break;
        }
        const auto result =
            f.catalog->publish(f.package, completions, f.budget, fault);
        CHECK(result.status == expected);
        const auto catalog_state = f.catalog->snapshot();
        CHECK(catalog_state.blobs == 0);
        CHECK(catalog_state.stashes == 0);
        CHECK(catalog_state.references == 0);
        const auto ledger_state = f.ledger.snapshot();
        CHECK(ledger_state.live_ops == 0);
        for (const auto & row : f.package.manifest.accounting) {
            const auto category =
                vbr_artifact_accounting_category(row.role);
            const auto domain =
                row.domain.residency ==
                        llama_cache_acct_residency::device
                    ? f.bindings[row.domain.device_ordinal].domain
                    : f.host;
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value = catalog_cell(
                    ledger_state, category, domain, measure);
                CHECK(value.state == llama_cache_acct_known::known);
                CHECK(value.value == 0);
            }
        }
    }
}

static void test_catalog_destructor_releases_live_references() {
    catalog_fixture f;
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    const auto second =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::adopted);
    CHECK(f.ledger.snapshot().live_ops > 0);

    f.catalog.reset();
    const auto snapshot = f.ledger.snapshot();
    CHECK(snapshot.live_ops == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::unit_version_payload,
        f.bindings[0].domain,
        llama_cache_acct_measure::resident_allocated).value == 0);
    CHECK(catalog_cell(
        snapshot,
        llama_cache_acct_category::artifact_reference_metadata,
        f.host,
        llama_cache_acct_measure::resident_allocated).value == 0);
}

static void test_catalog_dedup_race() {
    catalog_fixture f;
    const auto completions = f.completions();
    std::array<llama_vbr_artifact_publish_result, 2> results;
    std::thread a([&] {
        results[0] =
            f.catalog->publish(f.package, completions, f.budget);
    });
    std::thread b([&] {
        results[1] =
            f.catalog->publish(f.package, completions, f.budget);
    });
    a.join();
    b.join();
    const bool first_published =
        results[0].status ==
            llama_vbr_artifact_publish_status::published;
    const bool second_published =
        results[1].status ==
            llama_vbr_artifact_publish_status::published;
    CHECK(first_published != second_published);
    CHECK((results[0].status ==
               llama_vbr_artifact_publish_status::adopted) !=
          (results[1].status ==
               llama_vbr_artifact_publish_status::adopted));
    const auto state = f.catalog->snapshot();
    CHECK(state.blobs == 1 && state.stashes == 1 &&
          state.references == 2);
    CHECK(state.published == 1 && state.adopted == 1);
    CHECK(f.catalog->retire(results[0].reference_artifact));
    CHECK(f.catalog->retire(results[1].reference_artifact));
}

static void test_catalog_full_id_interning_and_stash_dedup() {
    catalog_fixture f;
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto second = f.catalog->publish(changed, {
        { 0, 1, true,  true, changed_storage.stash1.bytes },
        { 0, 0, false, true, changed_storage.payload0.bytes },
        { 0, 0, true,  true, changed_storage.stash0.bytes },
        { 0, 1, false, true, changed_storage.payload1.bytes },
    }, f.budget);
    CHECK(second.status ==
          llama_vbr_artifact_publish_status::published);
    CHECK(second.reference_artifact != first.reference_artifact);
    CHECK(second.unit_content != first.unit_content);
    CHECK(second.reference_lineage != first.reference_lineage);
    const auto state = f.catalog->snapshot();
    CHECK(state.blobs == 2);
    CHECK(state.stashes == 1);
    CHECK(state.references == 2);
    const auto accounting = f.ledger.snapshot();
    for (const auto & binding : f.bindings) {
        CHECK(catalog_cell(
            accounting,
            llama_cache_acct_category::clean_stash_payload,
            binding.domain,
            llama_cache_acct_measure::resident_allocated).value == 8);
    }
    CHECK(f.catalog->retire(first.reference_artifact));
    CHECK(f.catalog->retire(second.reference_artifact));
}

static void test_catalog_capacity_sequential_and_temporaries() {
    catalog_fixture f;
    for (auto & device : f.budget.devices) {
        device.physical_total = 28;
        device.physical_free = 28;
        device.configured_cache_cap = 28;
        device.cache_cap_state =
            llama_cache_budget_capacity_state::known;
    }
    const auto first =
        f.catalog->publish(f.package, f.completions(), f.budget);
    CHECK(first.status ==
          llama_vbr_artifact_publish_status::published);

    fixture_storage changed_storage;
    changed_storage.payload0.bytes[0] ^= 1;
    auto changed = make_package(changed_storage);
    const auto refused =
        f.catalog->publish(changed, {
            { 0, 0, false, true, changed_storage.payload0.bytes },
            { 0, 1, false, true, changed_storage.payload1.bytes },
            { 0, 0, true, true, changed_storage.stash0.bytes },
            { 0, 1, true, true, changed_storage.stash1.bytes },
        }, f.budget);
    CHECK(refused.status ==
          llama_vbr_artifact_publish_status::admission_refused);
    CHECK(f.catalog->snapshot().references == 1);
    CHECK(f.ledger.snapshot().live_ops > 0);

    llama_cache_budget_config generous = f.budget;
    for (auto & device : generous.devices) {
        device.physical_total = 100;
        device.physical_free = 70;
        device.configured_cache_cap = 100;
    }
    const auto domain = f.bindings[0].domain;
    llama_cache_authority_request staging;
    staging.category =
        llama_cache_acct_category::transfer_staging;
    staging.domain = domain;
    staging.expected_logical = 40;
    staging.expected_resident = 40;
    auto held = llama_cache_admit_reservation(
        f.ledger, generous, staging);
    CHECK(held.status == llama_cache_admission_status::admitted);

    llama_cache_authority_request workspace = staging;
    workspace.category =
        llama_cache_acct_category::codec_workspace;
    workspace.expected_logical = 70;
    workspace.expected_resident = 70;
    auto blocked = llama_cache_admit_reservation(
        f.ledger, generous, workspace);
    CHECK(blocked.status ==
          llama_cache_admission_status::exceeds_budget);
    CHECK(!blocked.claim.has_op());
    CHECK(f.catalog->retire(first.reference_artifact));
}

int main() {
    test_golden_and_native_lineage();
    test_identity_and_reference_separation();
    test_fail_closed_decode();
    test_encoder_rejects_source_mutation();
    test_validation_and_ordering();
    test_companion_payload();
    test_stream_larger_than_capture_ring();
    test_catalog_streaming_protocol();
    test_catalog_multi_unit_atomic_publish();
    test_catalog_streaming_companion_lifetime();
    test_catalog_charge_once_and_retire();
    test_catalog_all_shard_failures_and_rollback();
    test_catalog_destructor_releases_live_references();
    test_catalog_dedup_race();
    test_catalog_full_id_interning_and_stash_dedup();
    test_catalog_capacity_sequential_and_temporaries();
    if (failures != 0) {
        fprintf(stderr, "%d VBR artifact test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact format: PASS\n");
    return 0;
}
