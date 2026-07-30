#include "llama-vbr-artifact.h"
#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
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
    descriptor.pool_uuid = { 0x0102030405060708ull, 0x1112131415161718ull };
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
    controller.pool_uuid = descriptor.pool_uuid;
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
    reference.pool_uuid = descriptor.pool_uuid;
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
    CHECK(decoded.manifest.generation.controllers[0].pool_uuid ==
          package.manifest.generation.controllers[0].pool_uuid);
    CHECK(decoded.manifest.generation.controllers[0].pool_uuid ==
          package.unit_blobs[0].descriptor.pool_uuid);
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
    CHECK(payload_changed.unit_blobs[0].descriptor.pool_uuid ==
          first.unit_blobs[0].descriptor.pool_uuid);
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
    mismatched_tuple.manifest.unit_references[0].pool_uuid.lo ^= 1;
    CHECK(vbr_artifact_encode_vector(
              mismatched_tuple, bytes, 1024*1024) ==
          vbr_artifact_status::generation_mismatch);

    // A different logical unit may intern the same immutable clean-stash
    // subobject without sharing the unit address.
    auto shared_stash = make_package(storage);
    shared_stash.unit_blobs[0].descriptor.pool_uuid.lo++;
    shared_stash.manifest.generation.controllers[0].pool_uuid.lo++;
    shared_stash.manifest.unit_references[0].pool_uuid.lo++;
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
    CHECK(tuple_before.pool_uuid ==
          package.manifest.unit_references[0].pool_uuid);
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

int main() {
    test_golden_and_native_lineage();
    test_identity_and_reference_separation();
    test_fail_closed_decode();
    test_encoder_rejects_source_mutation();
    test_validation_and_ordering();
    test_companion_payload();
    test_stream_larger_than_capture_ring();
    if (failures != 0) {
        fprintf(stderr, "%d VBR artifact test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact format: PASS\n");
    return 0;
}
