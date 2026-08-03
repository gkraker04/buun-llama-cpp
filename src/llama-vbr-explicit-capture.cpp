#include "llama-vbr-explicit-capture.h"

#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-tree.h"
#include "llama-sha256.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-operation.h"
#include "turbo-rotation-data.h"

#include "ggml-turbo-meansub.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

std::array<uint8_t, 32> representation_hash_file_or_marker(
        const char * tag,
        const char * path,
        uint32_t type,
        bool value_side,
        const vbr_explicit_representation_policy & policy,
        bool & ok) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u32(type);
    writer.u32(value_side);
    if (path == nullptr || path[0] == '\0') {
        static constexpr char BUILTIN[] =
            "compiled-in/build-identity";
        writer.string(BUILTIN, sizeof(BUILTIN) - 1);
        if (policy.build_identity == nullptr ||
            policy.build_identity_len == 0) {
            ok = false;
            return {};
        }
        writer.string(
            policy.build_identity, policy.build_identity_len);
        return writer.finish();
    }
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        ok = false;
        return {};
    }
    std::array<uint8_t, 64*1024> buffer;
    for (;;) {
        const size_t size =
            fread(buffer.data(), 1, buffer.size(), file);
        if (size != 0) {
            writer.bytes(buffer.data(), size);
        }
        if (size != buffer.size()) {
            if (ferror(file)) {
                ok = false;
            }
            break;
        }
    }
    fclose(file);
    return ok ? writer.finish() : std::array<uint8_t, 32>{};
}

const char * representation_override(
        int32_t type,
        bool value_side) {
    switch (type) {
        case GGML_TYPE_TURBO8_0:
            return std::getenv("TURBO_CB_T8");
        case GGML_TYPE_TURBO4_0:
            return std::getenv("TURBO_CB_T4");
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO_TCQ_CB_V" : "TURBO_TCQ_CB_K");
            return side ? side : std::getenv("TURBO_TCQ_CB");
        }
        case GGML_TYPE_TURBO1_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO1_TCQ_CB_V" : "TURBO1_TCQ_CB_K");
            return side ? side : std::getenv("TURBO1_TCQ_CB");
        }
        default:
            return nullptr;
    }
}

std::array<uint8_t, 32> representation_rotation_identity(
        int32_t type,
        bool value_side) {
    llama_sha256_writer writer;
    static constexpr char DOMAIN[] =
        "buun.vbr.codec-rotation/v1";
    writer.string(DOMAIN, sizeof(DOMAIN) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);
    const auto * matrix = value_side
        ? TURBO_ROTATION_RT : TURBO_ROTATION_R;
    writer.bytes(matrix, 128*128*sizeof(matrix[0]));
    return writer.finish();
}

std::array<uint8_t, 32> representation_meansub_identity(
        int32_t type,
        bool value_side,
        const vbr_explicit_representation_policy & policy,
        bool & ok) {
    llama_sha256_writer writer;
    static constexpr char DOMAIN[] =
        "buun.vbr.codec-meansub/v1";
    writer.string(DOMAIN, sizeof(DOMAIN) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);

    const char * disabled = std::getenv("TURBO_MEANSUB_OFF");
    if (disabled != nullptr) {
        static constexpr char OFF[] = "disabled";
        writer.string(OFF, sizeof(OFF) - 1);
        return writer.finish();
    }
    const char * path = std::getenv(
        value_side ? "TURBO_VMEAN_SUB" : "TURBO_KMEAN_SUB");
    if (path != nullptr && path[0] != '\0') {
        return representation_hash_file_or_marker(
            "buun.vbr.codec-meansub-file/v1",
            path, uint32_t(type), value_side, policy, ok);
    }

    int max_layers = 0;
    int max_channels = 0;
    int live_layers = 0;
    const float * active = ggml_turbo_meansub_table(
        policy.turbo_meansub_id, value_side ? 1 : 0,
        &max_layers, &max_channels, &live_layers);
    if (active == nullptr || max_layers <= 0 ||
        max_channels <= 0 || live_layers <= 0) {
        static constexpr char INACTIVE[] = "inactive";
        writer.string(INACTIVE, sizeof(INACTIVE) - 1);
        return writer.finish();
    }
    if (size_t(max_layers) >
            std::numeric_limits<size_t>::max() /
                size_t(max_channels) ||
        size_t(max_layers)*size_t(max_channels) >
            std::numeric_limits<size_t>::max() / sizeof(float)) {
        ok = false;
        return {};
    }
    writer.u32(uint32_t(max_layers));
    writer.u32(uint32_t(max_channels));
    writer.u32(uint32_t(live_layers));
    writer.bytes(
        active,
        size_t(max_layers)*size_t(max_channels)*sizeof(float));
    return writer.finish();
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(digest.begin(), digest.end(),
        [](uint8_t value) { return value != 0; });
}

std::array<uint8_t, 32> tagged_digest(
        const char * tag,
        uint64_t a,
        uint64_t b = 0) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u64(a);
    writer.u64(b);
    return writer.finish();
}

class vector_io_writer final : public llama_io_write_i {
public:
    void write(const void * source, size_t size) override {
        if (size > std::numeric_limits<size_t>::max() - bytes.size()) {
            throw std::bad_alloc();
        }
        const auto * begin = static_cast<const uint8_t *>(source);
        bytes.insert(bytes.end(), begin, begin + size);
    }

    void write_tensor(
            ggml_tensor * tensor,
            size_t offset,
            size_t size) override {
        const size_t old = bytes.size();
        if (size > std::numeric_limits<size_t>::max() - old) {
            throw std::bad_alloc();
        }
        bytes.resize(old + size);
        ggml_backend_tensor_get(tensor, bytes.data() + old, offset, size);
    }

    size_t n_bytes() override {
        return bytes.size();
    }

    std::vector<uint8_t> bytes;
};

class counting_io_writer final : public llama_io_write_i {
public:
    void write(const void *, size_t size) override {
        add(size);
    }
    void write_tensor(ggml_tensor *, size_t, size_t size) override {
        add(size);
    }
    size_t n_bytes() override {
        return bytes;
    }
    void add(size_t size) {
        if (size > std::numeric_limits<size_t>::max() - bytes) {
            throw std::bad_alloc();
        }
        bytes += size;
    }
    size_t bytes = 0;
};

void add_accounting(
        std::vector<vbr_artifact_portable_accounting_row> & rows,
        vbr_artifact_accounting_role role,
        const vbr_artifact_portable_domain & domain,
        uint64_t bytes) {
    for (auto & row : rows) {
        if (row.role == role && row.domain == domain) {
            if (bytes > UINT64_MAX - row.logical_bytes ||
                bytes > UINT64_MAX - row.resident_bytes) {
                throw std::overflow_error("artifact accounting overflow");
            }
            row.logical_bytes += bytes;
            row.resident_bytes += bytes;
            return;
        }
    }
    rows.push_back({ role, domain, bytes, bytes,
        llama_cache_acct_attr_kind::artifact });
}

vbr_artifact_portable_domain portable_domain(
        uint32_t topology,
        uint16_t ordinal) {
    return {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        topology,
        ordinal,
    };
}

vbr_explicit_capture_status stream_status(
        vbr_capture_stream_status status) {
    switch (status) {
        case vbr_capture_stream_status::ok:
            return vbr_explicit_capture_status::ok;
        case vbr_capture_stream_status::ring_unavailable:
            return vbr_explicit_capture_status::ring_unavailable;
        case vbr_capture_stream_status::transfer_failed:
            return vbr_explicit_capture_status::transfer_failed;
        case vbr_capture_stream_status::short_read:
            return vbr_explicit_capture_status::short_read;
        case vbr_capture_stream_status::hash_mismatch:
            return vbr_explicit_capture_status::hash_mismatch;
        case vbr_capture_stream_status::accounting_unavailable:
        case vbr_capture_stream_status::stage_failed:
        case vbr_capture_stream_status::commit_failed:
            return vbr_explicit_capture_status::accounting_failed;
        case vbr_capture_stream_status::accounting_refused:
            return vbr_explicit_capture_status::admission_refused;
        case vbr_capture_stream_status::publication_failed:
            return vbr_explicit_capture_status::publication_failed;
        case vbr_capture_stream_status::format_rejected:
            return vbr_explicit_capture_status::dedup_validation_failed;
        case vbr_capture_stream_status::invalid_argument:
        case vbr_capture_stream_status::duplicate_segment:
        case vbr_capture_stream_status::missing_segment:
        case vbr_capture_stream_status::late_segment:
        case vbr_capture_stream_status::internal_error:
        case vbr_capture_stream_status::_count:
            return vbr_explicit_capture_status::internal_error;
    }
    return vbr_explicit_capture_status::internal_error;
}

} // namespace

bool vbr_explicit_capture_representation_identity(
        const void * context,
        int32_t current_type,
        bool value_side,
        vbr_explicit_representation_identity & output) noexcept {
    try {
        if (context == nullptr ||
            current_type < 0 || current_type >= GGML_TYPE_COUNT) {
            return false;
        }
        const auto & policy =
            *static_cast<const vbr_explicit_representation_policy *>(
                context);
        output.codec_id = uint32_t(current_type) + 1;
        output.codec_version = 1;
        bool ok = true;
        output.codebook_digest =
            representation_hash_file_or_marker(
                "buun.vbr.codec-codebook/v1",
                representation_override(current_type, value_side),
                uint32_t(current_type), value_side, policy, ok);
        output.rotation_digest =
            representation_rotation_identity(
                current_type, value_side);
        output.meansub_digest =
            representation_meansub_identity(
                current_type, value_side, policy, ok);
        return ok;
    } catch (...) {
        return false;
    }
}

class vbr_live_capture_adapter {
public:
    struct child {
        uint32_t child_id = 0;
        checkpoint_child_dependency_mode dependency_mode =
            checkpoint_child_dependency_mode::absent;
        llama_kv_cache * cache = nullptr;
        std::vector<llama_kv_cache::vbr_capture_unit_plan> units;
        llama_kv_cache::vbr_capture_stability_token stability;
        vbr_checkpoint_generation_controller generation;
        std::vector<vbr_artifact_stream_placement> placements;
    };

    static bool settle(llama_kv_cache & cache) {
        return cache.vbr_capture_settle();
    }

    static bool runtime_pools(
            llama_kv_cache & cache,
            std::vector<vbr_explicit_capture_runtime_pool> & output) {
        if (cache.other != nullptr) {
            return runtime_pools(*cache.other, output);
        }
        const auto instance = cache.vbr_instance_id();
        const auto * tracker =
            cache.vbr_generation_tracker_get();
        if (!cache.vbr_operation_armed() ||
            tracker == nullptr || !tracker->active() ||
            !vbr_controller_instance_id_is_set(instance) ||
            cache.vbr_pools_.empty()) {
            return false;
        }
        for (const auto & pool : cache.vbr_pools_) {
            if (pool.vmm == nullptr || pool.buf == nullptr ||
                pool.device < 0) {
                return false;
            }
            const auto backend_device =
                ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(pool.buf));
            if (backend_device == nullptr) {
                return false;
            }
            const auto duplicate = std::find_if(
                output.begin(), output.end(),
                [&](const auto & current) {
                    return current.instance_id == instance &&
                           current.device == pool.device;
                });
            if (duplicate != output.end()) {
                if (duplicate->backend_device != backend_device ||
                    (duplicate->backend != nullptr &&
                     pool.backend != nullptr &&
                     duplicate->backend != pool.backend)) {
                    return false;
                }
                continue;
            }
            output.push_back({
                instance, pool.device, backend_device, pool.backend,
            });
        }
        return true;
    }

    static bool capture_metadata(
            llama_kv_cache & cache,
            uint32_t child_id,
            checkpoint_child_dependency_mode mode,
            llama_seq_id sequence,
            llama_pos frontier,
            const std::vector<vbr_explicit_capture_pool_binding> & bindings,
            child & output,
            vbr_explicit_generation_failure & failure,
            vbr_explicit_size_failure & size_failure) {
        failure = vbr_explicit_generation_failure::none;
        size_failure = vbr_explicit_size_failure::none;
        llama_kv_cache::vbr_capture_unit_request request;
        request.child_id = child_id;
        request.bindings = &bindings;
        output.child_id = child_id;
        output.dependency_mode = mode;
        output.cache = &cache;
        // Snapshot the byte geometry/generation token first, then capture
        // ownership.  The final stability reread binds the ownership record
        // to that exact token: a mutation between these two calls advances a
        // monotone controller serial or unit publish_seq and fails closed.
        if (!cache.vbr_capture_size_pass(
                request, output.units, output.stability,
                &size_failure)) {
            failure = vbr_explicit_generation_failure::size_pass;
            return false;
        }
        vbr_artifact_stream_placement placement;
        auto * placement_out = mode ==
                checkpoint_child_dependency_mode::live_guarded ?
            &placement : nullptr;
        if (!cache.vbr_capture_generation_record(
                child_id, mode, sequence, frontier,
                output.generation, placement_out, &failure)) {
            if (failure == vbr_explicit_generation_failure::none) {
                failure =
                    vbr_explicit_generation_failure::internal_error;
            }
            return false;
        }
        if (placement_out != nullptr) {
            output.placements.push_back(std::move(placement));
        }
        if (!cache.vbr_capture_stability_matches(output.stability)) {
            failure =
                vbr_explicit_generation_failure::stability_reread_failed;
            return false;
        }
        return true;
    }

    static bool stable(const child & value) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stability_matches(value.stability);
    }

    static bool stream(
            const child & value,
            const llama_kv_cache::vbr_capture_unit_plan & unit,
            vbr_unit_build & sink,
            vbr_pinned_chunk_ring & ring,
            vbr_capture_stream_stats & stats) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stream_unit(
                   unit, sink, ring, stats);
    }
};

bool vbr_explicit_capture_runtime_pools(
        llama_memory_i & memory,
        std::vector<vbr_explicit_capture_runtime_pool> & pools,
        uint32_t & attention_children) noexcept {
    pools.clear();
    attention_children = 0;
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        for (const auto & node : tree) {
            if (node.attention == nullptr) {
                continue;
            }
            ++attention_children;
            if (!vbr_live_capture_adapter::runtime_pools(
                    *node.attention, pools)) {
                pools.clear();
                attention_children = 0;
                return false;
            }
        }
        return attention_children != 0 && !pools.empty();
    } catch (...) {
        pools.clear();
        attention_children = 0;
        return false;
    }
}

vbr_explicit_capture_result vbr_capture_explicit_manifest(
        llama_memory_i & memory,
        const vbr_explicit_capture_request & request,
        vbr_unit_version_sink & sink,
        const vbr_explicit_capture_accounting & accounting) noexcept {
    vbr_explicit_capture_result result;
    if (!request.idle_decode_thread) {
        result.status = vbr_explicit_capture_status::slot_not_idle;
        return result;
    }
    if (request.sequence < 0 || request.frontier.next_position < 0 ||
        request.ring == nullptr || accounting.budget == nullptr ||
        request.topologies.empty() || request.pool_bindings.empty() ||
        request.representation_identity == nullptr ||
        request.identity.token_count < 0 ||
        request.token_block.size() !=
            size_t(request.identity.token_count) ||
        request.identity.execution_identity.empty() ||
        request.identity.adapter_config_identity.empty() ||
        request.identity.media_content_identity.empty()) {
        result.status = vbr_explicit_capture_status::identity_unavailable;
        return result;
    }

    try {
        result.phase = vbr_explicit_capture_phase::memory_tree;
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }

        std::vector<vbr_live_capture_adapter::child> children;
        std::vector<llama_memory_recurrent *> recurrent;
        for (const auto & node : tree) {
            if (node.attention != nullptr) {
                if (!node.attention->vbr_operation_armed()) {
                    result.status = vbr_explicit_capture_status::not_armed;
                    return result;
                }
                vbr_live_capture_adapter::child child;
                child.child_id = node.child_id;
                child.dependency_mode = node.dependency_mode;
                child.cache = node.attention;
                children.push_back(std::move(child));
            }
            if (node.recurrent != nullptr) {
                recurrent.push_back(node.recurrent);
            }
        }
        if (children.empty()) {
            result.status = vbr_explicit_capture_status::not_armed;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::settlement;
        // Settlement is deliberately before both quiescence proofs. It flushes
        // only already-deferred housekeeping and dirty stash metadata.
        for (auto & child : children) {
            if (!vbr_live_capture_adapter::settle(*child.cache)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::pre_capture_quiescence;
        std::vector<vbr_controller_instance_id> instances;
        result.phase =
            vbr_explicit_capture_phase::metadata_and_manifest;
        for (auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_controller_instance_id_is_set(instance) ||
                std::any_of(instances.begin(), instances.end(),
                    [&](const auto & current) {
                        return current == instance;
                    })) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            instances.push_back(instance);
            if (vbr_recovery_pending_for(instance)) {
                result.status = vbr_explicit_capture_status::recovery_pending;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        for (auto & child : children) {
            if (!vbr_live_capture_adapter::capture_metadata(
                    *child.cache, child.child_id, child.dependency_mode,
                    request.sequence, request.frontier.next_position,
                    request.pool_bindings, child,
                    result.generation_failure,
                    result.size_failure)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        std::vector<vbr_identity_policy_digest_row> identity_policy;
        identity_policy.reserve(children.size());
        for (const auto & child : children) {
            identity_policy.push_back({
                child.child_id,
                child.dependency_mode,
                child.stability.lineage_uuid,
            });
        }
        const auto identity_policy_order_digest =
            vbr_identity_policy_digest(
                request.frontier, identity_policy);
        if (digest_nonzero(request.identity_policy_order_digest) &&
            request.identity_policy_order_digest !=
                identity_policy_order_digest) {
            result.status =
                vbr_explicit_capture_status::identity_unavailable;
            return result;
        }

        // Recurrent state uses the existing exact state codec. Accelerator
        // companions use equally typed injected existing codecs.
        vbr_artifact_package package;
        package.topologies = request.topologies;
        package.manifest.identity = request.identity;
        package.manifest.token_block.tokens = request.token_block;
        package.manifest.identity_policy_order_digest =
            identity_policy_order_digest;
        package.manifest.generation.version = 1;
        package.manifest.generation.status =
            vbr_checkpoint_generation_status::complete;
        package.manifest.generation.identity_policy_order_digest =
            identity_policy_order_digest;

        uint32_t global_unit = 0;
        for (auto & child : children) {
            package.manifest.generation.controllers.push_back(
                child.generation);
            package.manifest.stream_placements.insert(
                package.manifest.stream_placements.end(),
                child.placements.begin(), child.placements.end());
            vbr_artifact_controller_policy policy;
            policy.child_id = child.child_id;
            policy.dependency_mode = child.dependency_mode;
            policy.degrade_order_digest =
                child.stability.degrade_order_digest;
            policy.policy_digest = child.stability.policy_digest;
            policy.cursor = child.stability.degrade_cursor;
            policy.floor_type = child.stability.floor_type;
            policy.pressure_independent_settings =
                child.stability.pressure_independent_settings;
            policy.n_stream = child.units.front().n_stream;
            policy.unified = child.units.front().unified;
            policy.wm_cells = child.units.front().wm_cells;
            std::vector<ggml_type> current_types;
            current_types.reserve(child.units.size());
            for (const auto & unit : child.units) {
                current_types.push_back(
                    static_cast<ggml_type>(unit.generation.current_type));
            }
            policy.current_type_vector_digest =
                vbr_type_vector_digest(current_types);
            policy.completed_wave = child.stability.completed_wave;
            package.manifest.controller_policy.push_back(policy);

            for (auto & plan : child.units) {
                plan.capture_index = global_unit;
                vbr_artifact_unit_blob blob;
                auto & descriptor = blob.descriptor;
                descriptor.child_id = child.child_id;
                descriptor.logical_unit_id = plan.logical_unit;
                descriptor.lineage_uuid = child.stability.lineage_uuid;
                descriptor.repr_gen = plan.generation.repr_gen;
                descriptor.current_type = plan.generation.current_type;
                descriptor.last_source_type =
                    plan.generation.last_source_type;
                descriptor.promote_hops = plan.generation.promote_hops;
                descriptor.last_transition =
                    plan.generation.last_transition;
                descriptor.representation.kind =
                    plan.generation.current_type == GGML_TYPE_F16
                        ? vbr_artifact_representation_kind::raw
                        : vbr_artifact_representation_kind::approximate;
                vbr_explicit_representation_identity representation;
                if (!request.representation_identity(
                        request.representation_context,
                        plan.generation.current_type,
                        plan.is_v, representation) ||
                    representation.codec_id == 0 ||
                    representation.codec_version == 0 ||
                    !digest_nonzero(representation.codebook_digest) ||
                    !digest_nonzero(representation.rotation_digest) ||
                    !digest_nonzero(representation.meansub_digest)) {
                    result.status =
                        vbr_explicit_capture_status::
                            identity_unavailable;
                    return result;
                }
                descriptor.representation.codec_id =
                    representation.codec_id;
                descriptor.representation.codec_version =
                    representation.codec_version;
                llama_sha256_writer representation_hash;
                static constexpr char REPRESENTATION_DOMAIN[] =
                    "buun.vbr.capture/representation";
                representation_hash.string(
                    REPRESENTATION_DOMAIN,
                    sizeof(REPRESENTATION_DOMAIN) - 1);
                representation_hash.u32(
                    uint32_t(plan.generation.current_type));
                representation_hash.u32(
                    uint32_t(plan.generation.last_source_type));
                representation_hash.u32(representation.codec_id);
                representation_hash.u32(
                    representation.codec_version);
                representation_hash.bytes(
                    representation.codebook_digest.data(),
                    representation.codebook_digest.size());
                representation_hash.bytes(
                    representation.rotation_digest.data(),
                    representation.rotation_digest.size());
                representation_hash.bytes(
                    representation.meansub_digest.data(),
                    representation.meansub_digest.size());
                descriptor.representation.reference_digest =
                    representation_hash.finish();
                descriptor.representation.source_loss_history =
                    plan.generation.promote_hops;
                descriptor.side = plan.is_v
                    ? vbr_artifact_side::value
                    : vbr_artifact_side::key;
                descriptor.n_stream = plan.n_stream;
                descriptor.unified = plan.unified;
                descriptor.wm_cells = plan.wm_cells;
                descriptor.rank = 2;
                uint64_t total_columns = 0;
                for (const auto & shard : plan.shards) {
                    if (shard.columns >
                        std::numeric_limits<uint64_t>::max() -
                            total_columns) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    total_columns += shard.columns;
                }
                descriptor.dimensions =
                    std::array<uint64_t, 4> {
                        plan.wm_cells, total_columns, 0, 0,
                    };
                descriptor.row_alignment = 1;
                descriptor.row_codec_version = 1;
                descriptor.codebook_digest =
                    representation.codebook_digest;
                descriptor.rotation_digest =
                    representation.rotation_digest;
                descriptor.meansub_digest =
                    representation.meansub_digest;
                bool has_stash = false;
                uint32_t stash_rows = 0;
                uint64_t logical_offset = 0;
                for (const auto & shard : plan.shards) {
                    vbr_artifact_shard_descriptor wire;
                    wire.shard_index = shard.shard_index;
                    wire.topology_index = shard.topology_index;
                    wire.device_ordinal = shard.device_ordinal;
                    wire.logical_offset = logical_offset;
                    wire.row_count = plan.wm_cells;
                    wire.column_count = shard.columns;
                    wire.row_bytes = shard.row_bytes;
                    wire.payload_bytes = shard.payload_bytes;
                    descriptor.shards.push_back(wire);
                    logical_offset += shard.columns;
                    add_accounting(
                        package.manifest.accounting,
                        vbr_artifact_accounting_role::unit_payload,
                        portable_domain(
                            shard.topology_index,
                            shard.device_ordinal),
                        shard.payload_bytes);
                    if (shard.stash_bytes != 0) {
                        has_stash = true;
                        stash_rows =
                            uint32_t(shard.stash_bytes /
                                     (shard.columns*sizeof(uint16_t)));
                    }
                }
                descriptor.clean_stash_state = has_stash
                    ? vbr_artifact_clean_stash_state::present
                    : vbr_artifact_clean_stash_state::absent_at_source;
                if (has_stash) {
                    if (total_columns >
                        std::numeric_limits<uint64_t>::max() /
                            sizeof(uint16_t)) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    descriptor.clean_stash.valid_rows = stash_rows;
                    descriptor.clean_stash.domain =
                        vbr_repr_domain::tapped;
                    descriptor.clean_stash.row_count = stash_rows;
                    descriptor.clean_stash.column_count =
                        total_columns;
                    descriptor.clean_stash.row_bytes =
                        total_columns*sizeof(uint16_t);
                    logical_offset = 0;
                    for (const auto & shard : plan.shards) {
                        if (shard.stash_bytes == 0) {
                            result.status =
                                vbr_explicit_capture_status::stash_inconsistent;
                            return result;
                        }
                        vbr_artifact_shard_descriptor wire;
                        wire.shard_index = shard.shard_index;
                        wire.topology_index = shard.topology_index;
                        wire.device_ordinal = shard.device_ordinal;
                        wire.logical_offset = logical_offset;
                        wire.row_count = stash_rows;
                        wire.column_count = shard.columns;
                        wire.row_bytes = shard.columns*sizeof(uint16_t);
                        wire.payload_bytes = shard.stash_bytes;
                        descriptor.clean_stash.shards.push_back(wire);
                        logical_offset += shard.columns;
                        add_accounting(
                            package.manifest.accounting,
                            vbr_artifact_accounting_role::
                                clean_stash_payload,
                            portable_domain(
                                shard.topology_index,
                                shard.device_ordinal),
                            shard.stash_bytes);
                    }
                }
                package.unit_blobs.push_back(std::move(blob));

                vbr_artifact_unit_reference reference;
                reference.lineage_uuid = child.stability.lineage_uuid;
                reference.logical_unit_id = plan.logical_unit;
                reference.repr_gen = plan.generation.repr_gen;
                reference.authorized_stream_refs = { 0 };
                if (has_stash) {
                    if (child.generation.streams.empty()) {
                        result.status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return result;
                    }
                    const auto & stream =
                        child.generation.streams.front();
                    reference.has_stash_reference = true;
                    reference.stash_reference.valid_rows = stash_rows;
                    reference.stash_reference.domain =
                        vbr_repr_domain::tapped;
                    reference.stash_reference.row_count = stash_rows;
                    reference.stash_reference.column_count =
                        total_columns;
                    reference.stash_reference.row_bytes =
                        total_columns*sizeof(uint16_t);
                    reference.stash_reference.captured_sink_count =
                        stream.captured_dependency_count;
                    reference.stash_reference.covered_sink_pages =
                        stream.pages;
                }
                package.manifest.unit_references.push_back(
                    std::move(reference));
                ++global_unit;
            }
        }

        struct pending_companion {
            llama_memory_recurrent * recurrent = nullptr;
            const vbr_explicit_companion_provider * provider = nullptr;
            uint64_t bytes = 0;
        };
        std::vector<pending_companion> pending_companions;
        for (auto * memory_recurrent : recurrent) {
            counting_io_writer writer;
            memory_recurrent->state_write(
                writer, request.sequence, 0);
            if (writer.n_bytes() == 0) {
                result.status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                return result;
            }
            pending_companions.push_back({
                memory_recurrent, nullptr, writer.n_bytes(),
            });
            vbr_artifact_companion_payload companion;
            companion.kind = vbr_artifact_companion_kind::recurrent;
            companion.format_version = 1;
            companion.build_identity_digest = tagged_digest(
                "buun.vbr.capture/recurrent-codec", 1);
            companion.domain = {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
            companion.payload_bytes = writer.n_bytes();
            package.companions.push_back(companion);
        }
        for (const auto & provider : request.companions) {
            if (provider.size == nullptr ||
                provider.capture == nullptr ||
                provider.format_version == 0 ||
                !digest_nonzero(provider.build_identity_digest)) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            uint64_t companion_size = 0;
            if (!provider.size(
                    provider.context, request.sequence,
                    companion_size) ||
                companion_size == 0 ||
                companion_size >
                    std::numeric_limits<size_t>::max()) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            pending_companions.push_back({
                nullptr, &provider, companion_size,
            });
            vbr_artifact_companion_payload companion;
            companion.kind = provider.kind;
            companion.format_version = provider.format_version;
            companion.build_identity_digest =
                provider.build_identity_digest;
            companion.domain = provider.domain;
            companion.payload_bytes = companion_size;
            package.companions.push_back(companion);
        }
        package.manifest.companions = package.companions;
        for (const auto & companion : package.companions) {
            add_accounting(
                package.manifest.accounting,
                companion.kind ==
                    vbr_artifact_companion_kind::recurrent
                    ? vbr_artifact_accounting_role::recurrent_payload
                    : vbr_artifact_accounting_role::
                        typed_accelerator_payload,
                companion.domain, companion.payload_bytes);
            result.companion_bytes += companion.payload_bytes;
        }
        const auto metadata_domain = vbr_artifact_portable_domain {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::descriptor_metadata,
            metadata_domain,
            std::max<uint64_t>(1, package.unit_blobs.size()*256));
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::reference_metadata,
            metadata_domain,
            std::max<uint64_t>(1,
                package.manifest.unit_references.size()*128));
        package.manifest.consistency.kind =
            vbr_artifact_consistency_kind::capture_exact;

        result.phase =
            vbr_explicit_capture_phase::pre_transfer_stability;
        // Exact equality immediately before the first data byte.
        for (const auto & child : children) {
            if (!vbr_live_capture_adapter::stable(child)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::accounting_configuration;
        if (accounting.prepare != nullptr &&
            !accounting.prepare(accounting.context, package)) {
            result.status = vbr_explicit_capture_status::accounting_failed;
            return result;
        }
        result.phase =
            vbr_explicit_capture_phase::reservation_preparation;
        vbr_capture_stream_status begin_status;
        auto build = sink.begin_capture(
            package, *accounting.budget, accounting.fault,
            begin_status, &result.begin_diagnostics);
        if (!build) {
            result.inner_stream_status = begin_status;
            result.status = stream_status(begin_status);
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::companion_capture;
        // Durable + transfer-staging claims now exist. Only at this point may
        // companion codecs allocate their pageable byte images.
        for (size_t i = 0; i < pending_companions.size(); ++i) {
            std::vector<uint8_t> bytes;
            const auto & pending = pending_companions[i];
            if (pending.recurrent != nullptr) {
                vector_io_writer writer;
                pending.recurrent->state_write(
                    writer, request.sequence, 0);
                bytes = std::move(writer.bytes);
            } else if (pending.provider == nullptr ||
                       !pending.provider->capture(
                           pending.provider->context,
                           request.sequence, bytes)) {
                result.status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                return result;
            }
            // Companion size→data coherence relies on the required idle-slot,
            // no-decode route invariant. F3.3 must enforce that invariant;
            // size equality is intentionally the F3.2 guard, not a second
            // content-hash pass over the existing companion codecs.
            if (bytes.size() != pending.bytes) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
            auto chain = std::make_shared<artifact_segment_chain>();
            static constexpr size_t CHUNK = 1024*1024;
            for (size_t offset = 0; offset < bytes.size();) {
                const size_t size = std::min(
                    CHUNK, bytes.size() - offset);
                if (!chain->append(
                        bytes.data() + offset, size)) {
                    result.status =
                        vbr_explicit_capture_status::accounting_failed;
                    return result;
                }
                offset += size;
            }
            vbr_verified_companion verified;
            verified.companion_index = uint32_t(i);
            verified.bytes = chain;
            verified.streaming_digest =
                vbr_capture_stream_digest(*chain);
            const auto accepted =
                build->accept_verified_companion(verified);
            if (accepted != vbr_capture_stream_status::ok) {
                result.inner_stream_status = accepted;
                result.status = stream_status(accepted);
                return result;
            }
        }

        result.phase = vbr_explicit_capture_phase::unit_transfer;
        uint32_t unit_index = 0;
        for (const auto & child : children) {
            for (const auto & plan : child.units) {
                vbr_capture_stream_status unit_status;
                auto unit = build->begin_unit(unit_index, unit_status);
                if (!unit) {
                    result.inner_stream_status = unit_status;
                    result.status = stream_status(unit_status);
                    return result;
                }
                vbr_capture_stream_stats stats;
                if (!vbr_live_capture_adapter::stream(
                        child, plan, *unit, *request.ring, stats)) {
                    result.status =
                        vbr_explicit_capture_status::transfer_failed;
                    return result;
                }
                result.chunks += stats.chunks;
                result.backpressure_waits += stats.backpressure_waits;
                result.event_completions += stats.event_completions;
                result.synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                const auto sealed = unit->seal_unit();
                if (sealed != vbr_capture_stream_status::ok) {
                    result.inner_stream_status = sealed;
                    result.status =
                        vbr_explicit_capture_status::hash_mismatch;
                    return result;
                }
                for (const auto & shard : plan.shards) {
                    if (shard.payload_bytes >
                            UINT64_MAX - result.payload_bytes ||
                        shard.stash_bytes >
                            UINT64_MAX - result.stash_bytes) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    result.payload_bytes += shard.payload_bytes;
                    result.stash_bytes += shard.stash_bytes;
                }
                ++unit_index;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::post_transfer_stability;
        // Both levels of stability and quiescence are re-read after all D2H
        // completions, before the catalog's final reference publication.
        for (const auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_live_capture_adapter::stable(child) ||
                vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::publication;
        result.sink = build->publish_reference();
        result.inner_stream_status = result.sink.status;
        result.status = stream_status(result.sink.status);
        result.controllers = children.size();
        result.units = unit_index;
        result.companions = package.companions.size();
        if (result.status == vbr_explicit_capture_status::ok) {
            result.phase = vbr_explicit_capture_phase::complete;
        }
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        return result;
    }
}

const char * vbr_explicit_capture_phase_name(
        vbr_explicit_capture_phase phase) noexcept {
    switch (phase) {
        case vbr_explicit_capture_phase::validation: return "validation";
        case vbr_explicit_capture_phase::memory_tree: return "memory_tree";
        case vbr_explicit_capture_phase::settlement: return "settlement";
        case vbr_explicit_capture_phase::pre_capture_quiescence: return "pre_capture_quiescence";
        case vbr_explicit_capture_phase::metadata_and_manifest: return "metadata_and_manifest";
        case vbr_explicit_capture_phase::pre_transfer_stability: return "pre_transfer_stability";
        case vbr_explicit_capture_phase::accounting_configuration: return "accounting_configuration";
        case vbr_explicit_capture_phase::reservation_preparation: return "reservation_preparation";
        case vbr_explicit_capture_phase::companion_capture: return "companion_capture";
        case vbr_explicit_capture_phase::unit_transfer: return "unit_transfer";
        case vbr_explicit_capture_phase::post_transfer_stability: return "post_transfer_stability";
        case vbr_explicit_capture_phase::publication: return "publication";
        case vbr_explicit_capture_phase::complete: return "complete";
        case vbr_explicit_capture_phase::_count: return "_count";
    }
    return "invalid";
}

const char * vbr_explicit_generation_failure_name(
        vbr_explicit_generation_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_generation_failure::none: return "none";
        case vbr_explicit_generation_failure::size_pass: return "size_pass";
        case vbr_explicit_generation_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_generation_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_generation_failure::tracker_shadow_unavailable: return "tracker_shadow_unavailable";
        case vbr_explicit_generation_failure::invalid_sequence_or_frontier: return "invalid_sequence_or_frontier";
        case vbr_explicit_generation_failure::invalid_stream: return "invalid_stream";
        case vbr_explicit_generation_failure::ownership_index_missing: return "ownership_index_missing";
        case vbr_explicit_generation_failure::ownership_view_missing: return "ownership_view_missing";
        case vbr_explicit_generation_failure::ownership_view_unavailable: return "ownership_view_unavailable";
        case vbr_explicit_generation_failure::ownership_rank_failed: return "ownership_rank_failed";
        case vbr_explicit_generation_failure::ownership_enumeration_failed: return "ownership_enumeration_failed";
        case vbr_explicit_generation_failure::ownership_cardinality_mismatch: return "ownership_cardinality_mismatch";
        case vbr_explicit_generation_failure::stream_capture_failed: return "stream_capture_failed";
        case vbr_explicit_generation_failure::controller_capture_failed: return "controller_capture_failed";
        case vbr_explicit_generation_failure::stability_reread_failed: return "stability_reread_failed";
        case vbr_explicit_generation_failure::internal_error: return "internal_error";
        case vbr_explicit_generation_failure::_count: return "_count";
    }
    return "_count";
}

const char * vbr_explicit_size_failure_name(
        vbr_explicit_size_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_size_failure::none: return "none";
        case vbr_explicit_size_failure::not_armed: return "not_armed";
        case vbr_explicit_size_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_size_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_size_failure::bindings_missing: return "bindings_missing";
        case vbr_explicit_size_failure::stream_layout: return "stream_layout";
        case vbr_explicit_size_failure::policy_snapshot: return "policy_snapshot";
        case vbr_explicit_size_failure::unit_index: return "unit_index";
        case vbr_explicit_size_failure::extents_empty: return "extents_empty";
        case vbr_explicit_size_failure::extent_missing: return "extent_missing";
        case vbr_explicit_size_failure::vmm_missing: return "vmm_missing";
        case vbr_explicit_size_failure::backend_unavailable: return "backend_unavailable";
        case vbr_explicit_size_failure::wm_cells_zero: return "wm_cells_zero";
        case vbr_explicit_size_failure::extent_type_mismatch: return "extent_type_mismatch";
        case vbr_explicit_size_failure::promote_hops_mismatch: return "promote_hops_mismatch";
        case vbr_explicit_size_failure::domain_mismatch: return "domain_mismatch";
        case vbr_explicit_size_failure::shard_disagreement: return "shard_disagreement";
        case vbr_explicit_size_failure::binding_missing: return "binding_missing";
        case vbr_explicit_size_failure::topology_order: return "topology_order";
        case vbr_explicit_size_failure::bounds: return "bounds";
        case vbr_explicit_size_failure::stash_bounds: return "stash_bounds";
        case vbr_explicit_size_failure::stability_reread: return "stability_reread";
        case vbr_explicit_size_failure::internal_error: return "internal_error";
        case vbr_explicit_size_failure::_count: return "_count";
    }
    return "_count";
}

vbr_explicit_size_failure vbr_explicit_capture_validate_extent_generation(
        uint32_t wm_cells,
        int32_t extent_type,
        uint8_t extent_promote_hops,
        const vbr_unit_generation & generation) noexcept {
    if (wm_cells == 0) {
        return vbr_explicit_size_failure::wm_cells_zero;
    }
    if (extent_type != generation.current_type) {
        return vbr_explicit_size_failure::extent_type_mismatch;
    }
    if (extent_promote_hops != generation.promote_hops) {
        return vbr_explicit_size_failure::promote_hops_mismatch;
    }
    const auto expected_domain =
        generation.current_type == GGML_TYPE_F16 ||
        generation.current_type == GGML_TYPE_TURBO8_0
            ? vbr_repr_domain::full
            : vbr_repr_domain::tapped;
    if (generation.domain != expected_domain) {
        return vbr_explicit_size_failure::domain_mismatch;
    }
    return vbr_explicit_size_failure::none;
}

const char * vbr_explicit_capture_status_name(
        vbr_explicit_capture_status status) noexcept {
    switch (status) {
        case vbr_explicit_capture_status::ok: return "ok";
        case vbr_explicit_capture_status::not_armed: return "not_armed";
        case vbr_explicit_capture_status::unsupported_layout: return "unsupported_layout";
        case vbr_explicit_capture_status::slot_not_idle: return "slot_not_idle";
        case vbr_explicit_capture_status::identity_unavailable: return "identity_unavailable";
        case vbr_explicit_capture_status::generation_unavailable: return "generation_unavailable";
        case vbr_explicit_capture_status::registry_busy: return "registry_busy";
        case vbr_explicit_capture_status::recovery_pending: return "recovery_pending";
        case vbr_explicit_capture_status::geometry_mismatch: return "geometry_mismatch";
        case vbr_explicit_capture_status::stash_inconsistent: return "stash_inconsistent";
        case vbr_explicit_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_explicit_capture_status::size_overflow: return "size_overflow";
        case vbr_explicit_capture_status::ring_unavailable: return "ring_unavailable";
        case vbr_explicit_capture_status::admission_refused: return "admission_refused";
        case vbr_explicit_capture_status::transfer_failed: return "transfer_failed";
        case vbr_explicit_capture_status::short_read: return "short_read";
        case vbr_explicit_capture_status::event_failed: return "event_failed";
        case vbr_explicit_capture_status::source_changed: return "source_changed";
        case vbr_explicit_capture_status::hash_mismatch: return "hash_mismatch";
        case vbr_explicit_capture_status::dedup_validation_failed: return "dedup_validation_failed";
        case vbr_explicit_capture_status::accounting_failed: return "accounting_failed";
        case vbr_explicit_capture_status::publication_failed: return "publication_failed";
        case vbr_explicit_capture_status::internal_error: return "internal_error";
        case vbr_explicit_capture_status::_count: return "_count";
    }
    return "_count";
}
