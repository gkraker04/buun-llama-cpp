#include "llama-vbr-artifact-capture.h"

#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <new>
#include <utility>

const char * vbr_capture_stream_status_name(
        vbr_capture_stream_status status) noexcept {
    switch (status) {
        case vbr_capture_stream_status::ok:                  return "ok";
        case vbr_capture_stream_status::invalid_argument:    return "invalid_argument";
        case vbr_capture_stream_status::ring_unavailable:    return "ring_unavailable";
        case vbr_capture_stream_status::transfer_failed:     return "transfer_failed";
        case vbr_capture_stream_status::short_read:          return "short_read";
        case vbr_capture_stream_status::duplicate_segment:   return "duplicate_segment";
        case vbr_capture_stream_status::missing_segment:     return "missing_segment";
        case vbr_capture_stream_status::late_segment:        return "late_segment";
        case vbr_capture_stream_status::hash_mismatch:       return "hash_mismatch";
        case vbr_capture_stream_status::format_rejected:      return "format_rejected";
        case vbr_capture_stream_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_capture_stream_status::accounting_refused:  return "accounting_refused";
        case vbr_capture_stream_status::stage_failed:         return "stage_failed";
        case vbr_capture_stream_status::commit_failed:        return "commit_failed";
        case vbr_capture_stream_status::publication_failed:  return "publication_failed";
        case vbr_capture_stream_status::internal_error:      return "internal_error";
        case vbr_capture_stream_status::_count:              break;
    }
    return "invalid";
}

const char * vbr_capture_reservation_group_name(
        vbr_capture_reservation_group group) noexcept {
    switch (group) {
        case vbr_capture_reservation_group::none: return "none";
        case vbr_capture_reservation_group::transfer_staging: return "transfer_staging";
        case vbr_capture_reservation_group::durable_artifact: return "durable_artifact";
        case vbr_capture_reservation_group::_count: break;
    }
    return "invalid";
}

const char * vbr_capture_ring_create_failure_name(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::none:
            return "none";
        case vbr_capture_ring_create_failure::invalid_geometry:
            return "invalid_geometry";
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
            return "invalid_accounting_binding";
        case vbr_capture_ring_create_failure::existing_ring_charge:
            return "existing_ring_charge";
        case vbr_capture_ring_create_failure::accounting_update_failed:
            return "accounting_update_failed";
        case vbr_capture_ring_create_failure::budget_reset_failed:
            return "budget_reset_failed";
        case vbr_capture_ring_create_failure::budget_unavailable:
            return "budget_unavailable";
        case vbr_capture_ring_create_failure::budget_exceeded:
            return "budget_exceeded";
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
            return "global_capacity_exceeded";
        case vbr_capture_ring_create_failure::invalid_lane_binding:
            return "invalid_lane_binding";
        case vbr_capture_ring_create_failure::duplicate_device_lane:
            return "duplicate_device_lane";
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
            return "host_buffer_type_unavailable";
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
            return "host_buffer_allocation_failed";
        case vbr_capture_ring_create_failure::host_buffer_too_small:
            return "host_buffer_too_small";
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
            return "host_buffer_base_unavailable";
        case vbr_capture_ring_create_failure::lane_underprovisioned:
            return "lane_underprovisioned";
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return "accounting_charge_failed";
        case vbr_capture_ring_create_failure::internal_error:
            return "internal_error";
        case vbr_capture_ring_create_failure::_count:
            break;
    }
    return "invalid";
}

namespace {

vbr_capture_stream_status capture_status_for_ring_failure(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::budget_exceeded:
            return vbr_capture_stream_status::accounting_refused;
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
        case vbr_capture_ring_create_failure::existing_ring_charge:
        case vbr_capture_ring_create_failure::accounting_update_failed:
        case vbr_capture_ring_create_failure::budget_reset_failed:
        case vbr_capture_ring_create_failure::budget_unavailable:
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return vbr_capture_stream_status::accounting_unavailable;
        case vbr_capture_ring_create_failure::internal_error:
            return vbr_capture_stream_status::internal_error;
        case vbr_capture_ring_create_failure::none:
        case vbr_capture_ring_create_failure::invalid_geometry:
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
        case vbr_capture_ring_create_failure::invalid_lane_binding:
        case vbr_capture_ring_create_failure::duplicate_device_lane:
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
        case vbr_capture_ring_create_failure::host_buffer_too_small:
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
        case vbr_capture_ring_create_failure::lane_underprovisioned:
        case vbr_capture_ring_create_failure::_count:
            return vbr_capture_stream_status::ring_unavailable;
    }
    return vbr_capture_stream_status::ring_unavailable;
}

} // namespace

struct artifact_segment_chain::impl {
    std::vector<artifact_segment> segments;
    uint64_t total = 0;
    size_t max_segment = 0;
};

artifact_segment_chain::artifact_segment_chain()
    : impl_(new impl) {}
artifact_segment_chain::~artifact_segment_chain() = default;
artifact_segment_chain::artifact_segment_chain(
        artifact_segment_chain &&) noexcept = default;
artifact_segment_chain & artifact_segment_chain::operator=(
        artifact_segment_chain &&) noexcept = default;

bool artifact_segment_chain::append(
        const uint8_t * data, size_t size) noexcept {
    try {
        if ((!data && size != 0) ||
            size > std::numeric_limits<uint64_t>::max() -
                impl_->total) {
            return false;
        }
        auto bytes =
            std::make_shared<std::vector<uint8_t>>();
        if (size != 0) {
            bytes->assign(data, data + size);
        }
        impl_->segments.push_back({
            std::move(bytes), 0, uint64_t(size),
        });
        impl_->total += size;
        impl_->max_segment =
            std::max(impl_->max_segment, size);
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t artifact_segment_chain::size() const noexcept {
    return impl_->total;
}

size_t artifact_segment_chain::segment_count() const noexcept {
    return impl_->segments.size();
}

size_t artifact_segment_chain::max_segment_size() const noexcept {
    return impl_->max_segment;
}

bool artifact_segment_chain::read(
        uint64_t offset, uint8_t * destination,
        size_t size) const noexcept {
    if ((!destination && size != 0) ||
        offset > impl_->total ||
        size > impl_->total - offset) {
        return false;
    }
    uint64_t cursor = 0;
    size_t remaining = size;
    for (const auto & segment : impl_->segments) {
        const uint64_t end = cursor + segment.length;
        if (offset >= end) {
            cursor = end;
            continue;
        }
        const uint64_t within = offset > cursor
            ? offset - cursor : 0;
        const size_t available =
            size_t(segment.length - within);
        const size_t take = std::min(available, remaining);
        if (take != 0) {
            if (!segment.storage ||
                segment.offset + within >
                    segment.storage->size() ||
                take > segment.storage->size() -
                    size_t(segment.offset + within)) {
                return false;
            }
            std::memcpy(
                destination + (size - remaining),
                segment.storage->data() +
                    size_t(segment.offset + within),
                take);
            remaining -= take;
            offset += take;
            if (remaining == 0) {
                return true;
            }
        }
        cursor = end;
    }
    return remaining == 0;
}

namespace {

bool segment_source_read(
        const void * context, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto * chain =
        static_cast<const artifact_segment_chain *>(context);
    return chain &&
           chain->read(offset, destination, size);
}

} // namespace

vbr_artifact_byte_source artifact_segment_chain::source() const noexcept {
    return { size(), this, segment_source_read };
}

std::array<uint8_t, 32> vbr_capture_stream_digest(
        const artifact_segment_chain & chain) noexcept {
    static constexpr char DOMAIN[] =
        "buun.vbr.capture.segment-stream";
    llama_sha256_writer hash;
    hash.string(DOMAIN, sizeof(DOMAIN) - 1);
    hash.u64(chain.size());
    std::array<uint8_t, 64*1024> scratch;
    for (uint64_t offset = 0; offset < chain.size();) {
        const size_t count = size_t(std::min<uint64_t>(
            scratch.size(), chain.size() - offset));
        if (!chain.read(offset, scratch.data(), count)) {
            return {};
        }
        hash.bytes(scratch.data(), count);
        offset += count;
    }
    return hash.finish();
}

struct vbr_pinned_chunk_ring::impl {
    std::unique_ptr<vbr_bounded_pinned_ring_core> core;
};

vbr_pinned_chunk_ring::vbr_pinned_chunk_ring(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_pinned_chunk_ring::~vbr_pinned_chunk_ring() = default;

std::unique_ptr<vbr_pinned_chunk_ring>
vbr_pinned_chunk_ring::create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting,
        vbr_capture_ring_create_failure * failure) noexcept {
    status = vbr_capture_stream_status::ring_unavailable;
    vbr_capture_ring_create_failure reason =
        vbr_capture_ring_create_failure::none;
    try {
        std::unique_ptr<impl> state(new impl);
        state->core = vbr_bounded_pinned_ring_core::create(
            lanes, total_bytes, chunk_bytes, accounting, reason);
        if (!state->core) {
            status = capture_status_for_ring_failure(reason);
            if (failure) {
                *failure = reason;
            }
            return nullptr;
        }
        status = vbr_capture_stream_status::ok;
        if (failure) {
            *failure = reason;
        }
        return std::unique_ptr<vbr_pinned_chunk_ring>(
            new vbr_pinned_chunk_ring(std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::internal_error;
        if (failure) {
            *failure = vbr_capture_ring_create_failure::internal_error;
        }
        return nullptr;
    }
}

uint64_t vbr_pinned_chunk_ring::capacity_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->capacity_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::chunk_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->chunk_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::lane_count() const noexcept {
    return impl_ && impl_->core ? impl_->core->lane_count() : 0;
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    stats = {};
    if (source.lane >= impl_->core->lane_count() || source.size == 0) {
        return vbr_capture_stream_status::invalid_argument;
    }
    const auto * lane = impl_->core->lane_binding(source.lane);
    const bool tensor_source = source.tensor != nullptr;
    if (tensor_source) {
        // The store may be constructed before VBR lazily creates its dedicated
        // side-stream backend. Events and pinned buffers are device-scoped, so
        // bind the lane to the physical device rather than one backend handle.
        if (!source.backend || !source.device ||
            !lane || source.device != lane->device ||
            ggml_backend_get_device(source.backend) !=
                lane->device ||
            source.tensor_offset >
                std::numeric_limits<uint64_t>::max() -
                    source.size ||
            source.tensor_offset > ggml_nbytes(source.tensor) ||
            source.size >
                ggml_nbytes(source.tensor) -
                    source.tensor_offset) {
            return vbr_capture_stream_status::invalid_argument;
        }
    } else if (!source.read) {
        return vbr_capture_stream_status::invalid_argument;
    }
    const size_t chunk_size = impl_->core->chunk_bytes();

    std::deque<vbr_pinned_chunk_lease> pending;
    llama_sha256_writer hash;
    static constexpr char DOMAIN[] =
        "buun.vbr.capture.segment-stream";
    hash.string(DOMAIN, sizeof(DOMAIN) - 1);
    hash.u64(source.size);

    const auto synchronize_only = [&]() noexcept {
        for (auto & entry : pending) {
            bool event_completion = false;
            impl_->core->wait(entry, event_completion);
            impl_->core->release(entry);
        }
        pending.clear();
    };
    const auto drain_front = [&]() -> vbr_capture_stream_status {
        if (pending.empty()) {
            return vbr_capture_stream_status::ok;
        }
        auto entry = std::move(pending.front());
        pending.pop_front();
        bool event_completion = false;
        if (!impl_->core->wait(entry, event_completion)) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::internal_error;
        }
        if (event_completion) {
            stats.event_completions++;
        }
        // KNOWN LIMITATION: ggml's asynchronous copy/event APIs return no
        // transfer result. The synthetic seam can report transfer_failed,
        // while a real device error can only surface later as a length or
        // digest mismatch; the F3.2 hardware gate must account for that.
        if (stats.chunks == source.fail_completion_at) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::transfer_failed;
        }
        if (!destination.append(entry.data(), entry.valid())) {
            impl_->core->release(entry);
            return vbr_capture_stream_status::internal_error;
        }
        hash.bytes(entry.data(), entry.valid());
        stats.bytes += entry.valid();
        stats.chunks++;
        impl_->core->release(entry);
        return vbr_capture_stream_status::ok;
    };

    // TODO(F4.2a follow-up): lift shared drive(fill,consume) pump into the core.
    try {
        uint64_t offset = 0;
        while (offset < source.size) {
            bool would_block = false;
            auto entry = impl_->core->acquire(source.lane, would_block);
            if (!entry && would_block) {
                stats.backpressure_waits++;
                const auto drained = drain_front();
                if (drained != vbr_capture_stream_status::ok) {
                    synchronize_only();
                    return drained;
                }
                entry = impl_->core->acquire(source.lane, would_block);
            }
            if (!entry || would_block) {
                synchronize_only();
                return vbr_capture_stream_status::internal_error;
            }
            const size_t count = size_t(std::min<uint64_t>(
                chunk_size, source.size - offset));
            if (tensor_source) {
                ggml_backend_tensor_get_async(
                    source.backend, source.tensor,
                    entry.data(),
                    size_t(source.tensor_offset + offset),
                    count);
            } else if (!source.read(
                           source.context, offset,
                           entry.data(), count)) {
                impl_->core->release(entry);
                synchronize_only();
                return vbr_capture_stream_status::short_read;
            }
            bool synchronous_fallback = false;
            if (!impl_->core->submit(
                    entry, count,
                    tensor_source ? source.backend : nullptr,
                    synchronous_fallback)) {
                impl_->core->release(entry);
                synchronize_only();
                return vbr_capture_stream_status::internal_error;
            }
            if (synchronous_fallback) {
                stats.synchronous_fallbacks++;
            }
            pending.push_back(std::move(entry));
            offset += count;
        }
        while (!pending.empty()) {
            const auto drained = drain_front();
            if (drained != vbr_capture_stream_status::ok) {
                synchronize_only();
                return drained;
            }
        }
        if (stats.bytes != source.size) {
            return vbr_capture_stream_status::short_read;
        }
        stats.max_segment_size =
            destination.max_segment_size();
        stats.streaming_digest = hash.finish();
        return vbr_capture_stream_status::ok;
    } catch (...) {
        synchronize_only();
        stats = {};
        return vbr_capture_stream_status::internal_error;
    }
}
