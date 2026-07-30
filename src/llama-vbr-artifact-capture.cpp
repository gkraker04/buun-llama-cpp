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
    struct chunk {
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        std::vector<uint8_t> synthetic;
        uint8_t * data = nullptr;
        size_t valid = 0;
        bool busy = false;

        ~chunk() {
            if (event) {
                ggml_backend_event_free(event);
            }
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
        }
    };

    struct lane {
        vbr_capture_lane binding;
        std::vector<std::unique_ptr<chunk>> chunks;
        size_t next = 0;
    };

    uint64_t capacity = 0;
    size_t chunk_size = 0;
    std::vector<lane> lanes;
    llama_cache_acct_ledger * accounting = nullptr;
    llama_cache_acct_resource_domain accounting_domain;
    bool ring_charged = false;

};

vbr_pinned_chunk_ring::vbr_pinned_chunk_ring(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_pinned_chunk_ring::~vbr_pinned_chunk_ring() {
    if (!impl_) {
        return;
    }
    auto * ledger = impl_->accounting;
    const auto domain = impl_->accounting_domain;
    const bool charged = impl_->ring_charged;
    impl_->ring_charged = false;
    impl_.reset(); // free the pinned buffers before removing their gauge
    if (ledger && charged) {
        ledger->gauge_set(
            llama_cache_acct_category::pinned_preimage_ring,
            domain,
            llama_cache_acct_measure::logical_payload, 0);
        ledger->gauge_set(
            llama_cache_acct_category::pinned_preimage_ring,
            domain,
            llama_cache_acct_measure::resident_allocated, 0);
    }
}

std::unique_ptr<vbr_pinned_chunk_ring>
vbr_pinned_chunk_ring::create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting) noexcept {
    status = vbr_capture_stream_status::ring_unavailable;
    try {
        if (lanes.empty() || chunk_bytes == 0 ||
            total_bytes == 0 ||
            total_bytes >
                VBR_CAPTURE_PINNED_RING_MAX_BYTES ||
            lanes.size() >
                std::numeric_limits<size_t>::max()/2 ||
            total_bytes / chunk_bytes <
                lanes.size()*2 ||
            chunk_bytes >
                std::numeric_limits<uint64_t>::max() /
                    (total_bytes / chunk_bytes)) {
            return nullptr;
        }
        const size_t n_chunks =
            size_t(total_bytes / chunk_bytes);
        std::unique_ptr<impl> state(new impl);
        state->chunk_size = chunk_bytes;
        state->capacity = uint64_t(n_chunks)*chunk_bytes;
        if (accounting) {
            if (!accounting->ledger ||
                !accounting->budget ||
                accounting->domain.residency !=
                    llama_cache_acct_residency::pinned_host) {
                return nullptr;
            }
            auto & ledger = *accounting->ledger;
            const auto before = ledger.snapshot();
            const auto existing = std::find_if(
                before.cells.begin(), before.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category ==
                               llama_cache_acct_category::
                                   pinned_preimage_ring &&
                           row.domain == accounting->domain;
                });
            if (existing != before.cells.end()) {
                const auto resident =
                    existing->cell.measures[size_t(
                        llama_cache_acct_measure::
                            resident_allocated)];
                if (resident.state !=
                        llama_cache_acct_known::known ||
                    resident.value != 0) {
                    return nullptr;
                }
            }
            ledger.gauge_set(
                llama_cache_acct_category::pinned_preimage_ring,
                accounting->domain,
                llama_cache_acct_measure::logical_payload, 0);
            ledger.gauge_set(
                llama_cache_acct_category::pinned_preimage_ring,
                accounting->domain,
                llama_cache_acct_measure::resident_allocated, 0);
            const auto priced = ledger.snapshot();
            if (priced.faults_overflow !=
                    before.faults_overflow ||
                priced.faults_invalid_transition !=
                    before.faults_invalid_transition ||
                priced.faults_allocation !=
                    before.faults_allocation) {
                return nullptr;
            }
            llama_cache_budget_coordinator coordinator;
            coordinator.reset(priced, *accounting->budget);
            llama_cache_budget_plan plan;
            plan.accounting_serial = priced.serial;
            plan.entries.push_back({
                accounting->domain, state->capacity, 0,
            });
            if (coordinator.fits(plan).state !=
                    llama_cache_budget_fit_state::fits) {
                return nullptr;
            }
            state->accounting = &ledger;
            state->accounting_domain =
                accounting->domain;
        }
        state->lanes.resize(lanes.size());
        for (size_t i = 0; i < lanes.size(); ++i) {
            if ((lanes[i].device == nullptr) !=
                    (lanes[i].backend == nullptr) ||
                (lanes[i].backend &&
                 ggml_backend_get_device(
                     lanes[i].backend) !=
                     lanes[i].device)) {
                return nullptr;
            }
            if (lanes[i].device) {
                for (size_t j = 0; j < i; ++j) {
                    if (lanes[j].device ==
                            lanes[i].device) {
                        return nullptr;
                    }
                }
            }
            state->lanes[i].binding = lanes[i];
        }

        for (size_t i = 0; i < n_chunks; ++i) {
            auto & lane =
                state->lanes[i % state->lanes.size()];
            std::unique_ptr<impl::chunk> entry(
                new impl::chunk);
            if (lane.binding.device) {
                auto * host_buft =
                    ggml_backend_dev_host_buffer_type(
                        lane.binding.device);
                if (!host_buft) {
                    return nullptr;
                }
                entry->buffer =
                    ggml_backend_buft_alloc_buffer(
                        host_buft, chunk_bytes);
                if (!entry->buffer ||
                    ggml_backend_buffer_get_size(
                        entry->buffer) < chunk_bytes) {
                    return nullptr;
                }
                entry->data = static_cast<uint8_t *>(
                    ggml_backend_buffer_get_base(
                        entry->buffer));
                if (!entry->data) {
                    return nullptr;
                }
                if (!lane.binding.force_synchronous) {
                    entry->event =
                        ggml_backend_event_new(
                            lane.binding.device);
                }
            } else {
                entry->synthetic.resize(chunk_bytes);
                entry->data = entry->synthetic.data();
            }
            lane.chunks.push_back(std::move(entry));
        }
        for (const auto & lane : state->lanes) {
            if (lane.chunks.size() < 2) {
                return nullptr;
            }
        }
        if (state->accounting) {
            const auto before = state->accounting->snapshot();
            state->accounting->gauge_set(
                llama_cache_acct_category::pinned_preimage_ring,
                state->accounting_domain,
                llama_cache_acct_measure::logical_payload,
                state->capacity);
            state->accounting->gauge_set(
                llama_cache_acct_category::pinned_preimage_ring,
                state->accounting_domain,
                llama_cache_acct_measure::resident_allocated,
                state->capacity);
            const auto after = state->accounting->snapshot();
            if (after.faults_overflow !=
                    before.faults_overflow ||
                after.faults_invalid_transition !=
                    before.faults_invalid_transition ||
                after.faults_allocation !=
                    before.faults_allocation) {
                state->accounting->gauge_set(
                    llama_cache_acct_category::
                        pinned_preimage_ring,
                    state->accounting_domain,
                    llama_cache_acct_measure::
                        logical_payload, 0);
                state->accounting->gauge_set(
                    llama_cache_acct_category::
                        pinned_preimage_ring,
                    state->accounting_domain,
                    llama_cache_acct_measure::
                        resident_allocated, 0);
                return nullptr;
            }
            state->ring_charged = true;
        }
        status = vbr_capture_stream_status::ok;
        return std::unique_ptr<vbr_pinned_chunk_ring>(
            new vbr_pinned_chunk_ring(std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::ring_unavailable;
        return nullptr;
    }
}

uint64_t vbr_pinned_chunk_ring::capacity_bytes() const noexcept {
    return impl_->capacity;
}

size_t vbr_pinned_chunk_ring::chunk_bytes() const noexcept {
    return impl_->chunk_size;
}

size_t vbr_pinned_chunk_ring::lane_count() const noexcept {
    return impl_->lanes.size();
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    stats = {};
    if (source.lane >= impl_->lanes.size() ||
        source.size == 0) {
        return vbr_capture_stream_status::invalid_argument;
    }
    auto & lane = impl_->lanes[source.lane];
    const bool tensor_source = source.tensor != nullptr;
    if (tensor_source) {
        if (!source.backend || !source.device ||
            source.backend != lane.binding.backend ||
            source.device != lane.binding.device ||
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

    std::deque<impl::chunk *> pending;
    llama_sha256_writer hash;
    static constexpr char DOMAIN[] =
        "buun.vbr.capture.segment-stream";
    hash.string(DOMAIN, sizeof(DOMAIN) - 1);
    hash.u64(source.size);

    const auto synchronize_only = [&]() noexcept {
        for (auto * entry : pending) {
            if (entry->event) {
                ggml_backend_event_synchronize(entry->event);
            } else if (tensor_source) {
                ggml_backend_synchronize(source.backend);
            }
            entry->busy = false;
            entry->valid = 0;
        }
        pending.clear();
    };
    const auto drain_front = [&]() -> vbr_capture_stream_status {
        if (pending.empty()) {
            return vbr_capture_stream_status::ok;
        }
        auto * entry = pending.front();
        pending.pop_front();
        if (entry->event) {
            ggml_backend_event_synchronize(entry->event);
            stats.event_completions++;
        }
        // KNOWN LIMITATION: ggml's asynchronous copy/event APIs return no
        // transfer result. The synthetic seam can report transfer_failed,
        // while a real device error can only surface later as a length or
        // digest mismatch; the F3.2 hardware gate must account for that.
        if (stats.chunks == source.fail_completion_at) {
            entry->busy = false;
            entry->valid = 0;
            return vbr_capture_stream_status::transfer_failed;
        }
        if (!destination.append(entry->data, entry->valid)) {
            entry->busy = false;
            entry->valid = 0;
            return vbr_capture_stream_status::internal_error;
        }
        hash.bytes(entry->data, entry->valid);
        stats.bytes += entry->valid;
        stats.chunks++;
        entry->busy = false;
        entry->valid = 0;
        return vbr_capture_stream_status::ok;
    };

    try {
        uint64_t offset = 0;
        while (offset < source.size) {
            auto * entry =
                lane.chunks[lane.next].get();
            lane.next =
                (lane.next + 1) % lane.chunks.size();
            if (entry->busy) {
                stats.backpressure_waits++;
                const auto drained = drain_front();
                if (drained != vbr_capture_stream_status::ok) {
                    synchronize_only();
                    return drained;
                }
                if (entry->busy) {
                    synchronize_only();
                    return vbr_capture_stream_status::
                        internal_error;
                }
            }
            const size_t count = size_t(std::min<uint64_t>(
                impl_->chunk_size, source.size - offset));
            if (tensor_source) {
                ggml_backend_tensor_get_async(
                    source.backend, source.tensor,
                    entry->data,
                    size_t(source.tensor_offset + offset),
                    count);
                if (entry->event) {
                    ggml_backend_event_record(
                        entry->event, source.backend);
                } else {
                    ggml_backend_synchronize(source.backend);
                    stats.synchronous_fallbacks++;
                }
            } else if (!source.read(
                           source.context, offset,
                           entry->data, count)) {
                synchronize_only();
                return vbr_capture_stream_status::short_read;
            }
            entry->valid = count;
            entry->busy = true;
            pending.push_back(entry);
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
