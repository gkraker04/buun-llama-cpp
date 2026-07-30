#include "llama-vbr-artifact-capture.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", \
                    __FILE__, __LINE__, #cond); \
            failures++; \
        } \
    } while (0)

struct synthetic_source {
    std::vector<uint8_t> bytes;
    uint64_t fail_at = UINT64_MAX;
};

static bool read_synthetic(
        const void * opaque, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto & source =
        *static_cast<const synthetic_source *>(opaque);
    if (offset >= source.fail_at ||
        offset > source.bytes.size() ||
        size > source.bytes.size() - offset) {
        return false;
    }
    std::memcpy(destination, source.bytes.data() + offset, size);
    return true;
}

static std::vector<uint8_t> read_chain(
        const artifact_segment_chain & chain) {
    std::vector<uint8_t> out(chain.size());
    CHECK(chain.read(0, out.data(), out.size()));
    return out;
}

static void test_segment_chain_offsets() {
    artifact_segment_chain chain;
    const uint8_t a[] = { 0, 1, 2 };
    const uint8_t b[] = { 3, 4, 5, 6, 7 };
    CHECK(chain.append(a, sizeof(a)));
    CHECK(chain.append(b, sizeof(b)));
    CHECK(chain.size() == 8);
    CHECK(chain.segment_count() == 2);
    CHECK(chain.max_segment_size() == 5);
    uint8_t middle[5] = {};
    CHECK(chain.read(2, middle, sizeof(middle)));
    CHECK(std::vector<uint8_t>(middle, middle + 5) ==
          std::vector<uint8_t>({ 2, 3, 4, 5, 6 }));
    uint8_t one = 0;
    const auto source = chain.source();
    CHECK(source.read(source.context, 7, &one, 1));
    CHECK(one == 7);
    CHECK(!chain.read(8, &one, 1));
}

static void test_cpu_ring_boundaries() {
    vbr_capture_stream_status status;
    auto unavailable = vbr_pinned_chunk_ring::create(
        { {} }, 8, 8, status);
    CHECK(!unavailable);
    CHECK(status == vbr_capture_stream_status::ring_unavailable);

    auto ring = vbr_pinned_chunk_ring::create(
        { {}, {} }, 32, 8, status);
    CHECK(ring);
    CHECK(status == vbr_capture_stream_status::ok);
    CHECK(ring->lane_count() == 2);
    CHECK(ring->capacity_bytes() == 32);

    synthetic_source input;
    input.bytes.resize(41);
    for (size_t i = 0; i < input.bytes.size(); ++i) {
        input.bytes[i] = uint8_t((i*17 + 3) & 0xff);
    }
    vbr_capture_stream_source source;
    source.lane = 1;
    source.size = input.bytes.size();
    source.context = &input;
    source.read = read_synthetic;
    artifact_segment_chain chain;
    vbr_capture_stream_stats stats;
    CHECK(ring->stream(source, chain, stats) ==
          vbr_capture_stream_status::ok);
    CHECK(stats.bytes == input.bytes.size());
    CHECK(stats.chunks == 6);
    CHECK(stats.backpressure_waits > 0);
    CHECK(stats.max_segment_size <= 8);
    CHECK(chain.max_segment_size() <= 8);
    CHECK(chain.size() > ring->capacity_bytes());
    CHECK(read_chain(chain) == input.bytes);

    auto other = vbr_pinned_chunk_ring::create(
        { {} }, 14, 7, status);
    CHECK(other);
    artifact_segment_chain rechunked;
    vbr_capture_stream_stats other_stats;
    source.lane = 0;
    CHECK(other->stream(source, rechunked, other_stats) ==
          vbr_capture_stream_status::ok);
    CHECK(read_chain(rechunked) == input.bytes);
    CHECK(other_stats.streaming_digest == stats.streaming_digest);

    input.fail_at = 16;
    artifact_segment_chain short_chain;
    CHECK(other->stream(source, short_chain, other_stats) ==
          vbr_capture_stream_status::short_read);

    input.fail_at = UINT64_MAX;
    source.fail_completion_at = 1;
    artifact_segment_chain failed_completion;
    CHECK(other->stream(
        source, failed_completion, other_stats) ==
            vbr_capture_stream_status::transfer_failed);
}

static uint64_t ring_resident(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) {
    for (const auto & row : ledger.snapshot().cells) {
        if (row.category ==
                llama_cache_acct_category::pinned_preimage_ring &&
            row.domain == domain) {
            return row.cell.measures[size_t(
                llama_cache_acct_measure::
                    resident_allocated)].value;
        }
    }
    return 0;
}

static void test_ring_accounting_once() {
    llama_cache_acct_ledger ledger;
    const auto domain =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement required {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&required, 1));
    ledger.gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        domain, llama_cache_acct_measure::logical_payload, 0);
    ledger.gauge_set(
        llama_cache_acct_category::pinned_preimage_ring,
        domain, llama_cache_acct_measure::resident_allocated, 0);
    for (const auto category : {
            llama_cache_acct_category::full_snapshot_payload,
            llama_cache_acct_category::checkpoint_state_payload,
            llama_cache_acct_category::typed_accelerator_payload }) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated,
                llama_cache_acct_measure::reserved }) {
            ledger.gauge_set(category, domain, measure, 0);
        }
    }
    CHECK(ledger.certify_complete(
        domain, llama_cache_acct_producer::retention_sidecar));
    llama_cache_budget_config budget;
    budget.host.pinned_cap = 1024;
    budget.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    vbr_capture_ring_accounting accounting {
        &ledger, domain, &budget,
    };
    {
        const auto snapshot = ledger.snapshot();
        llama_cache_budget_coordinator coordinator;
        CHECK(coordinator.reset(snapshot, budget));
        llama_cache_budget_plan plan;
        plan.accounting_serial = snapshot.serial;
        plan.entries.push_back({ domain, 16, 0 });
        const auto fit = coordinator.fits(plan);
        CHECK(fit.state == llama_cache_budget_fit_state::fits);
    }
    vbr_capture_stream_status status;
    auto ring = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting);
    CHECK(ring);
    CHECK(ring_resident(ledger, domain) == 16);
    auto duplicate = vbr_pinned_chunk_ring::create(
        { {} }, 16, 8, status, &accounting);
    CHECK(!duplicate);
    CHECK(ring_resident(ledger, domain) == 16);
    ring.reset();
    CHECK(ring_resident(ledger, domain) == 0);
}

static void test_cuda_ring() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    if (!device) {
        fprintf(stderr, "FAIL: no GPU backend for F3.1 CUDA synthetic gate\n");
        failures++;
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }
    const size_t n = 5*1024*1024 + 3;
    std::vector<uint8_t> expected(n);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = uint8_t((i*29 + 11) & 0xff);
    }
    ggml_init_params params = {
        2*ggml_tensor_overhead(), nullptr, true,
    };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);
    ggml_tensor * tensor =
        context ? ggml_new_tensor_1d(
            context, GGML_TYPE_I8, n) : nullptr;
    ggml_backend_buffer_t buffer =
        tensor ? ggml_backend_alloc_ctx_tensors(
            context, backend) : nullptr;
    CHECK(tensor && buffer);
    if (tensor && buffer) {
        ggml_backend_tensor_set(
            tensor, expected.data(), 0, expected.size());
        vbr_capture_stream_status status;
        auto ring = vbr_pinned_chunk_ring::create(
            { { device, backend } },
            2*1024*1024, 1024*1024, status);
        CHECK(ring);
        artifact_segment_chain chain;
        vbr_capture_stream_stats stats;
        vbr_capture_stream_source source;
        source.lane = 0;
        source.size = expected.size();
        source.backend = backend;
        source.device = device;
        source.tensor = tensor;
        CHECK(ring->stream(source, chain, stats) ==
              vbr_capture_stream_status::ok);
        CHECK(chain.max_segment_size() <= 1024*1024);
        CHECK(read_chain(chain) == expected);
        CHECK(stats.event_completions > 0);

        auto sync_ring = vbr_pinned_chunk_ring::create(
            { { device, backend, true } },
            2*1024*1024, 1024*1024, status);
        CHECK(sync_ring);
        artifact_segment_chain sync_chain;
        vbr_capture_stream_stats sync_stats;
        CHECK(sync_ring->stream(source, sync_chain, sync_stats) ==
              vbr_capture_stream_status::ok);
        CHECK(sync_stats.synchronous_fallbacks > 0);
        CHECK(sync_stats.streaming_digest ==
              stats.streaming_digest);
        CHECK(read_chain(sync_chain) == expected);
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (context) {
        ggml_free(context);
    }
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    test_segment_chain_offsets();
    test_cpu_ring_boundaries();
    test_ring_accounting_once();
    if (argc == 2 && std::string(argv[1]) == "--cuda") {
        test_cuda_ring();
    }
    if (failures != 0) {
        fprintf(stderr, "%d F3.1 capture test(s) failed\n", failures);
        return 1;
    }
    printf("VBR artifact capture: PASS\n");
    return 0;
}
