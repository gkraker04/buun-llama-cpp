#include "server-vbr-artifact-store.h"

#include "build-info.h"

#include "../../src/llama-sha256.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <utility>

namespace {

bool capture_capacity_category_applies(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        bool include_live_scope) {
    const auto row = llama_cache_budget_classify(category);
    if (row.participation !=
            llama_cache_budget_capacity_participation::
                participating) {
        return false;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::by_domain) {
        return domain.residency ==
                   llama_cache_acct_residency::device ||
               domain.residency ==
                   llama_cache_acct_residency::pinned_host ||
               domain.residency ==
                   llama_cache_acct_residency::pageable_host;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::host) {
        return
            (domain.residency ==
                 llama_cache_acct_residency::pinned_host ||
             domain.residency ==
                 llama_cache_acct_residency::pageable_host);
    }
    return include_live_scope &&
           row.scope ==
               llama_cache_budget_residency_scope::device &&
           domain.residency ==
               llama_cache_acct_residency::device;
}

std::string opaque_reference(
        uint64_t nonce,
        uint64_t ordinal,
        llama_cache_acct_artifact_id artifact,
        const std::string & tenant) {
    llama_sha256_writer writer;
    static constexpr char DOMAIN[] = "buun.vbr.server-reference/v1";
    writer.string(DOMAIN, sizeof(DOMAIN) - 1);
    writer.u64(nonce);
    writer.u64(ordinal);
    writer.u64(artifact.v);
    writer.string(tenant.data(), tenant.size());
    const auto digest = writer.finish();
    static constexpr char HEX[] = "0123456789abcdef";
    std::string out = "vbrref_";
    out.reserve(7 + 32);
    for (size_t i = 0; i < 16; ++i) {
        out.push_back(HEX[digest[i] >> 4]);
        out.push_back(HEX[digest[i] & 0x0f]);
    }
    return out;
}

server_vbr_artifact_capture_status map_status(
        vbr_explicit_capture_status status) {
    switch (status) {
        case vbr_explicit_capture_status::ok:
            return server_vbr_artifact_capture_status::ok;
        case vbr_explicit_capture_status::not_armed:
        case vbr_explicit_capture_status::unsupported_layout:
            return server_vbr_artifact_capture_status::unsupported;
        case vbr_explicit_capture_status::slot_not_idle:
            return server_vbr_artifact_capture_status::slot_processing;
        case vbr_explicit_capture_status::identity_unavailable:
            return server_vbr_artifact_capture_status::identity_unavailable;
        case vbr_explicit_capture_status::required_companion_unavailable:
            return server_vbr_artifact_capture_status::
                required_companion_unavailable;
        case vbr_explicit_capture_status::admission_refused:
            return server_vbr_artifact_capture_status::admission_refused;
        case vbr_explicit_capture_status::source_changed:
            return server_vbr_artifact_capture_status::source_changed;
        case vbr_explicit_capture_status::generation_unavailable:
        case vbr_explicit_capture_status::registry_busy:
        case vbr_explicit_capture_status::recovery_pending:
        case vbr_explicit_capture_status::geometry_mismatch:
        case vbr_explicit_capture_status::stash_inconsistent:
        case vbr_explicit_capture_status::size_overflow:
        case vbr_explicit_capture_status::ring_unavailable:
        case vbr_explicit_capture_status::transfer_failed:
        case vbr_explicit_capture_status::short_read:
        case vbr_explicit_capture_status::event_failed:
        case vbr_explicit_capture_status::hash_mismatch:
        case vbr_explicit_capture_status::dedup_validation_failed:
        case vbr_explicit_capture_status::accounting_failed:
        case vbr_explicit_capture_status::publication_failed:
            return server_vbr_artifact_capture_status::unavailable;
        case vbr_explicit_capture_status::internal_error:
        case vbr_explicit_capture_status::_count:
            return server_vbr_artifact_capture_status::internal_error;
    }
    return server_vbr_artifact_capture_status::internal_error;
}

} // namespace

struct server_vbr_artifact_store::impl {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_vbr_artifact_catalog catalog;
    std::unique_ptr<vbr_pinned_chunk_ring> ring;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    void * budget_context = nullptr;
    server_vbr_artifact_store_config::sample_budget_fn sample_budget = nullptr;
    server_vbr_artifact_store_counters counters;
    uint64_t nonce = 0;
    uint64_t next_reference = 1;
    uint32_t n_attention_children = 0;

    explicit impl(llama_cache_acct_ledger & ledger)
        : ledger(&ledger), catalog(ledger) {
    }
};

server_vbr_artifact_store::server_vbr_artifact_store(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {
}

server_vbr_artifact_store::~server_vbr_artifact_store() = default;

bool server_vbr_artifact_store_observe_empty_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    try {
        if (!llama_cache_acct_resource_domain_valid(domain) ||
            (domain.residency !=
                 llama_cache_acct_residency::device &&
             domain.residency !=
                 llama_cache_acct_residency::pinned_host &&
             domain.residency !=
                 llama_cache_acct_residency::pageable_host)) {
            return false;
        }

        const auto before = ledger.snapshot();
        if (before.completeness_manifest !=
                llama_cache_acct_known::known) {
            return false;
        }
        if (std::none_of(
                before.completeness.begin(),
                before.completeness.end(),
                [&](const llama_cache_acct_completeness_row & row) {
                    return row.domain == domain &&
                           row.state !=
                               llama_cache_acct_known::
                                   unavailable;
                })) {
            return false;
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            const auto cell = std::find_if(
                before.cells.begin(), before.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == category &&
                           row.domain == domain;
                });
            if (cell == before.cells.end()) {
                return false;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value =
                    cell->cell.measures[size_t(measure)];
                if (value.state ==
                        llama_cache_acct_known::unavailable ||
                    (value.state == llama_cache_acct_known::known &&
                     value.value != 0)) {
                    return false;
                }
            }
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                ledger.gauge_set(category, domain, measure, 0);
            }
        }
        const auto after = ledger.snapshot();
        if (after.faults_invalid_transition !=
                before.faults_invalid_transition ||
            after.faults_overflow != before.faults_overflow ||
            after.faults_allocation !=
                before.faults_allocation) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_verify_accounting(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_resource_domain> &
            domains) noexcept {
    try {
        const auto snapshot = ledger.snapshot();
        if (snapshot.completeness_manifest !=
                llama_cache_acct_known::known ||
            domains.empty()) {
            return false;
        }
        for (size_t i = 0; i < domains.size(); ++i) {
            const auto & domain = domains[i];
            if (!llama_cache_acct_resource_domain_valid(domain) ||
                std::find(
                    domains.begin(), domains.begin() + i,
                    domain) != domains.begin() + i) {
                return false;
            }
            bool has_requirement = false;
            for (const auto & row : snapshot.completeness) {
                if (row.domain == domain) {
                    has_requirement = true;
                    if (row.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
            if (!has_requirement) {
                return false;
            }
        }
        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            const auto classification =
                llama_cache_budget_classify(category);
            for (const auto & domain : domains) {
                if (!capture_capacity_category_applies(
                        category, domain, true)) {
                    continue;
                }
                const auto cell = std::find_if(
                    snapshot.cells.begin(), snapshot.cells.end(),
                    [&](const llama_cache_acct_cell_row & row) {
                        return row.category == category &&
                               row.domain == domain;
                    });
                if (cell == snapshot.cells.end() ||
                    cell->certification !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                const auto resident =
                    cell->cell.measures[size_t(
                        llama_cache_acct_measure::
                            resident_allocated)];
                if (resident.state !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                if (classification.mode ==
                        llama_cache_budget_accounting_mode::
                            transactional) {
                    const auto reserved =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::reserved)];
                    if (reserved.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
                if (classification.scope !=
                        llama_cache_budget_residency_scope::
                            device) {
                    const auto logical =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::
                                logical_payload)];
                    if (logical.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_configure_pinned_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    const auto canonical =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    if (domain != canonical ||
        !server_vbr_artifact_store_observe_empty_accounting(
            ledger, domain) ||
        !ledger.certify_complete(
            domain,
            llama_cache_acct_producer::retention_sidecar)) {
        return false;
    }
    return server_vbr_artifact_store_verify_accounting(
        ledger, { domain });
}

std::unique_ptr<server_vbr_artifact_store>
server_vbr_artifact_store::create(
        const server_vbr_artifact_store_config & config,
        server_vbr_artifact_capture_status & status,
        server_vbr_artifact_store_create_diagnostics * diagnostics) noexcept {
    status = server_vbr_artifact_capture_status::unavailable;
    server_vbr_artifact_store_create_diagnostics observed;
    observed.requested_ring_bytes = config.ring_bytes;
    observed.chunk_bytes = config.chunk_bytes;
    observed.lane_count = config.lanes.size();
    observed.attention_children = config.attention_children;
    const auto fail =
        [&](server_vbr_artifact_store_create_failure failure) {
            observed.failure = failure;
            if (diagnostics) {
                *diagnostics = observed;
            }
        };
    try {
        if (config.ledger == nullptr) {
            fail(server_vbr_artifact_store_create_failure::ledger_missing);
            return nullptr;
        }
        if (config.sample_budget == nullptr) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sampler_missing);
            return nullptr;
        }
        if (config.topologies.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                topology_missing);
            return nullptr;
        }
        if (config.pool_bindings.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                pool_binding_missing);
            return nullptr;
        }
        if (config.lanes.empty()) {
            fail(server_vbr_artifact_store_create_failure::lane_missing);
            return nullptr;
        }
        if (config.attention_children == 0) {
            fail(server_vbr_artifact_store_create_failure::
                attention_child_missing);
            return nullptr;
        }
        if (config.ring_bytes == 0 ||
            config.ring_bytes >
                VBR_CAPTURE_PINNED_RING_MAX_BYTES) {
            fail(server_vbr_artifact_store_create_failure::
                ring_size_invalid);
            return nullptr;
        }
        if (config.chunk_bytes == 0 ||
            config.lanes.size() >
                std::numeric_limits<uint64_t>::max()/2 ||
            uint64_t(config.lanes.size()*2) >
                std::numeric_limits<uint64_t>::max() /
                    uint64_t(config.chunk_bytes)) {
            fail(server_vbr_artifact_store_create_failure::
                chunk_size_invalid);
            return nullptr;
        }
        auto state = std::make_unique<impl>(*config.ledger);
        state->topologies = config.topologies;
        state->pool_bindings = config.pool_bindings;
        state->budget_context = config.budget_context;
        state->sample_budget = config.sample_budget;
        state->n_attention_children = config.attention_children;
        std::random_device random;
        state->nonce = (uint64_t(random()) << 32) ^ random();
        if (state->nonce == 0) {
            state->nonce = 1;
        }

        llama_cache_budget_config budget;
        if (!state->sample_budget(state->budget_context, budget)) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sample_failed);
            return nullptr;
        }
        vbr_capture_ring_accounting accounting {
            config.ledger, config.pinned_domain, &budget,
        };
        const uint64_t minimum_ring_bytes =
            uint64_t(config.lanes.size()*2) *
            uint64_t(config.chunk_bytes);
        uint64_t attempt = config.ring_bytes;
        for (;;) {
            observed.attempted_ring_bytes = attempt;
            state->ring = vbr_pinned_chunk_ring::create(
                config.lanes, attempt, config.chunk_bytes,
                observed.ring_status, &accounting,
                &observed.ring_failure);
            if (state->ring) {
                break;
            }
            // Pinned allocation pressure is recoverable without weakening
            // the ring protocol: two chunks per physical lane are sufficient
            // for bounded producer/consumer overlap. Other failures are
            // evidence/configuration failures and remain fail closed.
            if (observed.ring_failure !=
                    vbr_capture_ring_create_failure::
                        host_buffer_allocation_failed ||
                attempt <= minimum_ring_bytes) {
                break;
            }
            uint64_t next =
                (attempt/2/uint64_t(config.chunk_bytes)) *
                uint64_t(config.chunk_bytes);
            next = std::max(next, minimum_ring_bytes);
            if (next >= attempt) {
                break;
            }
            attempt = next;
        }
        if (!state->ring) {
            status = observed.ring_status ==
                    vbr_capture_stream_status::accounting_refused
                ? server_vbr_artifact_capture_status::admission_refused
                : server_vbr_artifact_capture_status::unavailable;
            fail(server_vbr_artifact_store_create_failure::
                ring_create_failed);
            return nullptr;
        }
        observed.constructed_ring_bytes =
            state->ring->capacity_bytes();
        state->counters.pinned_bytes =
            observed.constructed_ring_bytes;
        status = server_vbr_artifact_capture_status::ok;
        if (diagnostics) {
            *diagnostics = observed;
        }
        return std::unique_ptr<server_vbr_artifact_store>(
            new server_vbr_artifact_store(std::move(state)));
    } catch (...) {
        status = server_vbr_artifact_capture_status::internal_error;
        fail(server_vbr_artifact_store_create_failure::internal_error);
        return nullptr;
    }
}

server_vbr_artifact_capture_output server_vbr_artifact_store::capture(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept {
    server_vbr_artifact_capture_output output;
    impl_->counters.requested++;
    try {
        if (tenant_key.empty()) {
            output.status =
                server_vbr_artifact_capture_status::unauthorized;
            impl_->counters.refused++;
            return output;
        }
        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            output.status =
                server_vbr_artifact_capture_status::unavailable;
            impl_->counters.unavailable++;
            return output;
        }
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;

        vbr_explicit_capture_accounting accounting;
        accounting.budget = &budget;
        accounting.context = &impl_->catalog;
        accounting.prepare = [](
                void * context,
                const vbr_artifact_package & package) noexcept {
            return static_cast<llama_vbr_artifact_catalog *>(context)
                ->prepare_capture_package(package);
        };
        const auto result = vbr_capture_explicit_manifest(
            memory, request, impl_->catalog, accounting);
        output.library_status = result.status;
        output.phase = result.phase;
        output.inner_stream_status =
            result.inner_stream_status;
        output.generation_failure =
            result.generation_failure;
        output.size_failure =
            result.size_failure;
        output.begin_diagnostics =
            result.begin_diagnostics;
        if (result.status != vbr_explicit_capture_status::_count) {
            impl_->counters.capture_outcomes[size_t(result.status)]++;
        }
        output.status = map_status(result.status);
        output.controllers = result.controllers;
        output.units = result.units;
        output.companions = result.companions;
        output.payload_bytes = result.payload_bytes;
        output.stash_bytes = result.stash_bytes;
        output.companion_bytes = result.companion_bytes;
        output.chunks = result.chunks;
        output.backpressure_waits = result.backpressure_waits;
        output.event_completions = result.event_completions;
        output.synchronous_fallbacks = result.synchronous_fallbacks;
        if (result.status != vbr_explicit_capture_status::ok) {
            if (output.status ==
                    server_vbr_artifact_capture_status::admission_refused) {
                impl_->counters.refused++;
            } else if (output.status ==
                    server_vbr_artifact_capture_status::internal_error) {
                impl_->counters.internal_error++;
            } else {
                impl_->counters.unavailable++;
            }
            return output;
        }
        if (result.sink.reference_artifact.v == 0) {
            output.status =
                server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            return output;
        }
        const auto after = impl_->catalog.snapshot();
        output.dedup = result.sink.adopted;
        output.reference = opaque_reference(
            impl_->nonce, impl_->next_reference++,
            result.sink.reference_artifact, tenant_key);
        output.consistency = vbr_artifact_consistency_kind::capture_exact;
        impl_->counters.exact_published++;
        impl_->counters.payload_bytes += result.payload_bytes;
        impl_->counters.stash_bytes += result.stash_bytes;
        impl_->counters.companion_bytes += result.companion_bytes;
        impl_->counters.chunks += result.chunks;
        impl_->counters.event_completions +=
            result.event_completions;
        impl_->counters.synchronous_fallbacks +=
            result.synchronous_fallbacks;
        impl_->counters.backpressure_waits += result.backpressure_waits;
        if (output.dedup) {
            impl_->counters.dedup_hits++;
        } else {
            impl_->counters.dedup_misses++;
        }
        impl_->counters.staging_overlap_refusals =
            after.staging_overlap_refusals;
        return output;
    } catch (...) {
        output.status =
            server_vbr_artifact_capture_status::internal_error;
        impl_->counters.internal_error++;
        return output;
    }
}

const server_vbr_artifact_store_counters &
server_vbr_artifact_store::counters() const noexcept {
    return impl_->counters;
}

uint32_t server_vbr_artifact_store::attention_children() const noexcept {
    return impl_->n_attention_children;
}

const char * server_vbr_artifact_store_create_failure_name(
        server_vbr_artifact_store_create_failure failure) noexcept {
    switch (failure) {
        case server_vbr_artifact_store_create_failure::none:
            return "none";
        case server_vbr_artifact_store_create_failure::ledger_missing:
            return "ledger_missing";
        case server_vbr_artifact_store_create_failure::
                budget_sampler_missing:
            return "budget_sampler_missing";
        case server_vbr_artifact_store_create_failure::topology_missing:
            return "topology_missing";
        case server_vbr_artifact_store_create_failure::
                pool_binding_missing:
            return "pool_binding_missing";
        case server_vbr_artifact_store_create_failure::lane_missing:
            return "lane_missing";
        case server_vbr_artifact_store_create_failure::
                attention_child_missing:
            return "attention_child_missing";
        case server_vbr_artifact_store_create_failure::
                ring_size_invalid:
            return "ring_size_invalid";
        case server_vbr_artifact_store_create_failure::
                chunk_size_invalid:
            return "chunk_size_invalid";
        case server_vbr_artifact_store_create_failure::
                budget_sample_failed:
            return "budget_sample_failed";
        case server_vbr_artifact_store_create_failure::
                ring_create_failed:
            return "ring_create_failed";
        case server_vbr_artifact_store_create_failure::internal_error:
            return "internal_error";
        case server_vbr_artifact_store_create_failure::_count:
            break;
    }
    return "invalid";
}

const char * server_vbr_artifact_capture_status_name(
        server_vbr_artifact_capture_status status) noexcept {
    switch (status) {
        case server_vbr_artifact_capture_status::ok: return "ok";
        case server_vbr_artifact_capture_status::unsupported: return "unsupported";
        case server_vbr_artifact_capture_status::unavailable: return "unavailable";
        case server_vbr_artifact_capture_status::invalid_slot: return "invalid_slot";
        case server_vbr_artifact_capture_status::slot_processing: return "slot_processing";
        case server_vbr_artifact_capture_status::stale_frontier: return "stale_frontier";
        case server_vbr_artifact_capture_status::identity_unavailable: return "identity_unavailable";
        case server_vbr_artifact_capture_status::unauthorized: return "unauthorized";
        case server_vbr_artifact_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case server_vbr_artifact_capture_status::admission_refused: return "admission_refused";
        case server_vbr_artifact_capture_status::source_changed: return "source_changed";
        case server_vbr_artifact_capture_status::internal_error: return "internal_error";
        case server_vbr_artifact_capture_status::_count: return "_count";
    }
    return "_count";
}
