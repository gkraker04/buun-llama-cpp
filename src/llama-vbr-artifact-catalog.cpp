#include "llama-vbr-artifact-catalog.h"

#include "llama-sha256.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <utility>

namespace {

using digest_key = std::array<uint8_t, 32>;

enum class intern_purpose : uint8_t {
    unit = 0,
    stash,
    manifest,
    logical_unit,
    _count,
};

using intern_key = std::pair<intern_purpose, digest_key>;

struct configured_cell {
    llama_cache_acct_category category =
        llama_cache_acct_category::container_overhead;
    llama_cache_acct_resource_domain domain;
};

bool operator<(const configured_cell & a, const configured_cell & b) {
    if (a.category != b.category) {
        return a.category < b.category;
    }
    if (a.domain.residency != b.domain.residency) {
        return a.domain.residency < b.domain.residency;
    }
    if (a.domain.kind != b.domain.kind) {
        return a.domain.kind < b.domain.kind;
    }
    if (a.domain.topology.v != b.domain.topology.v) {
        return a.domain.topology.v < b.domain.topology.v;
    }
    return a.domain.device_ordinal.v < b.domain.device_ordinal.v;
}

} // namespace

struct llama_vbr_artifact_catalog::impl {
    struct allocation {
        llama_cache_acct_category category =
            llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;
        uint64_t logical = 0;
        uint64_t resident = 0;
        llama_cache_acct_alloc_id alloc;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
    };

    struct blob {
        vbr_unit_version_id id;
        vbr_payload_digest payload_digest;
        vbr_stash_payload_id stash_id;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
        vbr_artifact_unit_descriptor descriptor;
        std::vector<std::shared_ptr<const artifact_segment_chain>>
            payload_shards;
        std::vector<allocation> allocations;
    };

    struct stash {
        vbr_stash_payload_id id;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
        vbr_artifact_clean_stash descriptor;
        std::vector<std::shared_ptr<const artifact_segment_chain>>
            shards;
        std::vector<allocation> allocations;
    };

    struct reference {
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest unit_content;
        llama_cache_acct_lineage_id lineage;
        vbr_unit_version_id unit_id;
        vbr_stash_payload_id stash_id;
        vbr_artifact_reference_manifest manifest;
        std::vector<llama_cache_acct_op_id> operations;
    };

    struct txn_leaf {
        allocation binding;
        uint64_t reserve_resident = 0;
        bool existing = false;
    };

    explicit impl(llama_cache_acct_ledger & ledger_) : ledger(ledger_) {}

    bool resolve_domain(
            const vbr_artifact_portable_domain & portable,
            llama_cache_acct_resource_domain & out) const {
        out = {};
        if (portable.residency == llama_cache_acct_residency::device) {
            const auto it = std::find_if(
                domains.begin(), domains.end(),
                [&](const llama_vbr_artifact_domain_binding & binding) {
                    return binding.topology_index == portable.topology_index &&
                           binding.device_ordinal == portable.device_ordinal;
                });
            if (it == domains.end()) {
                return false;
            }
            out = it->domain;
            return true;
        }
        if (portable.kind != llama_cache_acct_domain_kind::not_applicable ||
            portable.topology_index != UINT32_MAX ||
            portable.device_ordinal != UINT16_MAX ||
            portable.residency >= llama_cache_acct_residency::_count ||
            portable.residency == llama_cache_acct_residency::not_applicable) {
            return false;
        }
        out = llama_cache_acct_resource_domain::non_device(
            portable.residency);
        return true;
    }

    bool issue(uint64_t & next, uint64_t & out) {
        if (next == 0 || next == std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        out = next++;
        return true;
    }

    bool intern_content(
            intern_purpose purpose,
            const digest_key & key,
            llama_cache_acct_content_digest & out) {
        const intern_key typed { purpose, key };
        const auto found = content_ids.find(typed);
        if (found != content_ids.end()) {
            out = { found->second };
            return true;
        }
        uint64_t id;
        if (!issue(next_content, id)) {
            return false;
        }
        content_ids.emplace(typed, id);
        out = { id };
        return true;
    }

    bool intern_lineage(
            intern_purpose purpose,
            const digest_key & key,
            llama_cache_acct_lineage_id & out) {
        const intern_key typed { purpose, key };
        const auto found = lineage_ids.find(typed);
        if (found != lineage_ids.end()) {
            out = { found->second };
            return true;
        }
        uint64_t id;
        if (!issue(next_lineage, id)) {
            return false;
        }
        lineage_ids.emplace(typed, id);
        out = { id };
        return true;
    }

    bool issue_artifact(llama_cache_acct_artifact_id & out) {
        uint64_t id;
        if (!issue(next_artifact, id)) {
            return false;
        }
        out = { id };
        return true;
    }

    const allocation * find_allocation(
            const std::vector<allocation> & values,
            llama_cache_acct_category category,
            const llama_cache_acct_resource_domain & domain,
            uint64_t logical,
            uint64_t resident) const {
        const auto it = std::find_if(
            values.begin(), values.end(),
            [&](const allocation & value) {
                return value.category == category &&
                       value.domain == domain &&
                       value.logical == logical &&
                       value.resident == resident;
            });
        return it == values.end() ? nullptr : &*it;
    }

    llama_cache_acct_ledger & ledger;
    mutable std::mutex mutex;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<llama_vbr_artifact_domain_binding> domains;
    std::set<configured_cell> configured;
    std::map<digest_key, blob> blobs;
    std::map<digest_key, stash> stashes;
    std::map<uint64_t, reference> references;
    std::map<intern_key, uint64_t> content_ids;
    std::map<intern_key, uint64_t> lineage_ids;
    uint64_t next_artifact = 1;
    uint64_t next_content = 1;
    uint64_t next_lineage = 1;
    uint64_t n_published = 0;
    uint64_t n_adopted = 0;
    uint64_t n_refusals = 0;
    uint64_t n_staging_overlap_refusals = 0;
};

namespace {

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(
        digest.begin(), digest.end(),
        [](uint8_t byte) { return byte != 0; });
}

struct catalog_stream_state {
    // Borrowed by the move-only build: every build/unit handle must be
    // destroyed before its catalog and ledger.
    llama_vbr_artifact_catalog * catalog = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    vbr_artifact_package package;
    llama_cache_budget_config budget;
    llama_cache_transaction_fault fault;
    bool charge_transfer_staging = true;
    llama_cache_acct_artifact_id blob_artifact;
    llama_cache_acct_artifact_id stash_artifact;
    llama_cache_acct_artifact_id reference_artifact;
    std::vector<llama_cache_transaction_leaf>
        durable_prepared_leaves;
    std::vector<llama_cache_transaction_leaf> staging_leaves;
    std::vector<llama_cache_acct_op_id> durable_ops;
    std::vector<llama_cache_acct_alloc_id> durable_allocs;
    std::vector<llama_cache_acct_op_id> staging_ops;
    std::vector<llama_cache_acct_alloc_id> staging_allocs;
    llama_cache_prepared_claim_group durable_prepared;
    llama_cache_prepared_claim_group staging_prepared;
    bool staging_committed = false;
    std::vector<vbr_verified_segment> segments;
    bool unit_open = false;
    bool unit_sealed = false;
    bool published = false;
    vbr_capture_stream_status failed =
        vbr_capture_stream_status::ok;

    ~catalog_stream_state() {
        if (ledger && staging_committed) {
            for (const auto op : staging_ops) {
                if (op) {
                    const bool released = ledger->release(op);
                    GGML_ASSERT(released);
                }
            }
        }
    }

    bool commit_staging() noexcept {
        if (!charge_transfer_staging ||
            staging_committed) {
            return true;
        }
        const auto result =
            staging_prepared.materialize_and_commit(
                staging_leaves);
        if (result.status !=
                llama_cache_transaction_status::committed) {
            failed = result.status ==
                    llama_cache_transaction_status::stage_failed
                ? vbr_capture_stream_status::stage_failed
                : result.status ==
                      llama_cache_transaction_status::
                          commit_failed
                    ? vbr_capture_stream_status::commit_failed
                    : vbr_capture_stream_status::
                        accounting_refused;
            return false;
        }
        staging_committed = true;
        return true;
    }
};

class catalog_unit_build final : public vbr_unit_build {
public:
    explicit catalog_unit_build(
            std::shared_ptr<catalog_stream_state> state)
        : state_(std::move(state)) {}

    ~catalog_unit_build() override {
        if (state_ && !state_->unit_sealed &&
            state_->failed == vbr_capture_stream_status::ok) {
            state_->failed =
                vbr_capture_stream_status::missing_segment;
        }
    }

    vbr_capture_stream_status accept_verified_segment(
            const vbr_verified_segment & segment) noexcept override {
        try {
            if (!state_ || state_->published ||
                state_->unit_sealed) {
                return vbr_capture_stream_status::late_segment;
            }
            if (state_->failed !=
                    vbr_capture_stream_status::ok) {
                return state_->failed;
            }
            if (!state_->commit_staging()) {
                return state_->failed;
            }
            if (segment.unit_index != 0 ||
                !segment.bytes ||
                segment.bytes->size() == 0 ||
                !digest_nonzero(segment.streaming_digest) ||
                vbr_capture_stream_digest(*segment.bytes) !=
                    segment.streaming_digest) {
                state_->failed =
                    vbr_capture_stream_status::hash_mismatch;
                return state_->failed;
            }
            const auto duplicate = std::find_if(
                state_->segments.begin(),
                state_->segments.end(),
                [&](const vbr_verified_segment & current) {
                    return current.unit_index ==
                               segment.unit_index &&
                           current.shard_index ==
                               segment.shard_index &&
                           current.clean_stash ==
                               segment.clean_stash;
                });
            if (duplicate != state_->segments.end()) {
                state_->failed =
                    vbr_capture_stream_status::duplicate_segment;
                return state_->failed;
            }
            state_->segments.push_back(segment);
            return vbr_capture_stream_status::ok;
        } catch (...) {
            if (state_) {
                state_->failed =
                    vbr_capture_stream_status::internal_error;
            }
            return vbr_capture_stream_status::internal_error;
        }
    }

    vbr_capture_stream_status seal_unit() noexcept override {
        try {
            if (!state_ || state_->published ||
                state_->unit_sealed) {
                return vbr_capture_stream_status::late_segment;
            }
            if (state_->failed !=
                    vbr_capture_stream_status::ok) {
                return state_->failed;
            }
            const auto & descriptor =
                state_->package.unit_blobs[0].descriptor;
            const size_t expected =
                descriptor.shards.size() +
                (descriptor.clean_stash_state ==
                     vbr_artifact_clean_stash_state::present
                     ? descriptor.clean_stash.shards.size()
                     : 0);
            if (state_->segments.size() != expected) {
                state_->failed =
                    vbr_capture_stream_status::missing_segment;
                return state_->failed;
            }
            for (const auto & shard : descriptor.shards) {
                const auto found = std::find_if(
                    state_->segments.begin(),
                    state_->segments.end(),
                    [&](const vbr_verified_segment & value) {
                        return !value.clean_stash &&
                               value.shard_index ==
                                   shard.shard_index &&
                               value.bytes->size() ==
                                   shard.payload_bytes;
                    });
                if (found == state_->segments.end()) {
                    state_->failed =
                        vbr_capture_stream_status::
                            missing_segment;
                    return state_->failed;
                }
            }
            if (descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present) {
                for (const auto & shard :
                     descriptor.clean_stash.shards) {
                    const auto found = std::find_if(
                        state_->segments.begin(),
                        state_->segments.end(),
                        [&](const vbr_verified_segment & value) {
                            return value.clean_stash &&
                                   value.shard_index ==
                                       shard.shard_index &&
                                   value.bytes->size() ==
                                       shard.payload_bytes;
                        });
                    if (found == state_->segments.end()) {
                        state_->failed =
                            vbr_capture_stream_status::
                                missing_segment;
                        return state_->failed;
                    }
                }
            }
            state_->unit_sealed = true;
            return vbr_capture_stream_status::ok;
        } catch (...) {
            if (state_) {
                state_->failed =
                    vbr_capture_stream_status::internal_error;
            }
            return vbr_capture_stream_status::internal_error;
        }
    }

private:
    std::shared_ptr<catalog_stream_state> state_;
};

} // namespace

class llama_vbr_artifact_catalog_stream_build final
        : public vbr_capture_build {
public:
    explicit llama_vbr_artifact_catalog_stream_build(
            std::shared_ptr<catalog_stream_state> state)
        : state_(std::move(state)) {}

    std::unique_ptr<vbr_unit_build> begin_unit(
            uint32_t unit_index,
            vbr_capture_stream_status & status) noexcept override {
        status = vbr_capture_stream_status::invalid_argument;
        try {
            if (!state_ || state_->published ||
                state_->unit_open || unit_index != 0 ||
                state_->package.unit_blobs.size() != 1) {
                return nullptr;
            }
            state_->unit_open = true;
            status = vbr_capture_stream_status::ok;
            return std::unique_ptr<vbr_unit_build>(
                new catalog_unit_build(state_));
        } catch (...) {
            status = vbr_capture_stream_status::internal_error;
            return nullptr;
        }
    }

    vbr_capture_sink_result publish_reference() noexcept override {
        vbr_capture_sink_result out;
        if (!state_ || state_->published ||
            !state_->unit_sealed ||
            state_->failed != vbr_capture_stream_status::ok) {
            out.status = state_ &&
                    state_->failed !=
                        vbr_capture_stream_status::ok
                ? state_->failed
                : vbr_capture_stream_status::missing_segment;
            return out;
        }
        state_->published = true;
        const auto result = state_->catalog->publish_stream(
            state_->package, state_->segments,
            state_->budget, state_->fault,
            state_.get());
        out.reference_artifact = result.reference_artifact;
        out.unit_content = result.unit_content;
        out.reference_lineage = result.reference_lineage;
        out.adopted =
            result.status ==
                llama_vbr_artifact_publish_status::adopted;
        switch (result.status) {
            case llama_vbr_artifact_publish_status::published:
            case llama_vbr_artifact_publish_status::adopted:
                out.status = vbr_capture_stream_status::ok;
                break;
            case llama_vbr_artifact_publish_status::invalid_argument:
                out.status =
                    vbr_capture_stream_status::invalid_argument;
                break;
            case llama_vbr_artifact_publish_status::shard_failed:
                out.status =
                    vbr_capture_stream_status::transfer_failed;
                break;
            case llama_vbr_artifact_publish_status::duplicate_completion:
                out.status =
                    vbr_capture_stream_status::duplicate_segment;
                break;
            case llama_vbr_artifact_publish_status::missing_completion:
                out.status =
                    vbr_capture_stream_status::missing_segment;
                break;
            case llama_vbr_artifact_publish_status::format_rejected:
                out.status =
                    vbr_capture_stream_status::format_rejected;
                break;
            case llama_vbr_artifact_publish_status::accounting_unavailable:
                out.status =
                    vbr_capture_stream_status::
                        accounting_unavailable;
                break;
            case llama_vbr_artifact_publish_status::admission_refused:
                out.status =
                    vbr_capture_stream_status::accounting_refused;
                break;
            case llama_vbr_artifact_publish_status::stage_failed:
                out.status =
                    vbr_capture_stream_status::stage_failed;
                break;
            case llama_vbr_artifact_publish_status::commit_failed:
                out.status =
                    vbr_capture_stream_status::commit_failed;
                break;
            case llama_vbr_artifact_publish_status::publication_failed:
                out.status =
                    vbr_capture_stream_status::publication_failed;
                break;
            case llama_vbr_artifact_publish_status::internal_error:
            case llama_vbr_artifact_publish_status::_count:
                out.status =
                    vbr_capture_stream_status::internal_error;
                break;
        }
        return out;
    }

private:
    std::shared_ptr<catalog_stream_state> state_;
};

llama_vbr_artifact_catalog::llama_vbr_artifact_catalog(
        llama_cache_acct_ledger & ledger)
    : impl_(new impl(ledger)) {}

llama_vbr_artifact_catalog::~llama_vbr_artifact_catalog() {
    while (impl_ && !impl_->references.empty()) {
        const llama_cache_acct_artifact_id reference {
            impl_->references.begin()->first,
        };
        if (!retire(reference)) {
            GGML_ABORT(
                "VBR artifact catalog teardown could not release reference %" PRIu64,
                reference.v);
        }
    }
}

bool llama_vbr_artifact_catalog::bind_topologies(
        const std::vector<vbr_artifact_portable_topology> & topologies,
        std::vector<llama_vbr_artifact_domain_binding> & bindings) noexcept {
    bindings.clear();
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (topologies.empty() || !impl_->references.empty()) {
            return false;
        }
        if (!impl_->topologies.empty()) {
            if (impl_->topologies != topologies) {
                return false;
            }
            bindings = impl_->domains;
            return true;
        }

        std::vector<llama_vbr_artifact_domain_binding> built;
        for (uint32_t topology_index = 0;
             topology_index < topologies.size(); ++topology_index) {
            const auto & topology = topologies[topology_index];
            if (!topology.digest.valid() ||
                topology.digest !=
                    llama_cache_acct_compute_topology_digest(topology)) {
                return false;
            }
            for (uint16_t ordinal = 0;
                 ordinal < topology.device_count; ++ordinal) {
                llama_cache_acct_resource_domain domain;
                if (!impl_->ledger.make_device_domain(
                        topology,
                        llama_cache_acct_device_ordinal { ordinal },
                        domain)) {
                    return false;
                }
                built.push_back({ topology_index, ordinal, domain });
            }
        }
        impl_->topologies = topologies;
        impl_->domains = built;
        bindings = std::move(built);
        return true;
    } catch (...) {
        bindings.clear();
        return false;
    }
}

bool llama_vbr_artifact_catalog::configure_accounting(
        const vbr_artifact_package & package) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->topologies != package.topologies) {
            return false;
        }

        std::set<configured_cell> needed;
        needed.insert({
            llama_cache_acct_category::transfer_staging,
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host),
        });
        for (const auto & row : package.manifest.accounting) {
            llama_cache_acct_resource_domain domain;
            const auto category =
                vbr_artifact_accounting_category(row.role);
            if (category == llama_cache_acct_category::_count ||
                !impl_->resolve_domain(row.domain, domain)) {
                return false;
            }
            needed.insert({ category, domain });
            // Kept for the landed F2/F0 capacity tests and future
            // device-local codec staging. F3's pageable segment image uses
            // the single host-domain transfer_staging cell inserted above.
            needed.insert({
                llama_cache_acct_category::transfer_staging,
                domain,
            });
            needed.insert({
                llama_cache_acct_category::codec_workspace, domain,
            });
            if (domain.residency ==
                    llama_cache_acct_residency::pinned_host) {
                needed.insert({
                    llama_cache_acct_category::pinned_preimage_ring, domain,
                });
            }
        }

        const auto before = impl_->ledger.snapshot();
        std::vector<configured_cell> added;
        for (const auto & cell : needed) {
            if (impl_->configured.count(cell)) {
                continue;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                impl_->ledger.gauge_set(
                    cell.category, cell.domain, measure, 0);
            }
            added.push_back(cell);
        }

        const auto snapshot = impl_->ledger.snapshot();
        if (snapshot.faults_overflow != before.faults_overflow ||
            snapshot.faults_invalid_transition !=
                before.faults_invalid_transition ||
            snapshot.faults_allocation != before.faults_allocation) {
            return false;
        }
        for (const auto & cell : needed) {
            const auto found = std::find_if(
                snapshot.cells.begin(), snapshot.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == cell.category &&
                           row.domain == cell.domain;
                });
            if (found == snapshot.cells.end()) {
                return false;
            }
        }
        impl_->configured.insert(added.begin(), added.end());
        return true;
    } catch (...) {
        return false;
    }
}

std::unique_ptr<vbr_capture_build>
llama_vbr_artifact_catalog::begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status) noexcept {
    return begin_capture_impl(
        package, budget, fault, true, status);
}

std::unique_ptr<vbr_capture_build>
llama_vbr_artifact_catalog::begin_capture_impl(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        bool charge_transfer_staging,
        vbr_capture_stream_status & status) noexcept {
    status = vbr_capture_stream_status::invalid_argument;
    try {
        if (package.unit_blobs.size() != 1 ||
            package.manifest.unit_references.size() != 1 ||
            !package.companions.empty()) {
            return nullptr;
        }
        auto state = std::make_shared<catalog_stream_state>();
        state->catalog = this;
        state->ledger = &impl_->ledger;
        state->package = package;
        state->budget = budget;
        state->fault = fault;
        state->charge_transfer_staging =
            charge_transfer_staging;
        if (charge_transfer_staging) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            const bool has_stash =
                package.unit_blobs[0].descriptor
                    .clean_stash_state ==
                vbr_artifact_clean_stash_state::present;
            if (impl_->topologies != package.topologies ||
                !impl_->issue_artifact(
                    state->blob_artifact) ||
                (has_stash &&
                 !impl_->issue_artifact(
                     state->stash_artifact)) ||
                !impl_->issue_artifact(
                    state->reference_artifact)) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            llama_cache_acct_content_digest staging_content;
            llama_cache_acct_lineage_id staging_lineage;
            llama_sha256_writer staging_hash;
            static constexpr char STAGING_DOMAIN[] =
                "buun.vbr.capture.transfer-staging";
            staging_hash.string(
                STAGING_DOMAIN,
                sizeof(STAGING_DOMAIN) - 1);
            staging_hash.u64(
                package.manifest.accounting.size());
            for (const auto & row :
                 package.manifest.accounting) {
                staging_hash.u32(uint32_t(row.role));
                staging_hash.u32(
                    uint32_t(row.domain.residency));
                staging_hash.u32(
                    uint32_t(row.domain.kind));
                staging_hash.u32(
                    row.domain.topology_index);
                staging_hash.u32(
                    row.domain.device_ordinal);
                staging_hash.u64(row.logical_bytes);
                staging_hash.u64(row.resident_bytes);
            }
            const auto staging_digest =
                staging_hash.finish();
            if (!impl_->intern_content(
                    intern_purpose::manifest,
                    staging_digest,
                    staging_content) ||
                !impl_->intern_lineage(
                    intern_purpose::manifest,
                    staging_digest,
                    staging_lineage)) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }

            const size_t count =
                package.manifest.accounting.size();
            state->durable_ops.resize(count);
            state->durable_allocs.resize(count);
            state->staging_ops.resize(1);
            state->staging_allocs.resize(1);
            state->durable_prepared_leaves.reserve(count);
            state->staging_leaves.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                const auto & row =
                    package.manifest.accounting[i];
                llama_cache_acct_resource_domain domain;
                const auto category =
                    vbr_artifact_accounting_category(
                        row.role);
                if (!impl_->resolve_domain(
                        row.domain, domain) ||
                    !impl_->configured.count({
                        category, domain })) {
                    status =
                        vbr_capture_stream_status::
                            accounting_refused;
                    return nullptr;
                }
                const auto artifact =
                    row.role ==
                        vbr_artifact_accounting_role::
                            clean_stash_payload
                        ? state->stash_artifact
                        : row.role ==
                              vbr_artifact_accounting_role::
                                  reference_metadata
                            ? state->reference_artifact
                            : state->blob_artifact;
                llama_cache_transaction_leaf durable;
                durable.category = category;
                durable.domain = domain;
                durable.attribution = {
                    llama_cache_acct_attr_kind::artifact,
                    -1, artifact,
                };
                durable.expected_logical =
                    row.logical_bytes;
                durable.reserve_resident =
                    row.resident_bytes;
                durable.stage_resident =
                    row.resident_bytes;
                durable.artifact = artifact;
                durable.committed_op =
                    &state->durable_ops[i];
                durable.allocation_out =
                    &state->durable_allocs[i];
                state->durable_prepared_leaves.push_back(
                    durable);

            }
            uint64_t staging_bytes = 0;
            for (const auto & row :
                 package.manifest.accounting) {
                if (row.resident_bytes >
                        UINT64_MAX - staging_bytes) {
                    status =
                        vbr_capture_stream_status::
                            accounting_refused;
                    return nullptr;
                }
                staging_bytes += row.resident_bytes;
            }
            const auto staging_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::
                        pageable_host);
            if (staging_bytes == 0 ||
                !impl_->configured.count({
                    llama_cache_acct_category::
                        transfer_staging,
                    staging_domain })) {
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            llama_cache_transaction_leaf staging;
            staging.category =
                llama_cache_acct_category::transfer_staging;
            staging.domain = staging_domain;
            staging.attribution = {
                llama_cache_acct_attr_kind::artifact,
                -1, state->blob_artifact,
            };
            staging.expected_logical = staging_bytes;
            staging.reserve_resident = staging_bytes;
            staging.stage_resident = staging_bytes;
            staging.artifact = state->blob_artifact;
            staging.content = staging_content;
            staging.lineage = staging_lineage;
            staging.committed_op = &state->staging_ops[0];
            staging.allocation_out =
                &state->staging_allocs[0];
            state->staging_leaves.push_back(staging);
            state->staging_prepared =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    state->staging_leaves);
            if (!state->staging_prepared.ready()) {
                impl_->n_staging_overlap_refusals++;
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
            state->durable_prepared =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    state->durable_prepared_leaves);
            if (!state->durable_prepared.ready()) {
                impl_->n_staging_overlap_refusals++;
                status =
                    vbr_capture_stream_status::
                        accounting_refused;
                return nullptr;
            }
        }
        status = vbr_capture_stream_status::ok;
        return std::unique_ptr<vbr_capture_build>(
            new llama_vbr_artifact_catalog_stream_build(
                std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::internal_error;
        return nullptr;
    }
}

llama_vbr_artifact_publish_result llama_vbr_artifact_catalog::publish(
        const vbr_artifact_package & package,
        const std::vector<llama_vbr_artifact_fake_shard_completion> & completions,
        const llama_cache_budget_config & budget,
        const llama_vbr_artifact_publish_fault & fault) noexcept {
    llama_vbr_artifact_publish_result out;
    try {
        const uint64_t expected =
            package.unit_blobs.size() == 1
                ? package.unit_blobs[0].descriptor.shards.size() +
                      (package.unit_blobs[0].descriptor
                                   .clean_stash_state ==
                               vbr_artifact_clean_stash_state::present
                           ? package.unit_blobs[0].descriptor
                                 .clean_stash.shards.size()
                           : 0)
                : 0;
        if (completions.size() != expected) {
            out.status = completions.size() < expected
                ? llama_vbr_artifact_publish_status::
                    missing_completion
                : llama_vbr_artifact_publish_status::
                    duplicate_completion;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        vbr_capture_stream_status stream_status;
        // F2's fake completions are already caller-owned resident vectors;
        // unlike the F3 D2H stream they do not allocate a pageable transfer
        // image. They still drive the exact same seal/dedup/publication core.
        auto build = begin_capture_impl(
            package, budget, fault, false, stream_status);
        if (!build) {
            out.status =
                llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        auto unit = build->begin_unit(0, stream_status);
        if (!unit) {
            out.status =
                llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        for (const auto & completion : completions) {
            if (completion.unit_index != 0) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        invalid_argument;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            if (!completion.success) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        shard_failed;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            auto chain =
                std::make_shared<artifact_segment_chain>();
            if (!chain->append(
                    completion.bytes.data(),
                    completion.bytes.size())) {
                out.status =
                    llama_vbr_artifact_publish_status::
                        internal_error;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
            vbr_verified_segment segment;
            segment.unit_index = completion.unit_index;
            segment.shard_index = completion.shard_index;
            segment.clean_stash = completion.clean_stash;
            segment.bytes = std::move(chain);
            segment.streaming_digest =
                vbr_capture_stream_digest(*segment.bytes);
            stream_status =
                unit->accept_verified_segment(segment);
            if (stream_status !=
                    vbr_capture_stream_status::ok) {
                out.status =
                    stream_status ==
                        vbr_capture_stream_status::
                            duplicate_segment
                    ? llama_vbr_artifact_publish_status::
                        duplicate_completion
                    : llama_vbr_artifact_publish_status::
                        invalid_argument;
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->n_refusals++;
                return out;
            }
        }
        stream_status = unit->seal_unit();
        if (stream_status != vbr_capture_stream_status::ok) {
            out.status =
                stream_status ==
                    vbr_capture_stream_status::missing_segment
                ? llama_vbr_artifact_publish_status::
                    missing_completion
                : llama_vbr_artifact_publish_status::
                    invalid_argument;
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
            return out;
        }
        const auto streamed = build->publish_reference();
        out.reference_artifact = streamed.reference_artifact;
        out.unit_content = streamed.unit_content;
        out.reference_lineage = streamed.reference_lineage;
        if (streamed.status != vbr_capture_stream_status::ok) {
            switch (streamed.status) {
                case vbr_capture_stream_status::format_rejected:
                case vbr_capture_stream_status::hash_mismatch:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            format_rejected;
                    break;
                case vbr_capture_stream_status::
                        accounting_unavailable:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            accounting_unavailable;
                    break;
                case vbr_capture_stream_status::accounting_refused:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            admission_refused;
                    break;
                case vbr_capture_stream_status::stage_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            stage_failed;
                    break;
                case vbr_capture_stream_status::commit_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            commit_failed;
                    break;
                case vbr_capture_stream_status::publication_failed:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            publication_failed;
                    break;
                case vbr_capture_stream_status::duplicate_segment:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            duplicate_completion;
                    break;
                case vbr_capture_stream_status::missing_segment:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            missing_completion;
                    break;
                case vbr_capture_stream_status::transfer_failed:
                case vbr_capture_stream_status::short_read:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            shard_failed;
                    break;
                case vbr_capture_stream_status::invalid_argument:
                case vbr_capture_stream_status::ring_unavailable:
                case vbr_capture_stream_status::late_segment:
                case vbr_capture_stream_status::internal_error:
                case vbr_capture_stream_status::_count:
                    out.status =
                        llama_vbr_artifact_publish_status::
                            internal_error;
                    break;
                case vbr_capture_stream_status::ok:
                    break;
            }
            return out;
        }
        out.status = streamed.adopted
            ? llama_vbr_artifact_publish_status::adopted
            : llama_vbr_artifact_publish_status::published;
        return out;
    } catch (...) {
        out.status =
            llama_vbr_artifact_publish_status::internal_error;
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->n_refusals++;
        }
        return out;
    }
}

llama_vbr_artifact_publish_result
llama_vbr_artifact_catalog::publish_stream(
        const vbr_artifact_package & package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state) noexcept {
    llama_vbr_artifact_publish_result result;
    try {
        auto * stream_state =
            static_cast<catalog_stream_state *>(
                prepared_stream_state);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->topologies != package.topologies ||
            package.unit_blobs.size() != 1 ||
            package.manifest.unit_references.size() != 1 ||
            !package.companions.empty()) {
            result.status =
                llama_vbr_artifact_publish_status::invalid_argument;
            impl_->n_refusals++;
            return result;
        }

        vbr_artifact_package working = package;
        auto & descriptor = working.unit_blobs[0].descriptor;
        const uint64_t expected =
            descriptor.shards.size() +
            (descriptor.clean_stash_state ==
                 vbr_artifact_clean_stash_state::present
                 ? descriptor.clean_stash.shards.size() : 0);
        if (segments.size() != expected) {
            result.status =
                segments.size() < expected
                    ? llama_vbr_artifact_publish_status::missing_completion
                    : llama_vbr_artifact_publish_status::duplicate_completion;
            impl_->n_refusals++;
            return result;
        }

        std::set<std::pair<bool, uint32_t>> seen;
        for (const auto & segment : segments) {
            if (segment.unit_index != 0 || !segment.bytes) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            // The accepting pass hashes each completed D2H segment. Re-read
            // its immutable backing at the publication boundary so a source
            // mutation/corruption between completion and final artifact
            // encoding cannot silently mint a different content address.
            if (vbr_capture_stream_digest(*segment.bytes) !=
                    segment.streaming_digest) {
                result.status =
                    llama_vbr_artifact_publish_status::
                        format_rejected;
                impl_->n_refusals++;
                return result;
            }
            if (!seen.insert({
                    segment.clean_stash,
                    segment.shard_index }).second) {
                result.status =
                    llama_vbr_artifact_publish_status::duplicate_completion;
                impl_->n_refusals++;
                return result;
            }
            auto & shards = segment.clean_stash
                ? descriptor.clean_stash.shards
                : descriptor.shards;
            const auto shard = std::find_if(
                shards.begin(), shards.end(),
                [&](const vbr_artifact_shard_descriptor & candidate) {
                    return candidate.shard_index ==
                           segment.shard_index;
                });
            if (shard == shards.end()) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            shard->payload = segment.bytes->source();
            shard->payload_bytes = segment.bytes->size();
        }

        const auto prepared = vbr_artifact_prepare(working);
        if (prepared != vbr_artifact_status::ok) {
            result.status =
                llama_vbr_artifact_publish_status::format_rejected;
            impl_->n_refusals++;
            return result;
        }

        const auto unit_key =
            working.unit_blobs[0].unit_version_id.bytes();
        const auto stash_key =
            working.unit_blobs[0].descriptor.clean_stash.payload_id.bytes();
        const bool has_stash =
            working.unit_blobs[0].descriptor.clean_stash_state ==
                vbr_artifact_clean_stash_state::present;
        const auto blob_it = impl_->blobs.find(unit_key);
        const bool blob_exists = blob_it != impl_->blobs.end();
        auto stash_it = has_stash
            ? impl_->stashes.find(stash_key) : impl_->stashes.end();
        const bool stash_exists =
            has_stash && stash_it != impl_->stashes.end();
        if (blob_exists &&
            (blob_it->second.stash_id.valid() != has_stash ||
             (has_stash &&
              blob_it->second.stash_id.bytes() != stash_key))) {
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
        }

        impl::blob pending_blob;
        impl::stash pending_stash;
        impl::reference pending_reference;
        if (blob_exists) {
            pending_blob = blob_it->second;
        } else {
            pending_blob.id =
                working.unit_blobs[0].unit_version_id;
            pending_blob.payload_digest =
                working.unit_blobs[0].payload_digest;
            pending_blob.stash_id =
                working.unit_blobs[0].descriptor.clean_stash.payload_id;
            pending_blob.descriptor =
                working.unit_blobs[0].descriptor;
            for (auto & shard : pending_blob.descriptor.shards) {
                shard.payload = {};
            }
            for (auto & shard :
                 pending_blob.descriptor.clean_stash.shards) {
                shard.payload = {};
            }
            pending_blob.artifact =
                stream_state &&
                    stream_state->charge_transfer_staging
                ? stream_state->blob_artifact
                : llama_cache_acct_artifact_id {};
            if ((pending_blob.artifact.v == 0 &&
                 !impl_->issue_artifact(
                     pending_blob.artifact)) ||
                !impl_->intern_content(
                    intern_purpose::unit, unit_key,
                    pending_blob.content) ||
                !impl_->intern_lineage(
                    intern_purpose::logical_unit,
                    vbr_artifact_logical_unit_digest(descriptor),
                    pending_blob.lineage)) {
                result.status =
                    llama_vbr_artifact_publish_status::internal_error;
                impl_->n_refusals++;
                return result;
            }
        }

        if (has_stash) {
            if (stash_exists) {
                pending_stash = stash_it->second;
            } else {
                pending_stash.id =
                    descriptor.clean_stash.payload_id;
                pending_stash.descriptor =
                    descriptor.clean_stash;
                for (auto & shard : pending_stash.descriptor.shards) {
                    shard.payload = {};
                }
                pending_stash.artifact =
                    stream_state &&
                        stream_state->charge_transfer_staging
                    ? stream_state->stash_artifact
                    : llama_cache_acct_artifact_id {};
                if ((pending_stash.artifact.v == 0 &&
                     !impl_->issue_artifact(
                         pending_stash.artifact)) ||
                    !impl_->intern_content(
                        intern_purpose::stash, stash_key,
                        pending_stash.content) ||
                    !impl_->intern_lineage(
                        intern_purpose::stash, stash_key,
                        pending_stash.lineage)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
            }
        }

        pending_reference.artifact =
            stream_state &&
                stream_state->charge_transfer_staging
            ? stream_state->reference_artifact
            : llama_cache_acct_artifact_id {};
        if ((pending_reference.artifact.v == 0 &&
             !impl_->issue_artifact(
                 pending_reference.artifact)) ||
            !impl_->intern_lineage(
                intern_purpose::manifest,
                working.manifest.manifest_digest.bytes(),
                pending_reference.lineage)) {
            result.status =
                llama_vbr_artifact_publish_status::internal_error;
            impl_->n_refusals++;
            return result;
        }
        pending_reference.unit_content = pending_blob.content;
        pending_reference.unit_id = pending_blob.id;
        pending_reference.stash_id = pending_blob.stash_id;
        pending_reference.manifest = working.manifest;

        std::vector<impl::txn_leaf> leaves;
        leaves.reserve(working.manifest.accounting.size());
        for (const auto & row : working.manifest.accounting) {
            llama_cache_acct_resource_domain domain;
            const auto category =
                vbr_artifact_accounting_category(row.role);
            if (!impl_->resolve_domain(row.domain, domain) ||
                !impl_->configured.count({ category, domain })) {
                result.status =
                    llama_vbr_artifact_publish_status::accounting_unavailable;
                impl_->n_refusals++;
                return result;
            }

            impl::txn_leaf leaf;
            leaf.binding.category = category;
            leaf.binding.domain = domain;
            leaf.binding.logical = row.logical_bytes;
            leaf.binding.resident = row.resident_bytes;
            const std::vector<impl::allocation> * existing = nullptr;
            if (row.role ==
                    vbr_artifact_accounting_role::unit_payload ||
                row.role ==
                    vbr_artifact_accounting_role::descriptor_metadata) {
                leaf.binding.artifact = pending_blob.artifact;
                leaf.binding.content = pending_blob.content;
                leaf.binding.lineage = pending_blob.lineage;
                existing = blob_exists
                    ? &blob_it->second.allocations : nullptr;
            } else if (row.role ==
                    vbr_artifact_accounting_role::clean_stash_payload) {
                if (!has_stash) {
                    result.status =
                        llama_vbr_artifact_publish_status::format_rejected;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding.artifact = pending_stash.artifact;
                leaf.binding.content = pending_stash.content;
                leaf.binding.lineage = pending_stash.lineage;
                existing = stash_exists
                    ? &stash_it->second.allocations : nullptr;
            } else if (row.role ==
                    vbr_artifact_accounting_role::reference_metadata) {
                leaf.binding.artifact =
                    pending_reference.artifact;
                if (!impl_->intern_content(
                        intern_purpose::manifest,
                        working.manifest.manifest_digest.bytes(),
                        leaf.binding.content)) {
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding.lineage =
                    pending_reference.lineage;
            } else {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }

            if (existing) {
                const auto * allocation = impl_->find_allocation(
                    *existing, category, domain,
                    row.logical_bytes, row.resident_bytes);
                if (!allocation) {
                    result.status =
                        llama_vbr_artifact_publish_status::publication_failed;
                    impl_->n_refusals++;
                    return result;
                }
                leaf.binding = *allocation;
                leaf.existing = true;
                leaf.reserve_resident = 0;
            } else {
                leaf.reserve_resident = row.resident_bytes;
            }
            leaves.push_back(leaf);
        }

        std::vector<llama_cache_acct_op_id> committed(
            leaves.size());
        std::vector<llama_cache_acct_alloc_id> allocations(
            leaves.size());
        std::vector<llama_cache_transaction_leaf>
            transaction_leaves;
        transaction_leaves.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_transaction_leaf leaf;
            leaf.category = leaves[i].binding.category;
            leaf.domain = leaves[i].binding.domain;
            leaf.attribution = {
                llama_cache_acct_attr_kind::artifact,
                -1,
                leaves[i].binding.artifact,
            };
            leaf.expected_logical =
                leaves[i].binding.logical;
            leaf.reserve_resident =
                leaves[i].reserve_resident;
            leaf.stage_resident =
                leaves[i].binding.resident;
            leaf.artifact = leaves[i].binding.artifact;
            leaf.content = leaves[i].binding.content;
            leaf.lineage = leaves[i].binding.lineage;
            leaf.existing_allocation =
                leaves[i].existing
                    ? leaves[i].binding.alloc
                    : llama_cache_acct_alloc_id {};
            leaf.committed_op = &committed[i];
            leaf.allocation_out = &allocations[i];
            transaction_leaves.push_back(leaf);
        }

        // A real stream prepared both capacity groups before its first D2H
        // allocation.  The staging group is committed on first acceptance
        // and remains live through this terminal.  A content-addressed
        // adoption changes the allocation shape, so its deliberately
        // conservative fresh durable claims are aborted and re-priced while
        // staging is still live.
        llama_cache_prepared_claim_group local_durable;
        llama_cache_prepared_claim_group * prepared_durable =
            nullptr;
        const bool use_stream_preparation =
            stream_state &&
            stream_state->charge_transfer_staging &&
            !blob_exists && !stash_exists;
        if (use_stream_preparation) {
            prepared_durable =
                &stream_state->durable_prepared;
        } else {
            if (stream_state &&
                stream_state->charge_transfer_staging) {
                stream_state->durable_prepared = {};
            }
            local_durable =
                llama_cache_prepare_reservation_transaction(
                    impl_->ledger, budget,
                    transaction_leaves);
            prepared_durable = &local_durable;
        }
        if (!prepared_durable->ready()) {
            result.status =
                llama_vbr_artifact_publish_status::
                    admission_refused;
            if (stream_state &&
                stream_state->charge_transfer_staging) {
                impl_->n_staging_overlap_refusals++;
            }
            impl_->n_refusals++;
            return result;
        }

        struct materialize_context {
            impl::blob * blob = nullptr;
            impl::stash * stash = nullptr;
            const vbr_artifact_unit_descriptor * descriptor =
                nullptr;
            const std::vector<vbr_verified_segment> *
                    segments = nullptr;
            bool blob_exists = false;
            bool stash_exists = false;
            bool has_stash = false;
        } materialize {
            &pending_blob,
            &pending_stash,
            &descriptor,
            &segments,
            blob_exists,
            stash_exists,
            has_stash,
        };
        const auto materialize_storage = [](void * opaque) -> bool {
            auto * context =
                static_cast<materialize_context *>(opaque);
            if (!context || !context->blob ||
                !context->stash || !context->descriptor ||
                !context->segments) {
                return false;
            }
            if (!context->blob_exists) {
                context->blob->payload_shards.reserve(
                    context->descriptor->shards.size());
                for (const auto & shard :
                     context->descriptor->shards) {
                    const auto completion = std::find_if(
                        context->segments->begin(),
                        context->segments->end(),
                        [&](const auto & candidate) {
                            return !candidate.clean_stash &&
                                   candidate.shard_index ==
                                       shard.shard_index;
                        });
                    if (completion ==
                            context->segments->end() ||
                        !completion->bytes) {
                        return false;
                    }
                    context->blob->payload_shards.push_back(
                        completion->bytes);
                }
            }
            if (context->has_stash &&
                !context->stash_exists) {
                context->stash->shards.reserve(
                    context->descriptor->clean_stash.shards.size());
                for (const auto & shard :
                     context->descriptor->clean_stash.shards) {
                    const auto completion = std::find_if(
                        context->segments->begin(),
                        context->segments->end(),
                        [&](const auto & candidate) {
                            return candidate.clean_stash &&
                                   candidate.shard_index ==
                                       shard.shard_index;
                        });
                    if (completion ==
                            context->segments->end() ||
                        !completion->bytes) {
                        return false;
                    }
                    context->stash->shards.push_back(
                        completion->bytes);
                }
            }
            return true;
        };
        const llama_cache_transaction_after_admit after_admit {
            &materialize, materialize_storage,
        };
        const auto transaction =
            prepared_durable->materialize_and_commit(
                transaction_leaves, fault, after_admit);
        if (transaction.status !=
                llama_cache_transaction_status::committed) {
            switch (transaction.status) {
                case llama_cache_transaction_status::admission_refused:
                    result.status =
                        llama_vbr_artifact_publish_status::admission_refused;
                    break;
                case llama_cache_transaction_status::stage_failed:
                    result.status =
                        llama_vbr_artifact_publish_status::stage_failed;
                    break;
                case llama_cache_transaction_status::commit_failed:
                    result.status =
                        llama_vbr_artifact_publish_status::commit_failed;
                    break;
                case llama_cache_transaction_status::post_commit_fault:
                    result.status =
                        llama_vbr_artifact_publish_status::publication_failed;
                    break;
                case llama_cache_transaction_status::invalid_argument:
                case llama_cache_transaction_status::after_admit_failed:
                case llama_cache_transaction_status::internal_fault:
                case llama_cache_transaction_status::_count:
                    result.status =
                        llama_vbr_artifact_publish_status::internal_error;
                    break;
                case llama_cache_transaction_status::committed:
                    break;
            }
            impl_->n_refusals++;
            return result;
        }

        struct rollback_guard {
            llama_cache_acct_ledger * ledger = nullptr;
            std::vector<llama_cache_acct_op_id> * ops = nullptr;
            bool keep = false;
            ~rollback_guard() {
                if (!keep && ledger && ops) {
                    for (const auto op : *ops) {
                        ledger->release(op);
                    }
                }
            }
        } rollback { &impl_->ledger, &committed, false };

        for (size_t i = 0; i < leaves.size(); ++i) {
            leaves[i].binding.alloc = allocations[i];
            if (!leaves[i].existing) {
                if (leaves[i].binding.category ==
                        llama_cache_acct_category::clean_stash_payload) {
                    pending_stash.allocations.push_back(
                        leaves[i].binding);
                } else if (leaves[i].binding.category !=
                        llama_cache_acct_category::artifact_reference_metadata) {
                    pending_blob.allocations.push_back(
                        leaves[i].binding);
                }
            }
        }

        pending_reference.operations = committed;
        bool inserted_blob = false;
        bool inserted_stash = false;
        try {
            if (!blob_exists) {
                inserted_blob =
                    impl_->blobs.emplace(
                        unit_key, std::move(pending_blob)).second;
                if (!inserted_blob) {
                    throw 0;
                }
            }
            if (has_stash && !stash_exists) {
                inserted_stash =
                    impl_->stashes.emplace(
                        stash_key, std::move(pending_stash)).second;
                if (!inserted_stash) {
                    throw 0;
                }
            }
            const auto inserted_reference =
                impl_->references.emplace(
                    pending_reference.artifact.v,
                    std::move(pending_reference)).second;
            if (!inserted_reference) {
                throw 0;
            }
        } catch (...) {
            if (inserted_blob) {
                impl_->blobs.erase(unit_key);
            }
            if (inserted_stash) {
                impl_->stashes.erase(stash_key);
            }
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
        }

        rollback.keep = true;
        result.status = blob_exists
            ? llama_vbr_artifact_publish_status::adopted
            : llama_vbr_artifact_publish_status::published;
        const auto & stored =
            impl_->references.find(
                pending_reference.artifact.v)->second;
        result.reference_artifact = stored.artifact;
        result.unit_content = stored.unit_content;
        result.reference_lineage = stored.lineage;
        if (blob_exists) {
            impl_->n_adopted++;
        } else {
            impl_->n_published++;
        }
        return result;
    } catch (...) {
        result.status =
            llama_vbr_artifact_publish_status::internal_error;
        if (impl_) {
            impl_->n_refusals++;
        }
        return result;
    }
}

bool llama_vbr_artifact_catalog::retire(
        llama_cache_acct_artifact_id reference) noexcept {
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end()) {
            return false;
        }
        const auto serial = impl_->ledger.snapshot().serial;
        llama_cache_acct_release_set_preview preview;
        if (!impl_->ledger.preview_release_set(
                it->second.operations, serial, preview)) {
            return false;
        }
        for (const auto op : it->second.operations) {
            const bool released = impl_->ledger.release(op);
            GGML_ASSERT(released);
            if (!released) {
                return false;
            }
        }
        const auto unit = it->second.unit_id.bytes();
        const bool has_stash = it->second.stash_id.valid();
        const auto stash = it->second.stash_id.bytes();
        impl_->references.erase(it);

        const bool unit_live = std::any_of(
            impl_->references.begin(), impl_->references.end(),
            [&](const auto & row) {
                return row.second.unit_id.bytes() == unit;
            });
        if (!unit_live) {
            impl_->blobs.erase(unit);
        }
        if (has_stash) {
            const bool stash_live = std::any_of(
                impl_->references.begin(), impl_->references.end(),
                [&](const auto & row) {
                    return row.second.stash_id.valid() &&
                           row.second.stash_id.bytes() == stash;
                });
            if (!stash_live) {
                impl_->stashes.erase(stash);
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool llama_vbr_artifact_catalog::reference_tokens(
        llama_cache_acct_artifact_id reference,
        llama_vbr_artifact_reference_tokens & out) const noexcept {
    out = {};
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->references.find(reference.v);
        if (it == impl_->references.end()) {
            return false;
        }
        out = {
            it->second.artifact,
            it->second.unit_content,
            it->second.lineage,
        };
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

llama_vbr_artifact_catalog_snapshot
llama_vbr_artifact_catalog::snapshot() const noexcept {
    llama_vbr_artifact_catalog_snapshot out;
    try {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        out.blobs = impl_->blobs.size();
        out.stashes = impl_->stashes.size();
        out.references = impl_->references.size();
        out.published = impl_->n_published;
        out.adopted = impl_->n_adopted;
        out.refusals = impl_->n_refusals;
        out.staging_overlap_refusals =
            impl_->n_staging_overlap_refusals;
    } catch (...) {
        out = {};
    }
    return out;
}

const char * llama_vbr_artifact_publish_status_name(
        llama_vbr_artifact_publish_status status) noexcept {
    switch (status) {
        case llama_vbr_artifact_publish_status::published:              return "published";
        case llama_vbr_artifact_publish_status::adopted:                return "adopted";
        case llama_vbr_artifact_publish_status::invalid_argument:       return "invalid_argument";
        case llama_vbr_artifact_publish_status::shard_failed:           return "shard_failed";
        case llama_vbr_artifact_publish_status::duplicate_completion:   return "duplicate_completion";
        case llama_vbr_artifact_publish_status::missing_completion:     return "missing_completion";
        case llama_vbr_artifact_publish_status::format_rejected:        return "format_rejected";
        case llama_vbr_artifact_publish_status::accounting_unavailable: return "accounting_unavailable";
        case llama_vbr_artifact_publish_status::admission_refused:      return "admission_refused";
        case llama_vbr_artifact_publish_status::stage_failed:           return "stage_failed";
        case llama_vbr_artifact_publish_status::commit_failed:          return "commit_failed";
        case llama_vbr_artifact_publish_status::publication_failed:     return "publication_failed";
        case llama_vbr_artifact_publish_status::internal_error:         return "internal_error";
        case llama_vbr_artifact_publish_status::_count:                 break;
    }
    return "invalid";
}
