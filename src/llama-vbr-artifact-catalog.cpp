#include "llama-vbr-artifact-catalog.h"

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

bool source_read(
        const void * context,
        uint64_t offset,
        uint8_t * destination,
        size_t size) noexcept {
    const auto * bytes =
        static_cast<const std::vector<uint8_t> *>(context);
    if (!bytes || offset > bytes->size() ||
        size > bytes->size() - size_t(offset)) {
        return false;
    }
    std::copy_n(bytes->data() + size_t(offset), size, destination);
    return true;
}

vbr_artifact_byte_source byte_source(
        const std::vector<uint8_t> & bytes) {
    return { bytes.size(), &bytes, source_read };
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
        std::vector<std::vector<uint8_t>> payload_shards;
        std::vector<allocation> allocations;
    };

    struct stash {
        vbr_stash_payload_id id;
        llama_cache_acct_artifact_id artifact;
        llama_cache_acct_content_digest content;
        llama_cache_acct_lineage_id lineage;
        vbr_artifact_clean_stash descriptor;
        std::vector<std::vector<uint8_t>> shards;
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
        for (const auto & row : package.manifest.accounting) {
            llama_cache_acct_resource_domain domain;
            const auto category =
                vbr_artifact_accounting_category(row.role);
            if (category == llama_cache_acct_category::_count ||
                !impl_->resolve_domain(row.domain, domain)) {
                return false;
            }
            needed.insert({ category, domain });
            needed.insert({
                llama_cache_acct_category::transfer_staging, domain,
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

llama_vbr_artifact_publish_result llama_vbr_artifact_catalog::publish(
        const vbr_artifact_package & package,
        const std::vector<llama_vbr_artifact_fake_shard_completion> & completions,
        const llama_cache_budget_config & budget,
        const llama_vbr_artifact_publish_fault & fault) noexcept {
    llama_vbr_artifact_publish_result result;
    try {
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
        if (completions.size() != expected) {
            result.status =
                completions.size() < expected
                    ? llama_vbr_artifact_publish_status::missing_completion
                    : llama_vbr_artifact_publish_status::duplicate_completion;
            impl_->n_refusals++;
            return result;
        }

        std::set<std::pair<bool, uint32_t>> seen;
        for (const auto & completion : completions) {
            if (completion.unit_index != 0) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            if (!completion.success) {
                result.status =
                    llama_vbr_artifact_publish_status::shard_failed;
                impl_->n_refusals++;
                return result;
            }
            if (!seen.insert({
                    completion.clean_stash,
                    completion.shard_index }).second) {
                result.status =
                    llama_vbr_artifact_publish_status::duplicate_completion;
                impl_->n_refusals++;
                return result;
            }
            auto & shards = completion.clean_stash
                ? descriptor.clean_stash.shards
                : descriptor.shards;
            const auto shard = std::find_if(
                shards.begin(), shards.end(),
                [&](const vbr_artifact_shard_descriptor & candidate) {
                    return candidate.shard_index ==
                           completion.shard_index;
                });
            if (shard == shards.end()) {
                result.status =
                    llama_vbr_artifact_publish_status::invalid_argument;
                impl_->n_refusals++;
                return result;
            }
            shard->payload = byte_source(completion.bytes);
            shard->payload_bytes = completion.bytes.size();
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
            if (!impl_->issue_artifact(pending_blob.artifact) ||
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
                if (!impl_->issue_artifact(pending_stash.artifact) ||
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

        if (!impl_->issue_artifact(pending_reference.artifact) ||
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

        std::vector<llama_cache_admission_result> admissions(
            leaves.size());
        static constexpr uint32_t MAX_ATTEMPTS = 3;
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_admission_status last =
                llama_cache_admission_status::serial_conflict;
            for (uint32_t attempt = 0;
                 attempt < MAX_ATTEMPTS; ++attempt) {
                llama_cache_authority_request request;
                request.category = leaves[i].binding.category;
                request.domain = leaves[i].binding.domain;
                request.attribution = {
                    llama_cache_acct_attr_kind::artifact,
                    -1,
                    leaves[i].binding.artifact,
                };
                request.expected_logical =
                    leaves[i].binding.logical;
                request.expected_resident =
                    leaves[i].reserve_resident;
                admissions[i] = llama_cache_admit_reservation(
                    impl_->ledger, budget, request);
                last = admissions[i].status;
                if (last !=
                        llama_cache_admission_status::serial_conflict) {
                    break;
                }
            }
            if (last != llama_cache_admission_status::admitted) {
                result.status =
                    llama_vbr_artifact_publish_status::admission_refused;
                impl_->n_refusals++;
                return result;
            }
        }

        // The fake completion buffers belong to the caller. Catalog-owned
        // payload storage is allocated only after every capacity claim has
        // been admitted, matching F0b's reserve-before-mutate discipline.
        if (!blob_exists) {
            pending_blob.payload_shards.reserve(
                descriptor.shards.size());
            for (const auto & shard : descriptor.shards) {
                const auto completion = std::find_if(
                    completions.begin(), completions.end(),
                    [&](const auto & candidate) {
                        return !candidate.clean_stash &&
                               candidate.shard_index ==
                                   shard.shard_index;
                    });
                pending_blob.payload_shards.push_back(
                    completion->bytes);
            }
        }
        if (has_stash && !stash_exists) {
            pending_stash.shards.reserve(
                descriptor.clean_stash.shards.size());
            for (const auto & shard :
                 descriptor.clean_stash.shards) {
                const auto completion = std::find_if(
                    completions.begin(), completions.end(),
                    [&](const auto & candidate) {
                        return candidate.clean_stash &&
                               candidate.shard_index ==
                                   shard.shard_index;
                    });
                pending_stash.shards.push_back(
                    completion->bytes);
            }
        }

        for (size_t i = 0; i < leaves.size(); ++i) {
            if (fault.fail_stage_at == i) {
                result.status =
                    llama_vbr_artifact_publish_status::stage_failed;
                impl_->n_refusals++;
                return result;
            }
            if (!leaves[i].existing) {
                leaves[i].binding.alloc =
                    impl_->ledger.new_alloc();
            }
            // Joining an immutable allocation must pass through stage(), so
            // transient_peak observes the brief join even though no payload
            // bytes are copied and durable charge stays singular.
            if (!leaves[i].binding.alloc ||
                !impl_->ledger.stage(
                    admissions[i].claim.op(),
                    leaves[i].binding.alloc,
                    leaves[i].binding.resident,
                    leaves[i].binding.artifact,
                    leaves[i].binding.content,
                    leaves[i].binding.lineage)) {
                result.status =
                    llama_vbr_artifact_publish_status::stage_failed;
                impl_->n_refusals++;
                return result;
            }
        }

        std::vector<llama_cache_acct_op_id> committed;
        committed.reserve(leaves.size());
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
            if (fault.fail_commit_at == i) {
                result.status =
                    llama_vbr_artifact_publish_status::commit_failed;
                impl_->n_refusals++;
                return result;
            }
            llama_cache_acct_op_id op;
            if (!admissions[i].claim.commit(
                    leaves[i].binding.logical, op)) {
                result.status =
                    llama_vbr_artifact_publish_status::commit_failed;
                impl_->n_refusals++;
                return result;
            }
            committed.push_back(op);
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
        if (fault.fail_after_commit) {
            result.status =
                llama_vbr_artifact_publish_status::publication_failed;
            impl_->n_refusals++;
            return result;
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
