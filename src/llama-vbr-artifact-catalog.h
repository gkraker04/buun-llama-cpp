#pragma once

#include "llama-vbr-artifact-capture.h"

#include <cstdint>
#include <memory>
#include <vector>

// Internal F2.2 immutable artifact catalog. It is deliberately absent from
// public llama.h and has no server/backend dependency.
enum class llama_vbr_artifact_publish_status : uint8_t {
    published = 0,
    adopted,
    invalid_argument,
    shard_failed,
    duplicate_completion,
    missing_completion,
    format_rejected,
    accounting_unavailable,
    admission_refused,
    stage_failed,
    commit_failed,
    publication_failed,
    internal_error,
    _count,
};

struct llama_vbr_artifact_domain_binding {
    uint32_t topology_index = UINT32_MAX;
    uint16_t device_ordinal = UINT16_MAX;
    llama_cache_acct_resource_domain domain;
};

// F2.2 substitutes deterministic CPU bytes for future F3 backend completions.
// The vector order is intentionally irrelevant; (unit,stash,shard) is the
// participant identity and every sealed participant must occur exactly once.
struct llama_vbr_artifact_fake_shard_completion {
    uint32_t unit_index = UINT32_MAX;
    uint32_t shard_index = UINT32_MAX;
    bool clean_stash = false;
    bool success = true;
    std::vector<uint8_t> bytes;
};

// F2 keeps its public test vocabulary while using the one authority-owned
// transaction fault type shared with F0b.
using llama_vbr_artifact_publish_fault =
    llama_cache_transaction_fault;

struct llama_vbr_artifact_publish_result {
    llama_vbr_artifact_publish_status status =
        llama_vbr_artifact_publish_status::internal_error;
    llama_cache_acct_artifact_id reference_artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id reference_lineage;
};

struct llama_vbr_artifact_catalog_snapshot {
    uint64_t blobs = 0;
    uint64_t stashes = 0;
    uint64_t references = 0;
    uint64_t published = 0;
    uint64_t adopted = 0;
    uint64_t refusals = 0;
    uint64_t staging_overlap_refusals = 0;
};

struct llama_vbr_artifact_reference_tokens {
    llama_cache_acct_artifact_id artifact;
    llama_cache_acct_content_digest unit_content;
    llama_cache_acct_lineage_id lineage;
};

class llama_vbr_artifact_catalog : public vbr_unit_version_sink {
public:
    explicit llama_vbr_artifact_catalog(llama_cache_acct_ledger & ledger);
    ~llama_vbr_artifact_catalog();

    llama_vbr_artifact_catalog(const llama_vbr_artifact_catalog &) = delete;
    llama_vbr_artifact_catalog & operator=(const llama_vbr_artifact_catalog &) = delete;

    // Bind every portable topology/device ordinal through C's canonical
    // topology interner. Callers use the returned domains in their one-shot
    // completeness manifest and point-in-time budget configuration.
    bool bind_topologies(
        const std::vector<vbr_artifact_portable_topology> & topologies,
        std::vector<llama_vbr_artifact_domain_binding> & bindings) noexcept;

    // Create measured-zero cells for every package accounting row plus the F3
    // temporary leaves in those same capacity domains. This never certifies a
    // producer and never resets a cell already configured by this catalog.
    bool configure_accounting(const vbr_artifact_package & package) noexcept;

    // Bounded F2.2 per-unit publication. The format package must contain one
    // unit blob/reference; F3 composes checkpoint-wide capture from these
    // immutable unit publications.
    llama_vbr_artifact_publish_result publish(
        const vbr_artifact_package & package,
        const std::vector<llama_vbr_artifact_fake_shard_completion> & completions,
        const llama_cache_budget_config & budget,
        const llama_vbr_artifact_publish_fault & fault = {}) noexcept;

    // F3.1 streaming path and the abstract F3.2 capture sink entry point.
    std::unique_ptr<vbr_capture_build> begin_capture(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        vbr_capture_stream_status & status) noexcept override;

    // Release every ledger reference owned by this checkpoint reference.
    // Physical payload/stash bytes discharge only when C observes the last op.
    bool retire(llama_cache_acct_artifact_id reference) noexcept;

    bool reference_tokens(
        llama_cache_acct_artifact_id reference,
        llama_vbr_artifact_reference_tokens & out) const noexcept;
    llama_vbr_artifact_catalog_snapshot snapshot() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;

    llama_vbr_artifact_publish_result publish_stream(
        const vbr_artifact_package & package,
        const std::vector<vbr_verified_segment> & segments,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        void * prepared_stream_state) noexcept;

    std::unique_ptr<vbr_capture_build> begin_capture_impl(
        const vbr_artifact_package & package,
        const llama_cache_budget_config & budget,
        const llama_cache_transaction_fault & fault,
        bool charge_transfer_staging,
        vbr_capture_stream_status & status) noexcept;

    friend class llama_vbr_artifact_catalog_stream_build;
};

const char * llama_vbr_artifact_publish_status_name(
    llama_vbr_artifact_publish_status status) noexcept;
