#pragma once

#include "common-retention-sidecar.h"
#include "../../src/llama-cache-accounting.h"

#include <unordered_map>

struct common_prompt_checkpoint;
struct server_prompt_cache_state;

struct server_retention_instance_key {
    common_retention_artifact_kind kind =
        common_retention_artifact_kind::live_slot;
    int32_t owner_slot = -1;
    uintptr_t instance = 0;

    static server_retention_instance_key for_slot(int32_t slot_id) noexcept {
        return {
            common_retention_artifact_kind::live_slot,
            slot_id,
            uintptr_t(slot_id) + 1,
        };
    }

    static server_retention_instance_key for_checkpoint(
            int32_t owner_slot,
            const common_prompt_checkpoint * checkpoint) noexcept {
        return {
            common_retention_artifact_kind::checkpoint,
            owner_slot,
            reinterpret_cast<uintptr_t>(checkpoint),
        };
    }

    static server_retention_instance_key for_host_entry(
            const server_prompt_cache_state * entry) noexcept {
        return {
            common_retention_artifact_kind::host_entry,
            -1,
            reinterpret_cast<uintptr_t>(entry),
        };
    }
};

inline bool operator==(
        const server_retention_instance_key & a,
        const server_retention_instance_key & b) {
    return a.kind == b.kind &&
           a.owner_slot == b.owner_slot &&
           a.instance == b.instance;
}

struct server_retention_instance_key_hash {
    size_t operator()(const server_retention_instance_key & key) const noexcept;
};

void server_cache_acct_mark_shadow_unavailable(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) noexcept;

llama_cache_acct_op_id server_cache_acct_charge_shadow(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer,
        const llama_cache_acct_attribution & attribution,
        uint64_t logical_bytes,
        uint64_t resident_bytes) noexcept;

// Observer-owned D-S3 catalog. This server layer owns process-local accounting handles;
// the common codec remains a pure, serializable value format.
class server_retention_sidecar_store {
public:
    ~server_retention_sidecar_store();

    server_retention_sidecar_store() = default;
    server_retention_sidecar_store(const server_retention_sidecar_store &) = delete;
    server_retention_sidecar_store & operator=(const server_retention_sidecar_store &) = delete;

    void configure(
        llama_cache_acct_ledger * ledger,
        const llama_cache_acct_resource_domain & domain) noexcept;
    bool publish(
        const server_retention_instance_key & key,
        common_retention_pool pool,
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t turn_token_count,
        uint64_t coverage_tokens,
        bool coverage_valid) noexcept;
    bool clone(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    bool rebind(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept;
    // Interim D-S bridge: lifecycle choke points retire associations directly.
    // D-S5/D-S6 can consolidate this onto retire-by-artifact-id once D-S4 admission
    // owns the catalog mutation rather than merely carrying the strong id.
    void retire(const server_retention_instance_key & key) noexcept;
    void retire_slot(int32_t owner_slot) noexcept;
    llama_cache_acct_artifact_id artifact_id(
        const server_retention_instance_key & key) const noexcept;

    common_retention_sidecar_snapshot snapshot() const noexcept;
    bool import_snapshot(const common_retention_sidecar_snapshot & snapshot) noexcept;
    bool export_bytes(std::vector<uint8_t> & out) const noexcept;

    uint64_t live_bytes() const noexcept { return bytes_live; }
    uint64_t publish_ok() const noexcept { return n_publish_ok; }
    uint64_t unavailable() const noexcept { return n_unavailable; }

private:
    struct catalog_entry {
        common_retention_artifact_record record;
        uint64_t encoded_size = 0;
        llama_cache_acct_op_id accounting_op;
    };
    using association_map = std::unordered_map<
        server_retention_instance_key,
        llama_cache_acct_artifact_id,
        server_retention_instance_key_hash>;

    bool install(
        const server_retention_instance_key & key,
        common_retention_artifact_record && record) noexcept;
    void retire_association(association_map::iterator it) noexcept;
    void mark_unavailable() noexcept;
    static llama_cache_acct_artifact_id qualified_artifact_id(
        common_retention_pool pool, uint64_t stable_id) noexcept;

    llama_cache_acct_ledger * ledger = nullptr;
    llama_cache_acct_resource_domain domain;
    common_retention_allocator allocator;
    association_map associations;
    std::unordered_map<uint64_t, catalog_entry> catalog;
    uint64_t bytes_live = 0;
    uint64_t n_publish_ok = 0;
    uint64_t n_unavailable = 0;
};
