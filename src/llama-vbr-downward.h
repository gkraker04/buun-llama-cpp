#pragma once

#include "llama-cache-authority.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-policy.h"
#include "llama-vbr-transaction.h"

#include "ggml-vbr.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t VBR_DOWNWARD_RECIPE_VERSION = 1;

enum class vbr_downward_recipe_status : uint8_t {
    resolved = 0,
    equal_tier,
    upward_forbidden,
    unsupported_type,
    below_floor,
    nonmovable,
    invalid_argument,
    _count,
};

const char * vbr_downward_recipe_status_name(vbr_downward_recipe_status status) noexcept;

struct vbr_downward_edge {
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    vbr_repr_domain source_domain = vbr_repr_domain::full;
    vbr_repr_domain target_domain = vbr_repr_domain::full;
    bool capture_stash_before = false;
};

struct vbr_downward_recipe {
    uint32_t version = VBR_DOWNWARD_RECIPE_VERSION;
    ggml_type source_type = GGML_TYPE_COUNT;
    ggml_type target_type = GGML_TYPE_COUNT;
    std::array<vbr_downward_edge, 5> edges = {};
    size_t n_edges = 0;
};

vbr_downward_recipe_status vbr_downward_resolve_recipe(
        ggml_type source_type,
        ggml_type target_type,
        ggml_type floor_type,
        bool movable,
        vbr_downward_recipe & out) noexcept;

enum class vbr_downward_policy_status : uint8_t {
    coherent = 0,
    incoherent,
    exhausted,
    invalid,
    overflow,
    _count,
};

const char * vbr_downward_policy_status_name(vbr_downward_policy_status status) noexcept;

// The canonical type-vector identity digest lives in llama-vbr-identity-digest.h
// (vbr_type_vector_digest) so F3 capture does not depend on this import module.

struct vbr_downward_policy_child {
    llama_vbr_policy::child policy;
    std::vector<ggml_type> initial_types;
    std::vector<ggml_type> target_types;
};

struct vbr_downward_policy_projection {
    vbr_downward_policy_status status = vbr_downward_policy_status::invalid;
    std::vector<llama_vbr_policy::selection> prefix;
    std::vector<std::vector<ggml_type>> final_types;
    std::vector<std::array<uint8_t, 32>> child_type_digests;
    std::array<uint8_t, 32> tree_digest = {};
};

// One simulator for both the ordinary single-child cursor and the merged tree
// policy stream. It accepts only the first coherent prefix of that stream.
vbr_downward_policy_projection vbr_downward_project_policy_prefix(
        const std::vector<vbr_downward_policy_child> & children) noexcept;

struct vbr_downward_workspace_endpoint {
    const void * owner = nullptr;
    const ggml_vbr_backend_iface * iface = nullptr;
    ggml_backend_t backend = nullptr;
    int device = -1;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution attribution;
    std::vector<llama_vbr_transaction::workspace_request> requests;
};

// The KV cache owns vbr_pool, so its adapter is the only permitted way to
// expose vbr_stash_memory/vbr_stash_reserve without duplicating slab math.
struct vbr_downward_stash_endpoint {
    const void * owner = nullptr;
    // One pool projection covers the complete fixed slab. Every unit whose
    // requested stash shares that pool falls back independently if its one
    // grow-only reserve fails.
    std::vector<uint64_t> unit_ids;
    llama_cache_acct_resource_domain domain;
    llama_cache_acct_attribution attribution;
    void * context = nullptr;
    bool (*memory)(void *, uint64_t &, uint64_t &) = nullptr;
    bool (*reserve)(void *) = nullptr;
};

enum class vbr_downward_reserve_status : uint8_t {
    reserved = 0,
    reserved_stashless,
    projection_unavailable,
    accounting_refused,
    workspace_reserve_failed,
    internal_error,
    _count,
};

const char * vbr_downward_reserve_status_name(vbr_downward_reserve_status status) noexcept;

struct vbr_downward_reserve_result {
    vbr_downward_reserve_status status = vbr_downward_reserve_status::internal_error;
    llama_cache_transaction_status transaction_status = llama_cache_transaction_status::internal_fault;
    llama_cache_admission_status admission_status = llama_cache_admission_status::internal_fault;
    uint64_t workspace_growth = 0;
    uint64_t stash_growth = 0;
    std::vector<uint64_t> stashless_units;
};

// Owns the C references for persistent endpoints. The resource allocation and
// its accounting receipt intentionally have the same side-backend/pool
// lifetime. Existing bytes seen before this owner first projects are adopted
// as its uncharged baseline; only later endpoint growth is transacted.
class vbr_downward_resource_receipts {
public:
    explicit vbr_downward_resource_receipts(llama_cache_acct_ledger & ledger) noexcept;
    ~vbr_downward_resource_receipts();

    vbr_downward_resource_receipts(const vbr_downward_resource_receipts &) = delete;
    vbr_downward_resource_receipts & operator=(const vbr_downward_resource_receipts &) = delete;
    vbr_downward_resource_receipts(vbr_downward_resource_receipts &&) noexcept;
    vbr_downward_resource_receipts & operator=(vbr_downward_resource_receipts &&) noexcept;

    vbr_downward_reserve_result reserve_resources(
        const llama_cache_budget_config & budget,
        const std::vector<vbr_downward_workspace_endpoint> & workspaces,
        const std::vector<vbr_downward_stash_endpoint> & stashes) noexcept;

private:
    struct endpoint_key {
        const void * owner = nullptr;
        llama_cache_acct_category category = llama_cache_acct_category::container_overhead;
        llama_cache_acct_resource_domain domain;

        bool operator==(const endpoint_key & other) const noexcept {
            return owner == other.owner && category == other.category && domain == other.domain;
        }
    };

    struct record {
        endpoint_key key;
        uint64_t endpoint = 0;
    };

    record * find(const endpoint_key & key) noexcept;
    void release_ops() noexcept;

    llama_cache_acct_ledger * ledger_ = nullptr;
    std::vector<record> records_;
    std::vector<llama_cache_acct_op_id> ops_;
};

enum class vbr_downward_transform_status : uint8_t {
    transformed = 0,
    invalid_recipe,
    stash_unavailable,
    transform_failed,
    internal_error,
    _count,
};

struct vbr_downward_transform_iface {
    void * context = nullptr;
    bool (*capture_stash)(void *, ggml_type, const std::vector<uint8_t> &, std::vector<uint8_t> &) = nullptr;
    bool (*transcode)(void *, const vbr_downward_edge &, const std::vector<uint8_t> &,
                      const std::vector<uint8_t> *, std::vector<uint8_t> &) = nullptr;
};

struct vbr_downward_transform_result {
    vbr_downward_transform_status status = vbr_downward_transform_status::internal_error;
    std::vector<uint8_t> bytes;
    std::vector<uint8_t> stash;
    bool stash_regenerated = false;
};

// Injected CPU door used by the edge oracle now and by the live kernel adapter
// in F4.2b-2. Intermediate recipe tiers are never published.
vbr_downward_transform_result vbr_downward_execute_recipe(
        const vbr_downward_recipe & recipe,
        const std::vector<uint8_t> & source,
        const std::vector<uint8_t> * authorized_stash,
        const vbr_downward_transform_iface & iface) noexcept;
