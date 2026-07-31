// Single-source, INTERNAL-LINKAGE implementation of the D-A2-8 composite assembly core.
// Included (inside an anonymous namespace) by the production bridge TU
// src/llama-vbr-checkpoint.cpp and by the two named test files — and by nothing else.
// This keeps libllama's dynamic-export surface at the closed bridge operation inventory
// (F4/F12, verify round) while the tests exercise a token-identical copy without relinking
// the production bridge TU.

#include "llama-sha256.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-identity-digest.h"

#include <algorithm>
#include <array>
#include <vector>

namespace {

// One controller candidate per memory-tree child, in fixed pre-order. Armed live_guarded
// children capture through the callback; per-child applicability is vbr_operation_armed()
// (lineage UUID is identity, not liveness — consulted only after armed).
struct vbr_checkpoint_child_input {
    bool live_guarded = false;  // else payload_complete
    bool armed        = false;
    // unarmed live_guarded children: whether any live covered cells exist for the checkpoint
    // sequence below the frontier (nonempty coverage the shadow cannot represent -> row 16)
    bool live_covered = false;
    // armed children: exact nonzero lineage identity (digest envelope)
    vbr_lineage_uuid lineage_uuid = {};
    bool (*capture)(void * ctx, uint32_t child_id, vbr_checkpoint_generation_controller & out) = nullptr;
    void * capture_ctx = nullptr;
};

// Ordered (child_id, dependency_mode, lineage_uuid) digest envelope row.
struct vbr_checkpoint_child_policy {
    uint32_t                         child_id  = 0;
    checkpoint_child_dependency_mode mode      = checkpoint_child_dependency_mode::absent;
    vbr_lineage_uuid                 lineage_uuid = {};
};

inline bool vbr_checkpoint_frontier_valid(const vbr_checkpoint_frontier_fields & frontier) {
    return frontier.execution_identity != nullptr && frontier.adapter_config_identity != nullptr &&
           frontier.media_content_identity != nullptr && frontier.token_count >= 0 &&
           frontier.next_position >= 0;
}

// Canonical length-delimited identity/policy/order digest: SHA-256 over the checkpoint
// identity/frontier fields plus the ordered controller policy envelope. Capture and the future
// live-view builder share this helper; the evaluator's exact 32-byte comparison stays
// authoritative.
inline std::array<uint8_t, 32> vbr_checkpoint_identity_digest(
        const vbr_checkpoint_frontier_fields &           frontier,
        const std::vector<vbr_checkpoint_child_policy> & policy) {
    std::vector<vbr_identity_policy_digest_row> canonical;
    canonical.reserve(policy.size());
    for (const auto & row : policy) {
        canonical.push_back({
            row.child_id, row.mode, row.lineage_uuid,
        });
    }
    return vbr_identity_policy_digest(frontier, canonical);
}

// §9.1 all-or-nothing composite assembly over the pre-ordered children. Any armed-child
// capture failure, or an unarmed live_guarded child with nonempty live coverage, fails the
// WHOLE record. May throw on allocation failure; the opaque bridge API catches at its
// boundary.
inline vbr_checkpoint_capture_reason vbr_checkpoint_compose(
        const std::vector<vbr_checkpoint_child_input> & children,
        const vbr_checkpoint_frontier_fields &          frontier,
        vbr_checkpoint_generation_record &              out) {
    out = {};
    if (!vbr_checkpoint_frontier_valid(frontier)) {
        return vbr_checkpoint_capture_reason::invalid_arguments;
    }
    if (!std::any_of(children.begin(), children.end(),
                     [](const vbr_checkpoint_child_input & child) { return child.armed; })) {
        return vbr_checkpoint_capture_reason::not_applicable;
    }

    vbr_checkpoint_generation_record record;
    record.controllers.reserve(children.size());
    for (uint32_t child_id = 0; child_id < children.size(); ++child_id) {
        const auto & child = children[child_id];
        vbr_checkpoint_generation_controller controller;
        if (child.live_guarded && child.armed) {
            if (child.capture == nullptr ||
                !child.capture(child.capture_ctx, child_id, controller)) {
                return vbr_checkpoint_capture_reason::child_capture_failed;
            }
        } else if (child.live_guarded) {
            if (child.live_covered) {
                // §11.1 row 16: live coverage the shadow cannot represent — the matrix is
                // unavailable for this checkpoint, never server-invented coverage.
                return vbr_checkpoint_capture_reason::unarmed_live_covered;
            }
            controller.child_id        = child_id;
            controller.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        } else {
            // §11.1 row 17: payload_complete rows are vacuous (mode recorded, zero streams);
            // an armed child still contributes its exact lineage identity to the envelope.
            if (child.armed && !vbr_lineage_uuid_is_set(child.lineage_uuid)) {
                return vbr_checkpoint_capture_reason::child_capture_failed;
            }
            controller.child_id        = child_id;
            controller.dependency_mode = checkpoint_child_dependency_mode::payload_complete;
            controller.lineage_uuid    = child.armed ? child.lineage_uuid : vbr_lineage_uuid{};
        }
        record.controllers.push_back(std::move(controller));
    }

    std::vector<vbr_checkpoint_child_policy> policy;
    policy.reserve(record.controllers.size());
    for (const auto & controller : record.controllers) {
        policy.push_back({ controller.child_id, controller.dependency_mode, controller.lineage_uuid });
    }
    record.identity_policy_order_digest = vbr_checkpoint_identity_digest(frontier, policy);
    record.status                       = vbr_checkpoint_generation_status::complete;

    out = std::move(record);
    return vbr_checkpoint_capture_reason::ok;
}

}  // namespace
