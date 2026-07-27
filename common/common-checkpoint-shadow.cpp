#include "common-checkpoint-shadow.h"

// The one common TU with private src access (see common/CMakeLists.txt): every consumer above
// stays on the opaque helpers in common-checkpoint-shadow.h.
#include "llama-vbr-checkpoint.h"

#include <atomic>
#include <new>
#include <memory>
#include <utility>

// Opaque holder: owns exactly one bridge handle for the checkpoint's generation record.
struct common_checkpoint_shadow {
    llama_vbr_checkpoint_shadow * handle = nullptr;

    explicit common_checkpoint_shadow(llama_vbr_checkpoint_shadow * h) : handle(h) {}
    ~common_checkpoint_shadow() { llama_vbr_checkpoint_shadow_free(handle); }

    common_checkpoint_shadow(const common_checkpoint_shadow &)             = delete;
    common_checkpoint_shadow & operator=(const common_checkpoint_shadow &) = delete;
    common_checkpoint_shadow(common_checkpoint_shadow &&)                  = delete;
    common_checkpoint_shadow & operator=(common_checkpoint_shadow &&)      = delete;
};

namespace {

// The common reason enum mirrors the bridge enum value-for-value (the internal header cannot
// leak through common headers); pin the correspondence once so the mapping is a checked cast.
#define SHADOW_REASON_MIRROR(name) \
    static_assert(int(common_checkpoint_shadow_reason::name) == int(vbr_checkpoint_capture_reason::name), \
                  "capture reason enums diverged: " #name)
SHADOW_REASON_MIRROR(ok);
SHADOW_REASON_MIRROR(not_applicable);
SHADOW_REASON_MIRROR(invalid_arguments);
SHADOW_REASON_MIRROR(unarmed_live_covered);
SHADOW_REASON_MIRROR(child_capture_failed);
SHADOW_REASON_MIRROR(oracle_mismatch);
SHADOW_REASON_MIRROR(internal_error);
#undef SHADOW_REASON_MIRROR

common_checkpoint_shadow_reason common_shadow_reason(vbr_checkpoint_capture_reason reason) {
    return static_cast<common_checkpoint_shadow_reason>(reason);
}

std::atomic<uint64_t> g_shadow_dropped_on_copy{ 0 };

const llama_vbr_checkpoint_shadow * shadow_handle(const common_prompt_checkpoint & ckpt) {
    return ckpt.shadow ? ckpt.shadow->handle : nullptr;
}

}  // namespace

// Test seam: declared extern by the named lifecycle test (no header home — production callers
// use capture below, which mints handles only through the bridge).
void common_checkpoint_shadow_attach(common_prompt_checkpoint & ckpt, llama_vbr_checkpoint_shadow * handle);

void common_checkpoint_shadow_attach(common_prompt_checkpoint & ckpt, llama_vbr_checkpoint_shadow * handle) {
    ckpt.shadow.reset(handle != nullptr ? new (std::nothrow) common_checkpoint_shadow(handle) : nullptr);
    if (handle != nullptr && !ckpt.shadow) {
        // holder allocation failed: the handle must not leak, and the checkpoint stays
        // generation-unknown (capture reports internal_error below)
        llama_vbr_checkpoint_shadow_free(handle);
    }
}

common_prompt_checkpoint::common_prompt_checkpoint()  = default;
common_prompt_checkpoint::~common_prompt_checkpoint() = default;

// Copies drop the shadow by construction (finding-3 semantics): both implicit deep-copy sites
// (host-cache staging, prompt clones) then carry record-less checkpoints with no extra code.
// NOTE: keep the member list in sync with common_prompt_checkpoint.
common_prompt_checkpoint::common_prompt_checkpoint(const common_prompt_checkpoint & other) :
    n_tokens(other.n_tokens),
    id_task(other.id_task),
    pos_min(other.pos_min),
    pos_max(other.pos_max),
    representation_epoch(other.representation_epoch),
    representation_epoch_swa(other.representation_epoch_swa),
    computation_frontier(other.computation_frontier),
    data_tgt(other.data_tgt),
    data_dft(other.data_dft),
    accel(other.accel),
    shadow(nullptr) {
    if (other.shadow) {
        g_shadow_dropped_on_copy.fetch_add(1, std::memory_order_relaxed);
    }
}

common_prompt_checkpoint & common_prompt_checkpoint::operator=(const common_prompt_checkpoint & other) {
    if (this != &other) {
        *this = common_prompt_checkpoint(other);  // one member list: the copy ctor above
    }
    return *this;
}

common_prompt_checkpoint::common_prompt_checkpoint(common_prompt_checkpoint && other) noexcept            = default;
common_prompt_checkpoint & common_prompt_checkpoint::operator=(common_prompt_checkpoint && other) noexcept = default;

size_t common_prompt_checkpoint::size_without_shadow() const {
    return data_tgt.size() + data_dft.size() + accel.size(); // accel.ring was omitted pre-[R6]
}

size_t common_prompt_checkpoint::size() const {
    return size_without_shadow() + llama_vbr_checkpoint_shadow_size(shadow_handle(*this));
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;
    id_task  = -1; // was omitted [R6]

    pos_min = 0;
    pos_max = 0;

    representation_epoch     = 0; // [I9]
    representation_epoch_swa = 0;

    computation_frontier.clear();

    data_tgt.clear();
    data_dft.clear();
    accel.clear(); // ring omission fixed in [R6]

    shadow.reset();
}

common_checkpoint_shadow_reason common_checkpoint_shadow_capture(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier) {
    ckpt.shadow.reset();
    if (ctx == nullptr) {
        return common_checkpoint_shadow_reason::invalid_arguments;
    }

    vbr_checkpoint_frontier_fields fields;
    fields.execution_identity          = frontier.execution_identity.c_str();
    fields.execution_identity_len      = frontier.execution_identity.size();
    fields.adapter_config_identity     = frontier.adapter_config_identity.c_str();
    fields.adapter_config_identity_len = frontier.adapter_config_identity.size();
    fields.media_content_identity      = frontier.media_content_identity.c_str();
    fields.media_content_identity_len  = frontier.media_content_identity.size();
    fields.sequence_epoch              = frontier.sequence_epoch;
    fields.token_count                 = frontier.token_count;
    fields.next_position               = frontier.next_position;

    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(llama_get_memory(ctx), seq_id, &fields, &result);
    if (result.handle != nullptr) {
        common_checkpoint_shadow_attach(ckpt, result.handle);
        if (!ckpt.shadow) {
            return common_checkpoint_shadow_reason::internal_error;
        }
    }
    return common_shadow_reason(result.reason);
}

bool common_checkpoint_shadow_complete(const common_prompt_checkpoint & ckpt) {
    return llama_vbr_checkpoint_shadow_status(shadow_handle(ckpt)) ==
           vbr_checkpoint_generation_status::complete;
}

bool common_checkpoint_shadow_equal(const common_prompt_checkpoint & a, const common_prompt_checkpoint & b) {
    return llama_vbr_checkpoint_shadow_equal(shadow_handle(a), shadow_handle(b));
}

void common_checkpoint_shadow_adopt(common_prompt_checkpoint & dst, common_prompt_checkpoint & src) {
    dst.shadow = std::move(src.shadow);
}

uint64_t common_checkpoint_shadow_dropped_on_copy() {
    return g_shadow_dropped_on_copy.load(std::memory_order_relaxed);
}

const char * common_checkpoint_shadow_reason_name(common_checkpoint_shadow_reason reason) {
    // one string table: the bridge owns the log vocabulary
    return llama_vbr_checkpoint_shadow_reason_name(static_cast<vbr_checkpoint_capture_reason>(reason));
}

namespace {

// One component of the §9.3 proof. A retained payload must be reproduced byte-identically; an
// applicable component with nothing retained can only pass vacuously (nothing to prove against
// -> anything currently present refuses rather than guessing).
common_checkpoint_refresh_verdict refresh_check_component(
        const std::vector<uint8_t> & retained,
        const std::vector<uint8_t> * current,
        bool                         applicable) {
    if (retained.empty() && !applicable) {
        return common_checkpoint_refresh_verdict::proven;
    }
    // the component is retained and/or applicable: it MUST be reproduced (F1 — a null
    // observation is never proven for an applicable component)
    if (current == nullptr) {
        return common_checkpoint_refresh_verdict::refused_cannot_reproduce;
    }
    if (retained.empty()) {
        return current->empty() ? common_checkpoint_refresh_verdict::proven
                                : common_checkpoint_refresh_verdict::refused_cannot_reproduce;
    }
    return *current == retained ? common_checkpoint_refresh_verdict::proven
                                : common_checkpoint_refresh_verdict::refused_byte_mismatch;
}

}  // namespace

common_checkpoint_refresh_verdict common_checkpoint_shadow_refresh_proof(
        const common_prompt_checkpoint &              retained,
        const common_checkpoint_refresh_observation & current) {
    const common_checkpoint_refresh_verdict verdicts[] = {
        refresh_check_component(retained.data_tgt,   current.tgt,  true),
        refresh_check_component(retained.data_dft,   current.dft,  current.dft_applicable),
        refresh_check_component(retained.accel.ring, current.ring, current.ring_applicable),
        refresh_check_component(retained.accel.spec, current.spec, current.spec_applicable),
    };
    // a byte mismatch (nondeterminism evidence) dominates a cannot-reproduce refusal
    common_checkpoint_refresh_verdict result = common_checkpoint_refresh_verdict::proven;
    for (const auto verdict : verdicts) {
        if (verdict == common_checkpoint_refresh_verdict::refused_byte_mismatch) {
            return verdict;
        }
        if (verdict == common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
            result = verdict;
        }
    }
    return result;
}
