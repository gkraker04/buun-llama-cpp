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
SHADOW_REASON_MIRROR(controller_unavailable);
#undef SHADOW_REASON_MIRROR

#define SHADOW_SCOPE_MIRROR(name) \
    static_assert(int(common_checkpoint_reset_scope::name) == int(vbr_checkpoint_reset_scope::name), \
                  "reset scope enums diverged: " #name)
SHADOW_SCOPE_MIRROR(none);
SHADOW_SCOPE_MIRROR(capturing_slot);
SHADOW_SCOPE_MIRROR(global);
#undef SHADOW_SCOPE_MIRROR

#define SHADOW_CATEGORY_MIRROR(name) \
    static_assert(int(common_checkpoint_shadow_category::name) == int(vbr_checkpoint_shadow_category::name), \
                  "shadow category enums diverged: " #name)
SHADOW_CATEGORY_MIRROR(not_applicable);
SHADOW_CATEGORY_MIRROR(generation_unknown);
SHADOW_CATEGORY_MIRROR(strict_accept);
SHADOW_CATEGORY_MIRROR(live_rebased_shadow_accept);
SHADOW_CATEGORY_MIRROR(strict_reject);
#undef SHADOW_CATEGORY_MIRROR

#define SHADOW_EVAL_REASON_MIRROR(name) \
    static_assert(int(common_checkpoint_shadow_eval_reason::name) == int(vbr_checkpoint_shadow_reason::name), \
                  "shadow evaluation reason enums diverged: " #name)
SHADOW_EVAL_REASON_MIRROR(none);
SHADOW_EVAL_REASON_MIRROR(capability_not_applicable);
SHADOW_EVAL_REASON_MIRROR(record_unknown);
SHADOW_EVAL_REASON_MIRROR(record_version);
SHADOW_EVAL_REASON_MIRROR(identity_or_frontier);
SHADOW_EVAL_REASON_MIRROR(controller_shape);
SHADOW_EVAL_REASON_MIRROR(child_order);
SHADOW_EVAL_REASON_MIRROR(dependency_mode);
SHADOW_EVAL_REASON_MIRROR(controller_inactive);
SHADOW_EVAL_REASON_MIRROR(controller_unstable);
SHADOW_EVAL_REASON_MIRROR(pool_uuid);
SHADOW_EVAL_REASON_MIRROR(global_generation);
SHADOW_EVAL_REASON_MIRROR(unit_shape);
SHADOW_EVAL_REASON_MIRROR(unit_unstable);
SHADOW_EVAL_REASON_MIRROR(unit_generation);
SHADOW_EVAL_REASON_MIRROR(live_rebased_transition);
SHADOW_EVAL_REASON_MIRROR(stream_shape);
SHADOW_EVAL_REASON_MIRROR(stream_order);
SHADOW_EVAL_REASON_MIRROR(malformed_page_refs);
SHADOW_EVAL_REASON_MIRROR(page_out_of_range);
SHADOW_EVAL_REASON_MIRROR(dependency_changed);
SHADOW_EVAL_REASON_MIRROR(dependency_membership_lost);
SHADOW_EVAL_REASON_MIRROR(dependency_cardinality);
#undef SHADOW_EVAL_REASON_MIRROR

#define SHADOW_TOMBSTONE_MIRROR(name) \
    static_assert(int(common_checkpoint_shadow_tombstone::name) == int(vbr_checkpoint_shadow_tombstone::name), \
                  "shadow tombstone enums diverged: " #name)
SHADOW_TOMBSTONE_MIRROR(none);
SHADOW_TOMBSTONE_MIRROR(restore_one_behind);
SHADOW_TOMBSTONE_MIRROR(swa_wrap);
SHADOW_TOMBSTONE_MIRROR(explicit_destructive_trim);
SHADOW_TOMBSTONE_MIRROR(dependency_seq_removed);
SHADOW_TOMBSTONE_MIRROR(unexplained);
#undef SHADOW_TOMBSTONE_MIRROR

#define SHADOW_OBSERVATION_MIRROR(name) \
    static_assert(int(common_checkpoint_shadow_observation::name) == int(vbr_checkpoint_shadow_observation::name), \
                  "shadow observation enums diverged: " #name)
SHADOW_OBSERVATION_MIRROR(trivial_append);
SHADOW_OBSERVATION_MIRROR(boundary_refined);
SHADOW_OBSERVATION_MIRROR(destructive);
SHADOW_OBSERVATION_MIRROR(import_refined);
#undef SHADOW_OBSERVATION_MIRROR

#define SHADOW_ORACLE_MIRROR(name) \
    static_assert(int(common_checkpoint_oracle_outcome::name) == int(vbr_checkpoint_oracle_outcome::name), \
                  "oracle outcome enums diverged: " #name)
SHADOW_ORACLE_MIRROR(disabled);
SHADOW_ORACLE_MIRROR(not_due);
SHADOW_ORACLE_MIRROR(pass);
SHADOW_ORACLE_MIRROR(set_mismatch);
SHADOW_ORACLE_MIRROR(byte_mismatch);
SHADOW_ORACLE_MIRROR(set_and_byte_mismatch);
SHADOW_ORACLE_MIRROR(unavailable);
#undef SHADOW_ORACLE_MIRROR

// Closed-count pins (verify r1 finding 10): the LAST value of every mirrored enum is pinned
// numerically on BOTH sides of the seam, so any reorder or midlist insertion breaks this TU.
// Append-detection is completed by the src-side _count sentinels (substrate half).
static_assert(int(common_checkpoint_shadow_reason::controller_unavailable) == 7 &&
              int(vbr_checkpoint_capture_reason::controller_unavailable) == 7,
              "capture reason enum count drifted");
static_assert(int(common_checkpoint_reset_scope::global) == 2 &&
              int(vbr_checkpoint_reset_scope::global) == 2,
              "reset scope enum count drifted");
static_assert(int(common_checkpoint_shadow_category::strict_reject) == 4 &&
              int(vbr_checkpoint_shadow_category::strict_reject) == 4,
              "shadow category enum count drifted");
static_assert(int(common_checkpoint_shadow_eval_reason::dependency_cardinality) == 22 &&
              int(vbr_checkpoint_shadow_reason::dependency_cardinality) == 22,
              "shadow evaluation reason enum count drifted");
static_assert(int(common_checkpoint_shadow_tombstone::unexplained) == 5 &&
              int(vbr_checkpoint_shadow_tombstone::unexplained) == 5,
              "shadow tombstone enum count drifted");
static_assert(int(common_checkpoint_shadow_observation::import_refined) == 3 &&
              int(vbr_checkpoint_shadow_observation::import_refined) == 3,
              "shadow observation enum count drifted");
static_assert(int(common_checkpoint_oracle_outcome::unavailable) == 6 &&
              int(vbr_checkpoint_oracle_outcome::unavailable) == 6,
              "oracle outcome enum count drifted");

common_checkpoint_shadow_reason common_shadow_reason(vbr_checkpoint_capture_reason reason) {
    return static_cast<common_checkpoint_shadow_reason>(reason);
}

vbr_checkpoint_frontier_fields shadow_frontier_fields(const common_computation_frontier & frontier) {
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
    return fields;
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
    cache_family(other.cache_family),
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
    cache_family = {};

    data_tgt.clear();
    data_dft.clear();
    accel.clear(); // ring omission fixed in [R6]

    shadow.reset();
}

common_checkpoint_shadow_reason common_checkpoint_shadow_capture_scoped(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier,
        common_checkpoint_reset_scope &     reset_scope) {
    ckpt.shadow.reset();
    reset_scope = common_checkpoint_reset_scope::capturing_slot;
    if (ctx == nullptr) {
        return common_checkpoint_shadow_reason::invalid_arguments;
    }

    const auto fields = shadow_frontier_fields(frontier);

    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(llama_get_memory(ctx), seq_id, &fields, &result);
    reset_scope = static_cast<common_checkpoint_reset_scope>(result.reset_scope);
    if (result.handle != nullptr) {
        common_checkpoint_shadow_attach(ckpt, result.handle);
        if (!ckpt.shadow) {
            return common_checkpoint_shadow_reason::internal_error;
        }
    }
    return common_shadow_reason(result.reason);
}

common_checkpoint_shadow_reason common_checkpoint_shadow_capture(
        common_prompt_checkpoint &          ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier) {
    common_checkpoint_reset_scope scope;
    return common_checkpoint_shadow_capture_scoped(ckpt, ctx, seq_id, frontier, scope);
}

common_checkpoint_shadow_evaluation common_checkpoint_shadow_evaluate(
        const common_prompt_checkpoint &    ckpt,
        llama_context *                     ctx,
        llama_seq_id                        seq_id,
        const common_computation_frontier & frontier) {
    common_checkpoint_shadow_evaluation out;
    const auto * handle = shadow_handle(ckpt);
    if (handle == nullptr || ctx == nullptr) {
        return out;
    }

    const auto fields = shadow_frontier_fields(frontier);

    // The ONE common call into the G-only bridge (single-evaluator CI pins this file).
    llama_vbr_checkpoint_shadow_evaluation result;
    llama_vbr_checkpoint_shadow_evaluate(handle, llama_get_memory(ctx), seq_id, &fields, &result);

    out.strict              = result.strict;
    out.live_rebased_shadow = result.live_rebased_shadow;
    out.category            = static_cast<common_checkpoint_shadow_category>(result.category);
    out.reason              = static_cast<common_checkpoint_shadow_eval_reason>(result.reason);
    out.observation_class   = static_cast<common_checkpoint_shadow_observation>(result.observation_class);
    out.tombstone_class     = static_cast<common_checkpoint_shadow_tombstone>(result.tombstone_class);
    out.refinement_used     = result.refinement_used;
    out.rejecting_cells     = result.rejecting_cells;
    out.oracle_outcome      = static_cast<common_checkpoint_oracle_outcome>(result.oracle_outcome);
    out.evaluated           = result.evaluator_invocations == 1;
    return out;
}

const char * common_checkpoint_shadow_category_name(common_checkpoint_shadow_category category) {
    switch (category) {
        case common_checkpoint_shadow_category::not_applicable:             return "not_applicable";
        case common_checkpoint_shadow_category::generation_unknown:         return "generation_unknown";
        case common_checkpoint_shadow_category::strict_accept:              return "strict_accept";
        case common_checkpoint_shadow_category::live_rebased_shadow_accept: return "live_rebased_shadow_accept";
        case common_checkpoint_shadow_category::strict_reject:              return "strict_reject";
    }
    return "invalid";
}

const char * common_checkpoint_shadow_eval_reason_name(common_checkpoint_shadow_eval_reason reason) {
    switch (reason) {
        case common_checkpoint_shadow_eval_reason::none:                       return "none";
        case common_checkpoint_shadow_eval_reason::capability_not_applicable:  return "capability_not_applicable";
        case common_checkpoint_shadow_eval_reason::record_unknown:             return "record_unknown";
        case common_checkpoint_shadow_eval_reason::record_version:             return "record_version";
        case common_checkpoint_shadow_eval_reason::identity_or_frontier:       return "identity_or_frontier";
        case common_checkpoint_shadow_eval_reason::controller_shape:           return "controller_shape";
        case common_checkpoint_shadow_eval_reason::child_order:                return "child_order";
        case common_checkpoint_shadow_eval_reason::dependency_mode:            return "dependency_mode";
        case common_checkpoint_shadow_eval_reason::controller_inactive:        return "controller_inactive";
        case common_checkpoint_shadow_eval_reason::controller_unstable:        return "controller_unstable";
        case common_checkpoint_shadow_eval_reason::pool_uuid:                  return "pool_uuid";
        case common_checkpoint_shadow_eval_reason::global_generation:          return "global_generation";
        case common_checkpoint_shadow_eval_reason::unit_shape:                 return "unit_shape";
        case common_checkpoint_shadow_eval_reason::unit_unstable:              return "unit_unstable";
        case common_checkpoint_shadow_eval_reason::unit_generation:            return "unit_generation";
        case common_checkpoint_shadow_eval_reason::live_rebased_transition:    return "live_rebased_transition";
        case common_checkpoint_shadow_eval_reason::stream_shape:               return "stream_shape";
        case common_checkpoint_shadow_eval_reason::stream_order:               return "stream_order";
        case common_checkpoint_shadow_eval_reason::malformed_page_refs:        return "malformed_page_refs";
        case common_checkpoint_shadow_eval_reason::page_out_of_range:          return "page_out_of_range";
        case common_checkpoint_shadow_eval_reason::dependency_changed:         return "dependency_changed";
        case common_checkpoint_shadow_eval_reason::dependency_membership_lost: return "dependency_membership_lost";
        case common_checkpoint_shadow_eval_reason::dependency_cardinality:     return "dependency_cardinality";
    }
    return "unknown";
}

const char * common_checkpoint_shadow_tombstone_name(common_checkpoint_shadow_tombstone tombstone) {
    switch (tombstone) {
        case common_checkpoint_shadow_tombstone::none:                      return "none";
        case common_checkpoint_shadow_tombstone::restore_one_behind:        return "restore_one_behind";
        case common_checkpoint_shadow_tombstone::swa_wrap:                  return "swa_wrap";
        case common_checkpoint_shadow_tombstone::explicit_destructive_trim: return "explicit_destructive_trim";
        case common_checkpoint_shadow_tombstone::dependency_seq_removed:    return "dependency_seq_removed";
        case common_checkpoint_shadow_tombstone::unexplained:               return "unexplained";
    }
    return "unknown";
}

const char * common_checkpoint_shadow_observation_name(common_checkpoint_shadow_observation observation) {
    switch (observation) {
        case common_checkpoint_shadow_observation::trivial_append:  return "trivial_append";
        case common_checkpoint_shadow_observation::boundary_refined: return "boundary_refined";
        case common_checkpoint_shadow_observation::destructive:     return "destructive";
        case common_checkpoint_shadow_observation::import_refined:  return "import_refined";
    }
    return "unknown";
}

const char * common_checkpoint_oracle_outcome_name(common_checkpoint_oracle_outcome outcome) {
    switch (outcome) {
        case common_checkpoint_oracle_outcome::disabled:              return "disabled";
        case common_checkpoint_oracle_outcome::not_due:               return "not_due";
        case common_checkpoint_oracle_outcome::pass:                  return "pass";
        case common_checkpoint_oracle_outcome::set_mismatch:          return "set_mismatch";
        case common_checkpoint_oracle_outcome::byte_mismatch:         return "byte_mismatch";
        case common_checkpoint_oracle_outcome::set_and_byte_mismatch: return "set_and_byte_mismatch";
        case common_checkpoint_oracle_outcome::unavailable:           return "unavailable";
    }
    return "unknown";
}

const char * common_checkpoint_reset_scope_name(common_checkpoint_reset_scope scope) {
    switch (scope) {
        case common_checkpoint_reset_scope::none:           return "none";
        case common_checkpoint_reset_scope::capturing_slot: return "capturing_slot";
        case common_checkpoint_reset_scope::global:         return "global";
    }
    return "unknown";
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
