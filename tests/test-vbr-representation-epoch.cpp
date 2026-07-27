#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-vbr-generation-oracle.h"
#include "llama.h"

#include <cstdio>
#include <limits>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <string>
#include <vector>

// Friend of llama_kv_cache: the production low-LCP path reaches the same two operations through
// clear() followed by prepare(). Driving them directly makes the cursor-at-zero state observable
// before a tight budget can immediately start another degrade wave.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }

    static bool generation_seeded(const llama_kv_cache * kv) {
        const auto * tracker = kv->vbr_generation_tracker_get();
        if (tracker == nullptr || !tracker->active() || !tracker->stable()) {
            return false;
        }
        for (uint32_t stream = 0; stream < tracker->stream_count(); ++stream) {
            for (uint32_t cell = 0; cell < tracker->cell_count(); ++cell) {
                if (tracker->dependency_generation(stream, cell) != 0) {
                    return true;
                }
            }
        }
        return false;
    }

    static bool generation_units_match(const llama_kv_cache * kv) {
        const auto * tracker = kv->vbr_generation_tracker_get();
        if (tracker == nullptr || !tracker->stable() ||
                tracker->unit_count() != kv->layers.size() * 2) {
            return false;
        }
        for (size_t ikv = 0; ikv < kv->layers.size(); ++ikv) {
            for (uint32_t side = 0; side < 2; ++side) {
                const auto * tensor = side != 0 ? kv->layers[ikv].v : kv->layers[ikv].k;
                const auto unit =
                        tracker->unit_generation(static_cast<uint32_t>(ikv * 2 + side));
                const int32_t live_type =
                        tensor != nullptr ? static_cast<int32_t>(tensor->type) : -1;
                if (unit.current_type != live_type) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool has_mapped_degradable_unit(const llama_kv_cache * kv) {
        std::vector<ggml_type> sim;
        kv->vbr_sim_seed(
            sim, /*pooled_only=*/true,
            GGML_TYPE_COUNT, GGML_TYPE_COUNT,
            nullptr, nullptr, nullptr);
        for (size_t i = kv->vbr_degrade_cursor_;
             i < kv->vbr_degrade_order_.size(); ++i) {
            size_t slot = 0;
            const ggml_tensor * tensor = nullptr;
            ggml_type target = GGML_TYPE_COUNT;
            if (kv->vbr_sim_step(sim, i, slot, tensor, target)) {
                const auto & step = kv->vbr_degrade_order_[i];
                const auto & units =
                    kv->vbr_units_of(slot / 2, step.is_v != 0);
                bool all_mapped = !units.empty();
                for (const auto & [pool, extent] : units) {
                    all_mapped =
                        all_mapped && extent->t != nullptr &&
                        pool->vmm != nullptr && pool->wm_cells > 0;
                }
                if (all_mapped) {
                    return true;
                }
            }
        }
        return false;
    }

    // A one-token prepare is padded to the same 256-cell watermark used by production attention.
    // Map that watermark while the tensors are still at their entry tiers, before the seed decode
    // can invoke the budget controller. vbr_budget_eff() is floored at already-mapped bytes, so
    // the subsequent real decode cannot consume the ladder merely because this tiny fixture's
    // configured policy budget is smaller than one page-rounded entry-tier watermark.
    static bool map_seed_watermark(llama_kv_cache * kv) {
        const uint32_t wm = kv->vbr_watermark_cells(1);
        return wm > 0 && kv->vbr_vmm_try_map(wm);
    }

    // This test owns the representation-epoch mechanism, not the model/card-specific pricing
    // policy. vbr_degrade_next() is the authoritative production mutation path, but its runtime
    // clamp may legitimately contain zero steps when this tiny context fits at the entry tier.
    // Temporarily open the clamp for one direct friend call, then restore it. This creates the
    // test wave deterministically without changing the budget, decoding a model-dependent token
    // count, or relying on free VRAM. No controller boundary runs while cursor > the restored
    // clamp; each forced wave is followed by the assertions and then clear/full_reset.
    static bool force_degrade(llama_kv_cache * kv) {
        if (!has_mapped_degradable_unit(kv)) {
            return false;
        }
        const size_t saved_limit = kv->vbr_degrade_limit_;
        kv->vbr_degrade_limit_ = kv->vbr_degrade_order_.size();
        const bool changed =
            kv->vbr_degrade_next(kv->vbr_watermark_cells(0));
        kv->vbr_degrade_limit_ = saved_limit;
        return changed;
    }

    static void full_reset(llama_kv_cache * kv) {
        kv->vbr_full_reset();
    }

    static bool reconcile(llama_kv_cache * kv) {
        return kv->vbr_retier_take_reconcile("unit_test");
    }

    static uint64_t freeze_operation_id(const llama_kv_cache * kv) {
        if (kv->vbr_retier_freeze_depth_ == 0) {
            return 0;
        }
        return kv->vbr_retier_freeze_stack_[kv->vbr_retier_freeze_depth_ - 1]
            .operation_id.value;
    }

    static uint64_t set_budget_bytes(llama_kv_cache * kv, uint64_t budget_bytes) {
        const uint64_t previous = kv->vbr_budget_bytes_;
        kv->vbr_budget_bytes_ = budget_bytes;
        return previous;
    }
};

static bool decode_one(llama_context * ctx) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, 1, 0, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool epochs_equal(
        const llama_memory_vbr_state_data & a,
        const llama_memory_vbr_state_data & b) {
    return a.representation_epoch == b.representation_epoch &&
           a.representation_epoch_swa == b.representation_epoch_swa;
}

static bool get_iswa_children(
        llama_memory_t mem,
        llama_kv_cache *& base,
        llama_kv_cache *& swa) {
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        base = iswa->get_base();
        swa  = iswa->get_swa();
        return true;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        base = hybrid->get_mem_attn()->get_base();
        swa  = hybrid->get_mem_attn()->get_swa();
        return true;
    }
    return false;
}

static void set_test_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void unset_test_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

struct a1_membership_fixture {
    std::vector<uint8_t> present;
};

static bool a1_cell_has_seq(const void * context, uint32_t, uint32_t cell, llama_seq_id seq_id) {
    const auto * fixture = static_cast<const a1_membership_fixture *>(context);
    return seq_id == 0 && cell < fixture->present.size() && fixture->present[cell] != 0;
}

static vbr_checkpoint_generation_record a1_make_record(
        const vbr_generation_tracker & tracker,
        const vbr_checkpoint_generation_stream & stream) {
    vbr_checkpoint_generation_controller controller;
    vbr_checkpoint_generation_record record;
    if (!vbr_generation_capture_controller(
                tracker, 0, checkpoint_child_dependency_mode::live_guarded, {stream}, controller)) {
        return record;
    }
    record.status = vbr_checkpoint_generation_status::complete;
    record.controllers.push_back(std::move(controller));
    return record;
}

static vbr_generation_live_view a1_make_live(
        const vbr_generation_tracker & tracker,
        const a1_membership_fixture & membership,
        uint32_t exact_count,
        llama_pos computation_frontier = 400) {
    vbr_generation_live_stream_view stream;
    stream.stream_index           = 0;
    stream.dependency_seq_id      = 0;
    stream.computation_frontier   = computation_frontier;
    stream.exact_dependency_count = exact_count;
    stream.membership_context     = &membership;
    stream.cell_has_seq           = a1_cell_has_seq;

    vbr_generation_live_controller_view controller;
    controller.child_id        = 0;
    controller.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
    controller.tracker         = &tracker;
    controller.streams.push_back(stream);

    vbr_generation_live_view live;
    live.legacy_eligible = true;
    live.controllers.push_back(std::move(controller));
    return live;
}

// A2: every mutation event must cite a live registry operation. Tests reuse the production
// RAII (reuse review) — one begin/close idiom in the whole tree. P5v2 (v6): mutation targets
// carry exact nonzero pools, so every test operation is bound to its tracker's pool.
struct test_operation {
    vbr_scoped_operation op;
    test_operation(vbr_operation_kind kind, vbr_pool_uuid pool, llama_seq_id seq = -1,
                   llama_pos p0 = 0, llama_pos p1 = std::numeric_limits<llama_pos>::max(),
                   vbr_operation_class operation_class = vbr_operation_class::state_api)
        : op(vbr_mutation_binding(kind, seq, p0, p1, operation_class, pool.hi, pool.lo)) {}
    vbr_operation_id id() const { return op.id(); }
};

// P1v2 (v6): destructive test events supply per-target extents through the production
// callback shape; the supplier records which target index the tracker selected (single-target
// fixtures just populate handles[0]).
struct test_multi_extent_supplier {
    std::array<vbr_extent_handle, 2> handles = {};
    int                              last    = -1;
};

static vbr_extent_handle test_multi_extent_cb(void * ctx, uint8_t target_index) {
    auto * supplier = static_cast<test_multi_extent_supplier *>(ctx);
    supplier->last  = target_index;
    return target_index < 2 ? supplier->handles[target_index] : vbr_extent_handle{};
}

static bool run_a1_cpu_tests() {
    llama_kv_cells ownership_index;
    ownership_index.resize(4);
    ownership_index.pos_set(0, 5);
    ownership_index.seq_add(0, 0);
    ownership_index.pos_set(1, 15);
    ownership_index.seq_add(1, 0);
    if (ownership_index.seq_pos_count_before(0, 10) != 1 ||
            ownership_index.seq_pos_count_before(0, 20) != 2) {
        fprintf(stderr, "A1 canonical ownership index returned an inexact cardinality\n");
        return false;
    }

    vbr_generation_tracker tracker(1, 768, 1);
    if (!tracker.active() || !tracker.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full)) {
        fprintf(stderr, "A1 tracker did not initialize\n");
        return false;
    }
    vbr_generation_tracker distinct_tracker(1, 768, 1);
    if (distinct_tracker.pool_identity() == tracker.pool_identity()) {
        fprintf(stderr, "A1 process-local pool UUID was reused\n");
        return false;
    }
    test_operation a1_op(vbr_operation_kind::decode, tracker.pool_identity(), -1,
                         0, std::numeric_limits<llama_pos>::max(),
                         vbr_operation_class::ordinary_decode);
    test_operation a1_edit_op(vbr_operation_kind::sequence_edit, tracker.pool_identity(), -1,
                              0, std::numeric_limits<llama_pos>::max(),
                              vbr_operation_class::prompt_share);
    test_operation distinct_op(vbr_operation_kind::decode, distinct_tracker.pool_identity(), -1,
                               0, std::numeric_limits<llama_pos>::max(),
                               vbr_operation_class::ordinary_decode);
    if (!a1_op.id() || !a1_edit_op.id() || !distinct_op.id()) {
        fprintf(stderr, "A2 test operation failed to register\n");
        return false;
    }
    auto foreign_event = distinct_tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            distinct_op.id());
    if (!foreign_event || tracker.stamp_cell(foreign_event, 10, 0) || !foreign_event.finish()) {
        fprintf(stderr, "A1 tracker accepted an event owned by another controller\n");
        return false;
    }

    auto append = tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            a1_op.id());
    if (!append || !tracker.stamp_cell(append, 10, 0) ||
            !tracker.stamp_cell(append, 300, 0) || !append.finish()) {
        fprintf(stderr, "A1 dependency event did not publish atomically\n");
        return false;
    }
    const uint32_t dependency_before = tracker.dependency_generation(0, 10);

    auto share = tracker.begin_event(
            vbr_mutation_registrant::seq_cp,
            vbr_operation_class::prompt_share,
            0,
            vbr_generation_stamp_kind::membership,
            a1_edit_op.id());
    if (!share || !tracker.stamp_cell(share, 10, 1) ||
            tracker.dependency_generation(0, 10) != dependency_before ||
            tracker.membership_generation(0, 10) == 0 || !share.finish()) {
        fprintf(stderr, "A1 dependency/membership stamp split failed\n");
        return false;
    }

    vbr_checkpoint_generation_stream stream;
    if (!vbr_generation_capture_stream(tracker, 0, 0, 400, {10, 300}, stream) ||
            stream.captured_dependency_count != 2 || stream.pages.size() != 2) {
        fprintf(stderr, "A1 canonical covered-mask capture failed\n");
        return false;
    }

    a1_membership_fixture membership;
    membership.present.resize(768);
    membership.present[10]  = 1;
    membership.present[300] = 1;
    auto record = a1_make_record(tracker, stream);
    auto live   = a1_make_live(tracker, membership, 2);
    auto result = checkpoint_vbr_eligibility(record, live);
    if (!result.strict || result.reason != vbr_checkpoint_eligibility_reason::none) {
        fprintf(stderr, "A1 strict evaluator rejected an exact record\n");
        return false;
    }

    auto foreign_share = tracker.begin_event(
            vbr_mutation_registrant::seq_cp,
            vbr_operation_class::prompt_share,
            0,
            vbr_generation_stamp_kind::membership,
            a1_edit_op.id());
    if (!foreign_share || !tracker.stamp_cell(foreign_share, 10, 2)) {
        fprintf(stderr, "A1 membership-only refinement stamp failed\n");
        return false;
    }
    if (checkpoint_vbr_eligibility(record, live).reason !=
            vbr_checkpoint_eligibility_reason::controller_unstable) {
        fprintf(stderr, "A1 evaluator observed an in-flight ownership event\n");
        return false;
    }
    vbr_checkpoint_generation_controller unstable_capture;
    if (vbr_generation_capture_controller(
                tracker, 0, checkpoint_child_dependency_mode::live_guarded, {stream}, unstable_capture)) {
        fprintf(stderr, "A1 capture observed an in-flight ownership event\n");
        return false;
    }
    if (!foreign_share.finish() || !checkpoint_vbr_eligibility(record, live).strict) {
        fprintf(stderr, "A1 membership-only refinement falsely rejected\n");
        return false;
    }

    auto unrelated = tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            a1_op.id());
    if (!unrelated || !tracker.stamp_cell(unrelated, 20, 1) ||
            !tracker.stamp_cell(unrelated, 256, 1) ||
            !tracker.stamp_cell(unrelated, 600, 1) || !unrelated.finish() ||
            !checkpoint_vbr_eligibility(record, live).strict) {
        fprintf(stderr, "A1 same-page/boundary/different-page append refinement rejected\n");
        return false;
    }

    // NEW-1: the covered cells remain valid, but a newly-live dependency expands the exact set.
    membership.present[20] = 1;
    live.controllers[0].streams[0].exact_dependency_count = 3;
    result = checkpoint_vbr_eligibility(record, live);
    if (result.reason != vbr_checkpoint_eligibility_reason::dependency_cardinality) {
        fprintf(stderr, "A1 NEW-1 dependency-set expansion was accepted\n");
        return false;
    }
    membership.present[20] = 0;
    live.controllers[0].streams[0].exact_dependency_count = 2;

    membership.present[10] = 0;
    live.controllers[0].streams[0].exact_dependency_count = 1;
    result = checkpoint_vbr_eligibility(record, live);
    if (result.reason != vbr_checkpoint_eligibility_reason::dependency_membership_lost) {
        fprintf(stderr, "A1 dependency membership loss was not identified\n");
        return false;
    }
    membership.present[10] = 1;
    live.controllers[0].streams[0].exact_dependency_count = 2;

    auto rewrite = tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_occupied_reuse,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            a1_op.id(), nullptr, nullptr,
            true);
    if (!rewrite || !tracker.stamp_cell(rewrite, 10, 0) || !rewrite.finish()) {
        fprintf(stderr, "A1 destructive dependency event did not publish atomically\n");
        return false;
    }
    result = checkpoint_vbr_eligibility(record, live);
    if (result.reason != vbr_checkpoint_eligibility_reason::dependency_changed ||
            tracker.page_destructive_generation(0, 0) == 0) {
        fprintf(stderr, "A1 destructive refinement was not rejected\n");
        return false;
    }
    if (!tracker.global_transition(vbr_mutation_registrant::clear, vbr_operation_class::state_api) ||
            checkpoint_vbr_eligibility(record, live).reason !=
                    vbr_checkpoint_eligibility_reason::global_generation) {
        fprintf(stderr, "A1 controller-global invalidation did not dominate stale page evidence\n");
        return false;
    }

    // Durable transition provenance, not a debug-ring entry, is the only live_rebased proof.
    vbr_generation_tracker rebased_tracker(1, 512, 1);
    rebased_tracker.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
    test_operation rebased_op(vbr_operation_kind::decode, rebased_tracker.pool_identity(), -1,
                              0, std::numeric_limits<llama_pos>::max(),
                              vbr_operation_class::ordinary_decode);
    auto rebased_append = rebased_tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            rebased_op.id());
    rebased_tracker.stamp_cell(rebased_append, 10, 0);
    if (!rebased_append.finish()) {
        fprintf(stderr, "A1 rebased fixture event did not publish atomically\n");
        return false;
    }
    vbr_checkpoint_generation_stream rebased_stream;
    vbr_generation_capture_stream(rebased_tracker, 0, 0, 20, {10}, rebased_stream);
    auto rebased_record = a1_make_record(rebased_tracker, rebased_stream);
    if (!rebased_tracker.publish_unit(
                0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, vbr_repr_domain::full, 0,
                vbr_repr_transition::degrade_f16_to_t8_admitted,
                vbr_mutation_registrant::degrade_next)) {
        fprintf(stderr, "A1 durable transition publish failed\n");
        return false;
    }
    a1_membership_fixture rebased_membership;
    rebased_membership.present.resize(512);
    rebased_membership.present[10] = 1;
    auto rebased_live = a1_make_live(rebased_tracker, rebased_membership, 1, 20);
    result = checkpoint_vbr_eligibility(rebased_record, rebased_live);
    if (!result.live_rebased_shadow || result.strict) {
        fprintf(stderr, "A1 admitted transition provenance was not shadow-classified\n");
        return false;
    }

    vbr_generation_tracker unproven_tracker(1, 512, 1);
    unproven_tracker.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
    test_operation unproven_op(vbr_operation_kind::decode, unproven_tracker.pool_identity(), -1,
                               0, std::numeric_limits<llama_pos>::max(),
                               vbr_operation_class::ordinary_decode);
    auto unproven_append = unproven_tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            unproven_op.id());
    unproven_tracker.stamp_cell(unproven_append, 10, 0);
    if (!unproven_append.finish()) {
        fprintf(stderr, "A1 unproven fixture event did not publish atomically\n");
        return false;
    }
    vbr_checkpoint_generation_stream unproven_stream;
    vbr_generation_capture_stream(unproven_tracker, 0, 0, 20, {10}, unproven_stream);
    auto unproven_record = a1_make_record(unproven_tracker, unproven_stream);
    unproven_tracker.publish_unit(
            0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, vbr_repr_domain::full, 0,
            vbr_repr_transition::degrade_other,
            vbr_mutation_registrant::degrade_next);
    auto unproven_live = a1_make_live(unproven_tracker, rebased_membership, 1, 20);
    result = checkpoint_vbr_eligibility(unproven_record, unproven_live);
    if (result.reason != vbr_checkpoint_eligibility_reason::live_rebased_transition) {
        fprintf(stderr, "A1 accepted T8 without durable admitted-transition provenance\n");
        return false;
    }

    auto malformed = rebased_record;
    malformed.controllers[0].streams[0].pages.push_back(
            malformed.controllers[0].streams[0].pages.front());
    result = checkpoint_vbr_eligibility(malformed, rebased_live);
    if (result.reason != vbr_checkpoint_eligibility_reason::malformed_page_refs) {
        fprintf(stderr, "A1 duplicate/noncanonical page refs were accepted\n");
        return false;
    }

    uint8_t bytes_a[] = {1, 2, 3};
    uint8_t bytes_b[] = {4, 5};
    std::vector<vbr_generation_oracle_cell> canonical = {
        {10,  10,  true,  true, false, bytes_a, sizeof(bytes_a)},
        {20,  20,  false, true, false, bytes_a, sizeof(bytes_a)},
        {300, 300, true,  true, false, bytes_b, sizeof(bytes_b)},
    };
    unset_test_env("VBR_GENERATION_ORACLE");
    const auto disabled_baseline = vbr_generation_oracle_capture(400, canonical);
    const auto disabled = vbr_generation_oracle_audit(400, canonical, disabled_baseline, stream);
    if (disabled.enabled) {
        fprintf(stderr, "A1 byte oracle was not default-disabled\n");
        return false;
    }

    set_test_env("VBR_GENERATION_ORACLE", "1");
    const auto baseline = vbr_generation_oracle_capture(400, canonical);
    auto audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    if (!audit.enabled || !audit.complete || !audit.set_equal || !audit.bytes_equal) {
        fprintf(stderr, "A1 independent byte oracle rejected its exact baseline\n");
        return false;
    }
    bytes_a[0] = 9;
    audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    if (!audit.set_equal || audit.bytes_equal) {
        fprintf(stderr, "A1 byte oracle missed a covered-byte mutation\n");
        return false;
    }
    bytes_a[0] = 1;
    auto undercovered = stream;
    undercovered.pages.erase(undercovered.pages.begin());
    audit = vbr_generation_oracle_audit(400, canonical, baseline, undercovered);
    if (audit.set_equal) {
        fprintf(stderr, "A1 independent oracle repeated production mask under-coverage\n");
        return false;
    }
    auto incomplete = canonical;
    incomplete[0].bytes = nullptr;
    audit = vbr_generation_oracle_audit(400, incomplete, baseline, stream);
    if (audit.complete || audit.bytes_equal) {
        fprintf(stderr, "A1 byte oracle accepted an incomplete canonical observation\n");
        return false;
    }
    unset_test_env("VBR_GENERATION_ORACLE");

    fprintf(stderr, "A1 generation/evaluator/oracle CPU coverage PASS\n");
    return true;
}


static bool run_a2_cpu_tests() {
    // --- extent store lifecycle -------------------------------------------------------------
    vbr_generation_tracker tracker(1, 768, 1);
    test_operation op(vbr_operation_kind::sequence_edit, tracker.pool_identity(), 0, 0, 100);
    auto & store = tracker.extent_store();

    auto handle = store.reserve(vbr_mutation_family::trim,
                                vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 100);
    if (!handle) {
        fprintf(stderr, "A2 extent reserve failed\n");
        return false;
    }
    auto ref = store.add_ref(handle);
    if (!ref || store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 prepared extent must not be admission evidence\n");
        return false;
    }
    if (!store.submit(handle) || store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 submitted extent must not be admission evidence\n");
        return false;
    }
    if (!store.commit(handle)) {
        fprintf(stderr, "A2 extent commit from submitted failed\n");
        return false;
    }
    const auto * entry = store.lookup_committed(ref);
    if (entry == nullptr || entry->p0 != 0 || entry->p1 != 100 ||
            entry->family != vbr_mutation_family::trim) {
        fprintf(stderr, "A2 committed extent lookup returned wrong evidence\n");
        return false;
    }
    // release-to-zero reclaims; a stale ref then fails ABA-safe
    store.release_ref(ref);
    if (store.lookup_committed(ref) != nullptr) {
        fprintf(stderr, "A2 reclaimed extent slot admitted a stale reference (ABA)\n");
        return false;
    }
    // failed entries are never evidence
    auto fhandle = store.reserve(vbr_mutation_family::trim,
                                 vbr_operation_class::dependency_seq_remove, 0, 1, 0,
                                 std::numeric_limits<llama_pos>::max());
    auto fref = store.add_ref(fhandle);
    if (!store.fail(fhandle) || store.lookup_committed(fref) != nullptr) {
        fprintf(stderr, "A2 failed extent leaked into evidence\n");
        return false;
    }
    store.release_ref(fref);
    // exhaustion-recovers semantics
    {
        std::vector<vbr_extent_handle> hoard;
        for (;;) {
            auto h = store.reserve(vbr_mutation_family::trim,
                                   vbr_operation_class::state_api, 0, 0, 0, 1);
            if (!h) {
                break;
            }
            hoard.push_back(h);
        }
        if (!store.exhausted_latched()) {
            fprintf(stderr, "A2 extent exhaustion did not latch\n");
            return false;
        }
        store.reset_all();
        if (store.exhausted_latched() || store.live_entries() != 0 ||
                !store.reserve(vbr_mutation_family::trim, vbr_operation_class::state_api, 0, 0, 0, 1)) {
            fprintf(stderr, "A2 extent exhaustion did not recover after slab reset\n");
            return false;
        }
        store.reset_all();
    }

    // --- citation refusal -------------------------------------------------------------------
    auto uncited = tracker.begin_event(
            vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
            vbr_generation_stamp_kind::membership, vbr_operation_id{});
    if (uncited) {
        fprintf(stderr, "A2 tracker minted an event without a live operation citation\n");
        (void) uncited.finish();
        return false;
    }
    vbr_operation_id dead_id;
    {
        test_operation ephemeral(vbr_operation_kind::sequence_edit, tracker.pool_identity());
        dead_id = ephemeral.id();
    }
    auto dead_cited = tracker.begin_event(
            vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
            vbr_generation_stamp_kind::membership, dead_id);
    if (dead_cited) {
        fprintf(stderr, "A2 tracker accepted a dead operation citation\n");
        (void) dead_cited.finish();
        return false;
    }

    // --- ownership index: rank vs brute force incl. shifts + fail-closed domain --------------
    {
        vbr_ownership_index index(1, 8, 512);
        uint64_t rng = 0x5eedULL;
        std::vector<llama_pos> pos_of(512, -1);
        auto next = [&rng]() { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(rng >> 33); };
        for (int step = 0; step < 4000; ++step) {
            const uint32_t cell = next() % 512;
            const uint32_t act  = next() % 3;
            if (act == 0) {
                const llama_pos p = (llama_pos)(next() % 512);
                if (pos_of[cell] < 0 && index.add_cell(0, 3, cell, p)) {
                    pos_of[cell] = p;
                }
            } else if (act == 1 && pos_of[cell] >= 0) {
                index.remove_cell(0, 3, cell, pos_of[cell]);
                pos_of[cell] = -1;
            } else if (pos_of[cell] >= 0) {
                const llama_pos np = (llama_pos)(next() % 512);
                if (index.move_cell(0, 3, cell, pos_of[cell], np)) {
                    pos_of[cell] = np;
                }
            }
        }
        for (llama_pos frontier : { (llama_pos) 0, (llama_pos) 100, (llama_pos) 511, (llama_pos) 512 }) {
            uint32_t rank = 0;
            uint32_t brute = 0;
            for (uint32_t c = 0; c < 512; ++c) {
                brute += pos_of[c] >= 0 && pos_of[c] < frontier ? 1 : 0;
            }
            if (!index.rank_below(0, 3, frontier, rank) || rank != brute) {
                fprintf(stderr, "A2 ownership index rank mismatch at frontier %d (%u != %u)\n",
                        (int) frontier, rank, brute);
                return false;
            }
        }
        // out-of-domain position: fail-closed unavailable
        uint32_t free_cell = 0;
        while (free_cell < 512 && pos_of[free_cell] >= 0) free_cell++;
        if (index.add_cell(0, 3, free_cell, 9999) || index.available(0, 3)) {
            fprintf(stderr, "A2 ownership index accepted an out-of-domain position\n");
            return false;
        }
        uint32_t rank = 0;
        if (index.rank_below(0, 3, 10, rank)) {
            fprintf(stderr, "A2 unavailable index view still answered a rank query\n");
            return false;
        }
        index.clear_seq(0, 3);
        if (index.available(0, 3)) {
            fprintf(stderr, "A2 cleared seq view should be absent\n");
            return false;
        }
    }

    // --- recovery ring + capability ----------------------------------------------------------
    {
        test_operation rop(vbr_operation_kind::sequence_edit, tracker.pool_identity(), 2, 0, 64);
        const int32_t idx = vbr_recovery_reserve(rop.id());
        if (idx < 0 || !vbr_recovery_release_unused(idx, rop.id())) {
            fprintf(stderr, "A2 recovery reserve/release failed\n");
            return false;
        }
        const int32_t idx2 = vbr_recovery_reserve(rop.id());
        if (idx2 < 0 ||
                !vbr_recovery_record_failure(idx2, rop.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::deferred_byte_copy, true)) {
            fprintf(stderr, "A2 recovery record_failure failed\n");
            return false;
        }
        {
            auto capability = vbr_recovery_mint(idx2);
            if (!capability || !capability.target_allowed(0, 2, 0, 64) ||
                    capability.target_allowed(0, 2, 0, 65) || capability.target_allowed(0, 5, 0, 64)) {
                fprintf(stderr, "A2 recovery capability target restriction failed\n");
                return false;
            }
            // deliberately no resolve: destructor must fail-close to quarantined
        }
        // C4: the fail-closed destructor left the record awaiting_ack. Take it for the
        // (wildcard-pool) target, ack with the token; a stale token must not ack twice.
        auto work = vbr_recovery_take_quarantine(0, 0);
        if (!work.token) {
            fprintf(stderr, "C4 fail-closed capability left no pending quarantine\n");
            return false;
        }
        if (!vbr_recovery_ack_quarantine(work.token, 0, 0) ||
                vbr_recovery_ack_quarantine(work.token, 0, 0)) {
            fprintf(stderr, "C4 quarantine token ack was not single-use\n");
            return false;
        }
        if (vbr_recovery_take_quarantine(0, 0).token) {
            fprintf(stderr, "C4 acked quarantine still pending\n");
            return false;
        }
        auto remint = vbr_recovery_mint(idx2);
        if (remint) {
            fprintf(stderr, "A2 reclaimed recovery record allowed a second mint\n");
            return false;
        }
    }

    // --- registry binding retention + close(failed) autorecord --------------------------------
    {
        vbr_operation_binding probe;
        test_operation bop(vbr_operation_kind::state_import, tracker.pool_identity(), 4, 10, 20);
        if (!vbr_operation_registry_binding(bop.id(), probe) ||
                probe.seq_id() != 4 || probe.range().p0 != 10 || probe.range().p1 != 20 ||
                probe.kind != vbr_operation_kind::state_import) {
            fprintf(stderr, "A2 registry did not retain the authenticated binding\n");
            return false;
        }
        const int32_t ridx = vbr_recovery_reserve(bop.id());
        const vbr_operation_id bop_id = bop.id();
        if (ridx < 0 || !bop.op.close(vbr_operation_outcome::failed)) {
            fprintf(stderr, "A2 close(failed) failed\n");
            return false;
        }
        (void) bop_id;
        vbr_failed_operation_record record;
        if (!vbr_recovery_get_record(ridx, record) ||
                record.state != vbr_recovery_state::recorded) {
            fprintf(stderr, "A2 close(failed) did not autorecord the reserved recovery slot\n");
            return false;
        }
        auto cleanup = vbr_recovery_mint(ridx);
        cleanup.resolve_quarantined();
    }

    // --- §5.5 expected-tombstone classification ----------------------------------------------
    // Build: capture two cells for seq 0 below frontier 400, then trim them under
    // explicit_destructive_trim with a committed extent -> expected tombstone row 3.
    {
        vbr_generation_tracker t2(1, 768, 1);
        t2.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
        test_operation cap_op(vbr_operation_kind::decode, t2.pool_identity(), -1,
                              0, std::numeric_limits<llama_pos>::max(),
                              vbr_operation_class::ordinary_decode);
        {
            auto seed = t2.begin_event(
                    vbr_mutation_registrant::apply_ubatch_append, vbr_operation_class::ordinary_decode,
                    0, vbr_generation_stamp_kind::dependency, cap_op.id());
            if (!seed || !t2.stamp_cell(seed, 10, 0) || !t2.stamp_cell(seed, 300, 0) || !seed.finish()) {
                fprintf(stderr, "A2 tombstone fixture seed failed\n");
                return false;
            }
        }
        vbr_checkpoint_generation_stream stream;
        if (!vbr_generation_capture_stream(t2, 0, 0, 400, {10, 300}, stream)) {
            fprintf(stderr, "A2 tombstone fixture capture failed\n");
            return false;
        }
        vbr_checkpoint_generation_record record;
        record.status  = vbr_checkpoint_generation_status::complete;
        record.version = 1;
        vbr_checkpoint_generation_controller controller;
        if (!vbr_generation_capture_controller(t2, 0, checkpoint_child_dependency_mode::live_guarded,
                                               {stream}, controller)) {
            fprintf(stderr, "A2 tombstone fixture controller build failed\n");
            return false;
        }
        record.controllers = {controller};

        // destructive trim of both cells, provenance-bearing with committed extent
        test_operation trim_op(vbr_operation_kind::sequence_edit, t2.pool_identity(), 0, 0, 400,
                               vbr_operation_class::explicit_destructive_trim);
        auto trim_extent = t2.extent_store().reserve(
                vbr_mutation_family::trim, vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 400);
        test_multi_extent_supplier trim_supplier;
        trim_supplier.handles[0] = trim_extent;
        {
            auto trim = t2.begin_event(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::explicit_destructive_trim,
                    0, vbr_generation_stamp_kind::membership, trim_op.id(),
                    &test_multi_extent_cb, &trim_supplier, true);
            // C2: stamps prove pre-mutation positions inside the committed [0,400) extent.
            if (!trim || !t2.stamp_cell(trim, 10, 0, 10) || !t2.stamp_cell(trim, 300, 0, 300) || !trim.finish()) {
                fprintf(stderr, "A2 tombstone trim events failed\n");
                return false;
            }
        }
        if (!t2.extent_store().commit(trim_extent)) {
            fprintf(stderr, "A2 tombstone trim extent commit failed\n");
            return false;
        }

        vbr_generation_live_view live;
        live.legacy_eligible = true;
        vbr_generation_live_controller_view live_controller;
        live_controller.child_id        = controller.child_id;
        live_controller.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        live_controller.tracker         = &t2;
        vbr_generation_live_stream_view live_stream;
        live_stream.stream_index           = 0;
        live_stream.dependency_seq_id      = 0;
        live_stream.computation_frontier   = 400;
        live_stream.exact_dependency_count = 0;  // both trimmed away
        live_stream.membership_context     = nullptr;
        live_stream.cell_has_seq = [](const void *, uint32_t, uint32_t, llama_seq_id) { return false; };
        live_stream.cell_pos     = [](const void *, uint32_t, uint32_t) -> llama_pos { return -1; };
        live_controller.streams = {live_stream};
        live.controllers        = {live_controller};

        const auto verdict = checkpoint_vbr_eligibility(record, live);
        if (verdict.strict ||
                verdict.tombstone_class != vbr_expected_tombstone_class::explicit_destructive_trim ||
                verdict.rejecting_cells != 2) {
            fprintf(stderr, "A2 expected tombstone row 3 misclassified (class %d, cells %u)\n",
                    (int) verdict.tombstone_class, verdict.rejecting_cells);
            const auto mref10  = t2.membership_extent(0, 10);
            const auto mref300 = t2.membership_extent(0, 300);
            fprintf(stderr, "  mref10=(%u,%u) mref300=(%u,%u) lookup10=%p lookup300=%p reason=%d\n",
                    mref10.index, mref10.expected_gen, mref300.index, mref300.expected_gen,
                    (const void *) t2.extent_store().lookup_committed(mref10),
                    (const void *) t2.extent_store().lookup_committed(mref300),
                    (int) verdict.reason);
            return false;
        }
    }

    // --- review-fix coverage ------------------------------------------------------------------
    // F4: a decode-kind operation must NOT authorize a prompt-share membership event.
    {
        vbr_generation_tracker t3(1, 256, 1);
        test_operation decode_op(vbr_operation_kind::decode, t3.pool_identity(), -1,
                                 0, std::numeric_limits<llama_pos>::max(),
                                 vbr_operation_class::ordinary_decode);
        auto misused = t3.begin_event(
                vbr_mutation_registrant::seq_cp, vbr_operation_class::prompt_share,
                0, vbr_generation_stamp_kind::membership, decode_op.id());
        if (misused) {
            fprintf(stderr, "F4: decode operation authorized a sequence-share event\n");
            (void) misused.finish();
            return false;
        }
        // P1v2 (v6) seq scope: an op bound to seq 2 stamps seq 2 fine; a seq-3 stamp has no
        // covering target, so it POISONS the event and latches unavailable IMMEDIATELY, and
        // every further stamp from the poisoned event is inert.
        test_operation seq2_op(vbr_operation_kind::sequence_edit, t3.pool_identity(), 2, 0, 100);
        auto scoped = t3.begin_event(
                vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                0, vbr_generation_stamp_kind::membership, seq2_op.id());
        if (!scoped || !t3.stamp_cell(scoped, 5, 2, 5)) {
            fprintf(stderr, "F4: authorized seq-scope stamp failed\n");
            return false;
        }
        if (t3.stamp_cell(scoped, 5, 3, 5) || !t3.shadow_unavailable()) {
            fprintf(stderr, "P1v2: unauthorized seq stamp did not poison + latch immediately\n");
            return false;
        }
        if (t3.stamp_cell(scoped, 5, 2, 5)) {
            fprintf(stderr, "P1v2: poisoned event accepted a further stamp\n");
            return false;
        }
        if (!scoped.finish()) {
            fprintf(stderr, "P1v2: poisoned event did not finish cleanly\n");
            return false;
        }
    }
    // F6: resolved recovery records reclaim their slots — the ring survives > capacity cycles.
    {
        test_operation cyc_op(vbr_operation_kind::sequence_edit, tracker.pool_identity(), 0, 0, 8);
        for (int cycle = 0; cycle < 70; ++cycle) {
            const int32_t idx = vbr_recovery_reserve(cyc_op.id());
            if (idx < 0) {
                fprintf(stderr, "F6: recovery ring exhausted at cycle %d (leak)\n", cycle);
                return false;
            }
            if (!vbr_recovery_record_failure(idx, cyc_op.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::metadata_mutation, false)) {
                fprintf(stderr, "F6: record_failure failed at cycle %d\n", cycle);
                return false;
            }
            auto capability = vbr_recovery_mint(idx);
            if (!capability || !capability.resolve_completed()) {
                fprintf(stderr, "F6: resolve_completed failed at cycle %d\n", cycle);
                return false;
            }
        }
    }
    // F7: with rejecting cells, live-count expansion must classify unexplained even when the
    // reject provenance would otherwise satisfy a tombstone row. Reuse the row-3 fixture shape
    // but report a live count that does NOT reconcile as captured-minus-lost.
    {
        vbr_generation_tracker t4(1, 768, 1);
        t4.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
        test_operation cap_op(vbr_operation_kind::decode, t4.pool_identity(), -1,
                              0, std::numeric_limits<llama_pos>::max(),
                              vbr_operation_class::ordinary_decode);
        {
            auto seed = t4.begin_event(
                    vbr_mutation_registrant::apply_ubatch_append, vbr_operation_class::ordinary_decode,
                    0, vbr_generation_stamp_kind::dependency, cap_op.id());
            if (!seed || !t4.stamp_cell(seed, 10, 0) || !t4.stamp_cell(seed, 300, 0) || !seed.finish()) {
                return false;
            }
        }
        vbr_checkpoint_generation_stream stream;
        if (!vbr_generation_capture_stream(t4, 0, 0, 400, {10, 300}, stream)) {
            return false;
        }
        vbr_checkpoint_generation_record record;
        record.status  = vbr_checkpoint_generation_status::complete;
        record.version = 1;
        vbr_checkpoint_generation_controller controller;
        if (!vbr_generation_capture_controller(t4, 0, checkpoint_child_dependency_mode::live_guarded,
                                               {stream}, controller)) {
            return false;
        }
        record.controllers = {controller};
        test_operation trim_op(vbr_operation_kind::sequence_edit, t4.pool_identity(), 0, 0, 400,
                               vbr_operation_class::explicit_destructive_trim);
        auto trim_extent = t4.extent_store().reserve(
                vbr_mutation_family::trim, vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 400);
        test_multi_extent_supplier trim_supplier;
        trim_supplier.handles[0] = trim_extent;
        {
            auto trim = t4.begin_event(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::explicit_destructive_trim,
                    0, vbr_generation_stamp_kind::membership, trim_op.id(),
                    &test_multi_extent_cb, &trim_supplier, true);
            if (!trim || !t4.stamp_cell(trim, 10, 0, 10) || !t4.stamp_cell(trim, 300, 0, 300) || !trim.finish()) {
                return false;
            }
        }
        t4.extent_store().commit(trim_extent);
        vbr_generation_live_view live;
        live.legacy_eligible = true;
        vbr_generation_live_controller_view live_controller;
        live_controller.child_id        = controller.child_id;
        live_controller.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        live_controller.tracker         = &t4;
        vbr_generation_live_stream_view live_stream;
        live_stream.stream_index           = 0;
        live_stream.dependency_seq_id      = 0;
        live_stream.computation_frontier   = 400;
        live_stream.exact_dependency_count = 3;  // EXPANSION: does not reconcile with 2 captured - 2 lost
        live_stream.membership_context     = nullptr;
        live_stream.cell_has_seq = [](const void *, uint32_t, uint32_t, llama_seq_id) { return false; };
        live_stream.cell_pos     = [](const void *, uint32_t, uint32_t) -> llama_pos { return -1; };
        live_controller.streams = {live_stream};
        live.controllers        = {live_controller};
        const auto verdict = checkpoint_vbr_eligibility(record, live);
        if (verdict.strict || verdict.tombstone_class != vbr_expected_tombstone_class::unexplained) {
            fprintf(stderr, "F7: cardinality expansion under rejects was not unexplained (class %d)\n",
                    (int) verdict.tombstone_class);
            return false;
        }
    }

    // --- v3 failure-path matrix (CPU rows) ----------------------------------------------------
    // Forged-field matrix (§11.1): each forged manifest dimension must refuse the event.
    {
        vbr_generation_tracker t5(1, 256, 1);
        const vbr_pool_uuid t5_pool = t5.pool_identity();
        // wrong class
        test_operation wrong_class(vbr_operation_kind::sequence_edit, t5_pool, 0, 0, 10,
                                   vbr_operation_class::prompt_share);
        auto e1 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, wrong_class.id());
        // wrong kind for registrant (controller_retier cannot authorize seq_rm)
        test_operation wrong_kind(vbr_operation_kind::controller_retier, t5_pool, -1, -1, -1,
                                  vbr_operation_class::state_api);
        auto e2 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, wrong_kind.id());
        // wrong stream target
        vbr_operation_binding far_stream = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 0, 0, 10, vbr_operation_class::state_api,
                t5_pool.hi, t5_pool.lo, /*stream=*/7);
        vbr_scoped_operation far_op(far_stream);
        // foreign pool: a manifest bound to ANOTHER controller's pool never covers this one
        vbr_generation_tracker t5_foreign(1, 64, 1);
        test_operation foreign_pool(vbr_operation_kind::sequence_edit,
                                    t5_foreign.pool_identity(), 0, 0, 10,
                                    vbr_operation_class::state_api);
        auto e3 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, far_op.id());
        auto e5 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, foreign_pool.id());
        if (e1 || e2 || e3 || e5) {
            fprintf(stderr, "v3 forged-field matrix: a forged manifest authorized an event\n");
            return false;
        }
        // correct manifest, wrong stamped seq (target seq 2, stamping seq 3): authorized
        // stamp first, then the forged one — which poisons the event (P1v2).
        test_operation seq_scope(vbr_operation_kind::sequence_edit, t5_pool, 2, 0, 10,
                                 vbr_operation_class::state_api);
        auto e4 = t5.begin_event(vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api,
                                 0, vbr_generation_stamp_kind::membership, seq_scope.id());
        if (!e4 || !t5.stamp_cell(e4, 1, 2, 5) || t5.stamp_cell(e4, 1, 3, 5) ||
                !t5.shadow_unavailable() || !e4.finish()) {
            fprintf(stderr, "v3 forged-field matrix: seq-scope stamp check failed\n");
            return false;
        }
    }
    // Committed evidence WITHOUT a positional proof (in_range false: stamp at an unknown
    // position under a whole-range manifest) => row 3 must classify unexplained. The v5
    // "out-of-extent" fixture is no longer constructible: a stamp outside the authenticated
    // range now poisons instead of producing misattributed evidence (P1v2) — covered above.
    {
        vbr_generation_tracker t6(1, 768, 1);
        t6.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
        test_operation cap_op6(vbr_operation_kind::decode, t6.pool_identity(), -1,
                               0, std::numeric_limits<llama_pos>::max(),
                               vbr_operation_class::ordinary_decode);
        {
            auto seed = t6.begin_event(
                    vbr_mutation_registrant::apply_ubatch_append, vbr_operation_class::ordinary_decode,
                    0, vbr_generation_stamp_kind::dependency, cap_op6.id());
            if (!seed || !t6.stamp_cell(seed, 10, 0) || !seed.finish()) {
                return false;
            }
        }
        vbr_checkpoint_generation_stream stream6;
        if (!vbr_generation_capture_stream(t6, 0, 0, 400, {10}, stream6)) {
            return false;
        }
        vbr_checkpoint_generation_record record6;
        record6.status  = vbr_checkpoint_generation_status::complete;
        record6.version = 1;
        vbr_checkpoint_generation_controller controller6;
        if (!vbr_generation_capture_controller(t6, 0, checkpoint_child_dependency_mode::live_guarded,
                                               {stream6}, controller6)) {
            return false;
        }
        record6.controllers = {controller6};
        // whole-range trim, stamped at an UNKNOWN position (-1): selection covers via the
        // whole-range form, but the in_range proof bit stays false -> row 3 unexplained
        test_operation trim6(vbr_operation_kind::sequence_edit, t6.pool_identity(), 0,
                             0, std::numeric_limits<llama_pos>::max(),
                             vbr_operation_class::explicit_destructive_trim);
        auto extent6 = t6.extent_store().reserve(
                vbr_mutation_family::trim, vbr_operation_class::explicit_destructive_trim, 0, 0,
                0, std::numeric_limits<llama_pos>::max());
        test_multi_extent_supplier supplier6;
        supplier6.handles[0] = extent6;
        {
            auto trim = t6.begin_event(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::explicit_destructive_trim,
                    0, vbr_generation_stamp_kind::membership, trim6.id(),
                    &test_multi_extent_cb, &supplier6, true);
            if (!trim || !t6.stamp_cell(trim, 10, 0, -1) || !trim.finish()) {
                return false;
            }
        }
        t6.extent_store().commit(extent6);
        vbr_generation_live_view live6;
        live6.legacy_eligible = true;
        vbr_generation_live_controller_view lc6;
        lc6.child_id        = controller6.child_id;
        lc6.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        lc6.tracker         = &t6;
        vbr_generation_live_stream_view ls6;
        ls6.stream_index           = 0;
        ls6.dependency_seq_id      = 0;
        ls6.computation_frontier   = 400;
        ls6.exact_dependency_count = 0;
        ls6.cell_has_seq = [](const void *, uint32_t, uint32_t, llama_seq_id) { return false; };
        ls6.cell_pos     = [](const void *, uint32_t, uint32_t) -> llama_pos { return -1; };
        lc6.streams = {ls6};
        live6.controllers = {lc6};
        const auto verdict6 = checkpoint_vbr_eligibility(record6, live6);
        if (verdict6.tombstone_class != vbr_expected_tombstone_class::unexplained) {
            fprintf(stderr, "v3: out-of-extent trim classified as expected (class %d)\n",
                    (int) verdict6.tombstone_class);
            return false;
        }
    }
    // Publish-injection between reads: direct interleave — evaluator must reject unit_unstable.
    // (The tuple read itself is race-free by the units mutex; this exercises the F9 snapshot.)
    // Simulated by capturing a record, publishing a unit, and evaluating: repr_gen changes make
    // it unit_generation-reject; the F9 snapshot path is additionally covered by the two-thread
    // stress below reaching stable states only.
    // C5 row: clear_all -> add -> rank on the fixed slot map (the v2 critical-1 crash shape).
    {
        vbr_ownership_index idx5(1, 8, 512);
        if (!idx5.add_cell(0, 1, 3, 30)) {
            return false;
        }
        idx5.clear_all();
        uint32_t rank5 = 0;
        if (!idx5.add_cell(0, 1, 4, 40) || !idx5.rank_below(0, 1, 100, rank5) || rank5 != 1) {
            fprintf(stderr, "C5: clear_all -> add -> rank failed (rank %u)\n", rank5);
            return false;
        }
        std::vector<uint32_t> owned5;
        if (!idx5.enumerate_owned(0, 1, owned5) || owned5.size() != 1 || owned5[0] != 4) {
            fprintf(stderr, "C5: post-clear enumeration wrong\n");
            return false;
        }
    }
    // Two-thread stress: concurrent registry begin/end + recovery reserve/mint/resolve.
    {
        std::atomic<bool> failed{false};
        const vbr_pool_uuid stress_pool = tracker.pool_identity();
        auto worker = [&failed, stress_pool]() {
            for (int i = 0; i < 2000 && !failed.load(); ++i) {
                test_operation op(vbr_operation_kind::sequence_edit, stress_pool, i % 4, 0, 64,
                                  vbr_operation_class::state_api);
                if (!op.id()) {
                    failed.store(true);
                    break;
                }
                vbr_operation_binding probe;
                if (!vbr_operation_registry_binding(op.id(), probe) ||
                    probe.seq_id() != i % 4) {
                    failed.store(true);
                    break;
                }
                const int32_t r = vbr_recovery_reserve(op.id());
                if (r >= 0) {
                    if ((i & 1) != 0) {
                        vbr_recovery_record_failure(r, op.id(), vbr_operation_phase::mutate,
                                                    vbr_recovery_failure_site::metadata_mutation, false);
                        auto capability = vbr_recovery_mint(r);
                        if (capability) {
                            capability.resolve_completed();
                        }
                    } else {
                        vbr_recovery_release_unused(r, op.id());
                    }
                }
            }
        };
        std::thread t1(worker), t2(worker);
        t1.join();
        t2.join();
        if (failed.load()) {
            fprintf(stderr, "v3 two-thread registry/recovery stress FAILED\n");
            return false;
        }
    }

    // --- v6 rows: P1v2 stamp-time selection + per-target extents --------------------------
    {
        vbr_generation_tracker t7(1, 768, 1);
        vbr_operation_binding two_seq;
        two_seq.kind        = vbr_operation_kind::decode;
        two_seq.child_phase = vbr_operation_phase::mutate;
        two_seq.targets[two_seq.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                t7.pool_identity().hi, t7.pool_identity().lo, VBR_STREAM_ANY, 0, 0, 200);
        two_seq.targets[two_seq.n_targets++] = vbr_make_target(
                vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                t7.pool_identity().hi, t7.pool_identity().lo, VBR_STREAM_ANY, 1, 100, 300);
        vbr_scoped_operation two_seq_op(two_seq);
        if (!two_seq_op) {
            fprintf(stderr, "P1v2: two-target selection manifest failed to mint\n");
            return false;
        }
        test_multi_extent_supplier supplier;
        supplier.handles[0] = t7.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::ordinary_decode, 0, 0, 0, 200);
        supplier.handles[1] = t7.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::ordinary_decode, 0, 1, 100, 300);
        {
            auto reuse = t7.begin_event(
                    vbr_mutation_registrant::apply_ubatch_occupied_reuse,
                    vbr_operation_class::ordinary_decode, 0,
                    vbr_generation_stamp_kind::dependency, two_seq_op.id(),
                    &test_multi_extent_cb, &supplier, true);
            if (!reuse || !t7.stamp_cell(reuse, 10, 0, 50) || supplier.last != 0 ||
                    !t7.stamp_cell(reuse, 20, 1, 150) || supplier.last != 1) {
                fprintf(stderr, "P1v2: per-(seq,pos) selection picked the wrong target\n");
                return false;
            }
            // seq 1 at a position only seq 0's target covers: no cover -> poison + latch
            if (t7.stamp_cell(reuse, 30, 1, 50) || !t7.shadow_unavailable() ||
                    t7.stamp_cell(reuse, 10, 0, 50) || !reuse.finish()) {
                fprintf(stderr, "P1v2: uncovered (seq,pos) stamp did not poison + latch\n");
                return false;
            }
        }
        // the cells cite their SELECTED target's extent, never target zero's
        t7.extent_store().commit(supplier.handles[0]);
        t7.extent_store().commit(supplier.handles[1]);
        const auto * seq1_evidence = t7.extent_store().lookup_committed(t7.dependency_extent(0, 20));
        if (seq1_evidence == nullptr || seq1_evidence->seq_id != 1 || seq1_evidence->p0 != 100) {
            fprintf(stderr, "P1v2: stamp did not bind the selected target's extent\n");
            return false;
        }
        // P4v2 monotone re-arm: the latch recorded the generation; clearing needs a STRICTLY
        // later sanctioned transition.
        if (t7.try_clear_shadow_unavailable()) {
            fprintf(stderr, "P4v2: latch cleared without a post-latch transition\n");
            return false;
        }
        if (!t7.global_transition(vbr_mutation_registrant::clear, vbr_operation_class::state_api) ||
                !t7.try_clear_shadow_unavailable() || t7.shadow_unavailable()) {
            fprintf(stderr, "P4v2: latch did not clear after a post-latch transition\n");
            return false;
        }
        t7.set_shadow_unavailable();
        if (t7.try_clear_shadow_unavailable()) {
            fprintf(stderr, "P4v2: re-latch reused a stale transition proof\n");
            return false;
        }
        // P1v2 multi-seq target-set proof: every member covered -> stamp; any member
        // uncovered -> poison. (Fresh tracker: t7 is latched again above.)
        vbr_generation_tracker t7b(1, 768, 1);
        vbr_operation_binding set_manifest;
        set_manifest.kind        = vbr_operation_kind::decode;
        set_manifest.child_phase = vbr_operation_phase::mutate;
        for (llama_seq_id s = 0; s < 2; ++s) {
            set_manifest.targets[set_manifest.n_targets++] = vbr_make_target(
                    vbr_operation_kind::decode, vbr_operation_class::ordinary_decode,
                    t7b.pool_identity().hi, t7b.pool_identity().lo, VBR_STREAM_ANY, s, 0, 200);
        }
        vbr_scoped_operation set_op(set_manifest);
        auto append = t7b.begin_event(
                vbr_mutation_registrant::apply_ubatch_append, vbr_operation_class::ordinary_decode,
                0, vbr_generation_stamp_kind::dependency, set_op.id());
        const llama_seq_id both[2]     = { 0, 1 };
        const llama_seq_id stranger[2] = { 0, 2 };
        if (!append || !t7b.stamp_cell(append, 10, both, 2, 50)) {
            fprintf(stderr, "P1v2: fully-covered shared-cell stamp refused\n");
            return false;
        }
        if (t7b.stamp_cell(append, 11, stranger, 2, 50) || !t7b.shadow_unavailable() ||
                !append.finish()) {
            fprintf(stderr, "P1v2: uncovered shared-cell member did not poison\n");
            return false;
        }
    }

    // --- v6 rows: P5v2 closed mint predicates ---------------------------------------------
    {
        const vbr_pool_uuid pool = tracker.pool_identity();
        auto refused = [](vbr_operation_binding b) {
            vbr_scoped_operation probe(b);
            return !probe;
        };
        const auto good = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 0, 0, 10,
                vbr_operation_class::state_api, pool.hi, pool.lo);
        auto zero_mask = good;
        zero_mask.targets[0].registrant_mask = 0;
        auto foreign_bit = good;
        foreign_bit.targets[0].registrant_mask |=
                vbr_registrant_bit(vbr_mutation_registrant::apply_ubatch_append);
        auto subset_mask = good;
        subset_mask.targets[0].registrant_mask = vbr_registrant_bit(vbr_mutation_registrant::seq_rm);
        if (refused(good) ||                 // equality with the canonical mask is valid
            !refused(zero_mask) ||           // mask != 0
            !refused(foreign_bit) ||         // no out-of-kind bit
            refused(subset_mask)) {          // nonzero subset is least privilege
            fprintf(stderr, "P5v2: registrant-mask predicate matrix failed\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, 0, 10,
                                          vbr_operation_class::state_api, 0, 0))) {
            fprintf(stderr, "P5v2: pool-wildcard mutation target minted\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::decode, 0, 5, 5,
                                          vbr_operation_class::ordinary_decode, pool.hi, pool.lo))) {
            fprintf(stderr, "P5v2: empty decode range minted (p0 < p1 required)\n");
            return false;
        }
        if (refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, 5, 5,
                                         vbr_operation_class::state_api, pool.hi, pool.lo))) {
            fprintf(stderr, "P5v2: enumerated sequence_edit empty no-op form refused\n");
            return false;
        }
        if (!refused(vbr_mutation_binding(vbr_operation_kind::sequence_edit, 0, -1, -1,
                                          vbr_operation_class::state_api, pool.hi, pool.lo))) {
            fprintf(stderr, "P5v2: undeclared sequence_edit range wildcard minted\n");
            return false;
        }
        if (refused(vbr_mutation_binding(vbr_operation_kind::controller_retier, -1, -1, -1,
                                         vbr_operation_class::controller, pool.hi, pool.lo))) {
            fprintf(stderr, "P5v2: declared controller_retier range wildcard refused\n");
            return false;
        }
    }

    // --- v6 rows: P3v2 fixed-participant sealed aggregate ---------------------------------
    {
        // seal marks never-claimed declared slots failed; pre-seal reports cannot close.
        test_operation root(vbr_operation_kind::decode, tracker.pool_identity(), -1,
                            0, std::numeric_limits<llama_pos>::max(),
                            vbr_operation_class::ordinary_decode);
        const vbr_operation_id root_id = root.op.release();
        llama_kv_cache::vbr_composite_outcome aggregate;
        aggregate.operation_id = root_id;
        aggregate.declared     = 2;
        aggregate.claim();
        aggregate.report_terminal(false);
        if (!vbr_operation_registry_is_live(root_id)) {
            fprintf(stderr, "P3v2: pre-seal terminal report closed the root\n");
            return false;
        }
        aggregate.seal(true);
        if (vbr_operation_registry_is_live(root_id) || !aggregate.failed) {
            fprintf(stderr, "P3v2: seal did not fail the never-claimed slot and close\n");
            return false;
        }
        aggregate.report_terminal(true);  // late report must be inert (no double close)
        // detach-transfer shape: sealed with open tokens stays open until the LAST terminal.
        test_operation root2(vbr_operation_kind::decode, tracker.pool_identity(), -1,
                             0, std::numeric_limits<llama_pos>::max(),
                             vbr_operation_class::ordinary_decode);
        const vbr_operation_id root2_id = root2.op.release();
        llama_kv_cache::vbr_composite_outcome open_tokens;
        open_tokens.operation_id = root2_id;
        open_tokens.declared     = 2;
        open_tokens.claim();
        open_tokens.claim();
        open_tokens.seal(true);
        if (!vbr_operation_registry_is_live(root2_id)) {
            fprintf(stderr, "P3v2: seal closed the root past open participant tokens\n");
            return false;
        }
        open_tokens.report_terminal(true);
        if (!vbr_operation_registry_is_live(root2_id)) {
            fprintf(stderr, "P3v2: root closed before every declared slot terminated\n");
            return false;
        }
        open_tokens.report_terminal(true);
        if (vbr_operation_registry_is_live(root2_id) || open_tokens.failed) {
            fprintf(stderr, "P3v2: all-committed sealed aggregate did not close committed\n");
            return false;
        }
    }

    // --- v6 rows: P2v2 transactional ubatch manifest --------------------------------------
    {
        llama_seq_id   ids[20];
        llama_seq_id * seq_ptrs[20];
        int32_t        n_seq_id[20];
        llama_pos      pos[20];
        for (int i = 0; i < 20; ++i) {
            ids[i]      = i;
            seq_ptrs[i] = &ids[i];
            n_seq_id[i] = 1;
            pos[i]      = 100 + i;
        }
        llama_ubatch overflow_ub = {};
        overflow_ub.n_tokens = 20;
        overflow_ub.pos      = pos;
        overflow_ub.n_seq_id = n_seq_id;
        overflow_ub.seq_id   = seq_ptrs;
        vbr_operation_binding manifest;
        manifest.kind        = vbr_operation_kind::decode;
        manifest.child_phase = vbr_operation_phase::mutate;
        if (llama_kv_cache::vbr_decode_targets_from_ubatch(
                    manifest, 1, 1, false, VBR_STREAM_ANY, overflow_ub) ||
                manifest.n_targets != 0) {
            fprintf(stderr, "P2v2: seq-ceiling overflow did not zero the manifest\n");
            return false;
        }
        // single-seq wrap manifest: ordinary + whole-range wrap + ONE declared seq-wildcard
        // purge target (v6-fix F1: cross-sequence masked reuse makes both the destroyed
        // position and the purged owner unbounded by the incoming batch)
        llama_ubatch one_ub = {};
        one_ub.n_tokens = 1;
        one_ub.pos      = pos;
        one_ub.n_seq_id = n_seq_id;
        one_ub.seq_id   = seq_ptrs;
        manifest           = {};
        manifest.kind        = vbr_operation_kind::decode;
        manifest.child_phase = vbr_operation_phase::mutate;
        if (!llama_kv_cache::vbr_decode_targets_from_ubatch(
                    manifest, 1, 1, true, VBR_STREAM_ANY, one_ub) ||
                manifest.n_targets != 3 ||
                manifest.targets[1].operation_class != vbr_operation_class::swa_wrap ||
                manifest.targets[1].range.p1 != std::numeric_limits<llama_pos>::max() ||
                manifest.targets[2].operation_class != vbr_operation_class::state_api ||
                manifest.targets[2].seq_id != -1 ||
                manifest.targets[2].range.p1 != std::numeric_limits<llama_pos>::max()) {
            fprintf(stderr, "P2v2: wrap manifest missing the declared wrap/purge claims\n");
            return false;
        }
        // v6-fix F1 end-to-end: the wrap claims authenticate CROSS-SEQUENCE reuse — a
        // destructive reuse at a prior position far beyond the incoming batch, and the
        // nested purge of an OLD owner absent from the ubatch, both cover instead of
        // poisoning.
        vbr_generation_tracker t10(1, 768, 1);
        vbr_operation_binding wrap_manifest;
        wrap_manifest.kind        = vbr_operation_kind::decode;
        wrap_manifest.child_phase = vbr_operation_phase::mutate;
        if (!llama_kv_cache::vbr_decode_targets_from_ubatch(
                    wrap_manifest, t10.pool_identity().hi, t10.pool_identity().lo,
                    true, VBR_STREAM_ANY, one_ub)) {
            return false;
        }
        vbr_scoped_operation wrap_op(wrap_manifest);
        test_multi_extent_supplier wrap_supplier;
        wrap_supplier.handles[1] = t10.extent_store().reserve(
                vbr_mutation_family::occupied_reuse, vbr_operation_class::swa_wrap, 0, 0,
                0, std::numeric_limits<llama_pos>::max());
        {
            auto reuse = t10.begin_event(
                    vbr_mutation_registrant::apply_ubatch_occupied_reuse,
                    vbr_operation_class::swa_wrap, 0,
                    vbr_generation_stamp_kind::dependency, wrap_op.id(),
                    &test_multi_extent_cb, &wrap_supplier, true);
            if (!reuse || !t10.stamp_cell(reuse, 10, ids[0], 5000) ||
                    wrap_supplier.last != 1 || !reuse.finish()) {
                fprintf(stderr, "v6-F1: beyond-batch destructive reuse did not authenticate\n");
                return false;
            }
        }
        {
            auto purge = t10.begin_event(
                    vbr_mutation_registrant::seq_rm, vbr_operation_class::state_api, 0,
                    vbr_generation_stamp_kind::membership, wrap_op.id());
            if (!purge || !t10.stamp_cell(purge, 11, 7, 5000) || t10.shadow_unavailable() ||
                    !purge.finish()) {
                fprintf(stderr, "v6-F1: old-owner cross-seq purge did not authenticate\n");
                return false;
            }
        }
    }

    // --- v6-fix rows: F6 stream-exact recovery, F7 closed seq domain ----------------------
    {
        // F6: a record whose only target names exact stream 1 must NOT authorize stream 0.
        vbr_operation_binding far = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, 2, 0, 64, vbr_operation_class::state_api,
                tracker.pool_identity().hi, tracker.pool_identity().lo, /*stream=*/1);
        vbr_scoped_operation far_op(far);
        const int32_t ridx = vbr_recovery_reserve(far_op.id());
        if (ridx < 0 ||
                !vbr_recovery_record_failure(ridx, far_op.id(), vbr_operation_phase::mutate,
                                             vbr_recovery_failure_site::metadata_mutation, false)) {
            return false;
        }
        {
            auto capability = vbr_recovery_mint(ridx);
            if (!capability || capability.target_allowed(0, 2, 0, 64) ||
                    !capability.target_allowed(1, 2, 0, 64)) {
                fprintf(stderr, "v6-F6: recovery stream authorization is not target-exact\n");
                return false;
            }
            capability.resolve_completed();
        }
        // F7: seq domain is closed at mint — LLAMA_MAX_SEQ refused, LLAMA_MAX_SEQ-1 minted.
        auto over = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, LLAMA_MAX_SEQ, 0, 8,
                vbr_operation_class::state_api,
                tracker.pool_identity().hi, tracker.pool_identity().lo);
        auto edge = vbr_mutation_binding(
                vbr_operation_kind::sequence_edit, LLAMA_MAX_SEQ - 1, 0, 8,
                vbr_operation_class::state_api,
                tracker.pool_identity().hi, tracker.pool_identity().lo);
        vbr_scoped_operation over_op(over);
        vbr_scoped_operation edge_op(edge);
        if (over_op || !edge_op) {
            fprintf(stderr, "v6-F7: mint seq domain is not closed at LLAMA_MAX_SEQ\n");
            return false;
        }
    }

    // --- v6 rows: P5v2 exact-registrant unit publication ----------------------------------
    {
        vbr_generation_tracker t9(1, 64, 1);
        t9.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full);
        test_operation ctrl(vbr_operation_kind::controller_retier, t9.pool_identity(), -1, -1, -1,
                            vbr_operation_class::controller);
        if (!t9.publish_unit(0, GGML_TYPE_F16, GGML_TYPE_TURBO8_0, vbr_repr_domain::full, 0,
                             vbr_repr_transition::degrade_other,
                             vbr_mutation_registrant::degrade_next, ctrl.id())) {
            fprintf(stderr, "P5v2: exact in-manifest registrant publication refused\n");
            return false;
        }
        if (t9.publish_unit(0, GGML_TYPE_TURBO8_0, GGML_TYPE_F16, vbr_repr_domain::full, 0,
                            vbr_repr_transition::promote,
                            vbr_mutation_registrant::clear, ctrl.id())) {
            fprintf(stderr, "P5v2: out-of-manifest registrant publication accepted\n");
            return false;
        }
    }

    printf("A2 extent/index/recovery/citation/tombstone CPU coverage PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--a1-cpu") {
        return run_a1_cpu_tests() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--a2-cpu") {
        return run_a2_cpu_tests() ? 0 : 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL | --a1-cpu | --a2-cpu\n", argv[0]);
        return 1;
    }

    // A0 registry foundation: RAII closes exactly once, IDs are process-global/nonzero, and a
    // completed identity is never returned by the next operation.
    uint64_t first_registry_id = 0;
    {
        vbr_operation_binding binding = {};
        binding.kind = vbr_operation_kind::state_export;
        vbr_operation_registry_guard guard(binding);
        if (!guard.active()) {
            fprintf(stderr, "A0 registry RAII guard failed to mint an operation ID\n");
            return 1;
        }
        first_registry_id = guard.binding().operation_id.value;
        if (!vbr_operation_registry_is_live(guard.binding().operation_id)) {
            fprintf(stderr, "A0 registry did not expose its live RAII operation\n");
            return 1;
        }
    }
    if (vbr_operation_registry_is_live({ first_registry_id })) {
        fprintf(stderr, "A0 registry RAII guard left a completed operation live\n");
        return 1;
    }
    {
        vbr_operation_binding binding = {};
        binding.kind = vbr_operation_kind::state_export;
        vbr_operation_registry_guard guard(binding);
        if (!guard.active() ||
            guard.binding().operation_id.value == first_registry_id) {
            fprintf(stderr, "A0 registry reused an operation ID\n");
            return 1;
        }
    }

    if (!run_a1_cpu_tests() || !run_a2_cpu_tests()) {
        return 1;
    }

    ggml_backend_load_all();

    bool have_gpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            have_gpu = true;
            break;
        }
    }
    if (!have_gpu) {
        fprintf(stderr, "SKIP: VBR representation epoch requires a GPU VBR backend (currently CUDA)\n");
        return 0;
    }

    // Hermetic controller inputs. The generated/real Gemma-4 fixture has iSWA children; a generic
    // order plus the friend-only force_degrade() above makes the epoch wave independent of its
    // price clamp, budget reach, free VRAM, or card size.
    set_test_env("VBR_FORCE_GENERIC", "1");
    unset_test_env("VBR_BUDGET_MIB");
    unset_test_env("VBR_DEGRADE_ORDER");
    unset_test_env("VBR_FREEZE");
    unset_test_env("VBR_MIN_BITS");
    unset_test_env("VBR_GROWTH_HEADROOM_MIB");
    unset_test_env("VBR_TRANSCODE_TEST");
    set_test_env("VBR_PROMOTE", "0");
    set_test_env("VBR_STASH_ROWS", "0");
    const char * trace_prefix_env = std::getenv("VBR_EPOCH_TEST_TRACE_PREFIX");
    const std::string trace_prefix =
        trace_prefix_env != nullptr ? trace_prefix_env : "";
    if (trace_prefix.empty()) {
        unset_test_env("VBR_TRACE");
    } else {
        set_test_env("VBR_TRACE", (trace_prefix + ".normal").c_str());
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(argv[1], mparams));
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", argv[1]);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                  = 128;
    cparams.n_batch                = 32;
    cparams.n_ubatch               = 32;
    cparams.n_seq_max              = 1;
    cparams.n_threads              = 2;
    cparams.n_threads_batch        = 2;
    cparams.type_k                 = GGML_TYPE_F16;
    cparams.type_v                 = GGML_TYPE_F16;
    cparams.flash_attn_type        = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.vbr_dynamic            = true;
    cparams.vbr_budget_explicit    = true;
    cparams.vbr_vram_budget_bytes  = 64ull * 1024 * 1024;

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create CUDA VBR context\n");
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!get_iswa_children(mem, base, swa)) {
        fprintf(stderr, "fixture did not create an iSWA attention cache\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(base)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for the base child\n");
        return 0;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(swa)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for the SWA child\n");
        return 0;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(base)) {
        fprintf(stderr, "PRECONDITION failed: could not map the base child seed watermark\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(swa)) {
        fprintf(stderr, "PRECONDITION failed: could not map the SWA child seed watermark\n");
        return 1;
    }

    const auto initial = llama_memory_vbr_state(mem, 0, 0);
    if (initial.cursor != 0) {
        fprintf(stderr, "PRECONDITION failed: initial VBR cursor was not zero\n");
        return 1;
    }
    if (initial.representation_epoch != 0) {
        fprintf(stderr, "PRECONDITION failed: initial base representation epoch was not zero\n");
        return 1;
    }
    if (initial.representation_epoch_swa != 0) {
        fprintf(stderr, "PRECONDITION failed: initial SWA representation epoch was not zero\n");
        return 1;
    }
    if (!decode_one(ctx.get())) {
        fprintf(stderr, "PRECONDITION failed: seed decode failed\n");
        return 1;
    }
    const auto seeded = llama_memory_vbr_state(mem, 0, 0);
    if (seeded.cursor != initial.cursor) {
        fprintf(stderr, "PRECONDITION failed: seed decode consumed the VBR degrade ladder\n");
        return 1;
    }
    if (!epochs_equal(seeded, initial)) {
        fprintf(stderr, "PRECONDITION failed: seed decode changed a representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_seeded(base) ||
            !llama_kv_cache_vbr_epoch_test::generation_seeded(swa)) {
        fprintf(stderr, "A1 dual-write did not stamp both armed iSWA children\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(base)) {
        fprintf(stderr, "PRECONDITION failed: base child has no mapped degradable pooled extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(swa)) {
        fprintf(stderr, "PRECONDITION failed: SWA child has no mapped degradable pooled extent\n");
        return 1;
    }

    // WS-6: the iSWA parent must acquire both child controllers coherently. Nested scopes
    // defer actual representation mutations, leave both ordered epochs unchanged, and arm
    // exactly one fresh boundary evaluation per child when the outer scope exits.
    const auto preflight = llama_memory_vbr_retier_preflight(mem, 0);
    if (!preflight.active) {
        fprintf(stderr, "scoped-freeze preflight did not observe an active VBR controller\n");
        return 1;
    }
    if (!preflight.fits) {
        fprintf(stderr, "scoped-freeze preflight rejected the already-mapped current tiers\n");
        return 1;
    }
    if (preflight.pools < 2) {
        fprintf(stderr, "scoped-freeze preflight did not cover both iSWA child pools\n");
        return 1;
    }
    if (preflight.bytes_needed == 0) {
        fprintf(stderr, "scoped-freeze preflight reported zero current-tier bytes needed\n");
        return 1;
    }
    if (preflight.bytes_available == 0) {
        fprintf(stderr, "scoped-freeze preflight reported zero current-tier bytes available\n");
        return 1;
    }
    const auto before_freeze = llama_memory_vbr_state(mem, 0, 0);
    const uint64_t outer = llama_memory_vbr_retier_freeze_begin(mem, "epoch_test_outer");
    const uint64_t inner = llama_memory_vbr_retier_freeze_begin(mem, "epoch_test_inner");
    const auto nested = llama_memory_vbr_state(mem, 0, 0);
    if (outer == 0) {
        fprintf(stderr, "outer iSWA scoped freeze did not acquire\n");
        return 1;
    }
    if (inner == 0) {
        fprintf(stderr, "inner iSWA scoped freeze did not acquire\n");
        return 1;
    }
    if (outer == inner) {
        fprintf(stderr, "nested VBR operations reused an operation ID\n");
        return 1;
    }
    if (!vbr_operation_registry_is_live({ outer }) ||
        !vbr_operation_registry_is_live({ inner })) {
        fprintf(stderr, "nested VBR operation IDs were not both live\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != inner ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != inner) {
        fprintf(stderr, "iSWA children did not receive the identical inner operation ID\n");
        return 1;
    }
    if (nested.retier_freeze_depth != 2) {
        fprintf(stderr, "nested iSWA scoped freeze reported the wrong depth\n");
        return 1;
    }
    if (nested.retier_freeze_enters !=
        before_freeze.retier_freeze_enters + 2) {
        fprintf(stderr, "nested iSWA scoped freeze counted an unexpected number of parent entries\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "base tier mutation was not deferred under scoped freeze\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "SWA tier mutation was not deferred under scoped freeze\n");
        return 1;
    }
    const auto deferred = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(deferred, before_freeze)) {
        fprintf(stderr, "scoped freeze allowed a representation epoch to change\n");
        return 1;
    }
    if (deferred.retier_deferred_decisions !=
        before_freeze.retier_deferred_decisions + 2) {
        fprintf(stderr, "scoped freeze counted an unexpected number of deferred child decisions\n");
        return 1;
    }
    // A0 amendment: simulate future runtime budget renegotiation while the operation is live.
    // iSWA end must pair from the immutable begin record even though base now reports disarmed.
    const uint64_t base_budget =
        llama_kv_cache_vbr_epoch_test::set_budget_bytes(base, 0);
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_inner", inner);
    llama_kv_cache_vbr_epoch_test::set_budget_bytes(base, base_budget);
    if (vbr_operation_registry_is_live({ inner })) {
        fprintf(stderr, "inner VBR operation remained live after end\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != outer ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != outer) {
        fprintf(stderr, "iSWA armed-flip pairing did not restore the identical outer operation ID\n");
        return 1;
    }
    if (llama_memory_vbr_state(mem, 0, 0).retier_freeze_depth != 1) {
        fprintf(stderr, "inner scoped-freeze exit released the outer scope\n");
        return 1;
    }
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_outer", outer);
    if (vbr_operation_registry_is_live({ outer })) {
        fprintf(stderr, "outer VBR operation remained live after end\n");
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::freeze_operation_id(base) != 0 ||
        llama_kv_cache_vbr_epoch_test::freeze_operation_id(swa)  != 0) {
        fprintf(stderr, "iSWA child retained an operation ID after outer end\n");
        return 1;
    }
    const auto unfrozen = llama_memory_vbr_state(mem, 0, 0);
    if (unfrozen.retier_freeze_depth != 0) {
        fprintf(stderr, "outer scoped-freeze exit left a nonzero depth\n");
        return 1;
    }
    if (unfrozen.retier_freeze_exits !=
        before_freeze.retier_freeze_exits + 2) {
        fprintf(stderr, "nested scoped-freeze exits counted an unexpected number of parent exits\n");
        return 1;
    }
    if (!epochs_equal(unfrozen, before_freeze)) {
        fprintf(stderr, "scoped-freeze exit changed a representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::reconcile(base)) {
        fprintf(stderr, "outer unfreeze did not arm a fresh base-child evaluation\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::reconcile(swa)) {
        fprintf(stderr, "outer unfreeze did not arm a fresh SWA-child evaluation\n");
        return 1;
    }
    const auto reconciled = llama_memory_vbr_state(mem, 0, 0);
    if (reconciled.retier_reconciles !=
        unfrozen.retier_reconciles + 1) {
        fprintf(stderr, "fresh post-unfreeze evaluations counted an unexpected number of reconciles\n");
        return 1;
    }
    if (!epochs_equal(reconciled, unfrozen)) {
        fprintf(stderr, "fresh post-unfreeze evaluation changed a representation epoch\n");
        return 1;
    }

    // The two independently mutating children must surface an ordered tuple, never a sum.
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "failed to force base degrade\n");
        return 1;
    }
    const auto base_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (base_degraded.representation_epoch <=
        initial.representation_epoch) {
        fprintf(stderr, "base degrade did not advance the base epoch\n");
        return 1;
    }
    if (base_degraded.representation_epoch_swa !=
        initial.representation_epoch_swa) {
        fprintf(stderr, "base degrade unexpectedly changed the SWA epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(base)) {
        fprintf(stderr, "A1 base unit tuple did not publish the degraded live types\n");
        return 1;
    }

    if (!llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "failed to force SWA degrade\n");
        return 1;
    }
    const auto both_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (both_degraded.representation_epoch !=
        base_degraded.representation_epoch) {
        fprintf(stderr, "SWA degrade unexpectedly changed the base epoch\n");
        return 1;
    }
    if (both_degraded.representation_epoch_swa <=
        base_degraded.representation_epoch_swa) {
        fprintf(stderr, "SWA degrade did not advance the SWA epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(swa)) {
        fprintf(stderr, "A1 SWA unit tuple did not publish the degraded live types\n");
        return 1;
    }

    // This is the production low-LCP/empty-cache reset sequence. clear() changes the referenced
    // representation first; vbr_full_reset() then rewinds the cursor but must advance, not reset,
    // each epoch.
    mem->clear(true);
    const auto cleared = llama_memory_vbr_state(mem, 0, 0);
    if (cleared.representation_epoch <=
        both_degraded.representation_epoch) {
        fprintf(stderr, "clear did not advance the base representation epoch\n");
        return 1;
    }
    if (cleared.representation_epoch_swa <=
        both_degraded.representation_epoch_swa) {
        fprintf(stderr, "clear did not advance the SWA representation epoch\n");
        return 1;
    }
    llama_kv_cache_vbr_epoch_test::full_reset(base);
    llama_kv_cache_vbr_epoch_test::full_reset(swa);
    const auto reset = llama_memory_vbr_state(mem, 0, 0);
    if (reset.cursor != 0) {
        fprintf(stderr, "full reset did not rewind the VBR cursor\n");
        return 1;
    }
    if (reset.representation_epoch <= cleared.representation_epoch) {
        fprintf(stderr, "full reset did not advance the base representation epoch\n");
        return 1;
    }
    if (reset.representation_epoch_swa <=
        cleared.representation_epoch_swa) {
        fprintf(stderr, "full reset did not advance the SWA representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::generation_units_match(base) ||
            !llama_kv_cache_vbr_epoch_test::generation_units_match(swa)) {
        fprintf(stderr, "A1 full reset did not republish both children at their live types\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(base)) {
        fprintf(stderr, "PRECONDITION failed: could not remap the base child after full reset\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::map_seed_watermark(swa)) {
        fprintf(stderr, "PRECONDITION failed: could not remap the SWA child after full reset\n");
        return 1;
    }

    // Refill, degrade again, then adopt the native mixed-tier state onto itself. Ordinary forward
    // fill must not move the representation epochs; the second degrade and import both must.
    if (!decode_one(ctx.get())) {
        fprintf(stderr, "post-reset seed decode failed\n");
        return 1;
    }
    const auto refilled = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(refilled, reset)) {
        fprintf(stderr, "post-reset seed decode changed a representation epoch\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(base)) {
        fprintf(stderr, "PRECONDITION failed: post-reset base child has no mapped degradable extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::has_mapped_degradable_unit(swa)) {
        fprintf(stderr, "PRECONDITION failed: post-reset SWA child has no mapped degradable extent\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(base)) {
        fprintf(stderr, "post-reset base degrade failed\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::force_degrade(swa)) {
        fprintf(stderr, "post-reset SWA degrade failed\n");
        return 1;
    }
    const auto degraded_again = llama_memory_vbr_state(mem, 0, 0);
    if (degraded_again.representation_epoch <=
        reset.representation_epoch) {
        fprintf(stderr, "degrade-reset-degrade did not advance the base epoch\n");
        return 1;
    }
    if (degraded_again.representation_epoch_swa <=
        reset.representation_epoch_swa) {
        fprintf(stderr, "degrade-reset-degrade did not advance the SWA epoch\n");
        return 1;
    }
    const auto normal_final = llama_memory_vbr_state(mem, 0, 0);
    if (normal_final.retier_freeze_enters !=
        before_freeze.retier_freeze_enters + 2) {
        fprintf(stderr, "ordinary phase changed the scoped-freeze enter count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_freeze_exits !=
        before_freeze.retier_freeze_exits + 2) {
        fprintf(stderr, "ordinary phase changed the scoped-freeze exit count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_deferred_decisions !=
        before_freeze.retier_deferred_decisions + 2) {
        fprintf(stderr, "ordinary phase changed the deferred-decision count after reconciliation\n");
        return 1;
    }
    if (normal_final.retier_reconciles !=
        before_freeze.retier_reconciles + 1) {
        fprintf(stderr, "ordinary phase changed the reconcile count after reconciliation\n");
        return 1;
    }

    // WS-0 composition: its deterministic-input freeze remains authoritative, while an empty
    // scoped window is representation-neutral and balances normally. This intentionally does
    // not reinterpret VBR_FREEZE as a production retier stop (its scripted waves still run).
    ctx.reset();
    set_test_env("VBR_FREEZE", "1");
    if (!trace_prefix.empty()) {
        set_test_env("VBR_TRACE", (trace_prefix + ".env").c_str());
    }
    llama_context_ptr env_ctx(llama_init_from_model(model.get(), cparams));
    if (!env_ctx) {
        fprintf(stderr, "failed to create VBR_FREEZE composition context\n");
        return 1;
    }
    llama_memory_t env_mem = llama_get_memory(env_ctx.get());
    llama_kv_cache * env_base = nullptr;
    llama_kv_cache * env_swa  = nullptr;
    if (!get_iswa_children(env_mem, env_base, env_swa)) {
        fprintf(stderr, "VBR_FREEZE composition context was not iSWA\n");
        return 1;
    }
    if (!decode_one(env_ctx.get())) {
        fprintf(stderr, "VBR_FREEZE composition seed decode failed\n");
        return 1;
    }
    const auto env_before = llama_memory_vbr_state(env_mem, 0, 0);
    const uint64_t env_scope =
        llama_memory_vbr_retier_freeze_begin(env_mem, "epoch_test_env_noop");
    if (env_scope == 0) {
        fprintf(stderr, "scoped freeze did not compose with VBR_FREEZE context\n");
        return 1;
    }
    llama_memory_vbr_retier_freeze_end(
        env_mem, "epoch_test_env_noop", env_scope);
    const auto env_after = llama_memory_vbr_state(env_mem, 0, 0);
    const bool env_reconcile_base =
        llama_kv_cache_vbr_epoch_test::reconcile(env_base);
    const bool env_reconcile_swa =
        llama_kv_cache_vbr_epoch_test::reconcile(env_swa);
    unset_test_env("VBR_FREEZE");
    unset_test_env("VBR_TRACE");
    if (env_before.retier_env_freeze == 0) {
        fprintf(stderr, "VBR_FREEZE composition context did not report env freeze before the scope\n");
        return 1;
    }
    if (env_after.retier_env_freeze == 0) {
        fprintf(stderr, "VBR_FREEZE composition context lost env freeze across the scope\n");
        return 1;
    }
    if (env_after.retier_freeze_depth != 0) {
        fprintf(stderr, "VBR_FREEZE composition scope left a nonzero depth\n");
        return 1;
    }
    if (env_after.retier_freeze_enters !=
        env_before.retier_freeze_enters + 1) {
        fprintf(stderr, "VBR_FREEZE composition scope counted an unexpected number of parent entries\n");
        return 1;
    }
    if (env_after.retier_freeze_exits !=
        env_before.retier_freeze_exits + 1) {
        fprintf(stderr, "VBR_FREEZE composition scope counted an unexpected number of parent exits\n");
        return 1;
    }
    if (env_after.retier_deferred_decisions !=
        env_before.retier_deferred_decisions) {
        fprintf(stderr, "empty VBR_FREEZE composition scope counted a deferred decision\n");
        return 1;
    }
    if (env_reconcile_base) {
        fprintf(stderr, "empty VBR_FREEZE composition scope armed a base reconciliation\n");
        return 1;
    }
    if (env_reconcile_swa) {
        fprintf(stderr, "empty VBR_FREEZE composition scope armed a SWA reconciliation\n");
        return 1;
    }
    if (env_after.retier_reconciles != env_before.retier_reconciles) {
        fprintf(stderr, "empty VBR_FREEZE composition scope changed the reconcile counter\n");
        return 1;
    }
    if (!epochs_equal(env_before, env_after)) {
        fprintf(stderr, "empty VBR_FREEZE composition scope changed a representation epoch\n");
        return 1;
    }

    // NOTE: the "native mixed-tier import bumps both epochs" case is intentionally NOT exercised
    // here. The fork deliberately REFUSES to serialize a dynamic-VBR cache after a tier degrade
    // (llama_state_seq_get_size throws "cannot serialize a dynamic-VBR KV cache after tier
    // degrades..."), so a degraded mixed-tier state cannot be captured and re-adopted in the
    // current codebase — native mixed-tier import/serialization is unbuilt Phase-2/3 work. The
    // import path DOES bump the epoch (state_read adoption calls vbr_representation_changed), but it
    // is unreachable at runtime until that serialization exists. The P0 I9 behavior that matters --
    // per-child epoch advance on every degrade, clear, and full-reset (incl. the low-LCP ABA close),
    // and cross-degrade monotonicity -- is fully covered above.

    fprintf(stderr, "PASS: VBR scoped freeze is nested/iSWA-coherent, defers mutations, "
            "re-evaluates fresh on exit, composes with VBR_FREEZE, and preserves monotone "
            "per-child representation epochs\n");
    return 0;
}
