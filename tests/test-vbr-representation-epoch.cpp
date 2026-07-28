#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-io.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-vbr-checkpoint.h"
#include "llama-vbr-checkpoint-compose.inc"
#include "llama-vbr-generation-oracle.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
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

    static vbr_generation_tracker * tracker_mut(llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_mut();
    }

    static const vbr_generation_tracker * tracker_get(const llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_get();
    }

    struct serializer_count_complete {};

    class serializer_positions_writer : public llama_io_write_i {
    public:
        explicit serializer_positions_writer(bool has_ext) : has_ext(has_ext) {}

        void write(const void * src, size_t size) override {
            if (stage == 0) {
                if (size == sizeof(uint32_t)) {
                    std::memcpy(&streams, src, size);
                }
                valid = size == sizeof(uint32_t) && streams == 1;
                stage = 1;
                return;
            }
            if (stage == 1) {
                if (size == sizeof(uint32_t)) {
                    std::memcpy(&remaining, src, size);
                }
                valid = valid && size == sizeof(uint32_t);
                positions.reserve(remaining);
                if (remaining == 0) {
                    throw serializer_count_complete{};
                }
                stage = 2;
                return;
            }
            if (stage == 2) {
                llama_pos position = -1;
                if (size == sizeof(position)) {
                    std::memcpy(&position, src, size);
                }
                valid = valid && size == sizeof(position) && position >= 0;
                positions.push_back(position);
                stage = 3;
                return;
            }
            if (stage == 3) {
                uint32_t n_seq = 0;
                if (size == sizeof(n_seq)) {
                    std::memcpy(&n_seq, src, size);
                }
                valid = valid && size == sizeof(n_seq) && n_seq == 1;
                stage = has_ext ? 4 : 5;
                return;
            }
            if (stage == 4) {
                valid = valid && size == sizeof(llama_kv_cell_ext);
                stage = 5;
                return;
            }
            if (stage == 5) {
                llama_seq_id seq = -1;
                if (size == sizeof(seq)) {
                    std::memcpy(&seq, src, size);
                }
                valid = valid && size == sizeof(seq) && seq == 0 && remaining > 0;
                if (--remaining == 0) {
                    throw serializer_count_complete{};
                }
                stage = 2;
                return;
            }
            valid = false;
            throw serializer_count_complete{};
        }

        void write_tensor(ggml_tensor *, size_t, size_t) override {
            valid = false;
            throw serializer_count_complete{};
        }

        size_t n_bytes() override {
            return 0;
        }

        const bool             has_ext;
        uint32_t               streams   = 0;
        uint32_t               remaining = 0;
        uint32_t               stage     = 0;
        bool                   valid     = true;
        std::vector<llama_pos> positions;
    };

    static bool serializer_positions(
            const llama_kv_cache * kv,
            llama_seq_id seq_id,
            std::vector<llama_pos> & positions) {
        serializer_positions_writer writer(kv->hparams.n_pos_per_embd() > 1);
        try {
            kv->state_write(writer, seq_id);
        } catch (const serializer_count_complete &) {
            positions = std::move(writer.positions);
            return writer.valid && writer.remaining == 0;
        } catch (...) {
            return false;
        }
        return false;
    }

    // C2 rows (b)/(c): a REAL provenance-bearing root scope with a deliberately narrow
    // manifest (seq 0, positions [0,2)). Returned open so nested production mutations join
    // it; closing WITHOUT succeed() is the production FAILED close (autorecords recovery).
    static void * open_narrow_trim_scope(llama_kv_cache * kv) {
        auto * tracker = kv->vbr_generation_tracker_mut();
        if (tracker == nullptr) {
            return nullptr;
        }
        const auto pool = tracker->pool_identity();
        vbr_operation_binding binding;
        binding.kind        = vbr_operation_kind::sequence_edit;
        binding.child_phase = vbr_operation_phase::mutate;
        binding.n_targets   = 2;
        binding.targets[0]  = vbr_make_target(vbr_operation_kind::sequence_edit,
                                              vbr_operation_class::explicit_destructive_trim,
                                              pool.hi, pool.lo, 0, 0, 0, 2);
        // real nested seq_rm authenticates as membership-only state_api
        // (llama-kv-cache.cpp seq_rm scope); authorize that class on the SAME narrow range so
        // its refusal happens at per-stamp range selection (positions >= 2), never at
        // begin-time class authentication
        binding.targets[1]  = vbr_make_target(vbr_operation_kind::sequence_edit,
                                              vbr_operation_class::state_api,
                                              pool.hi, pool.lo, 0, 0, 0, 2);
        auto * op = new llama_kv_cache::vbr_mutation_op(kv, binding, /*provenance_bearing=*/true);
        if (!op->active()) {
            delete op;
            return nullptr;
        }
        return op;
    }

    static void close_scope_without_success(void * scope) {
        delete static_cast<llama_kv_cache::vbr_mutation_op *>(scope);
    }

    // C2 row (b): joined poison through the production citation/stamp path — the event cites
    // the open root scope and the stamped pre-mutation position (100) is outside the
    // manifest, so vbr_stamp() refuses it and poisons the root.
    static bool stamp_outside_manifest(llama_kv_cache * kv, void * scope) {
        auto * op = static_cast<llama_kv_cache::vbr_mutation_op *>(scope);
        auto event = kv->vbr_generation_begin(
                vbr_mutation_registrant::seq_rm,
                vbr_operation_class::explicit_destructive_trim,
                0,
                vbr_generation_stamp_kind::membership,
                /*destructive=*/true);
        if (!event) {
            return false;
        }
        kv->vbr_stamp(*op, event, /*cell=*/3, /*membership_seq=*/0, /*pre_mutation_pos=*/100);
        return event.finish();
    }

    // C2 row (e): fence-race seam — after decode SUBMISSION and before the synchronize
    // fence, one in-flight operation's per-target evidence goes stale via a slab reset. The
    // fence's commit then fails through the REAL terminal path (latch + fail handles +
    // FAILED close/report).
    static bool inject_stale_submitted_extent(llama_kv_cache * kv) {
        auto * tracker = kv->vbr_generation_tracker_mut();
        if (tracker == nullptr) {
            return false;
        }
        auto & store  = tracker->extent_store();
        auto   handle = store.reserve(vbr_mutation_family::trim,
                                      vbr_operation_class::explicit_destructive_trim, 0, 0, 0, 1);
        if (!handle || !store.submit(handle)) {
            return false;
        }
        store.reset_all();  // slab reset: the submitted handle is obsolete at the fence
        if (!kv->vbr_awaiting_commit_.empty()) {
            kv->vbr_awaiting_commit_.front().extents[0] = handle;
            return true;
        }
        if (!kv->vbr_pending_decode_ops_.empty()) {
            kv->vbr_pending_decode_ops_.front().extents[0] = handle;
            return true;
        }
        return false;
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
    std::vector<llama_pos> positions;
};

static bool a1_cell_has_seq(const void * context, uint32_t, uint32_t cell, llama_seq_id seq_id) {
    const auto * fixture = static_cast<const a1_membership_fixture *>(context);
    return seq_id == 0 && cell < fixture->present.size() && fixture->present[cell] != 0;
}

static llama_pos a1_cell_pos(const void * context, uint32_t, uint32_t cell) {
    const auto * fixture = static_cast<const a1_membership_fixture *>(context);
    if (cell >= fixture->present.size() || fixture->present[cell] == 0) {
        return -1;
    }
    return cell < fixture->positions.size() ? fixture->positions[cell] : (llama_pos) cell;
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
    stream.cell_pos               = a1_cell_pos;

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

    std::vector<vbr_generation_oracle_cell> canonical = {
        {10,  10,  true,  true, false, {1, 2, 3}},
        {20,  20,  false, true, false, {1, 2, 3}},
        {300, 300, true,  true, false, {4, 5}},
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
    canonical[0].dependency_bytes[0] = 9;
    audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    if (!audit.set_equal || audit.bytes_equal) {
        fprintf(stderr, "A1 byte oracle missed a covered-byte mutation\n");
        return false;
    }
    canonical[0].dependency_bytes[0] = 1;
    auto undercovered = stream;
    undercovered.pages.erase(undercovered.pages.begin());
    audit = vbr_generation_oracle_audit(400, canonical, baseline, undercovered);
    if (audit.set_equal) {
        fprintf(stderr, "A1 independent oracle repeated production mask under-coverage\n");
        return false;
    }
    auto incomplete = canonical;
    incomplete[0].dependency_bytes.clear();
    audit = vbr_generation_oracle_audit(400, incomplete, baseline, stream);
    if (audit.complete || audit.bytes_equal) {
        fprintf(stderr, "A1 byte oracle accepted an incomplete canonical observation\n");
        return false;
    }
    set_test_env("VBR_GENERATION_ORACLE_INJECT", "set");
    audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    vbr_generation_oracle_inject(audit);
    if (audit.set_equal || !audit.bytes_equal) {
        fprintf(stderr, "A1 oracle set-mismatch injection was not isolated\n");
        return false;
    }
    set_test_env("VBR_GENERATION_ORACLE_INJECT", "bytes");
    audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    vbr_generation_oracle_inject(audit);
    if (!audit.set_equal || audit.bytes_equal) {
        fprintf(stderr, "A1 oracle byte-mismatch injection was not isolated\n");
        return false;
    }
    set_test_env("VBR_GENERATION_ORACLE_INJECT", "unavailable");
    audit = vbr_generation_oracle_audit(400, canonical, baseline, stream);
    vbr_generation_oracle_inject(audit);
    if (audit.complete) {
        fprintf(stderr, "A1 oracle unavailable injection remained complete\n");
        return false;
    }
    unset_test_env("VBR_GENERATION_ORACLE_INJECT");
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

// --- C2 (commit 2): composite bridge assembly, identity digest, §6.2 policy, oracle rows ------

struct c2_capture_fixture {
    const vbr_generation_tracker *           tracker = nullptr;
    const vbr_checkpoint_generation_stream * stream  = nullptr;
};

static bool c2_capture_cb(void * ctx, uint32_t child_id, vbr_checkpoint_generation_controller & out) {
    const auto * fixture = static_cast<const c2_capture_fixture *>(ctx);
    return vbr_generation_capture_controller(
            *fixture->tracker, child_id, checkpoint_child_dependency_mode::live_guarded,
            { *fixture->stream }, out);
}

static bool run_c2_cpu_tests() {
    // armed-child fixture: one tracker with two stamped dependency cells and a captured stream
    vbr_generation_tracker tracker(1, 512, 1);
    if (!tracker.active() || !tracker.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full)) {
        fprintf(stderr, "C2 tracker did not initialize\n");
        return false;
    }
    test_operation op(vbr_operation_kind::decode, tracker.pool_identity(), -1,
                      0, std::numeric_limits<llama_pos>::max(),
                      vbr_operation_class::ordinary_decode);
    auto append = tracker.begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode,
            0,
            vbr_generation_stamp_kind::dependency,
            op.id());
    if (!append || !tracker.stamp_cell(append, 10, 0) ||
            !tracker.stamp_cell(append, 300, 0) || !append.finish()) {
        fprintf(stderr, "C2 fixture event did not publish atomically\n");
        return false;
    }
    vbr_checkpoint_generation_stream stream;
    if (!vbr_generation_capture_stream(tracker, 0, 0, 400, { 10, 300 }, stream)) {
        fprintf(stderr, "C2 fixture stream capture failed\n");
        return false;
    }
    c2_capture_fixture fixture{ &tracker, &stream };

    vbr_checkpoint_frontier_fields frontier;
    const std::string exec_id    = "c2-exec-identity";
    const std::string adapter_id = "c2-adapter-identity";
    const std::string media_id   = "c2-media-identity";
    frontier.execution_identity          = exec_id.c_str();
    frontier.execution_identity_len      = exec_id.size();
    frontier.adapter_config_identity     = adapter_id.c_str();
    frontier.adapter_config_identity_len = adapter_id.size();
    frontier.media_content_identity      = media_id.c_str();
    frontier.media_content_identity_len  = media_id.size();
    frontier.sequence_epoch = 7;
    frontier.token_count    = 400;
    frontier.next_position  = 400;

    vbr_checkpoint_child_input armed_lg;
    armed_lg.live_guarded = true;
    armed_lg.armed        = true;
    armed_lg.pool_uuid    = tracker.pool_identity();
    armed_lg.capture      = c2_capture_cb;
    armed_lg.capture_ctx  = &fixture;

    // §11.1 row 16: an unarmed live_guarded child WITH nonempty live coverage makes the whole
    // matrix unavailable — never server-invented coverage.
    {
        vbr_checkpoint_child_input unarmed_covered;
        unarmed_covered.live_guarded = true;
        unarmed_covered.live_covered = true;
        vbr_checkpoint_generation_record record;
        const auto reason = vbr_checkpoint_compose({ armed_lg, unarmed_covered }, frontier, record);
        if (reason != vbr_checkpoint_capture_reason::unarmed_live_covered ||
                record.status == vbr_checkpoint_generation_status::complete) {
            fprintf(stderr, "C2 row 16: unarmed live-covered child did not refuse the capture\n");
            return false;
        }
    }

    // §11.1 row 17 + composite shape: an unarmed payload_complete child is a vacuous row inside
    // a COMPLETE record; child ids are traversal ordinals.
    vbr_checkpoint_generation_record record;
    {
        vbr_checkpoint_child_input unarmed_pc;
        unarmed_pc.live_guarded = false;
        const auto reason = vbr_checkpoint_compose({ armed_lg, unarmed_pc }, frontier, record);
        if (reason != vbr_checkpoint_capture_reason::ok ||
                record.status != vbr_checkpoint_generation_status::complete ||
                record.controllers.size() != 2 ||
                record.controllers[0].child_id != 0 ||
                record.controllers[0].dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
                record.controllers[1].child_id != 1 ||
                record.controllers[1].dependency_mode != checkpoint_child_dependency_mode::payload_complete ||
                !record.controllers[1].streams.empty() ||
                !record.controllers[1].units.empty() ||
                record.controllers[1].pool_uuid.hi != 0 || record.controllers[1].pool_uuid.lo != 0) {
            fprintf(stderr, "C2 row 17: unarmed payload_complete child was not a vacuous row\n");
            return false;
        }
    }

    // fully unarmed memory: not applicable, never a failure
    {
        vbr_checkpoint_child_input unarmed_lg;
        unarmed_lg.live_guarded = true;
        vbr_checkpoint_child_input unarmed_pc;
        vbr_checkpoint_generation_record unused;
        if (vbr_checkpoint_compose({ unarmed_lg, unarmed_pc }, frontier, unused) !=
                vbr_checkpoint_capture_reason::not_applicable) {
            fprintf(stderr, "C2 fully-unarmed composite was not classified not_applicable\n");
            return false;
        }
    }

    // F5: an armed payload_complete child must carry its exact nonzero pool identity
    {
        vbr_checkpoint_child_input armed_pc;
        armed_pc.armed = true;  // pool_uuid deliberately zero
        vbr_checkpoint_generation_record unused;
        if (vbr_checkpoint_compose({ armed_lg, armed_pc }, frontier, unused) !=
                vbr_checkpoint_capture_reason::child_capture_failed) {
            fprintf(stderr, "C2 armed payload_complete child with zero pool identity was accepted\n");
            return false;
        }
        armed_pc.pool_uuid = tracker.pool_identity();
        if (vbr_checkpoint_compose({ armed_lg, armed_pc }, frontier, unused) !=
                    vbr_checkpoint_capture_reason::ok ||
                unused.controllers[1].pool_uuid != tracker.pool_identity()) {
            fprintf(stderr, "C2 armed payload_complete child did not record its pool identity\n");
            return false;
        }
    }

    // F2: canonical digest — shared helper reproduces the record digest; identity and policy
    // envelope changes both move it
    {
        std::vector<vbr_checkpoint_child_policy> policy;
        for (const auto & controller : record.controllers) {
            policy.push_back({ controller.child_id, controller.dependency_mode, controller.pool_uuid });
        }
        if (vbr_checkpoint_identity_digest(frontier, policy) != record.identity_policy_order_digest) {
            fprintf(stderr, "C2 digest helper diverged from the composed record digest\n");
            return false;
        }
        auto other_frontier = frontier;
        const std::string other_adapter = "c2-adapter-identity-b";
        other_frontier.adapter_config_identity     = other_adapter.c_str();
        other_frontier.adapter_config_identity_len = other_adapter.size();
        if (vbr_checkpoint_identity_digest(other_frontier, policy) == record.identity_policy_order_digest) {
            fprintf(stderr, "C2 digest ignored an adapter identity change\n");
            return false;
        }
        auto other_policy = policy;
        other_policy[1].mode = checkpoint_child_dependency_mode::live_guarded;
        if (vbr_checkpoint_identity_digest(frontier, other_policy) == record.identity_policy_order_digest) {
            fprintf(stderr, "C2 digest ignored a dependency-mode change\n");
            return false;
        }
    }

    // the composed two-child record round-trips through the sole evaluator
    a1_membership_fixture membership;
    membership.present.resize(512);
    membership.present[10]  = 1;
    membership.present[300] = 1;
    vbr_generation_live_view live;
    {
        live.legacy_eligible = true;
        live.identity_policy_order_digest = record.identity_policy_order_digest;

        vbr_generation_live_stream_view live_stream;
        live_stream.stream_index           = 0;
        live_stream.dependency_seq_id      = 0;
        live_stream.computation_frontier   = 400;
        live_stream.exact_dependency_count = 2;
        live_stream.membership_context     = &membership;
        live_stream.cell_has_seq           = a1_cell_has_seq;
        live_stream.cell_pos               = a1_cell_pos;

        vbr_generation_live_controller_view live_lg;
        live_lg.child_id        = 0;
        live_lg.dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        live_lg.tracker         = &tracker;
        live_lg.streams.push_back(live_stream);

        vbr_generation_live_controller_view live_pc;
        live_pc.child_id        = 1;
        live_pc.dependency_mode = checkpoint_child_dependency_mode::payload_complete;

        live.controllers.push_back(std::move(live_lg));
        live.controllers.push_back(std::move(live_pc));
    }
    auto result = checkpoint_vbr_eligibility(record, live);
    if (result.category != vbr_checkpoint_eligibility_category::strict_accept) {
        fprintf(stderr, "C2 composed two-child record was not strict-accepted (reason %d)\n",
                (int) result.reason);
        return false;
    }

    // Pin 6 mixed-child applicability: an unarmed live_guarded child with no covered cells is
    // a vacuous row in both capture and evaluation, not a demand for a nonexistent tracker.
    {
        vbr_checkpoint_child_input unarmed_lg;
        unarmed_lg.live_guarded = true;
        vbr_checkpoint_generation_record mixed_record;
        if (vbr_checkpoint_compose({ armed_lg, unarmed_lg }, frontier, mixed_record) !=
                vbr_checkpoint_capture_reason::ok) {
            fprintf(stderr, "C3 mixed-child vacuous live_guarded capture failed\n");
            return false;
        }
        auto mixed_live = live;
        mixed_live.controllers[1].dependency_mode = checkpoint_child_dependency_mode::live_guarded;
        std::vector<vbr_checkpoint_child_policy> mixed_policy;
        for (const auto & controller : mixed_record.controllers) {
            mixed_policy.push_back(
                    { controller.child_id, controller.dependency_mode, controller.pool_uuid });
        }
        mixed_live.identity_policy_order_digest =
            vbr_checkpoint_identity_digest(frontier, mixed_policy);
        if (!checkpoint_vbr_eligibility(mixed_record, mixed_live).strict) {
            fprintf(stderr, "C3 mixed-child vacuous live_guarded evaluation failed\n");
            return false;
        }
    }

    // payload_complete controllers with nonzero streams reject
    {
        auto malformed = record;
        malformed.controllers[1].streams.push_back(stream);
        if (checkpoint_vbr_eligibility(malformed, live).reason !=
                vbr_checkpoint_eligibility_reason::stream_shape) {
            fprintf(stderr, "C2 payload_complete controller with streams was accepted\n");
            return false;
        }
    }

    // non-increasing child ids reject (child_order)
    {
        auto malformed   = record;
        auto ordered_live = live;
        malformed.controllers[0].child_id    = 1;
        ordered_live.controllers[0].child_id = 1;
        if (checkpoint_vbr_eligibility(malformed, ordered_live).reason !=
                vbr_checkpoint_eligibility_reason::child_order) {
            fprintf(stderr, "C2 non-increasing child order was accepted\n");
            return false;
        }
    }

    // §6.2 sampling policy: pure, deterministic, crossing/forced always audit
    {
        std::array<uint8_t, 32> digest = {};
        digest[0] = 0;
        if (!vbr_generation_oracle_audit_due(false, digest, false) ||
                !vbr_generation_oracle_audit_due(true, digest, false)) {
            fprintf(stderr, "C2 §6.2 policy missed a mandatory audit\n");
            return false;
        }
        digest[0] = 1;
        if (vbr_generation_oracle_audit_due(false, digest, false) ||
                vbr_generation_oracle_audit_due(false, digest, false) !=
                        vbr_generation_oracle_audit_due(false, digest, false)) {
            fprintf(stderr, "C2 §6.2 append-only sample was not deterministic\n");
            return false;
        }
        if (!vbr_generation_oracle_audit_due(true, digest, false) ||
                !vbr_generation_oracle_audit_due(false, digest, true)) {
            fprintf(stderr, "C2 §6.2 crossing/forced audit did not fire\n");
            return false;
        }
        unset_test_env("VBR_GENERATION_FORCE_AUDIT");
        if (vbr_generation_oracle_audit_forced()) {
            fprintf(stderr, "C2 forced-audit override was not default-off\n");
            return false;
        }
        set_test_env("VBR_GENERATION_FORCE_AUDIT", "1");
        if (!vbr_generation_oracle_audit_forced()) {
            fprintf(stderr, "C2 forced-audit env override did not engage\n");
            return false;
        }
        unset_test_env("VBR_GENERATION_FORCE_AUDIT");
    }

    // F6: equal-cardinality wrong-cell injection — the independent oracle rejects a production
    // mask that covers the right COUNT of cells but a wrong member
    {
        std::vector<vbr_generation_oracle_cell> canonical = {
            { 10,  10,  true,  true, false, { 1 } },
            { 20,  20,  false, true, false, {} },
            { 300, 300, true,  true, false, { 2 } },
        };
        set_test_env("VBR_GENERATION_ORACLE", "1");
        const auto baseline = vbr_generation_oracle_capture(400, canonical);

        vbr_checkpoint_generation_stream wrong_cell;
        wrong_cell.stream_index              = 0;
        wrong_cell.dependency_seq_id         = 0;
        wrong_cell.computation_frontier      = 400;
        wrong_cell.captured_dependency_count = 2;
        vbr_generation_page_ref page0;
        page0.page_index = 0;
        page0.covered_mask[10 / 64] |= uint64_t(1) << (10 % 64);
        vbr_generation_page_ref page1;
        page1.page_index = 1;
        const uint32_t wrong_offset = 301 - VBR_GENERATION_PAGE_CELLS;  // covers 301, not 300
        page1.covered_mask[wrong_offset / 64] |= uint64_t(1) << (wrong_offset % 64);
        wrong_cell.pages = { page0, page1 };

        const auto audit = vbr_generation_oracle_audit(400, canonical, baseline, wrong_cell);
        if (!audit.enabled || audit.set_equal ||
                audit.independent_count != wrong_cell.captured_dependency_count) {
            fprintf(stderr, "C2 equal-cardinality wrong-cell production mask was not rejected\n");
            return false;
        }
        unset_test_env("VBR_GENERATION_ORACLE");
    }

    fprintf(stderr, "C2 composite bridge/digest/policy CPU coverage PASS\n");
    return true;
}

// --- C2-P9 rung-1 rows (a)-(e): queued commit-1 GPU evidence rows on the live armed fixture --
// Each row builds its own context so latched/poisoned state never leaks across rows. Row
// shapes were review-approved before the dorei gate; threshold/geometry constants (window
// coverage margins, drain-decode budget) may be tuned on the gate box without changing the
// asserted properties.

static bool c2_gpu_context(llama_model * model, uint32_t n_ctx, uint32_t n_batch,
                           uint32_t n_seq_max, llama_context_ptr & out, bool swa_full = false) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                 = n_ctx;
    cparams.n_batch               = n_batch;
    cparams.n_ubatch              = 32;
    cparams.n_seq_max             = n_seq_max;
    cparams.n_threads             = 2;
    cparams.n_threads_batch       = 2;
    cparams.kv_unified            = true;
    cparams.swa_full              = swa_full;
    cparams.type_k                = GGML_TYPE_F16;
    cparams.type_v                = GGML_TYPE_F16;
    cparams.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.vbr_dynamic           = true;
    cparams.vbr_budget_explicit   = true;
    // generous budget: these rows exercise evidence/recovery, not the degrade ladder
    cparams.vbr_vram_budget_bytes = 1024ull * 1024 * 1024;
    out.reset(llama_init_from_model(model, cparams));
    return static_cast<bool>(out);
}

static bool c2_gpu_children(llama_memory_t mem, llama_kv_cache *& base, llama_kv_cache *& swa) {
    if (!get_iswa_children(mem, base, swa)) {
        return false;
    }
    return llama_kv_cache_vbr_epoch_test::active(base) && llama_kv_cache_vbr_epoch_test::active(swa);
}

// batched decode of [pos_begin, pos_end) for one sequence, fenced (decode is asynchronous)
static bool c2_gpu_decode_range(llama_context * ctx, llama_seq_id seq,
                                llama_pos pos_begin, llama_pos pos_end, int32_t n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (llama_pos pos = pos_begin; pos < pos_end; ) {
        common_batch_clear(batch);
        const llama_pos stop = std::min<llama_pos>(pos_end, pos + n_batch);
        for (; pos < stop; ++pos) {
            common_batch_add(batch, 1, pos, { seq }, pos + 1 == stop);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
        llama_synchronize(ctx);
    }
    llama_batch_free(batch);
    return true;
}

static vbr_checkpoint_frontier_fields c2_gpu_frontier(int64_t n_past) {
    static const std::string exec_id    = "c2-gpu-exec";
    static const std::string adapter_id = "c2-gpu-adapter";
    static const std::string media_id   = "c2-gpu-media";
    vbr_checkpoint_frontier_fields frontier;
    frontier.execution_identity          = exec_id.c_str();
    frontier.execution_identity_len      = exec_id.size();
    frontier.adapter_config_identity     = adapter_id.c_str();
    frontier.adapter_config_identity_len = adapter_id.size();
    frontier.media_content_identity      = media_id.c_str();
    frontier.media_content_identity_len  = media_id.size();
    frontier.sequence_epoch = 1;
    frontier.token_count    = n_past;
    frontier.next_position  = (llama_pos) n_past;
    return frontier;
}

static llama_vbr_checkpoint_shadow * c2_bridge_capture(
        llama_memory_t mem, int64_t n_past, vbr_checkpoint_capture_reason & reason,
        vbr_checkpoint_reset_scope * reset_scope = nullptr) {
    const auto frontier = c2_gpu_frontier(n_past);
    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(mem, 0, &frontier, &result);
    reason = result.reason;
    if (reset_scope != nullptr) {
        *reset_scope = result.reset_scope;
    }
    return result.handle;
}

static bool c2_bridge_capture_ok(llama_memory_t mem, int64_t n_past,
                                 vbr_checkpoint_capture_reason & reason) {
    auto * handle = c2_bridge_capture(mem, n_past, reason);
    const bool ok = handle != nullptr;
    llama_vbr_checkpoint_shadow_free(handle);
    return ok;
}

// recovery-ring census for one owner pool: every non-free record reserved on behalf of that
// pool. Sampled before/after a scope lifetime, the delta is the exact mint count for the
// whole logical operation tree (the reservation's owner pool is immutable, v4 review F2).
static int32_t c2_recovery_census_pool(const vbr_pool_uuid & pool) {
    int32_t count = 0;
    vbr_failed_operation_record record;
    for (int32_t i = 0; vbr_recovery_get_record(i, record); ++i) {
        if (record.state != vbr_recovery_state::free_slot &&
                record.owner_pool_hi == pool.hi && record.owner_pool_lo == pool.lo) {
            ++count;
        }
    }
    return count;
}

// production quarantine drain: recovery work resolves at real decode boundaries — loop a
// bounded number of single-token decodes until the child rearms
static bool c2_gpu_drain_until_rearmed(llama_context * ctx, llama_kv_cache * kv,
                                       llama_pos & pos_cursor) {
    for (int attempt = 0; attempt < 8; ++attempt) {
        if (!c2_gpu_decode_range(ctx, 0, pos_cursor, pos_cursor + 1, 32)) {
            return false;
        }
        ++pos_cursor;
        const auto * tracker = llama_kv_cache_vbr_epoch_test::tracker_get(kv);
        if (tracker != nullptr && !tracker->shadow_unavailable()) {
            return true;
        }
    }
    return false;
}

// row (d): a single llama_decode spanning more than eight ubatches — the pending->awaiting
// evidence transfer must survive (commit-1 F5 capacity growth) and leave capture available
static int c2_gpu_row_d(llama_model * model) {
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, 512, 384, 1, ctx)) {
        fprintf(stderr, "row d: context creation failed\n");
        return 1;
    }
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!c2_gpu_children(llama_get_memory(ctx.get()), base, swa)) {
        fprintf(stderr, "row d: armed iSWA children unavailable\n");
        return 1;
    }
    // 288 tokens / 32-token ubatches = 9 ubatches in ONE decode call
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, 288, 288)) {
        fprintf(stderr, "row d: >8-ubatch decode failed\n");
        return 1;
    }
    const auto * tracker = llama_kv_cache_vbr_epoch_test::tracker_get(base);
    if (tracker == nullptr || !tracker->stable() || tracker->shadow_unavailable()) {
        fprintf(stderr, "row d: base tracker not stable/armed after 9-ubatch decode\n");
        return 1;
    }
    // Commit-3 G-only bridge: P and F are deliberately combined outside the bridge. Both
    // asymmetric rows must invoke the sole evaluator exactly once and produce distinct
    // P/G versus F/G candidates.
    vbr_checkpoint_capture_reason eval_capture_reason;
    auto * eval_handle =
        c2_bridge_capture(llama_get_memory(ctx.get()), 288, eval_capture_reason);
    if (eval_handle == nullptr) {
        fprintf(stderr, "row d: G-only bridge capture failed (%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(eval_capture_reason));
        return 1;
    }
    const auto eval_frontier = c2_gpu_frontier(288);
    for (const auto [p_eligible, f_eligible] :
            { std::pair<bool, bool>{ false, true }, { true, false } }) {
        llama_vbr_checkpoint_shadow_evaluation evaluation;
        llama_vbr_checkpoint_shadow_evaluate(
                eval_handle, llama_get_memory(ctx.get()), 0, &eval_frontier, &evaluation);
        const bool p_g = p_eligible && evaluation.strict;
        const bool f_g = f_eligible && evaluation.strict;
        if (evaluation.evaluator_invocations != 1 || !evaluation.strict || p_g == f_g ||
                p_g != p_eligible || f_g != f_eligible) {
            fprintf(stderr, "row d: G-only asymmetric P/F bridge row failed "
                    "(P=%d F=%d calls=%u strict=%d)\n",
                    (int) p_eligible, (int) f_eligible,
                    (unsigned) evaluation.evaluator_invocations, (int) evaluation.strict);
            llama_vbr_checkpoint_shadow_free(eval_handle);
            return 1;
        }
    }
    llama_vbr_checkpoint_shadow_free(eval_handle);

    vbr_checkpoint_capture_reason reason;
    if (!c2_bridge_capture_ok(llama_get_memory(ctx.get()), 288, reason)) {
        fprintf(stderr, "row d: capture unavailable after 9-ubatch decode (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    fprintf(stderr, "C2 GPU row (d) >8-ubatch evidence PASS\n");
    return 0;
}

// row (b): joined poison through the REAL scope/citation/stamp path — an event citing the
// open root scope stamps a position outside its authenticated manifest; the root poisons,
// its FAILED close autorecords exactly one recovery entry, and the composite stays
// unavailable until the real decode-boundary drain resolves it
static int c2_gpu_row_b(llama_model * model) {
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, 128, 32, 1, ctx)) {
        fprintf(stderr, "row b: context creation failed\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!c2_gpu_children(mem, base, swa)) {
        fprintf(stderr, "row b: armed iSWA children unavailable\n");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, 4, 32)) {
        fprintf(stderr, "row b: seed decode failed\n");
        return 1;
    }
    const auto * tracker = llama_kv_cache_vbr_epoch_test::tracker_get(base);
    const auto pool = tracker->pool_identity();
    const int32_t census_before = c2_recovery_census_pool(pool);
    {
        void * scope = llama_kv_cache_vbr_epoch_test::open_narrow_trim_scope(base);
        if (scope == nullptr) {
            fprintf(stderr, "row b: root scope did not open\n");
            return 1;
        }
        const bool stamped = llama_kv_cache_vbr_epoch_test::stamp_outside_manifest(base, scope);
        // production FAILED close: no succeed() — the eager recovery reservation records
        llama_kv_cache_vbr_epoch_test::close_scope_without_success(scope);
        if (!stamped) {
            fprintf(stderr, "row b: cited event was unavailable\n");
            return 1;
        }
    }
    if (!tracker->shadow_unavailable()) {
        fprintf(stderr, "row b: refused stamp did not latch shadow-unavailable\n");
        return 1;
    }
    // the root's failed close autorecorded EXACTLY ONE recovery entry for this pool
    if (c2_recovery_census_pool(pool) != census_before + 1 ||
            !vbr_recovery_pending_for(pool.hi, pool.lo)) {
        fprintf(stderr, "row b: failed close did not autorecord exactly one recovery entry\n");
        return 1;
    }
    vbr_checkpoint_capture_reason reason;
    vbr_checkpoint_reset_scope reset_scope;
    auto * poisoned = c2_bridge_capture(mem, 4, reason, &reset_scope);
    if (poisoned != nullptr ||
            reason != vbr_checkpoint_capture_reason::controller_unavailable ||
            reset_scope != vbr_checkpoint_reset_scope::global) {
        fprintf(stderr, "row b: poisoned child did not make the composite unavailable\n");
        llama_vbr_checkpoint_shadow_free(poisoned);
        return 1;
    }
    if (llama_kv_cache_vbr_epoch_test::tracker_mut(base)->try_rearm()) {
        fprintf(stderr, "row b: rearm succeeded with unresolved recovery work pending\n");
        return 1;
    }
    llama_pos pos_cursor = 4;
    if (!c2_gpu_drain_until_rearmed(ctx.get(), base, pos_cursor)) {
        fprintf(stderr, "row b: decode-boundary quarantine drain did not rearm the child\n");
        return 1;
    }
    if (vbr_recovery_pending_for(pool.hi, pool.lo)) {
        fprintf(stderr, "row b: quarantine remained pending after the drain\n");
        return 1;
    }
    if (!c2_bridge_capture_ok(mem, (int64_t) pos_cursor, reason)) {
        fprintf(stderr, "row b: capture still unavailable after recovery (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    fprintf(stderr, "C2 GPU row (b) joined-poison/recovery PASS\n");
    return 0;
}

// row (c): a narrow-manifest parent held open across a REAL nested iSWA mutation — the base
// child JOINS the parent (its trim stamps are outside the parent manifest and poison the
// root) while the SWA sibling proceeds under the wrapper's adopted identity. The whole
// refused tree mints exactly ONE recovery record (pool-keyed census delta), the sibling
// stays healthy, and the drain must restore full capture.
static int c2_gpu_row_c(llama_model * model) {
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, 128, 32, 1, ctx)) {
        fprintf(stderr, "row c: context creation failed\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!c2_gpu_children(mem, base, swa)) {
        fprintf(stderr, "row c: armed iSWA children unavailable\n");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, 8, 32)) {
        fprintf(stderr, "row c: seed decode failed\n");
        return 1;
    }
    const auto * base_tracker = llama_kv_cache_vbr_epoch_test::tracker_get(base);
    const auto * swa_tracker  = llama_kv_cache_vbr_epoch_test::tracker_get(swa);
    const auto pool = base_tracker->pool_identity();
    const int32_t census_before = c2_recovery_census_pool(pool);
    {
        void * scope = llama_kv_cache_vbr_epoch_test::open_narrow_trim_scope(base);
        if (scope == nullptr) {
            fprintf(stderr, "row c: parent scope did not open\n");
            return 1;
        }
        // REAL nested mutation through the iSWA parent: the base child joins the open parent
        // (whose manifest does not cover positions >= 6 -> refused stamps -> root poison);
        // the SWA sibling runs under the wrapper's own adopted identity and stays clean
        if (!llama_memory_seq_rm(mem, 0, 6, -1)) {
            llama_kv_cache_vbr_epoch_test::close_scope_without_success(scope);
            fprintf(stderr, "row c: nested trim rejected unexpectedly\n");
            return 1;
        }
        llama_kv_cache_vbr_epoch_test::close_scope_without_success(scope);
    }
    if (!base_tracker->shadow_unavailable()) {
        fprintf(stderr, "row c: joined-poisoned base child did not latch\n");
        return 1;
    }
    if (swa_tracker->shadow_unavailable()) {
        fprintf(stderr, "row c: healthy sibling was latched by the refused parent\n");
        return 1;
    }
    // one-mint census: the whole nested tree reserved exactly one recovery record
    if (c2_recovery_census_pool(pool) != census_before + 1) {
        fprintf(stderr, "row c: refused tree census delta != 1\n");
        return 1;
    }
    vbr_checkpoint_capture_reason reason;
    vbr_checkpoint_reset_scope reset_scope;
    auto * poisoned = c2_bridge_capture(mem, 6, reason, &reset_scope);
    if (poisoned != nullptr ||
            reason != vbr_checkpoint_capture_reason::controller_unavailable ||
            reset_scope != vbr_checkpoint_reset_scope::global) {
        fprintf(stderr, "row c: poisoned root did not make the composite unavailable\n");
        llama_vbr_checkpoint_shadow_free(poisoned);
        return 1;
    }
    llama_pos pos_cursor = 6;
    if (!c2_gpu_drain_until_rearmed(ctx.get(), base, pos_cursor)) {
        fprintf(stderr, "row c: decode-boundary drain did not rearm the base child\n");
        return 1;
    }
    if (vbr_recovery_pending_for(pool.hi, pool.lo)) {
        fprintf(stderr, "row c: recovery work remained pending after the drain\n");
        return 1;
    }
    if (!c2_bridge_capture_ok(mem, (int64_t) pos_cursor, reason)) {
        fprintf(stderr, "row c: capture did not recover after the drain (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    fprintf(stderr, "C2 GPU row (c) joined/adopted parent one-mint census PASS\n");
    return 0;
}

// row (e): mixed two-child FENCE outcome — one child's in-flight per-target evidence goes
// stale between decode submission and the synchronize fence (slab-reset commit race). The
// REAL fence terminal path must fail+latch that child, the sibling's committed evidence must
// stay tracker-local and truthful, and recovery must restore the composite.
static int c2_gpu_row_e(llama_model * model) {
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, 128, 32, 1, ctx)) {
        fprintf(stderr, "row e: context creation failed\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!c2_gpu_children(mem, base, swa)) {
        fprintf(stderr, "row e: armed iSWA children unavailable\n");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, 8, 32)) {
        fprintf(stderr, "row e: seed decode failed\n");
        return 1;
    }
    // async decode: submit WITHOUT fencing, then stale one submitted extent on the base
    // child before the synchronize fence commits it
    {
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, 1, 8, { 0 }, true);
        const bool decoded = llama_decode(ctx.get(), batch) == 0;
        llama_batch_free(batch);
        if (!decoded) {
            fprintf(stderr, "row e: fence-row decode failed\n");
            return 1;
        }
        if (!llama_kv_cache_vbr_epoch_test::inject_stale_submitted_extent(base)) {
            fprintf(stderr, "row e: no in-flight base operation to stale (tune the seam)\n");
            return 1;
        }
        llama_synchronize(ctx.get());  // the REAL fence: commit fails on the obsolete handle
    }
    const auto * base_tracker = llama_kv_cache_vbr_epoch_test::tracker_get(base);
    const auto * swa_tracker  = llama_kv_cache_vbr_epoch_test::tracker_get(swa);
    if (!base_tracker->shadow_unavailable()) {
        fprintf(stderr, "row e: fence commit failure did not latch the base child\n");
        return 1;
    }
    if (swa_tracker->shadow_unavailable()) {
        fprintf(stderr, "row e: sibling was latched by the other child's fence failure\n");
        return 1;
    }
    // the sibling's committed evidence stays tracker-local and truthful
    vbr_checkpoint_generation_controller swa_record;
    if (!swa->vbr_generation_capture_live_guarded(0, 0, 9, swa_record)) {
        fprintf(stderr, "row e: sibling capture failed after the mixed fence\n");
        return 1;
    }
    vbr_generation_live_controller_view swa_view;
    if (!swa->vbr_generation_live_guarded_view(0, 0, 9, swa_view)) {
        fprintf(stderr, "row e: sibling live view unavailable\n");
        return 1;
    }
    vbr_checkpoint_generation_record sibling;
    sibling.status = vbr_checkpoint_generation_status::complete;
    sibling.controllers.push_back(swa_record);
    vbr_generation_live_view live;
    live.legacy_eligible = true;
    live.controllers.push_back(swa_view);
    const auto sibling_result = checkpoint_vbr_eligibility(sibling, live);
    if (sibling_result.category != vbr_checkpoint_eligibility_category::strict_accept) {
        fprintf(stderr, "row e: sibling committed evidence was not preserved (reason %d)\n",
                (int) sibling_result.reason);
        return 1;
    }
    // the failed child blocks the composite until recovery resolves at a real boundary
    vbr_checkpoint_capture_reason reason;
    vbr_checkpoint_reset_scope reset_scope;
    auto * failed = c2_bridge_capture(mem, 9, reason, &reset_scope);
    if (failed != nullptr ||
            reason != vbr_checkpoint_capture_reason::controller_unavailable ||
            reset_scope != vbr_checkpoint_reset_scope::global) {
        fprintf(stderr, "row e: failed child did not make the composite unavailable\n");
        llama_vbr_checkpoint_shadow_free(failed);
        return 1;
    }
    llama_pos pos_cursor = 9;
    if (!c2_gpu_drain_until_rearmed(ctx.get(), base, pos_cursor)) {
        fprintf(stderr, "row e: failed child did not recover at the decode boundary\n");
        return 1;
    }
    if (!c2_bridge_capture_ok(mem, (int64_t) pos_cursor, reason)) {
        fprintf(stderr, "row e: composite still unavailable after child recovery (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    fprintf(stderr, "C2 GPU row (e) mixed fence outcome PASS\n");
    return 0;
}

// row (a): SWA wrap and cross-sequence reuse through REAL slot selection, in two arms sized
// so slot selection MUST reuse captured cells (single-seq contexts get a windowed SWA cache;
// n_seq_max>1 forces a full-size SWA cache, so the cross-seq arm fills it first). Each arm
// prechecks that the captured set was actually disturbed — an undisturbed arm fails as
// uncovered, never passes vacuously.
//
// Commit-3 adds the exact-cardinality correction and a third, in-domain same-sequence reuse
// arm. The first two arms retain the prior fail-closed proofs (out-of-domain wrap and
// cross-sequence reuse); arm 3 is the only row allowed to earn swa_wrap classification.
static int c2_gpu_row_a(llama_model * model) {
    const int32_t n_swa = llama_model_n_swa(model);
    if (n_swa <= 0 || n_swa > 1728) {
        fprintf(stderr, "row a: SKIP — model window %d outside fixture range\n", n_swa);
        return 0;
    }
    const llama_pos window = (llama_pos) n_swa;

    // --- arm 1: same-sequence wrap (windowed SWA cache; n_seq_max = 1) -----------------------
    {
        const uint32_t n_ctx = (uint32_t) window + 640;  // > any padded SWA child size
        llama_context_ptr ctx;
        if (!c2_gpu_context(model, n_ctx, 64, 1, ctx)) {
            fprintf(stderr, "row a: arm-1 context creation failed\n");
            return 1;
        }
        llama_kv_cache * base = nullptr;
        llama_kv_cache * swa  = nullptr;
        if (!c2_gpu_children(llama_get_memory(ctx.get()), base, swa)) {
            fprintf(stderr, "row a: arm-1 armed iSWA children unavailable\n");
            return 1;
        }
        if ((uint32_t) swa->get_size() >= n_ctx) {
            fprintf(stderr, "row a: arm-1 SWA cache is full-size (%u cells) — wrap unreachable, row uncovered\n",
                    (unsigned) swa->get_size());
            return 1;
        }
        const llama_pos frontier = window / 2;
        if (!c2_gpu_decode_range(ctx.get(), 0, 0, frontier, 64)) {
            fprintf(stderr, "row a: arm-1 seed decode failed\n");
            return 1;
        }
        vbr_checkpoint_generation_controller captured;
        if (!swa->vbr_generation_capture_live_guarded(0, 0, frontier, captured)) {
            fprintf(stderr, "row a: arm-1 SWA capture failed\n");
            return 1;
        }
        vbr_checkpoint_generation_record record;
        record.status = vbr_checkpoint_generation_status::complete;
        record.controllers.push_back(captured);

        // decode past the SWA child's cell count so real slot selection wraps: logical
        // positions leave the ownership index's positional domain, which latches the per-seq
        // view unavailable (Rev-5 pin 3 fail-closed domain restriction). The honest wrap
        // property on the REAL path is therefore fail-closed evidence UNAVAILABILITY —
        // never acceptance, never fabricated coverage. ([I8] keeps wrapped-SWA live evidence
        // out of production; §5.5 row 2 classification remains reachable only through
        // in-domain reuse — see arm 2 and the worklog SUBSTRATE note for the Sol ruling.)
        if (!c2_gpu_decode_range(ctx.get(), 0, frontier, (llama_pos) n_ctx - 1, 64)) {
            fprintf(stderr, "row a: arm-1 wrap decode failed\n");
            return 1;
        }
        vbr_generation_live_controller_view view;
        if (swa->vbr_generation_live_guarded_view(0, 0, frontier, view)) {
            fprintf(stderr, "row a: arm-1 wrapped view stayed available (out-of-domain positions must fail closed)\n");
            return 1;
        }
        vbr_checkpoint_generation_controller recapture;
        if (swa->vbr_generation_capture_live_guarded(0, 0, frontier, recapture)) {
            fprintf(stderr, "row a: arm-1 wrapped capture stayed available (must fail closed)\n");
            return 1;
        }
        GGML_UNUSED(record);
    }

    // --- arm 2: cross-sequence reuse through real slot selection. Windowed SWA cache (the
    // full-size cache deliberately never prunes, so a full unified cache just refuses the
    // second sequence); positions are kept strictly inside the ownership index domain by
    // sizing the fill from the ACTUAL SWA child cell count, so the captured view stays
    // available while seq 1's allocation prunes/reuses expired seq-0 cells. ----------------
    {
        // unified multi-seq sizing gives the SWA child n_swa*n_seq_max + n_ubatch cells;
        // the context must exceed that so the windowed child keeps prune slack
        const uint32_t n_ctx = (uint32_t) window * 2 + 640;
        llama_context_ptr ctx;
        if (!c2_gpu_context(model, n_ctx, 64, 2, ctx)) {
            fprintf(stderr, "row a: arm-2 context creation failed\n");
            return 1;
        }
        llama_kv_cache * base = nullptr;
        llama_kv_cache * swa  = nullptr;
        if (!c2_gpu_children(llama_get_memory(ctx.get()), base, swa)) {
            fprintf(stderr, "row a: arm-2 armed iSWA children unavailable\n");
            return 1;
        }
        const uint32_t swa_size = swa->get_size();
        if (swa_size >= n_ctx || swa_size < (uint32_t) window + 96) {
            fprintf(stderr, "row a: arm-2 SWA child size %u leaves no expired-reuse slack — row uncovered\n",
                    (unsigned) swa_size);
            return 1;
        }
        // fill every SWA cell with seq 0 (positions 0..swa_size-1: all inside the index
        // domain); cells at pos <= swa_size-1-window are expired-but-occupied
        const llama_pos fill = (llama_pos) swa_size;
        if (!c2_gpu_decode_range(ctx.get(), 0, 0, fill, 64)) {
            fprintf(stderr, "row a: arm-2 fill decode failed\n");
            return 1;
        }
        vbr_checkpoint_generation_controller captured;
        if (!swa->vbr_generation_capture_live_guarded(0, 0, fill, captured)) {
            fprintf(stderr, "row a: arm-2 SWA capture failed\n");
            return 1;
        }
        vbr_checkpoint_generation_record record;
        record.status = vbr_checkpoint_generation_status::complete;
        record.controllers.push_back(captured);
        // seq 1's allocation must prune/reuse expired seq-0 cells (zero free cells remain)
        if (!c2_gpu_decode_range(ctx.get(), 1, 0, 64, 64)) {
            fprintf(stderr, "row a: arm-2 cross-seq decode failed\n");
            return 1;
        }
        vbr_generation_live_controller_view view;
        if (!swa->vbr_generation_live_guarded_view(0, 0, fill, view)) {
            fprintf(stderr, "row a: arm-2 live view unavailable after cross-seq reuse\n");
            return 1;
        }
        if (view.streams.empty() ||
                view.streams[0].exact_dependency_count >= captured.streams[0].captured_dependency_count) {
            fprintf(stderr, "row a: arm-2 cross-seq decode left the captured set undisturbed — row uncovered\n");
            return 1;
        }
        vbr_generation_live_view live;
        live.legacy_eligible = true;
        live.controllers.push_back(view);
        const auto cross_result = checkpoint_vbr_eligibility(record, live);
        if (cross_result.category == vbr_checkpoint_eligibility_category::strict_accept ||
                cross_result.category == vbr_checkpoint_eligibility_category::live_rebased_shadow_accept) {
            fprintf(stderr, "row a: arm-2 cross-seq record was accepted\n");
            return 1;
        }
        if (cross_result.tombstone_class != vbr_expected_tombstone_class::unexplained) {
            fprintf(stderr, "row a: arm-2 cross-seq reuse classified %d, expected unexplained\n",
                    (int) cross_result.tombstone_class);
            return 1;
        }
    }

    // --- arm 3: in-domain SAME-SEQUENCE occupied reuse. Seq 0 first contributes more than
    // one SWA window but fewer cells than the physical child; seq 1 fills the remaining empty
    // cells. The next seq-0 token therefore has no free slot and must reuse an expired seq-0
    // dependency while its new logical position remains inside the ownership-index domain.
    // This is the real geometry for §5.5 row 2: membership survives, position advances past
    // the captured frontier, exact live rank shrinks, and only swa_wrap is admissible.
    {
        const uint32_t n_ctx = (uint32_t) window * 2 + 640;
        llama_context_ptr ctx;
        if (!c2_gpu_context(model, n_ctx, 64, 2, ctx)) {
            fprintf(stderr, "row a: arm-3 context creation failed\n");
            return 1;
        }
        llama_kv_cache * base = nullptr;
        llama_kv_cache * swa  = nullptr;
        if (!c2_gpu_children(llama_get_memory(ctx.get()), base, swa)) {
            fprintf(stderr, "row a: arm-3 armed iSWA children unavailable\n");
            return 1;
        }
        const uint32_t swa_size = swa->get_size();
        const llama_pos seq0_fill = window + 64;
        if (seq0_fill <= window || seq0_fill >= (llama_pos) swa_size ||
                swa_size - (uint32_t) seq0_fill < 64) {
            fprintf(stderr, "row a: arm-3 geometry cannot satisfy n_swa < pos < child-size "
                    "(window=%d fill=%d size=%u)\n",
                    (int) window, (int) seq0_fill, (unsigned) swa_size);
            return 1;
        }
        if (!c2_gpu_decode_range(ctx.get(), 0, 0, seq0_fill, 64)) {
            fprintf(stderr, "row a: arm-3 seq-0 fill failed\n");
            return 1;
        }
        const llama_pos seq1_fill = (llama_pos) swa_size - seq0_fill;
        if (!c2_gpu_decode_range(ctx.get(), 1, 0, seq1_fill, 64)) {
            fprintf(stderr, "row a: arm-3 seq-1 fill failed\n");
            return 1;
        }

        vbr_checkpoint_generation_controller captured;
        if (!swa->vbr_generation_capture_live_guarded(0, 0, seq0_fill, captured) ||
                captured.streams.empty()) {
            fprintf(stderr, "row a: arm-3 SWA capture failed\n");
            return 1;
        }
        vbr_checkpoint_generation_record record;
        record.status = vbr_checkpoint_generation_status::complete;
        record.controllers.push_back(captured);
        if (!c2_gpu_decode_range(ctx.get(), 0, seq0_fill, seq0_fill + 1, 1)) {
            fprintf(stderr, "row a: arm-3 occupied-reuse decode failed\n");
            return 1;
        }

        vbr_generation_live_controller_view view;
        if (!swa->vbr_generation_live_guarded_view(0, 0, seq0_fill, view) ||
                view.streams.empty()) {
            fprintf(stderr, "row a: arm-3 ownership view became unavailable\n");
            return 1;
        }
        const auto & stored_stream = captured.streams[0];
        const auto & live_stream   = view.streams[0];
        bool saw_same_seq_higher = false;
        for (const auto & page : stored_stream.pages) {
            const uint32_t page_base = page.page_index * VBR_GENERATION_PAGE_CELLS;
            for (uint32_t off = 0; off < VBR_GENERATION_PAGE_CELLS; ++off) {
                if ((page.covered_mask[off / 64] & (uint64_t(1) << (off % 64))) == 0) {
                    continue;
                }
                const uint32_t cell = page_base + off;
                if (live_stream.cell_has_seq(
                            live_stream.membership_context, live_stream.stream_index, cell, 0) &&
                        live_stream.cell_pos(
                            live_stream.membership_context, live_stream.stream_index, cell) >= seq0_fill) {
                    saw_same_seq_higher = true;
                }
            }
        }
        if (!saw_same_seq_higher ||
                live_stream.exact_dependency_count >= stored_stream.captured_dependency_count) {
            fprintf(stderr, "row a: arm-3 did not prove same-seq higher-position rank shrink\n");
            return 1;
        }

        vbr_generation_live_view live;
        live.legacy_eligible = true;
        live.controllers.push_back(view);
        const auto result = checkpoint_vbr_eligibility(record, live);
        if (result.category != vbr_checkpoint_eligibility_category::strict_reject ||
                result.tombstone_class != vbr_expected_tombstone_class::swa_wrap) {
            fprintf(stderr, "row a: arm-3 classified reason=%d tombstone=%d, expected only swa_wrap\n",
                    (int) result.reason, (int) result.tombstone_class);
            return 1;
        }
    }

    fprintf(stderr, "C2 GPU row (a) three-arm wrap/cross-seq/in-domain-swa PASS\n");
    return 0;
}

static void c3_oracle_cover_cell(
        vbr_checkpoint_generation_stream & stream,
        uint32_t cell) {
    const uint32_t page_index = cell / VBR_GENERATION_PAGE_CELLS;
    auto it = std::lower_bound(
            stream.pages.begin(), stream.pages.end(), page_index,
            [](const vbr_generation_page_ref & page, uint32_t index) {
                return page.page_index < index;
            });
    if (it == stream.pages.end() || it->page_index != page_index) {
        vbr_generation_page_ref page;
        page.page_index = page_index;
        it = stream.pages.insert(it, page);
    }
    const uint32_t offset = cell % VBR_GENERATION_PAGE_CELLS;
    it->covered_mask[offset / 64] |= uint64_t(1) << (offset % 64);
}

// Commit-3 finding-7 row: use a full-size SWA child so old, serializer-masked cells remain
// physically occupied beside visible cells for the SAME sequence. The canonical observer's
// independently scanned logical-position set must equal the real sequence serializer's set.
// Deliberately adding one masked physical cell to the production covered mask must then be
// caught as set_mismatch.
static int c3_gpu_oracle_serializer_visibility_row(llama_model * model) {
    const int32_t n_swa = llama_model_n_swa(model);
    if (n_swa <= 0 || n_swa > 1728) {
        fprintf(stderr, "C3 oracle visibility row: SKIP — model window %d outside fixture range\n",
                n_swa);
        return 0;
    }

    const llama_pos frontier = (llama_pos) n_swa + 32;
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, (uint32_t) frontier + 64, 64, 1, ctx, /*swa_full=*/true)) {
        fprintf(stderr, "C3 oracle visibility row: context creation failed\n");
        return 1;
    }
    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (!c2_gpu_children(llama_get_memory(ctx.get()), base, swa)) {
        fprintf(stderr, "C3 oracle visibility row: armed iSWA children unavailable\n");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, frontier, 64)) {
        fprintf(stderr, "C3 oracle visibility row: seed decode failed\n");
        return 1;
    }

    std::vector<vbr_generation_oracle_cell> observations;
    if (!swa->vbr_generation_oracle_observations(0, frontier, observations)) {
        fprintf(stderr, "C3 oracle visibility row: canonical observation unavailable\n");
        return 1;
    }

    vbr_checkpoint_generation_stream production;
    production.stream_index         = 0;
    production.dependency_seq_id    = 0;
    production.computation_frontier = frontier;
    uint32_t visible_count   = 0;
    uint32_t masked_count    = 0;
    uint32_t one_masked_cell = UINT32_MAX;
    for (const auto & cell : observations) {
        if (!cell.has_dependency_seq || cell.position < 0 || cell.position >= frontier) {
            continue;
        }
        if (!cell.attention_visible) {
            ++masked_count;
            one_masked_cell = cell.physical_cell;
            continue;
        }
        ++visible_count;
        ++production.captured_dependency_count;
        c3_oracle_cover_cell(production, cell.physical_cell);
    }

    std::vector<llama_pos> visible_positions;
    visible_positions.reserve(visible_count);
    for (const auto & cell : observations) {
        if (cell.has_dependency_seq && cell.attention_visible &&
                cell.position >= 0 && cell.position < frontier) {
            visible_positions.push_back(cell.position);
        }
    }
    std::vector<llama_pos> serializer_positions;
    if (visible_count == 0 || masked_count == 0 || one_masked_cell == UINT32_MAX ||
            !llama_kv_cache_vbr_epoch_test::serializer_positions(
                    swa, 0, serializer_positions) ||
            serializer_positions != visible_positions) {
        fprintf(stderr,
                "C3 oracle visibility row: fixture/serializer mismatch "
                "(visible=%u masked=%u serialized=%zu)\n",
                visible_count, masked_count, serializer_positions.size());
        return 1;
    }
    const uint32_t serializer_count = (uint32_t) serializer_positions.size();

    set_test_env("VBR_GENERATION_ORACLE", "1");
    const auto baseline = vbr_generation_oracle_capture(frontier, observations);
    auto audit = vbr_generation_oracle_audit(frontier, observations, baseline, production);
    if (!audit.complete || !audit.set_equal || !audit.bytes_equal ||
            audit.independent_count != serializer_count) {
        fprintf(stderr,
                "C3 oracle visibility row: independent set did not match serializer "
                "(complete=%d set=%d bytes=%d independent=%u serialized=%u)\n",
                (int) audit.complete, (int) audit.set_equal, (int) audit.bytes_equal,
                audit.independent_count, serializer_count);
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }

    c3_oracle_cover_cell(production, one_masked_cell);
    ++production.captured_dependency_count;
    audit = vbr_generation_oracle_audit(frontier, observations, baseline, production);
    if (!audit.complete || audit.set_equal ||
            audit.independent_count != serializer_count) {
        fprintf(stderr,
                "C3 oracle visibility row: masked-cell production inclusion escaped "
                "set mismatch (complete=%d set=%d independent=%u)\n",
                (int) audit.complete, (int) audit.set_equal, audit.independent_count);
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }

    // Payload-complete children are outside the live-dependency set even when their attention
    // cells are visible. This copy exercises the observer vocabulary directly; the production
    // bridge enforces the same exclusion structurally by never invoking the observer for
    // payload_complete children.
    auto payload_complete = observations;
    for (auto & cell : payload_complete) {
        cell.payload_supplied = true;
    }
    const auto payload_baseline =
            vbr_generation_oracle_capture(frontier, payload_complete);
    unset_test_env("VBR_GENERATION_ORACLE");
    if (!payload_baseline.complete || !payload_baseline.dependency_cells.empty()) {
        fprintf(stderr, "C3 oracle visibility row: payload-complete cells became live dependencies\n");
        return 1;
    }

    fprintf(stderr, "C3 oracle serializer-visibility/set-mismatch row PASS\n");
    return 0;
}

// Commit-3 §6.2 bridge row: capture two real dependency cells with the disabled-only byte
// sidecar enabled, append within the same page so strict evaluation refines, then audit the
// independently observed set+bytes. The fault seam pins each closed outcome, and a handle
// captured before enable proves late enable is unavailable rather than baseline fabrication.
static int c3_gpu_oracle_bridge_row(llama_model * model) {
    llama_context_ptr ctx;
    if (!c2_gpu_context(model, 128, 32, 1, ctx)) {
        fprintf(stderr, "C3 oracle row: context creation failed\n");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 0, 2, 2)) {
        fprintf(stderr, "C3 oracle row: seed decode failed\n");
        return 1;
    }
    auto mem = llama_get_memory(ctx.get());
    vbr_checkpoint_capture_reason reason;

    unset_test_env("VBR_GENERATION_ORACLE");
    auto * no_sidecar = c2_bridge_capture(mem, 2, reason);
    if (no_sidecar == nullptr) {
        fprintf(stderr, "C3 oracle row: sidecar-less capture failed\n");
        return 1;
    }

    set_test_env("VBR_GENERATION_ORACLE", "1");
    auto * audited = c2_bridge_capture(mem, 2, reason);
    if (audited == nullptr) {
        fprintf(stderr, "C3 oracle row: audited capture failed (%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        llama_vbr_checkpoint_shadow_free(no_sidecar);
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }
    if (!c2_gpu_decode_range(ctx.get(), 0, 2, 3, 1)) {
        fprintf(stderr, "C3 oracle row: append refinement decode failed\n");
        llama_vbr_checkpoint_shadow_free(no_sidecar);
        llama_vbr_checkpoint_shadow_free(audited);
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }

    set_test_env("VBR_GENERATION_FORCE_AUDIT", "1");
    const auto frontier = c2_gpu_frontier(2);
    auto evaluate = [&](llama_vbr_checkpoint_shadow * handle) {
        llama_vbr_checkpoint_shadow_evaluation evaluation;
        llama_vbr_checkpoint_shadow_evaluate(handle, mem, 0, &frontier, &evaluation);
        return evaluation;
    };
    auto evaluation = evaluate(audited);
    if (evaluation.evaluator_invocations != 1 || !evaluation.strict ||
            !evaluation.refinement_used ||
            evaluation.oracle_outcome != vbr_checkpoint_oracle_outcome::pass) {
        fprintf(stderr, "C3 oracle row: real set+byte audit did not pass "
                "(calls=%u strict=%d refine=%d outcome=%d)\n",
                (unsigned) evaluation.evaluator_invocations, (int) evaluation.strict,
                (int) evaluation.refinement_used, (int) evaluation.oracle_outcome);
        llama_vbr_checkpoint_shadow_free(no_sidecar);
        llama_vbr_checkpoint_shadow_free(audited);
        unset_test_env("VBR_GENERATION_FORCE_AUDIT");
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }
    for (const auto & fault : {
            std::pair<const char *, vbr_checkpoint_oracle_outcome>{
                "set", vbr_checkpoint_oracle_outcome::set_mismatch },
            { "bytes", vbr_checkpoint_oracle_outcome::byte_mismatch },
            { "unavailable", vbr_checkpoint_oracle_outcome::unavailable },
         }) {
        set_test_env("VBR_GENERATION_ORACLE_INJECT", fault.first);
        evaluation = evaluate(audited);
        if (evaluation.oracle_outcome != fault.second) {
            fprintf(stderr, "C3 oracle row: injected %s produced outcome %d\n",
                    fault.first, (int) evaluation.oracle_outcome);
            llama_vbr_checkpoint_shadow_free(no_sidecar);
            llama_vbr_checkpoint_shadow_free(audited);
            unset_test_env("VBR_GENERATION_ORACLE_INJECT");
            unset_test_env("VBR_GENERATION_FORCE_AUDIT");
            unset_test_env("VBR_GENERATION_ORACLE");
            return 1;
        }
    }
    unset_test_env("VBR_GENERATION_ORACLE_INJECT");
    evaluation = evaluate(no_sidecar);
    if (evaluation.evaluator_invocations != 1 || !evaluation.strict ||
            evaluation.oracle_outcome != vbr_checkpoint_oracle_outcome::unavailable) {
        fprintf(stderr, "C3 oracle row: late enable fabricated a baseline\n");
        llama_vbr_checkpoint_shadow_free(no_sidecar);
        llama_vbr_checkpoint_shadow_free(audited);
        unset_test_env("VBR_GENERATION_FORCE_AUDIT");
        unset_test_env("VBR_GENERATION_ORACLE");
        return 1;
    }
    llama_vbr_checkpoint_shadow_free(no_sidecar);
    llama_vbr_checkpoint_shadow_free(audited);
    unset_test_env("VBR_GENERATION_FORCE_AUDIT");
    unset_test_env("VBR_GENERATION_ORACLE");
    fprintf(stderr, "C3 oracle sidecar/real-byte/audit bridge row PASS\n");
    return 0;
}

static int run_c2_gpu_rows(llama_model * model) {
    if (c2_gpu_row_d(model) != 0 || c2_gpu_row_b(model) != 0 ||
            c2_gpu_row_c(model) != 0 || c2_gpu_row_e(model) != 0 ||
            c2_gpu_row_a(model) != 0 ||
            c3_gpu_oracle_serializer_visibility_row(model) != 0 ||
            c3_gpu_oracle_bridge_row(model) != 0) {
        return 1;
    }
    fprintf(stderr, "C2 GPU fixture rows (a)-(e) PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--a1-cpu") {
        return run_a1_cpu_tests() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--a2-cpu") {
        return run_a2_cpu_tests() ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--c2-cpu") {
        return run_c2_cpu_tests() ? 0 : 1;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL | --a1-cpu | --a2-cpu | --c2-cpu\n", argv[0]);
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

    if (!run_a1_cpu_tests() || !run_a2_cpu_tests() || !run_c2_cpu_tests()) {
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

    if (run_c2_gpu_rows(model.get()) != 0) {
        return 1;
    }

    fprintf(stderr, "PASS: VBR scoped freeze is nested/iSWA-coherent, defers mutations, "
            "re-evaluates fresh on exit, composes with VBR_FREEZE, and preserves monotone "
            "per-child representation epochs\n");
    return 0;
}
