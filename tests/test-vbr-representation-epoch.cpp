#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama.h"

#include <cstdio>
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

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL\n", argv[0]);
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
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_inner", inner);
    if (llama_memory_vbr_state(mem, 0, 0).retier_freeze_depth != 1) {
        fprintf(stderr, "inner scoped-freeze exit released the outer scope\n");
        return 1;
    }
    llama_memory_vbr_retier_freeze_end(mem, "epoch_test_outer", outer);
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
