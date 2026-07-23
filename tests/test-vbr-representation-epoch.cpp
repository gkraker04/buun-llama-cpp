#include "common.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// Friend of llama_kv_cache: the production low-LCP path reaches the same two operations through
// clear() followed by prepare(). Driving them directly makes the cursor-at-zero state observable
// before a tight budget can immediately start another degrade wave.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }

    static bool degrade(llama_kv_cache * kv) {
        return kv->vbr_degrade_next(kv->vbr_watermark_cells(0));
    }

    static void full_reset(llama_kv_cache * kv) {
        kv->vbr_full_reset();
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

    // The synthetic fixture has no baked order. Keep the controller roomy so only the explicit
    // test waves mutate tiers, and exclude ambient developer overrides from the result.
    set_test_env("VBR_FORCE_GENERIC", "1");
    unset_test_env("VBR_BUDGET_MIB");
    unset_test_env("VBR_DEGRADE_ORDER");
    set_test_env("VBR_PROMOTE", "0");

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
    if (!llama_kv_cache_vbr_epoch_test::active(base) ||
        !llama_kv_cache_vbr_epoch_test::active(swa)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for both iSWA children\n");
        return 0;
    }

    const auto initial = llama_memory_vbr_state(mem, 0, 0);
    if (initial.cursor != 0 || initial.representation_epoch != 0 ||
        initial.representation_epoch_swa != 0 || !decode_one(ctx.get())) {
        fprintf(stderr, "initial VBR state or seed decode failed\n");
        return 1;
    }

    // The two independently mutating children must surface an ordered tuple, never a sum.
    if (!llama_kv_cache_vbr_epoch_test::degrade(base)) {
        fprintf(stderr, "failed to force base degrade\n");
        return 1;
    }
    const auto base_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (base_degraded.representation_epoch <= initial.representation_epoch ||
        base_degraded.representation_epoch_swa != initial.representation_epoch_swa) {
        fprintf(stderr, "base degrade did not advance only the base epoch\n");
        return 1;
    }

    if (!llama_kv_cache_vbr_epoch_test::degrade(swa)) {
        fprintf(stderr, "failed to force SWA degrade\n");
        return 1;
    }
    const auto both_degraded = llama_memory_vbr_state(mem, 0, 0);
    if (both_degraded.representation_epoch != base_degraded.representation_epoch ||
        both_degraded.representation_epoch_swa <= base_degraded.representation_epoch_swa) {
        fprintf(stderr, "SWA degrade did not advance only the SWA epoch\n");
        return 1;
    }

    // This is the production low-LCP/empty-cache reset sequence. clear() changes the referenced
    // representation first; vbr_full_reset() then rewinds the cursor but must advance, not reset,
    // each epoch.
    mem->clear(true);
    const auto cleared = llama_memory_vbr_state(mem, 0, 0);
    if (cleared.representation_epoch <= both_degraded.representation_epoch ||
        cleared.representation_epoch_swa <= both_degraded.representation_epoch_swa) {
        fprintf(stderr, "clear did not advance both child epochs\n");
        return 1;
    }
    llama_kv_cache_vbr_epoch_test::full_reset(base);
    llama_kv_cache_vbr_epoch_test::full_reset(swa);
    const auto reset = llama_memory_vbr_state(mem, 0, 0);
    if (reset.cursor != 0 ||
        reset.representation_epoch <= cleared.representation_epoch ||
        reset.representation_epoch_swa <= cleared.representation_epoch_swa) {
        fprintf(stderr, "full reset did not preserve monotonic epochs while rewinding cursor\n");
        return 1;
    }

    // Refill, degrade again, then adopt the native mixed-tier state onto itself. Ordinary forward
    // fill must not move the representation epochs; the second degrade and import both must.
    if (!decode_one(ctx.get())) {
        fprintf(stderr, "post-reset seed decode failed\n");
        return 1;
    }
    const auto refilled = llama_memory_vbr_state(mem, 0, 0);
    if (!epochs_equal(refilled, reset) ||
        !llama_kv_cache_vbr_epoch_test::degrade(base) ||
        !llama_kv_cache_vbr_epoch_test::degrade(swa)) {
        fprintf(stderr, "post-reset degrade setup failed\n");
        return 1;
    }
    const auto degraded_again = llama_memory_vbr_state(mem, 0, 0);
    if (degraded_again.representation_epoch <= reset.representation_epoch ||
        degraded_again.representation_epoch_swa <= reset.representation_epoch_swa) {
        fprintf(stderr, "degrade-reset-degrade sequence was not monotone\n");
        return 1;
    }

    std::vector<uint8_t> state(llama_state_seq_get_size(ctx.get(), 0));
    if (state.empty() ||
        llama_state_seq_get_data(ctx.get(), state.data(), state.size(), 0) != state.size()) {
        fprintf(stderr, "failed to capture mixed-tier state\n");
        return 1;
    }
    const auto before_import = llama_memory_vbr_state(mem, 0, 0);
    if (llama_state_seq_set_data(ctx.get(), state.data(), state.size(), 0) != state.size()) {
        fprintf(stderr, "failed to adopt mixed-tier state\n");
        return 1;
    }
    const auto after_import = llama_memory_vbr_state(mem, 0, 0);
    if (after_import.representation_epoch <= before_import.representation_epoch ||
        after_import.representation_epoch_swa <= before_import.representation_epoch_swa) {
        fprintf(stderr, "native mixed-tier import did not advance both child epochs\n");
        return 1;
    }

    fprintf(stderr, "PASS: per-child VBR representation epochs are monotone across "
            "degrade, clear/reset, degrade, and native import\n");
    return 0;
}
