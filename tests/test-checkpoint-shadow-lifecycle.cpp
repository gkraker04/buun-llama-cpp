#include "common.h"
#include "common-checkpoint-shadow.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-vbr-checkpoint.h"
#include "llama-vbr-checkpoint-compose.inc"
#include "llama-vbr-generation.h"
#include "llama-vbr-generation-types.h"
#include "llama-vbr-operation.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Test-only record factory (F12): the ONLY handle-minting path outside bridge capture, compiled
// solely into this test target — production libllama exports no factory symbol and this file
// never recompiles the production bridge TU. The struct definition below mirrors
// src/llama-vbr-checkpoint.cpp token-identically; CI compares the two definitions.
struct llama_vbr_checkpoint_shadow {
    vbr_checkpoint_generation_record record;
};

// Test seam into the one common holder bridge TU (defined in common-checkpoint-shadow.cpp).
void common_checkpoint_shadow_attach(common_prompt_checkpoint & ckpt, llama_vbr_checkpoint_shadow * handle);

static llama_vbr_checkpoint_shadow * make_handle(vbr_checkpoint_generation_record record) {
    return new llama_vbr_checkpoint_shadow{ std::move(record) };
}

static vbr_checkpoint_generation_record make_complete_record() {
    vbr_checkpoint_generation_record record;
    record.status = vbr_checkpoint_generation_status::complete;
    record.identity_policy_order_digest.fill(0x5a);

    vbr_checkpoint_generation_stream stream;
    stream.stream_index              = 0;
    stream.dependency_seq_id         = 0;
    stream.computation_frontier      = 128;
    stream.captured_dependency_count = 2;
    vbr_generation_page_ref page;
    page.page_index        = 0;
    page.captured_page_gen = 7;
    page.covered_mask[0]   = (uint64_t(1) << 10) | (uint64_t(1) << 20);
    stream.pages.push_back(page);

    vbr_checkpoint_generation_controller controller;
    controller.child_id          = 0;
    controller.dependency_mode   = checkpoint_child_dependency_mode::live_guarded;
    controller.pool_uuid         = { 0x1111, 0x2222 };
    controller.global_generation = 3;
    controller.streams.push_back(std::move(stream));

    record.controllers.push_back(std::move(controller));
    return record;
}

static common_prompt_checkpoint make_checkpoint() {
    common_prompt_checkpoint ckpt;
    ckpt.clear();
    ckpt.n_tokens  = 128;
    ckpt.id_task   = 42;
    ckpt.pos_min   = 0;
    ckpt.pos_max   = 127;
    ckpt.data_tgt  = { 1, 2, 3, 4 };
    ckpt.data_dft  = {};
    ckpt.accel.ring = { 9, 9 };
    return ckpt;
}

static bool run_holder_tests() {
    common_prompt_checkpoint a = make_checkpoint();
    if (common_checkpoint_shadow_complete(a) || a.size() != a.size_without_shadow()) {
        fprintf(stderr, "shadow-less checkpoint was not clean\n");
        return false;
    }

    common_checkpoint_shadow_attach(a, make_handle(make_complete_record()));
    if (!common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "attached complete record did not report complete\n");
        return false;
    }
    const size_t legacy_size = a.size_without_shadow();
    if (legacy_size != a.data_tgt.size() + a.data_dft.size() + a.accel.size() ||
            a.size() <= legacy_size) {
        fprintf(stderr, "size accounting: live size must be legacy + resident shadow bytes\n");
        return false;
    }

    // copy drops the shadow (fresh generation-unknown) and counts the drop; legacy fields copy
    const uint64_t drops_before = common_checkpoint_shadow_dropped_on_copy();
    common_prompt_checkpoint b(a);
    if (common_checkpoint_shadow_complete(b) || b.shadow != nullptr ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 1 ||
            b.n_tokens != a.n_tokens || b.id_task != a.id_task ||
            b.data_tgt != a.data_tgt || b.accel.ring != a.accel.ring ||
            b.size() != a.size_without_shadow()) {
        fprintf(stderr, "copy construction did not drop the shadow with counted legacy fidelity\n");
        return false;
    }
    common_prompt_checkpoint c = make_checkpoint();
    c = a;
    if (common_checkpoint_shadow_complete(c) ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 2) {
        fprintf(stderr, "copy assignment did not drop the shadow\n");
        return false;
    }
    // a still holds its shadow after being copied from
    if (!common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "copy source lost its shadow\n");
        return false;
    }

    // self-assignment is guarded: state (including the shadow) is unchanged, nothing counted
    auto * self = &a;
    a = *self;
    if (!common_checkpoint_shadow_complete(a) ||
            common_checkpoint_shadow_dropped_on_copy() != drops_before + 2) {
        fprintf(stderr, "self-assignment was not a guarded no-op\n");
        return false;
    }

    // moves transfer the shadow
    common_prompt_checkpoint d(std::move(a));
    if (!common_checkpoint_shadow_complete(d) || common_checkpoint_shadow_complete(a)) {
        fprintf(stderr, "move construction did not transfer the shadow\n");
        return false;
    }
    common_prompt_checkpoint e;
    e.clear();
    e = std::move(d);
    if (!common_checkpoint_shadow_complete(e) || common_checkpoint_shadow_complete(d)) {
        fprintf(stderr, "move assignment did not transfer the shadow\n");
        return false;
    }

    // clear destroys; double-clear is a no-op
    e.clear();
    e.clear();
    if (common_checkpoint_shadow_complete(e) || e.size() != 0) {
        fprintf(stderr, "clear did not destroy the shadow\n");
        return false;
    }

    // host-cache staging parity (F3): admission prices size_without_shadow() on the LIVE list,
    // and the invalidate-first copies then really are exactly that size
    std::vector<common_prompt_checkpoint> live;
    live.push_back(make_checkpoint());
    common_checkpoint_shadow_attach(live.back(), make_handle(make_complete_record()));
    live.push_back(make_checkpoint());
    size_t priced = 0;
    for (const auto & ckpt : live) {
        priced += ckpt.size_without_shadow();
    }
    std::vector<common_prompt_checkpoint> staged_copy = live;
    size_t staged_size = 0;
    for (const auto & ckpt : staged_copy) {
        staged_size += ckpt.size();
    }
    if (priced != staged_size) {
        fprintf(stderr, "staging admission price diverged from the staged copy's size\n");
        return false;
    }

    fprintf(stderr, "holder lifecycle rows PASS\n");
    return true;
}

static bool run_equality_tests() {
    common_prompt_checkpoint a = make_checkpoint();
    common_prompt_checkpoint b = make_checkpoint();
    common_checkpoint_shadow_attach(a, make_handle(make_complete_record()));
    common_checkpoint_shadow_attach(b, make_handle(make_complete_record()));
    if (!common_checkpoint_shadow_equal(a, b) || !common_checkpoint_shadow_equal(a, a)) {
        fprintf(stderr, "equal records did not compare equal (or not reflexive)\n");
        return false;
    }

    // a single covered-mask bit difference is detected
    auto flipped = make_complete_record();
    flipped.controllers[0].streams[0].pages[0].covered_mask[0] ^= uint64_t(1) << 20;
    common_prompt_checkpoint c = make_checkpoint();
    common_checkpoint_shadow_attach(c, make_handle(std::move(flipped)));
    if (common_checkpoint_shadow_equal(a, c)) {
        fprintf(stderr, "a covered-mask bit difference was not detected\n");
        return false;
    }

    // absence/unknown is availability, never equality
    common_prompt_checkpoint none = make_checkpoint();
    auto unknown_record   = make_complete_record();
    unknown_record.status = vbr_checkpoint_generation_status::generation_unknown;
    common_prompt_checkpoint unknown = make_checkpoint();
    common_checkpoint_shadow_attach(unknown, make_handle(std::move(unknown_record)));
    if (common_checkpoint_shadow_equal(a, none) || common_checkpoint_shadow_equal(none, none) ||
            common_checkpoint_shadow_equal(a, unknown) ||
            common_checkpoint_shadow_complete(unknown)) {
        fprintf(stderr, "absent/unknown records leaked into the equality relation\n");
        return false;
    }

    // §9.3 step 5 adopt: only shadow state moves
    common_prompt_checkpoint fresh = make_checkpoint();
    common_checkpoint_shadow_attach(fresh, make_handle(make_complete_record()));
    const std::vector<uint8_t> retained_payload = c.data_tgt;
    common_checkpoint_shadow_adopt(c, fresh);
    if (!common_checkpoint_shadow_equal(a, c) || common_checkpoint_shadow_complete(fresh) ||
            c.data_tgt != retained_payload) {
        fprintf(stderr, "adopt did not swap exactly the shadow record\n");
        return false;
    }

    fprintf(stderr, "equality/adopt rows PASS\n");
    return true;
}

// §11.1 rows 11/12: the refresh byte-proof over every retained payload.
static bool run_refresh_proof_tests() {
    common_prompt_checkpoint retained = make_checkpoint();
    common_checkpoint_shadow_attach(retained, make_handle(make_complete_record()));

    std::vector<uint8_t> cur_tgt  = retained.data_tgt;
    std::vector<uint8_t> cur_ring = retained.accel.ring;
    common_checkpoint_refresh_observation obs;
    obs.tgt             = &cur_tgt;
    obs.ring            = &cur_ring;
    obs.ring_applicable = true;

    // row 11: byte-identical reproduction of every retained payload proves the refresh
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::proven) {
        fprintf(stderr, "row 11: byte-identical state did not prove the refresh\n");
        return false;
    }

    // row 12: a mismatching reproduction refuses as nondeterminism evidence
    cur_tgt[0] ^= 0xff;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_byte_mismatch) {
        fprintf(stderr, "row 12: a target byte mismatch did not refuse the refresh\n");
        return false;
    }
    cur_tgt[0] ^= 0xff;

    // a retained payload that cannot be reproduced refuses (never mutates anything)
    obs.tgt = nullptr;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "an unreproducible target payload did not refuse the refresh\n");
        return false;
    }
    obs.tgt = &cur_tgt;

    // F1: applicable accelerator payloads are part of the proof — ring mismatch refuses
    cur_ring.push_back(1);
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_byte_mismatch) {
        fprintf(stderr, "F1: an accel.ring mismatch did not refuse the refresh\n");
        return false;
    }
    cur_ring.pop_back();

    // retained ring with no reproduction path refuses
    obs.ring            = nullptr;
    obs.ring_applicable = false;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "F1: a retained ring without a save path did not refuse\n");
        return false;
    }
    obs.ring            = &cur_ring;
    obs.ring_applicable = true;

    // an applicable component with nothing retained cannot be proven against nonempty state
    std::vector<uint8_t> cur_dft = { 7 };
    obs.dft            = &cur_dft;
    obs.dft_applicable = true;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "a draft context with no retained draft payload did not refuse\n");
        return false;
    }
    cur_dft.clear();
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::proven) {
        fprintf(stderr, "an empty applicable component was not vacuous\n");
        return false;
    }

    // F1 (verify round): an APPLICABLE component whose observation is null must refuse even
    // when nothing is retained (e.g. the spec-state getter failed under can_speculate)
    obs.spec_applicable = true;
    obs.spec            = nullptr;
    if (common_checkpoint_shadow_refresh_proof(retained, obs) !=
            common_checkpoint_refresh_verdict::refused_cannot_reproduce) {
        fprintf(stderr, "F1: a null observation for an applicable empty component was proven\n");
        return false;
    }
    obs.spec_applicable = false;

    fprintf(stderr, "refresh byte-proof rows PASS\n");
    return true;
}

static bool run_bridge_boundary_tests() {
    // null memory / null frontier fail closed with a closed reason, never a crash
    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(nullptr, 0, nullptr, &result);
    if (result.handle != nullptr ||
            result.reason != vbr_checkpoint_capture_reason::invalid_arguments) {
        fprintf(stderr, "null capture arguments were not refused\n");
        return false;
    }
    llama_vbr_checkpoint_shadow_capture(nullptr, 0, nullptr, nullptr);

    if (llama_vbr_checkpoint_shadow_status(nullptr) !=
            vbr_checkpoint_generation_status::generation_unknown ||
            llama_vbr_checkpoint_shadow_size(nullptr) != 0 ||
            llama_vbr_checkpoint_shadow_equal(nullptr, nullptr)) {
        fprintf(stderr, "null handles were not generation-unknown/zero/unequal\n");
        return false;
    }
    llama_vbr_checkpoint_shadow_free(nullptr);

    for (const auto reason : {
                 vbr_checkpoint_capture_reason::ok,
                 vbr_checkpoint_capture_reason::not_applicable,
                 vbr_checkpoint_capture_reason::invalid_arguments,
                 vbr_checkpoint_capture_reason::unarmed_live_covered,
                 vbr_checkpoint_capture_reason::child_capture_failed,
                 vbr_checkpoint_capture_reason::oracle_mismatch,
                 vbr_checkpoint_capture_reason::internal_error,
         }) {
        const char * name = llama_vbr_checkpoint_shadow_reason_name(reason);
        if (name == nullptr || strlen(name) == 0) {
            fprintf(stderr, "capture reason without a name\n");
            return false;
        }
    }

    fprintf(stderr, "bridge boundary rows PASS\n");
    return true;
}

// --- C2-P9 rung-1 rows (f)-(h): real armed iSWA composite capture (GPU fixture) ---------------
// Run with a gemma-4 iSWA model argument on the dorei gate box; the no-arg ctest invocation
// runs only the CPU rows above.

// Same-named friend as the epoch test (each test binary defines its own shim): tracker access
// for row (g)'s !stable() and shadow-unavailable refusals.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }
    static vbr_generation_tracker * tracker_mut(llama_kv_cache * kv) {
        return kv->vbr_generation_tracker_mut();
    }
};

static bool gpu_decode_at(llama_context * ctx, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, 1, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (ok) {
        // decode is asynchronous; the fixture inspects/mutates memory next, so fence to the
        // completed-state boundary (this is also where submitted evidence commits)
        llama_synchronize(ctx);
    }
    return ok;
}

static vbr_checkpoint_frontier_fields gpu_frontier(int64_t n_past) {
    static const std::string exec_id    = "shadow-fixture-exec";
    static const std::string adapter_id = "shadow-fixture-adapter";
    static const std::string media_id   = "shadow-fixture-media";
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

static llama_vbr_checkpoint_shadow * gpu_capture(llama_memory_t mem, int64_t n_past,
                                                 vbr_checkpoint_capture_reason & reason) {
    const auto frontier = gpu_frontier(n_past);
    llama_vbr_checkpoint_capture_result result;
    llama_vbr_checkpoint_shadow_capture(mem, 0, &frontier, &result);
    reason = result.reason;
    return result.handle;
}

static int run_gpu_fixture_rows(const char * model_path) {
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fprintf(stderr, "failed to load model %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                 = 128;
    cparams.n_batch               = 32;
    cparams.n_ubatch              = 32;
    cparams.n_seq_max             = 1;
    cparams.n_threads             = 2;
    cparams.n_threads_batch       = 2;
    cparams.type_k                = GGML_TYPE_F16;
    cparams.type_v                = GGML_TYPE_F16;
    cparams.flash_attn_type       = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.vbr_dynamic           = true;
    cparams.vbr_budget_explicit   = true;
    cparams.vbr_vram_budget_bytes = 64ull * 1024 * 1024;

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fprintf(stderr, "failed to create CUDA VBR context\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx.get());

    llama_kv_cache * base = nullptr;
    llama_kv_cache * swa  = nullptr;
    if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        base = iswa->get_base();
        swa  = iswa->get_swa();
    } else if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        base = hybrid->get_mem_attn()->get_base();
        swa  = hybrid->get_mem_attn()->get_swa();
    } else {
        fprintf(stderr, "fixture did not create an iSWA attention cache\n");
        return 1;
    }
    if (!llama_kv_cache_vbr_epoch_test::active(base) ||
            !llama_kv_cache_vbr_epoch_test::active(swa)) {
        fprintf(stderr, "SKIP: loaded GPU backend does not provide VBR VMM for both children\n");
        return 0;
    }

    int64_t n_past = 0;
    for (; n_past < 4; ++n_past) {
        if (!gpu_decode_at(ctx.get(), (llama_pos) n_past)) {
            fprintf(stderr, "seed decode failed at pos %" PRId64 "\n", n_past);
            return 1;
        }
    }

    // row (f): composite bridge capture on the real armed iSWA pair — complete record, child
    // order [base, swa], serializer-derived modes, armed pool identities, shared digest helper
    vbr_checkpoint_capture_reason reason;
    llama_vbr_checkpoint_shadow * first = gpu_capture(mem, n_past, reason);
    if (first == nullptr || reason != vbr_checkpoint_capture_reason::ok) {
        fprintf(stderr, "row f: armed iSWA capture failed (reason=%s)\n",
                llama_vbr_checkpoint_shadow_reason_name(reason));
        return 1;
    }
    {
        const auto & record = first->record;
        if (record.status != vbr_checkpoint_generation_status::complete ||
                record.controllers.size() != 2 ||
                record.controllers[0].child_id != 0 ||
                record.controllers[0].dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
                record.controllers[1].child_id != 1 ||
                record.controllers[1].dependency_mode != checkpoint_child_dependency_mode::payload_complete ||
                !record.controllers[1].streams.empty()) {
            fprintf(stderr, "row f: composite child order/modes did not match [base=live_guarded, swa=payload_complete]\n");
            return 1;
        }
        if (record.controllers[0].pool_uuid != base->vbr_pool_id() ||
                record.controllers[1].pool_uuid != swa->vbr_pool_id() ||
                record.controllers[1].pool_uuid.hi == 0) {
            fprintf(stderr, "row f: armed children did not record their exact pool identities\n");
            return 1;
        }
        std::vector<vbr_checkpoint_child_policy> policy;
        for (const auto & controller : record.controllers) {
            policy.push_back({ controller.child_id, controller.dependency_mode, controller.pool_uuid });
        }
        if (vbr_checkpoint_identity_digest(gpu_frontier(n_past), policy) !=
                record.identity_policy_order_digest) {
            fprintf(stderr, "row f: record digest diverged from the shared digest helper\n");
            return 1;
        }
    }

    // row (h): no-op recapture is equal; a real mutation (frontier advance, and a same-frontier
    // destructive trim) makes captures unequal
    llama_vbr_checkpoint_shadow * again = gpu_capture(mem, n_past, reason);
    if (again == nullptr || !llama_vbr_checkpoint_shadow_equal(first, again)) {
        fprintf(stderr, "row h: no-op recapture was not equal\n");
        return 1;
    }
    if (!gpu_decode_at(ctx.get(), (llama_pos) n_past)) {
        fprintf(stderr, "row h: mutation decode failed\n");
        return 1;
    }
    ++n_past;
    llama_vbr_checkpoint_shadow * moved = gpu_capture(mem, n_past, reason);
    if (moved == nullptr || llama_vbr_checkpoint_shadow_equal(first, moved)) {
        fprintf(stderr, "row h: a frontier-advancing decode did not change the record\n");
        return 1;
    }
    if (!llama_memory_seq_rm(mem, 0, (llama_pos) (n_past - 1), -1)) {
        fprintf(stderr, "row h: tail trim was rejected\n");
        return 1;
    }
    --n_past;
    llama_vbr_checkpoint_shadow * trimmed = gpu_capture(mem, n_past, reason);
    if (trimmed == nullptr || llama_vbr_checkpoint_shadow_equal(first, trimmed)) {
        fprintf(stderr, "row h: a same-shape destructive trim did not change the record\n");
        return 1;
    }

    // row (g): capture refuses while a child is mid-mutation (!stable()) and while its shadow
    // is latched unavailable
    {
        auto * tracker = llama_kv_cache_vbr_epoch_test::tracker_mut(base);
        if (tracker == nullptr) {
            fprintf(stderr, "row g: base tracker unavailable\n");
            return 1;
        }
        vbr_scoped_operation op(vbr_mutation_binding(
                vbr_operation_kind::decode, 0, 0,
                std::numeric_limits<llama_pos>::max(),
                vbr_operation_class::ordinary_decode,
                tracker->pool_identity().hi, tracker->pool_identity().lo));
        if (!op.id()) {
            fprintf(stderr, "row g: test operation failed to register\n");
            return 1;
        }
        {
            auto event = tracker->begin_event(
                    vbr_mutation_registrant::apply_ubatch_append,
                    vbr_operation_class::ordinary_decode,
                    0,
                    vbr_generation_stamp_kind::dependency,
                    op.id());
            if (!event) {
                fprintf(stderr, "row g: could not open a mid-mutation event\n");
                return 1;
            }
            llama_vbr_checkpoint_shadow * refused = gpu_capture(mem, n_past, reason);
            if (refused != nullptr ||
                    reason != vbr_checkpoint_capture_reason::child_capture_failed) {
                fprintf(stderr, "row g: capture was not refused mid-mutation (reason=%s)\n",
                        llama_vbr_checkpoint_shadow_reason_name(reason));
                llama_vbr_checkpoint_shadow_free(refused);
                return 1;
            }
            if (!event.finish()) {
                fprintf(stderr, "row g: mid-mutation event did not close\n");
                return 1;
            }
        }
        tracker->set_shadow_unavailable();
        llama_vbr_checkpoint_shadow * latched = gpu_capture(mem, n_past, reason);
        if (latched != nullptr || reason != vbr_checkpoint_capture_reason::child_capture_failed) {
            fprintf(stderr, "row g: capture was not refused while shadow-unavailable (reason=%s)\n",
                    llama_vbr_checkpoint_shadow_reason_name(reason));
            llama_vbr_checkpoint_shadow_free(latched);
            return 1;
        }
    }

    llama_vbr_checkpoint_shadow_free(first);
    llama_vbr_checkpoint_shadow_free(again);
    llama_vbr_checkpoint_shadow_free(moved);
    llama_vbr_checkpoint_shadow_free(trimmed);

    printf("GPU armed-iSWA composite rows (f)-(h) PASS\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (!run_holder_tests() || !run_equality_tests() || !run_refresh_proof_tests() ||
            !run_bridge_boundary_tests()) {
        return 1;
    }
    printf("checkpoint shadow lifecycle PASS\n");
    if (argc == 2) {
        return run_gpu_fixture_rows(argv[1]);
    }
    return 0;
}
