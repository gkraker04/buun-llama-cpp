#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "llama-vbr-generation.h"

#include "ggml-vbr.h" // backend interface for turbo KV / dynamic VBR (resolved at init, never linked)
#include "llama-vram-ledger.h" // co-tenancy peer claim/marker types (P2)

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <exception>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

// Dynamic VBR (S3): one step of the measured decode-time degrade price order —
// knock (layer il, K/V side) down to `tier` (a vbr_tier index, see llama-kv-cache.cpp).
struct vbr_degrade_step {
    uint8_t il;
    uint8_t is_v;
    uint8_t tier;
};

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share = nullptr,
        const llama_memory_vbr_params & vbr = {});

    ~llama_kv_cache(); // frees the VBR VMM pool (if any); = default otherwise

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;
    bool can_seq_rm_partial() const override { return true; }

    void breathe() override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    // monotone counter of in-place VBR tier flips — graph reuse fences on it.
    // A share-linked cache (mem_other) views the owner's tensors, so its graphs must
    // fence on the OWNER's flips: delegate to the source cache (shared-KV drafters,
    // e.g. the gemma4 MTP assistant, follow the target's VBR tier changes this way).
    uint64_t vbr_tier_epoch() const { return other ? other->vbr_tier_epoch() : vbr_tier_epoch_; }

    // Checkpoint-facing semantic counter: unlike the graph tier fence, this also covers
    // clear/reset/import. It never resets, so cursor rewind cannot create an ABA.
    uint64_t vbr_representation_epoch() const {
        return other ? other->vbr_representation_epoch() : vbr_representation_epoch_;
    }

    // Revision-9 A1 adapters over the cache's canonical dependency index. These are
    // read-only shadow helpers: A2 will compose child controllers and route selectors
    // through checkpoint_vbr_eligibility(), but A1 changes no read authority.
    bool vbr_generation_capture_live_guarded(
            uint32_t child_id,
            llama_seq_id seq_id,
            llama_pos computation_frontier,
            vbr_checkpoint_generation_controller & output) const;
    bool vbr_generation_live_guarded_view(
            uint32_t child_id,
            llama_seq_id seq_id,
            llama_pos computation_frontier,
            vbr_generation_live_controller_view & output) const;

    // Debug-oracle trust domain (env-gated callers only): canonical per-cell observations built
    // by a direct cell scan, deliberately WITHOUT the ownership index, so the independent
    // reconstruction can catch an index that drifted from the canonical cells.
    bool vbr_generation_oracle_observations(
            llama_seq_id seq_id,
            std::vector<struct vbr_generation_oracle_cell> & output) const;

    // effective bits/value of this cache at the CURRENT tensor types (llama_memory_i)
    double kv_bpv() const override;

    llama_memory_vbr_state_data memory_vbr_state(llama_seq_id seq_id, uint32_t n_tokens_extra) const override;
    bool vbr_operation_armed() const override;
    // C4 boundary service: true while this cache's tracker is latched unavailable or its pool
    // has unresolved recovery-ring work. The update context reports an update NEEDED in this
    // state so the quarantine drain + monotone re-arm in update() actually run at quiet decode
    // boundaries (a NO_UPDATE short-circuit would starve recovery until an unrelated shift).
    bool vbr_recovery_service_pending() const;
    bool vbr_retier_freeze_begin(const char * owner, vbr_operation_id operation_id) override;
    void vbr_retier_freeze_end(const char * owner, vbr_operation_id operation_id) override;
    llama_memory_vbr_preflight_data vbr_retier_preflight(uint32_t n_tokens_extra) const override;
    bool vbr_retier_freeze_active() const {
        return other ? other->vbr_retier_freeze_active() : vbr_retier_freeze_depth_ > 0;
    }
    // totals for cross-cache aggregation (iSWA weights its children by stored values)
    void   kv_bpv_accum(double & bits, double & vals) const;

    // co-tenancy: bytes a demand-driven shed could free on `device` — max shed over the
    // remaining f16->t8 band, net of the projected dequant-scratch growth it would cost.
    // 0 when the band is spent/absent, the order is a custom override, or no VMM pool
    // lives on the device. Memoized; safe to call between boundaries (marker writes).
    size_t vbr_shed_available(int device) const;

    void vbr_cotenancy_accum(uint64_t & decrement, uint32_t & grants,
                             uint64_t & offer, uint64_t & pending) const override;

    double memory_vbr_floor_bits_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;
    double memory_vbr_scratch_bytes_per_token(ggml_type entry_k, ggml_type entry_v, double floor_bpv) override;

    // shared ladder-sim primitives: seed a per-(layer,side) type view + per-step
    // applicability under vbr_degrade_next's exact skip rules (see impl comment) — the
    // floor sim, the bpv-if-degraded walk and the co-tenancy offer all ride these
    void vbr_sim_seed(std::vector<ggml_type> & sim, bool pooled_only,
                      ggml_type entry_k, ggml_type entry_v,
                      double * sum_bits, int64_t * sum_vals, size_t * n_pinned) const;
    bool vbr_sim_step(const std::vector<ggml_type> & sim, size_t i,
                      size_t & slot, const ggml_tensor *& t, ggml_type & type_B) const;

    // shared floor-walk core (runtime clamp + fit capacity math), see impl comment
    struct vbr_floor_sim_result {
        size_t clamp_step     = 0;     // steps applied before the clamp (== order size if unclamped)
        size_t n_pinned       = 0;
        double next_bpv       = 0.0;   // aggregate the clamping step would have produced
        double bits_per_token = 0.0;   // end-state KV bits per token (0 = no units)
        std::vector<ggml_type> end_types; // [layers*2] end-state tier, GGML_TYPE_COUNT = absent
    };
    vbr_floor_sim_result vbr_floor_sim(double floor_bpv, bool pooled_only,
            ggml_type entry_k = GGML_TYPE_COUNT, ggml_type entry_v = GGML_TYPE_COUNT) const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;


    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(
            const slot_info & sinfo, const llama_ubatch & ubatch, bool generation_commit = true);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    // Dynamic VBR (M2): per-(layer,side) descriptor over the shared KV pool buffer. Tier is NOT
    // mirrored here — the cache tensor (layers[ikv].k/.v) is the single source of truth for the
    // TYPE; a degrade flips the tensor and this descriptor only tracks placement. Cell WIDTH is
    // per-pool: `t` is the tensor instance whose bytes live in this pool — the cache tensor
    // itself under -sm layer, or this device's shard of it under -sm tensor (same name, same
    // type, ne0 = this device's slice of the head*dim axis). All per-pool byte math (row sizes,
    // slots, stash sizing) derives from `t`, never from the canonical layers[] tensor.
    struct vbr_extent {
        ggml_tensor * t    = nullptr;           // pool-local tensor instance (canonical or shard)
        size_t    byte_off = 0;                 // offset of this tensor's data within the pool buffer
        ggml_type type0    = GGML_TYPE_COUNT;   // ENTRY tier (immutable; full-clear reset target)
        size_t    stash_off   = 0;              // offset into the f16 sink-stash buffer
        uint32_t  stash_valid = 0;              // rows captured (0 = not yet)
        // promote transcodes with live rows since the last full reset: each one re-encodes the
        // aged rows from their degraded recon, so error compounds per hop — cap bounds the damage
        uint8_t   promote_hops = 0;
    };
    // Multi-GPU: one vbr_pool per KV-hosting device buffer. Extent vectors stay indexed by ikv in
    // EVERY pool; only entries whose tensor (or tensor shard) lives in that pool's buffer are
    // populated (e.t == nullptr elsewhere). Under -sm layer the populated sets are DISJOINT (each
    // device owns whole layers); under -sm tensor every pool holds a per-device SHARD of every
    // (layer,side), so a tier flip transcodes in every pool that has a nonzero extent for it.
    // With a single GPU there is exactly one pool and all logic reduces to the previous
    // single-pool controller bit-for-bit.
    struct vbr_pool {
        ggml_backend_buffer_t buf     = nullptr; // non-owning (lives in ctxs_bufs); one KV buffer
        char *                base    = nullptr; // ggml_backend_buffer_get_base(buf)
        size_t                size    = 0;        // total pool bytes
        size_t                used    = 0;        // high-water of placed extents (log-only)
        size_t                budget  = 0;         // per-pool share of vbr_budget_bytes_ (VA-size proportional)
        size_t                budget_base = 0;      // init-armed/fallback share: the re-derivation floor
        // vbr_budget_eff memo: one live free-VRAM query per pool per boundary (the degrade loop
        // and promote hysteresis both consult it repeatedly within one boundary)
        mutable uint64_t      budget_eff_stamp = ~0ull;
        mutable size_t        budget_eff_cache = 0;
        std::vector<vbr_extent> k;                // indexed by kv-cache layer id (ikv)
        std::vector<vbr_extent> v;
        // backend VBR vtable that owns this pool's device (resolved from the buffer type's
        // registry at init; a pool only exists if the backend exports it)
        const ggml_vbr_backend_iface * be = nullptr;
        // S2 (option C): VMM-backed pool — per-tensor fixed VA slots, physical pages mapped on
        // demand. When set, `size` is the VA reservation (not physical); each extent's byte_off is
        // page-aligned so tensor-tail unmaps never straddle a neighbor's pages.
        struct ggml_vbr_vmm_pool * vmm = nullptr;
        uint32_t wm_cells    = 0;                 // cells already backed for every extent
        int      device      = -1;                // backend device ordinal backing the pool
        size_t   gran        = 0;                 // page granularity
        size_t   mapped_base = 0;                 // bytes mapped up front (rotation matrices)
        // #88 scratch-reserve memo: widest f16 row per dequant-active side, valid while no tier
        // flips (keyed on vbr_tier_epoch_; ~0 forces the first compute)
        uint64_t scratch_rows_epoch = ~0ull;
        size_t   scratch_k_row      = 0;
        size_t   scratch_v_row      = 0;
        size_t   scratch_k_reserved = 0; // largest successful backend-global reserve requested here
        size_t   scratch_v_reserved = 0;
        // co-tenancy (P2): PCI bus id (resolved once from the backend device; empty = none)
        // and the summed unamortized grant decrement vbr_budget_eff subtracts
        std::string busid;
        mutable size_t grant_decrement = 0;
        // per-device transcode side stream (lazy) + S5 overlap state: transcodes run async on
        // backend's stream; the next decode graph GPU-waits via the armed per-device fence
        // (be->fence_arm). Tail pages a transcode may still READ (rA extent >
        // kept rB extent) can only be unmapped once it finishes — queue them and flush at the
        // next decode boundary, when the wave is long done.
        ggml_backend_t backend      = nullptr;
        bool           wave_pending = false;      // async GPU work enqueued, fence not yet armed
        std::vector<std::pair<size_t, size_t>> unmap_deferred; // {pool byte_off, len}
        // f16 sink-stash (VBR_STASH_ROWS env; 0 = off): pristine first-degrade snapshot of the
        // first N rows per tensor — permanently-hot sink rows re-encode from it at every hop
        ggml_backend_buffer_t stash_buf = nullptr;
    };
    void vbr_vmm_ensure_mapped(); // grow physical backing to the current cell watermark
    bool vbr_vmm_try_map(uint32_t wm); // same, recoverable: false on physical exhaustion

    // S3/S4: decode-time degrade controller (VMM mode only). The price order and its cursor stay
    // GLOBAL (layer-global price order); each step resolves the pool that owns its tensor.
    llama_memory_vbr_params vbr_params_;              // API/CLI inputs (ctor copy; env can override)
    // bumped on every in-place tier flip (degrade/promote/full reset). Graph reuse must be
    // fenced on it: a reused graph carries the OLD type/strides baked into its K/V views, and
    // a free-VRAM-clamp wave (or a promote map-retry) can flip tiers MID-band where the n_kv
    // shape check alone would still allow reuse.
    uint64_t vbr_tier_epoch_ = 0;
    // Bumped once per representation-changing operation (degrade, promote, occupied-cell reuse,
    // clear/full-reset, native state import). Never derive this from or reset it with the cursor.
    uint64_t vbr_representation_epoch_ = 0;
    // Revision-9 A1 shadow generations. Allocated only for a construction-final armed VBR
    // controller; aliases delegate to their canonical owner and inert caches allocate nothing.
    // No current checkpoint read consults this store until the A2 four-way ratchet lands.
    std::unique_ptr<vbr_generation_tracker> vbr_generation_;
    // A2 dual-view ownership index: updated in the SAME registrant transactions that stamp the
    // tracker; capture consumes rank_below for scan-free exact dependency cardinality.
    std::unique_ptr<vbr_ownership_index>    vbr_ownership_;
    std::vector<vbr_degrade_step> vbr_degrade_order_; // global price order, F16->t8 band first
    size_t         vbr_degrade_cursor_ = 0;
    size_t         vbr_budget_bytes_   = 0;           // global mapped-physical budget; 0 = no trigger
    uint32_t       vbr_stash_rows_     = 0;           // sink-stash rows per (layer,side); 0 = off
    // --vbr-floor (env VBR_MIN_BITS): first order step the aggregate bits/value floor forbids;
    // the cursor never advances past it (default = order size, i.e. unclamped)
    size_t vbr_degrade_limit_ = (size_t) -1;
    // co-tenancy: end of the leading f16->t8 band of the order (demand sheds stop here);
    // 0 = no band (custom VBR_DEGRADE_ORDER carries no band guarantee -> demand shed off)
    size_t t8_band_end_ = 0;
    // peer-yield consent bound (buun 2026-07-20, explicit-floor-as-consent): a TYPED
    // --vbr-floor (flag or VBR_MIN_BITS env) consents demand sheds down to the floor —
    // the ledger is per-uid, so the demander is the same human who typed it. A defaulted
    // floor keeps the conservative restorable band. 0 = demand shedding disabled.
    size_t vbr_demand_limit() const {
        if (t8_band_end_ == 0) {
            return 0;
        }
        return vbr_floor_typed_ ? vbr_degrade_limit_
                                : std::min(vbr_degrade_limit_, t8_band_end_);
    }
    bool vbr_floor_typed_ = false;
    // vbr_shed_available memo: per-pool freed-bytes projection, keyed on (tier epoch,
    // watermark padded to the 256-cell quantum) — budget is deliberately NOT an input
    mutable uint64_t            shed_avail_epoch_ = ~0ull;
    mutable uint32_t            shed_avail_wm_    = 0;
    mutable std::vector<size_t> shed_avail_pool_;

    // ---- co-tenancy donor state (P2) ----
    // grant rows: private in-memory liabilities recording a demand-shed's decrement,
    // keyed (pid, starttime, ver) with the demanded device's busid; one row per pool the
    // wave freed bytes in. Collateral rows (lockstep frees on non-demanded devices) carry
    // the full decrement until the lift event (delta_i = 0 — this also keeps the promote
    // cursor frozen so a promote cannot undo a lockstep shed).
    struct vbr_grant_row {
        std::string busid;      // device the demand named (claim file key)
        int32_t     pid;
        uint64_t    starttime;
        uint64_t    ver;
        size_t      pool_idx;
        uint64_t    bytes;
        uint64_t    bytes_now_at_grant;
        bool        collateral;
    };
    std::vector<vbr_grant_row> vbr_grants_;
    // reader-side heartbeat aging (shared llama_vram_hb_obs; claim keys "busid-pid",
    // marker keys "m-busid-pid")
    std::map<std::string, llama_vram_hb_obs> vbr_claim_obs_;
    // ledger scan pacing: dir-mtime pre-check baseline + last full-scan clock
    uint64_t vbr_ledger_mtime_  = 0;
    uint64_t vbr_last_scan_ns_  = 0;
    bool     vbr_ledger_force_  = false; // pre-check hit: run the full controller path
    // last published marker fields per busid (republish = rename only on change)
    std::map<std::string, std::pair<uint64_t, uint64_t>> vbr_marker_pub_; // {shed_avail, grant_pending}
    // our marker's created_ts (donor-rank input; 0 until first publish) and the per-device
    // granted-but-not-yet-flushed bytes (set at shed commit, cleared at the first scan
    // event after that wave's deferred unmaps flush)
    uint64_t vbr_marker_created_ts_ = 0;
    std::map<std::string, uint64_t> vbr_grant_pending_;

    // ---- P3 presence census ----
    // effective N_live per busid (self + live peer markers). Arrivals count immediately
    // (growing headroom is the safe direction); departures only after the raw count holds
    // for DEBOUNCE consecutive scan events (a GC'd marker of a crashed-and-restarting peer
    // must not flap the budgets). Promotes are presence-quiet gated on the change scan.
    struct vbr_presence { uint32_t cur = 0, raw = 0, stable = 0; };
    std::map<std::string, vbr_presence> vbr_presence_;
    uint32_t vbr_scan_events_          = 0;
    uint32_t vbr_nlive_change_scan_    = 0;
    uint32_t vbr_pool_n_live(const vbr_pool & p) const;
    bool     vbr_presence_quiet() const; // promote gate: no N_live change within DEBOUNCE scans

    // ---- P3 runtime-growth demand (idle-donor only) ----
    // a resident that spent its own consent window and is still over budget publishes a
    // phase=runtime claim; only donors idle >= IDLE honor it (active-vs-active residents
    // self-serve via their own ladders). CLEAR is demander-owned: the first boundary
    // where the recomputed shortage <= 0 unlinks (the donors' lift signal).
    uint64_t vbr_runtime_ver_      = 0;
    uint64_t vbr_last_prepare_ns_  = 0; // decode-based idleness input (ticks never update it)
    std::set<std::string> vbr_runtime_live_; // busids with a live runtime claim
    std::vector<size_t> vbr_pre_deficit_;    // per-pool pre-own-loop deficit (the honest ask)
    void vbr_runtime_demand_update(uint32_t wm_next, bool was_over);

    void   vbr_ledger_precheck();                 // every boundary, outside the stable gate
    void   vbr_ledger_scan_service(uint32_t wm_next); // composes the four phases below
    void   vbr_presence_census(const std::vector<llama_vram_peer_marker> & peers);
    bool   vbr_grants_upkeep(const std::vector<llama_vram_peer_claim> & claims, uint64_t now);
    bool   vbr_service_demands(const std::vector<llama_vram_peer_claim> & claims,
                               const std::vector<llama_vram_peer_marker> & peers,
                               uint64_t now, uint32_t wm_next);
    void   vbr_grant_pending_clear();
    void   vbr_markers_publish(uint64_t now);
    void   vbr_maybe_promote(uint32_t wm_next); // gated promote step (boundary + tick)
    void   vbr_arm_wave_fences();               // arm fences for queued transcode waves
    vbr_pool * vbr_find_pool(const std::string & busid);
    void   vbr_apply_grant_decrements();          // recompute per-pool sums, bust memos
    size_t vbr_total_grant_decrement() const;     // promote freeze gate
    const std::string & vbr_pool_busid(vbr_pool & p) const;

public:
    // co-tenancy: exactly one cache per memory tree runs the ledger protocol; composite
    // parents (iSWA) demote all but one child and hand the owner a sibling pointer so the
    // tree's offer is the SUM and demand targets split by offer weight. Non-owners keep
    // every local mechanism (budget, band, waves, grants on their own pools) but never
    // scan, serve, or publish themselves.
    void vbr_set_ledger_owner(bool owner) { vbr_ledger_owner_ = owner; }
    void vbr_set_ledger_sibling(llama_kv_cache * sib) { vbr_ledger_sibling_ = sib; }
    size_t vbr_execute_shed(const llama_vram_peer_claim & c, uint64_t target, uint32_t wm_next);
private:
    bool vbr_ledger_owner_ = true;
    llama_kv_cache * vbr_ledger_sibling_ = nullptr;
    size_t vbr_floor_cost_bytes_ = 0;                 // page-exact cost of the floor layout at full
                                                      // kv_size (fallback budget in dynamic mode)
    bool   vbr_budget_warned_ = false;                // budget-unmeetable warning fired (terminal)
    // prepare() boundaries since the last applied degrade step — promote cooldown basis
    // (deterministic, unlike wall time); promotes wait for a quiet window after any degrade
    uint32_t vbr_quiet_boundaries_ = 0;
    // auto-budget runtime re-derivation (explicit budgets never move): boundary counter (the
    // FIRST boundary is skipped — lazy cuBLAS/CUDA-graph pools have not allocated yet and free
    // overstates reality) + resolved growth headroom
    uint64_t vbr_boundary_count_   = 0;
    size_t   vbr_growth_headroom_  = 0;
    bool     vbr_budget_explicit_  = false;
    // WS-0 (P1) deterministic freeze mode — env VBR_FREEZE, TEST/GATING ONLY. Neutralizes the two
    // live-VRAM / co-tenancy inputs that make the tier schedule irreproducible run-to-run: the
    // vbr_budget_eff clamp (live cudaMemGetInfo) and the ledger scan/precheck + wall-clock gates,
    // so the schedule becomes a pure function of the fixed budget + occupancy. Requires an explicit
    // VBR_BUDGET_MIB (else vbr_pool_reach re-derivation, which is !explicit-gated, is not frozen).
    // OFF => every gated branch runs verbatim: a freeze-off build is bit-identical to a pre-freeze
    // build (the P0 base-numerics ratchet). Never a production degrade-policy lever.
    bool     vbr_freeze_           = false;
    // WS-6: production scoped freeze of representation mutation. Orthogonal to WS-0's
    // deterministic-input freeze above: nesting never changes the ledger/presence machinery.
    struct vbr_retier_freeze_frame {
        vbr_operation_id operation_id = {};
        uint64_t started_ns = 0;
    };
    static constexpr size_t VBR_RETIER_FREEZE_MAX_DEPTH = 64;
    std::array<vbr_retier_freeze_frame, VBR_RETIER_FREEZE_MAX_DEPTH> vbr_retier_freeze_stack_ = {};
    uint32_t vbr_retier_freeze_depth_       = 0;
    uint64_t vbr_retier_freeze_enters_      = 0;
    uint64_t vbr_retier_freeze_exits_       = 0;
    uint64_t vbr_retier_deferred_decisions_ = 0;
    uint64_t vbr_retier_reconciles_         = 0;
    uint64_t vbr_retier_outer_deferred_base_ = 0;
    bool     vbr_retier_reconcile_pending_  = false;
    bool     vbr_retier_defer(const char * decision);
    bool     vbr_retier_take_reconcile(const char * boundary);
    // WS-0 (P1) schedule-trace recorder — env VBR_TRACE=<path>, TEST/GATING ONLY. One line per
    // boundary: phase, boundary#, degrade cursor, tier-vector FNV digest, watermark, used cells,
    // mapped bytes. The L2 null arm needs two runs proven schedule-IDENTICAL (not just same output);
    // this makes the schedule diffable and localizes the first divergent boundary. null => no-op.
    // RAII (Sol review F5): a throwing ctor after the open still closes the handle during unwinding.
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> vbr_trace_fp_{nullptr, &std::fclose};
    void     vbr_trace_emit(const char * phase, uint32_t wm, uint32_t used);
    // what this pool's device can give it right now: device_share x (mapped + free - headroom),
    // 64 MiB-quantized. Shared by the init-time auto-budget arm (fit-less modes, e.g.
    // SPLIT_MODE_TENSOR) and the periodic re-derivation.
    size_t   vbr_pool_reach(const vbr_pool & p) const;
    // Fast-path stability tracking: skip per-batch VBR bookkeeping when settled (avoids ~1ms/token)
    uint32_t vbr_last_used_        = 0;   // observed cell count last prepare() pass
    void     vbr_rederive_budget();
    // sink-stash staleness guard: set when any cell below stash_rows is freed (its content can be
    // rewritten by another request; injecting the old snapshot would corrupt the new rows)
    bool   vbr_stash_dirty_   = false;
    void     vbr_full_reset();                        // cache empty: undo every degrade (lossless)
    void     vbr_representation_changed();             // monotone checkpoint change detector
    vbr_generation_tracker *       vbr_generation_tracker_mut();
    const vbr_generation_tracker * vbr_generation_tracker_get() const;
    static bool vbr_generation_cell_has_seq_cb(
            const void * context, uint32_t stream, uint32_t cell, llama_seq_id seq_id);
    static llama_pos vbr_generation_cell_pos_cb(
            const void * context, uint32_t stream, uint32_t cell);

    // A2 explicit operation binding (§7.2): every mutation entry point opens ONE scope carrying
    // its authenticated multi-target manifest. The scope registers the operation and — for
    // provenance-bearing mutations — reserves the recovery record BEFORE any mutation; damage
    // extents reserve lazily per SELECTED target at the first destructive stamp (P1v2).
    // Events minted while the scope is open cite its operation id. Close follows the
    // per-family commit-boundary table (design Rev 5.1): synchronous families commit at scope
    // end; deferred families transfer everything to the pending owner via detach_deferred().
    // A2 (review F3): decode operations stay open past apply_ubatch — closed only when the
    // decode outcome is known. One entry per in-flight committed ubatch.
  public:
    // P3v2 (v6): FIXED parent-declared participant slots with a sealed-registration phase.
    // The parent declares every armed child before the first apply; each child claims its
    // slot in its scope constructor (before mutation), and the slot reports terminal EXACTLY
    // once — setup/decode/submit failure, or synchronize-time delivery. Detach transfers the
    // still-OPEN token to pending ownership (never terminal). seal() folds the wrapper
    // result, fails any never-claimed declared slot, and seals registration; only
    // `sealed && every declared slot terminal` closes the root, failure dominating. No
    // dynamic remaining++ anywhere — the v5 premature-close class is unrepresentable.
    // (Public: the iSWA wrapper constructs it; methods live in llama-kv-cache.cpp so the
    // registry close stays in that trust domain.)
    struct vbr_composite_outcome {
        vbr_operation_id operation_id = {};
        int32_t          declared     = 0;
        int32_t          claimed      = 0;
        int32_t          terminal     = 0;
        bool             sealed       = false;
        bool             failed       = false;
        bool             closed       = false;

        void claim();
        void report_terminal(bool ok);
        void seal(bool wrapper_ok);

      private:
        void try_close();
    };

  private:
    struct vbr_pending_decode_op {
        vbr_operation_id  operation_id   = {};
        // P1v2 (v6): per-target damage extents — submit, commit, fail, and recovery cover
        // every handle; each cell cited its SELECTED target's extent at stamp time.
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> extents = {};
        int32_t           recovery_index = -1;
        // Single-cache ops close directly (owns_close). Composite children instead report
        // their terminal result into the shared sealed aggregate, which closes the root.
        bool              owns_close     = true;
        std::shared_ptr<vbr_composite_outcome> composite;
    };

    class vbr_mutation_op {
      public:
        vbr_mutation_op(llama_kv_cache *    cache,
                        vbr_operation_kind  kind,
                        vbr_operation_class operation_class,
                        llama_seq_id        seq_id,
                        llama_pos           p0,
                        llama_pos           p1,
                        bool                provenance_bearing = false,
                        uint16_t            extent_stream      = 0);
        // Multi-target form (decode composites): the caller supplies the full manifest.
        vbr_mutation_op(llama_kv_cache *              cache,
                        const vbr_operation_binding & binding,
                        bool                          provenance_bearing);
        ~vbr_mutation_op();

        vbr_mutation_op(const vbr_mutation_op &)             = delete;
        vbr_mutation_op & operator=(const vbr_mutation_op &) = delete;
        vbr_mutation_op(vbr_mutation_op &&)                  = delete;
        vbr_mutation_op & operator=(vbr_mutation_op &&)      = delete;

        bool active() const { return static_cast<bool>(operation_id_); }
        std::optional<vbr_pending_decode_op> detach_deferred();
        // P1v2 (v6): per-target lazy extent — reserved at the FIRST destructive stamp that
        // SELECTS manifest target `target_index` (the tracker calls through the trampoline).
        // Idempotent per target; empty on reservation failure (availability path taken).
        vbr_extent_handle ensure_extent_for(uint8_t target_index);
        static vbr_extent_handle extent_trampoline(void * ctx, uint8_t target_index);
        // P1v2: a refused/unauthorized stamp poisons the whole LOGICAL operation — failure
        // ownership follows the same root link as extent ownership (v6-fix F2), so a poison
        // in a joined child fails the root: it reports FAILED (into its aggregate for
        // composite children, at its close for owned scopes) and its recovery reservation
        // survives to quarantine through the failed close's autorecord.
        void poison() {
            poisoned_ = true;
            if (extent_owner_ != nullptr && extent_owner_ != this) {
                extent_owner_->poison();
            }
        }
        // v3.1 amendment 4: explicit success required — destruction without succeed() closes
        // the operation FAILED (exception unwind and forgotten paths fail by construction).
        // A poisoned scope can never succeed.
        void succeed() {
            if (!poisoned_) {
                succeeded_ = true;
            }
        }

        // For the always-succeed metadata-edit family: ONE opt-in per function. Succeeds at
        // scope exit UNLESS an exception entered flight — the default-fail pin holds on
        // unwind while every normal return path stops hand-spelling succeed().
        class success_on_return {
          public:
            explicit success_on_return(vbr_mutation_op & op)
                : op_(op), exceptions_at_entry_(std::uncaught_exceptions()) {}
            ~success_on_return() {
                if (std::uncaught_exceptions() == exceptions_at_entry_) {
                    op_.succeed();
                }
            }
          private:
            vbr_mutation_op & op_;
            int               exceptions_at_entry_;
        };

      private:
        void abort_to_shadow_unavailable();
        void fail_extents();
        // Owned scopes read their manifest from the RAII (which retains the identical
        // binding); only adopted scopes hold their own registry-fetched copy.
        const vbr_operation_binding & scope_manifest() const {
            return owned_op_ ? owned_op_->binding() : manifest_;
        }

        friend class llama_kv_cache;
        llama_kv_cache *      cache_          = nullptr;
        vbr_mutation_op *     outer_          = nullptr;
        // The root scope owning the per-target extents this chain stamps against; joined
        // scopes point at their root so the tracker's extent callback lands there.
        vbr_mutation_op *     extent_owner_   = nullptr;
        // Minting scopes own a registry operation; joining scopes (nested/adopted) borrow the
        // outer/adopted id and own nothing (review F10).
        std::optional<vbr_scoped_operation> owned_op_;
        vbr_operation_id      operation_id_   = {};
        vbr_operation_kind    kind_           = vbr_operation_kind::sequence_edit;
        // P1v2 (v6): adopted scopes' authenticated manifest copy (owned scopes read the
        // RAII's retained binding via scope_manifest()); one lazy extent per target.
        vbr_operation_binding manifest_       = {};
        std::array<vbr_extent_handle, vbr_operation_binding::MAX_TARGETS> extents_ = {};
        // P3v2: the participant aggregate this adopted child claimed a slot in.
        std::shared_ptr<vbr_composite_outcome> composite_;
        int32_t               recovery_index_ = -1;
        bool                  succeeded_      = false;
        bool                  poisoned_       = false;
        bool                  detached_       = false;   // token transferred to pending owner
        bool                  joined_         = false;   // nested: borrows outer identity fully
        bool                  adopted_        = false;   // C1: shared id, OWN reservations
    };
    friend class vbr_mutation_op;
    // The innermost open mutation scope; vbr_generation_begin cites it.
    vbr_mutation_op * vbr_current_mutation_ = nullptr;
    // A2 (review F10b): an operation adopted from a composite wrapper — child mutation scopes
    // join it instead of minting, so iSWA/hybrid children share ONE id per logical mutation.
    vbr_operation_id vbr_adopted_operation_ = {};
    // P2v2 (v6): the composite root's mint was REFUSED — child scopes open refused (fail
    // closed to shadow-unavailable) instead of minting independently (A0 one-id in refusal).
    bool vbr_adopted_refused_ = false;
    // v4 review F1: the composite aggregate the adopted children report their terminal
    // results into (set only for deferred/decode composites).
    std::shared_ptr<vbr_composite_outcome> vbr_adopted_composite_;
    // A2 (review F3): decode operations stay open past apply_ubatch — closed only when the
    // decode outcome is known. One entry per in-flight committed ubatch.
    std::vector<vbr_pending_decode_op> vbr_pending_decode_ops_;
    // C1: records whose extents are `submitted`, awaiting terminal delivery at the sync fence.
    // Their registry operations and recovery reservations remain OPEN until then.
    std::vector<vbr_pending_decode_op> vbr_awaiting_commit_;
    uint64_t vbr_pending_commit_failures_ = 0;  // sync-boundary commit failures, counted

  public:
    // Promote submitted extents to committed. Called from the context's existing synchronize
    // point — never introduces a new fence (Rev 5.1). No-op when nothing is pending.
    void vbr_commit_submitted();
    // A2 (review F3): resolve in-flight decode operations at the decode boundary where the
    // outcome is known. ok=true: extents -> submitted, ops close committed. ok=false: extents
    // fail, ops close failed (autorecording their reserved recovery slots).
    void vbr_decode_ops_finish(bool ok);
    // A2 (review F10b): composite adoption — wrappers mint once and adopt into children.
    void vbr_adopt_operation(vbr_operation_id operation_id);
    void vbr_adopt_composite(std::shared_ptr<vbr_composite_outcome> composite);
    void vbr_adopt_refused();
    void vbr_release_adopted();
    // v4 review F4 + P2v2/v6-fix F1: decode manifest bound to the ubatch's actual sequences —
    // per touched seq an ordinary target over its exact position range, plus (when wrapping
    // is possible) a whole-range swa_wrap target per seq and ONE declared seq-wildcard
    // whole-range state_api target authorizing the nested §7.3 prefix purge: cross-sequence
    // masked reuse makes the destroyed position and the purged old owner unbounded by the
    // batch, and the owner is chosen by slot selection AFTER an adopted manifest has minted.
    // TRANSACTIONAL: any target-ceiling overflow zeroes the manifest and returns false — the
    // registry then refuses the mint (fail-closed shadow-unavailable), never partial.
    static bool vbr_decode_targets_from_ubatch(vbr_operation_binding & binding,
                                               uint64_t pool_hi, uint64_t pool_lo,
                                               bool wrap_possible, uint16_t stream,
                                               const llama_ubatch & ubatch);
    // Pool identity for manifest construction ({0,0} = unarmed/wildcard).
    vbr_pool_uuid vbr_pool_id() const {
        const auto * tracker = vbr_generation_tracker_get();
        return tracker != nullptr ? tracker->pool_identity() : vbr_pool_uuid{};
    }

  private:

    vbr_generation_event vbr_generation_begin(
            vbr_mutation_registrant registrant,
            vbr_operation_class operation_class,
            uint32_t stream,
            vbr_generation_stamp_kind stamp_kind,
            bool destructive = false,
            bool imported = false);
    // P1v2 (v6): the ONE spelling of "a refused stamp fails the owning operation" — inert on
    // an empty (unarmed/latched) event, poisons the scope on refusal. Runtime fail-closed,
    // never an assert.
    void vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                   llama_seq_id membership_seq, llama_pos pre_mutation_pos = -1);
    void vbr_stamp(vbr_mutation_op & op, vbr_generation_event & event, uint32_t cell,
                   const llama_seq_id * seqs, int32_t n_seqs, llama_pos pre_mutation_pos);
    void vbr_generation_global(vbr_mutation_registrant registrant, vbr_operation_class operation_class);
    void vbr_ownership_rebuild();  // A2: import/install-boundary index rebuild (sanctioned scan)
    void vbr_ownership_update_all_seqs(uint32_t stream, uint32_t cell, llama_pos pos,
                                       bool add, llama_seq_id exclude_seq = -1);
    void     vbr_shrink_watermark();                  // occupancy dropped: release phantom tail pages
    bool     vbr_promote_next(uint32_t wm_next);      // occupancy dropped: re-promote one container
    void     vbr_floor_clamp_order();
    size_t   vbr_flush_deferred_unmaps(); // returns the number of entries flushed
    bool     vbr_scratch_reserve(uint32_t wm_cells);  // #88: boundary-time f16 dequant scratch grow
    char *   vbr_stash_ensure(vbr_pool & p);          // lazy per-pool sink-stash buffer; returns base
    void     vbr_load_degrade_order();                // baked table, VBR_DEGRADE_ORDER=<file>, or generic fallback
    void     vbr_synth_generic_order();               // cross-model curves for unsupported archs (VBR_FORCE_GENERIC=1 to force)
    size_t   vbr_vmm_projected_bytes(const vbr_pool & p, uint32_t wm_cells) const;
    size_t   vbr_budget_eff(const vbr_pool & p) const; // live-clamped per-pool budget (shared basis)
    size_t   vbr_budget_eff_uncached(const vbr_pool & p) const; // restore preflight: fresh live capacity
    bool     vbr_vmm_active() const;                  // any pool is VMM-backed
    bool     vbr_over_budget(uint32_t wm_cells) const; // any VMM pool projected past its budget
    vbr_pool *       vbr_pool_of(const ggml_tensor * t);       // pool owning the tensor (by buffer)
    const vbr_pool * vbr_pool_of(const ggml_tensor * t) const;
    // every VMM pool holding an extent for one (layer,side) unit: exactly one under -sm layer,
    // one per device under -sm tensor (each with that device's shard), empty for static units.
    // A tier step applies to the unit — i.e. to EVERY entry returned here.
    // every VMM pool holding unit (ikv, side): a const ref into vbr_units_tab_, precomputed
    // once after pool construction (membership is fixed for the cache's lifetime) — the
    // degrade/promote paths call this per decode boundary, so it must not allocate
    const std::vector<std::pair<vbr_pool *, vbr_extent *>> & vbr_units_of(size_t ikv, bool is_v) const;
    bool vbr_unit_pooled(size_t ikv, bool is_v) const;         // any VMM pool holds this unit
    // side pinned via mixed config (-ctk turbo8 -ctv vbr): ladder never touches it
    bool vbr_side_pinned(bool is_v) const { return is_v ? vbr_params_.pin_v : vbr_params_.pin_k; }
    // unified pin contract: a unit may be stepped only if its current type is a vbr tier AND
    // its side is not flag-pinned — every degrade/promote/sim walk must use this predicate
    bool vbr_unit_movable(ggml_type t, bool is_v) const;
    uint32_t vbr_watermark_cells(uint32_t extra_tokens) const; // shared by prepare() + ensure_mapped
    bool     vbr_degrade_next(uint32_t wm_next);      // one step down the order; false = exhausted
                                                      // wm_next = projected watermark incl. the
                                                      // incoming batch (bounds live pages/scrub)

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // Dynamic VBR shared KV pools (M2 bookkeeping; M3 transcode/relocate) — one per KV buffer
    // (per device under -sm layer; exactly one on a single GPU)
    std::vector<vbr_pool> vbr_pools_;
    // [ikv*2 + is_v] -> (pool, extent) units; built once at ctor end, immutable after
    std::vector<std::vector<std::pair<vbr_pool *, vbr_extent *>>> vbr_units_tab_;

    // Permanent transcode oracle (env VBR_TRANSCODE_TEST): synthetic turbo8 A->A byte round-trip +
    // turbo8->turbo4 in-place-vs-separate identity, on a scoped CUDA backend. See definition.
    void vbr_transcode_anchor_test();

    friend struct llama_kv_cache_vbr_epoch_test;

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // VBR tier-flip epoch of the underlying cache (0 when VBR is off — the counter never moves)
    uint64_t get_vbr_epoch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;


    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
