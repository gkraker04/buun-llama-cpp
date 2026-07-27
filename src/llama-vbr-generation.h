#pragma once

#include "llama-vbr-generation-types.h"
#include "llama-vbr-operation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct vbr_unit_generation {
    uint64_t            repr_gen         = 0;
    uint64_t            publish_seq      = 0;
    int32_t             current_type     = -1;
    int32_t             last_source_type = -1;
    vbr_repr_domain     domain           = vbr_repr_domain::full;
    uint8_t             promote_hops     = 0;
    vbr_repr_transition last_transition  = vbr_repr_transition::initial;
    uint8_t             flags            = 0;
};

enum class vbr_generation_stamp_kind : uint8_t {
    dependency,
    membership,
};

class vbr_generation_tracker;

struct vbr_generation_event {
    vbr_generation_event() = default;
    ~vbr_generation_event();

    vbr_generation_event(const vbr_generation_event &)             = delete;
    vbr_generation_event & operator=(const vbr_generation_event &) = delete;
    vbr_generation_event(vbr_generation_event && other) noexcept;
    vbr_generation_event & operator=(vbr_generation_event &&) = delete;

    bool finish();

    explicit operator bool() const { return serial != 0; }

  private:
    friend class vbr_generation_tracker;
    uint64_t                  serial          = 0;
    uint32_t                  stream          = 0;
    vbr_generation_stamp_kind stamp_kind      = vbr_generation_stamp_kind::dependency;
    vbr_mutation_family       family          = vbr_mutation_family::append;
    vbr_operation_class       operation_class = vbr_operation_class::ordinary_decode;
    bool                      destructive     = false;
    bool                      imported        = false;
    vbr_generation_tracker *  owner_          = nullptr;
};

// Armed-only dual-write store. The raw arrays stay private so every read comparison can be
// isolated in checkpoint_vbr_eligibility(). Mutation sites use the closed A0 registrant table
// through begin_event(); no caller supplies an open-ended family string.
class vbr_generation_tracker {
  public:
    vbr_generation_tracker(uint32_t n_stream, uint32_t n_cells, uint32_t n_units);
    ~vbr_generation_tracker();

    vbr_generation_tracker(const vbr_generation_tracker &)             = delete;
    vbr_generation_tracker & operator=(const vbr_generation_tracker &) = delete;
    vbr_generation_tracker(vbr_generation_tracker &&)                  = delete;
    vbr_generation_tracker & operator=(vbr_generation_tracker &&)      = delete;

    bool     active() const;
    uint32_t stream_count() const;
    uint32_t cell_count() const;
    uint32_t unit_count() const;
    bool     stable() const;

    vbr_pool_uuid pool_identity() const;
    uint64_t      controller_generation() const;

    vbr_generation_event begin_event(vbr_mutation_registrant   registrant,
                                     vbr_operation_class       operation_class,
                                     uint32_t                  stream,
                                     vbr_generation_stamp_kind stamp_kind,
                                     bool                      destructive = false,
                                     bool                      imported    = false);
    bool stamp_cell(const vbr_generation_event & event, uint32_t cell, llama_seq_id membership_seq = -1);

    bool global_transition(vbr_mutation_registrant registrant, vbr_operation_class operation_class);
    bool initialize_unit(uint32_t unit, int32_t type, vbr_repr_domain domain);
    bool publish_unit(uint32_t            unit,
                      int32_t             source_type,
                      int32_t             target_type,
                      vbr_repr_domain     domain,
                      uint8_t             promote_hops,
                      vbr_repr_transition transition);

    // Read-only accessors are intentionally value-returning. Callers may capture them, but CI
    // forbids comparison of raw stamp fields outside checkpoint_vbr_eligibility().
    uint32_t            page_generation(uint32_t stream, uint32_t page) const;
    uint32_t            page_destructive_generation(uint32_t stream, uint32_t page) const;
    uint32_t            page_import_generation(uint32_t stream, uint32_t page) const;
    uint32_t            dependency_generation(uint32_t stream, uint32_t cell) const;
    uint32_t            membership_generation(uint32_t stream, uint32_t cell) const;
    uint16_t            dependency_provenance(uint32_t stream, uint32_t cell) const;
    uint16_t            membership_provenance(uint32_t stream, uint32_t cell) const;
    llama_seq_id        last_membership_seq(uint32_t stream, uint32_t cell) const;
    vbr_unit_generation unit_generation(uint32_t unit) const;

  private:
    friend struct vbr_generation_event;
    struct stream_state;
    bool reset_page_generations_before_wrap();
    bool reset_unit_generations_before_wrap();
    bool finish_event(uint64_t serial);

    vbr_pool_uuid                         pool_uuid_          = {};
    uint64_t                              global_generation_  = 1;
    uint64_t                              mutation_serial_    = 0;
    uint64_t                              event_serial_       = 0;
    uint32_t                              active_event_depth_ = 0;
    static constexpr uint32_t             MAX_EVENT_DEPTH     = 64;
    std::array<uint64_t, MAX_EVENT_DEPTH> active_event_stack_ = {};
    uint32_t                              n_cells_            = 0;
    std::vector<stream_state>             streams_;
    std::vector<vbr_unit_generation>      units_;
};

// Production capture consumes a controller-owned exact dependency index and passes its canonical
// physical cells here. This helper never discovers dependencies by scanning the cache; the
// independent oracle has a separate implementation and trust boundary.
bool vbr_generation_capture_stream(const vbr_generation_tracker &     tracker,
                                   uint32_t                           stream,
                                   llama_seq_id                       dependency_seq_id,
                                   llama_pos                          computation_frontier,
                                   const std::vector<uint32_t> &      canonical_dependency_cells,
                                   vbr_checkpoint_generation_stream & output);

bool vbr_generation_capture_controller(const vbr_generation_tracker &                        tracker,
                                       uint32_t                                              child_id,
                                       checkpoint_child_dependency_mode                      dependency_mode,
                                       const std::vector<vbr_checkpoint_generation_stream> & streams,
                                       vbr_checkpoint_generation_controller &                output);

using vbr_generation_cell_has_seq_fn = bool (*)(const void * context,
                                                uint32_t     stream,
                                                uint32_t     cell,
                                                llama_seq_id seq_id);

struct vbr_generation_live_stream_view {
    uint32_t                       stream_index           = 0;
    llama_seq_id                   dependency_seq_id      = -1;
    llama_pos                      computation_frontier   = -1;
    uint32_t                       exact_dependency_count = 0;
    const void *                   membership_context     = nullptr;
    vbr_generation_cell_has_seq_fn cell_has_seq           = nullptr;
};

struct vbr_generation_live_controller_view {
    uint32_t                                     child_id        = 0;
    checkpoint_child_dependency_mode             dependency_mode = checkpoint_child_dependency_mode::absent;
    const vbr_generation_tracker *               tracker         = nullptr;
    std::vector<vbr_generation_live_stream_view> streams;
};

struct vbr_generation_live_view {
    bool                                             legacy_eligible              = false;
    bool                                             identity_frontier_eligible   = true;
    bool                                             capability_applicable        = true;
    std::array<uint8_t, 32>                          identity_policy_order_digest = {};
    std::vector<vbr_generation_live_controller_view> controllers;
};

enum class vbr_checkpoint_eligibility_category : uint8_t {
    not_applicable,
    generation_unknown,
    strict_accept,
    live_rebased_shadow_accept,
    strict_reject,
};

enum class vbr_checkpoint_eligibility_reason : uint8_t {
    none,
    capability_not_applicable,
    record_unknown,
    record_version,
    identity_or_frontier,
    controller_shape,
    child_order,
    dependency_mode,
    controller_inactive,
    controller_unstable,
    pool_uuid,
    global_generation,
    unit_shape,
    unit_unstable,
    unit_generation,
    live_rebased_transition,
    stream_shape,
    stream_order,
    malformed_page_refs,
    page_out_of_range,
    dependency_changed,
    dependency_membership_lost,
    dependency_cardinality,
};

struct vbr_checkpoint_eligibility {
    bool                                legacy              = false;
    bool                                strict              = false;
    bool                                live_rebased_shadow = false;
    vbr_checkpoint_eligibility_category category            = vbr_checkpoint_eligibility_category::strict_reject;
    vbr_checkpoint_eligibility_reason   reason              = vbr_checkpoint_eligibility_reason::none;
};

// The sole raw-generation comparison authority. A1 does not route any live selector here; A2
// migrates every reader and flips only after the four-way ratchet qualifies.
vbr_checkpoint_eligibility checkpoint_vbr_eligibility(const vbr_checkpoint_generation_record & checkpoint,
                                                      const vbr_generation_live_view &         live);
