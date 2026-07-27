#include "llama-vbr-generation.h"

#include "ggml.h"
#include "llama-cparams.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>

namespace {

static_assert(LLAMA_MAX_SEQ <= std::numeric_limits<int16_t>::max(),
              "A1 packed membership provenance must widen with LLAMA_MAX_SEQ");

std::atomic<uint64_t> g_vbr_pool_uuid_counter{ 1 };
std::atomic<bool>     g_vbr_pool_uuid_exhausted{ false };

enum class generation_dispatch_effect : uint8_t {
    dependency,
    membership,
    global,
    unit,
    delegated_transaction,
};

// VBR_GENERATION_MUTATION_DISPATCH_EXHAUSTIVE
constexpr std::array<generation_dispatch_effect,
                     static_cast<size_t>(vbr_mutation_registrant::count)>
    VBR_GENERATION_DISPATCH = {
        {
         generation_dispatch_effect::dependency,             // apply_ubatch_append
            generation_dispatch_effect::dependency,             // apply_ubatch_occupied_reuse
            generation_dispatch_effect::membership,             // seq_rm
            generation_dispatch_effect::membership,             // seq_cp
            generation_dispatch_effect::membership,             // seq_keep
            generation_dispatch_effect::dependency,             // seq_add
            generation_dispatch_effect::dependency,             // seq_div
            generation_dispatch_effect::delegated_transaction,  // state_read_meta
            generation_dispatch_effect::delegated_transaction,  // state_read_data
            generation_dispatch_effect::global,                 // state_read_install
            generation_dispatch_effect::delegated_transaction,  // state_read_cleanup
            generation_dispatch_effect::global,                 // whole_import
            generation_dispatch_effect::global,                 // explicit_restore_adopt
            generation_dispatch_effect::global,                 // clear
            generation_dispatch_effect::global,                 // full_reset
            generation_dispatch_effect::unit,                   // degrade_next
            generation_dispatch_effect::unit,                   // promote_next
            generation_dispatch_effect::delegated_transaction,  // execute_shed -> degrade_next
            generation_dispatch_effect::global,                 // authenticated_recovery
        }
};
static_assert(VBR_GENERATION_DISPATCH.size() == static_cast<size_t>(vbr_mutation_registrant::count),
              "every closed A0 mutation registrant must have an A1 generation effect");

uint16_t pack_provenance(vbr_mutation_family family, vbr_operation_class operation_class) {
    return uint16_t(static_cast<uint8_t>(family)) | uint16_t(uint16_t(static_cast<uint8_t>(operation_class)) << 8);
}

const vbr_mutation_registration * registration_for(vbr_mutation_registrant registrant) {
    const size_t index = static_cast<size_t>(registrant);
    if (index >= VBR_MUTATION_REGISTRY.size()) {
        return nullptr;
    }
    const auto & registration = VBR_MUTATION_REGISTRY[index];
    return registration.registrant == registrant ? &registration : nullptr;
}

bool class_allowed(const vbr_mutation_registration & registration, vbr_operation_class operation_class) {
    const size_t index = static_cast<size_t>(operation_class);
    return index < static_cast<size_t>(vbr_operation_class::count) &&
           (registration.allowed_classes & (uint16_t(1u) << index)) != 0;
}

uint32_t page_count(uint32_t cells) {
    return (cells + VBR_GENERATION_PAGE_CELLS - 1) / VBR_GENERATION_PAGE_CELLS;
}

vbr_pool_uuid allocate_pool_uuid() {
    if (g_vbr_pool_uuid_exhausted.load(std::memory_order_acquire)) {
        return {};
    }
    uint64_t expected = g_vbr_pool_uuid_counter.load(std::memory_order_relaxed);
    for (;;) {
        if (expected == 0) {
            g_vbr_pool_uuid_exhausted.store(true, std::memory_order_release);
            return {};
        }
        const uint64_t next = expected == std::numeric_limits<uint64_t>::max() ? 0 : expected + 1;
        if (g_vbr_pool_uuid_counter.compare_exchange_weak(expected, next, std::memory_order_acq_rel,
                                                          std::memory_order_relaxed)) {
            if (next == 0) {
                g_vbr_pool_uuid_exhausted.store(true, std::memory_order_release);
            }
            return { UINT64_C(0x56425247454e4131), expected };
        }
    }
}

bool mask_test(const std::array<uint64_t, VBR_GENERATION_MASK_WORDS> & mask, uint32_t offset) {
    return (mask[offset / 64] & (uint64_t(1) << (offset % 64))) != 0;
}

void mask_set(std::array<uint64_t, VBR_GENERATION_MASK_WORDS> & mask, uint32_t offset) {
    mask[offset / 64] |= uint64_t(1) << (offset % 64);
}

uint32_t mask_popcount(const std::array<uint64_t, VBR_GENERATION_MASK_WORDS> & mask) {
    uint32_t result = 0;
    for (uint64_t word : mask) {
#if defined(__GNUC__) || defined(__clang__)
        result += static_cast<uint32_t>(__builtin_popcountll(word));
#else
        while (word != 0) {
            word &= word - 1;
            ++result;
        }
#endif
    }
    return result;
}

bool unit_equal(const vbr_checkpoint_unit_generation & captured, const vbr_unit_generation & current) {
    return captured.repr_gen == current.repr_gen && captured.current_type == current.current_type &&
           captured.last_source_type == current.last_source_type && captured.domain == current.domain &&
           captured.promote_hops == current.promote_hops && captured.last_transition == current.last_transition;
}

bool unit_live_rebased_shape(const vbr_checkpoint_unit_generation & captured, const vbr_unit_generation & current) {
    return captured.current_type == GGML_TYPE_F16 && captured.domain == vbr_repr_domain::full &&
           current.current_type == GGML_TYPE_TURBO8_0 && current.domain == vbr_repr_domain::full &&
           captured.repr_gen != std::numeric_limits<uint64_t>::max() && current.repr_gen == captured.repr_gen + 1 &&
           current.last_source_type == GGML_TYPE_F16 && current.promote_hops == captured.promote_hops;
}

bool unit_live_rebased(const vbr_checkpoint_unit_generation & captured, const vbr_unit_generation & current) {
    return unit_live_rebased_shape(captured, current) &&
           current.last_transition == vbr_repr_transition::degrade_f16_to_t8_admitted;
}

vbr_checkpoint_eligibility reject(
    bool                                legacy,
    vbr_checkpoint_eligibility_reason   reason,
    vbr_checkpoint_eligibility_category category = vbr_checkpoint_eligibility_category::strict_reject) {
    vbr_checkpoint_eligibility result;
    result.legacy   = legacy;
    result.category = category;
    result.reason   = reason;
    return result;
}

// --- A2 §5.5 expected-tombstone classification (evaluator-private) ---------------------------

constexpr uint32_t REJECT_INSPECT_CAP = 4096;

struct audit_state {
    bool     refinement_used       = false;
    bool     saw_destructive       = false;
    bool     saw_import            = false;
    bool     overflowed            = false;
    bool     cardinality_violated  = false;
    uint32_t rejecting_cells       = 0;
};

struct audit_cell {
    const vbr_generation_tracker * tracker = nullptr;
    bool           membership_lost = false;
    bool           doubly_rejected = false;  // F7: dep AND membership — mixed reason
    bool           in_range        = false;  // C2 stamp-time range proof (committed evidence)
    uint16_t       provenance      = 0;
    vbr_extent_ref extent          = {};
    llama_pos      current_pos     = -1;
    llama_pos      frontier        = -1;
    llama_seq_id   dep_seq         = -1;
};

void fill_audit(vbr_checkpoint_eligibility & result, const audit_state & audit) {
    result.refinement_used = audit.refinement_used;
    result.rejecting_cells = audit.rejecting_cells;
    result.observation_class = audit.saw_import      ? vbr_observation_class::import_refined
                             : audit.saw_destructive ? vbr_observation_class::destructive
                             : audit.refinement_used ? vbr_observation_class::boundary_refined
                                                     : vbr_observation_class::trivial_append;
}

// §5.5 closed table. A strict reject is an expected tombstone ONLY when every rejecting covered
// cell carries one uniform allowed (family, class) provenance whose committed extent entry — the
// same single entry for all cells — satisfies the row's extent predicate. Anything mixed,
// overwritten, non-committed, or overflowed is `unexplained` (fail-closed disagreement).
vbr_expected_tombstone_class classify_expected_tombstone(const std::vector<audit_cell> & cells,
                                                         bool                            overflowed,
                                                         bool                            cardinality_violated) {
    if (overflowed || cardinality_violated || cells.empty()) {
        return vbr_expected_tombstone_class::unexplained;
    }
    // F8: one uniform provenance AND one tracker — extent refs are store-local, so cells from
    // different controllers can never share evidence.
    const vbr_generation_tracker * tracker    = cells.front().tracker;
    const uint16_t                 provenance = cells.front().provenance;
    const vbr_extent_ref           extent_ref = cells.front().extent;
    if (tracker == nullptr) {
        return vbr_expected_tombstone_class::unexplained;
    }
    for (const auto & rc : cells) {
        // F7 (§5.5 rule 4): a doubly-rejecting cell is a mixed reason — never expected.
        if (rc.doubly_rejected || rc.provenance != provenance || rc.tracker != tracker ||
            rc.extent.index != extent_ref.index || rc.extent.expected_gen != extent_ref.expected_gen) {
            return vbr_expected_tombstone_class::unexplained;
        }
    }
    const auto family          = static_cast<vbr_mutation_family>(provenance & 0xFF);
    const auto operation_class = static_cast<vbr_operation_class>((provenance >> 8) & 0xFF);

    // One committed extent entry must back every rejecting cell (equality checked above).
    const vbr_extent_entry * entry = tracker->extent_store().lookup_committed(extent_ref);
    if (entry == nullptr || entry->family != family || entry->operation_class != operation_class) {
        return vbr_expected_tombstone_class::unexplained;
    }

    if (family == vbr_mutation_family::trim &&
        operation_class == vbr_operation_class::restore_one_behind_trim) {
        // Row 1: every lost covered position >= p0 (implied: the cells cite this trim), and
        // p0 >= checkpoint_frontier - 1.
        for (const auto & rc : cells) {
            // C2: the stamp-time range proof replaces assumption — every lost position was
            // INSIDE the committed trim extent, and the extent starts at/after frontier-1.
            if (!rc.membership_lost || !rc.in_range || entry->p0 < rc.frontier - 1) {
                return vbr_expected_tombstone_class::unexplained;
            }
        }
        return vbr_expected_tombstone_class::restore_one_behind;
    }
    if (family == vbr_mutation_family::occupied_reuse &&
        operation_class == vbr_operation_class::swa_wrap) {
        // Row 2: current occupant is the same dependency sequence at a strictly higher logical
        // position. Positions written after capture are >= frontier > every covered position,
        // so the frontier bound is the sound check. Membership persisting IS the same-seq
        // occupancy (still_has_seq was derivable as !membership_lost — simplification review).
        for (const auto & rc : cells) {
            if (rc.membership_lost || rc.current_pos < rc.frontier) {
                return vbr_expected_tombstone_class::unexplained;
            }
        }
        return vbr_expected_tombstone_class::swa_wrap;
    }
    if (family == vbr_mutation_family::trim &&
        operation_class == vbr_operation_class::explicit_destructive_trim) {
        // Row 3: every rejecting cell is inside the registry-bound committed [p0,p1). A covered
        // cell mutated OUTSIDE that range necessarily carries different provenance and already
        // failed the uniformity check above.
        for (const auto & rc : cells) {
            if (!rc.membership_lost || !rc.in_range) {
                return vbr_expected_tombstone_class::unexplained;
            }
        }
        return vbr_expected_tombstone_class::explicit_destructive_trim;
    }
    if (family == vbr_mutation_family::trim &&
        operation_class == vbr_operation_class::dependency_seq_remove) {
        // Row 4: the registered removal spans the whole dependency sequence and that sequence
        // is absent from every covered cell.
        const bool whole_seq = entry->p0 <= 0 && entry->p1 == std::numeric_limits<llama_pos>::max();
        for (const auto & rc : cells) {
            if (!whole_seq || !rc.membership_lost || entry->seq_id != rc.dep_seq) {
                return vbr_expected_tombstone_class::unexplained;
            }
        }
        return vbr_expected_tombstone_class::dependency_seq_removed;
    }
    return vbr_expected_tombstone_class::unexplained;
}

}  // namespace

vbr_generation_event::~vbr_generation_event() {
    if (owner_ != nullptr && !finish()) {
        std::abort();
    }
}

vbr_generation_event::vbr_generation_event(vbr_generation_event && other) noexcept :
    serial(other.serial),
    stream(other.stream),
    stamp_kind(other.stamp_kind),
    family(other.family),
    operation_class(other.operation_class),
    destructive(other.destructive),
    imported(other.imported),
    operation_id(other.operation_id),
    manifest(other.manifest),
    registrant_bit(other.registrant_bit),
    poisoned(other.poisoned),
    extent_fn(other.extent_fn),
    extent_ctx(other.extent_ctx),
    owner_(other.owner_) {
    other.serial       = 0;
    other.operation_id = {};
    other.extent_fn    = nullptr;
    other.extent_ctx   = nullptr;
    other.owner_       = nullptr;
}

bool vbr_generation_event::finish() {
    if (owner_ == nullptr) {
        return false;
    }
    if (!owner_->finish_event(serial)) {
        return false;
    }
    owner_ = nullptr;
    serial = 0;
    return true;
}

struct vbr_generation_tracker::stream_state {
    std::vector<uint32_t> page_event_gen;
    std::vector<uint32_t> page_last_destructive_gen;
    std::vector<uint32_t> page_last_import_gen;
    std::vector<uint64_t> page_event_serial;

    std::vector<uint32_t> cell_last_dependency_gen;
    std::vector<uint32_t> cell_last_membership_gen;
    std::vector<uint16_t> cell_dependency_provenance;
    std::vector<uint16_t> cell_membership_provenance;
    std::vector<int16_t>  cell_last_membership_seq;

    // A2: durable committed-extent references (design D-A2-4v3). One per stamp kind — a cell
    // can retain two different events (latest dependency + latest membership).
    std::vector<vbr_extent_ref> cell_dependency_extent;
    std::vector<vbr_extent_ref> cell_membership_extent;
    // C2: stamp-time range-proof bits (position was inside the authenticated target range).
    std::vector<uint64_t> cell_dependency_in_range;
    std::vector<uint64_t> cell_membership_in_range;
};

static void set_range_bit(std::vector<uint64_t> & bits, uint32_t cell, bool value) {
    if (value) {
        bits[cell / 64] |= uint64_t(1) << (cell % 64);
    } else {
        bits[cell / 64] &= ~(uint64_t(1) << (cell % 64));
    }
}

static bool get_range_bit(const std::vector<uint64_t> & bits, uint32_t cell) {
    return (bits[cell / 64] & (uint64_t(1) << (cell % 64))) != 0;
}

vbr_generation_tracker::~vbr_generation_tracker() {
    if (active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0) {
        std::abort();
    }
}

vbr_generation_tracker::vbr_generation_tracker(uint32_t n_stream, uint32_t n_cells, uint32_t n_units) :
    n_cells_(n_cells),
    streams_(n_stream),
    units_(n_units) {
    const uint32_t n_pages = page_count(n_cells);
    for (auto & stream : streams_) {
        stream.page_event_gen.resize(n_pages);
        stream.page_last_destructive_gen.resize(n_pages);
        stream.page_last_import_gen.resize(n_pages);
        stream.page_event_serial.resize(n_pages);
        stream.cell_last_dependency_gen.resize(n_cells);
        stream.cell_last_membership_gen.resize(n_cells);
        stream.cell_dependency_provenance.resize(n_cells);
        stream.cell_membership_provenance.resize(n_cells);
        stream.cell_last_membership_seq.resize(n_cells, -1);
        stream.cell_dependency_extent.resize(n_cells);
        stream.cell_membership_extent.resize(n_cells);
        stream.cell_dependency_in_range.resize((n_cells + 63) / 64);
        stream.cell_membership_in_range.resize((n_cells + 63) / 64);
    }

    // A process-local construction identity, not a security nonce. A fixed nonzero domain tag
    // prevents the all-zero sentinel; exhaustion latches instead of wrapping into an ABA.
    pool_uuid_ = allocate_pool_uuid();
}

bool vbr_generation_tracker::active() const {
    return pool_uuid_.hi != 0 && pool_uuid_.lo != 0 && !streams_.empty() && n_cells_ != 0;
}

uint32_t vbr_generation_tracker::stream_count() const {
    return static_cast<uint32_t>(streams_.size());
}

uint32_t vbr_generation_tracker::cell_count() const {
    return n_cells_;
}

uint32_t vbr_generation_tracker::unit_count() const {
    return static_cast<uint32_t>(units_.size());
}

bool vbr_generation_tracker::stable() const {
    if ((mutation_serial_ & 1u) != 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(units_mutex_);
    for (const auto & unit : units_) {
        if ((unit.publish_seq & 1u) != 0) {
            return false;
        }
    }
    return true;
}

vbr_pool_uuid vbr_generation_tracker::pool_identity() const {
    return pool_uuid_;
}

uint64_t vbr_generation_tracker::controller_generation() const {
    return global_generation_;
}

vbr_generation_event vbr_generation_tracker::begin_event(vbr_mutation_registrant   registrant,
                                                         vbr_operation_class       operation_class,
                                                         uint32_t                  stream,
                                                         vbr_generation_stamp_kind stamp_kind,
                                                         vbr_operation_id          operation_id,
                                                         vbr_event_extent_fn       extent_fn,
                                                         void *                    extent_ctx,
                                                         bool                      destructive,
                                                         bool                      imported) {
    // One named return object for every path (NRVO): the ~656-byte manifest is copied exactly
    // once, registry slot -> event, and refused paths return it with serial 0 (falsy).
    vbr_generation_event result;
    const auto * registration = registration_for(registrant);
    const size_t registrant_index = static_cast<size_t>(registrant);
    const generation_dispatch_effect expected_effect =
            stamp_kind == vbr_generation_stamp_kind::dependency
                    ? generation_dispatch_effect::dependency
                    : generation_dispatch_effect::membership;
    if (!active() || shadow_unavailable_ ||
        registration == nullptr || !class_allowed(*registration, operation_class) ||
        registrant_index >= VBR_GENERATION_DISPATCH.size() ||
        VBR_GENERATION_DISPATCH[registrant_index] != expected_effect ||
        stream >= streams_.size() || event_serial_ == std::numeric_limits<uint64_t>::max() ||
        active_event_depth_ == MAX_EVENT_DEPTH ||
        (active_event_depth_ == 0 && mutation_serial_ == std::numeric_limits<uint64_t>::max())) {
        return result;
    }

    // C2 (v3.2, Sol CONCUR): full manifest authentication. The event must cite a live
    // operation whose manifest (a) lists this registrant in its closed mask, (b) declares
    // this exact operation class, (c) is in the mutate phase, and (d) carries a target
    // covering this tracker's pool and the event's stream.
    if (!operation_id || !vbr_operation_registry_binding(operation_id, result.manifest)) {
        return result;
    }
    // P1v2 (v6): begin still refuses when NO target could ever cover this
    // pool/stream/class/registrant; the per-(seq, position) selection happens at EACH STAMP
    // against the event's manifest copy, so multi-target manifests authenticate
    // multi-sequence ubatches exactly instead of citing target zero.
    if (result.manifest.find_covering_target(pool_uuid_.hi, pool_uuid_.lo, stream, operation_class,
                                             vbr_registrant_bit(registrant)) == nullptr) {
        return result;
    }

    if (active_event_depth_ == 0) {
        ++mutation_serial_;
    }
    ++event_serial_;
    active_event_stack_[active_event_depth_] = event_serial_;
    ++active_event_depth_;
    result.serial          = event_serial_;
    result.stream          = stream;
    result.stamp_kind      = stamp_kind;
    result.family          = registration->family;
    result.operation_class = operation_class;
    result.destructive     = destructive;
    result.imported        = imported;
    result.operation_id    = operation_id;
    result.registrant_bit  = vbr_registrant_bit(registrant);
    result.extent_fn       = extent_fn;
    result.extent_ctx      = extent_ctx;
    result.owner_          = this;
    return result;
}

bool vbr_generation_tracker::finish_event(uint64_t serial) {
    if (serial == 0 || active_event_depth_ == 0 || active_event_stack_[active_event_depth_ - 1] != serial ||
        mutation_serial_ == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    --active_event_depth_;
    active_event_stack_[active_event_depth_] = 0;
    if (active_event_depth_ == 0) {
        ++mutation_serial_;
    }
    return true;
}

bool vbr_generation_tracker::stamp_cell(vbr_generation_event & event,
                                        uint32_t               cell,
                                        llama_seq_id           membership_seq,
                                        llama_pos              pre_mutation_pos) {
    return stamp_cell(event, cell, &membership_seq, 1, pre_mutation_pos);
}

bool vbr_generation_tracker::stamp_cell(vbr_generation_event & event,
                                        uint32_t               cell,
                                        const llama_seq_id *   seqs,
                                        int32_t                n_seqs,
                                        llama_pos              pre_mutation_pos) {
    // P1v2 (v6): a poisoned event stays inert — no further metadata moves under it.
    if (event.poisoned) {
        return false;
    }
    bool event_is_live = false;
    for (uint32_t depth = 0; depth < active_event_depth_; ++depth) {
        event_is_live = event_is_live || active_event_stack_[depth] == event.serial;
    }
    // Wiring-bug refusals (wrong owner, dead event, out-of-bounds): plain false, no poison —
    // these are not authorization failures against this event's manifest.
    const bool binds_evidence = event.destructive || event.imported;
    if (!event || event.owner_ != this || !event_is_live ||
        event.stream >= streams_.size() || cell >= n_cells_ || seqs == nullptr || n_seqs < 1 ||
        (event.stamp_kind == vbr_generation_stamp_kind::membership && n_seqs != 1)) {
        return false;
    }
    // Destructive/import evidence binds to exactly ONE selected target; a shared multi-member
    // cell has no single exact citation, so it goes unavailable instead (v6 P1 rule 3).
    if (binds_evidence && n_seqs != 1) {
        event.poisoned = true;
        set_shadow_unavailable();
        return false;
    }
    // P1v2 (v6): per-stamp covering-target selection over the event's authenticated manifest,
    // keyed by (pool, stream, class, registrant, seq, pre-mutation position). EVERY member of
    // a shared cell's sequence set needs a covering target (target-set proof); the first
    // member's selection supplies the durable evidence binding. NO cover => the stamp
    // refuses, POISONS the event, and latches shadow-unavailable IMMEDIATELY — metadata may
    // already have moved under an unauthenticated claim, so no strict accept can be allowed
    // to form until a sanctioned transition provably follows (P4v2).
    uint8_t selected_index = 0;
    bool    all_real_range = true;
    for (int32_t i = 0; i < n_seqs; ++i) {
        if (seqs[i] < -1 || seqs[i] > std::numeric_limits<int16_t>::max()) {
            return false;  // wiring-bug refusal: out-of-domain value, not an auth failure
        }
        uint8_t      index    = 0;
        const auto * covering = event.manifest.find_covering_target_at(
                pool_uuid_.hi, pool_uuid_.lo, static_cast<uint16_t>(event.stream),
                event.operation_class, event.registrant_bit, seqs[i], pre_mutation_pos, &index);
        if (covering == nullptr) {
            event.poisoned = true;
            set_shadow_unavailable();
            return false;
        }
        all_real_range = all_real_range && covering->range.p0 >= 0;
        if (i == 0) {
            selected_index = index;
        }
    }
    vbr_extent_handle extent = {};
    if (binds_evidence && event.extent_fn != nullptr) {
        extent = event.extent_fn(event.extent_ctx, selected_index);
        if (!extent) {
            // Per-target reservation failed — the owning scope already took the availability
            // path; poison so the rest of this event is inert and the operation reports
            // failed instead of committing partial evidence.
            event.poisoned = true;
            set_shadow_unavailable();
            return false;
        }
    }
    // C2: the range proof is RECORDED (in_range bit) so tombstone rows 1/3 can PROVE
    // membership from committed evidence. True only when the position is known AND every
    // member's selected target carries a real (non-wildcard) range containing it.
    const bool in_authorized_range = pre_mutation_pos >= 0 && all_real_range;
    const llama_seq_id membership_seq = seqs[0];

    auto &         stream = streams_[event.stream];
    const uint32_t page   = cell / VBR_GENERATION_PAGE_CELLS;
    if (stream.page_event_serial[page] != event.serial) {
        if (stream.page_event_gen[page] == std::numeric_limits<uint32_t>::max()) {
            if (!reset_page_generations_before_wrap()) {
                return false;
            }
        }
        stream.page_event_serial[page] = event.serial;
        ++stream.page_event_gen[page];
        if (event.destructive) {
            stream.page_last_destructive_gen[page] = stream.page_event_gen[page];
        }
        if (event.imported) {
            stream.page_last_import_gen[page] = stream.page_event_gen[page];
        }
    }

    const uint32_t generation = stream.page_event_gen[page];
    const uint16_t provenance = pack_provenance(event.family, event.operation_class);
    if (event.stamp_kind == vbr_generation_stamp_kind::dependency) {
        stream.cell_last_dependency_gen[cell]   = generation;
        stream.cell_dependency_provenance[cell] = provenance;
        extents_.release_ref(stream.cell_dependency_extent[cell]);
        stream.cell_dependency_extent[cell] = extent ? extents_.add_ref(extent) : vbr_extent_ref{};
        set_range_bit(stream.cell_dependency_in_range, cell, in_authorized_range);
    } else {
        stream.cell_last_membership_gen[cell]   = generation;
        stream.cell_membership_provenance[cell] = provenance;
        stream.cell_last_membership_seq[cell]   = static_cast<int16_t>(membership_seq);
        extents_.release_ref(stream.cell_membership_extent[cell]);
        stream.cell_membership_extent[cell] = extent ? extents_.add_ref(extent) : vbr_extent_ref{};
        set_range_bit(stream.cell_membership_in_range, cell, in_authorized_range);
    }
    return true;
}

vbr_extent_ref vbr_generation_tracker::dependency_extent(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_dependency_extent.at(cell);
}

vbr_extent_ref vbr_generation_tracker::membership_extent(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_membership_extent.at(cell);
}

bool vbr_generation_tracker::dependency_in_range(uint32_t stream, uint32_t cell) const {
    return get_range_bit(streams_.at(stream).cell_dependency_in_range, cell);
}

bool vbr_generation_tracker::membership_in_range(uint32_t stream, uint32_t cell) const {
    return get_range_bit(streams_.at(stream).cell_membership_in_range, cell);
}

bool vbr_generation_tracker::global_invalidate_and_reset_extents(vbr_mutation_registrant registrant,
                                                                 vbr_operation_class     operation_class,
                                                                 vbr_operation_id        operation_id) {
    if (!global_transition(registrant, operation_class, operation_id)) {
        return false;
    }
    // Every stored extent reference is obsolete after the global invalidation; drop them so
    // reset_all() reclaims a coherent slab (design Rev 4 item 3).
    for (auto & stream : streams_) {
        std::fill(stream.cell_dependency_extent.begin(), stream.cell_dependency_extent.end(), vbr_extent_ref{});
        std::fill(stream.cell_membership_extent.begin(), stream.cell_membership_extent.end(), vbr_extent_ref{});
    }
    extents_.reset_all();
    return true;
}

bool vbr_generation_tracker::reset_page_generations_before_wrap() {
    if (global_generation_ == std::numeric_limits<uint64_t>::max() ||
        (active_event_depth_ == 0 &&
         mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2)) {
        return false;
    }
    const bool owns_stability_barrier = active_event_depth_ == 0;
    if (owns_stability_barrier) {
        ++mutation_serial_;
    } else if ((mutation_serial_ & 1u) == 0) {
        return false;
    }
    ++global_generation_;
    for (auto & stream : streams_) {
        std::fill(stream.page_event_gen.begin(), stream.page_event_gen.end(), 0);
        std::fill(stream.page_last_destructive_gen.begin(), stream.page_last_destructive_gen.end(), 0);
        std::fill(stream.page_last_import_gen.begin(), stream.page_last_import_gen.end(), 0);
        std::fill(stream.page_event_serial.begin(), stream.page_event_serial.end(), 0);
        std::fill(stream.cell_last_dependency_gen.begin(), stream.cell_last_dependency_gen.end(), 0);
        std::fill(stream.cell_last_membership_gen.begin(), stream.cell_last_membership_gen.end(), 0);
        std::fill(stream.cell_dependency_provenance.begin(), stream.cell_dependency_provenance.end(), 0);
        std::fill(stream.cell_membership_provenance.begin(), stream.cell_membership_provenance.end(), 0);
        std::fill(stream.cell_last_membership_seq.begin(), stream.cell_last_membership_seq.end(), -1);
        // Pre-wrap reset is a global invalidation: every stored extent reference is obsolete.
        std::fill(stream.cell_dependency_extent.begin(), stream.cell_dependency_extent.end(), vbr_extent_ref{});
        std::fill(stream.cell_membership_extent.begin(), stream.cell_membership_extent.end(), vbr_extent_ref{});
    }
    extents_.reset_all();
    if (owns_stability_barrier) {
        ++mutation_serial_;
    }
    return true;
}

bool vbr_generation_tracker::try_rearm() {
    if (!shadow_unavailable_) {
        return true;
    }
    if (vbr_recovery_pending_for(pool_uuid_.hi, pool_uuid_.lo) ||
        !vbr_operation_registry_has_capacity()) {
        return false;
    }
    if (!global_invalidate_and_reset_extents(vbr_mutation_registrant::authenticated_recovery,
                                             vbr_operation_class::controller)) {
        return false;
    }
    return try_clear_shadow_unavailable();
}

bool vbr_generation_tracker::global_transition(vbr_mutation_registrant registrant,
                                               vbr_operation_class     operation_class,
                                               vbr_operation_id        operation_id) {
    // v3 review B6 / v4 review F5: cited operations validate at manifest depth — the cited
    // binding must carry a covering target for THIS pool authorizing this registrant + class.
    // The recovery drain and registry-refusal fallback run OUTSIDE any operation (empty id) —
    // a NAMED exemption: they are the paths that CREATE availability.
    if (operation_id) {
        vbr_operation_binding cited;
        if (!vbr_operation_registry_binding(operation_id, cited) ||
            cited.find_covering_target(pool_uuid_.hi, pool_uuid_.lo, 0, operation_class,
                                       vbr_registrant_bit(registrant)) == nullptr) {
            return false;
        }
    }
    const auto * registration = registration_for(registrant);
    const size_t registrant_index = static_cast<size_t>(registrant);
    if (!active() || registration == nullptr || !class_allowed(*registration, operation_class) ||
        registrant_index >= VBR_GENERATION_DISPATCH.size() ||
        VBR_GENERATION_DISPATCH[registrant_index] != generation_dispatch_effect::global ||
        active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0 ||
        global_generation_ == std::numeric_limits<uint64_t>::max() ||
        mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2) {
        return false;
    }
    ++mutation_serial_;
    ++global_generation_;
    ++mutation_serial_;
    // v3 review B7: the unavailable state does NOT auto-clear here — the cause (registry or
    // slab exhaustion) may persist. try_clear_shadow_unavailable() probes the cause.
    return true;
}

bool vbr_generation_tracker::initialize_unit(uint32_t unit, int32_t type, vbr_repr_domain domain) {
    std::lock_guard<std::mutex> lock(units_mutex_);
    if (unit >= units_.size()) {
        return false;
    }
    auto & state           = units_[unit];
    state.repr_gen         = 1;
    state.current_type     = type;
    state.last_source_type = type;
    state.domain           = domain;
    state.last_transition  = vbr_repr_transition::initial;
    return true;
}

bool vbr_generation_tracker::publish_unit(uint32_t                unit,
                                          int32_t                 source_type,
                                          int32_t                 target_type,
                                          vbr_repr_domain         domain,
                                          uint8_t                 promote_hops,
                                          vbr_repr_transition     transition,
                                          vbr_mutation_registrant registrant,
                                          vbr_operation_id        operation_id) {
    if (operation_id) {
        vbr_operation_binding cited;
        // v4-F5 + P5v2 (v6): the citation authenticates at manifest depth with the EXACT
        // registrant driving this publication — never an OR of plausible ones.
        if (!vbr_operation_registry_binding(operation_id, cited) ||
            cited.find_covering_target(pool_uuid_.hi, pool_uuid_.lo, 0,
                                       vbr_operation_class::controller,
                                       vbr_registrant_bit(registrant)) == nullptr) {
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(units_mutex_);
    if (unit >= units_.size() || active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0 ||
        units_[unit].current_type != source_type || (units_[unit].publish_seq & 1u) != 0) {
        return false;
    }
    if ((units_[unit].repr_gen == std::numeric_limits<uint64_t>::max() ||
         units_[unit].publish_seq > std::numeric_limits<uint64_t>::max() - 2) &&
        !reset_unit_generations_before_wrap()) {
        return false;
    }
    auto & state = units_[unit];
    ++state.publish_seq;
    ++state.repr_gen;
    state.last_source_type = source_type;
    state.current_type     = target_type;
    state.domain           = domain;
    state.promote_hops     = promote_hops;
    state.last_transition  = transition;
    ++state.publish_seq;
    return true;
}

bool vbr_generation_tracker::reset_unit_generations_before_wrap() {
    if (global_generation_ == std::numeric_limits<uint64_t>::max() ||
        mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2) {
        return false;
    }
    for (const auto & unit : units_) {
        if ((unit.publish_seq & 1u) != 0) {
            return false;
        }
    }
    ++mutation_serial_;
    ++global_generation_;
    for (auto & unit : units_) {
        unit.repr_gen    = 1;
        unit.publish_seq = 0;
        unit.flags       = 0;
    }
    ++mutation_serial_;
    return true;
}

uint32_t vbr_generation_tracker::page_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_event_gen.at(page);
}

uint32_t vbr_generation_tracker::page_destructive_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_last_destructive_gen.at(page);
}

uint32_t vbr_generation_tracker::page_import_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_last_import_gen.at(page);
}

uint32_t vbr_generation_tracker::dependency_generation(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_dependency_gen.at(cell);
}

uint32_t vbr_generation_tracker::membership_generation(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_membership_gen.at(cell);
}

uint16_t vbr_generation_tracker::dependency_provenance(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_dependency_provenance.at(cell);
}

uint16_t vbr_generation_tracker::membership_provenance(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_membership_provenance.at(cell);
}

llama_seq_id vbr_generation_tracker::last_membership_seq(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_membership_seq.at(cell);
}

vbr_unit_generation vbr_generation_tracker::unit_generation(uint32_t unit) const {
    std::lock_guard<std::mutex> lock(units_mutex_);
    return units_.at(unit);
}

bool vbr_generation_capture_stream(const vbr_generation_tracker &     tracker,
                                   uint32_t                           stream,
                                   llama_seq_id                       dependency_seq_id,
                                   llama_pos                          computation_frontier,
                                   const std::vector<uint32_t> &      canonical_dependency_cells,
                                   vbr_checkpoint_generation_stream & output) {
    if (!tracker.active() || !tracker.stable() || stream >= tracker.stream_count() || dependency_seq_id < 0 ||
        computation_frontier < 0 ||
        canonical_dependency_cells.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    output                           = {};
    output.stream_index              = stream;
    output.dependency_seq_id         = dependency_seq_id;
    output.computation_frontier      = computation_frontier;
    output.captured_dependency_count = static_cast<uint32_t>(canonical_dependency_cells.size());

    uint32_t previous = std::numeric_limits<uint32_t>::max();
    for (uint32_t cell : canonical_dependency_cells) {
        if (cell >= tracker.cell_count() || (previous != std::numeric_limits<uint32_t>::max() && cell <= previous)) {
            output = {};
            return false;
        }
        previous = cell;

        const uint32_t page = cell / VBR_GENERATION_PAGE_CELLS;
        if (output.pages.empty() || output.pages.back().page_index != page) {
            vbr_generation_page_ref ref;
            ref.page_index        = page;
            ref.captured_page_gen = tracker.page_generation(stream, page);
            output.pages.push_back(ref);
        }
        mask_set(output.pages.back().covered_mask, cell % VBR_GENERATION_PAGE_CELLS);
    }
    return true;
}

bool vbr_generation_capture_controller(const vbr_generation_tracker &                        tracker,
                                       uint32_t                                              child_id,
                                       checkpoint_child_dependency_mode                      dependency_mode,
                                       const std::vector<vbr_checkpoint_generation_stream> & streams,
                                       vbr_checkpoint_generation_controller &                output) {
    if (!tracker.active() || tracker.shadow_unavailable() || !tracker.stable() ||
        dependency_mode != checkpoint_child_dependency_mode::live_guarded) {
        return false;
    }

    output                   = {};
    output.child_id          = child_id;
    output.dependency_mode   = dependency_mode;
    output.pool_uuid         = tracker.pool_identity();
    output.global_generation = tracker.controller_generation();
    output.streams           = streams;
    uint32_t previous_stream  = std::numeric_limits<uint32_t>::max();
    for (const auto & stream : output.streams) {
        if (stream.stream_index >= tracker.stream_count() ||
            (previous_stream != std::numeric_limits<uint32_t>::max() &&
             stream.stream_index <= previous_stream)) {
            output = {};
            return false;
        }
        previous_stream = stream.stream_index;
        for (const auto & page : stream.pages) {
            if (page.page_index >= page_count(tracker.cell_count()) ||
                tracker.page_generation(stream.stream_index, page.page_index) !=
                        page.captured_page_gen) {
                output = {};
                return false;
            }
        }
    }
    output.units.reserve(tracker.unit_count());
    for (uint32_t unit = 0; unit < tracker.unit_count(); ++unit) {
        const auto live_unit = tracker.unit_generation(unit);
        if ((live_unit.publish_seq & 1u) != 0) {
            output = {};
            return false;
        }
        output.units.push_back({
            live_unit.repr_gen,
            live_unit.current_type,
            live_unit.last_source_type,
            live_unit.domain,
            live_unit.promote_hops,
            live_unit.last_transition,
        });
    }

    if (!tracker.stable() || tracker.pool_identity() != output.pool_uuid ||
        tracker.controller_generation() != output.global_generation) {
        output = {};
        return false;
    }
    for (uint32_t unit = 0; unit < tracker.unit_count(); ++unit) {
        if (!unit_equal(output.units[unit], tracker.unit_generation(unit))) {
            output = {};
            return false;
        }
    }
    for (const auto & stream : output.streams) {
        for (const auto & page : stream.pages) {
            if (tracker.page_generation(stream.stream_index, page.page_index) !=
                    page.captured_page_gen) {
                output = {};
                return false;
            }
        }
    }
    return true;
}

// VBR_GENERATION_ELIGIBILITY_AUTHORITY
vbr_checkpoint_eligibility checkpoint_vbr_eligibility(const vbr_checkpoint_generation_record & checkpoint,
                                                      const vbr_generation_live_view &         live) {
    if (!live.capability_applicable) {
        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::capability_not_applicable,
                      vbr_checkpoint_eligibility_category::not_applicable);
    }
    if (checkpoint.status != vbr_checkpoint_generation_status::complete) {
        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::record_unknown,
                      vbr_checkpoint_eligibility_category::generation_unknown);
    }
    if (checkpoint.version != 1) {
        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::record_version);
    }
    if (!live.identity_frontier_eligible ||
        checkpoint.identity_policy_order_digest != live.identity_policy_order_digest) {
        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::identity_or_frontier);
    }
    if (checkpoint.controllers.empty() || checkpoint.controllers.size() != live.controllers.size()) {
        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::controller_shape);
    }

    bool any_live_rebased = false;
    audit_state             audit;
    std::vector<audit_cell> reject_cells;
    vbr_checkpoint_eligibility_reason first_reject = vbr_checkpoint_eligibility_reason::none;
    uint32_t previous_child = std::numeric_limits<uint32_t>::max();
    for (size_t ci = 0; ci < checkpoint.controllers.size(); ++ci) {
        const auto & captured = checkpoint.controllers[ci];
        const auto & current  = live.controllers[ci];
        if (captured.child_id != current.child_id ||
            (previous_child != std::numeric_limits<uint32_t>::max() &&
             captured.child_id <= previous_child)) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::child_order);
        }
        previous_child = captured.child_id;
        if (captured.dependency_mode != current.dependency_mode) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::dependency_mode);
        }
        if (captured.dependency_mode == checkpoint_child_dependency_mode::absent ||
            captured.dependency_mode == checkpoint_child_dependency_mode::payload_complete) {
            if (!captured.streams.empty()) {
                return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::stream_shape);
            }
            continue;
        }
        if (current.tracker == nullptr || !current.tracker->active()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::controller_inactive,
                          vbr_checkpoint_eligibility_category::not_applicable);
        }
        if (!current.tracker->stable()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::controller_unstable);
        }
        const uint64_t serial_at_read_start = current.tracker->mutation_serial();
        std::vector<std::pair<uint32_t, uint64_t>> unit_seq_snapshot;
        unit_seq_snapshot.reserve(captured.units.size());
        if (captured.pool_uuid != current.tracker->pool_identity()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::pool_uuid);
        }
        if (captured.global_generation != current.tracker->controller_generation()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::global_generation);
        }
        if (captured.units.size() != current.tracker->unit_count()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::unit_shape);
        }
        for (size_t ui = 0; ui < captured.units.size(); ++ui) {
            const auto current_unit = current.tracker->unit_generation(static_cast<uint32_t>(ui));
            if ((current_unit.publish_seq & 1u) != 0) {
                return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::unit_unstable);
            }
            unit_seq_snapshot.emplace_back(static_cast<uint32_t>(ui), current_unit.publish_seq);
            if (unit_equal(captured.units[ui], current_unit)) {
                continue;
            }
            if (!unit_live_rebased(captured.units[ui], current_unit)) {
                return reject(live.legacy_eligible, unit_live_rebased_shape(captured.units[ui], current_unit) ?
                                                        vbr_checkpoint_eligibility_reason::live_rebased_transition :
                                                        vbr_checkpoint_eligibility_reason::unit_generation);
            }
            any_live_rebased = true;
        }

        if (captured.streams.size() != current.streams.size()) {
            return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::stream_shape);
        }
        for (size_t si = 0; si < captured.streams.size(); ++si) {
            const auto & stored_stream = captured.streams[si];
            const auto & live_stream   = current.streams[si];
            if (stored_stream.stream_index != live_stream.stream_index ||
                stored_stream.stream_index >= current.tracker->stream_count()) {
                return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::stream_order);
            }
            if (stored_stream.dependency_seq_id < 0 || stored_stream.computation_frontier < 0 ||
                live_stream.cell_has_seq == nullptr) {
                return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::stream_shape);
            }
            if (stored_stream.dependency_seq_id != live_stream.dependency_seq_id ||
                stored_stream.computation_frontier != live_stream.computation_frontier) {
                return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::identity_or_frontier);
            }

            uint32_t covered_count  = 0;
            uint32_t lost_in_stream = 0;
            uint32_t previous_page  = std::numeric_limits<uint32_t>::max();
            for (const auto & page : stored_stream.pages) {
                if ((previous_page != std::numeric_limits<uint32_t>::max() && page.page_index <= previous_page) ||
                    mask_popcount(page.covered_mask) == 0) {
                    return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::malformed_page_refs);
                }
                previous_page = page.page_index;
                if (page.page_index >= page_count(current.tracker->cell_count())) {
                    return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::page_out_of_range);
                }

                const uint32_t current_page_gen =
                    current.tracker->page_generation(stored_stream.stream_index, page.page_index);
                const bool     fast_match = current_page_gen == page.captured_page_gen;
                if (!fast_match) {
                    audit.refinement_used = true;
                }
                if (current.tracker->page_destructive_generation(stored_stream.stream_index, page.page_index) >
                        page.captured_page_gen) {
                    audit.saw_destructive = true;
                }
                if (current.tracker->page_import_generation(stored_stream.stream_index, page.page_index) >
                        page.captured_page_gen) {
                    audit.saw_import = true;
                }
                const uint32_t base = page.page_index * VBR_GENERATION_PAGE_CELLS;
                for (uint32_t offset = 0; offset < VBR_GENERATION_PAGE_CELLS; ++offset) {
                    if (!mask_test(page.covered_mask, offset)) {
                        continue;
                    }
                    const uint32_t cell = base + offset;
                    if (cell >= current.tracker->cell_count()) {
                        return reject(live.legacy_eligible, vbr_checkpoint_eligibility_reason::page_out_of_range);
                    }
                    ++covered_count;
                    // §5.5 requires inspecting EVERY rejecting covered cell — collect instead of
                    // early-returning so classification sees the complete reject set.
                    const bool dep_changed =
                        !fast_match && current.tracker->dependency_generation(stored_stream.stream_index, cell) >
                                           page.captured_page_gen;
                    const bool membership_lost =
                        !live_stream.cell_has_seq(live_stream.membership_context, stored_stream.stream_index,
                                                  cell, stored_stream.dependency_seq_id);
                    if (membership_lost) {
                        ++lost_in_stream;
                    }
                    if (dep_changed || membership_lost) {
                        if (first_reject == vbr_checkpoint_eligibility_reason::none) {
                            first_reject = dep_changed ? vbr_checkpoint_eligibility_reason::dependency_changed
                                                       : vbr_checkpoint_eligibility_reason::dependency_membership_lost;
                        }
                        ++audit.rejecting_cells;
                        if (audit.rejecting_cells > REJECT_INSPECT_CAP) {
                            audit.overflowed = true;
                            continue;
                        }
                        if (reject_cells.empty()) {
                            reject_cells.reserve(64);
                        }
                        audit_cell rc;
                        rc.tracker         = current.tracker;
                        rc.membership_lost = membership_lost;
                        rc.doubly_rejected = dep_changed && membership_lost;
                        rc.in_range        = dep_changed
                                ? current.tracker->dependency_in_range(stored_stream.stream_index, cell)
                                : current.tracker->membership_in_range(stored_stream.stream_index, cell);
                        rc.provenance      = dep_changed
                                ? current.tracker->dependency_provenance(stored_stream.stream_index, cell)
                                : current.tracker->membership_provenance(stored_stream.stream_index, cell);
                        rc.extent          = dep_changed
                                ? current.tracker->dependency_extent(stored_stream.stream_index, cell)
                                : current.tracker->membership_extent(stored_stream.stream_index, cell);
                        rc.current_pos     = live_stream.cell_pos != nullptr
                                ? live_stream.cell_pos(live_stream.membership_context,
                                                       stored_stream.stream_index, cell)
                                : -1;
                        rc.frontier        = stored_stream.computation_frontier;
                        rc.dep_seq         = stored_stream.dependency_seq_id;
                        reject_cells.push_back(rc);
                    }
                }
            }
            // F7: cardinality agreement is validated in BOTH outcomes. Clean streams must match
            // exactly; rejecting streams must reconcile as captured-minus-lost (any expansion or
            // malformed count forces the whole reject set to unexplained).
            if (covered_count != stored_stream.captured_dependency_count) {
                audit.cardinality_violated = true;
            }
            const uint32_t expected_live =
                stored_stream.captured_dependency_count >= lost_in_stream
                    ? stored_stream.captured_dependency_count - lost_in_stream
                    : 0;
            if (live_stream.exact_dependency_count != expected_live ||
                stored_stream.captured_dependency_count < lost_in_stream) {
                audit.cardinality_violated = true;
            }
            if (first_reject == vbr_checkpoint_eligibility_reason::none && audit.cardinality_violated) {
                auto result            = reject(live.legacy_eligible,
                                                vbr_checkpoint_eligibility_reason::dependency_cardinality);
                result.tombstone_class = vbr_expected_tombstone_class::unexplained;
                fill_audit(result, audit);
                return result;
            }
        }

        // F9 (v3.2): per-unit publish_seq must be IDENTICAL after all reads — unit publication
        // does not move the controller serial, so this is the only proof no unit published
        // even->odd->even mid-evaluation. Tuple copies themselves are race-free by the units
        // mutex; this catches the interleaving.
        for (const auto & [ui, seq_before] : unit_seq_snapshot) {
            if (current.tracker->unit_generation(ui).publish_seq != seq_before) {
                auto result = reject(live.legacy_eligible,
                                     vbr_checkpoint_eligibility_reason::unit_unstable);
                fill_audit(result, audit);
                return result;
            }
        }
        // F9: post-read stability recheck — a mutation that raced these reads (even -> odd ->
        // even) invalidates everything read above. Reject rather than re-read: the caller's
        // next scan sees a coherent state.
        if (current.tracker->mutation_serial() != serial_at_read_start ||
            !current.tracker->stable()) {
            auto result = reject(live.legacy_eligible,
                                 vbr_checkpoint_eligibility_reason::controller_unstable);
            fill_audit(result, audit);
            return result;
        }
    }

    if (first_reject != vbr_checkpoint_eligibility_reason::none) {
        auto result = reject(live.legacy_eligible, first_reject);
        result.tombstone_class =
            classify_expected_tombstone(reject_cells, audit.overflowed, audit.cardinality_violated);
        fill_audit(result, audit);
        return result;
    }

    vbr_checkpoint_eligibility result;
    result.legacy              = live.legacy_eligible;
    result.strict              = !any_live_rebased;
    result.live_rebased_shadow = any_live_rebased;
    result.category            = any_live_rebased ? vbr_checkpoint_eligibility_category::live_rebased_shadow_accept :
                                                    vbr_checkpoint_eligibility_category::strict_accept;
    result.reason              = vbr_checkpoint_eligibility_reason::none;
    fill_audit(result, audit);
    return result;
}
