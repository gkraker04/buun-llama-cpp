#pragma once

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>

// Process-local coordination identity. It is deliberately a strong internal type so it cannot
// accidentally enter a checkpoint/state envelope. The public C freeze API exposes only its
// opaque uint64_t value.
struct vbr_operation_id {
    uint64_t value = 0;

    explicit operator bool() const {
        return value != 0;
    }
};

inline bool operator==(vbr_operation_id lhs, vbr_operation_id rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(vbr_operation_id lhs, vbr_operation_id rhs) {
    return !(lhs == rhs);
}

// Rev-9 closed provenance vocabulary. New values require updating the registry inventory below;
// its compile-time coverage checks intentionally make an unregistered extension fail the build.
enum class vbr_mutation_family : uint8_t {
    append,
    occupied_reuse,
    trim,
    seq_share,
    seq_keep,
    shift,
    import,
    restore,
    clear,
    reset,
    degrade,
    promote,
    shed,
    recovery,
    count,
};

enum class vbr_operation_class : uint8_t {
    ordinary_decode,
    checkpoint_restore,
    restore_one_behind_trim,
    swa_wrap,
    explicit_destructive_trim,
    dependency_seq_remove,
    speculative_backup,
    prompt_share,
    sibling_owner_remove,
    host_import,
    state_api,
    controller,
    count,
};

enum class vbr_operation_kind : uint8_t {
    retier_freeze,
    decode,
    sequence_edit,
    checkpoint_restore,
    state_import,
    state_export,
    controller_retier,
    recovery,
    count,
};

// VBR_OPERATION_KIND_EXHAUSTIVE
constexpr std::array<const char *, static_cast<size_t>(vbr_operation_kind::count)>
        VBR_OPERATION_KIND_NAMES = {{
    "retier_freeze",
    "decode",
    "sequence_edit",
    "checkpoint_restore",
    "state_import",
    "state_export",
    "controller_retier",
    "recovery",
}};
static_assert(VBR_OPERATION_KIND_NAMES.size() ==
        static_cast<size_t>(vbr_operation_kind::count),
        "every VBR operation kind must have closed registry vocabulary");

enum class vbr_operation_phase : uint8_t {
    root,
    prepare,
    mutate,
    publish,
    cleanup,
    recovery,
    count,
};

struct vbr_operation_range {
    llama_pos p0 = -1;
    llama_pos p1 = -1;
};

struct vbr_operation_binding {
    vbr_operation_id    operation_id = {};
    vbr_operation_kind  kind         = vbr_operation_kind::retier_freeze;
    llama_seq_id        seq_id       = -1;
    vbr_operation_range range        = {};
    vbr_operation_phase child_phase  = vbr_operation_phase::root;
};

enum class vbr_mutation_registrant : uint8_t {
    apply_ubatch_append,
    apply_ubatch_occupied_reuse,
    seq_rm,
    seq_cp,
    seq_keep,
    seq_add,
    seq_div,
    state_read_meta,
    state_read_data,
    state_read_install,
    state_read_cleanup,
    whole_import,
    explicit_restore_adopt,
    clear,
    full_reset,
    degrade_next,
    promote_next,
    execute_shed,
    authenticated_recovery,
    count,
};

constexpr uint16_t vbr_operation_class_bit(vbr_operation_class operation_class) {
    return uint16_t(1u) << static_cast<uint8_t>(operation_class);
}
static_assert(static_cast<size_t>(vbr_operation_class::count) <= 16,
        "VBR registry allowed-class mask must be widened with the closed class enum");

struct vbr_mutation_registration {
    vbr_mutation_registrant registrant;
    vbr_mutation_family     family;
    vbr_operation_phase     phase;
    uint16_t                allowed_classes;
};

// Closed mutation-registration inventory. apply_ubatch owns assignment registration; find_slot
// remains a read-only planner and is intentionally absent. This table is inert in A0; subsequent
// WS-A commits attach dispatch at these audited sites without reopening the vocabulary.
constexpr std::array<vbr_mutation_registration,
        static_cast<size_t>(vbr_mutation_registrant::count)> VBR_MUTATION_REGISTRY = {{
    { vbr_mutation_registrant::apply_ubatch_append,         vbr_mutation_family::append,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::ordinary_decode) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) },
    { vbr_mutation_registrant::apply_ubatch_occupied_reuse, vbr_mutation_family::occupied_reuse,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::ordinary_decode) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) },
    { vbr_mutation_registrant::seq_rm,                      vbr_mutation_family::trim,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::restore_one_behind_trim) |
      vbr_operation_class_bit(vbr_operation_class::swa_wrap) |
      vbr_operation_class_bit(vbr_operation_class::explicit_destructive_trim) |
      vbr_operation_class_bit(vbr_operation_class::dependency_seq_remove) |
      vbr_operation_class_bit(vbr_operation_class::speculative_backup) |
      vbr_operation_class_bit(vbr_operation_class::sibling_owner_remove) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::seq_cp,                      vbr_mutation_family::seq_share,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::speculative_backup) |
      vbr_operation_class_bit(vbr_operation_class::prompt_share) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::seq_keep,                    vbr_mutation_family::seq_keep,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::seq_add,                     vbr_mutation_family::shift,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::seq_div,                     vbr_mutation_family::shift,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::state_read_meta,             vbr_mutation_family::import,
      vbr_operation_phase::prepare,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_data,             vbr_mutation_family::import,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_install,          vbr_mutation_family::import,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::state_read_cleanup,          vbr_mutation_family::import,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) |
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::whole_import,                vbr_mutation_family::import,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::host_import) |
      vbr_operation_class_bit(vbr_operation_class::state_api) },
    { vbr_mutation_registrant::explicit_restore_adopt,      vbr_mutation_family::restore,
      vbr_operation_phase::publish,
      vbr_operation_class_bit(vbr_operation_class::checkpoint_restore) },
    { vbr_mutation_registrant::clear,                       vbr_mutation_family::clear,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::state_api) |
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::full_reset,                  vbr_mutation_family::reset,
      vbr_operation_phase::cleanup,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::degrade_next,                vbr_mutation_family::degrade,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::promote_next,                vbr_mutation_family::promote,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::execute_shed,                vbr_mutation_family::shed,
      vbr_operation_phase::mutate,
      vbr_operation_class_bit(vbr_operation_class::controller) },
    { vbr_mutation_registrant::authenticated_recovery,      vbr_mutation_family::recovery,
      vbr_operation_phase::recovery,
      vbr_operation_class_bit(vbr_operation_class::controller) },
}};

constexpr bool vbr_mutation_registry_is_exhaustive() {
    std::array<bool, static_cast<size_t>(vbr_mutation_registrant::count)> registrants = {};
    std::array<bool, static_cast<size_t>(vbr_mutation_family::count)> families = {};
    std::array<bool, static_cast<size_t>(vbr_operation_class::count)> classes = {};
    std::array<bool, static_cast<size_t>(vbr_operation_phase::count)> phases = {};

    for (const auto & registration : VBR_MUTATION_REGISTRY) {
        registrants[static_cast<size_t>(registration.registrant)] = true;
        families[static_cast<size_t>(registration.family)] = true;
        phases[static_cast<size_t>(registration.phase)] = true;
        for (size_t i = 0; i < classes.size(); ++i) {
            if ((registration.allowed_classes & (uint16_t(1u) << i)) != 0) {
                classes[i] = true;
            }
        }
    }
    for (bool present : registrants) {
        if (!present) {
            return false;
        }
    }
    for (bool present : families) {
        if (!present) {
            return false;
        }
    }
    for (bool present : classes) {
        if (!present) {
            return false;
        }
    }
    // Root is the operation owner rather than a mutation site; all mutation phases must occur.
    for (size_t i = 1; i < phases.size(); ++i) {
        if (!phases[i]) {
            return false;
        }
    }
    return true;
}

// VBR_MUTATION_INVENTORY_EXHAUSTIVE
static_assert(vbr_mutation_registry_is_exhaustive(),
        "VBR mutation registrants, families, classes, and phases must remain closed and exhaustive");

enum class vbr_stable_read_registrant : uint8_t {
    checkpoint_capture,
    state_export,
    oracle_read,
    count,
};

// Stable readers are registry participants but do not stamp a mutation family.
// VBR_STABLE_READ_INVENTORY_EXHAUSTIVE
constexpr std::array<vbr_stable_read_registrant,
        static_cast<size_t>(vbr_stable_read_registrant::count)> VBR_STABLE_READ_REGISTRY = {{
    vbr_stable_read_registrant::checkpoint_capture,
    vbr_stable_read_registrant::state_export,
    vbr_stable_read_registrant::oracle_read,
}};
constexpr bool vbr_stable_read_registry_is_exhaustive() {
    std::array<bool, static_cast<size_t>(vbr_stable_read_registrant::count)> present = {};
    for (vbr_stable_read_registrant registrant : VBR_STABLE_READ_REGISTRY) {
        present[static_cast<size_t>(registrant)] = true;
    }
    for (bool registered : present) {
        if (!registered) {
            return false;
        }
    }
    return true;
}
static_assert(vbr_stable_read_registry_is_exhaustive(),
        "capture, export, and oracle stable-read guards must stay exhaustive");

// The sole process-global minting entry point. Composite memories must only forward its result.
vbr_operation_id vbr_operation_registry_begin(vbr_operation_binding & binding);
bool vbr_operation_registry_end(vbr_operation_id operation_id);
bool vbr_operation_registry_is_live(vbr_operation_id operation_id);

// Internal owner for operations whose call shape permits ordinary C++ lifetime management.
// The legacy public freeze begin/end ABI is manually paired at its C boundary, while its sole
// server caller is already protected by server_vbr_retier_freeze_scope.
class vbr_operation_registry_guard {
public:
    explicit vbr_operation_registry_guard(vbr_operation_binding binding);
    ~vbr_operation_registry_guard();

    vbr_operation_registry_guard(const vbr_operation_registry_guard &) = delete;
    vbr_operation_registry_guard & operator=(const vbr_operation_registry_guard &) = delete;
    vbr_operation_registry_guard(vbr_operation_registry_guard &&) = delete;
    vbr_operation_registry_guard & operator=(vbr_operation_registry_guard &&) = delete;

    bool active() const {
        return static_cast<bool>(binding_.operation_id);
    }

    const vbr_operation_binding & binding() const {
        return binding_;
    }

    bool finish();

private:
    vbr_operation_binding binding_;
};
