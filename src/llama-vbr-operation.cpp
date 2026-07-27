#include "llama-vbr-operation.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>

namespace {

constexpr size_t VBR_OPERATION_REGISTRY_CAPACITY = 4096;

// VBR_OPERATION_ALLOCATOR_DEFINITION
std::atomic<uint64_t> g_vbr_next_operation_id { 1 };
std::atomic<bool> g_vbr_operation_id_exhausted { false };
std::array<std::atomic<uint64_t>, VBR_OPERATION_REGISTRY_CAPACITY> g_vbr_live_operations {};

vbr_operation_id vbr_operation_allocate() {
    if (g_vbr_operation_id_exhausted.load(std::memory_order_acquire)) {
        return {};
    }

    uint64_t expected = g_vbr_next_operation_id.load(std::memory_order_relaxed);
    for (;;) {
        if (expected == 0) {
            g_vbr_operation_id_exhausted.store(true, std::memory_order_release);
            return {};
        }

        const uint64_t next =
            expected == std::numeric_limits<uint64_t>::max() ? 0 : expected + 1;
        if (g_vbr_next_operation_id.compare_exchange_weak(
                    expected, next, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            if (next == 0) {
                g_vbr_operation_id_exhausted.store(true, std::memory_order_release);
            }
            return { expected };
        }
    }
}

size_t vbr_operation_slot(vbr_operation_id operation_id) {
    return static_cast<size_t>(operation_id.value % VBR_OPERATION_REGISTRY_CAPACITY);
}

} // namespace

vbr_operation_id vbr_operation_registry_begin(vbr_operation_binding & binding) {
    // VBR_OPERATION_MINT_SITE
    if (binding.operation_id) {
        return {};
    }
    if (static_cast<uint8_t>(binding.kind) >=
            static_cast<uint8_t>(vbr_operation_kind::count) ||
        static_cast<uint8_t>(binding.child_phase) >=
            static_cast<uint8_t>(vbr_operation_phase::count)) {
        return {};
    }

    const vbr_operation_id operation_id = vbr_operation_allocate();
    if (!operation_id) {
        return {};
    }

    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        auto & slot = g_vbr_live_operations[(first + i) % VBR_OPERATION_REGISTRY_CAPACITY];
        uint64_t empty = 0;
        if (slot.compare_exchange_strong(
                    empty, operation_id.value, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            binding.operation_id = operation_id;
            return operation_id;
        }
    }

    // The ID is intentionally burned: allocation never reuses an identity even when the bounded
    // live-operation registry is temporarily full.
    return {};
}

bool vbr_operation_registry_end(vbr_operation_id operation_id) {
    if (!operation_id) {
        return false;
    }

    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        auto & slot = g_vbr_live_operations[(first + i) % VBR_OPERATION_REGISTRY_CAPACITY];
        uint64_t expected = operation_id.value;
        if (slot.compare_exchange_strong(
                    expected, 0, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool vbr_operation_registry_is_live(vbr_operation_id operation_id) {
    if (!operation_id) {
        return false;
    }

    const size_t first = vbr_operation_slot(operation_id);
    for (size_t i = 0; i < VBR_OPERATION_REGISTRY_CAPACITY; ++i) {
        if (g_vbr_live_operations[(first + i) % VBR_OPERATION_REGISTRY_CAPACITY].load(
                    std::memory_order_acquire) == operation_id.value) {
            return true;
        }
    }
    return false;
}

vbr_operation_registry_guard::vbr_operation_registry_guard(vbr_operation_binding binding) :
        binding_(binding) {
    binding_.operation_id = {};
    vbr_operation_registry_begin(binding_);
}

vbr_operation_registry_guard::~vbr_operation_registry_guard() {
    if (active() && !finish()) {
        std::abort();
    }
}

bool vbr_operation_registry_guard::finish() {
    if (!active()) {
        return false;
    }
    const vbr_operation_id operation_id = binding_.operation_id;
    if (!vbr_operation_registry_end(operation_id)) {
        return false;
    }
    binding_.operation_id = {};
    return true;
}
