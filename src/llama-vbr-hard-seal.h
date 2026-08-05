#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// One measured degrade-order step. Shared with the E1 read-only hard-seal
// classifier so controller policy continues to have one canonical order.
struct vbr_degrade_step {
    uint8_t il = 0;
    uint8_t is_v = 0;
    uint8_t tier = 0;
};

struct vbr_hard_seal_subject {
    uint8_t il = 0;
    bool is_v = false;
    size_t order_ordinal = 0;
};

struct vbr_hard_seal_classification {
    std::vector<vbr_hard_seal_subject> affected;
};

// Frozen E1 default: crossing from the restorable T8 band into T4.
inline constexpr ggml_type VBR_HARD_SEAL_DEFAULT_FLOOR =
    GGML_TYPE_TURBO4_0;

// Central, read-only classification kernel. It never changes the order,
// cursor, floor, controller serial, or any backend state.
bool vbr_classify_hard_seal(
    const std::vector<vbr_degrade_step> & order,
    uint8_t seal_tier,
    vbr_hard_seal_classification & out) noexcept;
