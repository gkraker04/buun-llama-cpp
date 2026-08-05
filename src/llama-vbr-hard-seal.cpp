#include "llama-vbr-hard-seal.h"

#include <utility>

bool vbr_classify_hard_seal(
        const std::vector<vbr_degrade_step> & order,
        uint8_t seal_tier,
        vbr_hard_seal_classification & out) noexcept {
    vbr_hard_seal_classification result;
    try {
        result.affected.reserve(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i].tier == seal_tier) {
                result.affected.push_back({
                    order[i].il, order[i].is_v != 0, i });
            }
        }
    } catch (...) {
        return false;
    }
    out = std::move(result);
    return true;
}
