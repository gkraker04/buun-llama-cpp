#include "llama-vbr-hard-seal.h"

#include <cstdio>

int main() {
    const std::vector<vbr_degrade_step> order = {
        { 2, 0, 0 },
        { 7, 1, 1 },
        { 2, 0, 1 },
        { 7, 1, 2 },
    };
    vbr_hard_seal_classification out;
    if (!vbr_classify_hard_seal(order, 1, out) ||
        VBR_HARD_SEAL_DEFAULT_FLOOR != GGML_TYPE_TURBO4_0 ||
        out.affected.size() != 2 ||
        out.affected[0].il != 7 || !out.affected[0].is_v ||
        out.affected[0].order_ordinal != 1 ||
        out.affected[1].il != 2 || out.affected[1].is_v ||
        out.affected[1].order_ordinal != 2) {
        std::fputs("VBR hard-seal classification failed\n", stderr);
        return 1;
    }
    std::puts("VBR hard-seal classification passed");
    return 0;
}
