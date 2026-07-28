#include "llama-vbr-generation-oracle.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace {

bool env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

uint64_t hash_bytes(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool oracle_dependency(llama_pos computation_frontier, const vbr_generation_oracle_cell & cell) {
    return cell.has_dependency_seq && cell.position >= 0 && cell.position < computation_frontier &&
           cell.attention_visible && !cell.payload_supplied;
}

vbr_generation_oracle_baseline independently_reconstruct(
    llama_pos                                       computation_frontier,
    const std::vector<vbr_generation_oracle_cell> & canonical_cells) {
    vbr_generation_oracle_baseline result;
    result.dependency_byte_hash = UINT64_C(1469598103934665603);
    result.complete             = true;

    std::vector<const vbr_generation_oracle_cell *> dependencies;
    for (const auto & cell : canonical_cells) {
        if (!oracle_dependency(computation_frontier, cell)) {
            continue;
        }
        dependencies.push_back(&cell);
    }
    std::sort(dependencies.begin(), dependencies.end(),
              [](const auto * lhs, const auto * rhs) { return lhs->physical_cell < rhs->physical_cell; });

    for (const auto * cell : dependencies) {
        if (!result.dependency_cells.empty() &&
            result.dependency_cells.back() == cell->physical_cell) {
            result.complete = false;
        }
        if (cell->dependency_bytes.empty()) {
            result.complete = false;
        }
        result.dependency_cells.push_back(cell->physical_cell);
        result.dependency_byte_hash =
            hash_bytes(result.dependency_byte_hash, &cell->physical_cell, sizeof(cell->physical_cell));
        const size_t byte_count = cell->dependency_bytes.size();
        result.dependency_byte_hash =
            hash_bytes(result.dependency_byte_hash, &byte_count, sizeof(byte_count));
        if (!cell->dependency_bytes.empty()) {
            result.dependency_byte_hash =
                hash_bytes(result.dependency_byte_hash, cell->dependency_bytes.data(),
                           cell->dependency_bytes.size());
        }
    }
    return result;
}

std::vector<uint32_t> production_covered_set(const vbr_checkpoint_generation_stream & production_record) {
    std::vector<uint32_t> result;
    for (const auto & page : production_record.pages) {
        const uint32_t base = page.page_index * VBR_GENERATION_PAGE_CELLS;
        for (uint32_t offset = 0; offset < VBR_GENERATION_PAGE_CELLS; ++offset) {
            if ((page.covered_mask[offset / 64] & (uint64_t(1) << (offset % 64))) != 0) {
                result.push_back(base + offset);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

bool vbr_generation_oracle_enabled() {
    // Deliberately probed per call: tests toggle the gate mid-process, and the call is
    // boundary-rate (efficiency review's static-once idea traded away test agility for ~nothing).
    return env_flag("VBR_GENERATION_ORACLE");
}

vbr_generation_oracle_baseline vbr_generation_oracle_capture(
    llama_pos                                       computation_frontier,
    const std::vector<vbr_generation_oracle_cell> & canonical_cells) {
    if (!vbr_generation_oracle_enabled()) {
        return {};
    }
    return independently_reconstruct(computation_frontier, canonical_cells);
}

vbr_generation_oracle_result vbr_generation_oracle_audit(
    llama_pos                                       computation_frontier,
    const std::vector<vbr_generation_oracle_cell> & canonical_cells,
    const vbr_generation_oracle_baseline &          baseline,
    const vbr_checkpoint_generation_stream &        production_record) {
    vbr_generation_oracle_result result;
    result.enabled = vbr_generation_oracle_enabled();
    if (!result.enabled) {
        return result;
    }

    const auto independent       = independently_reconstruct(computation_frontier, canonical_cells);
    const auto covered           = production_covered_set(production_record);
    result.independent_count     = static_cast<uint32_t>(independent.dependency_cells.size());
    result.independent_byte_hash = independent.dependency_byte_hash;
    result.complete = independent.complete && baseline.complete;
    result.set_equal =
        result.complete &&
        independent.dependency_cells == baseline.dependency_cells && independent.dependency_cells == covered &&
        static_cast<size_t>(production_record.captured_dependency_count) == independent.dependency_cells.size();
    result.bytes_equal = result.complete &&
                         independent.dependency_byte_hash == baseline.dependency_byte_hash;
    return result;
}

bool vbr_generation_oracle_audit_due(bool                            destructive_or_import_crossing,
                                     const std::array<uint8_t, 32> & identity_digest,
                                     bool                            forced_audit) {
    if (forced_audit || destructive_or_import_crossing) {
        return true;
    }
    // deterministic 1/256 append-only sample: a pure function of the identity digest
    return identity_digest[0] == 0;
}

bool vbr_generation_oracle_audit_forced() {
    return env_flag("VBR_GENERATION_FORCE_AUDIT");
}

void vbr_generation_oracle_inject(vbr_generation_oracle_result & result) {
    const char * fault = std::getenv("VBR_GENERATION_ORACLE_INJECT");
    if (fault == nullptr) {
        return;
    }
    if (std::strcmp(fault, "set") == 0) {
        result.set_equal = false;
    } else if (std::strcmp(fault, "bytes") == 0) {
        result.bytes_equal = false;
    } else if (std::strcmp(fault, "unavailable") == 0) {
        result.complete = false;
    }
}
