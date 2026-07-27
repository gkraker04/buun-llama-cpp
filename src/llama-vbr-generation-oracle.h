#pragma once

#include "llama-vbr-generation-types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Canonical read-only observation supplied to the debug oracle. The adapter must populate this
// directly from llama_kv_cells and the serializer traversal; production ownership indices and
// cached manifests are forbidden inputs.
struct vbr_generation_oracle_cell {
    uint32_t        physical_cell      = 0;
    llama_pos       position           = -1;
    bool            has_dependency_seq = false;
    bool            attention_visible  = false;
    bool            payload_supplied   = false;
    const uint8_t * bytes              = nullptr;
    size_t          byte_count         = 0;
};

struct vbr_generation_oracle_baseline {
    std::vector<uint32_t> dependency_cells;
    uint64_t              dependency_byte_hash = 0;
    bool                  complete             = false;
};

struct vbr_generation_oracle_result {
    bool     enabled               = false;
    bool     complete              = false;
    bool     set_equal             = false;
    bool     bytes_equal           = false;
    uint32_t independent_count     = 0;
    uint64_t independent_byte_hash = 0;
};

bool vbr_generation_oracle_enabled();

// Both calls independently derive the dependency set from canonical observations. audit() then
// diffs that independently-derived set against both the capture baseline and the production
// covered mask. The oracle is debug evidence only; checkpoint_vbr_eligibility() never calls it.
vbr_generation_oracle_baseline vbr_generation_oracle_capture(
    llama_pos                                       computation_frontier,
    const std::vector<vbr_generation_oracle_cell> & canonical_cells);

vbr_generation_oracle_result vbr_generation_oracle_audit(
    llama_pos                                       computation_frontier,
    const std::vector<vbr_generation_oracle_cell> & canonical_cells,
    const vbr_generation_oracle_baseline &          baseline,
    const vbr_checkpoint_generation_stream &        production_record);

// §6.2 strict-accept audit sampling policy. Pure and SHIPPED DISABLED: no production caller
// until the commit-3 selection/evaluation consumer integrates it; never an admission input.
// Destructive/import crossings always audit; append-only observations audit deterministically
// at 1/256 keyed on the identity digest (same digest => same verdict); forced always audits.
bool vbr_generation_oracle_audit_due(bool                            destructive_or_import_crossing,
                                     const std::array<uint8_t, 32> & identity_digest,
                                     bool                            forced_audit);

// Env probe for the forced-audit override (VBR_GENERATION_FORCE_AUDIT), probed per call like
// the oracle gate so tests can toggle it mid-process.
bool vbr_generation_oracle_audit_forced();
