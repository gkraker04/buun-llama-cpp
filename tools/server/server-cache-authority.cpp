#include "server-cache-authority.h"

#include "server-common.h"
#include "server-task.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace {

bool add_checked(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        out = 0;
        return false;
    }
    out = a + b;
    return true;
}

} // namespace

bool server_cache_weighted_price_us(
        long double base_us,
        uint32_t weight_milli,
        uint64_t & out) noexcept {
    out = 0;
    const long double weighted = base_us * weight_milli /
        SERVER_CACHE_HOST_WEIGHT_SCALE;
    if (!std::isfinite((double) weighted) || weighted < 0.0L ||
        weighted > (long double) std::numeric_limits<long long>::max()) {
        return false;
    }
    out = uint64_t(std::llround(weighted));
    return true;
}

bool server_cache_retention_weight_milli(
        bool soft_leased,
        bool main_family,
        uint32_t additional_weight_milli,
        uint32_t & weight_milli) noexcept {
    weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    return (!soft_leased || server_cache_multiply_retention_weight(
                weight_milli, SERVER_CACHE_HOST_SOFT_LEASE_WEIGHT)) &&
           (!main_family || server_cache_multiply_retention_weight(
                weight_milli, SERVER_CACHE_HOST_MAIN_FAMILY_WEIGHT)) &&
           server_cache_multiply_retention_weight(
                weight_milli, additional_weight_milli);
}

bool server_cache_host_retention_price_us(
        const common_cache_plan_calib & calib,
        uint64_t bytes,
        bool soft_leased,
        bool main_family,
        uint32_t & weight_milli,
        uint64_t & price_us,
        uint32_t additional_weight_milli) noexcept {
    price_us = 0;
    if (!server_cache_retention_weight_milli(
            soft_leased, main_family, additional_weight_milli,
            weight_milli)) {
        return false;
    }
    double restore_us = 0.0;
    double workspace_us = 0.0;
    return common_cache_plan_restore_us(
               calib, bytes, restore_us, workspace_us) &&
           server_cache_weighted_price_us(
               (long double) restore_us + workspace_us,
               weight_milli, price_us);
}

server_cache_checkpoint_trade_plan server_cache_plan_checkpoint_thinning(
        const std::vector<server_cache_checkpoint_trade_input> & candidates,
        const common_cache_plan_calib * calib) noexcept {
    server_cache_checkpoint_trade_plan out;
    if (!calib || !std::isfinite(calib->replay_us_per_token) ||
        calib->replay_us_per_token < 0.0) {
        out.reason = common_cache_plan_destruction_reason::profile_unfitted;
        return out;
    }
    common_cache_plan_destruction_reason refusal =
        common_cache_plan_destruction_reason::recovery_unavailable;
    server_cache_checkpoint_protection protection =
        server_cache_checkpoint_protection::none;
    try {
        for (const auto & candidate : candidates) {
            if (candidate.artifact.v == 0 || !candidate.identity_known ||
                candidate.payload_bytes == 0 ||
                candidate.weight_milli == 0) {
                refusal =
                    common_cache_plan_destruction_reason::manifest_incomplete;
                continue;
            }
            if (candidate.seam_heuristic_protected ||
                candidate.mandatory_anchor) {
                refusal =
                    common_cache_plan_destruction_reason::mandatory_anchor;
                if (candidate.seam_heuristic_protected) {
                    protection =
                        server_cache_checkpoint_protection::seam_heuristic;
                } else if (protection !=
                        server_cache_checkpoint_protection::seam_heuristic) {
                    protection =
                        server_cache_checkpoint_protection::mandatory_anchor;
                }
                continue;
            }
            if (candidate.hard_leased) {
                refusal =
                    common_cache_plan_destruction_reason::hard_lease_blocked;
                if (protection ==
                        server_cache_checkpoint_protection::none) {
                    protection =
                        server_cache_checkpoint_protection::hard_lease;
                }
                continue;
            }
            if (!candidate.recovery_available ||
                candidate.recovery_ordinal == UINT32_MAX) {
                refusal =
                    common_cache_plan_destruction_reason::recovery_unavailable;
                continue;
            }
            double restore_us = 0.0;
            double workspace_us = 0.0;
            if (!common_cache_plan_restore_us(
                    *calib, candidate.payload_bytes,
                    restore_us, workspace_us)) {
                refusal =
                    common_cache_plan_destruction_reason::capacity_refused;
                continue;
            }
            const long double replay_us =
                (long double) candidate.replay_tokens *
                calib->replay_us_per_token;
            const long double base = replay_us + restore_us + workspace_us;
            uint64_t price = 0;
            if (!server_cache_weighted_price_us(
                    base, candidate.weight_milli, price)) {
                refusal =
                    common_cache_plan_destruction_reason::capacity_refused;
                continue;
            }
            const auto key = std::make_tuple(
                price, candidate.stable_id, candidate.ordinal);
            const auto best = std::make_tuple(
                out.price_us, out.stable_id, out.ordinal);
            if (!out.selected || key < best) {
                out.selected = true;
                out.ordinal = candidate.ordinal;
                out.recovery_ordinal = candidate.recovery_ordinal;
                out.price_us = price;
                out.stable_id = candidate.stable_id;
                out.weight_milli = candidate.weight_milli;
                out.protection =
                    server_cache_checkpoint_protection::none;
                out.reason = common_cache_plan_destruction_reason::none;
            }
        }
    } catch (...) {
        out = {};
        out.reason = common_cache_plan_destruction_reason::internal_fault;
    }
    if (!out.selected) {
        out.protection = protection;
        out.reason = refusal;
        switch (protection) {
            case server_cache_checkpoint_protection::seam_heuristic:
            case server_cache_checkpoint_protection::mandatory_anchor:
                out.reason =
                    common_cache_plan_destruction_reason::mandatory_anchor;
                break;
            case server_cache_checkpoint_protection::hard_lease:
                out.reason =
                    common_cache_plan_destruction_reason::hard_lease_blocked;
                break;
            case server_cache_checkpoint_protection::none:
            case server_cache_checkpoint_protection::_count:
                break;
        }
    }
    return out;
}

bool server_cache_checkpoint_bounded_replay(
        const common_prompt_checkpoint & recovery,
        const common_prompt_checkpoint & later,
        uint64_t max_replay_tokens) noexcept {
    return recovery.computation_frontier.valid() &&
           later.computation_frontier.valid() &&
           later.n_tokens >= recovery.n_tokens &&
           uint64_t(later.n_tokens - recovery.n_tokens) <= max_replay_tokens &&
           recovery.computation_frontier.sequence_epoch ==
               later.computation_frontier.sequence_epoch &&
           recovery.computation_frontier.execution_identity ==
               later.computation_frontier.execution_identity &&
           recovery.computation_frontier.adapter_config_identity ==
               later.computation_frontier.adapter_config_identity &&
           recovery.computation_frontier.media_content_identity ==
               later.computation_frontier.media_content_identity &&
           recovery.checkpoint_epoch == later.checkpoint_epoch &&
           recovery.checkpoint_epoch_swa == later.checkpoint_epoch_swa;
}

size_t server_cache_checkpoint_rebase_preserved_suffix(
        std::list<common_prompt_checkpoint> & checkpoints,
        const llama_memory_vbr_state_data & before,
        const llama_memory_vbr_state_data & after,
        llama_pos suffix_begin) noexcept {
    if (suffix_begin < 0) {
        return 0;
    }
    size_t rebased = 0;
    for (auto & checkpoint : checkpoints) {
        if (checkpoint.pos_max >= 0 && checkpoint.pos_max < suffix_begin &&
            common_prompt_checkpoint_lineage_matches(checkpoint, before)) {
            checkpoint.checkpoint_epoch     = after.checkpoint_epoch;
            checkpoint.checkpoint_epoch_swa = after.checkpoint_epoch_swa;
            rebased++;
        }
    }
    return rebased;
}

server_cache_checkpoint_floor_plan server_cache_plan_checkpoint_capacity_floor(
        const std::vector<server_cache_checkpoint_floor_input> & candidates) noexcept {
    server_cache_checkpoint_floor_plan out;
    uint32_t heuristic = UINT32_MAX;
    try {
        for (const auto & candidate : candidates) {
            if (candidate.recovery_pinned ||
                candidate.protection ==
                    server_cache_checkpoint_protection::mandatory_anchor ||
                candidate.protection ==
                    server_cache_checkpoint_protection::hard_lease) {
                if (candidate.protection ==
                        server_cache_checkpoint_protection::hard_lease) {
                    out.reason =
                        common_cache_plan_destruction_reason::hard_lease_blocked;
                }
                continue;
            }
            if (candidate.protection ==
                    server_cache_checkpoint_protection::seam_heuristic) {
                if (heuristic == UINT32_MAX) {
                    heuristic = candidate.ordinal;
                }
                continue;
            }
            out.selected = true;
            out.ordinal = candidate.ordinal;
            out.reason = common_cache_plan_destruction_reason::none;
            return out;
        }
        if (heuristic != UINT32_MAX) {
            out.selected = true;
            out.ordinal = heuristic;
            out.reason = common_cache_plan_destruction_reason::none;
        }
    } catch (...) {
        out = {};
        out.reason = common_cache_plan_destruction_reason::internal_fault;
    }
    return out;
}

bool server_cache_authority::sample_budget(
        llama_cache_budget_config & config,
        uint64_t pending_host_bytes) noexcept {
    try {
        config = {};
        config.devices = budget_devices;
        for (auto & input : config.devices) {
            size_t free = 0;
            size_t total = 0;
            auto * device = reinterpret_cast<ggml_backend_dev_t>(
                const_cast<void *>(input.backend_device));
            ggml_backend_dev_memory(device, &free, &total);
            input.physical_free  = uint64_t(free);
            input.physical_total = uint64_t(total);
            input.phys_state =
                total > 0 && free <= total
                    ? llama_cache_budget_capacity_state::known
                    : llama_cache_budget_capacity_state::unavailable;
        }

        config.host.pinned_cap = 0;
        config.host.pinned_state =
            llama_cache_budget_capacity_state::known;
        config.host.total_state =
            llama_cache_budget_capacity_state::unbounded;
        config.global_cap_state =
            llama_cache_budget_capacity_state::unbounded;

        // PROPOSAL §9 requires pre-flip FIFO eviction/list order to remain legacy-identical.
        // Therefore the prompt cache's configured rotation limit is not an authority ceiling.
        // Price new host bytes against physical CPU-memory headroom instead. The detached entry
        // has already allocated pending_host_bytes, so add those bytes back before comparing the
        // canonical accounting `before` to the point-in-time headroom.
        ggml_backend_dev_t cpu =
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        size_t free = 0;
        size_t total = 0;
        if (!cpu) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        ggml_backend_dev_memory(cpu, &free, &total);
        if (total == 0 || free > total) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }

        // Obtain the canonical host-domain `before` from the same closed category tables used by
        // fits(), rather than growing a second cache-byte classifier in the server.
        llama_cache_acct_snapshot snapshot = ledger.snapshot();
        llama_cache_budget_config probe = config;
        probe.host.pageable_state =
            llama_cache_budget_capacity_state::unbounded;
        llama_cache_budget_coordinator coordinator;
        const uint64_t serial = snapshot.serial;
        if (!coordinator.reset(std::move(snapshot), probe)) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        llama_cache_budget_plan baseline;
        baseline.accounting_serial = serial;
        const llama_cache_budget_result current = coordinator.fits(baseline);
        const auto host_domain =
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host);
        const auto row = std::find_if(
            current.domains.begin(), current.domains.end(),
            [&](const llama_cache_budget_row & candidate) {
                return candidate.resource.domain == host_domain;
            });
        if (row == current.domains.end() ||
            row->before.state != llama_cache_acct_known::known) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }

        uint64_t available = 0;
        uint64_t cap = 0;
        if (!add_checked(uint64_t(free), pending_host_bytes, available) ||
            !add_checked(row->before.value, available, cap)) {
            config.host.pageable_state =
                llama_cache_budget_capacity_state::unavailable;
            return true;
        }
        config.host.pageable_cap = cap;
        config.host.pageable_state =
            llama_cache_budget_capacity_state::known;
        return true;
    } catch (...) {
        config = {};
        return false;
    }
}

bool server_cache_authority::project_release(
        const llama_cache_acct_release_set_preview & release,
        std::vector<common_cache_plan_yield_domain> & out) noexcept {
    out.clear();
    try {
        llama_cache_budget_config config;
        if (!sample_budget(config)) {
            return false;
        }
        auto snapshot = ledger.snapshot();
        if (snapshot.serial != release.accounting_serial) {
            return false;
        }
        llama_cache_budget_coordinator coordinator;
        if (!coordinator.reset(std::move(snapshot), config)) {
            return false;
        }
        llama_cache_budget_plan plan;
        if (!server_cache_yield_release_plan(
                release, release.accounting_serial, plan)) {
            return false;
        }
        const auto fit = coordinator.fits(plan);
        if (fit.state != llama_cache_budget_fit_state::fits ||
            fit.accounting_serial != release.accounting_serial) {
            return false;
        }
        out.reserve(fit.domains.size());
        for (const auto & row : fit.domains) {
            if (std::none_of(
                    release.rows.begin(), release.rows.end(),
                    [&](const auto & released) {
                        return released.domain == row.resource.domain;
                    })) {
                continue;
            }
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(row, lowered)) {
                return false;
            }
            out.push_back(lowered);
        }
        return !out.empty();
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_authority::observe_release_domains(
        const std::vector<common_cache_plan_yield_domain> & projected,
        std::vector<common_cache_plan_yield_domain> & out) noexcept {
    out.clear();
    if (projected.empty()) {
        return true;
    }
    try {
        llama_cache_budget_config config;
        if (!sample_budget(config)) {
            return false;
        }
        auto snapshot = ledger.snapshot();
        const uint64_t serial = snapshot.serial;
        llama_cache_budget_coordinator coordinator;
        if (!coordinator.reset(std::move(snapshot), config)) {
            return false;
        }
        llama_cache_budget_plan plan;
        plan.accounting_serial = serial;
        const auto fit = coordinator.fits(plan);
        if (fit.accounting_serial != serial ||
            fit.state == llama_cache_budget_fit_state::unavailable) {
            return false;
        }
        out.reserve(projected.size());
        for (const auto & expected : projected) {
            const auto row = std::find_if(
                fit.domains.begin(), fit.domains.end(),
                [&](const auto & current) {
                    return current.resource.kind ==
                               llama_cache_budget_resource_kind::
                                   accounting_domain &&
                           current.resource.domain == expected.domain;
                });
            if (row == fit.domains.end()) {
                out.clear();
                return false;
            }
            common_cache_plan_yield_domain lowered;
            if (!server_cache_yield_lower_domain(*row, lowered)) {
                out.clear();
                return false;
            }
            out.push_back(std::move(lowered));
        }
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

bool server_cache_authority::admit_host_entry(
        server_prompt_cache_state & entry) noexcept {
    if (!configured) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: substrate unavailable");
        return false;
    }

    std::array<server_prompt_cache_payload_leaf, 3> leaves;
    if (!server_prompt_cache::payload_leaves(entry, leaves) ||
        server_fault("acct_unavailable")) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: payload accounting unavailable");
        return false;
    }

    uint64_t pending_host_bytes = 0;
    for (const auto & leaf : leaves) {
        if (!add_checked(pending_host_bytes, leaf.bytes, pending_host_bytes)) {
            admission_refusals++;
            SRV_WRN("%s\n",
                    "CACHE_AUTHORITY host publish refused: payload total overflow");
            return false;
        }
    }

    // One physical-capacity sample per publish. Serial-conflict retries refresh the coherent
    // accounting snapshot inside the composer; they do not require re-reading physical capacity.
    llama_cache_budget_config config;
    if (!sample_budget(config, pending_host_bytes)) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: budget sample failed");
        return false;
    }

    std::array<llama_cache_acct_op_id, 3> committed = {};
    std::vector<llama_cache_transaction_leaf> transaction_leaves;
    try {
        transaction_leaves.reserve(leaves.size());
        for (size_t i = 0; i < leaves.size(); ++i) {
            llama_cache_transaction_leaf leaf;
            leaf.category = leaves[i].category;
            leaf.domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pageable_host);
            leaf.expected_logical = leaves[i].bytes;
            leaf.reserve_resident = leaves[i].bytes;
            leaf.stage_resident = leaves[i].bytes;
            leaf.committed_op = &committed[i];
            transaction_leaves.push_back(leaf);
        }
    } catch (...) {
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: transaction setup failed");
        return false;
    }

    llama_cache_transaction_fault fault;
    fault.fail_after_commit =
        server_fault("cache_lifecycle_after_commit");
    const auto transaction =
        llama_cache_execute_reservation_transaction(
            ledger, config, transaction_leaves, fault);
    admission_retries += transaction.serial_retries;
    admission_rollbacks += transaction.rolled_back;

    if (transaction.status !=
            llama_cache_transaction_status::committed) {
        admission_refusals++;
        const auto category =
            transaction.failed_leaf < leaves.size()
                ? leaves[transaction.failed_leaf].category
                : llama_cache_acct_category::container_overhead;
        switch (transaction.status) {
            case llama_cache_transaction_status::admission_refused:
                SRV_WRN(
                    "CACHE_AUTHORITY host publish refused: category=%u status=%s attempts=%u\n",
                    unsigned(category),
                    llama_cache_admission_status_name(
                        transaction.admission_status),
                    transaction.attempts);
                break;
            case llama_cache_transaction_status::stage_failed:
                SRV_WRN(
                    "CACHE_AUTHORITY host publish refused: category=%u status=stage_failed\n",
                    unsigned(category));
                break;
            case llama_cache_transaction_status::commit_failed:
                SRV_WRN(
                    "CACHE_AUTHORITY host publish refused: category=%u status=commit_failed\n",
                    unsigned(category));
                break;
            case llama_cache_transaction_status::post_commit_fault:
                SRV_WRN("%s\n",
                        "CACHE_AUTHORITY host publish refused: injected post-commit failure");
                break;
            case llama_cache_transaction_status::invalid_argument:
            case llama_cache_transaction_status::after_admit_failed:
            case llama_cache_transaction_status::internal_fault:
            case llama_cache_transaction_status::_count:
                SRV_WRN(
                    "CACHE_AUTHORITY host publish refused: status=%s\n",
                    llama_cache_transaction_status_name(
                        transaction.status));
                break;
            case llama_cache_transaction_status::committed:
                break;
        }
        return false;
    }

    for (size_t i = 0; i < leaves.size(); ++i) {
        *leaves[i].operation = committed[i];
    }
    admission_commits++;
    return true;
}

bool server_cache_authority::admit_live_checkpoints(
        std::vector<server_cache_live_checkpoint_admission> & batch) noexcept {
    const uint64_t refusal_count = std::max<size_t>(batch.size(), 1);
    const auto refuse = [&]() noexcept {
        for (auto & member : batch) {
            member.committed.clear();
        }
        admission_refusals += refusal_count;
        return false;
    };

    uint64_t pending = 0;
    if (!configured || batch.empty()) {
        return refuse();
    }
    try {
        for (auto & member : batch) {
            member.committed.clear();
            member.committed.reserve(member.accelerator_bytes > 0 ? 2 : 1);
            uint64_t member_bytes = 0;
            if (member.artifact.v == 0 || member.checkpoint_bytes == 0 ||
                !add_checked(member.checkpoint_bytes,
                             member.accelerator_bytes, member_bytes) ||
                !add_checked(pending, member_bytes, pending)) {
                return refuse();
            }
        }
    } catch (...) {
        return refuse();
    }

    llama_cache_budget_config config;
    if (!sample_budget(config, pending)) {
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY checkpoint ownership refused: budget sample unavailable");
        return refuse();
    }

    std::vector<std::array<llama_cache_acct_op_id, 2>> outputs;
    std::vector<llama_cache_transaction_leaf> leaves;
    try {
        outputs.resize(batch.size());
        leaves.reserve(batch.size() * 2);
        for (size_t i = 0; i < batch.size(); ++i) {
            const auto add_leaf = [&](llama_cache_acct_category category,
                                      uint64_t bytes,
                                      llama_cache_acct_op_id * output) {
                if (bytes == 0) {
                    return;
                }
                llama_cache_transaction_leaf leaf;
                leaf.category = category;
                leaf.domain = llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pageable_host);
                leaf.attribution = {
                    llama_cache_acct_attr_kind::artifact, -1,
                    batch[i].artifact,
                };
                leaf.expected_logical = bytes;
                leaf.reserve_resident = bytes;
                leaf.stage_resident = bytes;
                leaf.artifact = batch[i].artifact;
                leaf.committed_op = output;
                leaves.push_back(leaf);
            };
            add_leaf(
                llama_cache_acct_category::checkpoint_state_payload,
                batch[i].checkpoint_bytes, &outputs[i][0]);
            add_leaf(
                llama_cache_acct_category::typed_accelerator_payload,
                batch[i].accelerator_bytes, &outputs[i][1]);
        }
    } catch (...) {
        return refuse();
    }

    const auto transaction = llama_cache_execute_reservation_transaction(
        ledger, config, leaves);
    admission_retries += transaction.serial_retries;
    admission_rollbacks += transaction.rolled_back;
    if (transaction.status != llama_cache_transaction_status::committed) {
        SRV_WRN(
            "CACHE_AUTHORITY checkpoint ownership refused: status=%s admission=%s\n",
            llama_cache_transaction_status_name(transaction.status),
            llama_cache_admission_status_name(transaction.admission_status));
        return refuse();
    }
    try {
        for (size_t i = 0; i < batch.size(); ++i) {
            for (const auto op : outputs[i]) {
                if (op) {
                    batch[i].committed.push_back(op);
                }
            }
        }
    } catch (...) {
        for (const auto & member : outputs) {
            for (const auto op : member) {
                if (op) {
                    (void) ledger.release(op);
                }
            }
        }
        admission_rollbacks += leaves.size();
        return refuse();
    }
    // Preserve the established per-checkpoint admission counter semantics
    // even though the accounting terminal is now one batch transaction.
    admission_commits += batch.size();
    return true;
}

bool server_cache_authority::admit_live_checkpoint(
        llama_cache_acct_artifact_id artifact,
        uint64_t checkpoint_bytes,
        uint64_t accelerator_bytes,
        std::vector<llama_cache_acct_op_id> & committed) noexcept {
    committed.clear();
    try {
        std::vector<server_cache_live_checkpoint_admission> batch(1);
        batch[0].artifact = artifact;
        batch[0].checkpoint_bytes = checkpoint_bytes;
        batch[0].accelerator_bytes = accelerator_bytes;
        if (!admit_live_checkpoints(batch)) {
            return false;
        }
        committed = std::move(batch[0].committed);
        return !committed.empty();
    } catch (...) {
        admission_refusals++;
        return false;
    }
}

void server_cache_authority::observe_host_destruction(
        common_cache_plan_destruction_receipt receipt,
        bool observe_classification) noexcept {
    destruction_counters.observe(
        common_cache_plan_selection::none, receipt, observe_classification);
    destruction_counters.last_receipt = std::move(receipt);
    destruction_counters.has_receipt = true;
}
