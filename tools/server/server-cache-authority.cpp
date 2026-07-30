#include "server-cache-authority.h"

#include "server-common.h"
#include "server-task.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <limits>
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
