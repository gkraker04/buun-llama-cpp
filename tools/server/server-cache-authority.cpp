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
    static constexpr uint32_t MAX_ATTEMPTS = 3;

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

    std::array<llama_cache_admission_result, 3> admissions;
    for (size_t i = 0; i < leaves.size(); ++i) {
        llama_cache_admission_status last =
            llama_cache_admission_status::serial_conflict;
        uint32_t attempts = 0;
        for (; attempts < MAX_ATTEMPTS; ++attempts) {
            llama_cache_authority_request request;
            request.category          = leaves[i].category;
            request.domain            =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pageable_host);
            request.expected_logical  = leaves[i].bytes;
            request.expected_resident = leaves[i].bytes;
            admissions[i] = llama_cache_admit_reservation(
                ledger, config, request);
            last = admissions[i].status;
            if (last != llama_cache_admission_status::serial_conflict) {
                attempts++;
                break;
            }
            admission_retries++;
        }
        if (last != llama_cache_admission_status::admitted) {
            admission_refusals++;
            SRV_WRN(
                "CACHE_AUTHORITY host publish refused: category=%u status=%s attempts=%u\n",
                unsigned(leaves[i].category),
                llama_cache_admission_status_name(last),
                attempts);
            return false;
        }

        const auto alloc = ledger.new_alloc();
        if (!alloc ||
            !ledger.stage(
                admissions[i].claim.op(), alloc, leaves[i].bytes)) {
            admission_refusals++;
            SRV_WRN(
                "CACHE_AUTHORITY host publish refused: category=%u status=stage_failed\n",
                unsigned(leaves[i].category));
            return false;
        }
    }

    std::array<llama_cache_acct_op_id, 3> committed = {};
    for (size_t i = 0; i < leaves.size(); ++i) {
        if (!admissions[i].claim.commit(leaves[i].bytes, committed[i])) {
            for (size_t j = 0; j < i; ++j) {
                if (committed[j] && ledger.release(committed[j])) {
                    admission_rollbacks++;
                }
            }
            admission_refusals++;
            SRV_WRN(
                "CACHE_AUTHORITY host publish refused: category=%u status=commit_failed\n",
                unsigned(leaves[i].category));
            return false;
        }
    }

    if (server_fault("cache_lifecycle_after_commit")) {
        for (const auto op : committed) {
            if (ledger.release(op)) {
                admission_rollbacks++;
            }
        }
        admission_refusals++;
        SRV_WRN("%s\n",
                "CACHE_AUTHORITY host publish refused: injected post-commit failure");
        return false;
    }

    for (size_t i = 0; i < leaves.size(); ++i) {
        *leaves[i].operation = committed[i];
    }
    admission_commits++;
    return true;
}
