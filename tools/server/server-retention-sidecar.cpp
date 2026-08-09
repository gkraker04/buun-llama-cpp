#include "server-retention-sidecar.h"
#include "server-cache-lease.h"
#include "server-cache-destruction-quote.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace {
constexpr size_t MAX_CATALOG_ARTIFACTS = 8192;

bool checked_add(uint64_t & dst, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - dst) {
        return false;
    }
    dst += value;
    return true;
}
}

void server_cache_acct_mark_shadow_unavailable(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer) noexcept {
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated }) {
        ledger.mark_unavailable(category, domain, measure);
    }
    ledger.mark_producer_unavailable(domain, producer);
}

llama_cache_acct_op_id server_cache_acct_charge_shadow(
        llama_cache_acct_ledger & ledger,
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        llama_cache_acct_producer producer,
        const llama_cache_acct_attribution & attribution,
        uint64_t logical_bytes,
        uint64_t resident_bytes) noexcept {
    const auto op = ledger.reserve(
        category, domain, attribution, logical_bytes, resident_bytes);
    const auto artifact =
        attribution.kind == llama_cache_acct_attr_kind::artifact
            ? attribution.artifact
            : llama_cache_acct_artifact_id{};
    if (!op ||
        !ledger.stage(op, ledger.new_alloc(), resident_bytes, artifact) ||
        !ledger.commit(op, logical_bytes)) {
        if (op) {
            (void) ledger.abort(op);
        }
        server_cache_acct_mark_shadow_unavailable(
            ledger, category, domain, producer);
        return {};
    }
    return op;
}

size_t server_retention_instance_key_hash::operator()(
        const server_retention_instance_key & key) const noexcept {
    size_t result = std::hash<uintptr_t>{}(key.instance);
    result ^= std::hash<int32_t>{}(key.owner_slot) +
        0x9e3779b9 + (result << 6) + (result >> 2);
    result ^= std::hash<uint8_t>{}(uint8_t(key.kind)) +
        0x9e3779b9 + (result << 6) + (result >> 2);
    return result;
}

server_retention_sidecar_store::~server_retention_sidecar_store() {
    while (!associations.empty()) {
        retire_association(associations.begin());
    }
}

void server_retention_sidecar_store::configure(
        llama_cache_acct_ledger * ledger_in,
        const llama_cache_acct_resource_domain & domain_in,
        server_cache_lease_table * leases_in) noexcept {
    ledger = ledger_in;
    domain = domain_in;
    leases = leases_in;
}

llama_cache_acct_artifact_id
server_retention_sidecar_store::qualified_artifact_id(
        common_retention_pool pool, uint64_t stable_id) noexcept {
    if (stable_id == 0 ||
        stable_id > COMMON_RETENTION_MAX_POOL_COUNTER ||
        pool >= common_retention_pool::_count) {
        return {};
    }
    return { (stable_id << 1) | uint64_t(pool) };
}

void server_retention_sidecar_store::mark_unavailable() noexcept {
    n_unavailable++;
    if (!ledger) {
        return;
    }
    server_cache_acct_mark_shadow_unavailable(
        *ledger,
        llama_cache_acct_category::artifact_descriptor_metadata,
        domain,
        llama_cache_acct_producer::retention_sidecar);
}

bool server_retention_sidecar_store::install(
        const server_retention_instance_key & key,
        common_retention_artifact_record && record,
        const server_cache_lease_identity * checkpoint_identity,
        const server_cache_lease_frontier * replacement_frontier) noexcept {
    try {
        if (catalog.size() >= MAX_CATALOG_ARTIFACTS) {
            mark_unavailable();
            return false;
        }
        uint64_t bytes = 0;
        if (!common_retention_sidecar_artifact_encoded_size(record, bytes)) {
            mark_unavailable();
            return false;
        }

        const auto artifact = qualified_artifact_id(
            record.stamp.pool, record.stamp.stable_id);
        if (artifact.v == 0 || catalog.find(artifact.v) != catalog.end()) {
            mark_unavailable();
            return false;
        }
        auto old = associations.find(key);
        const auto old_artifact = old == associations.end()
            ? llama_cache_acct_artifact_id{} : old->second;

        catalog_entry entry;
        entry.record = std::move(record);
        if (checkpoint_identity && checkpoint_identity->valid()) {
            entry.checkpoint_identity = *checkpoint_identity;
            entry.checkpoint_identity_known = true;
        }
        entry.encoded_size = bytes;
        auto inserted = catalog.emplace(artifact.v, std::move(entry));
        if (!inserted.second) {
            mark_unavailable();
            return false;
        }
        inserted.first->second.owner = this;
        inserted.first->second.artifact = artifact;

        auto & installed = inserted.first->second;
        if (ledger) {
            const llama_cache_acct_attribution attribution {
                llama_cache_acct_attr_kind::artifact, -1, artifact,
            };
            installed.accounting_op = server_cache_acct_charge_shadow(
                *ledger,
                llama_cache_acct_category::artifact_descriptor_metadata,
                domain,
                llama_cache_acct_producer::retention_sidecar,
                attribution,
                bytes,
                bytes);
            if (!installed.accounting_op) {
                catalog.erase(inserted.first);
                return false;
            }
        }
        if (!checked_add(bytes_live, bytes)) {
            bytes_live = std::numeric_limits<uint64_t>::max();
            mark_unavailable();
        }

        // Publish all new backing/accounting state before changing the one
        // association. A same-key append replacement can then migrate leases
        // without exposing a missing-artifact interval on the scheduler
        // thread. New-key insertion is the only throwing step after charge;
        // roll its complete catalog entry back on failure.
        if (old != associations.end()) {
            old->second = artifact;
        } else {
            try {
                if (!associations.emplace(key, artifact).second) {
                    retire_catalog_entry(inserted.first);
                    mark_unavailable();
                    return false;
                }
            } catch (...) {
                retire_catalog_entry(inserted.first);
                throw;
            }
        }

        if (old_artifact.v != 0) {
            bool continued = false;
            if (leases && checkpoint_identity && replacement_frontier) {
                continued = leases->artifact_replaced(
                    { old_artifact, key.kind, key.owner_slot },
                    { artifact, key.kind, key.owner_slot },
                    *checkpoint_identity, *replacement_frontier);
            }
            if (leases && !continued) {
                leases->artifact_retired(old_artifact);
            }
            const auto old_entry = catalog.find(old_artifact.v);
            if (old_entry == catalog.end()) {
                mark_unavailable();
            } else if (old_entry->second.recovery_pins != 0) {
                old_entry->second.retire_pending = true;
            } else {
                retire_catalog_entry(old_entry);
            }
        }
        n_publish_ok++;
        return true;
    } catch (...) {
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::publish(
        const server_retention_instance_key & key,
        common_retention_pool pool,
        const common_chat_msg_spans & spans,
        bool source_known,
        uint64_t turn_token_count,
        uint64_t coverage_tokens,
        bool coverage_valid,
        const server_cache_lease_identity * checkpoint_identity,
        const server_cache_lease_frontier * replacement_frontier) noexcept {
    common_retention_artifact_record record;
    record.kind = key.kind;
    if (!allocator.issue(pool, record.stamp)) {
        retire(key);
        mark_unavailable();
        return false;
    }
    record.stamp.coverage_tokens = coverage_tokens;
    if (!common_retention_build_turn_table(
            spans, source_known, turn_token_count, record.turns)) {
        record.turns = {};
    }
    if (!coverage_valid ||
        !common_retention_score(record.turns, coverage_tokens, record.stamp)) {
        record.stamp.state = common_retention_score_state::unavailable;
        record.stamp.mandatory_anchor = false;
        record.stamp.mapped_turn_ordinal = 0;
        record.stamp.anchor_rank = 0;
    }
    if (!install(
            key, std::move(record), checkpoint_identity,
            replacement_frontier)) {
        retire(key);
        return false;
    }
    return true;
}

bool server_retention_sidecar_store::clone(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    try {
        const auto assoc = associations.find(source);
        if (assoc == associations.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto item = catalog.find(assoc->second.v);
        if (item == catalog.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        auto record = item->second.record;
        server_cache_lease_identity checkpoint_identity;
        const server_cache_lease_identity * checkpoint_identity_ptr = nullptr;
        if (item->second.checkpoint_identity_known) {
            checkpoint_identity = item->second.checkpoint_identity;
            checkpoint_identity_ptr = &checkpoint_identity;
        }
        record.kind = destination.kind;
        if (!allocator.issue(record.stamp.pool, record.stamp)) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        if (!install(
                destination, std::move(record), checkpoint_identity_ptr,
                nullptr)) {
            retire(destination);
            return false;
        }
        return true;
    } catch (...) {
        retire(destination);
        mark_unavailable();
        return false;
    }
}

bool server_retention_sidecar_store::rebind(
        const server_retention_instance_key & source,
        const server_retention_instance_key & destination) noexcept {
    try {
        const auto src = associations.find(source);
        if (src == associations.end()) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        const auto item = catalog.find(src->second.v);
        if (item == catalog.end() ||
            item->second.record.kind != destination.kind) {
            retire(destination);
            retire(source);
            mark_unavailable();
            return false;
        }
        const auto artifact = src->second;
        auto old = associations.find(destination);
        if (old != associations.end() && old != src) {
            retire_association(old);
        }
        if (!associations.emplace(destination, artifact).second) {
            retire(source);
            mark_unavailable();
            return false;
        }
        associations.erase(source);
        return true;
    } catch (...) {
        retire(source);
        mark_unavailable();
        return false;
    }
}

void server_retention_sidecar_store::retire_association(
        association_map::iterator it) noexcept {
    const auto artifact = it->second;
    const auto entry = catalog.find(artifact.v);
    if (entry == catalog.end()) {
        associations.erase(it);
        if (leases) {
            leases->artifact_retired(artifact);
        }
        mark_unavailable();
        return;
    }
    // The policy inventory excludes pinned entries. If latent catalog drift
    // reaches this legacy terminal anyway, fail soft: surface unavailable,
    // detach the stale association, and defer catalog/accounting retirement
    // until the final pin closes. The pin callback owns that terminal.
    if (entry->second.recovery_pins != 0) {
        entry->second.retire_pending = true;
        associations.erase(it);
        if (leases) {
            leases->artifact_retired(artifact);
        }
        mark_unavailable();
        return;
    }
    associations.erase(it);
    if (leases) {
        leases->artifact_retired(artifact);
    }
    retire_catalog_entry(entry);
}

void server_retention_sidecar_store::retire_catalog_entry(
        catalog_map::iterator entry) noexcept {
    if (ledger && !entry->second.release_ops.empty()) {
        auto release = llama_cache_prepare_release_set(
            *ledger, entry->second.release_ops,
            ledger->snapshot().serial);
        if (!release.ready() || release.commit() !=
                llama_cache_conditional_release_status::released) {
            // A failed conditional commit never releases a member. The
            // legacy drop has already won, so discharge each still-live op
            // and mark the catalog unavailable rather than aborting.
            mark_unavailable();
            for (const auto op : entry->second.release_ops) {
                (void) ledger->release(op);
            }
        }
    }
    if (entry->second.accounting_op && ledger) {
        if (!ledger->release(entry->second.accounting_op)) {
            mark_unavailable();
        }
    }
    const uint64_t bytes = entry->second.encoded_size;
    bytes_live = bytes <= bytes_live ? bytes_live - bytes : 0;
    catalog.erase(entry);
}

bool server_retention_sidecar_store::recovery_pinned(
        const server_retention_instance_key & key) const noexcept {
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    return item != catalog.end() && item->second.recovery_pins != 0;
}

void server_retention_sidecar_store::retire(
        const server_retention_instance_key & key) noexcept {
    const auto it = associations.find(key);
    if (it != associations.end()) {
        retire_association(it);
    }
}

void server_retention_sidecar_store::retire_slot(int32_t owner_slot) noexcept {
    for (auto it = associations.begin(); it != associations.end();) {
        if (it->first.owner_slot == owner_slot) {
            auto victim = it++;
            retire_association(victim);
        } else {
            ++it;
        }
    }
}

llama_cache_acct_artifact_id server_retention_sidecar_store::artifact_id(
        const server_retention_instance_key & key) const noexcept {
    const auto it = associations.find(key);
    return it == associations.end() ? llama_cache_acct_artifact_id{} : it->second;
}

bool server_retention_sidecar_store::candidate_for_instance(
        const server_retention_instance_key & key,
        server_retention_candidate & out) const noexcept {
    out = {};
    try {
        const auto association = associations.find(key);
        if (association == associations.end()) {
            return false;
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end()) {
            return false;
        }
        out.artifact_id = association->second;
        out.instance_key = key;
        out.record = item->second.record;
        out.provenance_op = item->second.accounting_op;
        out.release_ops = item->second.release_ops;
        out.avail = server_retention_candidate_availability::available;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool server_retention_sidecar_store::checkpoint_admission_artifact(
        const server_retention_instance_key & key,
        llama_cache_acct_artifact_id & artifact) const noexcept {
    artifact = {};
    if (key.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() || !item->second.release_ops.empty()) {
        return false;
    }
    artifact = association->second;
    return artifact.v != 0;
}

bool server_retention_sidecar_store::checkpoint_inventory(
        const server_retention_instance_key & key,
        server_retention_checkpoint_inventory & out) const noexcept {
    out = {};
    if (key.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    const auto association = associations.find(key);
    if (association == associations.end()) {
        return false;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() ||
        item->second.record.kind != common_retention_artifact_kind::checkpoint) {
        return false;
    }
    out.artifact_id = association->second;
    out.pool = item->second.record.stamp.pool;
    out.stable_id = item->second.record.stamp.stable_id;
    out.mandatory_anchor = item->second.record.stamp.mandatory_anchor;
    out.release_owned = !item->second.release_ops.empty();
    out.recovery_pinned = item->second.recovery_pins != 0;
    out.identity_known = leases &&
        item->second.checkpoint_identity_known &&
        item->second.checkpoint_identity.valid();
    if (out.identity_known) {
        out.lease = leases->inspect(
            out.artifact_id, item->second.checkpoint_identity);
    }
    return out.artifact_id.v != 0;
}

bool server_retention_sidecar_store::attach_release_ops(
        const server_retention_instance_key & key,
        std::vector<llama_cache_acct_op_id> ops) noexcept {
    const auto release_supplied = [&]() noexcept {
        if (!ledger) {
            return;
        }
        for (const auto op : ops) {
            if (op && !ledger->release(op)) {
                mark_unavailable();
            }
        }
    };
    try {
        if (ops.empty() || std::any_of(ops.begin(), ops.end(),
                [](const auto op) { return !op; })) {
            release_supplied();
            return false;
        }
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        const auto association = associations.find(key);
        if (association == associations.end()) {
            release_supplied();
            return false;
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end() || !item->second.release_ops.empty()) {
            release_supplied();
            return false;
        }
        item->second.release_ops = std::move(ops);
        return true;
    } catch (...) {
        release_supplied();
        mark_unavailable();
        return false;
    }
}

void server_retention_sidecar_store::release_recovery_pin(
        void * context) noexcept {
    auto * entry = static_cast<catalog_entry *>(context);
    if (!entry || entry->recovery_pins == 0) {
        return;
    }
    entry->recovery_pins--;
    if (entry->recovery_pins == 0 && entry->retire_pending &&
        entry->owner && entry->artifact.v != 0) {
        auto * owner = entry->owner;
        const auto found = owner->catalog.find(entry->artifact.v);
        if (found != owner->catalog.end() && &found->second == entry) {
            owner->retire_catalog_entry(found);
        } else {
            owner->mark_unavailable();
        }
    }
}

server_cache_recovery_pin
server_retention_sidecar_store::acquire_recovery_pin(
        const server_retention_instance_key & key) noexcept {
    try {
        const auto association = associations.find(key);
        if (association == associations.end()) {
            return {};
        }
        const auto item = catalog.find(association->second.v);
        if (item == catalog.end() || item->second.release_ops.empty() ||
            item->second.recovery_pins == UINT32_MAX) {
            return {};
        }
        item->second.recovery_pins++;
        auto pin = server_cache_recovery_pin::acquire(
            &item->second,
            release_recovery_pin,
            { association->second },
            item->second.release_ops);
        if (!pin.valid()) {
            item->second.recovery_pins--;
        }
        return pin;
    } catch (...) {
        mark_unavailable();
        return {};
    }
}

void server_retention_sidecar_store::retire_after_committed_release(
        const server_retention_instance_key & key) noexcept {
    const auto association = associations.find(key);
    if (association == associations.end()) {
        mark_unavailable();
        return;
    }
    const auto item = catalog.find(association->second.v);
    if (item == catalog.end() || item->second.recovery_pins != 0 ||
        item->second.release_ops.empty()) {
        mark_unavailable();
        return;
    }
    item->second.release_ops.clear();
    retire_association(association);
}

bool server_retention_sidecar_store::retire_slot_after_committed_release(
        int32_t owner_slot,
        const std::vector<llama_cache_acct_artifact_id> & selected_attention,
        const std::vector<llama_cache_acct_artifact_id> & selected_recurrent) noexcept {
    const auto selected = [&](llama_cache_acct_artifact_id artifact) {
        return std::find(selected_attention.begin(), selected_attention.end(), artifact) !=
                   selected_attention.end() ||
               std::find(selected_recurrent.begin(), selected_recurrent.end(), artifact) !=
                   selected_recurrent.end();
    };
    bool found = false;
    for (const auto & association : associations) {
        if (association.first.owner_slot != owner_slot) {
            continue;
        }
        found = true;
        const auto item = catalog.find(association.second.v);
        if (item == catalog.end() || item->second.recovery_pins != 0 ||
            !selected(association.second)) {
            mark_unavailable();
            return false;
        }
    }
    if (!found) {
        mark_unavailable();
        return false;
    }
    for (auto it = associations.begin(); it != associations.end();) {
        if (it->first.owner_slot != owner_slot) {
            ++it;
            continue;
        }
        auto victim = it++;
        const auto item = catalog.find(victim->second.v);
        GGML_ASSERT(item != catalog.end());
        GGML_ASSERT(item->second.recovery_pins == 0);
        item->second.release_ops.clear();
        item->second.accounting_op = {};
        retire_association(victim);
    }
    return true;
}

std::vector<server_retention_candidate>
server_retention_sidecar_store::candidate_snapshot() const noexcept {
    try {
        std::vector<server_retention_candidate> out;
        out.reserve(associations.size());
        for (const auto & [key, artifact] : associations) {
            server_retention_candidate candidate;
            candidate.artifact_id = artifact;
            candidate.instance_key = key;
            const auto item = catalog.find(artifact.v);
            if (item != catalog.end()) {
                candidate.record = item->second.record;
                candidate.provenance_op = item->second.accounting_op;
                candidate.release_ops = item->second.release_ops;
                candidate.avail =
                    server_retention_candidate_availability::available;
            }
            out.push_back(std::move(candidate));
        }
        std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) {
            if (a.record.stamp.pool != b.record.stamp.pool) {
                return a.record.stamp.pool < b.record.stamp.pool;
            }
            if (a.record.stamp.stable_id != b.record.stamp.stable_id) {
                return a.record.stamp.stable_id < b.record.stamp.stable_id;
            }
            return a.artifact_id.v < b.artifact_id.v;
        });
        return out;
    } catch (...) {
        server_retention_candidate failed;
        failed.avail =
            server_retention_candidate_availability::backing_missing_or_stale;
        try {
            return { std::move(failed) };
        } catch (...) {
            return {};
        }
    }
}

common_retention_sidecar_snapshot
server_retention_sidecar_store::snapshot() const noexcept {
    try {
        common_retention_sidecar_snapshot out;
        for (const auto pool : {
                common_retention_pool::attention,
                common_retention_pool::recurrent }) {
            const size_t i = size_t(pool);
            out.recency_high_water[i] = allocator.recency_high_water(pool);
            out.stable_high_water[i] = allocator.stable_high_water(pool);
        }
        out.artifacts.reserve(catalog.size());
        for (const auto & item : catalog) {
            out.artifacts.push_back(item.second.record);
        }
        std::sort(out.artifacts.begin(), out.artifacts.end(),
            [](const auto & a, const auto & b) {
                if (a.stamp.pool != b.stamp.pool) {
                    return a.stamp.pool < b.stamp.pool;
                }
                return a.stamp.stable_id < b.stamp.stable_id;
            });
        return out;
    } catch (...) {
        common_retention_sidecar_snapshot invalid;
        invalid.version = 0;
        return invalid;
    }
}

bool server_retention_sidecar_store::import_snapshot(
        const common_retention_sidecar_snapshot & imported) noexcept {
    if (!allocator.import_snapshot(imported)) {
        mark_unavailable();
        return false;
    }
    return true;
}

bool server_retention_sidecar_store::export_bytes(
        std::vector<uint8_t> & out) const noexcept {
    return common_retention_sidecar_encode(snapshot(), out);
}
