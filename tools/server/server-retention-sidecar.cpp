#include "server-retention-sidecar.h"
#include "server-cache-lease.h"

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
        common_retention_artifact_record && record) noexcept {
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
        if (old != associations.end()) {
            retire_association(old);
        }

        catalog_entry entry;
        entry.record = std::move(record);
        entry.encoded_size = bytes;
        auto inserted = catalog.emplace(artifact.v, std::move(entry));
        if (!inserted.second ||
            !associations.emplace(key, artifact).second) {
            if (inserted.second) {
                catalog.erase(inserted.first);
            }
            mark_unavailable();
            return false;
        }

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
                associations.erase(key);
                catalog.erase(inserted.first);
                return false;
            }
        }
        if (!checked_add(bytes_live, bytes)) {
            bytes_live = std::numeric_limits<uint64_t>::max();
            mark_unavailable();
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
        bool coverage_valid) noexcept {
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
    if (!install(key, std::move(record))) {
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
        record.kind = destination.kind;
        if (!allocator.issue(record.stamp.pool, record.stamp)) {
            retire(destination);
            mark_unavailable();
            return false;
        }
        if (!install(destination, std::move(record))) {
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
    associations.erase(it);
    if (leases) {
        leases->artifact_retired(artifact);
    }
    const auto entry = catalog.find(artifact.v);
    if (entry == catalog.end()) {
        mark_unavailable();
        return;
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
        out.avail = server_retention_candidate_availability::available;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
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
