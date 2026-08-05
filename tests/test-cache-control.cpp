#include "server-cache-control.h"
#include "server-cache-destruction-quote.h"
#include "server-vbr-artifact-store.h"
#include "llama-cache-authority.h"

#include <cstdio>
#include <memory>
#include <unordered_map>

static int failures = 0;
#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #value); \
    ++failures; } } while (0)

class control_test_clock final : public server_cache_lease_clock {
public:
    uint64_t now_ns() noexcept override { return now; }
    uint64_t now = 1;
};

class control_test_tokens final : public server_cache_control_token_source {
public:
    bool next(server_cache_control_token & out) noexcept override {
        out = { 0x1000000000000000ULL + next_value,
                0x2000000000000000ULL + next_value };
        ++next_value;
        return true;
    }
    uint64_t next_value = 1;
};

struct frontier_source {
    std::unordered_map<uintptr_t, server_cache_lease_frontier> values;
};

struct vbr_test_resolver {
    std::shared_ptr<int> owner = std::make_shared<int>(1);
};

struct host_test_resolver {
    std::shared_ptr<int> owner = std::make_shared<int>(1);
};

static server_cache_durable_fallback_proof acquire_host_proof(
        void * context,
        const server_cache_control_selector &) noexcept {
    auto * resolver = static_cast<host_test_resolver *>(context);
    return server_cache_durable_fallback_proof_for_test(
        server_cache_lease_fallback_state::available, resolver->owner);
}

static server_cache_control_status resolve_vbr(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_subject & subject,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier,
        server_cache_durable_fallback_proof & pin) noexcept {
    auto * resolver = static_cast<vbr_test_resolver *>(context);
    if (selector.tenant_key != "tenant" ||
        (selector.reference != "subject" &&
         selector.reference != "fallback")) {
        return server_cache_control_status::not_found;
    }
    subject = {
        { selector.reference == "subject" ? 900u : 901u },
        common_retention_artifact_kind::host_entry,
        -1,
    };
    identity = { "e", "a", "m" };
    frontier = { 1, 8, 8 };
    pin = server_cache_durable_fallback_proof_for_test(
        server_cache_lease_fallback_state::available, resolver->owner);
    return server_cache_control_status::ok;
}

static bool refresh_frontier(
        void * context,
        const server_cache_control_selector & selector,
        server_cache_lease_identity & identity,
        server_cache_lease_frontier & frontier) noexcept {
    auto * source = static_cast<frontier_source *>(context);
    const auto found = source->values.find(selector.retention_key.instance);
    if (found == source->values.end()) {
        return false;
    }
    identity = selector.identity;
    frontier = found->second;
    return true;
}

static bool sample_control_store_budget(
        void *, llama_cache_budget_config & output) noexcept {
    output = {};
    output.host.pageable_state =
        llama_cache_budget_capacity_state::unbounded;
    output.host.pinned_cap = 1024;
    output.host.pinned_state =
        llama_cache_budget_capacity_state::known;
    output.host.total_state =
        llama_cache_budget_capacity_state::unbounded;
    output.global_cap_state =
        llama_cache_budget_capacity_state::unbounded;
    return true;
}

static vbr_artifact_portable_topology control_store_topology() {
    llama_cache_acct_shard_topology topology;
    CHECK(llama_cache_acct_build_shard_topology(
        { "e1-control-store-device" },
        LLAMA_SPLIT_MODE_NONE, 0, nullptr, topology));
    return topology;
}

static common_chat_msg_spans spans() {
    common_chat_msg_spans value;
    value.add(COMMON_CHAT_ROLE_USER, 0, 4);
    value.add(COMMON_CHAT_ROLE_ASSISTANT, 4, 4);
    return value;
}

static void configure_retention(
        llama_cache_acct_ledger & ledger,
        server_cache_lease_table & leases,
        server_retention_sidecar_store & retention,
        const llama_cache_acct_resource_domain & domain) {
    const llama_cache_acct_completeness_requirement requirement = {
        domain, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::artifact_descriptor_metadata,
            domain, measure, 0);
        ledger.gauge_set(
            llama_cache_acct_category::checkpoint_state_payload,
            domain, measure, 0);
    }
    CHECK(ledger.certify_complete(
        domain, llama_cache_acct_producer::retention_sidecar));
    retention.configure(&ledger, domain, &leases);
}

static std::array<llama_cache_acct_op_id, 3> attach_all(
        llama_cache_acct_ledger & ledger,
        server_retention_sidecar_store & retention,
        const llama_cache_acct_resource_domain & domain,
        const server_retention_instance_key & key,
        const server_cache_lease_identity & identity) {
    CHECK(retention.publish(
        key, common_retention_pool::attention, spans(), true,
        8, 8, true, &identity));
    const auto artifact = retention.artifact_id(key);
    std::array<llama_cache_acct_op_id, 3> ops;
    for (auto & op : ops) {
        op = server_cache_acct_charge_shadow(
            ledger, llama_cache_acct_category::checkpoint_state_payload,
            domain, llama_cache_acct_producer::retention_sidecar,
            { llama_cache_acct_attr_kind::artifact, -1, artifact }, 32, 32);
        CHECK(op);
    }
    CHECK(retention.attach_release_ops(
        key, std::vector<llama_cache_acct_op_id>(ops.begin(), ops.end())));
    return ops;
}

static llama_cache_acct_op_id attach(
        llama_cache_acct_ledger & ledger,
        server_retention_sidecar_store & retention,
        const llama_cache_acct_resource_domain & domain,
        const server_retention_instance_key & key,
        const server_cache_lease_identity & identity) {
    return attach_all(ledger, retention, domain, key, identity).front();
}

static server_cache_control_request create_holder(uint64_t ttl) {
    server_cache_control_request request;
    request.ttl_ns = ttl;
    return request;
}

static server_cache_control_result execute(
        server_cache_control_authority & authority,
        server_cache_control_operation operation,
        const server_cache_control_request & request) {
    return authority.execute(operation, request);
}

static server_cache_control_selector selector(
        server_cache_control_subject_kind kind,
        server_retention_instance_key key,
        const server_cache_lease_identity & identity,
        server_cache_lease_frontier frontier) {
    server_cache_control_selector value;
    value.kind = kind;
    value.retention_key = key;
    value.identity = identity;
    value.frontier = frontier;
    return value;
}

static void test_holder_hard_orphan_frontier_and_proof() {
    control_test_clock clock;
    control_test_tokens tokens;
    server_cache_lease_table leases(&clock);
    llama_cache_acct_ledger ledger;
    server_retention_sidecar_store retention;
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    configure_retention(ledger, leases, retention, domain);

    const server_cache_lease_identity identity = {
        "execution", "adapter", "media",
    };
    const auto live = server_retention_instance_key::for_slot(1);
    const auto fallback = server_retention_instance_key::for_host_entry(
        reinterpret_cast<const server_prompt_cache_state *>(uintptr_t(2)));
    (void) attach(ledger, retention, domain, live, identity);
    (void) attach(ledger, retention, domain, fallback, identity);

    frontier_source frontiers;
    frontiers.values[live.instance] = { 7, 8, 8 };
    frontiers.values[fallback.instance] = { 7, 8, 8 };
    server_cache_control_config config;
    config.leases = &leases;
    config.retention = &retention;
    config.clock = &clock;
    config.tokens = &tokens;
    config.refresh_context = &frontiers;
    config.refresh_subject = refresh_frontier;
    vbr_test_resolver vbr;
    config.resolve_vbr_context = &vbr;
    config.resolve_vbr = resolve_vbr;
    host_test_resolver host;
    config.host_proof_context = &host;
    config.acquire_host_proof = acquire_host_proof;
    server_cache_control_authority authority(config);
    CHECK(authority.available());

    const auto holder = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(100));
    CHECK(holder.status == server_cache_control_status::ok);
    server_cache_control_request acquire;
    acquire.holder = holder.holder;
    acquire.requested_class = server_cache_lease_class::hard;
    acquire.ttl_ns = 50;
    acquire.idempotency_key = 44;
    acquire.subject = selector(
        server_cache_control_subject_kind::live_prefix,
        live, identity, { 7, 8, 8 });
    acquire.fallback = selector(
        server_cache_control_subject_kind::host_snapshot,
        fallback, identity, { 7, 8, 8 });
    const auto granted = execute(
        authority, server_cache_control_operation::lease_acquire, acquire);
    CHECK(granted.status == server_cache_control_status::ok);
    const auto replayed = execute(
        authority, server_cache_control_operation::lease_acquire, acquire);
    CHECK(replayed.status == server_cache_control_status::ok);
    CHECK(replayed.lease == granted.lease);
    CHECK(granted.granted_class == server_cache_lease_class::hard);
    CHECK(host.owner.use_count() > 1);
    CHECK(server_cache_lease_is_hard(
        leases.inspect_range(retention.artifact_id(live), identity, 7, 0, 8)));
    CHECK(!server_cache_lease_is_hard(
        leases.inspect_range(retention.artifact_id(live), identity, 7, 8, 4)));

    server_cache_control_request inspect;
    inspect.holder = holder.holder;
    inspect.lease = granted.lease;
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::ok);
    frontiers.values[live.instance] = { 7, 12, 12 };
    const auto partial = execute(
        authority, server_cache_control_operation::lease_inspect, inspect);
    CHECK(partial.status == server_cache_control_status::partially_stale);
    CHECK(partial.proven_frontier.token_count == 8);
    CHECK(partial.lease_frontier.token_count == 12);

    clock.now = 55;
    authority.lifecycle_point();
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status ==
          server_cache_control_status::orphaned);
    server_cache_control_request renew = inspect;
    renew.ttl_ns = 60;
    frontiers.values[fallback.instance] = { 7, 12, 12 };
    CHECK(execute(authority, server_cache_control_operation::lease_renew,
                  renew).status == server_cache_control_status::fallback_invalid);
    renew.fallback = selector(
        server_cache_control_subject_kind::host_snapshot,
        fallback, identity, { 7, 12, 12 });
    CHECK(execute(authority, server_cache_control_operation::lease_renew,
                  renew).status == server_cache_control_status::ok);
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::ok);

    // The table clone shares the proof owner and explicit scope. Retiring the
    // original subject does not leak the cloned hard entry; owner release
    // below closes the whole scope and finally drops the recovery pin.
    const auto cloned_live = server_retention_instance_key::for_slot(3);
    CHECK(retention.clone(live, cloned_live));
    const auto cloned_artifact = retention.artifact_id(cloned_live);
    CHECK(leases.artifact_cloned(
        { retention.artifact_id(live),
          common_retention_artifact_kind::live_slot, 1 },
        { cloned_artifact, common_retention_artifact_kind::live_slot, 3 },
        identity));
    const auto clone_eval = leases.inspect(cloned_artifact, identity);
    CHECK(clone_eval.state == server_cache_lease_eval_state::known);
    CHECK(server_cache_lease_is_hard(clone_eval));
    retention.retire(live);

    clock.now = 200;
    authority.lifecycle_point();
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::not_found);
    CHECK(host.owner.use_count() == 1);
    CHECK(!server_cache_lease_is_hard(
        leases.inspect(cloned_artifact, identity)));
    server_cache_control_request reattach;
    reattach.recovery = holder.holder_recovery;
    reattach.ttl_ns = 100;
    auto wrong_reattach = reattach;
    wrong_reattach.recovery.low++;
    CHECK(execute(authority,
                  server_cache_control_operation::holder_reattach,
                  wrong_reattach).status == server_cache_control_status::not_found);
    const auto resumed = execute(
        authority, server_cache_control_operation::holder_reattach, reattach);
    CHECK(resumed.status == server_cache_control_status::ok);
    inspect.holder = resumed.holder;
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::subject_lost);

    server_cache_control_request release = inspect;
    CHECK(execute(authority, server_cache_control_operation::lease_release,
                  release).status == server_cache_control_status::ok);
    CHECK(host.owner.use_count() == 1);
    CHECK(!server_cache_lease_is_hard(
        leases.inspect(cloned_artifact, identity)));
}

static void test_refusals_soft_expiry_and_ownership() {
    control_test_clock clock;
    control_test_tokens tokens;
    server_cache_lease_table leases(&clock);
    llama_cache_acct_ledger ledger;
    server_retention_sidecar_store retention;
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    configure_retention(ledger, leases, retention, domain);
    const server_cache_lease_identity identity = { "e", "a", "m" };
    const auto live = server_retention_instance_key::for_slot(4);
    (void) attach(ledger, retention, domain, live, identity);
    frontier_source frontiers;
    frontiers.values[live.instance] = { 1, 8, 8 };
    server_cache_control_config config;
    config.leases = &leases;
    config.retention = &retention;
    config.clock = &clock;
    config.tokens = &tokens;
    config.refresh_context = &frontiers;
    config.refresh_subject = refresh_frontier;
    vbr_test_resolver vbr;
    config.resolve_vbr_context = &vbr;
    config.resolve_vbr = resolve_vbr;
    host_test_resolver host;
    config.host_proof_context = &host;
    config.acquire_host_proof = acquire_host_proof;
    server_cache_control_authority authority(config);
    const auto first = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(500));
    const auto second = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(50));
    server_cache_control_request own_events;
    own_events.holder = second.holder;
    const auto second_events = execute(
        authority, server_cache_control_operation::events, own_events);
    CHECK(second_events.status == server_cache_control_status::ok);
    CHECK(second_events.events.size() == 1);

    const auto fallback = server_retention_instance_key::for_host_entry(
        reinterpret_cast<const server_prompt_cache_state *>(uintptr_t(5)));
    (void) attach(ledger, retention, domain, fallback, identity);
    frontiers.values[fallback.instance] = { 1, 8, 8 };

    // All three v1 subject kinds traverse the same registry. Host and F are
    // immutable; F owns a package-equivalent pin for the lease lifetime.
    server_cache_control_request immutable;
    immutable.holder = first.holder;
    immutable.requested_class = server_cache_lease_class::soft;
    immutable.ttl_ns = 25;
    immutable.subject = selector(
        server_cache_control_subject_kind::host_snapshot,
        fallback, identity, { 1, 8, 8 });
    const auto host_lease = execute(
        authority, server_cache_control_operation::lease_acquire, immutable);
    CHECK(host_lease.status == server_cache_control_status::ok);
    immutable.subject = {};
    immutable.subject.kind = server_cache_control_subject_kind::vbr_reference;
    immutable.subject.reference = "subject";
    immutable.subject.tenant_key = "tenant";
    const auto f_lease = execute(
        authority, server_cache_control_operation::lease_acquire, immutable);
    CHECK(f_lease.status == server_cache_control_status::ok);
    CHECK(vbr.owner.use_count() == 1);

    server_cache_control_request acquire;
    acquire.holder = first.holder;
    acquire.requested_class = server_cache_lease_class::hard;
    acquire.ttl_ns = 20;
    acquire.subject = selector(
        server_cache_control_subject_kind::live_prefix,
        live, identity, { 1, 8, 8 });
    acquire.fallback = acquire.subject;
    CHECK(execute(authority, server_cache_control_operation::lease_acquire,
                  acquire).status ==
          server_cache_control_status::fallback_invalid);
    acquire.fallback = selector(
        server_cache_control_subject_kind::host_snapshot,
        fallback, { "wrong", "a", "m" }, { 1, 8, 8 });
    CHECK(execute(authority, server_cache_control_operation::lease_acquire,
                  acquire).status ==
          server_cache_control_status::fallback_invalid);
    acquire.fallback.retention_key =
        server_retention_instance_key::for_host_entry(
            reinterpret_cast<const server_prompt_cache_state *>(uintptr_t(99)));
    CHECK(execute(authority, server_cache_control_operation::lease_acquire,
                  acquire).status ==
          server_cache_control_status::fallback_unavailable);
    acquire.fallback = {};
    acquire.fallback.kind = server_cache_control_subject_kind::vbr_reference;
    acquire.fallback.reference = "fallback";
    acquire.fallback.tenant_key = "tenant";
    const auto hard_with_fallback = execute(
        authority, server_cache_control_operation::lease_acquire, acquire);
    CHECK(hard_with_fallback.status == server_cache_control_status::ok);
    CHECK(vbr.owner.use_count() > 1);
    acquire.subject.kind = server_cache_control_subject_kind::live_checkpoint;
    CHECK(execute(authority, server_cache_control_operation::lease_acquire,
                  acquire).status ==
          server_cache_control_status::not_supported);

    const auto checkpoint = server_retention_instance_key::for_checkpoint(
        4, reinterpret_cast<const common_prompt_checkpoint *>(uintptr_t(7)));
    (void) attach(ledger, retention, domain, checkpoint, identity);
    frontiers.values[checkpoint.instance] = { 1, 8, 8 };
    acquire.subject = selector(
        server_cache_control_subject_kind::live_prefix,
        live, identity, { 1, 8, 8 });
    acquire.fallback = selector(
        server_cache_control_subject_kind::live_checkpoint,
        checkpoint, identity, { 1, 8, 8 });
    const auto checkpoint_grant = execute(
        authority, server_cache_control_operation::lease_acquire, acquire);
    CHECK(checkpoint_grant.status == server_cache_control_status::not_supported);

    acquire.subject = selector(
        server_cache_control_subject_kind::live_prefix,
        live, identity, { 1, 8, 8 });
    acquire.requested_class = server_cache_lease_class::soft;
    const auto soft = execute(
        authority, server_cache_control_operation::lease_acquire, acquire);
    CHECK(soft.status == server_cache_control_status::ok);
    server_cache_control_request inspect;
    inspect.holder = second.holder;
    inspect.lease = soft.lease;
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::not_found);
    clock.now = 100;
    authority.lifecycle_point();
    inspect.holder = first.holder;
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::lease_expired);

    CHECK(leases.artifact_rebound(
        retention.artifact_id(live), { "rebound", "a", "m" }));
    CHECK(vbr.owner.use_count() == 1);
    inspect.lease = hard_with_fallback.lease;
    CHECK(execute(authority, server_cache_control_operation::lease_inspect,
                  inspect).status == server_cache_control_status::subject_lost);
    CHECK(execute(authority, server_cache_control_operation::lease_release,
                  inspect).status == server_cache_control_status::ok);
}

static void test_poisoned_evidence_still_releases() {
    control_test_clock clock;
    control_test_tokens tokens;
    server_cache_lease_table leases(&clock);
    llama_cache_acct_ledger ledger;
    server_retention_sidecar_store retention;
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    configure_retention(ledger, leases, retention, domain);
    vbr_test_resolver vbr;
    const auto run = [&](bool fail_note, bool fail_remember) {
        server_cache_control_config config;
        config.leases = &leases;
        config.retention = &retention;
        config.clock = &clock;
        config.tokens = &tokens;
        config.resolve_vbr_context = &vbr;
        config.resolve_vbr = resolve_vbr;
        config.test_fail_note_after = fail_note ? 1 :
            std::numeric_limits<size_t>::max();
        config.test_fail_remember = fail_remember;
        server_cache_control_authority authority(config);
        const auto holder = execute(
            authority, server_cache_control_operation::holder_create,
            create_holder(100));
        CHECK(holder.status == server_cache_control_status::ok);
        server_cache_control_request acquire;
        acquire.holder = holder.holder;
        acquire.idempotency_key = fail_remember ? 7 : 0;
        acquire.requested_class = server_cache_lease_class::hard;
        acquire.ttl_ns = 50;
        acquire.subject.kind = server_cache_control_subject_kind::vbr_reference;
        acquire.subject.reference = "subject";
        acquire.subject.tenant_key = "tenant";
        acquire.fallback.kind = server_cache_control_subject_kind::vbr_reference;
        acquire.fallback.reference = "fallback";
        acquire.fallback.tenant_key = "tenant";
        const auto granted = execute(
            authority, server_cache_control_operation::lease_acquire, acquire);
        CHECK(granted.status == server_cache_control_status::ok);
        CHECK(vbr.owner.use_count() > 1);
        if (fail_note) {
            server_cache_control_request release;
            release.holder = holder.holder;
            release.lease = granted.lease;
            CHECK(execute(authority,
                          server_cache_control_operation::lease_release,
                          release).status == server_cache_control_status::ok);
        } else {
            server_cache_control_request close;
            close.holder = holder.holder;
            CHECK(execute(authority,
                          server_cache_control_operation::holder_close,
                          close).status == server_cache_control_status::ok);
        }
        CHECK(vbr.owner.use_count() == 1);
    };
    run(true, false);
    run(false, true);
}

static void test_task_lifecycle_gate() {
    CHECK(server_cache_control_task_precheck(false, true, true) ==
          server_cache_control_status::invalid_request);
    CHECK(server_cache_control_task_precheck(true, false, true) ==
          server_cache_control_status::not_supported);
    CHECK(server_cache_control_task_precheck(true, true, false) ==
          server_cache_control_status::not_supported);
    CHECK(server_cache_control_task_precheck(true, true, true) ==
          server_cache_control_status::ok);
}

static void test_family_registry_and_binding() {
    control_test_clock clock;
    control_test_tokens tokens;
    server_cache_lease_table leases(&clock);
    server_retention_sidecar_store retention;
    server_cache_control_config config;
    config.leases = &leases;
    config.retention = &retention;
    config.clock = &clock;
    config.tokens = &tokens;
    server_cache_control_authority authority(config);
    CHECK(authority.available());

    const auto holder = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(100));
    CHECK(holder.status == server_cache_control_status::ok);

    server_cache_control_request register_family;
    register_family.holder = holder.holder;
    register_family.idempotency_key = 901;
    const auto family = execute(
        authority, server_cache_control_operation::family_register,
        register_family);
    CHECK(family.status == server_cache_control_status::ok);
    CHECK(bool(family.family));
    const auto family_replay = execute(
        authority, server_cache_control_operation::family_register,
        register_family);
    CHECK(family_replay.family == family.family);

    server_cache_control_request bind;
    bind.holder = holder.holder;
    bind.family = family.family;
    bind.family_role = common_cache_family_role::main;
    bind.idempotency_key = 902;
    const auto binding = execute(
        authority, server_cache_control_operation::family_bind, bind);
    CHECK(binding.status == server_cache_control_status::ok);
    CHECK(bool(binding.family_binding));
    CHECK(binding.cache_family.declared());
    CHECK(binding.cache_family.role == common_cache_family_role::main);
    const auto binding_replay = execute(
        authority, server_cache_control_operation::family_bind, bind);
    CHECK(binding_replay.family_binding == binding.family_binding);

    common_cache_family_binding resolved;
    CHECK(authority.resolve_family_binding(
              binding.family_binding, resolved) ==
          server_cache_control_status::ok);
    CHECK(resolved == binding.cache_family);

    const auto other = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(100));
    auto foreign = bind;
    foreign.holder = other.holder;
    foreign.idempotency_key = 0;
    CHECK(execute(authority, server_cache_control_operation::family_bind,
                  foreign).status == server_cache_control_status::not_found);

    clock.now = 101;
    authority.lifecycle_point();
    resolved = {};
    CHECK(authority.resolve_family_binding(
              binding.family_binding, resolved) ==
          server_cache_control_status::not_found);
    CHECK(!resolved.declared());

    // A poisoned idempotency/grant path is an internal authority fault, not
    // capacity pressure. This mirrors lease-acquire's status discipline.
    server_cache_lease_table poisoned_leases(&clock);
    server_retention_sidecar_store poisoned_retention;
    auto poisoned_config = config;
    poisoned_config.leases = &poisoned_leases;
    poisoned_config.retention = &poisoned_retention;
    poisoned_config.test_fail_remember = true;
    server_cache_control_authority poisoned(poisoned_config);
    auto poisoned_holder_request = create_holder(200);
    poisoned_holder_request.idempotency_key = 99;
    const auto poisoned_holder = execute(
        poisoned, server_cache_control_operation::holder_create,
        poisoned_holder_request);
    CHECK(poisoned_holder.status == server_cache_control_status::ok);
    server_cache_control_request poisoned_family;
    poisoned_family.holder = poisoned_holder.holder;
    CHECK(execute(
        poisoned, server_cache_control_operation::family_register,
        poisoned_family).status ==
            server_cache_control_status::internal_fault);
    std::puts("E1_FAMILY registry_binding PASS role=main expired=not_found");
}

static void test_production_store_resolver_leg() {
    llama_cache_acct_ledger ledger;
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    const llama_cache_acct_completeness_requirement requirement {
        pinned, llama_cache_acct_producer::retention_sidecar,
    };
    CHECK(ledger.configure_required_producers(&requirement, 1));
    for (const auto measure : {
            llama_cache_acct_measure::logical_payload,
            llama_cache_acct_measure::resident_allocated,
            llama_cache_acct_measure::reserved }) {
        ledger.gauge_set(
            llama_cache_acct_category::pinned_preimage_ring,
            pinned, measure, 0);
    }
    CHECK(ledger.certify_complete(
        pinned, llama_cache_acct_producer::retention_sidecar));
    CHECK(server_vbr_artifact_store_configure_pinned_accounting(
        ledger, pinned));

    server_vbr_artifact_store_config store_config;
    store_config.ledger = &ledger;
    store_config.pinned_domain = pinned;
    store_config.topologies.push_back(control_store_topology());
    store_config.pool_bindings.push_back({
        { 0x1111, 0x2222 }, 0, 0, 0, 0,
    });
    store_config.lanes.push_back({});
    store_config.attention_children = 1;
    store_config.ring_bytes = 16;
    store_config.chunk_bytes = 8;
    store_config.sample_budget = sample_control_store_budget;
    server_vbr_artifact_capture_status store_status;
    auto store = server_vbr_artifact_store::create(
        store_config, store_status);
    CHECK(store != nullptr);
    CHECK(store_status == server_vbr_artifact_capture_status::ok);
    if (!store) {
        return;
    }

    control_test_clock clock;
    control_test_tokens tokens;
    server_cache_lease_table leases(&clock);
    server_retention_sidecar_store retention;
    server_cache_control_config config;
    config.leases = &leases;
    config.retention = &retention;
    config.artifacts = store.get();
    config.clock = &clock;
    config.tokens = &tokens;
    // Deliberately omit resolve_vbr: this must install and execute the same
    // store-backed default callback production uses.
    server_cache_control_authority authority(config);
    const auto holder = execute(
        authority, server_cache_control_operation::holder_create,
        create_holder(100));
    CHECK(holder.status == server_cache_control_status::ok);
    server_cache_control_request acquire;
    acquire.holder = holder.holder;
    acquire.requested_class = server_cache_lease_class::soft;
    acquire.ttl_ns = 20;
    acquire.subject.kind =
        server_cache_control_subject_kind::vbr_reference;
    acquire.subject.reference = "vbrref_missing";
    acquire.subject.tenant_key = "tenant";
    CHECK(execute(authority, server_cache_control_operation::lease_acquire,
                  acquire).status == server_cache_control_status::not_found);
    CHECK(store->counters().requested == 0);
    std::puts("E1_STORE_RESOLVER default_leg PASS status=not_found");
}

int main() {
    test_holder_hard_orphan_frontier_and_proof();
    test_refusals_soft_expiry_and_ownership();
    test_poisoned_evidence_still_releases();
    test_task_lifecycle_gate();
    test_family_registry_and_binding();
    test_production_store_resolver_leg();
    if (failures != 0) {
        std::fprintf(stderr, "test-cache-control: %d failures\n", failures);
        return 1;
    }
    std::puts("test-cache-control: PASS");
    return 0;
}
