#pragma once

#include <cstddef>
#include <cstdint>

struct common_cache_family_id {
    uint64_t value = 0;

    constexpr explicit operator bool() const noexcept { return value != 0; }
};

constexpr bool operator==(
        common_cache_family_id a,
        common_cache_family_id b) noexcept {
    return a.value == b.value;
}

constexpr bool operator!=(
        common_cache_family_id a,
        common_cache_family_id b) noexcept {
    return !(a == b);
}

enum class common_cache_family_role : uint8_t {
    main = 0,
    branch,
    background,
    _count,
};

// Strong scheduler-resolved family policy. The default remains undeclared;
// only the holder-owned E1 registry may construct a production declaration.
struct common_cache_family_binding {
    common_cache_family_id family;
    common_cache_family_role role = common_cache_family_role::background;

    constexpr bool declared() const noexcept {
        return bool(family) && role < common_cache_family_role::_count;
    }
};

constexpr bool operator==(
        const common_cache_family_binding & a,
        const common_cache_family_binding & b) noexcept {
    return a.family == b.family && a.role == b.role;
}

constexpr bool operator!=(
        const common_cache_family_binding & a,
        const common_cache_family_binding & b) noexcept {
    return !(a == b);
}

// The single family-role -> main_family resolver. An absent declaration keeps
// the historical automatic parent/child result byte-for-byte.
constexpr bool common_cache_family_main_family(
        const common_cache_family_binding & binding,
        bool automatic_main_family) noexcept {
    return binding.declared()
        ? binding.role == common_cache_family_role::main
        : automatic_main_family;
}

// A declared role already occupies the one main_family price carrier. The
// independent policy callback must therefore be neutral for that entry.
constexpr bool common_cache_family_allows_additional_weight(
        const common_cache_family_binding & binding) noexcept {
    return !binding.declared();
}

enum class common_cache_retention_provenance : uint8_t {
    neutral = 0,
    proven_server_parent,
    _count,
};

// One copy-safe carrier for all retention-role evidence. The declaration is
// holder-authenticated E1 state; automatic provenance is deliberately much
// narrower and can only be minted by the scheduler from an established
// server parent/child relation.
struct common_cache_retention_lineage {
    common_cache_family_binding declaration;
    common_cache_retention_provenance automatic_provenance =
        common_cache_retention_provenance::neutral;
};

constexpr bool operator==(
        const common_cache_retention_lineage & a,
        const common_cache_retention_lineage & b) noexcept {
    return a.declaration == b.declaration &&
           a.automatic_provenance == b.automatic_provenance;
}

constexpr bool operator!=(
        const common_cache_retention_lineage & a,
        const common_cache_retention_lineage & b) noexcept {
    return !(a == b);
}

constexpr bool common_cache_retention_lineage_main(
        const common_cache_retention_lineage & lineage) noexcept {
    return lineage.declaration.declared()
        ? lineage.declaration.role == common_cache_family_role::main
        : lineage.automatic_provenance ==
              common_cache_retention_provenance::proven_server_parent;
}

// One policy-aware spelling of the lineage -> main_family carrier. ZC's
// intentional policy consumes only the declared/proven lineage; the historical
// path preserves its caller-supplied automatic parent/child classification.
constexpr bool common_cache_retention_main_family(
        const common_cache_retention_lineage & lineage,
        bool intentional_retention,
        bool historical_automatic_main) noexcept {
    return intentional_retention
        ? common_cache_retention_lineage_main(lineage)
        : common_cache_family_main_family(
              lineage.declaration, historical_automatic_main);
}

constexpr const char * common_cache_retention_lineage_name(
        const common_cache_retention_lineage & lineage) noexcept {
    if (lineage.declaration.declared()) {
        return lineage.declaration.role == common_cache_family_role::main
            ? "declared_main"
            : lineage.declaration.role == common_cache_family_role::branch
                ? "declared_branch" : "declared_background";
    }
    return lineage.automatic_provenance ==
            common_cache_retention_provenance::proven_server_parent
        ? "proven_server_parent" : "neutral";
}

constexpr common_cache_retention_lineage
common_cache_retention_follow_lineage(
        const common_cache_retention_lineage & current,
        const common_cache_retention_lineage & incoming,
        size_t retained_prefix,
        size_t current_tokens) noexcept {
    return current_tokens != 0 && retained_prefix == current_tokens
        ? current : incoming;
}
