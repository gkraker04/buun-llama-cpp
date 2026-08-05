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

// Family provenance follows the retained immutable content. A zero-overlap
// launch replaces the conversation lineage with the incoming declaration (or
// the undeclared default); any retained prefix keeps the existing lineage.
constexpr common_cache_family_binding common_cache_family_follow_lineage(
        const common_cache_family_binding & current,
        const common_cache_family_binding & incoming,
        size_t retained_prefix) noexcept {
    return retained_prefix == 0 ? incoming : current;
}
