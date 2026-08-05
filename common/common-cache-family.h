#pragma once

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

// E1.0 foundation only. Production does not construct a declared binding
// until the scheduler-owned family registry lands in E1.1b.
struct common_cache_family_binding {
    common_cache_family_id family;
    common_cache_family_role role = common_cache_family_role::background;

    constexpr bool declared() const noexcept {
        return bool(family) && role < common_cache_family_role::_count;
    }
};

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
