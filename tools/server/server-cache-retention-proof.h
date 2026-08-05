#pragma once

#include "server-cache-lease.h"

class server_cache_recovery_pin;

// Private E1.0 test adapter. E1.1a replaces this door with the
// scheduler-owned identity-authorized resolver.
server_cache_durable_fallback_proof
server_cache_retention_fallback_proof_for_test(
    server_cache_recovery_pin && pin) noexcept;
