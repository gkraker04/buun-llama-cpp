#pragma once

#include "server-cache-lease.h"

class vbr_artifact_package_view;

// Private E1.0 test adapter. E1.1a replaces this door with the tenant-bound
// F-reference authorization resolver.
server_cache_durable_fallback_proof
server_cache_vbr_fallback_proof_for_test(
    vbr_artifact_package_view && package) noexcept;
