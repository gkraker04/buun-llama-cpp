# Merge tripwire: every llama_memory_i subclass must be either handled by the
# checkpoint memory-tree walker or explicitly declared unsupported here.
#
# The walker (src/llama-memory-tree.cpp) is an ordered dynamic_cast ladder that
# falls through to `return false`, which makes checkpoint capture/adopt
# UNAVAILABLE for any memory class it does not name — silently, with no compile
# error and no merge conflict, because the walker is fork-owned and upstream
# never touches it. This scan turns that silence into a test failure: adding a
# memory class (ours or upstream's) forces an explicit decision — extend the
# ladder, or record the class below with a reason.
#
# Run: cmake -DSOURCE_ROOT=<repo> -P tests/check-memory-tree-coverage.cmake

if (NOT DEFINED SOURCE_ROOT)
    message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

# Classes deliberately NOT covered by the walker. Every entry needs a reason and
# an owner decision; stale entries (class gone, or class now in the ladder) fail
# the scan so this list cannot rot.
# None of these were ever *decided* unsupported — they postdate the walker and no
# one extended it. This list records that fact instead of letting it stay silent.
# The walker only feeds the VBR artifact capture/adopt paths; ordinary context
# checkpoints do not route through it.
#   llama_kv_cache_dsa  — composes TWO coupled llama_kv_cache children (kv_mla +
#                         kv_lid lightning-indexer keys, layer-filtered); the
#                         walker taxonomy has no coupled-attention-pair kind, and
#                         the inner caches accept turbo/VBR types (epoch already
#                         forwarded as the sum of both) — supporting this is a
#                         real design task, not a ladder entry.
#   llama_kv_cache_dsv4 — postdates the walker; composition unaudited; no VBR
#                         wiring in its header today, so capture is unreachable
#                         for it until that changes.
#   llama_kv_cache_msa  — arrived with the 2026-08-09 sync (74ce15741); same
#                         status as dsv4.
set(KNOWN_UNCOVERED
    llama_kv_cache_dsa
    llama_kv_cache_dsv4
    llama_kv_cache_msa)

file(GLOB memory_headers
    "${SOURCE_ROOT}/src/llama-memory*.h"
    "${SOURCE_ROOT}/src/llama-kv-cache*.h")

set(subclasses "")
foreach(path IN LISTS memory_headers)
    file(READ "${path}" text)
    string(REGEX MATCHALL "class (llama_[a-z0-9_]+)[ \t\r\n]*:[ \t\r\n]*public llama_memory_i" hits "${text}")
    foreach(hit IN LISTS hits)
        string(REGEX REPLACE "class (llama_[a-z0-9_]+).*" "\\1" name "${hit}")
        list(APPEND subclasses "${name}")
    endforeach()
endforeach()
list(REMOVE_DUPLICATES subclasses)
list(LENGTH subclasses n_subclasses)
if (n_subclasses EQUAL 0)
    message(FATAL_ERROR "memory-tree scan matched no llama_memory_i subclasses; the scan regex is stale, fix it rather than deleting the test")
endif()

file(READ "${SOURCE_ROOT}/src/llama-memory-tree.cpp" walker)
string(REGEX MATCHALL "dynamic_cast<(llama_[a-z0-9_]+)[ \t]*\\*>" casts "${walker}")
set(ladder "")
foreach(hit IN LISTS casts)
    string(REGEX REPLACE "dynamic_cast<(llama_[a-z0-9_]+).*" "\\1" name "${hit}")
    list(APPEND ladder "${name}")
endforeach()
list(REMOVE_DUPLICATES ladder)

set(errors "")
foreach(cls IN LISTS subclasses)
    list(FIND ladder "${cls}" in_ladder)
    list(FIND KNOWN_UNCOVERED "${cls}" in_known)
    if (in_ladder EQUAL -1 AND in_known EQUAL -1)
        string(APPEND errors "  NEW UNHANDLED MEMORY CLASS: ${cls} — checkpoint capture will silently return unavailable for it. Extend the ladder in src/llama-memory-tree.cpp (mind cast ORDER: derived before base) or add it to KNOWN_UNCOVERED with a reason.\n")
    endif()
endforeach()
foreach(cls IN LISTS KNOWN_UNCOVERED)
    list(FIND subclasses "${cls}" still_exists)
    if (still_exists EQUAL -1)
        string(APPEND errors "  STALE KNOWN_UNCOVERED ENTRY: ${cls} no longer subclasses llama_memory_i — remove it from the list.\n")
    endif()
    list(FIND ladder "${cls}" now_covered)
    if (NOT now_covered EQUAL -1)
        string(APPEND errors "  STALE KNOWN_UNCOVERED ENTRY: ${cls} is now handled by the walker — remove it from the list.\n")
    endif()
endforeach()
foreach(cls IN LISTS ladder)
    list(FIND subclasses "${cls}" still_exists)
    if (still_exists EQUAL -1)
        string(APPEND errors "  LADDER CASTS A NON-SUBCLASS: ${cls} is in the walker but no longer subclasses llama_memory_i — upstream refactored the hierarchy; re-derive the ladder.\n")
    endif()
endforeach()

if (errors)
    message(FATAL_ERROR "memory-tree coverage scan FAILED:\n${errors}")
endif()
message(STATUS "memory-tree coverage OK: ${n_subclasses} llama_memory_i subclasses = ladder + KNOWN_UNCOVERED, no stale entries")
