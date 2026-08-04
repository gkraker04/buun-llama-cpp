# Shared primitive for the source-contract scan scripts (check-*.cmake). One definition —
# a counting bug here would weaken every contract gate at once.

function(count_literal text needle output)
    string(LENGTH "${text}" before)
    string(REPLACE "${needle}" "" stripped "${text}")
    string(LENGTH "${stripped}" after)
    string(LENGTH "${needle}" needle_length)
    math(EXPR count "(${before} - ${after}) / ${needle_length}")
    set(${output} "${count}" PARENT_SCOPE)
endfunction()

# Extract the half-open source region [begin marker, end marker). Callers use
# the explicit found result so a missing/misordered marker cannot accidentally
# become an empty region that passes a forbidden-token scan.
function(contract_extract_region text begin_marker end_marker output found_output)
    string(FIND "${text}" "${begin_marker}" begin)
    string(FIND "${text}" "${end_marker}" end)
    if (begin EQUAL -1 OR end LESS_EQUAL begin)
        set(${output} "" PARENT_SCOPE)
        set(${found_output} FALSE PARENT_SCOPE)
        return()
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${text}" ${begin} ${length} region)
    set(${output} "${region}" PARENT_SCOPE)
    set(${found_output} TRUE PARENT_SCOPE)
endfunction()

# Return every literal from ARGN present in text. Keeping extraction and
# forbidden matching in one shared utility avoids subtly weaker ratchet scans.
function(contract_find_forbidden text output)
    set(hits "")
    foreach(token IN LISTS ARGN)
        string(FIND "${text}" "${token}" found)
        if (NOT found EQUAL -1)
            list(APPEND hits "${token}")
        endif()
    endforeach()
    set(${output} "${hits}" PARENT_SCOPE)
endfunction()
