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
