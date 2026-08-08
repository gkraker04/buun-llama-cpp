include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-cache-authority.cpp
    server-cache-authority.h
    server-cache-control.cpp
    server-cache-control-wire.cpp
    server-cache-control.h
    server-cache-destruction-quote.cpp
    server-cache-destruction-quote.h
    server-cache-plan-authority.cpp
    server-cache-plan-authority.h
    server-cache-plan-preflight.cpp
    server-cache-plan-preflight.h
    server-cache-plan-preflight-internal.h
    server-cache-observer.cpp
    server-cache-observer.h
    server-cache-calibration-store.cpp
    server-cache-calibration-store.h
    server-cache-fingerprint.cpp
    server-cache-fingerprint.h
    server-cache-lease.cpp
    server-cache-lease.h
    server-cache-retention-proof.cpp
    server-cache-retention-proof.h
    server-cache-retention-policy.cpp
    server-cache-retention-policy.h
    server-cache-vbr-proof.cpp
    server-cache-vbr-proof.h
    server-cache-yield.cpp
    server-cache-yield.h
    server-vbr-artifact-store.cpp
    server-vbr-artifact-store.h
    server-retention-sidecar.cpp
    server-retention-sidecar.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
if (WIN32)
    target_link_libraries(${TARGET} PRIVATE bcrypt)
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# C++ unit test for server_tokens::get_common_prefix (U3 empty-id media guard). Lives here rather
# than tests/ because that subdir is configured before tools/, so server-context/mtmd targets are
# not yet defined there. No model or mmproj needed — runs on CPU.
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-media-prefix ${CMAKE_SOURCE_DIR}/tests/test-server-media-prefix.cpp)
    target_link_libraries(test-server-media-prefix PRIVATE server-context mtmd llama llama-common)
    target_include_directories(test-server-media-prefix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    target_compile_features(test-server-media-prefix PRIVATE cxx_std_17)
    add_test(NAME test-server-media-prefix COMMAND test-server-media-prefix)
    set_property(TEST test-server-media-prefix PROPERTY LABELS main)

    add_executable(test-server-prompt-cache ${CMAKE_SOURCE_DIR}/tests/test-server-prompt-cache.cpp)
    target_link_libraries(test-server-prompt-cache PRIVATE server-context mtmd llama llama-common)
    target_include_directories(test-server-prompt-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/tools/mtmd)
    target_compile_features(test-server-prompt-cache PRIVATE cxx_std_17)
    add_test(NAME test-server-prompt-cache COMMAND test-server-prompt-cache)
    set_property(TEST test-server-prompt-cache PROPERTY LABELS main)
    add_test(NAME test-server-prompt-cache-clone-fault
        COMMAND test-server-prompt-cache --clone-fault)
    set_tests_properties(test-server-prompt-cache-clone-fault PROPERTIES
        LABELS main
        ENVIRONMENT "LLAMA_SERVER_FAULT=load_clone_fail")

    add_executable(test-server-retention-policy
        ${CMAKE_SOURCE_DIR}/tests/test-server-retention-policy.cpp)
    target_link_libraries(test-server-retention-policy PRIVATE server-context)
    target_include_directories(test-server-retention-policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test-server-retention-policy PRIVATE cxx_std_17)
    add_test(NAME test-server-retention-policy COMMAND test-server-retention-policy)
    set_property(TEST test-server-retention-policy PROPERTY LABELS main)

    add_executable(test-server-cache-observer
        ${CMAKE_SOURCE_DIR}/tests/test-server-cache-observer.cpp)
    target_link_libraries(test-server-cache-observer PRIVATE server-context)
    target_include_directories(test-server-cache-observer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test-server-cache-observer PRIVATE cxx_std_17)
    add_test(NAME test-server-cache-observer COMMAND test-server-cache-observer)
    set_property(TEST test-server-cache-observer PROPERTY LABELS main)

    add_executable(test-server-cache-fingerprint
        ${CMAKE_SOURCE_DIR}/tests/test-server-cache-fingerprint.cpp)
    target_link_libraries(test-server-cache-fingerprint PRIVATE server-context)
    target_include_directories(test-server-cache-fingerprint PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test-server-cache-fingerprint PRIVATE cxx_std_17)
    add_test(NAME test-server-cache-fingerprint COMMAND test-server-cache-fingerprint)
    set_property(TEST test-server-cache-fingerprint PROPERTY LABELS main)

    add_executable(test-server-cache-calibration-store
        ${CMAKE_SOURCE_DIR}/tests/test-server-cache-calibration-store.cpp)
    target_link_libraries(test-server-cache-calibration-store PRIVATE server-context)
    target_include_directories(test-server-cache-calibration-store PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test-server-cache-calibration-store PRIVATE cxx_std_17)
    add_test(NAME test-server-cache-calibration-store
        COMMAND test-server-cache-calibration-store)
    set_property(TEST test-server-cache-calibration-store PROPERTY LABELS main)

    add_executable(zc1-retention-grid bench/zc1-retention-grid.cpp)
    target_link_libraries(zc1-retention-grid PRIVATE server-context)
    target_include_directories(zc1-retention-grid PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(zc1-retention-grid PRIVATE cxx_std_17)
    find_program(ZC1_PYTHON_EXECUTABLE NAMES python3 REQUIRED)
    add_test(NAME test-zc1-retention-tools
        COMMAND ${ZC1_PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/test-zc1-retention-tools.py
            --grid $<TARGET_FILE:zc1-retention-grid>)
    set_property(TEST test-zc1-retention-tools PROPERTY LABELS main)
endif()
