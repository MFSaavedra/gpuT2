include(FetchContent)

# First, try to use a system-installed GTest
find_package(GTest CONFIG QUIET)

# If not found, fetch it from upstream
if(NOT GTest_FOUND)
    message(STATUS "GTest not found locally, fetching from GitHub...")

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
    )

    # Prevent GoogleTest from overriding compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# Compile-smoke test: includes every interface-layer header so the scaffolding is
# verified to build before any engine/renderer bodies exist.
add_executable(gol_tests
    compile_smoke_test.cpp
)

target_link_libraries(gol_tests
    PRIVATE
        gol_core
        GTest::gtest_main
)

gtest_discover_tests(gol_tests)

# Engine + loader tests: CpuEngine correctness under the variant rule, the RLE
# loader, and the sequential-vs-parallel equivalence. PATTERNS_DIR points the
# tests at the repo's RLE fixtures regardless of the working directory.
add_executable(gol_engine_tests
    rules_test.cpp
    rle_loader_test.cpp
    cpu_parallel_test.cpp
)

target_link_libraries(gol_engine_tests
    PRIVATE
        gol_core
        gol_cpu
        GTest::gtest_main
)

target_compile_definitions(gol_engine_tests
    PRIVATE
        PATTERNS_DIR="${CMAKE_SOURCE_DIR}/patterns"
)

gtest_discover_tests(gol_engine_tests)
