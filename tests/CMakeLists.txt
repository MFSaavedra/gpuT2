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
