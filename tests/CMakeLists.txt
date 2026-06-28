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

# Cross-backend equivalence: CudaEngine vs the CpuEngine oracle. Only built when
# the CUDA engine was configured (-DBUILD_CUDA=ON). Requires a GPU at run time.
if(GOL_CUDA_ENABLED)
    add_executable(gol_cuda_tests cuda_equivalence_test.cpp)
    target_link_libraries(gol_cuda_tests
        PRIVATE
            gol_core
            gol_cpu
            gol_cuda
            GTest::gtest_main
    )
    gtest_discover_tests(gol_cuda_tests)
endif()

# Cross-backend equivalence: OpenCLEngine vs the CpuEngine oracle, plus the
# host<->device round-trip and buffer-reuse lifecycle checks. Only built when the
# OpenCL engine was configured (-DBUILD_OPENCL=ON). Requires an OpenCL device at
# run time.
if(GOL_OPENCL_ENABLED)
    add_executable(gol_opencl_tests opencl_equivalence_test.cpp)
    target_link_libraries(gol_opencl_tests
        PRIVATE
            gol_core
            gol_cpu
            gol_opencl
            GTest::gtest_main
    )
    gtest_discover_tests(gol_opencl_tests)
endif()

# Cross-backend equivalence: HybridEngine (CPU + GPU, static load balancing) vs the
# CpuEngine oracle, across CPU fractions, both edge modes, and the calibrated split.
# Only built when the hybrid engine was configured (a GPU backend present). Requires
# a GPU/OpenCL device at run time. GOL_HAVE_{CUDA,OPENCL} tell the test which GPU
# backends to exercise.
if(GOL_HYBRID_ENABLED)
    add_executable(gol_hybrid_tests hybrid_equivalence_test.cpp)
    target_link_libraries(gol_hybrid_tests
        PRIVATE
            gol_core
            gol_cpu
            gol_hybrid
            GTest::gtest_main
    )
    if(GOL_CUDA_ENABLED)
        target_compile_definitions(gol_hybrid_tests PRIVATE GOL_HAVE_CUDA)
    endif()
    if(GOL_OPENCL_ENABLED)
        target_compile_definitions(gol_hybrid_tests PRIVATE GOL_HAVE_OPENCL)
    endif()
    gtest_discover_tests(gol_hybrid_tests)
endif()
