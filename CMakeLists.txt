cmake_minimum_required(VERSION 3.28)
project(gpuT2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ----------------------------------------------------------------------------
# Core: backend-agnostic layer (Grid, Config + CLI, LifeRules, ISimEngine,
# IRenderer, patterns + RLE loader). A STATIC library now that the parsing and
# pattern sources exist; the header-only pieces (Grid, LifeRules, Timer) still
# live entirely in include/.
# ----------------------------------------------------------------------------
add_library(gol_core STATIC
    src/core/Config.cpp
    src/patterns/Pattern.cpp
    src/patterns/RleLoader.cpp
)
target_include_directories(gol_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(gol_core PUBLIC cxx_std_23)

# ----------------------------------------------------------------------------
# CPU engine (always built — no toolkit required). One implementation serves both
# the sequential baseline (1 core) and the data-parallel mode (N cores).
# ----------------------------------------------------------------------------
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_library(gol_cpu src/engines/cpu/CpuEngine.cpp)
target_link_libraries(gol_cpu PUBLIC gol_core Threads::Threads)

# ----------------------------------------------------------------------------
# Renderers (NullRenderer is header-only; TextRenderer has a body).
# ----------------------------------------------------------------------------
add_library(gol_render
    src/render/TextRenderer.cpp
    src/render/AnsiRenderer.cpp
)
target_link_libraries(gol_render PUBLIC gol_core)

# ----------------------------------------------------------------------------
# Application: owns the loop and wires a chosen engine + renderer (CPU + null/text
# today). Builds and runs with no CUDA toolkit and no OpenCL present.
# ----------------------------------------------------------------------------
add_executable(gol src/core/main.cpp)
target_link_libraries(gol PRIVATE gol_core gol_cpu gol_render)

# ----------------------------------------------------------------------------
# Optional CUDA engine (opt-in, and only once its sources exist).
# ----------------------------------------------------------------------------
option(BUILD_CUDA "Build the CUDA engine" OFF)
if(BUILD_CUDA)
    set(_cuda_src ${CMAKE_CURRENT_SOURCE_DIR}/src/engines/cuda/CudaEngine.cu)
    if(EXISTS ${_cuda_src})
        enable_language(CUDA)
        add_library(gol_cuda
            ${_cuda_src}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/engines/cuda/kernel.cu
        )
        # gol_cuda only consumes header-only pieces (LifeRules.hpp, Grid.hpp), so it
        # needs gol_core's include dir but must NOT link gol_core -- that would
        # propagate the cxx_std_23 requirement onto nvcc, which has no such flag.
        target_include_directories(gol_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        set_target_properties(gol_cuda PROPERTIES
            CUDA_ARCHITECTURES 75              # GTX 1660 Ti (Turing); override as needed
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_STANDARD 17                   # device TUs need only C++17; host/app stays C++23
            CUDA_STANDARD_REQUIRED ON
        )
        target_compile_options(gol_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>) # source attribution for ncu

        # Wire the CUDA engine into the application.
        target_link_libraries(gol PRIVATE gol_cuda)
        target_compile_definitions(gol PRIVATE GOL_HAVE_CUDA)
        set(GOL_CUDA_ENABLED ON)               # consumed by tests/ (added later)
    else()
        message(WARNING "BUILD_CUDA=ON but ${_cuda_src} does not exist yet; "
                        "skipping the CUDA engine.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Optional OpenCL engine (opt-in, and only once its source exists).
# ----------------------------------------------------------------------------
option(BUILD_OPENCL "Build the OpenCL engine" OFF)
if(BUILD_OPENCL)
    set(_opencl_src ${CMAKE_CURRENT_SOURCE_DIR}/src/engines/opencl/OpenCLEngine.cpp)
    if(EXISTS ${_opencl_src})
        find_package(OpenCL REQUIRED)
        add_library(gol_opencl ${_opencl_src})
        target_link_libraries(gol_opencl PUBLIC gol_core OpenCL::OpenCL)
    else()
        message(WARNING "BUILD_OPENCL=ON but ${_opencl_src} does not exist yet; "
                        "skipping the OpenCL engine.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
include(CTest)

if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_TESTING "Build tests" ON)
else()
    option(BUILD_TESTING "Build tests" OFF)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
