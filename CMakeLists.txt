cmake_minimum_required(VERSION 3.28)
project(gpuT2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ----------------------------------------------------------------------------
# Core: backend-agnostic interface layer (Grid, Config, LifeRules, ISimEngine,
# IRenderer, patterns). Header-only for now, so this is an INTERFACE target that
# just propagates the include directory and the C++23 requirement. It becomes a
# STATIC library once src/core/Grid.cpp and src/core/Config.cpp are implemented.
# ----------------------------------------------------------------------------
add_library(gol_core INTERFACE)
target_include_directories(gol_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(gol_core INTERFACE cxx_std_23)

# ----------------------------------------------------------------------------
# CPU engine (always built once its source exists). The CpuEngine.cpp body is
# the next pass; the target is wired here but skipped until the source lands so
# the default configuration builds cleanly today.
#
#   add_library(gol_cpu src/engines/cpu/CpuEngine.cpp)
#   target_link_libraries(gol_cpu PUBLIC gol_core)
# ----------------------------------------------------------------------------
set(_cpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/engines/cpu/CpuEngine.cpp)
if(EXISTS ${_cpu_src})
    add_library(gol_cpu ${_cpu_src})
    target_link_libraries(gol_cpu PUBLIC gol_core)
endif()

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
        target_link_libraries(gol_cuda PUBLIC gol_core)
        set_target_properties(gol_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
