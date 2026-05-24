cmake_minimum_required(VERSION 3.28)
project(gpuT2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# ----------------------------------------------------------------------------
# Core library: CPU (sequential) Game of Life
# ----------------------------------------------------------------------------
add_library(GameOfLife_lib
  src/GameOfLife.cpp
)

target_include_directories(GameOfLife_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------------------------------------------------------------------------
# Optional CUDA build
# ----------------------------------------------------------------------------
option(BUILD_CUDA "Build CUDA implementation" OFF)
if(BUILD_CUDA)
    enable_language(CUDA)
    add_library(GameOfLife_cuda
        src/GameOfLife.cu
    )
    target_include_directories(GameOfLife_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    set_target_properties(GameOfLife_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ----------------------------------------------------------------------------
# Optional OpenCL build
# ----------------------------------------------------------------------------
option(BUILD_OPENCL "Build OpenCL implementation" OFF)
if(BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    add_library(GameOfLife_opencl
        src/GameOfLife_cl.cpp
    )
    target_include_directories(GameOfLife_opencl
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(GameOfLife_opencl
        PUBLIC
            OpenCL::OpenCL
    )
endif()

include(CTest)

if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_TESTING "Build tests" ON)
else()
    option(BUILD_TESTING "Build tests" OFF)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
