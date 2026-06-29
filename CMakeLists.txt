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
set(GOL_RENDER_SOURCES
        src/render/TextRenderer.cpp
)

if(NOT WIN32)
    list(APPEND GOL_RENDER_SOURCES
            src/render/AnsiRenderer.cpp
    )
endif()

add_library(gol_render ${GOL_RENDER_SOURCES})
target_link_libraries(gol_render PUBLIC gol_core)
# ----------------------------------------------------------------------------
# Application: owns the loop and wires a chosen engine + renderer. Base build is the
# CPU engine + null/text/ansi renderers; the CUDA engine links in under -DBUILD_CUDA=ON.
# Builds and runs with no CUDA toolkit and no OpenCL present.
# ----------------------------------------------------------------------------
add_executable(gol src/core/main.cpp)
target_link_libraries(gol PRIVATE gol_core gol_cpu gol_render)

# ----------------------------------------------------------------------------
# Optional CUDA engine (opt-in; double-gated on BUILD_CUDA and the source existing).
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

        # kernel.cl is read at runtime (per the design), so bake in its absolute
        # source path -- otherwise the binary only finds the kernel when launched
        # from the repo root.
        target_compile_definitions(gol_opencl PRIVATE
            GOL_OPENCL_KERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/engines/opencl/kernel.cl"
            # The host code uses the OpenCL 1.2 API surface (clCreateCommandQueue,
            # etc.); pinning the target version silences the headers' "version not
            # defined" pragma and the 2.0+ deprecation warning.
            CL_TARGET_OPENCL_VERSION=120)

        target_link_libraries(gol PRIVATE gol_opencl)
        target_compile_definitions(gol PRIVATE GOL_HAVE_OPENCL)

        set(GOL_OPENCL_ENABLED ON)
    else()
        message(WARNING "BUILD_OPENCL=ON but ${_opencl_src} does not exist yet; "
                "skipping the OpenCL engine.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Optional Qt OpenGL viewer (opt-in). Builds a separate `gol_gui` executable; the
# headless `gol` benchmark binary is untouched and never links Qt or OpenGL.
#
# Runs on the CPU engine alone (no CUDA toolkit needed). When the CUDA engine is
# also configured (-DBUILD_CUDA=ON), it additionally offers the zero-copy CUDA/GL
# interop display path; otherwise (and as a fallback) it uses host-upload display.
# ----------------------------------------------------------------------------
option(BUILD_GUI "Build the Qt OpenGL Game of Life viewer" OFF)
if(BUILD_GUI)
    find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)

    set(_gui_libs gol_core gol_cpu
        Qt6::Core Qt6::Gui Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets)
    set(_gui_defs "")

    if(GOL_CUDA_ENABLED)
        find_package(CUDAToolkit REQUIRED)
        # CUDA<->GL interop bridge: compiled by nvcc (it includes cuda_gl_interop.h),
        # but exposes a CUDA-free header so the Qt code (host compiler) can call it.
        add_library(gol_cudagl ${CMAKE_CURRENT_SOURCE_DIR}/src/render/CudaGlInterop.cu)
        target_include_directories(gol_cudagl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        set_target_properties(gol_cudagl PROPERTIES
            CUDA_ARCHITECTURES 75
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON)
        target_link_libraries(gol_cudagl PUBLIC CUDA::cudart)

        list(APPEND _gui_libs gol_cuda gol_cudagl CUDA::cudart)
        list(APPEND _gui_defs GOL_HAVE_CUDA)
    endif()

    add_executable(gol_gui
        src/gui/main_gui.cpp
        src/gui/GolGlWidget.cpp
        src/gui/MainWindow.cpp
        # Headers carry Q_OBJECT and live apart from their .cpp, so list them
        # explicitly for AUTOMOC to scan (it only auto-pairs same-dir basenames).
        include/gol/gui/GolGlWidget.hpp
        include/gol/gui/MainWindow.hpp
    )
    set_target_properties(gol_gui PROPERTIES AUTOMOC ON)
    target_link_libraries(gol_gui PRIVATE ${_gui_libs})
    # Shaders are read at runtime (like kernel.cl), so bake in their source dir.
    target_compile_definitions(gol_gui PRIVATE
        GOL_GUI_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/gui/shaders" ${_gui_defs})
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
