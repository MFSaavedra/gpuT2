# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University assignment (Tarea 2, course CC7515-1 "Computación en GPU", U. de Chile): implement
Conway's Game of Life three times — sequential **CPU**, **CUDA**, and **OpenCL** — then benchmark
cells-evaluated-per-second across grid sizes and report the GPU speed-up vs CPU.

**Rule variant (do not use vanilla Conway rules):** a dead cell is born on **exactly 3 OR exactly 6**
live neighbours; a live cell survives on 2 or 3. The extra "birth on 6" rule is the one detail that's
easy to get wrong — it must hold in every implementation (CPU, CUDA, OpenCL).

## Current state — skeleton being restructured

The repo started as a single `GameOfLife` class with a `// TODO` `step()` stub. It is being migrated to
the engine/renderer **strategy** layout described below:

- The old `GameOfLife` class splits into a backend-agnostic `Grid` plus a `CpuEngine` that implements
  `ISimEngine`. The stubbed `step()` logic becomes `CpuEngine::step()`.
- CUDA (`CudaEngine`) and OpenCL (`OpenCLEngine`) are greenfield — no device sources exist yet.
- GPU CMake targets are opt-in via `-DBUILD_CUDA=ON` / `-DBUILD_OPENCL=ON` and won't configure until
  their sources exist.

## Build & test

Requires a C++23 compiler and CMake ≥ 3.28.

```bash
cmake -S . -B build              # core + CPU engine + tests (GPU targets OFF by default)
cmake --build build
ctest --test-dir build --output-on-failure
```

GPU targets are opt-in:

```bash
cmake -S . -B build -DBUILD_CUDA=ON -DBUILD_OPENCL=ON
```

Run a subset of tests via ctest regex:

```bash
ctest --test-dir build -R Rules --output-on-failure
```

GoogleTest is found via `find_package(GTest CONFIG)` if installed system-wide; otherwise it is fetched
(v1.17.0) and built automatically. Tests are registered with `gtest_discover_tests`.

## Architecture

Strategy pattern. A backend-agnostic core (`Grid`, `Config`, patterns) plus two swappable interfaces —
`ISimEngine` (the only thing that genuinely differs per backend) and `IRenderer` (output). The
application owns the loop and wires a chosen engine + renderer at runtime:

```
Application (main loop, Config/CLI)
  |-- ISimEngine   <- CpuEngine | CudaEngine | OpenCLEngine
  |-- IRenderer    <- NullRenderer | TextRenderer | (GuiRenderer later)
  '-- Grid + Patterns (shared, backend-agnostic)
```

Loop: `engine->step()` -> `engine->download(grid)` (no-op on CPU) -> `renderer->render(grid)` -> repeat.

Interfaces:

- `ISimEngine`: `upload(const Grid&)`, `step()`, `download(Grid&)`, `lastKernelMillis() const`,
  `name() const`, virtual dtor. CUDA/OpenCL engines own their device buffers and do the double-buffer
  ping-pong internally (read A -> write B -> swap).
- `IRenderer`: `render(const Grid&, uint64_t gen)`, `shouldClose() const = false`. `NullRenderer` does
  nothing — **always benchmark against it** so render/print cost never pollutes cells/sec.

Grid: a flat, row-major `std::vector<unsigned char>` indexed `r * cols_ + c`. It exposes dims, an
`at(x,y)` accessor, and the raw pointer; engines hold their own ping-pong buffers. Keep this layout
identical across backends so one host-side harness and the equivalence test can drive all three.

Rule sharing (minimise duplication): write the variant rule **once** in `LifeRules.hpp` as an inline
neighbour-count + birth/survive helper, qualified `__host__ __device__` under `#ifdef __CUDACC__` so
CPU and CUDA share it verbatim. OpenCL can't include C++ headers, so it mirrors the same helper in a
plain-C `kernel.cl` that is **read at runtime** (not embedded as a string literal); keep it diff-able
against `LifeRules.hpp`. Don't chase zero duplication between CUDA and OpenCL — it costs more clarity
than it saves. Edge handling (bounded vs toroidal) must also be identical across backends — pick one
and document it here.

CMake: a core library (always), a CPU engine (always), CUDA/OpenCL engine libraries gated on
`BUILD_CUDA` / `BUILD_OPENCL`, and renderer sources. The project must build and run (CPU + text) with
no CUDA toolkit and no OpenCL present.

Suggested layout:

```
include/gol/   Grid.hpp ISimEngine.hpp IRenderer.hpp Config.hpp LifeRules.hpp Timer.hpp
               patterns/Pattern.hpp patterns/RleLoader.hpp
src/core/      Grid.cpp Config.cpp main.cpp
src/engines/   cpu/CpuEngine.cpp  cuda/CudaEngine.cu cuda/kernel.cu  opencl/OpenCLEngine.cpp opencl/kernel.cl
src/render/    NullRenderer.cpp TextRenderer.cpp
tests/         rules_test.cpp equivalence_test.cpp
analysis/      *.ipynb results/*.csv
data/          *.rle
```

Per the report plan (`report/main.tex`), the two kernel-config variations to benchmark are block sizes
{32, 64, 128, 256} and shared/local memory vs none. Both should be runtime-selectable on the GPU
engines (e.g. `--block`, `--shared`) so the same binary drives every benchmark configuration.

## Timing & profiling

Headline metric: **cells evaluated per second** = `rows * cols * generations / elapsed`. Always
benchmark with the `NullRenderer` so output cost doesn't pollute the numbers.

Use uniform in-program event timing so all three backends are measured the same way (don't rely on a
profiler for the comparison numbers), exposed via `ISimEngine::lastKernelMillis()`. Report kernel time
separately from host<->device transfers:

- CPU: `std::chrono`.
- CUDA: `cudaEvent_t` around the kernel launch.
- OpenCL: command-queue events with `CL_QUEUE_PROFILING_ENABLE`.

Deep-dive on the *best* parallel solution only:

- **Nsight Compute (`ncu`)** — CUDA kernel-internal metrics (occupancy, stalls, memory throughput).
  CUDA/OptiX only; it will **not** profile OpenCL.
- **Nsight Systems (`nsys`)** — system/timeline view; can also trace OpenCL on NVIDIA hardware.

Analysis to keep in mind: this is a memory-bound, arithmetically trivial 9-point stencil over 1-byte
cells. Block size should give a concave curve plateauing by ~128-256; shared memory may give little or
even negative gain because the 1-byte stencil already lives in cache — the *why* is the interesting
part of the report. None of the optimizations help at small grids (launch/transfer overhead dominates),
so run the sweeps at large N x M.

## Tests

Planned, in dependency order:

1. **CPU `CpuEngine::step()` correctness** on known patterns under the variant rule (block stays still,
   blinker oscillates, and at least one case exercising the birth-on-6 rule).
2. **Cross-backend equivalence**: seed CPU, CUDA, and OpenCL identically, run N generations, assert the
   grids are bit-for-bit equal. `CpuEngine` is the reference oracle, so implement and verify it first —
   every later backend is checked against it.

## Report

LaTeX sources are in `report/`, built from a vendored template (`report/src/`, `report/template.tex`).
Edit `report/main.tex` — it has the full section skeleton with the grading rubric noted in comments.

```bash
cd report && latexmk -pdf main.tex
```

The assignment statement is `Tarea_2_Enunciado.pdf` (in Spanish) — the source of truth for requirements.
