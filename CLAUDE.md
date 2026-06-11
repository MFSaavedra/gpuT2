# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University assignment (Tarea 2, course CC7515-1 "Computación en GPU", U. de Chile): implement
Conway's Game of Life three times — sequential **CPU**, **CUDA**, and **OpenCL** — then benchmark
cells-evaluated-per-second across grid sizes and report the GPU speed-up vs CPU.

**Rule variant (do not use vanilla Conway rules):** a dead cell is born on **exactly 3 OR exactly 6**
live neighbours; a live cell survives on 2 or 3. The extra "birth on 6" rule is the one detail that's
easy to get wrong — it must hold in every implementation (CPU, CUDA, OpenCL). This is the named rule
**HighLife** (`B36/S23`), which is convenient: Golly and other tools simulate it directly, so they can
cross-check our backends (see Tests).

## Current state

The engine/renderer **strategy** layout below is in place. What exists and works:

- **Core** — backend-agnostic `Grid` (header-only), `Config` + CLI parser, `LifeRules.hpp` (the shared
  variant rule), the `Pattern` / `RleLoader` data pipeline, and the `ISimEngine` / `IRenderer` interfaces.
- **CpuEngine** (`ISimEngine`) — implemented. A single code path serves both the **sequential** baseline
  (`threads == 1`, no synchronisation) and the **data-parallel** mode (`threads >= 2`, a persistent worker
  pool synchronised with `std::barrier`; `threads == 0` = all hardware cores). Both only partition the
  rows, so they produce bit-for-bit identical boards.
- **Renderers** — `NullRenderer` (header-only, for benchmarking), `TextRenderer` (scrolling text
  dump), and `AnsiRenderer` (in-place ANSI animation; clips big grids to the terminal viewport). Both
  visible renderers draw each live cell as **two** full blocks `██` (U+2588) and dead as two spaces, so
  cells read as squares despite the ~1:2 terminal cell aspect ratio; glyph fields are `std::string`
  (configurable). `AnsiRenderer` clips by per-cell display width (2 cols/cell), not raw column count.
  On a TTY it is **interactive**: arrow keys pan the viewport over a grid larger than the screen,
  `space`/`p` pauses (the sim halts but you can still pan), and `q` quits (via `shouldClose()`).
  Scrollbars (`┃`/`━` thumbs along the right/bottom edges) show the scroll position when clipped, and
  the viewer **holds on the final frame until `q`** (via `staysOpen()`). It uses raw, non-blocking
  input and restores the terminal on exit and on SIGINT/SIGTERM. Pause is contained in `render()` (a
  `do/while`), so the main loop/engine/interface are untouched; piped (non-TTY), the controls are
  disabled, `staysOpen()` is false, and the top-left window is shown once.
- **App** — `main.cpp` owns the loop, wires a CPU engine + renderer, seeds from RLE or a deterministic
  random fill, and prints kernel/wall cells-per-second.

- **CudaEngine** (`ISimEngine`) — implemented. One thread per cell over a 2D grid of blocks; owns two
  device buffers and ping-pongs on the device. Two runtime-selectable kernels: plain global-memory and
  a shared-memory tiled variant with a 1-cell halo (`--shared`); block size (`--block`) maps to a
  32-wide tile for coalesced loads. Kernel timed with `cudaEvent_t` (excludes transfers); reuses
  `nextState` from `LifeRules.hpp` verbatim. Verified **bit-for-bit against the `CpuEngine` oracle**
  across block sizes, shared/global, and both edge modes (`cuda_equivalence_test.cpp`). Opt-in via
  `-DBUILD_CUDA=ON`; targets `sm_75` (GTX 1660 Ti) with `-lineinfo` for `ncu`.

- **OpenCLEngine** (`ISimEngine`) — implemented. Mirrors the CUDA design: one work-item per cell over a
  2D NDRange, two device buffers ping-ponged on the device, and two runtime-selectable kernels — plain
  global-memory `life_global` and a local-memory tiled variant with a 1-cell halo (`life_local`, via
  `--shared`); the work-group maps to a 32-wide local size, like CUDA's tile. The kernel source
  `kernel.cl` is **read at runtime** (`clCreateProgramWithSource`) and mirrors `nextState` + the edge
  rule in plain C; kernel timed with `CL_QUEUE_PROFILING_ENABLE` command-queue events (excludes
  transfers). Verified **bit-for-bit against the `CpuEngine` oracle** at runtime via
  `--engine opencl --verify` (no dedicated gtest yet — see Tests). Opt-in via `-DBUILD_OPENCL=ON`; on an
  NVIDIA box the platform is "NVIDIA CUDA", so it lowers through the same PTX/SASS backend as CUDA.

- **Benchmarks** — run. `scripts/sweep.sh` drives the CPU(threads) + CUDA(block×shared) sweeps to
  `results/sweep_*.csv`; `analysis/results.ipynb` (Spanish) plots them to `report/img/bench_*.png`.
  Headline: CUDA peaks ~32 Gcells/s, ~200× over the best parallel CPU; **shared memory is slower**
  (~0.55×) and block 64–128 is optimal. `ncu` corrected the roofline: the kernel is **not** DRAM-bound —
  L2 serves ~88% of requests so DRAM sees ~2 B/cell at ~20% util; it's latency/L2-pipe-bound (~65% SOL).
  `nsys`: ~30 µs/step launch overhead, H2D 8.4 GB/s (pageable `std::vector`). Report Resultados/Análisis,
  §Cotas, and §Perfilamiento are filled from this (`results/profiling/*.ncu-rep`).

All three backends now run and verify against the `CpuEngine` reference oracle. GPU CMake targets are
opt-in and double-gated (the option **and** the source must exist); the project still builds and runs
(CPU + text) with neither toolkit present. The chosen optimization options to benchmark are **#1 block
size** and **#3 shared/local memory** (the `--block` / `--shared` knobs); option #2 (2D vs 1D arrays)
is not used — the `Grid` is flat 1-D.

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
  |-- IRenderer    <- NullRenderer | TextRenderer | AnsiRenderer | (GuiRenderer later)
  '-- Grid + Patterns (shared, backend-agnostic)
```

Loop: render the seed as generation 0, then `engine->step()` -> (only if a renderer is attached)
`engine->download(grid)` -> `renderer->render(grid, gen+1)` -> repeat. Under `NullRenderer` the
download + render are skipped entirely, so host<->device transfers never enter the benchmark.

Interfaces:

- `ISimEngine`: `upload(const Grid&)`, `step()`, `download(Grid&)`, `lastKernelMillis() const`,
  `name() const`, virtual dtor. CUDA/OpenCL engines own their device buffers and do the double-buffer
  ping-pong internally (read A -> write B -> swap).
- `IRenderer`: `render(const Grid&, uint64_t gen)`, `shouldClose()`/`staysOpen()` (both default false).
  `NullRenderer` does
  nothing — **always benchmark against it** so render/print cost never pollutes cells/sec. `TextRenderer`
  appends each frame (scrolls); `AnsiRenderer` redraws in place via ANSI escapes and clips to the
  terminal viewport (it never wraps, since a wrapped row would desync its cursor-home math).

Grid: a flat, row-major `std::vector<unsigned char>` indexed `r * cols_ + c`. It exposes dims, an
`at(x,y)` accessor, and the raw pointer; engines hold their own ping-pong buffers. Keep this layout
identical across backends so one host-side harness and the equivalence test can drive all three.

Rule sharing (minimise duplication): write the variant rule **once** in `LifeRules.hpp` as an inline
neighbour-count + birth/survive helper, qualified `__host__ __device__` under `#ifdef __CUDACC__` so
CPU and CUDA share it verbatim. OpenCL can't include C++ headers, so it mirrors the same helper in a
plain-C `kernel.cl` that is **read at runtime** (not embedded as a string literal); keep it diff-able
against `LifeRules.hpp`. Don't chase zero duplication between CUDA and OpenCL — it costs more clarity
than it saves. Edge handling (bounded vs toroidal) must also be identical across backends. **Default is
bounded** (out-of-range neighbours count as dead); `--wrap` selects toroidal. `CpuEngine` implements
both, the seq-vs-parallel test covers each mode, and the GPU backends must match it bit-for-bit in both.

CMake: a core library (always), a CPU engine (always), CUDA/OpenCL engine libraries gated on
`BUILD_CUDA` / `BUILD_OPENCL`, and renderer sources. The project must build and run (CPU + text) with
no CUDA toolkit and no OpenCL present.

Layout (parenthesised notes are clarifications — header-only units, directory contents):

```
include/gol/   Grid.hpp ISimEngine.hpp IRenderer.hpp Config.hpp LifeRules.hpp Timer.hpp
               engines/CpuEngine.hpp  render/NullRenderer.hpp render/TextRenderer.hpp render/AnsiRenderer.hpp
               patterns/Pattern.hpp patterns/RleLoader.hpp
src/core/      Config.cpp main.cpp                         (Grid/LifeRules/Timer are header-only)
src/engines/   cpu/CpuEngine.cpp  cuda/CudaEngine.cu cuda/kernel.cu  opencl/OpenCLEngine.cpp opencl/kernel.cl
include/gol/engines/  CudaEngine.hpp  cuda/kernel.cuh  OpenCLEngine.hpp
src/render/    TextRenderer.cpp AnsiRenderer.cpp           (NullRenderer is header-only)
src/patterns/  Pattern.cpp RleLoader.cpp
tests/         compile_smoke_test.cpp rules_test.cpp rle_loader_test.cpp cpu_parallel_test.cpp cuda_equivalence_test.cpp
patterns/      *.rle                                       (block, blinker, birth_on_six, highlife_replicator, highlife_spaceship)
scripts/       sweep.sh                                    (CPU+CUDA benchmark sweeps -> results/*.csv)
results/       sweep_cuda_opt.csv sweep_scaling.csv         (Pandas-ready sweep data)
analysis/      results.ipynb                               (loads results/, writes report/img/bench_*.png)
```

Per the report plan (`report/main.tex`), the two kernel-config variations to benchmark are block sizes
{32, 64, 128, 256} and shared/local memory vs none. Both are runtime-selectable on the GPU
engines (e.g. `--block`, `--shared`) so the same binary drives every benchmark configuration.

## Timing & profiling

Headline metric: **cells evaluated per second** = `rows * cols * generations / elapsed`. Always
benchmark with the `NullRenderer` so output cost doesn't pollute the numbers.

For sweep data, `--csv` prints one data row per run (columns:
`backend,rows,cols,generations,threads,block,shared,wrap,kernel_ms,wall_ms,mcells_kernel,mcells_wall`)
and `--csv-header` prints just the header line and exits — so a script does
`gol --csv-header > out.csv` once, then appends `gol … --csv >> out.csv` per config (Pandas-ready).

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

Analysis (now measured + profiled): this is a memory-bound, arithmetically trivial 9-point stencil over
1-byte cells. Measured: block size is concave, peaking at **64-128**; shared memory **hurts** (~0.55x)
because L2 already serves the reuse. `ncu` refined the picture — the global kernel is **not**
DRAM-bandwidth-bound (DRAM at ~20%, q≈2 B/cell via 88% L2 hit) but **latency / L2-pipe-bound** (~65% SOL).
Launch/transfer overhead dominates only at small grids, so run the sweeps at large N x M.

## Tests

In dependency order (`CpuEngine` is the oracle, so it is verified first):

1. **CpuEngine correctness** under the variant rule — done. Block stays still, blinker oscillates
   period-2, and the birth-on-6 case brings a dead cell with 6 neighbours alive (`rules_test.cpp`).
2. **Sequential vs. parallel equivalence** — done. Same seed + generations must give a bit-for-bit
   identical board across thread counts and both edge modes, including non-divisible grids that exercise
   the row-remainder logic (`cpu_parallel_test.cpp`).
3. **RLE loader + `Pattern::applyTo`** — done. Parsing, stamping, clipping, missing-file errors
   (`rle_loader_test.cpp`). `compile_smoke_test.cpp` also includes every header so the scaffolding stays
   build-clean.
4. **Cross-backend equivalence** — CUDA done (gtest); OpenCL verified at runtime, gtest pending.
   `cuda_equivalence_test.cpp` seeds CUDA and CPU identically and asserts bit-for-bit equality against
   the `CpuEngine` oracle across block sizes {32,64,128,256}, shared/global kernels, and both edge modes
   (plus a birth-on-6 device check). Built only when `-DBUILD_CUDA=ON` (`gol_cuda_tests`); needs a GPU
   at run time. OpenCL is currently checked the same way only at runtime, via `--engine opencl --verify`
   (passes across sizes and both edge modes); a dedicated `gol_opencl_tests` following the CUDA pattern
   is still to be added.

**Golly as an external oracle.** Golly is installed locally, including the headless `bgolly` runner
(infinite grid, no edge effects). It runs our exact rule (`B36/S23`), so it is a second, independent
reference: `bgolly -m <gens> -i <step> file.rle` prints populations to compare against ours. The
`highlife_replicator.rle` / `highlife_spaceship.rle` fixtures in `patterns/` were verified this way and
reproduce `bgolly` bit-for-bit, so they make good large-grid cross-checks for the GPU backends.

## Report

LaTeX sources are in `report/`, built from a vendored template (`report/src/`, `report/template.tex`).
There are **two** standalone documents, both built the same way:

- `report/main.tex` — the **graded assignment report** (Spanish). Full section skeleton with the
  grading rubric noted in comments; holds experiments, results, speed-up analysis.
- `report/manual.tex` — the **developer manual** (English). Architecture, mechanism, and a
  line-by-line code walkthrough; the reference for how/why the code works.

```bash
cd report && latexmk -pdf main.tex      # or: latexmk -pdf manual.tex
```

**Architecture UML.** The class diagram source is `report/diagrams/architecture.mmd` (Mermaid),
rendered with mermaid-cli to `report/img/uml_architecture.{svg,pdf,png}` — the PDF is embedded in the
manual (Figure 1) and the PNG is for the assignment report. Regenerate after architecture changes:

```bash
# -c mermaid-config.json sets htmlLabels:false -> native SVG <text> (not <foreignObject>),
# so rsvg-convert renders text into the PDF; -p sets Chromium's --no-sandbox for this env.
mmdc -i report/diagrams/architecture.mmd -o report/img/uml_architecture.svg \
     -p report/diagrams/puppeteer.json -c report/diagrams/mermaid-config.json -b white -t neutral
rsvg-convert -f pdf -o report/img/uml_architecture.pdf report/img/uml_architecture.svg
mmdc -i report/diagrams/architecture.mmd -o report/img/uml_architecture.png \
     -p report/diagrams/puppeteer.json -c report/diagrams/mermaid-config.json -b white -t neutral -s 3
```

**CUDA index-mapping diagrams.** Three Graphviz sources in `report/diagrams/` explain how the 1-D
flat `Grid` maps onto the 2-D CUDA thread/block geometry (the "three coordinate spaces" subsection in
the manual and the CUDA implementation subsection in the report): `cuda_coordinate_spaces.dot`
(execution → board → memory + the two maps), `cuda_block_mapping.dot` (a block is a rectangle on the
board but two discontiguous runs in memory, `cols - bx` apart — scaled down to a 4×2 block to fit),
and `cuda_stencil_addressing.dot` (the 9-point stencil as memory offsets: ±1 horizontal, ±cols
vertical). Each renders to `report/img/<name>.{pdf,png}` (PDF embedded in both `.tex` via
`\insertimage`). Regenerate after changing the kernel's index math:

```bash
for d in cuda_coordinate_spaces cuda_block_mapping cuda_stencil_addressing; do
  dot -Tpdf report/diagrams/$d.dot -o report/img/$d.pdf
  dot -Tpng -Gdpi=200 report/diagrams/$d.dot -o report/img/$d.png
done
```

The assignment statement is `Tarea_2_Enunciado.pdf` (in Spanish) — the source of truth for requirements.

## Documentation maintenance

**IMPORTANT — keep the docs in sync with the code.** This repo treats three documents as living
artifacts: `report/manual.tex` (developer manual), `report/main.tex` (assignment report), and this
`CLAUDE.md`. Whenever you **add, remove, or change a source file in a way that alters behaviour,
structure, the file layout, the CLI/flag surface, the timing methodology, or the rule/edge handling,
update all three affected documents in the same change** — do not defer it. Specifically:

- **`report/manual.tex`** — update the relevant section and the file map (Table `tab:files`). New
  source files get a section + a row; changed mechanisms get their walkthrough/snippet updated;
  removed files get their entry deleted. Keep code snippets matching the real code (em-dashes inside
  `sourcecode` listings must be `--`; escape `_`, `$`, `#`, `%`, `&` in *captions* and prose; a raw
  `█`/Unicode glyph in LaTeX text breaks pdflatex — describe it as "U+2588" instead). If the class or
  relationship structure changes, update the UML source `report/diagrams/architecture.mmd` and
  re-render `report/img/uml_architecture.*` (commands under "Report").
- **`report/main.tex`** — update only when results, experiments, or methodology move (e.g. a GPU
  backend lands, a benchmark is run, a config variation changes). It is the graded report.
- **`CLAUDE.md`** — update "Current state", the layout tree, and any affected section (Build, Tests,
  Architecture, Timing) so this file never describes code that no longer exists.

The most common trigger is landing the CUDA or OpenCL backend: that touches the manual's rule,
CPU-oracle, tests, timing, and roadmap sections; the report's implementation + results sections; and
this file's "Current state" and layout. After editing, rebuild both PDFs (`latexmk -pdf`) to confirm
they still compile.
