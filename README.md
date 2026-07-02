# GPU computation - Assignments 2 & 3

Conway's Game of Life on CPU, CUDA and OpenCL, benchmarked by cells evaluated
per second to report the GPU speed-up over the CPU (Tarea 2), plus a real-time
Qt + OpenGL viewer that displays the GPU-computed board through CUDA↔OpenGL
interop with no host round trip (Tarea 3).

The variant used here adds a third rule to the original game: a dead cell
becomes alive if it has **exactly 6** live neighbours (in addition to the
classic birth on exactly 3); a live cell survives on 2 or 3. This is the named
rule **HighLife** (`B36/S23`), and it must hold identically in every backend.

## Status

* **CPU engine** — implemented. One code path serves both the sequential
  baseline (1 core) and the data-parallel mode (N cores); they produce
  bit-for-bit identical results.
* **CUDA engine** — implemented. One thread per cell with device-side
  ping-pong and two runtime-selectable kernels (plain global memory and a
  shared-memory tiled variant), verified bit-for-bit against the CPU oracle.
  Opt in with `-DBUILD_CUDA=ON`.
* **OpenCL engine** — implemented. Mirrors the CUDA design (one work-item per
  cell, device-side ping-pong, two runtime-selectable kernels: plain global
  memory and a local-memory tiled variant via `--shared`). The kernel source
  `kernel.cl` is read at runtime and verified bit-for-bit against the CPU oracle
  (`--engine opencl --verify`). Opt in with `-DBUILD_OPENCL=ON`.
* **Qt + OpenGL viewer (`gol_gui`)** — a real-time viewer for the same
  simulation (Tarea 3, "Interop – Shaders"). It uses **CUDA↔OpenGL interop** to
  display the GPU-computed board with no host round trip, and falls back to a
  host-upload path for the CPU engine (so it builds with no CUDA toolkit). Opt in
  with `-DBUILD_GUI=ON`; see **Viewer** below.

All three backends run and verify against the CPU oracle. The CPU/CUDA/OpenCL
benchmark sweeps and the Nsight (`ncu`/`nsys`) profiling are done (Nsight Compute
is CUDA-only); see **Benchmarking** and **Report** below.

## Requirements

* C++ 23 compiler.
* CMake >= 3.28.
* Will use a local googletest install if available; otherwise it is fetched
  and built automatically.
* CUDA toolkit (only for the CUDA target).
* OpenCL headers and runtime (only for the OpenCL target).

## Build

```bash
cmake -S . -B build
cmake --build build
```

To opt in to the GPU targets (each is independent; build either or both):

```bash
cmake -S . -B build -DBUILD_CUDA=ON -DBUILD_OPENCL=ON
cmake --build build
```

## Run

```bash
./build/gol [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-r, --rows N` | 256 | grid height |
| `-c, --cols N` | 256 | grid width |
| `-g, --gens N` | 100 | generations to simulate |
| `-t, --threads N` | 1 | CPU cores: 1 = sequential, N = parallel, 0 = all hardware cores |
| `--wrap` | off | toroidal edges (default: bounded, out-of-range neighbours count as dead) |
| `--seed N` | 1 | RNG seed for the random fill |
| `--rle PATH` | — | seed from an RLE pattern (centred) instead of a random fill |
| `--renderer null\|text\|ansi` | null | `null` for benchmarking, `text` for a scrolling ASCII dump, `ansi` for an in-place animation (interactive on a TTY) |
| `--engine cpu\|cuda\|opencl` | cpu | simulation backend (all three implemented; `cuda`/`opencl` need their target built in) |
| `--verify` | off | run the selected backend and the sequential CPU on the same seed and assert bit-for-bit equality |
| `--block N`, `--shared` | 256, off | GPU kernel knobs (block size and shared-memory tiling), swept in the report (ignored by the CPU engine) |
| `--csv`, `--csv-header` | — | emit one CSV data row / the header line, for benchmark sweep scripts |

The headline metric is **cells evaluated per second** = `rows * cols * gens /
time`. Always benchmark against the `null` renderer so output cost never
pollutes the number; the binary reports both kernel and wall throughput.

```bash
# Benchmark: 1024x1024, 50 generations, sequential vs. all cores
./build/gol -r 1024 -c 1024 -g 50 -t 1
./build/gol -r 1024 -c 1024 -g 50 -t 0

# Watch a pattern evolve (generation 0 is the seed itself)
./build/gol -r 22 -c 22 -g 12 --rle patterns/highlife_replicator.rle --renderer text
```

## Viewer (Qt + OpenGL)

`gol_gui` is a real-time viewer for the same simulation. It drives any engine and
draws the board with an OpenGL fragment shader. With a CUDA engine on an NVIDIA GL
context it uses **CUDA↔OpenGL interop** so the GPU-computed board is displayed
without a host round trip (zero-copy); the CPU engine — and any context where
interop is unavailable — falls back to a host-upload path, so the viewer builds and
runs **with no CUDA toolkit**.

Build it with `-DBUILD_GUI=ON` (needs Qt6: Widgets, OpenGL, OpenGLWidgets):

```bash
cmake -S . -B build -DBUILD_CUDA=ON -DBUILD_GUI=ON
cmake --build build --target gol_gui
./scripts/run_gui.sh 1024x1024 --rle patterns/highlife_c98_gun.rle  # NVIDIA GL context (zero-copy)
./build/gol_gui --engine cpu 512x512                                # CPU engine, runs anywhere
```

`scripts/run_gui.sh` sets PRIME-offload env vars so the GL context lands on the
NVIDIA GPU (this dev box is an Optimus laptop, where GL otherwise defaults to the
Intel iGPU and interop is unavailable). The CLI mirrors the headless flags
(`--rows/--cols/--gens/--threads/--wrap/--seed/--rle/--engine cpu|cuda/--block`,
plus a `COLSxROWS` shorthand); `--gens` becomes an interactive auto-pause.

* **Display** — three colour modes (binary, live-neighbour count, and an *age
  heatmap* that colours a live cell by how many generations it has survived)
  through a selectable palette (**grayscale** by default, plus phosphor, amber,
  magma, ice). Grid lines appear around cells once zoomed in past ~4 px/cell.
* **Controls** — wheel zoom, middle/right-drag pan, left-drag paint (Shift
  erases). Keys: `space` play/pause, `S` step, `R` reseed, `C` clear, `I` reset to
  generation 0, `F` fit view, `P` save a PNG screenshot of the view. A dockable
  panel adds play/step/reset/reseed/clear, **Open RLE…**, Screenshot, and speed,
  colour-mode, palette, and toroidal-edges controls.

## Benchmarking

For parameter sweeps the binary has a CSV mode (`--csv` / `--csv-header`).
`scripts/sweep.sh` drives the CPU-thread and CUDA/OpenCL block/shared sweeps into
`results/*.csv` (the Linux re-run used for the report lives in `results/linux/`),
and `analysis/results.ipynb` turns them into the report's figures:

```bash
./scripts/sweep.sh        # -> results/sweep_gpu_opt.csv, results/sweep_scaling.csv
```

## Patterns

RLE seed patterns live in `patterns/`:

| File | What it shows |
| --- | --- |
| `block.rle` | 2x2 still life — survival rule, board never changes |
| `blinker.rle` | period-2 oscillator |
| `birth_on_six.rle` | a dead cell with exactly 6 live neighbours is born — the variant-only rule |
| `glider.rle` | classic 5-cell diagonal spaceship (needs only birth-on-3, so it behaves identically here) |
| `acorn.rle` | 7-cell methuselah — chaotic growth that peaks near gen 200 and settles by ~gen 600 |
| `r_pentomino.rle` | Conway's famous 1103-gen methuselah, but under B36/S23 it dies out in ~7 gens (a rule-difference demo) |
| `highlife_replicator.rle` | minimal 12-cell HighLife replicator (copies itself every 12 generations) |
| `highlife_spaceship.rle` | a c/18 HighLife spaceship built from replicators (period 72) |
| `highlife_c98_gun.rle` | a working HighLife glider gun firing c/98 orthogonal spaceships (583x546) |

The `highlife_*` patterns come from Golly's library, and those plus `glider`,
`acorn`, and `r_pentomino` were cross-checked against Golly's reference engine
(`bgolly`), which runs our exact rule (`B36/S23`) — so they double as correctness
oracles for the variant rule (the first three are simple fixtures the unit tests
cover). Note the Gosper glider gun is *not* included — birth-on-6 breaks it under
this rule, so `highlife_c98_gun.rle` is the gun that actually works.

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Report

LaTeX sources live under `report/`: `main.tex` is the graded assignment report
(Spanish, with results and the Nsight profiling) and `manual.tex` is the
developer manual (English). Build either with `latexmk -pdf <file>.tex` from
that directory.
