# GPU computation - Assignment 2

Conway's Game of Life on CPU, CUDA and OpenCL, benchmarked by cells evaluated
per second to report the GPU speed-up over the CPU.

The variant used here adds a third rule to the original game: a dead cell
becomes alive if it has **exactly 6** live neighbours (in addition to the
classic birth on exactly 3); a live cell survives on 2 or 3. This is the named
rule **HighLife** (`B36/S23`), and it must hold identically in every backend.

## Status

* **CPU engine** — implemented. One code path serves both the sequential
  baseline (1 core) and the data-parallel mode (N cores); they produce
  bit-for-bit identical results.
* **CUDA / OpenCL engines** — planned. The CLI flags and CMake targets exist,
  but the device sources do not yet, so `--engine cuda|opencl` currently errors.

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

To opt in to GPU targets (once their sources exist):

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
| `--renderer null\|text` | null | `null` for benchmarking, `text` for an ASCII dump per generation |
| `--engine cpu\|cuda\|opencl` | cpu | simulation backend (only `cpu` works today) |
| `--block N`, `--shared` | 256, off | GPU kernel knobs, swept in the report (ignored by the CPU engine) |

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

## Patterns

RLE seed patterns live in `patterns/`:

| File | What it shows |
| --- | --- |
| `block.rle` | 2x2 still life — survival rule, board never changes |
| `blinker.rle` | period-2 oscillator |
| `birth_on_six.rle` | a dead cell with exactly 6 live neighbours is born — the variant-only rule |
| `highlife_replicator.rle` | minimal 12-cell HighLife replicator (copies itself every 12 generations) |
| `highlife_spaceship.rle` | a c/18 HighLife spaceship built from replicators (period 72) |

The two HighLife patterns were imported from Golly's library and verified
bit-for-bit against its reference engine (`bgolly`), so they double as
correctness oracles for the variant rule.

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Report

LaTeX sources live under `report/`. Build with `latexmk -pdf main.tex` from
that directory.
