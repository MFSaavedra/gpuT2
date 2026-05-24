# GPU computation - Assignment 2

Conway's Game of Life on CPU, CUDA and OpenCL.

The variant used here adds a third rule to the original game: a dead cell
becomes alive if it has **exactly 6** live neighbours (in addition to the
classic birth on exactly 3).

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

To opt in to GPU targets:

```bash
cmake -S . -B build -DBUILD_CUDA=ON -DBUILD_OPENCL=ON
cmake --build build
```

## Run tests

```bash
ctest --test-dir build --output-on-failure
```

## Report

LaTeX sources live under `report/`. Build with `latexmk -pdf main.tex` from
that directory.
