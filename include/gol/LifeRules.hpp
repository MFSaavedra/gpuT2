#pragma once

// Single source of truth for the assignment's rule VARIANT, shared verbatim by
// the CPU and CUDA backends. Under nvcc the helper is qualified so it compiles
// for both host and device; in a plain C++ translation unit it is just inline.
// OpenCL cannot include this header, so kernel.cl mirrors this function in
// plain C and must be kept diff-able against it.
#ifdef __CUDACC__
#define GOL_FN __host__ __device__ inline
#else
#define GOL_FN inline
#endif

namespace gol {

// Next state of one cell given its current state and live-neighbour count.
//
// VARIANT (NOT vanilla Conway):
//   - a dead cell is born on EXACTLY 3 OR EXACTLY 6 live neighbours;
//   - a live cell survives on 2 or 3 live neighbours;
//   - everything else dies / stays dead.
GOL_FN unsigned char nextState(unsigned char alive, int neighbors) {
  if (alive) {
    return (neighbors == 2 || neighbors == 3) ? 1u : 0u; // survive
  }
  return (neighbors == 3 || neighbors == 6) ? 1u : 0u; // birth (3 or 6)
}

} // namespace gol
