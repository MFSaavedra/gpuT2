/**
 * @file kernel.cu
 * @brief CUDA Game of Life step kernels (global-memory and shared-memory tiled)
 *        plus the host launcher.
 *
 * The per-cell rule is shared verbatim with the CPU backend through
 * LifeRules.hpp: under nvcc, nextState() is qualified `__host__ __device__`.
 * Edge handling matches CpuEngine::countNeighbors exactly (bounded: out-of-range
 * neighbours count as dead; toroidal: indices wrap).
 */

#include "gol/engines/cuda/kernel.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "gol/LifeRules.hpp" // nextState() -> __host__ __device__ under nvcc

namespace gol {

namespace {

// Plain global-memory kernel: one thread per cell, each reads its 8 neighbours
// directly from global memory.
__global__ void lifeGlobal(const unsigned char* __restrict__ src,
                           unsigned char* __restrict__ dst,
                           int rows, int cols, bool wrap) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= cols || row >= rows) return;

  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int nc = col + dx;
      int nr = row + dy;
      if (wrap) {
        nc = (nc + cols) % cols;
        nr = (nr + rows) % rows;
      } else if (nc < 0 || nr < 0 || nc >= cols || nr >= rows) {
        continue; // bounded: out-of-range neighbour is dead
      }
      n += src[static_cast<std::size_t>(nr) * cols + nc];
    }
  }
  const std::size_t idx = static_cast<std::size_t>(row) * cols + col;
  dst[idx] = nextState(src[idx], n);
}

// Shared-memory tiled kernel: the block cooperatively stages a (bx+2)x(by+2)
// tile (its cells plus a 1-cell halo) into shared memory, then each thread reads
// its 8 neighbours from the tile. The halo load applies the same edge rule, so
// the result is bit-for-bit identical to lifeGlobal / the CPU oracle.
__global__ void lifeShared(const unsigned char* __restrict__ src,
                           unsigned char* __restrict__ dst,
                           int rows, int cols, bool wrap) {
  extern __shared__ unsigned char tile[];
  const int bx = blockDim.x;
  const int by = blockDim.y;
  const int tileW = bx + 2;
  const int tileH = by + 2;

  // Cooperatively fill the tile (including halo). Each tile cell maps to a global
  // (gr, gc) offset by -1 for the halo; apply the edge rule per cell.
  for (int t = threadIdx.y * bx + threadIdx.x; t < tileW * tileH; t += bx * by) {
    const int tx = t % tileW;
    const int ty = t / tileW;
    int gc = blockIdx.x * bx + tx - 1;
    int gr = blockIdx.y * by + ty - 1;
    unsigned char v;
    if (wrap) {
      gc = (gc + cols) % cols;
      gr = (gr + rows) % rows;
      v = src[static_cast<std::size_t>(gr) * cols + gc];
    } else {
      v = (gc < 0 || gr < 0 || gc >= cols || gr >= rows)
              ? 0u
              : src[static_cast<std::size_t>(gr) * cols + gc];
    }
    tile[ty * tileW + tx] = v;
  }
  __syncthreads();

  const int col = blockIdx.x * bx + threadIdx.x;
  const int row = blockIdx.y * by + threadIdx.y;
  if (col >= cols || row >= rows) return;

  const int lx = threadIdx.x + 1;
  const int ly = threadIdx.y + 1;
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      n += tile[(ly + dy) * tileW + (lx + dx)];
    }
  }
  dst[static_cast<std::size_t>(row) * cols + col] = nextState(tile[ly * tileW + lx], n);
}

// Halo (ghost-row) kernel for a hybrid CPU+GPU partition. The buffer has height
// realRows+2: row 0 is the top ghost, rows [1, realRows] are the real rows, and
// row realRows+1 is the bottom ghost. Each thread computes one real row, reading
// its vertical neighbours straight from the buffer (ghost rows included -- no
// vertical edge logic) and applying the horizontal edge rule per `wrap`. The host
// fills the ghosts each generation, so the result is bit-for-bit identical to the
// CpuEngine oracle in both edge modes.
__global__ void lifeHalo(const unsigned char* __restrict__ src,
                         unsigned char* __restrict__ dst,
                         int realRows, int cols, bool wrap) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y; // real-row index in [0, realRows)
  if (col >= cols || r >= realRows) return;

  const int br = r + 1; // buffer row of this real row (row 0 is the top ghost)

  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      int nc = col + dx;
      if (wrap) {
        nc = (nc + cols) % cols;
      } else if (nc < 0 || nc >= cols) {
        continue; // bounded: out-of-range column is dead
      }
      const int nr = br + dy; // vertical neighbour is always a valid buffer row
      n += src[static_cast<std::size_t>(nr) * cols + nc];
    }
  }
  const std::size_t idx = static_cast<std::size_t>(br) * cols + col;
  dst[idx] = nextState(src[idx], n);
}

} // namespace

void launchLifeStep(const unsigned char* dSrc, unsigned char* dDst,
                    std::size_t rows, std::size_t cols, bool wrap,
                    int blockSize, bool useShared) {
  if (blockSize < 32) blockSize = 32;
  const int bx = 32;                              // 32-wide rows -> coalesced loads
  const int by = (blockSize + bx - 1) / bx;       // 32/64/128/256 -> 1/2/4/8
  const dim3 block(bx, by);
  const dim3 grid(static_cast<unsigned>((cols + bx - 1) / bx),
                  static_cast<unsigned>((rows + by - 1) / by));
  const int ri = static_cast<int>(rows);
  const int ci = static_cast<int>(cols);

  if (useShared) {
    const std::size_t shmem = static_cast<std::size_t>(bx + 2) * (by + 2);
    lifeShared<<<grid, block, shmem>>>(dSrc, dDst, ri, ci, wrap);
  } else {
    lifeGlobal<<<grid, block>>>(dSrc, dDst, ri, ci, wrap);
  }

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

void launchLifeHaloStep(const unsigned char* dSrc, unsigned char* dDst,
                        std::size_t realRows, std::size_t cols, bool wrap,
                        int blockSize) {
  if (realRows == 0) return; // partition owns no rows -> nothing to do
  if (blockSize < 32) blockSize = 32;
  const int bx = 32;                              // 32-wide rows -> coalesced loads
  const int by = (blockSize + bx - 1) / bx;       // 32/64/128/256 -> 1/2/4/8
  const dim3 block(bx, by);
  const dim3 grid(static_cast<unsigned>((cols + bx - 1) / bx),
                  static_cast<unsigned>((realRows + by - 1) / by));

  lifeHalo<<<grid, block>>>(dSrc, dDst, static_cast<int>(realRows),
                            static_cast<int>(cols), wrap);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA halo kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

} // namespace gol
