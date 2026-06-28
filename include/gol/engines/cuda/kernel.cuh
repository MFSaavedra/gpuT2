#pragma once

#include <cstddef>

/**
 * @file kernel.cuh
 * @brief Host-side launcher for the CUDA Game of Life step kernels.
 *
 * Plain C++ signature (no CUDA types) so it can be declared here and called from
 * CudaEngine.cu; the definition and the device kernels live in kernel.cu.
 */

namespace gol {

/**
 * @brief Launch one Game of Life generation on the device (read @p dSrc, write @p dDst).
 *
 * Does not synchronise; the caller brackets the launch with cudaEvent timing.
 * @param dSrc      Device source (current) buffer.
 * @param dDst      Device destination (next) buffer.
 * @param rows      Board height.
 * @param cols      Board width.
 * @param wrap      false = bounded edges, true = toroidal.
 * @param blockSize Threads per block; mapped to a 32 x ceil(blockSize/32) tile.
 * @param useShared true = shared-memory tiled kernel, false = global-memory kernel.
 * @throws std::runtime_error if the kernel launch fails.
 */
void launchLifeStep(const unsigned char* dSrc, unsigned char* dDst,
                    std::size_t rows, std::size_t cols, bool wrap,
                    int blockSize, bool useShared);

/**
 * @brief Launch one generation of the HALO (ghost-row) kernel for a hybrid
 *        partition (read @p dSrc, write @p dDst), without synchronising.
 *
 * Operates on a buffer of height `realRows + 2`: buffer row 0 is the top ghost,
 * rows [1, realRows] are the real rows, and row `realRows + 1` is the bottom
 * ghost. Only the real rows are written. Vertical neighbours are read straight
 * from the ghost rows (no vertical edge logic — the host fills the ghosts to
 * realise bounded or toroidal edges); the horizontal (column) edge rule is
 * applied per @p wrap. Used by HybridEngine for its GPU share (Barlas chapter
 * 11.3 row-wise domain decomposition).
 * @param dSrc     Device source buffer of `(realRows + 2) * cols` bytes.
 * @param dDst     Device destination buffer of the same size.
 * @param realRows Number of real rows owned by the partition.
 * @param cols     Board width.
 * @param wrap     false = bounded columns, true = toroidal columns.
 * @param blockSize Threads per block; mapped to a 32 x ceil(blockSize/32) tile.
 * @throws std::runtime_error if the kernel launch fails.
 */
void launchLifeHaloStep(const unsigned char* dSrc, unsigned char* dDst,
                        std::size_t realRows, std::size_t cols, bool wrap,
                        int blockSize);

} // namespace gol
