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

} // namespace gol
