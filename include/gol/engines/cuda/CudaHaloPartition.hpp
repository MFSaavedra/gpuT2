#pragma once

#include <memory>

#include "gol/engines/IHaloPartition.hpp"

/**
 * @file CudaHaloPartition.hpp
 * @brief Factory for the CUDA implementation of IHaloPartition.
 *
 * Only the factory is exposed (no CUDA headers/types leak out), so HybridEngine
 * can build a GPU partition without depending on the CUDA toolkit. The concrete
 * class lives in CudaHaloPartition.cu inside the gated gol_cuda library.
 */

namespace gol {

/**
 * @brief Construct a CUDA-backed halo partition.
 * @param blockSize Threads per block ({32,64,128,256}); mapped onto a 32-wide tile.
 * @return Owning pointer to the partition as an IHaloPartition.
 */
std::unique_ptr<IHaloPartition> makeCudaHaloPartition(int blockSize);

} // namespace gol
