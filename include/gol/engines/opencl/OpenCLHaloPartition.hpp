#pragma once

#include <memory>

#include "gol/engines/IHaloPartition.hpp"

/**
 * @file OpenCLHaloPartition.hpp
 * @brief Factory for the OpenCL implementation of IHaloPartition.
 *
 * Only the factory is exposed (no OpenCL headers/types leak out), so HybridEngine
 * can build a GPU partition without depending on OpenCL. The concrete class lives
 * in OpenCLHaloPartition.cpp inside the gated gol_opencl library.
 */

namespace gol {

/**
 * @brief Construct an OpenCL-backed halo partition.
 * @param blockSize Work-items per group ({32,64,128,256}); mapped onto a 32-wide local size.
 * @return Owning pointer to the partition as an IHaloPartition.
 */
std::unique_ptr<IHaloPartition> makeOpenCLHaloPartition(int blockSize);

} // namespace gol
