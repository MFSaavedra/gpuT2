#pragma once

#include <string>

/**
 * @file HybridNode.hpp
 * @brief Description of one node (compute device) in the hybrid engine's ordered
 *        row-wise decomposition. Kept dependency-free (in gol_core) so both the CLI
 *        parser (Config) and the hybrid engine can share the type.
 *
 * The hybrid engine splits the board into a top-to-bottom ordered list of nodes;
 * each node owns a contiguous band of rows sized by Divisible Load Theory. A node
 * is the host CPU pool, the discrete GPU (CUDA or OpenCL), or the integrated GPU
 * (OpenCL). The order determines adjacency for the one-row ghost halo exchange.
 */

namespace gol {

/// @brief Which compute device backs a hybrid node.
enum class NodeKind {
  Cpu,    ///< Host CPU worker pool.
  Cuda,   ///< A CUDA halo partition (the discrete NVIDIA GPU).
  OpenCL  ///< An OpenCL halo partition (device chosen by @ref HybridNode::deviceHint).
};

/// @brief One node in the hybrid decomposition (kind + how to pick its device).
struct HybridNode {
  NodeKind kind = NodeKind::Cpu;
  /// OpenCL device name/vendor substring ("intel", "nvidia", or "" = first GPU);
  /// unused for Cpu/Cuda. The GOL_OCL_DEVICE env var still overrides it.
  std::string deviceHint;
  /// Human label for reports/output ("cpu", "igpu", "dgpu", "dgpu-ocl").
  std::string label;
};

} // namespace gol
