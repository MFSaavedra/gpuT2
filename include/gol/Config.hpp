#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

/**
 * @file Config.hpp
 * @brief Fully-resolved run configuration and the argv parser that builds it.
 */

namespace gol {

/**
 * @brief Which simulation backend to instantiate.
 *
 * The strategy is chosen at runtime so one binary drives every benchmark
 * configuration.
 */
enum class EngineKind {
  Cpu,    ///< CpuEngine (sequential or data-parallel).
  Cuda,   ///< CudaEngine (not yet implemented).
  OpenCL  ///< OpenCLEngine (not yet implemented).
};

/**
 * @brief Which output strategy to instantiate.
 *
 * Always benchmark against Null so render cost never pollutes the cells/sec
 * headline metric.
 */
enum class RendererKind {
  Null, ///< NullRenderer: no output (use for benchmarking).
  Text, ///< TextRenderer: ASCII dump to stdout (one appended frame per step).
  Ansi  ///< AnsiRenderer: in-place ANSI animation (clips to the terminal viewport).
};

/**
 * @brief Fully resolved run configuration.
 *
 * Populated by parse() from argv (or built by hand in tests). Backend-agnostic:
 * GPU-only knobs (blockSize, useShared) are simply ignored by CpuEngine.
 */
struct Config {
  std::size_t rows = 256;          ///< Grid height.
  std::size_t cols = 256;          ///< Grid width.
  std::uint64_t generations = 100; ///< Number of steps to simulate.

  EngineKind engine = EngineKind::Cpu;        ///< Backend to run.
  RendererKind renderer = RendererKind::Null; ///< Output strategy.

  /**
   * @brief CPU engine worker count.
   *
   * 1 = sequential baseline; N = data-parallel over row blocks; 0 = use all
   * hardware cores. Sequential and parallel run the identical per-cell rule --
   * more cores only partition the rows, never change the result (see CpuEngine).
   */
  unsigned threads = 1;

  int blockSize = 256;    ///< GPU threads per block ({32,64,128,256}); mapped onto a 2D tile.
  bool useShared = false; ///< Use shared/local-memory tiling on the GPU engines.

  /**
   * @brief Edge handling.
   *
   * false = bounded (out-of-bounds neighbours count as dead), true = toroidal
   * wrap. Must be identical across all three backends.
   */
  bool wrap = false;

  std::uint64_t seed = 1; ///< Seed for Grid::randomize when no pattern is loaded.

  std::optional<std::string> rlePath; ///< Optional RLE pattern to stamp instead of a random seed.

  bool csv = false;       ///< Emit one CSV data row instead of the human-readable summary.
  bool csvHeader = false; ///< Print the CSV header line and exit (for sweep scripts).
};

/**
 * @brief Parse argv into a Config.
 * @param argc Argument count, as received by main().
 * @param argv Argument vector, as received by main().
 * @return The resolved configuration. Exits the process on `--help` or a usage error.
 */
Config parse(int argc, char** argv);

} // namespace gol
