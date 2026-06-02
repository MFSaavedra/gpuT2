#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace gol {

// Which simulation backend to instantiate. The strategy is chosen at runtime so
// one binary drives every benchmark configuration.
enum class EngineKind { Cpu, Cuda, OpenCL };

// Which output strategy to instantiate. Always benchmark against Null so render
// cost never pollutes the cells/sec headline metric.
enum class RendererKind { Null, Text };

// Fully resolved run configuration. Populated by parse() from argv (or built by
// hand in tests). Backend-agnostic: GPU-only knobs (blockSize, useShared) are
// simply ignored by CpuEngine.
struct Config {
  std::size_t rows = 256;       // grid height
  std::size_t cols = 256;       // grid width
  std::uint64_t generations = 100; // number of steps to simulate

  EngineKind engine = EngineKind::Cpu;
  RendererKind renderer = RendererKind::Null;

  // CPU engine: number of worker cores. 1 = sequential baseline; N = data-parallel
  // over row blocks; 0 = use all hardware cores. Sequential and parallel run the
  // identical per-cell rule — more cores only partition the rows, never change the
  // result (see CpuEngine).
  unsigned threads = 1;

  // GPU kernel-config knobs swept in the report. blockSize is the total threads
  // per block ({32, 64, 128, 256}); the GPU engines map it onto a 2D tile.
  int blockSize = 256;
  bool useShared = false; // use shared/local memory tiling on the GPU engines

  // Edge handling: false = bounded (out-of-bounds neighbours count as dead),
  // true = toroidal wrap. Must be identical across all three backends.
  bool wrap = false;

  std::uint64_t seed = 1; // seed for Grid::randomize when no pattern is loaded

  // Optional RLE pattern to stamp instead of a random seed.
  std::optional<std::string> rlePath;
};

// Parse argv into a Config. Implemented in a later pass once the flag surface is
// agreed; declared here so the interface layer is complete and compilable.
Config parse(int argc, char** argv);

} // namespace gol
