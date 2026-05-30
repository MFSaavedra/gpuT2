#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace gol {

// Backend-agnostic state of the board: a flat, row-major buffer of 1-byte cells
// indexed `y * cols + x` (0 = dead, 1 = alive). This exact layout is shared by
// every backend so a single host-side harness and the equivalence test can drive
// the CPU, CUDA and OpenCL engines interchangeably. The Grid owns no device
// memory and contains no device code; engines hold their own ping-pong buffers
// and only read/write this through data() during upload()/download().
class Grid {
public:
  Grid(std::size_t rows, std::size_t cols, unsigned char fillValue = 0)
      : rows_(rows), cols_(cols), cells_(rows * cols, fillValue) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  std::size_t size() const { return cells_.size(); }

  // Cell access by (x, y) = (column, row). Unchecked: the simulation hot path
  // must stay branch-free, and bounds are the caller's responsibility.
  unsigned char& at(std::size_t x, std::size_t y) { return cells_[y * cols_ + x]; }
  unsigned char at(std::size_t x, std::size_t y) const { return cells_[y * cols_ + x]; }

  // Raw contiguous buffer, for host<->device transfers.
  unsigned char* data() { return cells_.data(); }
  const unsigned char* data() const { return cells_.data(); }

  // Set every cell to `value`.
  void fill(unsigned char value) {
    std::fill(cells_.begin(), cells_.end(), value);
  }

  // Deterministic random seeding: same seed => same board, so all three backends
  // can start from an identical state in the equivalence test.
  void randomize(std::uint64_t seed, double aliveProbability = 0.3) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution alive(aliveProbability);
    for (auto& c : cells_) c = alive(rng) ? 1u : 0u;
  }

  // Bit-for-bit comparison, used as the equivalence-test oracle check.
  bool operator==(const Grid& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ && cells_ == other.cells_;
  }
  bool operator!=(const Grid& other) const { return !(*this == other); }

private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<unsigned char> cells_; // row-major, size rows_ * cols_
};

} // namespace gol
