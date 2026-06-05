#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * @file Grid.hpp
 * @brief Backend-agnostic board state: the flat, row-major cell buffer shared
 *        by every simulation backend.
 */

/// @namespace gol
/// @brief Root namespace for the Game of Life project (core, engines, renderers).
namespace gol {

/**
 * @brief Backend-agnostic state of the board.
 *
 * A flat, row-major buffer of 1-byte cells indexed `y * cols + x` (0 = dead,
 * 1 = alive). This exact layout is shared by every backend so a single
 * host-side harness and the equivalence test can drive the CPU, CUDA and
 * OpenCL engines interchangeably. The Grid owns no device memory and contains
 * no device code; engines hold their own ping-pong buffers and only read/write
 * this through data() during upload()/download().
 */
class Grid {
public:
  /**
   * @brief Construct a `rows` x `cols` board with every cell set to @p fillValue.
   * @param rows      Number of rows (board height).
   * @param cols      Number of columns (board width).
   * @param fillValue Initial value for every cell (0 = dead, 1 = alive).
   */
  Grid(std::size_t rows, std::size_t cols, unsigned char fillValue = 0)
      : rows_(rows), cols_(cols), cells_(rows * cols, fillValue) {}

  std::size_t rows() const { return rows_; } ///< @return Number of rows (board height).
  std::size_t cols() const { return cols_; } ///< @return Number of columns (board width).
  std::size_t size() const { return cells_.size(); } ///< @return Total cell count (rows * cols).

  /**
   * @brief Mutable cell access by (x, y) = (column, row).
   *
   * Unchecked: the simulation hot path must stay branch-free, and bounds are
   * the caller's responsibility.
   * @param x Column index, expected in [0, cols()).
   * @param y Row index, expected in [0, rows()).
   * @return Reference to the cell at (x, y).
   */
  unsigned char& at(std::size_t x, std::size_t y) { return cells_[y * cols_ + x]; }

  /**
   * @brief Read-only cell access by (x, y) = (column, row). Unchecked.
   * @param x Column index, expected in [0, cols()).
   * @param y Row index, expected in [0, rows()).
   * @return Value of the cell at (x, y).
   */
  unsigned char at(std::size_t x, std::size_t y) const { return cells_[y * cols_ + x]; }

  unsigned char* data() { return cells_.data(); } ///< @return Raw contiguous buffer, for host<->device transfers.
  const unsigned char* data() const { return cells_.data(); } ///< @return Read-only raw contiguous buffer.

  /**
   * @brief Set every cell to @p value.
   * @param value Value written to all cells (0 = dead, 1 = alive).
   */
  void fill(unsigned char value) {
    std::fill(cells_.begin(), cells_.end(), value);
  }

  /**
   * @brief Deterministic random seeding.
   *
   * The same @p seed produces the same board, so all three backends can start
   * from an identical state in the equivalence test.
   * @param seed             Seed for the PRNG; equal seeds yield equal boards.
   * @param aliveProbability Probability that any given cell starts alive.
   */
  void randomize(std::uint64_t seed, double aliveProbability = 0.3) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution alive(aliveProbability);
    for (auto& c : cells_) c = alive(rng) ? 1u : 0u;
  }

  /**
   * @brief Bit-for-bit comparison (dimensions and cell contents).
   *
   * Used as the equivalence-test oracle check.
   * @param other Grid to compare against.
   * @return true if both grids have identical dimensions and cells.
   */
  bool operator==(const Grid& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ && cells_ == other.cells_;
  }

  /**
   * @brief Negation of operator==.
   * @param other Grid to compare against.
   * @return true if the grids differ in dimensions or any cell.
   */
  bool operator!=(const Grid& other) const { return !(*this == other); }

private:
  std::size_t rows_;                 ///< Board height.
  std::size_t cols_;                 ///< Board width.
  std::vector<unsigned char> cells_; ///< Row-major cell buffer, size rows_ * cols_.
};

} // namespace gol
