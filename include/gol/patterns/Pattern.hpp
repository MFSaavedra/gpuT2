#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "gol/Grid.hpp"

/**
 * @file Pattern.hpp
 * @brief Backend-agnostic seed pattern (a sparse set of live cells) and its
 *        stamping operation.
 */

namespace gol {

/**
 * @brief A backend-agnostic seed pattern.
 *
 * The set of live-cell coordinates inside a bounding box, in the same (x, y)
 * convention as Grid. Produced by the RLE loader (or built by hand) and stamped
 * into a Grid before simulation.
 */
struct Pattern {
  std::size_t width = 0;  ///< Bounding-box width.
  std::size_t height = 0; ///< Bounding-box height.

  /// Live cells as (x, y) offsets relative to the pattern's top-left corner.
  std::vector<std::pair<std::size_t, std::size_t>> liveCells;

  /**
   * @brief Stamp the live cells into @p grid with the pattern's top-left at
   *        (@p originX, @p originY).
   *
   * Cells falling outside the grid are skipped (clipped, not an error).
   * @param[in,out] grid    Destination board.
   * @param originX Column of the pattern's top-left corner in the grid.
   * @param originY Row of the pattern's top-left corner in the grid.
   */
  void applyTo(Grid& grid, std::size_t originX, std::size_t originY) const;
};

} // namespace gol
