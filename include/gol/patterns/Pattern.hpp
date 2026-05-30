#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "gol/Grid.hpp"

namespace gol {

// A backend-agnostic seed pattern: the set of live-cell coordinates inside a
// bounding box, in the same (x, y) convention as Grid. Produced by the RLE
// loader (or built by hand) and stamped into a Grid before simulation.
struct Pattern {
  std::size_t width = 0;  // bounding-box width
  std::size_t height = 0; // bounding-box height

  // Live cells as (x, y) offsets relative to the pattern's top-left corner.
  std::vector<std::pair<std::size_t, std::size_t>> liveCells;

  // Stamp the live cells into `grid` with the pattern's top-left at (originX,
  // originY). Cells falling outside the grid are skipped. Implemented later.
  void applyTo(Grid& grid, std::size_t originX, std::size_t originY) const;
};

} // namespace gol
