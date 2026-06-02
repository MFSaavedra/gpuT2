#include "gol/patterns/Pattern.hpp"

#include "gol/Grid.hpp"

namespace gol {

void Pattern::applyTo(Grid& grid, std::size_t originX, std::size_t originY) const {
  for (const auto& [dx, dy] : liveCells) {
    const std::size_t x = originX + dx;
    const std::size_t y = originY + dy;
    if (x < grid.cols() && y < grid.rows()) {
      grid.at(x, y) = 1u;
    }
  }
}

} // namespace gol
