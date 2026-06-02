#include "gol/render/TextRenderer.hpp"

#include <cstdio>
#include <string>

#include "gol/Grid.hpp"

namespace gol {

TextRenderer::TextRenderer(char alive, char dead) : alive_(alive), dead_(dead) {}

void TextRenderer::render(const Grid& grid, std::uint64_t generation) {
  // Build the whole frame in one buffer, then a single write — cheaper and less
  // flickery than per-cell putchar.
  std::string out;
  out.reserve(grid.size() + grid.rows() + 32);
  out += "generation ";
  out += std::to_string(generation);
  out += '\n';
  for (std::size_t y = 0; y < grid.rows(); ++y) {
    for (std::size_t x = 0; x < grid.cols(); ++x) {
      out += grid.at(x, y) ? alive_ : dead_;
    }
    out += '\n';
  }
  out += '\n';
  std::fputs(out.c_str(), stdout);
}

} // namespace gol
