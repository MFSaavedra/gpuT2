/**
 * @file TextRenderer.cpp
 * @brief Implementation of the text renderer.
 */

#include "gol/render/TextRenderer.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

#include "gol/Grid.hpp"

namespace gol {

TextRenderer::TextRenderer(std::string alive, std::string dead)
    : alive_(std::move(alive)), dead_(std::move(dead)) {}

void TextRenderer::render(const Grid& grid, std::uint64_t generation) {
  // Build the whole frame in one buffer, then a single write — cheaper and less
  // flickery than per-cell putchar.
  const std::size_t glyph = std::max(alive_.size(), dead_.size());
  std::string out;
  out.reserve(grid.size() * glyph + grid.rows() + 32);
  out += "generation ";
  out += std::to_string(generation);
  out += '\n';
  for (std::size_t y = 0; y < grid.rows(); ++y) {
    for (std::size_t x = 0; x < grid.cols(); ++x) {
      out += (grid.at(x, y) ? alive_ : dead_);
    }
    out += '\n';
  }
  out += '\n';
  std::fputs(out.c_str(), stdout);
}

} // namespace gol
