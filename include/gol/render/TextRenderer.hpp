#pragma once

#include <cstdint>

#include "gol/IRenderer.hpp"

/**
 * @file TextRenderer.hpp
 * @brief ASCII renderer that dumps the board to stdout, one frame per generation.
 */

namespace gol {

/**
 * @brief Dumps the board to stdout as ASCII, one frame per generation.
 *
 * For small demos and eyeballing correctness only -- never use it in a benchmark
 * (use NullRenderer).
 */
class TextRenderer final : public IRenderer {
public:
  /**
   * @brief Construct with the glyphs used for live and dead cells.
   * @param alive Character printed for a live cell.
   * @param dead  Character printed for a dead cell.
   */
  explicit TextRenderer(char alive = '#', char dead = '.');

  /**
   * @brief Print one frame (a generation header line followed by the board).
   * @param grid       Board to print.
   * @param generation Generation index shown in the header.
   */
  void render(const Grid& grid, std::uint64_t generation) override;

private:
  char alive_; ///< Glyph for a live cell.
  char dead_;  ///< Glyph for a dead cell.
};

} // namespace gol
