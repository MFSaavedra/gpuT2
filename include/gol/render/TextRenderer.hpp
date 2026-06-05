#pragma once

#include <cstdint>
#include <string>

#include "gol/IRenderer.hpp"

/**
 * @file TextRenderer.hpp
 * @brief Text renderer that dumps the board to stdout, one frame per generation.
 */

namespace gol {

/**
 * @brief Dumps the board to stdout, one frame per generation.
 *
 * Each frame is appended (the terminal scrolls). To counter the ~1:2 aspect
 * ratio of a terminal character cell, a live cell defaults to TWO full blocks
 * ("██") and a dead cell to two spaces, so each cell reads as a square rather
 * than a tall rectangle. Glyphs are strings so multi-byte UTF-8 works. For small
 * demos and eyeballing correctness only -- never use it in a benchmark (use
 * NullRenderer).
 */
class TextRenderer final : public IRenderer {
public:
  /**
   * @brief Construct with the glyphs used for live and dead cells.
   * @param alive Glyph for a live cell (default: two U+2588 full blocks "██").
   * @param dead  Glyph for a dead cell (default: two spaces). Use the same
   *              display width as @p alive so the grid stays aligned.
   */
  explicit TextRenderer(std::string alive = "██", std::string dead = "  ");

  /**
   * @brief Print one frame (a generation header line followed by the board).
   * @param grid       Board to print.
   * @param generation Generation index shown in the header.
   */
  void render(const Grid& grid, std::uint64_t generation) override;

private:
  std::string alive_; ///< Glyph for a live cell.
  std::string dead_;  ///< Glyph for a dead cell.
};

} // namespace gol
