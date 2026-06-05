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
 * Each frame is appended (the terminal scrolls). The live-cell glyph defaults to
 * a Unicode full block (U+2588); glyphs are strings so multi-byte UTF-8 works.
 * For small demos and eyeballing correctness only -- never use it in a benchmark
 * (use NullRenderer).
 */
class TextRenderer final : public IRenderer {
public:
  /**
   * @brief Construct with the glyphs used for live and dead cells.
   * @param alive Glyph for a live cell (default: U+2588 full block "█").
   * @param dead  Glyph for a dead cell (default: ".").
   */
  explicit TextRenderer(std::string alive = "█", std::string dead = ".");

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
