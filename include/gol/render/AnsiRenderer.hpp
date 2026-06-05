#pragma once

#include <cstdint>
#include <string>

#include "gol/IRenderer.hpp"

/**
 * @file AnsiRenderer.hpp
 * @brief In-place terminal renderer using ANSI/DEC escape codes.
 */

namespace gol {

/**
 * @brief Animates the board in place in an ANSI terminal.
 *
 * Unlike TextRenderer (which appends each frame, so the terminal scrolls), this
 * renderer homes the cursor and overwrites the same region every frame, giving a
 * stable animation. It hides the cursor and disables line wrapping on
 * construction and restores both on destruction (RAII).
 *
 * Big grids: the board is CLIPPED to the visible terminal viewport (the top-left
 * region that fits), never wrapped -- a grid wider/taller than the terminal
 * shows a window with a "[cols x rows of W x H]" note in the header. For honest
 * benchmarks use NullRenderer; this is a small-grid visualisation aid.
 *
 * The live-cell glyph defaults to a Unicode full block (U+2588), which is one
 * display column wide so the grid stays aligned; glyphs are strings so multi-byte
 * UTF-8 works. Intended for an interactive TTY; if stdout is not a terminal the
 * size query falls back to 80x24 and the escape codes are still emitted.
 */
class AnsiRenderer final : public IRenderer {
public:
  /**
   * @brief Enter animation mode (hide cursor, disable wrap, clear the screen).
   * @param alive   Glyph for a live cell (default: U+2588 full block "█").
   * @param dead    Glyph for a dead cell (default: ".").
   * @param delayMs Pause after each frame, in milliseconds, so the animation is
   *                watchable (the simulation loop is otherwise uncapped). 0
   *                disables the pause.
   */
  explicit AnsiRenderer(std::string alive = "█", std::string dead = ".",
                        unsigned delayMs = 50);

  /// @brief Restore the terminal (re-enable wrap, show cursor) and drop below the frame.
  ~AnsiRenderer() override;

  AnsiRenderer(const AnsiRenderer&) = delete;            ///< Non-copyable (owns terminal state).
  AnsiRenderer& operator=(const AnsiRenderer&) = delete; ///< Non-copyable.
  AnsiRenderer(AnsiRenderer&&) = delete;                 ///< Non-movable.
  AnsiRenderer& operator=(AnsiRenderer&&) = delete;      ///< Non-movable.

  /**
   * @brief Redraw the board in place for the given generation.
   * @param grid       Board to draw (clipped to the terminal viewport).
   * @param generation Generation index shown in the header.
   */
  void render(const Grid& grid, std::uint64_t generation) override;

private:
  std::string alive_; ///< Glyph for a live cell.
  std::string dead_;  ///< Glyph for a dead cell.
  unsigned delayMs_;  ///< Per-frame pause in milliseconds.
};

} // namespace gol
