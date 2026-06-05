#pragma once

#include <cstddef>
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
 * Square cells: a terminal character cell is about twice as tall as it is wide,
 * so each grid cell is drawn TWO terminal columns wide -- the live glyph defaults
 * to two full blocks ("██") and dead to two spaces -- which reads as a square.
 * The viewport clip below divides the terminal width by the per-cell display
 * width accordingly. Glyphs are strings so multi-byte UTF-8 works.
 *
 * Big grids: the board is CLIPPED to the visible terminal viewport, never wrapped
 * -- a grid larger than the terminal shows a window with a "view(ox,oy) ... of
 * W x H" note in the header. For honest benchmarks use NullRenderer; this is a
 * small-grid visualisation aid.
 *
 * Interactive controls: when stdin is a TTY the renderer puts the terminal in
 * raw mode (non-blocking) and reads keys each frame -- the arrow keys pan the
 * viewport over a grid larger than the screen, `space`/`p` toggles pause (the
 * simulation halts but you can still pan), and `q` requests termination via
 * shouldClose(). Pause is contained entirely here (render() loops while paused),
 * so the main loop and engine are untouched. The terminal is restored on
 * destruction and on SIGINT/SIGTERM. When stdin is not a TTY (piped), the
 * controls are disabled and the top-left window is shown.
 */
class AnsiRenderer final : public IRenderer {
public:
  /**
   * @brief Enter animation mode (hide cursor, disable wrap, clear the screen).
   * @param alive   Glyph for a live cell (default: two U+2588 full blocks "██").
   * @param dead    Glyph for a dead cell (default: two spaces). Use the same
   *                display width as @p alive so the grid stays aligned.
   * @param delayMs Pause after each frame, in milliseconds, so the animation is
   *                watchable (the simulation loop is otherwise uncapped). 0
   *                disables the pause.
   */
  explicit AnsiRenderer(std::string alive = "██", std::string dead = "  ",
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

  /// @return true once the user has pressed `q` (interactive mode only).
  bool shouldClose() const override { return quit_; }

private:
  std::string alive_;       ///< Glyph for a live cell.
  std::string dead_;        ///< Glyph for a dead cell.
  unsigned delayMs_;        ///< Per-frame pause in milliseconds.
  std::size_t offsetX_ = 0; ///< Viewport top-left column in grid coordinates.
  std::size_t offsetY_ = 0; ///< Viewport top-left row in grid coordinates.
  bool interactive_ = false;///< stdin is a TTY -> raw mode + arrow-key panning.
  bool quit_ = false;       ///< User pressed `q` (drives shouldClose()).
  bool paused_ = false;     ///< Pause toggle (space/p): sim halts, panning still works.
};

} // namespace gol
