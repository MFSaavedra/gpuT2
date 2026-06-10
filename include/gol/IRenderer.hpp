#pragma once

#include <cstdint>

#include "gol/Grid.hpp"

/**
 * @file IRenderer.hpp
 * @brief Strategy interface for output (NullRenderer, TextRenderer, ...).
 */

namespace gol {

/**
 * @brief Strategy interface for output.
 *
 * Implementations: NullRenderer (does nothing -- always benchmark against it so
 * render cost never pollutes cells/sec), TextRenderer (ASCII dump), and a
 * GuiRenderer later. The Application calls render() once per generation after
 * download().
 */
class IRenderer {
public:
  virtual ~IRenderer() = default;

  /**
   * @brief Present the board for the given generation index.
   * @param grid       Board to render.
   * @param generation Generation index (0 = seed, then 1, 2, ...).
   */
  virtual void render(const Grid& grid, std::uint64_t generation) = 0;

  /**
   * @brief Lets an interactive renderer (e.g. a GUI window) request loop termination.
   *
   * Non-interactive renderers keep the default.
   * @return true to ask the main loop to stop; false to continue.
   */
  virtual bool shouldClose() const { return false; }

  /**
   * @brief Whether the app should keep displaying the final frame after the run,
   *        until shouldClose(), so an interactive viewer can be explored.
   *
   * Non-interactive renderers keep the default (false), so the app exits when the
   * generation loop ends.
   * @return true to hold on the final frame; false to exit when the run finishes.
   */
  virtual bool staysOpen() const { return false; }
};

} // namespace gol
