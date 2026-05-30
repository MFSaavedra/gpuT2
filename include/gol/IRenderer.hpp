#pragma once

#include <cstdint>

#include "gol/Grid.hpp"

namespace gol {

// Strategy interface for output. Implementations: NullRenderer (does nothing —
// always benchmark against it so render cost never pollutes cells/sec),
// TextRenderer (ASCII dump), and a GuiRenderer later. The Application calls
// render() once per generation after download().
class IRenderer {
public:
  virtual ~IRenderer() = default;

  // Present the board for the given generation index.
  virtual void render(const Grid& grid, std::uint64_t generation) = 0;

  // Lets an interactive renderer (e.g. a GUI window) request loop termination.
  // Non-interactive renderers keep the default.
  virtual bool shouldClose() const { return false; }
};

} // namespace gol
