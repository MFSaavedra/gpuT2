#pragma once

#include <cstdint>

#include "gol/IRenderer.hpp"

namespace gol {

// Dumps the board to stdout as ASCII, one frame per generation. For small demos
// and eyeballing correctness only — never use it in a benchmark (use NullRenderer).
class TextRenderer final : public IRenderer {
public:
  explicit TextRenderer(char alive = '#', char dead = '.');
  void render(const Grid& grid, std::uint64_t generation) override;

private:
  char alive_;
  char dead_;
};

} // namespace gol
