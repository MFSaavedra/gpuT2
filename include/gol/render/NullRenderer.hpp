#pragma once

#include <cstdint>

#include "gol/IRenderer.hpp"

namespace gol {

// Does nothing. Always benchmark against this so output cost never pollutes the
// cells/sec headline metric. Header-only because there is nothing to define.
class NullRenderer final : public IRenderer {
public:
  void render(const Grid& /*grid*/, std::uint64_t /*generation*/) override {}
};

} // namespace gol
