#pragma once

#include <cstdint>

#include "gol/IRenderer.hpp"

/**
 * @file NullRenderer.hpp
 * @brief No-op renderer used to keep output cost out of the benchmark.
 */

namespace gol {

/**
 * @brief Does nothing.
 *
 * Always benchmark against this so output cost never pollutes the cells/sec
 * headline metric. Header-only because there is nothing to define.
 */
class NullRenderer final : public IRenderer {
public:
  /// @brief No-op. @param grid Ignored. @param generation Ignored.
  void render(const Grid& /*grid*/, std::uint64_t /*generation*/) override {}
};

} // namespace gol
