#pragma once

#include <string>

#include "gol/patterns/Pattern.hpp"

namespace gol {

// Loads a Game of Life pattern from an RLE file (the de-facto Life format) into
// a Pattern. Parsing is implemented in a later pass; declared here so the data
// pipeline is visible in the interface layer.
struct RleLoader {
  static Pattern load(const std::string& path);
};

} // namespace gol
