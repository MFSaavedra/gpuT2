#pragma once

#include <string>

#include "gol/patterns/Pattern.hpp"

/**
 * @file RleLoader.hpp
 * @brief Loader for Game of Life patterns in the RLE file format.
 */

namespace gol {

/**
 * @brief Loads a Game of Life pattern from an RLE file (the de-facto Life format)
 *        into a Pattern.
 */
struct RleLoader {
  /**
   * @brief Parse an RLE file into a Pattern.
   * @param path Filesystem path to the .rle file.
   * @return The decoded pattern (bounding box + live cells).
   * @throws std::runtime_error if the file cannot be opened.
   */
  static Pattern load(const std::string& path);
};

} // namespace gol
