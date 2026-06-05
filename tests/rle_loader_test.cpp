/**
 * @file rle_loader_test.cpp
 * @brief RLE loader + Pattern::applyTo tests.
 *
 * PATTERNS_DIR is injected by CMake and points at the repo's patterns/ directory.
 */

#include <string>

#include <gtest/gtest.h>

#include "gol/Grid.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"

using namespace gol;

namespace {
std::string pat(const char* name) { return std::string(PATTERNS_DIR) + "/" + name; }

bool hasCell(const Pattern& p, std::size_t x, std::size_t y) {
  for (const auto& [cx, cy] : p.liveCells)
    if (cx == x && cy == y) return true;
  return false;
}
} // namespace

TEST(RleLoader, ParsesBlock) {
  Pattern p = RleLoader::load(pat("block.rle"));
  EXPECT_EQ(p.width, 2u);
  EXPECT_EQ(p.height, 2u);
  ASSERT_EQ(p.liveCells.size(), 4u);
  EXPECT_TRUE(hasCell(p, 0, 0));
  EXPECT_TRUE(hasCell(p, 1, 0));
  EXPECT_TRUE(hasCell(p, 0, 1));
  EXPECT_TRUE(hasCell(p, 1, 1));
}

TEST(RleLoader, ParsesBlinker) {
  Pattern p = RleLoader::load(pat("blinker.rle"));
  EXPECT_EQ(p.width, 3u);
  EXPECT_EQ(p.height, 1u);
  ASSERT_EQ(p.liveCells.size(), 3u);
  EXPECT_TRUE(hasCell(p, 0, 0));
  EXPECT_TRUE(hasCell(p, 1, 0));
  EXPECT_TRUE(hasCell(p, 2, 0));
}

TEST(RleLoader, ParsesBirthOnSix) {
  // 3o$obo$o!  ->  6 live cells, with the centre (1,1) explicitly dead.
  Pattern p = RleLoader::load(pat("birth_on_six.rle"));
  EXPECT_EQ(p.width, 3u);
  EXPECT_EQ(p.height, 3u);
  ASSERT_EQ(p.liveCells.size(), 6u);
  EXPECT_FALSE(hasCell(p, 1, 1)); // the dead centre is the whole point
  EXPECT_TRUE(hasCell(p, 1, 0));
  EXPECT_TRUE(hasCell(p, 2, 1));
}

TEST(RleLoader, MissingFileThrows) {
  EXPECT_THROW(RleLoader::load(pat("does_not_exist.rle")), std::exception);
}

TEST(PatternApply, StampsAtOriginAndClips) {
  Pattern p = RleLoader::load(pat("block.rle"));
  Grid g(5, 5);
  p.applyTo(g, 1, 1); // block occupies (1,1),(2,1),(1,2),(2,2)
  EXPECT_EQ(g.at(1, 1), 1u);
  EXPECT_EQ(g.at(2, 1), 1u);
  EXPECT_EQ(g.at(1, 2), 1u);
  EXPECT_EQ(g.at(2, 2), 1u);
  EXPECT_EQ(g.at(0, 0), 0u);

  // Cells stamped outside the grid are silently clipped, not a crash.
  Grid small(2, 2);
  p.applyTo(small, 1, 1); // only (1,1) lands inside
  EXPECT_EQ(small.at(1, 1), 1u);
}
