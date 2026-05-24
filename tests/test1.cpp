#include "GameOfLife.h"

#include <gtest/gtest.h>
#include <stdexcept>

// ---------------------------------------------
// Construction
// ---------------------------------------------
TEST(ConstructorTest, BasicDimensions) {
  GameOfLife g(4, 6);
  EXPECT_EQ(g.rows(), 4u);
  EXPECT_EQ(g.cols(), 6u);
}

TEST(ConstructorTest, AllCellsDeadByDefault) {
  GameOfLife g(3, 3);
  for (std::size_t r = 0; r < g.rows(); ++r)
    for (std::size_t c = 0; c < g.cols(); ++c)
      EXPECT_FALSE(g.at(r, c));
}

// ---------------------------------------------
// Access
// ---------------------------------------------
TEST(AccessTest, SetAndGet) {
  GameOfLife g(2, 2);
  g.set(0, 1, true);
  EXPECT_TRUE(g.at(0, 1));
  EXPECT_FALSE(g.at(0, 0));
}

TEST(AccessTest, OutOfRangeRow) {
  GameOfLife g(2, 2);
  EXPECT_THROW(g.at(2, 0), std::out_of_range);
}

TEST(AccessTest, OutOfRangeCol) {
  GameOfLife g(2, 2);
  EXPECT_THROW(g.set(0, 2, true), std::out_of_range);
}
