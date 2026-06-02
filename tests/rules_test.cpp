// CpuEngine correctness on known patterns under the VARIANT rule
// (born on exactly 3 OR exactly 6; survive on 2 or 3). CpuEngine is the
// reference oracle every later backend is checked against, so it is verified
// first. PATTERNS_DIR is injected by CMake.

#include <string>

#include <gtest/gtest.h>

#include "gol/Grid.hpp"
#include "gol/engines/CpuEngine.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"

using namespace gol;

namespace {
std::string pat(const char* name) { return std::string(PATTERNS_DIR) + "/" + name; }

// Build a grid of the given size and stamp an RLE pattern at (ox, oy).
Grid seeded(std::size_t rows, std::size_t cols, const char* rle,
            std::size_t ox, std::size_t oy) {
  Grid g(rows, cols);
  RleLoader::load(pat(rle)).applyTo(g, ox, oy);
  return g;
}

// Advance a copy `gens` generations with the given engine config and return it.
Grid run(const Grid& start, unsigned threads, bool wrap, int gens) {
  CpuEngine eng(threads, wrap);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}
} // namespace

TEST(Rules, BlockIsStillLife) {
  // A block must not change: each live cell has 3 neighbours (survives) and no
  // dead cell reaches 3 or 6 neighbours.
  Grid g = seeded(5, 5, "block.rle", 1, 1);
  EXPECT_TRUE(run(g, 1, false, 1) == g);
  EXPECT_TRUE(run(g, 1, false, 10) == g); // still stationary after many steps
}

TEST(Rules, BlinkerOscillatesPeriod2) {
  // Horizontal bar at row 2, cols 1..3 -> vertical bar at col 2, rows 1..3, and back.
  Grid horizontal = seeded(5, 5, "blinker.rle", 1, 2);

  Grid expectedVertical(5, 5);
  expectedVertical.at(2, 1) = 1;
  expectedVertical.at(2, 2) = 1;
  expectedVertical.at(2, 3) = 1;

  Grid afterOne = run(horizontal, 1, false, 1);
  EXPECT_TRUE(afterOne == expectedVertical);

  Grid afterTwo = run(horizontal, 1, false, 2);
  EXPECT_TRUE(afterTwo == horizontal); // period 2
}

TEST(Rules, BirthOnSixBringsCentreAlive) {
  // The defining variant case: the dead centre has exactly 6 live neighbours and
  // must be born. (Vanilla Conway would leave it dead.)
  Grid g = seeded(5, 5, "birth_on_six.rle", 1, 1); // centre dead cell lands at (2,2)
  ASSERT_EQ(g.at(2, 2), 0u);

  Grid next = run(g, 1, false, 1);
  EXPECT_EQ(next.at(2, 2), 1u) << "dead cell with 6 live neighbours must be born";
}

TEST(Rules, BirthOnSixToroidalAgrees) {
  // Same outcome with wrap edges, since the pattern is interior here.
  Grid g = seeded(5, 5, "birth_on_six.rle", 1, 1);
  Grid next = run(g, 1, true, 1);
  EXPECT_EQ(next.at(2, 2), 1u);
}
