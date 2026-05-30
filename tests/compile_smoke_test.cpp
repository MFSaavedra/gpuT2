// Compile-smoke test for the strategy interface layer.
//
// Its job is to #include every new header so the whole scaffolding is verified
// to build (and the headers stay mutually consistent) via the normal ctest flow,
// before any engine, renderer or kernel bodies exist. It only exercises the
// header-only inline helpers; declared-but-undefined symbols (Config::parse,
// RleLoader::load, Pattern::applyTo) are intentionally not called so the test
// links without those future translation units.

#include "gol/Config.hpp"
#include "gol/Grid.hpp"
#include "gol/IRenderer.hpp"
#include "gol/ISimEngine.hpp"
#include "gol/LifeRules.hpp"
#include "gol/Timer.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"

#include <gtest/gtest.h>

using namespace gol;

TEST(CompileSmoke, GridBasics) {
  Grid g(3, 4); // rows=3, cols=4
  EXPECT_EQ(g.rows(), 3u);
  EXPECT_EQ(g.cols(), 4u);
  EXPECT_EQ(g.size(), 12u);

  g.at(2, 1) = 1; // (x=2, y=1)
  EXPECT_EQ(g.at(2, 1), 1u);

  Grid copy = g;
  EXPECT_TRUE(g == copy);
  copy.at(0, 0) = 1;
  EXPECT_TRUE(g != copy);

  g.fill(0);
  EXPECT_EQ(g.at(2, 1), 0u);
}

TEST(CompileSmoke, RandomizeIsDeterministic) {
  Grid a(8, 8), b(8, 8);
  a.randomize(42);
  b.randomize(42);
  EXPECT_TRUE(a == b);
}

TEST(CompileSmoke, VariantRule) {
  // Born on exactly 3 OR exactly 6; survive on 2 or 3.
  EXPECT_EQ(nextState(0, 3), 1u);
  EXPECT_EQ(nextState(0, 6), 1u);
  EXPECT_EQ(nextState(0, 2), 0u);
  EXPECT_EQ(nextState(0, 5), 0u);
  EXPECT_EQ(nextState(1, 2), 1u);
  EXPECT_EQ(nextState(1, 3), 1u);
  EXPECT_EQ(nextState(1, 1), 0u);
  EXPECT_EQ(nextState(1, 6), 0u);
}

TEST(CompileSmoke, TimerRuns) {
  Timer t;
  EXPECT_GE(t.elapsedMillis(), 0.0);
}

TEST(CompileSmoke, ConfigDefaults) {
  Config c;
  EXPECT_EQ(c.engine, EngineKind::Cpu);
  EXPECT_EQ(c.renderer, RendererKind::Null);
  EXPECT_FALSE(c.wrap);
}
