/**
 * @file cpu_parallel_test.cpp
 * @brief Sequential-vs-parallel equivalence for CpuEngine.
 *
 * Proves the core design claim: the parallel CPU solution is the sequential one
 * with the rows partitioned across cores. Same seed + same generations must give
 * a bit-for-bit identical board regardless of thread count and edge mode.
 */

#include <gtest/gtest.h>

#include "gol/Grid.hpp"
#include "gol/engines/CpuEngine.hpp"

using namespace gol;

namespace {
Grid runN(const Grid& start, unsigned threads, bool wrap, int gens) {
  CpuEngine eng(threads, wrap);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}
} // namespace

class SeqVsParallel : public ::testing::TestWithParam<bool> {};

TEST_P(SeqVsParallel, IdenticalAcrossThreadCounts) {
  const bool wrap = GetParam();
  Grid start(128, 130); // non-square, non-multiple-of-thread-count on purpose
  start.randomize(20260602);

  const Grid reference = runN(start, 1, wrap, 25); // sequential oracle
  for (unsigned threads : {2u, 3u, 4u, 8u, 0u /* all hw cores */}) {
    Grid parallel = runN(start, threads, wrap, 25);
    EXPECT_TRUE(parallel == reference)
        << "threads=" << threads << " wrap=" << wrap
        << " diverged from the sequential result";
  }
}

INSTANTIATE_TEST_SUITE_P(EdgeModes, SeqVsParallel, ::testing::Values(false, true));

TEST(SeqVsParallel, RemainderRowsHandled) {
  // Row count not divisible by the thread count exercises the remainder logic in
  // rangeFor(); every partition scheme must still match the sequential result.
  Grid start(17, 9);
  start.randomize(7);
  const Grid reference = runN(start, 1, false, 13);
  EXPECT_TRUE(runN(start, 4, false, 13) == reference);
  EXPECT_TRUE(runN(start, 5, false, 13) == reference);
}
