/**
 * @file cuda_equivalence_test.cpp
 * @brief Cross-backend equivalence: CudaEngine must match the CpuEngine oracle
 *        bit-for-bit, across block sizes, shared/global kernels, and edge modes.
 *
 * Only built when -DBUILD_CUDA=ON produced gol_cuda (see CMake). Requires a GPU
 * at run time.
 */

#include <gtest/gtest.h>

#include "gol/Grid.hpp"
#include "gol/engines/CpuEngine.hpp"
#include "gol/engines/CudaEngine.hpp"

using namespace gol;

namespace {

Grid runCpu(const Grid& start, bool wrap, int gens) {
  CpuEngine eng(1, wrap);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}

Grid runCuda(const Grid& start, bool wrap, int block, bool shared, int gens) {
  CudaEngine eng(block, wrap, shared);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}

} // namespace

class CudaVsCpu : public ::testing::TestWithParam<bool> {};

TEST_P(CudaVsCpu, MatchesOracleAcrossConfigs) {
  const bool wrap = GetParam();
  Grid start(130, 127); // non-square, not a multiple of any block dim, on purpose
  start.randomize(20260605);

  const Grid reference = runCpu(start, wrap, 20); // CPU oracle

  for (int block : {32, 64, 128, 256}) {
    for (bool shared : {false, true}) {
      Grid cuda = runCuda(start, wrap, block, shared, 20);
      EXPECT_TRUE(cuda == reference)
          << "CUDA diverged: block=" << block << " shared=" << shared
          << " wrap=" << wrap;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(EdgeModes, CudaVsCpu, ::testing::Values(false, true));

TEST(CudaVsCpu, BirthOnSixOnDevice) {
  // The defining variant case, built by hand: a dead centre (2,2) with exactly
  // six live neighbours must be born after one step on the GPU too.
  Grid g(5, 5);
  for (auto [x, y] : {std::pair{1, 1}, {2, 1}, {3, 1}, {1, 2}, {3, 2}, {1, 3}})
    g.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = 1u;
  ASSERT_EQ(g.at(2, 2), 0u);

  Grid next = runCuda(g, /*wrap=*/false, /*block=*/64, /*shared=*/false, 1);
  EXPECT_EQ(next.at(2, 2), 1u) << "dead cell with 6 live neighbours must be born";
}
