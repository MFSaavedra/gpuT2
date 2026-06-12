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

// upload() then download() with NO step must return the seed unchanged. This
// isolates the host<->device transfer path (the H2D/D2H cudaMemcpy) from the
// kernel: a failure here is a transfer bug, not a simulation bug, which a
// step-based test could not tell apart.
TEST(CudaVsCpu, RoundTripPreservesGrid) {
  Grid start(96, 71);
  start.randomize(424242);

  CudaEngine eng(128, /*wrap=*/false, /*shared=*/false);
  eng.upload(start);
  Grid out(start.rows(), start.cols());
  eng.download(out);

  EXPECT_TRUE(out == start) << "upload->download (no step) must be the identity";
}

// One engine instance, two uploads of DIFFERENT (non-divisible) sizes: the device
// buffers must be reallocated and the cached dimensions updated for the second
// grid. A stale-buffer or stale-dimension bug would corrupt the second result or
// read out of bounds -- a real hazard since upload() cudaFree's and re-mallocs.
TEST(CudaVsCpu, ReuseAcrossSizes) {
  CudaEngine eng(64, /*wrap=*/true, /*shared=*/true);

  Grid a(64, 64);
  a.randomize(1);
  eng.upload(a);
  for (int i = 0; i < 7; ++i) eng.step();
  Grid aOut(a.rows(), a.cols());
  eng.download(aOut);
  EXPECT_TRUE(aOut == runCpu(a, /*wrap=*/true, 7)) << "first (64x64) run diverged";

  Grid b(101, 73); // different shape, not a multiple of the block
  b.randomize(2);
  eng.upload(b);
  for (int i = 0; i < 7; ++i) eng.step();
  Grid bOut(b.rows(), b.cols());
  eng.download(bOut);
  EXPECT_TRUE(bOut == runCpu(b, /*wrap=*/true, 7))
      << "second (101x73) run after re-upload diverged -- buffers not resized?";
}
