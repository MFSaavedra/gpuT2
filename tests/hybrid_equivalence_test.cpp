/**
 * @file hybrid_equivalence_test.cpp
 * @brief Cross-backend equivalence: HybridEngine (CPU + GPU, static load
 *        balancing) must match the CpuEngine oracle bit-for-bit, across CPU
 *        fractions (including the degenerate pure-CPU / pure-GPU splits and a
 *        calibrated split) and both edge modes.
 *
 * Only built when a GPU backend was configured (see CMake). Requires a GPU /
 * OpenCL device at run time. GOL_HAVE_{CUDA,OPENCL} select which GPU partitions
 * to exercise.
 */

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "gol/Grid.hpp"
#include "gol/engines/CpuEngine.hpp"
#include "gol/engines/HybridEngine.hpp"

using namespace gol;

namespace {

// GPU backends compiled into this build (the hybrid pairs the CPU with each).
std::vector<GpuBackend> backends() {
  std::vector<GpuBackend> v;
#ifdef GOL_HAVE_CUDA
  v.push_back(GpuBackend::Cuda);
#endif
#ifdef GOL_HAVE_OPENCL
  v.push_back(GpuBackend::OpenCL);
#endif
  return v;
}

const char* backendName(GpuBackend b) {
  switch (b) {
    case GpuBackend::Cuda: return "cuda";
    case GpuBackend::OpenCL: return "opencl";
    default: return "auto";
  }
}

Grid runCpu(const Grid& start, bool wrap, int gens) {
  CpuEngine eng(1, wrap);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}

Grid runHybrid(const Grid& start, bool wrap, GpuBackend backend,
               std::optional<double> cpuFrac, unsigned threads, int gens) {
  HybridEngine eng(threads, /*blockSize=*/128, wrap, backend, cpuFrac,
                   /*calibSteps=*/5);
  eng.upload(start);
  for (int i = 0; i < gens; ++i) eng.step();
  Grid out(start.rows(), start.cols());
  eng.download(out);
  return out;
}

} // namespace

class HybridVsCpu : public ::testing::TestWithParam<bool> {};

// A non-square grid that is not a multiple of any block dim, so the split point
// lands mid-block and the row-remainder logic of the CPU pool is exercised.
TEST_P(HybridVsCpu, MatchesOracleAcrossSplits) {
  const bool wrap = GetParam();
  Grid start(130, 127);
  start.randomize(20260615);

  const Grid reference = runCpu(start, wrap, 20); // CPU oracle

  for (GpuBackend backend : backends()) {
    // Explicit static splits, including the degenerate pure-CPU / pure-GPU ends.
    for (double frac : {0.0, 0.25, 0.5, 0.75, 1.0}) {
      Grid hybrid = runHybrid(start, wrap, backend, frac, /*threads=*/4, 20);
      EXPECT_TRUE(hybrid == reference)
          << "hybrid diverged: gpu=" << backendName(backend)
          << " cpu-frac=" << frac << " wrap=" << wrap;
    }
    // Auto-calibrated split (nullopt): the chosen split must still be exact.
    Grid calibrated = runHybrid(start, wrap, backend, std::nullopt,
                                /*threads=*/4, 20);
    EXPECT_TRUE(calibrated == reference)
        << "hybrid (calibrated) diverged: gpu=" << backendName(backend)
        << " wrap=" << wrap;
    // Sequential CPU slice (threads=1) at a mid split.
    Grid seq = runHybrid(start, wrap, backend, 0.5, /*threads=*/1, 20);
    EXPECT_TRUE(seq == reference)
        << "hybrid (threads=1) diverged: gpu=" << backendName(backend)
        << " wrap=" << wrap;
  }
}

INSTANTIATE_TEST_SUITE_P(EdgeModes, HybridVsCpu, ::testing::Values(false, true));

// The defining variant case across the seam: a dead centre with exactly six live
// neighbours must be born. Forced to a 50/50 split so the live cluster straddles
// the CPU/GPU boundary (the seam is the row-2 of a 5-row grid).
TEST(HybridVsCpu, BirthOnSixAcrossSeam) {
  for (GpuBackend backend : backends()) {
    Grid g(5, 5);
    for (auto [x, y] : {std::pair{1, 1}, {2, 1}, {3, 1}, {1, 2}, {3, 2}, {1, 3}})
      g.at(static_cast<std::size_t>(x), static_cast<std::size_t>(y)) = 1u;
    ASSERT_EQ(g.at(2, 2), 0u);

    Grid next = runHybrid(g, /*wrap=*/false, backend, /*cpuFrac=*/0.4,
                          /*threads=*/2, 1);
    EXPECT_EQ(next.at(2, 2), 1u)
        << "dead cell with 6 live neighbours must be born across the seam ("
        << backendName(backend) << ")";
  }
}

// upload() then download() with NO step must return the seed unchanged. Isolates
// the host/device assembly path (CPU buffer copy + GPU region D2H) from the kernel.
TEST(HybridVsCpu, RoundTripPreservesGrid) {
  for (GpuBackend backend : backends()) {
    Grid start(96, 71);
    start.randomize(424242);

    HybridEngine eng(/*threads=*/4, /*blockSize=*/128, /*wrap=*/false, backend,
                     /*cpuFrac=*/0.5, /*calibSteps=*/5);
    eng.upload(start);
    Grid out(start.rows(), start.cols());
    eng.download(out);

    EXPECT_TRUE(out == start)
        << "upload->download (no step) must be the identity (" << backendName(backend) << ")";
  }
}

// One engine instance, two uploads of DIFFERENT (non-divisible) sizes: the CPU and
// GPU buffers must be resized and the split recomputed for the second grid.
TEST(HybridVsCpu, ReuseAcrossSizes) {
  for (GpuBackend backend : backends()) {
    HybridEngine eng(/*threads=*/4, /*blockSize=*/128, /*wrap=*/true, backend,
                     /*cpuFrac=*/0.5, /*calibSteps=*/5);

    Grid a(64, 64);
    a.randomize(1);
    eng.upload(a);
    for (int i = 0; i < 7; ++i) eng.step();
    Grid aOut(a.rows(), a.cols());
    eng.download(aOut);
    EXPECT_TRUE(aOut == runCpu(a, /*wrap=*/true, 7))
        << "first (64x64) run diverged (" << backendName(backend) << ")";

    Grid b(101, 73);
    b.randomize(2);
    eng.upload(b);
    for (int i = 0; i < 7; ++i) eng.step();
    Grid bOut(b.rows(), b.cols());
    eng.download(bOut);
    EXPECT_TRUE(bOut == runCpu(b, /*wrap=*/true, 7))
        << "second (101x73) run after re-upload diverged (" << backendName(backend) << ")";
  }
}
