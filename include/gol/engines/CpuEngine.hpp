#pragma once

#include <atomic>
#include <barrier>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"

namespace gol {

// CPU backend for the simulation. One implementation serves both the SEQUENTIAL
// baseline and the data-PARALLEL variant, because the Game of Life step has no
// intra-generation dependencies: every cell reads only the current buffer and
// writes the next, so parallelising is purely a matter of splitting the rows
// across worker threads. The per-cell rule (LifeRules::nextState) is identical
// in both modes — running with a single core simply reduces the partition to one
// block and falls back to a straight sequential sweep with zero synchronisation.
//
//   threads == 1  -> sequential baseline (no worker threads, no barriers)
//   threads >= 2  -> N row-blocks computed in parallel
//   threads == 0  -> resolved to std::thread::hardware_concurrency()
//
// The parallel path uses a persistent worker pool synchronised with a std::barrier
// so per-generation thread-creation cost never pollutes the cells/sec metric.
class CpuEngine final : public ISimEngine {
public:
  // `threads`: see the table above. `wrap`: false = bounded edges (out-of-range
  // neighbours count as dead), true = toroidal wrap. Edge handling must match
  // every other backend in the equivalence test.
  explicit CpuEngine(unsigned threads = 1, bool wrap = false);
  ~CpuEngine() override;

  CpuEngine(const CpuEngine&) = delete;
  CpuEngine& operator=(const CpuEngine&) = delete;
  CpuEngine(CpuEngine&&) = delete;
  CpuEngine& operator=(CpuEngine&&) = delete;

  void upload(const Grid& initial) override;
  void step() override;
  void download(Grid& out) override;
  double lastKernelMillis() const override { return lastMs_; }
  std::string name() const override { return "cpu"; }

  // Resolved worker count (after mapping 0 -> hardware_concurrency). Exposed for
  // logs and CSV columns so the sequential/parallel runs are distinguishable.
  unsigned threads() const { return threads_; }

private:
  void startPool();
  void workerLoop(unsigned id);

  // Compute the next state for rows [yBegin, yEnd) of `src` into `dst`.
  void computeRows(const unsigned char* src, unsigned char* dst,
                   std::size_t yBegin, std::size_t yEnd) const;

  // Live-neighbour count of cell (x, y) in `g`, honouring the edge mode.
  int countNeighbors(const unsigned char* g, std::size_t x, std::size_t y) const;

  // Half-open row range owned by partition `id` of `threads_` (id 0 is the main
  // thread; ids 1..threads_-1 are workers). Remainder rows go to the first ones.
  std::pair<std::size_t, std::size_t> rangeFor(unsigned id) const;

  static unsigned resolveThreads(unsigned requested);

  unsigned threads_;
  bool wrap_;
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<unsigned char> cur_; // current generation (read by step)
  std::vector<unsigned char> nxt_; // scratch for the next generation
  double lastMs_ = 0.0;

  // Persistent worker pool (only populated when threads_ >= 2).
  std::vector<std::thread> workers_;
  std::unique_ptr<std::barrier<>> barrier_;
  const unsigned char* src_ = nullptr; // published to workers each step
  unsigned char* dst_ = nullptr;
  std::atomic<bool> stop_{false};
};

} // namespace gol
