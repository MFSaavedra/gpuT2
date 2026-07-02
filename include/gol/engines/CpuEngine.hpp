#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gol/Barrier.hpp"
#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"

/**
 * @file CpuEngine.hpp
 * @brief CPU simulation backend serving both the sequential baseline and the
 *        data-parallel mode from one code path.
 */

namespace gol {

/**
 * @brief CPU backend for the simulation.
 *
 * One implementation serves both the SEQUENTIAL baseline and the data-PARALLEL
 * variant, because the Game of Life step has no intra-generation dependencies:
 * every cell reads only the current buffer and writes the next, so parallelising
 * is purely a matter of splitting the rows across worker threads. The per-cell
 * rule (LifeRules::nextState) is identical in both modes -- running with a single
 * core simply reduces the partition to one block and falls back to a straight
 * sequential sweep with zero synchronisation.
 *
 *   - threads == 1  -> sequential baseline (no worker threads, no barriers)
 *   - threads >= 2  -> N row-blocks computed in parallel
 *   - threads == 0  -> resolved to std::thread::hardware_concurrency()
 *
 * The parallel path uses a persistent worker pool synchronised with a reusable
 * Barrier (see Barrier.hpp) so per-generation thread-creation cost never pollutes
 * the cells/sec metric.
 */
class CpuEngine final : public ISimEngine {
public:
  /**
   * @brief Construct the engine and, if parallel, start the worker pool.
   * @param threads See the class table: 1 = sequential, N = parallel, 0 = all cores.
   * @param wrap    false = bounded edges (out-of-range neighbours count as dead),
   *                true = toroidal wrap. Must match every other backend.
   */
  explicit CpuEngine(unsigned threads = 1, bool wrap = false);

  /// @brief Stop and join the worker pool (if any).
  ~CpuEngine() override;

  CpuEngine(const CpuEngine&) = delete;            ///< Non-copyable (owns threads/barrier).
  CpuEngine& operator=(const CpuEngine&) = delete; ///< Non-copyable.
  CpuEngine(CpuEngine&&) = delete;                 ///< Non-movable.
  CpuEngine& operator=(CpuEngine&&) = delete;      ///< Non-movable.

  void upload(const Grid& initial) override;   ///< @copydoc ISimEngine::upload
  void step() override;                         ///< @copydoc ISimEngine::step
  void download(Grid& out) override;            ///< @copydoc ISimEngine::download
  double lastKernelMillis() const override { return lastMs_; } ///< @copydoc ISimEngine::lastKernelMillis
  std::string name() const override { return "cpu"; }          ///< @copydoc ISimEngine::name
  void pokeCell(std::size_t x, std::size_t y, unsigned char value) override; ///< @copydoc ISimEngine::pokeCell

  /**
   * @brief Resolved worker count (after mapping 0 -> hardware_concurrency).
   *
   * Exposed for logs and CSV columns so the sequential/parallel runs are
   * distinguishable.
   * @return Number of partitions/threads actually in use.
   */
  unsigned threads() const { return threads_; }

private:
  /// @brief Create the barrier and spawn the persistent worker threads.
  void startPool();

  /**
   * @brief Body of a worker thread: park on the barrier, compute its rows, repeat.
   * @param id Partition id in [1, threads_) owned by this worker.
   */
  void workerLoop(unsigned id);

  /**
   * @brief Compute the next state for rows [yBegin, yEnd) of @p src into @p dst.
   * @param src    Source (current) buffer.
   * @param dst    Destination (next) buffer.
   * @param yBegin First row to compute (inclusive).
   * @param yEnd   One past the last row to compute (exclusive).
   */
  void computeRows(const unsigned char* src, unsigned char* dst,
                   std::size_t yBegin, std::size_t yEnd) const;

  /**
   * @brief Live-neighbour count of cell (x, y) in @p g, honouring the edge mode.
   * @param g Buffer to read.
   * @param x Column index.
   * @param y Row index.
   * @return Number of live neighbours (0..8).
   */
  int countNeighbors(const unsigned char* g, std::size_t x, std::size_t y) const;

  /**
   * @brief Half-open row range owned by partition @p id of threads_.
   *
   * Id 0 is the main thread; ids 1..threads_-1 are workers. Remainder rows go to
   * the first partitions.
   * @param id Partition id in [0, threads_).
   * @return {begin, end} half-open row range.
   */
  std::pair<std::size_t, std::size_t> rangeFor(unsigned id) const;

  /**
   * @brief Map a requested thread count to a concrete one.
   * @param requested 0 = all hardware cores, otherwise the value itself.
   * @return Resolved count (>= 1).
   */
  static unsigned resolveThreads(unsigned requested);

  unsigned threads_;               ///< Resolved worker/partition count (>= 1).
  bool wrap_;                       ///< Edge mode: true = toroidal, false = bounded.
  std::size_t rows_ = 0;           ///< Board height (set by upload()).
  std::size_t cols_ = 0;           ///< Board width (set by upload()).
  std::vector<unsigned char> cur_; ///< Current generation (read by step).
  std::vector<unsigned char> nxt_; ///< Scratch for the next generation.
  double lastMs_ = 0.0;            ///< Compute time of the last step(), in ms.

  // Persistent worker pool (only populated when threads_ >= 2).
  std::vector<std::thread> workers_;   ///< Worker threads (size threads_ - 1).
  std::unique_ptr<Barrier> barrier_;   ///< Two-phase sync; threads_ participants.
  const unsigned char* src_ = nullptr;      ///< Source buffer published to workers each step.
  unsigned char* dst_ = nullptr;            ///< Destination buffer published to workers each step.
  std::atomic<bool> stop_{false};           ///< Set at destruction to release the workers.
};

} // namespace gol
