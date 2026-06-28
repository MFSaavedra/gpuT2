#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gol/Barrier.hpp"
#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"
#include "gol/engines/IHaloPartition.hpp"

/**
 * @file HybridEngine.hpp
 * @brief Hybrid CPU+GPU backend with STATIC load balancing (Divisible Load
 *        Theory, Barlas chapter 11.3).
 */

namespace gol {

/**
 * @brief Which GPU device the hybrid pairs the host CPU with.
 */
enum class GpuBackend {
  Auto,  ///< Prefer CUDA if compiled in, else OpenCL.
  Cuda,  ///< Force the CUDA partition.
  OpenCL ///< Force the OpenCL partition.
};

/**
 * @brief Hybrid backend: the CPU and a GPU process one board together, split by
 *        rows, with a statically chosen partition.
 *
 * Row-wise domain decomposition: the CPU owns rows [0, s) and the GPU owns
 * [s, R). Each side keeps its slice resident across generations (CPU in host
 * buffers, GPU on device) and ping-pongs locally; the full board is reassembled
 * only on download(). At the seam (and, under --wrap, the far edges) the two
 * sides exchange a one-row ghost halo each generation -- the only per-step
 * host<->device traffic. The GPU runs the halo kernel (vertical neighbours from
 * ghosts, horizontal edge rule applied) so the result is bit-for-bit identical to
 * the CpuEngine oracle in both edge modes.
 *
 * Static load balancing (the chapter 11.3 core): model each "node" cost as affine
 * in its share; the optimum makes both finish together, giving
 * `part_gpu = R_gpu / (R_cpu + R_gpu)` from the measured throughputs R = 1/p. The
 * split is chosen ONCE -- either from `--cpu-frac` (an offline cost-model
 * parameter) or from a short a-priori calibration phase (time a few CPU-only and
 * GPU-only steps on the actual grid) -- and then frozen for the whole run.
 *
 * The CPU half uses a persistent Barrier worker pool (same design as
 * CpuEngine) so per-generation thread-creation never pollutes the metric; while
 * the GPU kernel is in flight, the host computes the CPU rows concurrently.
 */
class HybridEngine final : public ISimEngine {
public:
  /**
   * @brief Construct the hybrid engine.
   * @param threads         CPU worker count for the CPU slice (1 = sequential,
   *                        N = parallel, 0 = all hardware cores).
   * @param blockSize       GPU threads/work-items per block for the halo kernel.
   * @param wrap            false = bounded edges, true = toroidal. Must match the oracle.
   * @param backend         Which GPU device to pair with (Auto/Cuda/OpenCL).
   * @param cpuFracOverride If set, the fixed fraction of ROWS given to the CPU
   *                        (skips calibration). nullopt = auto-calibrate.
   * @param calibSteps      Steps timed per node during the calibration phase.
   */
  explicit HybridEngine(unsigned threads = 0, int blockSize = 128, bool wrap = false,
                        GpuBackend backend = GpuBackend::Auto,
                        std::optional<double> cpuFracOverride = std::nullopt,
                        unsigned calibSteps = 10);

  ~HybridEngine() override;

  HybridEngine(const HybridEngine&) = delete;            ///< Non-copyable (owns threads + device memory).
  HybridEngine& operator=(const HybridEngine&) = delete; ///< Non-copyable.
  HybridEngine(HybridEngine&&) = delete;                 ///< Non-movable.
  HybridEngine& operator=(HybridEngine&&) = delete;      ///< Non-movable.

  void upload(const Grid& initial) override; ///< @copydoc ISimEngine::upload
  void step() override;                       ///< @copydoc ISimEngine::step
  void download(Grid& out) override;          ///< @copydoc ISimEngine::download
  double lastKernelMillis() const override { return lastMs_; } ///< Wall of the concurrent CPU+GPU region.
  std::string name() const override { return "hybrid"; }        ///< @copydoc ISimEngine::name

  std::size_t cpuRows() const { return cpuRows_; } ///< @return Rows assigned to the CPU.
  std::size_t gpuRows() const { return gpuRows_; } ///< @return Rows assigned to the GPU.
  double cpuFraction() const { return rows_ ? static_cast<double>(cpuRows_) / static_cast<double>(rows_) : 0.0; } ///< @return CPU share of the rows.
  double cpuRate() const { return cpuRate_; }     ///< @return Measured CPU throughput (cells/s); 0 if not calibrated.
  double gpuRate() const { return gpuRate_; }     ///< @return Measured GPU throughput (cells/s); 0 if not calibrated.
  bool calibrated() const { return calibrated_; } ///< @return Whether the split came from calibration.
  std::string gpuName() const;                    ///< @return GPU partition name ("cuda"/"opencl"), or "none".

private:
  void ensurePartition();                                  ///< Lazily create the GPU partition for the chosen backend.
  void calibrate(const Grid& initial);                     ///< Time CPU-only and GPU-only steps to set cpuRate_/gpuRate_.
  void setupBuffers(const Grid& initial);                  ///< Allocate CPU/GPU buffers for the chosen split and seed them.
  void exchangeGhosts();                                   ///< Fill both sides' ghost rows from the current buffers.

  // CPU worker pool (mirrors CpuEngine). Computes real rows [1, poolRows_] of the
  // current CPU buffer (vertical neighbours from ghost rows, horizontal edge rule).
  void startPool();
  void workerLoop(unsigned id);
  void runCpuParallel(const unsigned char* src, unsigned char* dst, std::size_t nReal); ///< Compute nReal rows across the pool.
  void computeCpuRows(const unsigned char* src, unsigned char* dst,
                      std::size_t brBegin, std::size_t brEnd) const;       ///< Buffer-row range [brBegin, brEnd).
  std::pair<std::size_t, std::size_t> bufferRangeFor(unsigned id) const;   ///< Buffer-row range owned by pool partition id.

  unsigned threads_;                       ///< Resolved CPU worker count (>= 1).
  int blockSize_;                          ///< GPU block size for the halo kernel.
  bool wrap_;                              ///< Edge mode: true = toroidal.
  GpuBackend backend_;                     ///< Requested GPU backend.
  std::optional<double> cpuFracOverride_;  ///< Manual CPU fraction, or nullopt to calibrate.
  unsigned calibSteps_;                    ///< Steps per node in the calibration phase.

  std::size_t rows_ = 0;                   ///< Board height R.
  std::size_t cols_ = 0;                   ///< Board width C.
  std::size_t cpuRows_ = 0;                ///< Rows owned by the CPU (split point s).
  std::size_t gpuRows_ = 0;                ///< Rows owned by the GPU (R - s).

  double cpuRate_ = 0.0;                   ///< Measured CPU throughput (cells/s).
  double gpuRate_ = 0.0;                   ///< Measured GPU throughput (cells/s).
  bool calibrated_ = false;                ///< Split came from calibration (vs override).

  // CPU host buffers (height cpuRows_+2: top ghost, real rows, bottom ghost).
  std::vector<unsigned char> cpuCur_;      ///< Current CPU buffer (read by step).
  std::vector<unsigned char> cpuNxt_;      ///< Next CPU buffer (written by step).
  std::vector<unsigned char> rowBuf_;      ///< Scratch for one row (ghost transfers).

  std::unique_ptr<IHaloPartition> partition_; ///< GPU partition (null when gpuRows_ == 0 and no calibration).

  double lastMs_ = 0.0;                     ///< Wall time of the last step()'s compute region.

  // Persistent worker pool (only populated when threads_ >= 2).
  std::vector<std::thread> workers_;
  std::unique_ptr<Barrier> barrier_;
  const unsigned char* poolSrc_ = nullptr;  ///< Source buffer published to workers.
  unsigned char* poolDst_ = nullptr;        ///< Destination buffer published to workers.
  std::size_t poolRows_ = 0;                ///< Real-row count published to workers.
  std::atomic<bool> stop_{false};           ///< Set at destruction to release workers.
};

} // namespace gol
