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
#include "gol/HybridNode.hpp"
#include "gol/ISimEngine.hpp"
#include "gol/engines/IHaloPartition.hpp"

/**
 * @file HybridEngine.hpp
 * @brief Hybrid multi-device backend with STATIC load balancing (Divisible Load
 *        Theory, Barlas chapter 11.3).
 */

namespace gol {

/**
 * @brief Which GPU device the (legacy) two-node CPU+GPU constructor pairs with.
 *
 * Retained for the CPU+GPU constructor and its tests; the general constructor
 * takes an explicit ordered node list (see @ref HybridNode) instead.
 */
enum class GpuBackend {
  Auto,  ///< Prefer CUDA if compiled in, else OpenCL.
  Cuda,  ///< Force the CUDA partition.
  OpenCL ///< Force the OpenCL partition.
};

/**
 * @brief Hybrid backend: several compute nodes process one board together, split
 *        by rows, with a statically chosen partition per node.
 *
 * Row-wise domain decomposition over an ORDERED list of nodes (top to bottom).
 * The default composition is the integrated GPU plus the discrete GPU (the CPU is
 * negligible here and off by default), but any list of {CPU, CUDA-dGPU,
 * OpenCL-dGPU, OpenCL-iGPU} nodes is allowed. Each node keeps its band of rows
 * resident (CPU in host buffers, each GPU on its device) and ping-pongs locally;
 * the full board is reassembled only on download(). Between adjacent nodes (and,
 * under --wrap, the far edges) a one-row ghost halo is exchanged each generation
 * -- the only per-step host<->device traffic; inter-GPU seams stage through host
 * (no cross-vendor peer copy). Every GPU node runs the halo kernel so the result
 * is bit-for-bit identical to the CpuEngine oracle in both edge modes.
 *
 * Static load balancing (the chapter 11.3 core): model each node's cost as affine
 * in its share; all-finish-together gives `part_i = R_i / sum_j R_j` from the
 * measured throughputs R = 1/p. The split is chosen ONCE -- from explicit
 * per-node fractions or from a short a-priori calibration phase (time a few steps
 * per node on the actual grid) -- and then frozen for the whole run.
 *
 * A CPU node (if present) computes on the calling thread + a persistent Barrier
 * worker pool while every GPU node's kernel is in flight concurrently.
 */
class HybridEngine final : public ISimEngine {
public:
  /**
   * @brief General constructor: an explicit ordered list of nodes.
   * @param threads    CPU worker count for a CPU node (1 = sequential, 0 = all cores).
   * @param blockSize  GPU threads/work-items per block for the halo kernel.
   * @param wrap       false = bounded edges, true = toroidal. Must match the oracle.
   * @param nodes      Ordered nodes (top to bottom). Empty => a single Auto GPU node.
   * @param fracs      If set, explicit per-node row fractions (size == nodes.size(),
   *                   normalised); nullopt = auto-calibrate.
   * @param calibSteps Steps timed per node during the calibration phase.
   */
  HybridEngine(unsigned threads, int blockSize, bool wrap,
               std::vector<HybridNode> nodes,
               std::optional<std::vector<double>> fracs = std::nullopt,
               unsigned calibSteps = 10);

  /**
   * @brief Legacy two-node CPU+GPU constructor (kept for tests / --cpu-frac).
   * @param cpuFracOverride Fixed fraction of ROWS for the CPU (nullopt = calibrate).
   *
   * Equivalent to a [CPU, GPU(backend)] node list; with @p cpuFracOverride the
   * fractions are {F, 1-F}.
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
  double lastKernelMillis() const override { return lastMs_; } ///< Wall of the concurrent multi-node region.
  std::string name() const override { return "hybrid"; }        ///< @copydoc ISimEngine::name

  /// @brief Per-node summary for the run banner (label, rows, measured rate cells/s).
  struct NodeReport {
    std::string label;
    std::size_t rows = 0;
    double rate = 0.0; ///< Calibrated throughput (cells/s), 0 if not calibrated.
  };
  std::vector<NodeReport> nodeReports() const; ///< Active nodes, top to bottom.

  std::size_t cpuRows() const { return cpuRows_; } ///< @return Rows assigned to the CPU node (0 if none).
  std::size_t gpuRows() const { return gpuRows_; } ///< @return Rows assigned to all GPU nodes.
  double cpuFraction() const { return rows_ ? static_cast<double>(cpuRows_) / static_cast<double>(rows_) : 0.0; } ///< @return CPU share of the rows.
  double cpuRate() const { return cpuRate_; }     ///< @return Measured CPU throughput (cells/s); 0 if none/uncalibrated.
  double gpuRate() const { return gpuRate_; }     ///< @return Measured throughput of the first GPU node (cells/s).
  bool calibrated() const { return calibrated_; } ///< @return Whether the split came from calibration.
  std::string gpuName() const;                    ///< @return First GPU node's name, or "none".

private:
  /// One node in the decomposition: its spec, device (GPU) and assigned band.
  struct Node {
    HybridNode spec;
    std::unique_ptr<IHaloPartition> part; ///< GPU partition; null for a CPU node.
    double rate = 0.0;                    ///< Measured throughput (cells/s).
    std::size_t rows = 0;                 ///< Rows assigned to this node (0 = inactive).
    bool isCpu() const { return part == nullptr && spec.kind == NodeKind::Cpu; }
  };

  void buildNodes();                     ///< Materialise nodes_ from nodeSpec_ (create GPU partitions lazily).
  std::unique_ptr<IHaloPartition> makePartition(const HybridNode& n) const;
  void calibrate(const Grid& initial);   ///< Time each node on the full board -> Node::rate.
  double measureCpuRate(const Grid& initial);
  double measureGpuRate(IHaloPartition& part, const Grid& initial);
  void assignRows();                     ///< fracs/rates -> Node::rows; build active_ (non-empty), cpuNode_.
  void setupBuffers(const Grid& initial);///< Allocate/seed each active node's buffers.
  void exchangeGhosts();                 ///< Exchange one-row halos across adjacent active nodes.

  // Uniform per-node boundary access (host memcpy for CPU, IHaloPartition for GPU).
  void readTopRealRow(const Node& n, unsigned char* out);
  void readBottomRealRow(const Node& n, unsigned char* out);
  void setTopGhost(const Node& n, const unsigned char* in);
  void setBottomGhost(const Node& n, const unsigned char* in);

  // CPU worker pool (mirrors CpuEngine); computes the CPU node's rows.
  void startPool();
  void workerLoop(unsigned id);
  void runCpuParallel(const unsigned char* src, unsigned char* dst, std::size_t nReal);
  void computeCpuRows(const unsigned char* src, unsigned char* dst,
                      std::size_t brBegin, std::size_t brEnd) const;
  std::pair<std::size_t, std::size_t> bufferRangeFor(unsigned id) const;

  unsigned threads_;                       ///< Resolved CPU worker count (>= 1).
  int blockSize_;                          ///< GPU block size for the halo kernel.
  bool wrap_;                              ///< Edge mode: true = toroidal.
  unsigned calibSteps_;                    ///< Steps per node in the calibration phase.
  std::vector<HybridNode> nodeSpec_;       ///< Requested nodes, top to bottom.
  std::optional<std::vector<double>> fracsOverride_; ///< Explicit per-node fractions, or nullopt.
  bool hasCpuNode_ = false;                ///< A CPU node is present in the spec.

  std::size_t rows_ = 0;                   ///< Board height R.
  std::size_t cols_ = 0;                   ///< Board width C.
  bool calibrated_ = false;                ///< Split came from calibration (vs override).

  std::vector<Node> nodes_;                ///< All spec'd nodes (owns partitions); reserved, never reallocated.
  std::vector<Node*> active_;              ///< Non-empty nodes, top to bottom (into nodes_).
  Node* cpuNode_ = nullptr;                ///< The active CPU node, or null.

  // CPU host buffers (height cpuRows+2: top ghost, real rows, bottom ghost).
  std::vector<unsigned char> cpuCur_;      ///< Current CPU buffer (read by step).
  std::vector<unsigned char> cpuNxt_;      ///< Next CPU buffer (written by step).
  std::vector<unsigned char> rowBuf_;      ///< Scratch for one row (ghost transfers).

  double lastMs_ = 0.0;                     ///< Wall time of the last step()'s compute region.

  // Cached for the legacy accessors.
  std::size_t cpuRows_ = 0;                ///< Rows on the CPU node.
  std::size_t gpuRows_ = 0;                ///< Rows on all GPU nodes.
  double cpuRate_ = 0.0;                   ///< CPU node throughput (cells/s).
  double gpuRate_ = 0.0;                   ///< First GPU node throughput (cells/s).

  // Persistent worker pool (only populated when a CPU node exists and threads_ >= 2).
  std::vector<std::thread> workers_;
  std::unique_ptr<Barrier> barrier_;
  const unsigned char* poolSrc_ = nullptr;  ///< Source buffer published to workers.
  unsigned char* poolDst_ = nullptr;        ///< Destination buffer published to workers.
  std::size_t poolRows_ = 0;                ///< Real-row count published to workers.
  std::atomic<bool> stop_{false};           ///< Set at destruction to release workers.
};

/**
 * @brief The default node composition for this build (backend-aware).
 * @return {iGPU(OpenCL), dGPU(CUDA)} when both are compiled; {dGPU(CUDA)} for a
 *         CUDA-only build; {iGPU, dGPU-ocl} (both OpenCL) for an OpenCL-only build.
 */
std::vector<HybridNode> defaultHybridNodes();

/**
 * @brief Parse a "--nodes" spec into an ordered node list.
 * @param spec Comma-separated tokens: `cpu`, `dgpu` (CUDA), `dgpu-ocl`
 *             (OpenCL/NVIDIA), `igpu` (OpenCL/Intel).
 * @return The ordered nodes (top to bottom).
 * @throws std::invalid_argument on an unknown token or a token whose backend is
 *         not compiled into this build.
 */
std::vector<HybridNode> parseHybridNodes(const std::string& spec);

} // namespace gol
