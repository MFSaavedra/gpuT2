/**
 * @file HybridEngine.cpp
 * @brief Implementation of the hybrid CPU+GPU backend with static load balancing
 *        (Divisible Load Theory, Barlas chapter 11.3).
 *
 * Row-wise domain decomposition: the CPU owns rows [0, s) and the GPU owns
 * [s, R). The split s is chosen ONCE -- from --cpu-frac or from a short a-priori
 * calibration phase -- and then frozen. Each generation the two sides compute
 * concurrently and exchange a one-row ghost halo at the seam (and the far edges
 * under --wrap); the bulk slice never moves until download().
 */

#include "gol/engines/HybridEngine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "gol/LifeRules.hpp"
#include "gol/Timer.hpp"

#ifdef GOL_HAVE_CUDA
#include "gol/engines/cuda/CudaHaloPartition.hpp"
#endif
#ifdef GOL_HAVE_OPENCL
#include "gol/engines/opencl/OpenCLHaloPartition.hpp"
#endif

namespace gol {

namespace {
unsigned resolveThreads(unsigned requested) {
  if (requested != 0) return requested;
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1u : hw;
}
} // namespace

HybridEngine::HybridEngine(unsigned threads, int blockSize, bool wrap,
                           GpuBackend backend, std::optional<double> cpuFracOverride,
                           unsigned calibSteps)
    : threads_(resolveThreads(threads)),
      blockSize_(blockSize < 32 ? 32 : blockSize),
      wrap_(wrap),
      backend_(backend),
      cpuFracOverride_(cpuFracOverride),
      calibSteps_(calibSteps == 0 ? 1u : calibSteps) {
  if (threads_ >= 2) startPool();
}

HybridEngine::~HybridEngine() {
  if (!workers_.empty()) {
    stop_.store(true, std::memory_order_relaxed);
    barrier_->arrive_and_wait(); // workers observe stop_ and return
    for (auto& w : workers_) w.join();
  }
}

// ---------------------------------------------------------------------------
// GPU partition factory
// ---------------------------------------------------------------------------
void HybridEngine::ensurePartition() {
  if (partition_) return;
  GpuBackend b = backend_;
  if (b == GpuBackend::Auto) {
#if defined(GOL_HAVE_CUDA)
    b = GpuBackend::Cuda;
#elif defined(GOL_HAVE_OPENCL)
    b = GpuBackend::OpenCL;
#else
    throw std::runtime_error("hybrid: no GPU backend compiled in");
#endif
  }
  if (b == GpuBackend::Cuda) {
#if defined(GOL_HAVE_CUDA)
    partition_ = makeCudaHaloPartition(blockSize_);
#else
    throw std::runtime_error("hybrid: CUDA backend requested but not built");
#endif
  } else { // OpenCL
#if defined(GOL_HAVE_OPENCL)
    partition_ = makeOpenCLHaloPartition(blockSize_);
#else
    throw std::runtime_error("hybrid: OpenCL backend requested but not built");
#endif
  }
}

std::string HybridEngine::gpuName() const {
  return partition_ ? partition_->name() : std::string("none");
}

// ---------------------------------------------------------------------------
// Calibration (the a-priori "initialization phase" of DLT): measure each node's
// pure compute throughput, independently, on the actual grid. Ghost contents do
// not affect kernel/compute cost, so the loops run with zero ghosts.
// ---------------------------------------------------------------------------
void HybridEngine::calibrate(const Grid& initial) {
  const std::size_t R = rows_;
  const std::size_t C = cols_;
  const double cells = static_cast<double>(R) * static_cast<double>(C) *
                       static_cast<double>(calibSteps_);
  constexpr unsigned warmup = 2;

  // --- CPU: full board across the worker pool ---
  std::vector<unsigned char> a((R + 2) * C, 0u);
  std::vector<unsigned char> b((R + 2) * C, 0u);
  std::copy(initial.data(), initial.data() + R * C, a.data() + C);
  for (unsigned i = 0; i < warmup; ++i) {
    runCpuParallel(a.data(), b.data(), R);
    std::swap(a, b);
  }
  Timer tc;
  for (unsigned i = 0; i < calibSteps_; ++i) {
    runCpuParallel(a.data(), b.data(), R);
    std::swap(a, b);
  }
  const double cpuMs = tc.elapsedMillis();
  cpuRate_ = cpuMs > 0.0 ? cells / (cpuMs / 1000.0) : 0.0;

  // --- GPU: full board on the device ---
  ensurePartition();
  partition_->uploadRegion(initial.data(), R, C, wrap_);
  for (unsigned i = 0; i < warmup; ++i) { partition_->launchStep(); partition_->finishStep(); }
  Timer tg;
  for (unsigned i = 0; i < calibSteps_; ++i) { partition_->launchStep(); partition_->finishStep(); }
  const double gpuMs = tg.elapsedMillis();
  gpuRate_ = gpuMs > 0.0 ? cells / (gpuMs / 1000.0) : 0.0;
}

// ---------------------------------------------------------------------------
// upload(): decide the split, then size and seed the buffers.
// ---------------------------------------------------------------------------
void HybridEngine::upload(const Grid& initial) {
  rows_ = initial.rows();
  cols_ = initial.cols();
  rowBuf_.assign(cols_, 0u);

  double cpuFrac;
  if (cpuFracOverride_) {
    cpuFrac = std::clamp(*cpuFracOverride_, 0.0, 1.0);
    calibrated_ = false;
  } else {
    calibrate(initial); // sets cpuRate_, gpuRate_
    const double total = cpuRate_ + gpuRate_;
    cpuFrac = total > 0.0 ? cpuRate_ / total : 0.5;
    calibrated_ = true;
  }

  cpuRows_ = static_cast<std::size_t>(std::llround(cpuFrac * static_cast<double>(rows_)));
  if (cpuRows_ > rows_) cpuRows_ = rows_;
  gpuRows_ = rows_ - cpuRows_;

  setupBuffers(initial);
}

void HybridEngine::setupBuffers(const Grid& initial) {
  const std::size_t C = cols_;
  if (cpuRows_ > 0) {
    cpuCur_.assign((cpuRows_ + 2) * C, 0u);
    cpuNxt_.assign((cpuRows_ + 2) * C, 0u);
    // Board rows [0, cpuRows_) -> buffer rows [1, cpuRows_].
    std::copy(initial.data(), initial.data() + cpuRows_ * C, cpuCur_.data() + C);
  } else {
    cpuCur_.clear();
    cpuNxt_.clear();
  }
  if (gpuRows_ > 0) {
    ensurePartition();
    partition_->uploadRegion(initial.data() + cpuRows_ * C, gpuRows_, C, wrap_);
  }
}

// ---------------------------------------------------------------------------
// Ghost exchange: fill both sides' ghost rows from the current buffers. Reads the
// current generation; called before each step's compute.
// ---------------------------------------------------------------------------
void HybridEngine::exchangeGhosts() {
  const std::size_t C = cols_;
  unsigned char* cpu = cpuCur_.data();

  if (cpuRows_ > 0 && gpuRows_ > 0) {
    // Seam (always present): exchange the two adjacent boundary rows.
    partition_->readTopRow(rowBuf_.data());                          // GPU first real (board cpuRows_)
    std::copy(rowBuf_.begin(), rowBuf_.end(), cpu + (cpuRows_ + 1) * C); // -> CPU bottom ghost
    partition_->setTopGhost(cpu + cpuRows_ * C);                     // CPU last real -> GPU top ghost
    if (wrap_) {
      partition_->readBottomRow(rowBuf_.data());                     // GPU last real (board R-1)
      std::copy(rowBuf_.begin(), rowBuf_.end(), cpu);                // -> CPU top ghost
      partition_->setBottomGhost(cpu + C);                           // CPU first real (board 0) -> GPU bottom ghost
    }
    // Bounded far-edge ghosts stay zero (set at upload, never written).
  } else if (cpuRows_ > 0) {
    // Pure CPU: the far edges wrap onto the CPU's own rows.
    if (wrap_) {
      std::copy(cpu + cpuRows_ * C, cpu + (cpuRows_ + 1) * C, cpu);              // last real -> top ghost
      std::copy(cpu + C, cpu + 2 * C, cpu + (cpuRows_ + 1) * C);                 // first real -> bottom ghost
    }
  } else if (gpuRows_ > 0) {
    // Pure GPU: the far edges wrap onto the GPU's own rows (host bounce).
    if (wrap_) {
      partition_->readBottomRow(rowBuf_.data()); partition_->setTopGhost(rowBuf_.data());
      partition_->readTopRow(rowBuf_.data()); partition_->setBottomGhost(rowBuf_.data());
    }
  }
}

// ---------------------------------------------------------------------------
// step(): ghost exchange -> launch GPU async -> compute CPU concurrently -> sync.
// ---------------------------------------------------------------------------
void HybridEngine::step() {
  Timer t;
  exchangeGhosts();
  if (gpuRows_ > 0) partition_->launchStep(); // async: returns before the kernel finishes
  if (cpuRows_ > 0) runCpuParallel(cpuCur_.data(), cpuNxt_.data(), cpuRows_);
  if (gpuRows_ > 0) partition_->finishStep(); // wait + swap device buffers
  if (cpuRows_ > 0) std::swap(cpuCur_, cpuNxt_);
  lastMs_ = t.elapsedMillis();
}

void HybridEngine::download(Grid& out) {
  const std::size_t C = cols_;
  if (cpuRows_ > 0) {
    std::copy(cpuCur_.data() + C, cpuCur_.data() + (cpuRows_ + 1) * C, out.data());
  }
  if (gpuRows_ > 0) {
    partition_->downloadRegion(out.data() + cpuRows_ * C);
  }
}

// ---------------------------------------------------------------------------
// CPU worker pool (persistent Barrier pool, mirrors CpuEngine).
// ---------------------------------------------------------------------------
void HybridEngine::startPool() {
  barrier_ = std::make_unique<Barrier>(static_cast<std::ptrdiff_t>(threads_));
  workers_.reserve(threads_ - 1);
  for (unsigned id = 1; id < threads_; ++id) {
    workers_.emplace_back([this, id] { workerLoop(id); });
  }
}

void HybridEngine::workerLoop(unsigned id) {
  while (true) {
    barrier_->arrive_and_wait(); // wait for runCpuParallel to publish work
    if (stop_.load(std::memory_order_relaxed)) return;
    auto [b, e] = bufferRangeFor(id);
    computeCpuRows(poolSrc_, poolDst_, b, e);
    barrier_->arrive_and_wait(); // signal completion
  }
}

std::pair<std::size_t, std::size_t> HybridEngine::bufferRangeFor(unsigned id) const {
  const std::size_t base = poolRows_ / threads_;
  const std::size_t rem = poolRows_ % threads_;
  const std::size_t begin = id * base + std::min<std::size_t>(id, rem);
  const std::size_t count = base + (id < rem ? 1u : 0u);
  // Real rows map to buffer rows [1 + begin, 1 + begin + count).
  return {1 + begin, 1 + begin + count};
}

void HybridEngine::runCpuParallel(const unsigned char* src, unsigned char* dst,
                                  std::size_t nReal) {
  poolSrc_ = src;
  poolDst_ = dst;
  poolRows_ = nReal;
  if (workers_.empty()) {
    computeCpuRows(src, dst, 1, nReal + 1); // sequential: all real rows
  } else {
    barrier_->arrive_and_wait();            // release workers
    auto [b, e] = bufferRangeFor(0);
    computeCpuRows(src, dst, b, e);         // main thread owns partition 0
    barrier_->arrive_and_wait();            // wait for every partition
  }
}

void HybridEngine::computeCpuRows(const unsigned char* src, unsigned char* dst,
                                  std::size_t brBegin, std::size_t brEnd) const {
  const long C = static_cast<long>(cols_);
  for (std::size_t br = brBegin; br < brEnd; ++br) {
    const std::size_t rowOff = br * cols_;
    for (long x = 0; x < C; ++x) {
      int n = 0;
      for (long dy = -1; dy <= 1; ++dy) {
        for (long dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          long nx = x + dx;
          if (wrap_) {
            nx = (nx + C) % C;
          } else if (nx < 0 || nx >= C) {
            continue; // bounded: out-of-range column is dead
          }
          // Vertical neighbour is always a valid buffer row (ghost rows present).
          const long nbrRow = static_cast<long>(br) + dy;
          n += src[static_cast<std::size_t>(nbrRow) * cols_ +
                   static_cast<std::size_t>(nx)];
        }
      }
      dst[rowOff + static_cast<std::size_t>(x)] =
          nextState(src[rowOff + static_cast<std::size_t>(x)], n);
    }
  }
}

} // namespace gol
