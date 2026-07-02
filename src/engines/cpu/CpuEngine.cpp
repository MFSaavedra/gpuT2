/**
 * @file CpuEngine.cpp
 * @brief Implementation of the CPU engine (sequential baseline + data-parallel
 *        worker pool), the reference oracle for the GPU backends.
 */

#include "gol/engines/CpuEngine.hpp"

#include <algorithm>

#include "gol/LifeRules.hpp"
#include "gol/Timer.hpp"

namespace gol {

unsigned CpuEngine::resolveThreads(unsigned requested) {
  if (requested != 0) return requested;
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1u : hw; // hardware_concurrency may report 0 if unknown
}

CpuEngine::CpuEngine(unsigned threads, bool wrap)
    : threads_(resolveThreads(threads)), wrap_(wrap) {
  if (threads_ >= 2) startPool();
}

CpuEngine::~CpuEngine() {
  if (!workers_.empty()) {
    // Release the workers one last time; they observe stop_ and return without
    // arriving at the completion barrier, so the main thread does not wait.
    stop_.store(true, std::memory_order_relaxed);
    barrier_->arrive_and_wait();
    for (auto& w : workers_) w.join();
  }
}

void CpuEngine::startPool() {
  // barrier participants = main thread + (threads_ - 1) workers = threads_.
  barrier_ = std::make_unique<Barrier>(static_cast<std::ptrdiff_t>(threads_));
  workers_.reserve(threads_ - 1);
  for (unsigned id = 1; id < threads_; ++id) {
    workers_.emplace_back([this, id] { workerLoop(id); });
  }
}

void CpuEngine::workerLoop(unsigned id) {
  while (true) {
    barrier_->arrive_and_wait();                 // wait for step() to publish work
    if (stop_.load(std::memory_order_relaxed)) return;
    auto [yBegin, yEnd] = rangeFor(id);
    computeRows(src_, dst_, yBegin, yEnd);
    barrier_->arrive_and_wait();                 // signal completion to step()
  }
}

std::pair<std::size_t, std::size_t> CpuEngine::rangeFor(unsigned id) const {
  const std::size_t base = rows_ / threads_;
  const std::size_t rem = rows_ % threads_;
  const std::size_t begin = id * base + std::min<std::size_t>(id, rem);
  const std::size_t count = base + (id < rem ? 1u : 0u);
  return {begin, begin + count};
}

int CpuEngine::countNeighbors(const unsigned char* g, std::size_t x,
                              std::size_t y) const {
  int n = 0;
  if (wrap_) {
    // Toroidal: indices wrap around the grid edges.
    const long cols = static_cast<long>(cols_);
    const long rows = static_cast<long>(rows_);
    for (long dy = -1; dy <= 1; ++dy) {
      for (long dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        const std::size_t nx = static_cast<std::size_t>((static_cast<long>(x) + dx + cols) % cols);
        const std::size_t ny = static_cast<std::size_t>((static_cast<long>(y) + dy + rows) % rows);
        n += g[ny * cols_ + nx];
      }
    }
  } else {
    // Bounded: out-of-range neighbours count as dead.
    for (long dy = -1; dy <= 1; ++dy) {
      for (long dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        const long nx = static_cast<long>(x) + dx;
        const long ny = static_cast<long>(y) + dy;
        if (nx < 0 || ny < 0 || nx >= static_cast<long>(cols_) ||
            ny >= static_cast<long>(rows_))
          continue;
        n += g[static_cast<std::size_t>(ny) * cols_ + static_cast<std::size_t>(nx)];
      }
    }
  }
  return n;
}

void CpuEngine::computeRows(const unsigned char* src, unsigned char* dst,
                            std::size_t yBegin, std::size_t yEnd) const {
  for (std::size_t y = yBegin; y < yEnd; ++y) {
    const std::size_t row = y * cols_;
    for (std::size_t x = 0; x < cols_; ++x) {
      dst[row + x] = nextState(src[row + x], countNeighbors(src, x, y));
    }
  }
}

void CpuEngine::upload(const Grid& initial) {
  rows_ = initial.rows();
  cols_ = initial.cols();
  cur_.assign(initial.data(), initial.data() + initial.size());
  nxt_.assign(cur_.size(), 0u);
}

void CpuEngine::step() {
  Timer t;
  if (workers_.empty()) {
    // Sequential baseline: a single full sweep, no synchronisation. This is the
    // parallel path with exactly one partition (rangeFor(0) == [0, rows_)).
    computeRows(cur_.data(), nxt_.data(), 0, rows_);
  } else {
    src_ = cur_.data();
    dst_ = nxt_.data();
    barrier_->arrive_and_wait();                 // release workers onto their rows
    auto [yBegin, yEnd] = rangeFor(0);
    computeRows(src_, dst_, yBegin, yEnd);        // main thread owns partition 0
    barrier_->arrive_and_wait();                 // wait for every partition to finish
  }
  std::swap(cur_, nxt_);
  lastMs_ = t.elapsedMillis();
}

void CpuEngine::download(Grid& out) {
  // CPU works in host memory already; this is just a copy into the caller's Grid
  // (which must have matching dimensions).
  std::copy(cur_.begin(), cur_.end(), out.data());
}

void CpuEngine::pokeCell(std::size_t x, std::size_t y, unsigned char value) {
  if (x >= cols_ || y >= rows_) return;
  cur_[y * cols_ + x] = value; // in-place edit of the live buffer
}

} // namespace gol
