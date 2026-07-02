#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>

/**
 * @file Barrier.hpp
 * @brief A minimal reusable (cyclic) thread barrier.
 *
 * Drop-in replacement for the single std::barrier<> use in CpuEngine, provided so
 * the project builds under C++17 (std::barrier is a C++20 library feature). Only
 * the operation the engine needs -- arrive_and_wait() with a fixed participant
 * count -- is implemented; the semantics match std::barrier<>'s for that use.
 */

namespace gol {

/**
 * @brief Reusable barrier: N participants each call arrive_and_wait(); none return
 *        until all N have arrived, then the barrier rearms for the next phase.
 *
 * Implemented with a generation counter so it can be reused across generations
 * without reconstruction. The last thread to arrive advances the generation (which
 * releases the waiters) and resets the arrival count for the following phase.
 */
class Barrier {
public:
  /// @param count Number of participating threads (must be >= 1).
  explicit Barrier(std::ptrdiff_t count)
      : threshold_(count), count_(count), generation_(0) {}

  Barrier(const Barrier&) = delete;
  Barrier& operator=(const Barrier&) = delete;

  /**
   * @brief Block until all @p count participants have arrived at this phase.
   *
   * The last arriver rearms the counter, advances the generation, and wakes
   * everyone; earlier arrivers wait until the generation they captured on entry no
   * longer matches (i.e. the gate has opened).
   */
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::size_t gen = generation_;
    if (--count_ == 0) {
      ++generation_;       // open the gate for this phase
      count_ = threshold_; // rearm for the next phase
      lock.unlock();
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  const std::ptrdiff_t threshold_; ///< Participant count (fixed for the barrier's life).
  std::ptrdiff_t count_;           ///< Remaining arrivals in the current phase.
  std::size_t generation_;         ///< Phase counter; advances when the gate opens.
};

} // namespace gol
