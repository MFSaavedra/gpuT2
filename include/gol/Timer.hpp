#pragma once

#include <chrono>

/**
 * @file Timer.hpp
 * @brief Minimal monotonic stopwatch for host-side timing.
 */

namespace gol {

/**
 * @brief Minimal monotonic stopwatch for host-side timing (the CPU engine and harness).
 *
 * GPU engines time their kernels with backend-native events instead. Built on
 * std::chrono::steady_clock so it cannot run backwards if the system wall clock
 * is adjusted mid-run.
 */
class Timer {
public:
  /// @brief Construct the timer and start it from the current instant.
  Timer() : start_(Clock::now()) {}

  /// @brief Restart the stopwatch from now.
  void reset() { start_ = Clock::now(); }

  /**
   * @brief Elapsed time since construction or the last reset().
   * @return Elapsed time in fractional milliseconds.
   */
  double elapsedMillis() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

private:
  using Clock = std::chrono::steady_clock; ///< Monotonic clock source.
  Clock::time_point start_;                ///< Instant of construction/last reset.
};

} // namespace gol
