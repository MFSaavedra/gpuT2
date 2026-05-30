#pragma once

#include <chrono>

namespace gol {

// Minimal monotonic stopwatch for host-side timing (the CPU engine and harness).
// GPU engines time their kernels with backend-native events instead.
class Timer {
public:
  Timer() : start_(Clock::now()) {}

  // Restart the stopwatch from now.
  void reset() { start_ = Clock::now(); }

  // Elapsed time since construction/reset, in fractional milliseconds.
  double elapsedMillis() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

} // namespace gol
