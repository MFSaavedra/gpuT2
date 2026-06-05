#pragma once

#include <string>

#include "gol/Grid.hpp"

/**
 * @file ISimEngine.hpp
 * @brief Strategy interface for a simulation backend (CPU / CUDA / OpenCL).
 */

namespace gol {

/**
 * @brief Strategy interface for a simulation backend (CPU / CUDA / OpenCL).
 *
 * This is the only thing that genuinely differs per backend; everything else in
 * the core is shared. The Application owns the loop and wires one ISimEngine +
 * one IRenderer at runtime.
 *
 * Ping-pong contract: an engine keeps two equally-sized buffers internally.
 * Each step() reads the "current" buffer and writes the "next" buffer, then
 * swaps them, so step() advances exactly one generation and never aliases its
 * input and output. GPU engines do this swap on the device; the host Grid is
 * only touched by upload()/download().
 *
 * Typical loop:
 * @code
 *   engine->upload(grid);
 *   for (gen = 0; gen < N; ++gen) {
 *     engine->step();
 *     engine->download(grid);   // no-op cost on CPU
 *     renderer->render(grid, gen);
 *   }
 * @endcode
 */
class ISimEngine {
public:
  virtual ~ISimEngine() = default;

  /**
   * @brief Copy an initial host Grid into the engine's "current" device buffer
   *        and size the ping-pong pair.
   *
   * Must be called before the first step().
   * @param initial Seed board to upload.
   */
  virtual void upload(const Grid& initial) = 0;

  /**
   * @brief Advance the simulation by exactly one generation (read current,
   *        write next, swap).
   *
   * The variant rule and edge handling must match every other backend.
   */
  virtual void step() = 0;

  /**
   * @brief Copy the engine's current buffer back into @p out.
   *
   * @p out must already have matching dimensions. A no-op-cost copy for the CPU
   * engine, which works in-place.
   * @param[out] out Destination grid (dimensions must match the uploaded board).
   */
  virtual void download(Grid& out) = 0;

  /**
   * @brief Wall time of the most recent step()'s kernel/compute region.
   *
   * Measured with backend-native event timing (chrono / cudaEvent / CL events),
   * excluding host<->device transfers. Drives the cells/sec metric.
   * @return Compute time of the last step() in milliseconds.
   */
  virtual double lastKernelMillis() const = 0;

  /**
   * @brief Human-readable backend name for logs and CSV columns.
   * @return Backend name, e.g. "cpu", "cuda".
   */
  virtual std::string name() const = 0;
};

} // namespace gol
