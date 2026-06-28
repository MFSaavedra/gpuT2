#pragma once

#include <cstddef>
#include <string>

/**
 * @file IHaloPartition.hpp
 * @brief Strategy interface for a GPU sub-domain ("partition") that owns a
 *        contiguous block of board rows plus a one-row ghost halo above and
 *        below it.
 *
 * Used by HybridEngine to drive a CUDA or OpenCL device for its share of a
 * row-wise domain decomposition (Divisible Load Theory, Barlas chapter 11.3).
 * Deliberately free of any CUDA/OpenCL headers/types so HybridEngine.cpp (host
 * C++23) can hold a std::unique_ptr<IHaloPartition> without pulling in a toolkit;
 * the concrete partitions live in the gated gol_cuda / gol_opencl libraries.
 *
 * Buffer layout: a partition owning @c realRows board rows keeps device buffers
 * of height `realRows + 2`. Buffer row 0 is the TOP ghost, rows [1, realRows] are
 * the real rows (board row = bufferRow - 1 within the partition), and buffer row
 * `realRows + 1` is the BOTTOM ghost. The step kernel reads vertical neighbours
 * straight from the ghost rows (no vertical edge logic) and applies only the
 * horizontal (column) edge rule, so the host fills the ghosts each generation to
 * realise either bounded or toroidal edges. This makes the device result
 * bit-for-bit identical to the CpuEngine oracle in both edge modes.
 *
 * Ping-pong: the partition holds two device buffers and a notion of "current"
 * (the one the next step reads). All ghost set/row read operations act on the
 * current buffer; finishStep() swaps current and next.
 */

namespace gol {

/**
 * @brief A GPU-side partition of the board with a one-row ghost halo.
 */
class IHaloPartition {
public:
  virtual ~IHaloPartition() = default;

  /**
   * @brief Allocate `(realRows + 2) x cols` device buffers (both ping-pong
   *        halves), zero them, and copy @p region into the real rows of the
   *        current buffer.
   *
   * Zeroing both buffers means the bounded far-edge ghosts (never written by the
   * kernel) stay dead for the lifetime of the run with no per-step work.
   * @param region   Host pointer to `realRows * cols` bytes (the real rows only).
   * @param realRows Number of real board rows owned by this partition.
   * @param cols     Board width (full width; the decomposition is row-wise only).
   * @param wrap     Horizontal edge mode: false = bounded, true = toroidal.
   */
  virtual void uploadRegion(const unsigned char* region, std::size_t realRows,
                            std::size_t cols, bool wrap) = 0;

  /**
   * @brief Copy one @c cols-byte row into the current buffer's TOP ghost (row 0).
   * @param row Host pointer to @c cols bytes.
   */
  virtual void setTopGhost(const unsigned char* row) = 0;

  /**
   * @brief Copy one @c cols-byte row into the current buffer's BOTTOM ghost
   *        (row realRows + 1).
   * @param row Host pointer to @c cols bytes.
   */
  virtual void setBottomGhost(const unsigned char* row) = 0;

  /**
   * @brief Enqueue one generation of the halo kernel (current -> next) WITHOUT
   *        synchronising, and start the kernel timer.
   *
   * Returning before the kernel completes is what lets the host compute the CPU
   * partition concurrently. Pair every launchStep() with a finishStep().
   */
  virtual void launchStep() = 0;

  /**
   * @brief Wait for the most recent launchStep() to finish, record its kernel
   *        time, and swap the current/next device buffers.
   */
  virtual void finishStep() = 0;

  /**
   * @brief Copy the FIRST real row (buffer row 1) of the current buffer to host.
   * @param[out] out Host pointer to at least @c cols bytes.
   */
  virtual void readTopRow(unsigned char* out) = 0;

  /**
   * @brief Copy the LAST real row (buffer row realRows) of the current buffer to host.
   * @param[out] out Host pointer to at least @c cols bytes.
   */
  virtual void readBottomRow(unsigned char* out) = 0;

  /**
   * @brief Copy all real rows [1, realRows] of the current buffer to host.
   * @param[out] out Host pointer to at least `realRows * cols` bytes.
   */
  virtual void downloadRegion(unsigned char* out) = 0;

  /**
   * @brief Kernel time of the most recent finishStep(), in milliseconds.
   * @return Device kernel time (excludes host<->device ghost transfers).
   */
  virtual double lastKernelMillis() const = 0;

  /**
   * @brief Human-readable backend name (e.g. "cuda", "opencl").
   * @return Backend name of the underlying device.
   */
  virtual std::string name() const = 0;
};

} // namespace gol
