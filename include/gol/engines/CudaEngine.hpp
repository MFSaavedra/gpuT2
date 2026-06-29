#pragma once

#include <cstddef>
#include <string>

#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"

/**
 * @file CudaEngine.hpp
 * @brief CUDA simulation backend (declaration).
 *
 * Deliberately free of any CUDA headers/types so it can be included by plain C++
 * translation units (e.g. main.cpp compiled with the host compiler). Device
 * handles are stored as opaque pointers and cast inside CudaEngine.cu.
 */

namespace gol {

/**
 * @brief GPU backend using CUDA.
 *
 * One thread per cell over a 2D grid of blocks. Owns two device buffers and does
 * the ping-pong on the device; the host Grid is only touched by upload() and
 * download(). The variant rule is shared verbatim with the CPU backend via
 * LifeRules.hpp (compiled `__host__ __device__` under nvcc), and edge handling
 * (bounded / toroidal) matches CpuEngine bit-for-bit.
 *
 * Two kernel-config knobs are swept in the report and selected at runtime:
 * the block size (mapped onto a 32-wide 2D tile for coalesced loads) and whether
 * to stage a tile + halo in shared memory.
 */
class CudaEngine final : public ISimEngine {
public:
  /**
   * @brief Construct the engine and create the timing events.
   * @param blockSize Threads per block ({32,64,128,256}); mapped to a 32 x (N/32) tile.
   * @param wrap      false = bounded edges, true = toroidal. Must match other backends.
   * @param useShared true = shared-memory tiled kernel, false = plain global-memory kernel.
   */
  explicit CudaEngine(int blockSize = 256, bool wrap = false, bool useShared = false);
  ~CudaEngine() override;

  CudaEngine(const CudaEngine&) = delete;            ///< Non-copyable (owns device memory/events).
  CudaEngine& operator=(const CudaEngine&) = delete; ///< Non-copyable.
  CudaEngine(CudaEngine&&) = delete;                 ///< Non-movable.
  CudaEngine& operator=(CudaEngine&&) = delete;      ///< Non-movable.

  void upload(const Grid& initial) override;   ///< @copydoc ISimEngine::upload
  void step() override;                          ///< @copydoc ISimEngine::step
  void download(Grid& out) override;             ///< @copydoc ISimEngine::download
  double lastKernelMillis() const override { return lastMs_; } ///< Kernel-only time (cudaEvent).
  std::string name() const override { return "cuda"; }         ///< @copydoc ISimEngine::name

  int blockSize() const { return blockSize_; } ///< @return Threads per block in use.
  bool useShared() const { return useShared_; } ///< @return Whether the shared-memory kernel is used.

  // -- Interop accessors (used by the Qt/OpenGL viewer; no-op for the headless app) --
  std::size_t rows() const { return rows_; } ///< @return Board height set by upload().
  std::size_t cols() const { return cols_; } ///< @return Board width set by upload().
  /**
   * @brief Opaque device pointer to the current-generation buffer.
   *
   * Points into CUDA device memory (the same buffer step() writes). The viewer
   * does a device->device copy from here into a GL-registered PBO, so the board
   * is displayed without ever crossing PCIe to the host. Returned as void* to
   * keep this header free of CUDA types.
   * @return Device pointer to the current buffer, or nullptr before upload().
   */
  const void* currentDeviceBuffer() const { return dCur_; }
  /**
   * @brief Set a single cell in the current device buffer (interactive editing).
   *
   * A 1-byte host->device write into the live buffer; out-of-range coordinates
   * are ignored. Used by the viewer's mouse paint tool.
   * @param x     Column, [0, cols()).
   * @param y     Row, [0, rows()).
   * @param value New cell value (0 = dead, 1 = alive).
   */
  void pokeCell(std::size_t x, std::size_t y, unsigned char value) override;

private:
  int blockSize_;        ///< Threads per block.
  bool wrap_;            ///< Edge mode: true = toroidal, false = bounded.
  bool useShared_;       ///< Use the shared-memory tiled kernel.
  std::size_t rows_ = 0; ///< Board height (set by upload()).
  std::size_t cols_ = 0; ///< Board width (set by upload()).
  std::size_t bytes_ = 0;///< Device buffer size in bytes (rows_*cols_).
  unsigned char* dCur_ = nullptr; ///< Device "current" buffer (read by step).
  unsigned char* dNxt_ = nullptr; ///< Device "next" buffer (written by step).
  void* evStart_ = nullptr;       ///< cudaEvent_t marking the kernel start (opaque).
  void* evStop_ = nullptr;        ///< cudaEvent_t marking the kernel stop (opaque).
  double lastMs_ = 0.0;           ///< Kernel time of the last step(), in ms.
};

} // namespace gol
