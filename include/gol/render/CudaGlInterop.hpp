#pragma once

#include <cstddef>

/**
 * @file CudaGlInterop.hpp
 * @brief CUDA <-> OpenGL interop bridge (declaration).
 *
 * Lets the GPU-computed board be displayed without a host round trip: an OpenGL
 * Pixel Buffer Object (PBO) is registered with CUDA once, then each frame the
 * current device buffer is copied device->device straight into the PBO's memory,
 * which OpenGL streams into a texture. No cudaMemcpy ever crosses PCIe.
 *
 * Deliberately free of any CUDA/GL headers so the Qt widget (compiled by the host
 * compiler) can use it; the cudaGraphicsResource handle is stored as an opaque
 * pointer and the implementation lives in CudaGlInterop.cu (compiled by nvcc).
 */

namespace gol {

/**
 * @brief Owns the registration of one GL buffer with CUDA and the per-frame copy.
 */
class CudaGlInterop {
public:
  CudaGlInterop() = default;
  ~CudaGlInterop();

  CudaGlInterop(const CudaGlInterop&) = delete;            ///< Non-copyable (owns a CUDA resource).
  CudaGlInterop& operator=(const CudaGlInterop&) = delete; ///< Non-copyable.

  /**
   * @brief Register an OpenGL buffer (PBO) so CUDA can write into it.
   *
   * Any previously registered buffer is released first. The GL buffer must
   * already be sized to at least @p bytes and a GL context must be current.
   * @param glBufferId OpenGL buffer name (GLuint) to register.
   * @param bytes      Number of bytes copied per frame (the board size).
   * @throws std::runtime_error on a CUDA error.
   */
  void registerBuffer(unsigned int glBufferId, std::size_t bytes);

  /**
   * @brief Copy @p deviceSrc into the registered PBO (map -> D2D memcpy -> unmap).
   * @param deviceSrc CUDA device pointer to the current board buffer.
   * @throws std::runtime_error if no buffer is registered or on a CUDA error.
   */
  void copyFromDevice(const void* deviceSrc);

  /// @brief Release the registered buffer, if any (also called by the destructor).
  void unregister();

private:
  void* resource_ = nullptr; ///< cudaGraphicsResource* (opaque), like CudaEngine's events.
  std::size_t bytes_ = 0;    ///< Bytes to copy per frame.
};

} // namespace gol
