/**
 * @file CudaGlInterop.cu
 * @brief Implementation of the CUDA <-> OpenGL interop bridge.
 *
 * Compiled by nvcc (it needs cuda_gl_interop.h). The header it implements is
 * CUDA-free, so the Qt widget can call it from a host translation unit.
 */

#include "gol/render/CudaGlInterop.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace gol {

namespace {
/// @brief Throw a std::runtime_error if @p e is not cudaSuccess.
inline void cudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA-GL interop error in ") + what +
                             ": " + cudaGetErrorString(e));
  }
}
} // namespace

CudaGlInterop::~CudaGlInterop() {
  // Destructor must not throw; ignore errors during teardown.
  if (resource_) {
    cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(resource_));
    resource_ = nullptr;
  }
}

void CudaGlInterop::registerBuffer(unsigned int glBufferId, std::size_t bytes) {
  unregister();
  cudaGraphicsResource_t res = nullptr;
  // WriteDiscard: CUDA fully overwrites the buffer each frame, so the driver need
  // not preserve its previous contents.
  const cudaError_t e = cudaGraphicsGLRegisterBuffer(
      &res, glBufferId, cudaGraphicsRegisterFlagsWriteDiscard);
  if (e != cudaSuccess) {
    // Registration can legitimately fail when the current GL context is on a
    // different GPU than CUDA (e.g. an Optimus iGPU context), in which case the
    // caller falls back to host-upload. The failed call has recorded the error in
    // the runtime's per-thread "last error" slot, though; clear it here (it is a
    // non-sticky validation error, so the context stays usable) so that a later
    // engine kernel launch's cudaGetLastError() check does not inherit it and
    // throw a spurious "kernel launch failed".
    cudaGetLastError();
    throw std::runtime_error(
        std::string("CUDA-GL interop error in cudaGraphicsGLRegisterBuffer: ") +
        cudaGetErrorString(e));
  }
  resource_ = res;
  bytes_ = bytes;
}

void CudaGlInterop::copyFromDevice(const void* deviceSrc) {
  if (!resource_) {
    throw std::runtime_error("CudaGlInterop::copyFromDevice: no registered buffer");
  }
  auto res = static_cast<cudaGraphicsResource_t>(resource_);

  cudaCheck(cudaGraphicsMapResources(1, &res, /*stream=*/0),
            "cudaGraphicsMapResources");

  void* dPtr = nullptr;
  std::size_t mappedBytes = 0;
  cudaCheck(cudaGraphicsResourceGetMappedPointer(&dPtr, &mappedBytes, res),
            "cudaGraphicsResourceGetMappedPointer");

  const std::size_t n = mappedBytes < bytes_ ? mappedBytes : bytes_;
  cudaCheck(cudaMemcpy(dPtr, deviceSrc, n, cudaMemcpyDeviceToDevice),
            "cudaMemcpy(D2D into PBO)");

  cudaCheck(cudaGraphicsUnmapResources(1, &res, /*stream=*/0),
            "cudaGraphicsUnmapResources");
}

void CudaGlInterop::unregister() {
  if (resource_) {
    cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(resource_));
    resource_ = nullptr;
  }
  bytes_ = 0;
}

} // namespace gol
