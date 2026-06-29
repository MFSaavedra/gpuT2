/**
 * @file CudaEngine.cu
 * @brief Implementation of the CUDA backend.
 *
 * Owns two device buffers, does the ping-pong on the device, and times the
 * kernel region with cudaEvent_t (excluding host<->device transfers) so
 * lastKernelMillis() is comparable to the CPU's chrono timing.
 */

#include "gol/engines/CudaEngine.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "gol/Grid.hpp"
#include "gol/engines/cuda/kernel.cuh"

namespace gol {

namespace {
/// @brief Throw a std::runtime_error if @p e is not cudaSuccess.
inline void cudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                             cudaGetErrorString(e));
  }
}
} // namespace

CudaEngine::CudaEngine(int blockSize, bool wrap, bool useShared)
    : blockSize_(blockSize), wrap_(wrap), useShared_(useShared) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaCheck(cudaEventCreate(&start), "cudaEventCreate(start)");
  cudaCheck(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  evStart_ = start;
  evStop_ = stop;
}

CudaEngine::~CudaEngine() {
  // Destructor must not throw; ignore errors during teardown.
  if (dCur_) cudaFree(dCur_);
  if (dNxt_) cudaFree(dNxt_);
  if (evStart_) cudaEventDestroy(static_cast<cudaEvent_t>(evStart_));
  if (evStop_) cudaEventDestroy(static_cast<cudaEvent_t>(evStop_));
}

void CudaEngine::upload(const Grid& initial) {
  rows_ = initial.rows();
  cols_ = initial.cols();
  bytes_ = initial.size();
  if (dCur_) { cudaFree(dCur_); dCur_ = nullptr; }
  if (dNxt_) { cudaFree(dNxt_); dNxt_ = nullptr; }
  cudaCheck(cudaMalloc(&dCur_, bytes_), "cudaMalloc(cur)");
  cudaCheck(cudaMalloc(&dNxt_, bytes_), "cudaMalloc(nxt)");
  cudaCheck(cudaMemcpy(dCur_, initial.data(), bytes_, cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D upload)");
}

void CudaEngine::step() {
  cudaEvent_t start = static_cast<cudaEvent_t>(evStart_);
  cudaEvent_t stop = static_cast<cudaEvent_t>(evStop_);
  cudaCheck(cudaEventRecord(start), "cudaEventRecord(start)");
  launchLifeStep(dCur_, dNxt_, rows_, cols_, wrap_, blockSize_, useShared_);
  cudaCheck(cudaEventRecord(stop), "cudaEventRecord(stop)");
  cudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  lastMs_ = static_cast<double>(ms);
  std::swap(dCur_, dNxt_); // ping-pong on the device
}

void CudaEngine::download(Grid& out) {
  cudaCheck(cudaMemcpy(out.data(), dCur_, bytes_, cudaMemcpyDeviceToHost),
            "cudaMemcpy(D2H download)");
}

void CudaEngine::pokeCell(std::size_t x, std::size_t y, unsigned char value) {
  if (!dCur_ || x >= cols_ || y >= rows_) return;
  cudaCheck(cudaMemcpy(dCur_ + y * cols_ + x, &value, 1, cudaMemcpyHostToDevice),
            "cudaMemcpy(pokeCell)");
}

} // namespace gol
