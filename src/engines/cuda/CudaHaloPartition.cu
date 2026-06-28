/**
 * @file CudaHaloPartition.cu
 * @brief CUDA implementation of IHaloPartition: a device-resident block of board
 *        rows with a one-row ghost halo, driven by HybridEngine.
 *
 * Owns two `(realRows + 2) * cols` device buffers and ping-pongs them on the
 * device. launchStep() enqueues the halo kernel without synchronising so the host
 * can compute the CPU partition concurrently; finishStep() waits, records the
 * cudaEvent kernel time, and swaps. Ghost rows and boundary rows move one row at a
 * time (cols bytes) -- the only per-generation host<->device traffic.
 */

#include "gol/engines/cuda/CudaHaloPartition.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "gol/engines/cuda/kernel.cuh"

namespace gol {

namespace {

inline void cudaCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

/// @brief CUDA-backed halo partition (see IHaloPartition for the contract).
class CudaHaloPartition final : public IHaloPartition {
public:
  explicit CudaHaloPartition(int blockSize) : blockSize_(blockSize) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "cudaEventCreate(start)");
    cudaCheck(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    evStart_ = start;
    evStop_ = stop;
  }

  ~CudaHaloPartition() override {
    if (dCur_) cudaFree(dCur_);
    if (dNxt_) cudaFree(dNxt_);
    if (evStart_) cudaEventDestroy(static_cast<cudaEvent_t>(evStart_));
    if (evStop_) cudaEventDestroy(static_cast<cudaEvent_t>(evStop_));
  }

  void uploadRegion(const unsigned char* region, std::size_t realRows,
                    std::size_t cols, bool wrap) override {
    realRows_ = realRows;
    cols_ = cols;
    wrap_ = wrap;
    const std::size_t bufBytes = (realRows + 2) * cols;
    if (dCur_) { cudaFree(dCur_); dCur_ = nullptr; }
    if (dNxt_) { cudaFree(dNxt_); dNxt_ = nullptr; }
    cudaCheck(cudaMalloc(&dCur_, bufBytes), "cudaMalloc(cur)");
    cudaCheck(cudaMalloc(&dNxt_, bufBytes), "cudaMalloc(nxt)");
    // Zero both buffers so the bounded far-edge ghosts (never written by the
    // kernel) stay dead for the whole run with no per-step work.
    cudaCheck(cudaMemset(dCur_, 0, bufBytes), "cudaMemset(cur)");
    cudaCheck(cudaMemset(dNxt_, 0, bufBytes), "cudaMemset(nxt)");
    // Copy the real rows into [1, realRows] of the current buffer.
    cudaCheck(cudaMemcpy(dCur_ + cols, region, realRows * cols,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(H2D region)");
  }

  void setTopGhost(const unsigned char* row) override {
    cudaCheck(cudaMemcpy(dCur_, row, cols_, cudaMemcpyHostToDevice),
              "cudaMemcpy(H2D top ghost)");
  }

  void setBottomGhost(const unsigned char* row) override {
    cudaCheck(cudaMemcpy(dCur_ + (realRows_ + 1) * cols_, row, cols_,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(H2D bottom ghost)");
  }

  void launchStep() override {
    cudaCheck(cudaEventRecord(static_cast<cudaEvent_t>(evStart_)),
              "cudaEventRecord(start)");
    launchLifeHaloStep(dCur_, dNxt_, realRows_, cols_, wrap_, blockSize_);
    cudaCheck(cudaEventRecord(static_cast<cudaEvent_t>(evStop_)),
              "cudaEventRecord(stop)");
  }

  void finishStep() override {
    auto stop = static_cast<cudaEvent_t>(evStop_);
    cudaCheck(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(evStart_), stop),
              "cudaEventElapsedTime");
    lastMs_ = static_cast<double>(ms);
    std::swap(dCur_, dNxt_); // ping-pong on the device
  }

  void readTopRow(unsigned char* out) override {
    cudaCheck(cudaMemcpy(out, dCur_ + cols_, cols_, cudaMemcpyDeviceToHost),
              "cudaMemcpy(D2H top row)");
  }

  void readBottomRow(unsigned char* out) override {
    cudaCheck(cudaMemcpy(out, dCur_ + realRows_ * cols_, cols_,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(D2H bottom row)");
  }

  void downloadRegion(unsigned char* out) override {
    cudaCheck(cudaMemcpy(out, dCur_ + cols_, realRows_ * cols_,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy(D2H region)");
  }

  double lastKernelMillis() const override { return lastMs_; }
  std::string name() const override { return "cuda"; }

private:
  int blockSize_;
  bool wrap_ = false;
  std::size_t realRows_ = 0;
  std::size_t cols_ = 0;
  unsigned char* dCur_ = nullptr; ///< Device current buffer ((realRows+2)*cols).
  unsigned char* dNxt_ = nullptr; ///< Device next buffer.
  void* evStart_ = nullptr;       ///< cudaEvent_t marking kernel start (opaque).
  void* evStop_ = nullptr;        ///< cudaEvent_t marking kernel stop (opaque).
  double lastMs_ = 0.0;
};

} // namespace

std::unique_ptr<IHaloPartition> makeCudaHaloPartition(int blockSize) {
  return std::make_unique<CudaHaloPartition>(blockSize);
}

} // namespace gol
