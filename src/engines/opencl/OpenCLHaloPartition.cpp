/**
 * @file OpenCLHaloPartition.cpp
 * @brief OpenCL implementation of IHaloPartition: a device-resident block of board
 *        rows with a one-row ghost halo, driven by HybridEngine.
 *
 * Mirrors CudaHaloPartition: owns two `(realRows + 2) * cols` device buffers and
 * ping-pongs them on the device. launchStep() enqueues the life_halo kernel on an
 * in-order, profiling-enabled queue without blocking so the host can compute the
 * CPU partition concurrently; finishStep() waits, reads the CL profiling event for
 * the kernel time, and swaps. The kernel source is the same kernel.cl read at
 * runtime by OpenCLEngine.
 */

#include "gol/engines/opencl/OpenCLHaloPartition.hpp"

#include <CL/cl.h>

#include "OclDeviceSelect.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef GOL_OPENCL_KERNEL_PATH
#define GOL_OPENCL_KERNEL_PATH "src/engines/opencl/kernel.cl"
#endif

namespace gol {

namespace {

void clCheck(cl_int err, const char* what) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error(std::string("OpenCL error in ") + what +
                             ": code " + std::to_string(err));
  }
}

std::string readTextFile(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("OpenCL: cannot open kernel file: " + path);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

std::string getBuildLog(cl_program program, cl_device_id device) {
  size_t logSize = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
  std::string log(logSize, '\0');
  if (logSize > 0) {
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize,
                          log.data(), nullptr);
  }
  return log;
}

size_t roundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

/// @brief OpenCL-backed halo partition (see IHaloPartition for the contract).
class OpenCLHaloPartition final : public IHaloPartition {
public:
  OpenCLHaloPartition(int blockSize, std::string deviceHint)
      : blockSize_(blockSize < 32 ? 32 : blockSize),
        deviceHint_(std::move(deviceHint)) {
    cl_int err = CL_SUCCESS;
    auto [platform, device] = gol::ocl::chooseDevice(deviceHint_);
    (void)platform;
    device_ = device;
    std::cerr << "[opencl] halo partition on: " << gol::ocl::deviceName(device)
              << " [" << gol::ocl::deviceVendor(device) << "]\n";

    context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    clCheck(err, "clCreateContext");
    queue_ = clCreateCommandQueue(static_cast<cl_context>(context_), device,
                                  CL_QUEUE_PROFILING_ENABLE, &err);
    clCheck(err, "clCreateCommandQueue");

    const std::string source = readTextFile(GOL_OPENCL_KERNEL_PATH);
    const char* src = source.c_str();
    const size_t len = source.size();
    cl_program program = clCreateProgramWithSource(
        static_cast<cl_context>(context_), 1, &src, &len, &err);
    clCheck(err, "clCreateProgramWithSource");
    program_ = program;

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("OpenCL halo kernel build failed on device '" +
                               gol::ocl::deviceName(device) + "':\n" +
                               getBuildLog(program, device));
    }
    kernel_ = clCreateKernel(program, "life_halo", &err);
    clCheck(err, "clCreateKernel(life_halo)");
  }

  ~OpenCLHaloPartition() override {
    if (dCur_) clReleaseMemObject(static_cast<cl_mem>(dCur_));
    if (dNxt_) clReleaseMemObject(static_cast<cl_mem>(dNxt_));
    if (kernel_) clReleaseKernel(static_cast<cl_kernel>(kernel_));
    if (program_) clReleaseProgram(static_cast<cl_program>(program_));
    if (queue_) clReleaseCommandQueue(static_cast<cl_command_queue>(queue_));
    if (context_) clReleaseContext(static_cast<cl_context>(context_));
  }

  void uploadRegion(const unsigned char* region, std::size_t realRows,
                    std::size_t cols, bool wrap) override {
    if (realRows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("OpenCL: partition dimensions exceed int range");
    }
    realRows_ = realRows;
    cols_ = cols;
    wrap_ = wrap;
    const std::size_t bufBytes = (realRows + 2) * cols;

    cl_context context = static_cast<cl_context>(context_);
    cl_command_queue queue = static_cast<cl_command_queue>(queue_);
    if (dCur_) { clReleaseMemObject(static_cast<cl_mem>(dCur_)); dCur_ = nullptr; }
    if (dNxt_) { clReleaseMemObject(static_cast<cl_mem>(dNxt_)); dNxt_ = nullptr; }

    cl_int err = CL_SUCCESS;
    cl_mem cur = clCreateBuffer(context, CL_MEM_READ_WRITE, bufBytes, nullptr, &err);
    clCheck(err, "clCreateBuffer(cur)");
    cl_mem nxt = clCreateBuffer(context, CL_MEM_READ_WRITE, bufBytes, nullptr, &err);
    clCheck(err, "clCreateBuffer(nxt)");

    // Zero both buffers so bounded far-edge ghosts stay dead for the whole run.
    const unsigned char zero = 0;
    clCheck(clEnqueueFillBuffer(queue, cur, &zero, 1, 0, bufBytes, 0, nullptr, nullptr),
            "clEnqueueFillBuffer(cur)");
    clCheck(clEnqueueFillBuffer(queue, nxt, &zero, 1, 0, bufBytes, 0, nullptr, nullptr),
            "clEnqueueFillBuffer(nxt)");
    // Copy the real rows into [1, realRows] of the current buffer (offset = cols).
    clCheck(clEnqueueWriteBuffer(queue, cur, CL_TRUE, cols, realRows * cols, region,
                                 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(region)");
    dCur_ = cur;
    dNxt_ = nxt;
  }

  void setTopGhost(const unsigned char* row) override {
    clCheck(clEnqueueWriteBuffer(static_cast<cl_command_queue>(queue_),
                                 static_cast<cl_mem>(dCur_), CL_TRUE, 0, cols_, row,
                                 0, nullptr, nullptr),
            "clEnqueueWriteBuffer(top ghost)");
  }

  void setBottomGhost(const unsigned char* row) override {
    clCheck(clEnqueueWriteBuffer(static_cast<cl_command_queue>(queue_),
                                 static_cast<cl_mem>(dCur_), CL_TRUE,
                                 (realRows_ + 1) * cols_, cols_, row, 0, nullptr,
                                 nullptr),
            "clEnqueueWriteBuffer(bottom ghost)");
  }

  void launchStep() override {
    const size_t localX = 32;
    const size_t localY = static_cast<size_t>((blockSize_ + 31) / 32);
    const size_t global[2] = {roundUp(cols_, localX), roundUp(realRows_, localY)};
    const size_t local[2] = {localX, localY};

    cl_kernel kernel = static_cast<cl_kernel>(kernel_);
    cl_mem cur = static_cast<cl_mem>(dCur_);
    cl_mem nxt = static_cast<cl_mem>(dNxt_);
    int realRows = static_cast<int>(realRows_);
    int cols = static_cast<int>(cols_);
    int wrap = wrap_ ? 1 : 0;

    clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &cur), "clSetKernelArg(src)");
    clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &nxt), "clSetKernelArg(dst)");
    clCheck(clSetKernelArg(kernel, 2, sizeof(int), &realRows), "clSetKernelArg(real_rows)");
    clCheck(clSetKernelArg(kernel, 3, sizeof(int), &cols), "clSetKernelArg(cols)");
    clCheck(clSetKernelArg(kernel, 4, sizeof(int), &wrap), "clSetKernelArg(wrap)");

    event_ = nullptr;
    cl_event ev = nullptr;
    cl_command_queue queue = static_cast<cl_command_queue>(queue_);
    clCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0,
                                   nullptr, &ev),
            "clEnqueueNDRangeKernel(life_halo)");
    event_ = ev; // not waited here -> host overlaps with the device
    // Flush so the device starts now (in-order queue would otherwise only begin
    // at the finishStep() wait, serialising the GPU after the CPU slice).
    clCheck(clFlush(queue), "clFlush(launch)");
  }

  void finishStep() override {
    cl_event ev = static_cast<cl_event>(event_);
    clCheck(clWaitForEvents(1, &ev), "clWaitForEvents(kernel)");
    cl_ulong start = 0, end = 0;
    clCheck(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start),
                                    &start, nullptr),
            "clGetEventProfilingInfo(start)");
    clCheck(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end,
                                    nullptr),
            "clGetEventProfilingInfo(end)");
    lastMs_ = static_cast<double>(end - start) * 1e-6;
    clReleaseEvent(ev);
    event_ = nullptr;
    std::swap(dCur_, dNxt_);
  }

  void readTopRow(unsigned char* out) override {
    clCheck(clEnqueueReadBuffer(static_cast<cl_command_queue>(queue_),
                                static_cast<cl_mem>(dCur_), CL_TRUE, cols_, cols_,
                                out, 0, nullptr, nullptr),
            "clEnqueueReadBuffer(top row)");
  }

  void readBottomRow(unsigned char* out) override {
    clCheck(clEnqueueReadBuffer(static_cast<cl_command_queue>(queue_),
                                static_cast<cl_mem>(dCur_), CL_TRUE,
                                realRows_ * cols_, cols_, out, 0, nullptr, nullptr),
            "clEnqueueReadBuffer(bottom row)");
  }

  void downloadRegion(unsigned char* out) override {
    clCheck(clEnqueueReadBuffer(static_cast<cl_command_queue>(queue_),
                                static_cast<cl_mem>(dCur_), CL_TRUE, cols_,
                                realRows_ * cols_, out, 0, nullptr, nullptr),
            "clEnqueueReadBuffer(region)");
  }

  double lastKernelMillis() const override { return lastMs_; }
  std::string name() const override { return "opencl"; }

private:
  int blockSize_;
  std::string deviceHint_;
  bool wrap_ = false;
  std::size_t realRows_ = 0;
  std::size_t cols_ = 0;

  void* device_ = nullptr;
  void* context_ = nullptr;
  void* queue_ = nullptr;
  void* program_ = nullptr;
  void* kernel_ = nullptr;
  void* dCur_ = nullptr;
  void* dNxt_ = nullptr;
  void* event_ = nullptr; ///< In-flight kernel event between launchStep/finishStep.
  double lastMs_ = 0.0;
};

} // namespace

std::unique_ptr<IHaloPartition> makeOpenCLHaloPartition(int blockSize,
                                                       std::string deviceHint) {
  return std::make_unique<OpenCLHaloPartition>(blockSize, std::move(deviceHint));
}

} // namespace gol
