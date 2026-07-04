#include "gol/engines/OpenCLEngine.hpp"

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

#include "gol/Grid.hpp"

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
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          logSize, log.data(), nullptr);
  }

  return log;
}

// First GPU on the first platform (then CPU) by default, but honours the
// GOL_OCL_DEVICE environment variable so `--engine opencl` can be pinned to the
// iGPU or the discrete card for isolated measurement (see OclDeviceSelect.hpp).
std::pair<cl_platform_id, cl_device_id> chooseDevice() {
  return gol::ocl::chooseDevice();
}

std::string deviceName(cl_device_id device) {
  size_t size = 0;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size);

  std::string name(size, '\0');
  if (size > 0) {
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr);
  }

  while (!name.empty() && name.back() == '\0') {
    name.pop_back();
  }

  return name;
}

size_t roundUp(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

OpenCLEngine::OpenCLEngine(int blockSize, bool wrap, bool useShared)
    : blockSize_(blockSize), wrap_(wrap), useShared_(useShared) {
  cl_int err = CL_SUCCESS;

  auto [platform, device] = chooseDevice();
  platform_ = platform;
  device_ = device;
  std::cerr << "[opencl] engine on: " << deviceName(device) << "\n";

  cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  clCheck(err, "clCreateContext");
  context_ = context;

  cl_command_queue queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  clCheck(err, "clCreateCommandQueue");
  queue_ = queue;

  const std::string source = readTextFile(GOL_OPENCL_KERNEL_PATH);
  const char* src = source.c_str();
  const size_t len = source.size();

  cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &err);
  clCheck(err, "clCreateProgramWithSource");
  program_ = program;

  err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    const std::string log = getBuildLog(program, device);
    throw std::runtime_error("OpenCL kernel build failed on device '" +
                             deviceName(device) + "':\n" + log);
  }

  cl_kernel kernelGlobal = clCreateKernel(program, "life_global", &err);
  clCheck(err, "clCreateKernel(life_global)");
  kernelGlobal_ = kernelGlobal;
  cl_kernel kernelLocal = clCreateKernel(program, "life_local", &err);
  clCheck(err, "clCreateKernel(life_local)");
  kernelLocal_ = kernelLocal;
}

OpenCLEngine::~OpenCLEngine() {
  if (dCur_) clReleaseMemObject(static_cast<cl_mem>(dCur_));
  if (dNxt_) clReleaseMemObject(static_cast<cl_mem>(dNxt_));

  if (kernelGlobal_) clReleaseKernel(static_cast<cl_kernel>(kernelGlobal_));
  if (kernelLocal_) clReleaseKernel(static_cast<cl_kernel>(kernelLocal_));
  if (program_) clReleaseProgram(static_cast<cl_program>(program_));
  if (queue_) clReleaseCommandQueue(static_cast<cl_command_queue>(queue_));
  if (context_) clReleaseContext(static_cast<cl_context>(context_));
}

void OpenCLEngine::upload(const Grid& initial) {
  rows_ = initial.rows();
  cols_ = initial.cols();
  bytes_ = initial.size() * sizeof(unsigned char);

  if (rows_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      cols_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("OpenCL: grid dimensions exceed int range");
  }

  cl_context context = static_cast<cl_context>(context_);
  cl_command_queue queue = static_cast<cl_command_queue>(queue_);

  if (dCur_) {
    clReleaseMemObject(static_cast<cl_mem>(dCur_));
    dCur_ = nullptr;
  }

  if (dNxt_) {
    clReleaseMemObject(static_cast<cl_mem>(dNxt_));
    dNxt_ = nullptr;
  }

  cl_int err = CL_SUCCESS;

  cl_mem cur = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_, nullptr, &err);
  clCheck(err, "clCreateBuffer(cur)");

  cl_mem nxt = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes_, nullptr, &err);
  clCheck(err, "clCreateBuffer(nxt)");

  clCheck(clEnqueueWriteBuffer(queue, cur, CL_TRUE, 0, bytes_,
                               initial.data(), 0, nullptr, nullptr),
          "clEnqueueWriteBuffer(upload)");

  dCur_ = cur;
  dNxt_ = nxt;
}

void OpenCLEngine::step() {

  if (!dCur_ || !dNxt_) {
    throw std::runtime_error("OpenCL: upload() must be called before step()");
  }
  const size_t localX = 32;
  const size_t localY = static_cast<size_t>((blockSize_ + 31) / 32);

  const size_t globalX = roundUp(cols_, localX);
  const size_t globalY = roundUp(rows_, localY);

  const size_t global[2] = {globalX, globalY};
  const size_t local[2] = {localX, localY};

  cl_command_queue queue = static_cast<cl_command_queue>(queue_);
cl_kernel kernel = static_cast<cl_kernel>(useShared_ ? kernelLocal_ : kernelGlobal_);
  cl_mem cur = static_cast<cl_mem>(dCur_);
  cl_mem nxt = static_cast<cl_mem>(dNxt_);

  int rows = static_cast<int>(rows_);
  int cols = static_cast<int>(cols_);
  int wrap = wrap_ ? 1 : 0;

  clCheck(clSetKernelArg(kernel, 0, sizeof(cl_mem), &cur), "clSetKernelArg(src)");
  clCheck(clSetKernelArg(kernel, 1, sizeof(cl_mem), &nxt), "clSetKernelArg(dst)");
  clCheck(clSetKernelArg(kernel, 2, sizeof(int), &rows), "clSetKernelArg(rows)");
  clCheck(clSetKernelArg(kernel, 3, sizeof(int), &cols), "clSetKernelArg(cols)");
  clCheck(clSetKernelArg(kernel, 4, sizeof(int), &wrap), "clSetKernelArg(wrap)");
  if (useShared_) {
    const size_t tileBytes = (localX + 2) * (localY + 2) * sizeof(unsigned char);
    clCheck(clSetKernelArg(kernel, 5, tileBytes, nullptr),
            "clSetKernelArg(local tile)");
  }

  if (blockSize_ < 32) {
    blockSize_ = 32;
  }


  cl_event event = nullptr;
  clCheck(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
                                 global, local, 0, nullptr, &event),
          "clEnqueueNDRangeKernel(life_global)");

  clCheck(clWaitForEvents(1, &event), "clWaitForEvents(kernel)");

  cl_ulong start = 0;
  cl_ulong end = 0;

  clCheck(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                  sizeof(start), &start, nullptr),
          "clGetEventProfilingInfo(start)");

  clCheck(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                  sizeof(end), &end, nullptr),
          "clGetEventProfilingInfo(end)");

  lastMs_ = static_cast<double>(end - start) * 1e-6;

  clReleaseEvent(event);

  std::swap(dCur_, dNxt_);
}

void OpenCLEngine::download(Grid& out) {
  if (out.rows() != rows_ || out.cols() != cols_) {
    throw std::runtime_error("OpenCL: download grid has incompatible dimensions");
  }

  cl_command_queue queue = static_cast<cl_command_queue>(queue_);
  cl_mem cur = static_cast<cl_mem>(dCur_);

  clCheck(clEnqueueReadBuffer(queue, cur, CL_TRUE, 0, bytes_,
                              out.data(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(download)");
}

} // namespace gol