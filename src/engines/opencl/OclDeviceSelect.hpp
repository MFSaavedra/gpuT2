#pragma once

/**
 * @file OclDeviceSelect.hpp
 * @brief Shared OpenCL device-selection helper for the OpenCL engine and the
 *        OpenCL halo partition (both in the gated gol_opencl library).
 *
 * The default policy (empty hint) is the historical one: the first GPU on the
 * first platform, then a CPU device as a fallback. A nonempty @p hint restricts
 * the choice to a device whose NAME or VENDOR contains that substring
 * (case-insensitive) --- e.g. "intel" for the integrated GPU, "nvidia" for the
 * discrete card. The environment variable GOL_OCL_DEVICE overrides the hint, so a
 * run can be pinned to a device without recompiling (this is what makes an
 * iGPU-vs-dGPU measurement airtight). Mirrors the selectIGPU idiom from the
 * Barlas mandelbrot_hybrid reference (Chapter 11 load-balancing).
 */

#include <CL/cl.h>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gol {
namespace ocl {

/// @brief Read a string device-info field (CL_DEVICE_NAME / CL_DEVICE_VENDOR).
inline std::string deviceInfoString(cl_device_id device, cl_device_info field) {
  size_t size = 0;
  clGetDeviceInfo(device, field, 0, nullptr, &size);
  std::string s(size, '\0');
  if (size > 0) clGetDeviceInfo(device, field, size, s.data(), nullptr);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

inline std::string deviceName(cl_device_id device) {
  return deviceInfoString(device, CL_DEVICE_NAME);
}

inline std::string deviceVendor(cl_device_id device) {
  return deviceInfoString(device, CL_DEVICE_VENDOR);
}

/// @brief Case-insensitive substring test; an empty needle matches anything.
inline bool containsCI(const std::string& haystack, const std::string& needle) {
  if (needle.empty()) return true;
  auto lower = [](std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
  };
  return lower(haystack).find(lower(needle)) != std::string::npos;
}

/**
 * @brief Choose an OpenCL platform+device.
 * @param hint Name/vendor substring to require ("" = any). GOL_OCL_DEVICE overrides.
 * @return {platform, device}. GPU devices are preferred over CPU.
 * @throws std::runtime_error if no platform exists, or a nonempty selector matches
 *         nothing (the message lists every device that was seen).
 */
inline std::pair<cl_platform_id, cl_device_id> chooseDevice(const std::string& hint = "") {
  const char* env = std::getenv("GOL_OCL_DEVICE");
  const std::string want = (env && *env) ? std::string(env) : hint;

  cl_uint numPlatforms = 0;
  clGetPlatformIDs(0, nullptr, &numPlatforms);
  if (numPlatforms == 0) throw std::runtime_error("OpenCL: no platforms found");
  std::vector<cl_platform_id> platforms(numPlatforms);
  clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

  // Enumerate GPU devices first (preferred), then CPU. Record every device so a
  // failed selection can report what was actually available.
  std::vector<std::pair<cl_platform_id, cl_device_id>> seen;
  for (cl_device_type type : {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU}) {
    for (cl_platform_id platform : platforms) {
      cl_uint numDevices = 0;
      if (clGetDeviceIDs(platform, type, 0, nullptr, &numDevices) != CL_SUCCESS ||
          numDevices == 0)
        continue;
      std::vector<cl_device_id> devices(numDevices);
      clGetDeviceIDs(platform, type, numDevices, devices.data(), nullptr);
      for (cl_device_id device : devices) {
        if (!want.empty() &&
            (containsCI(deviceName(device), want) || containsCI(deviceVendor(device), want)))
          return {platform, device};
        seen.emplace_back(platform, device);
      }
    }
  }

  if (!want.empty()) {
    std::string msg = "OpenCL: no device matching '" + want + "'. Devices seen:";
    for (const auto& [p, d] : seen) msg += "\n  - " + deviceName(d) + " [" + deviceVendor(d) + "]";
    if (seen.empty()) msg += " (none)";
    throw std::runtime_error(msg);
  }
  if (seen.empty()) throw std::runtime_error("OpenCL: no GPU or CPU devices found");
  return seen.front(); // default: first GPU (GPUs enumerated before CPUs)
}

} // namespace ocl
} // namespace gol
