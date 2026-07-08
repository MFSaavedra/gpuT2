/**
 * @file GolBench.cpp
 * @brief Implementation of the headless present-path throughput benchmark.
 *
 * Mirrors the per-frame work of GolGlWidget::paintGL (the interop D2D-into-PBO copy
 * and the host-upload download+texture upload), but against an offscreen GL context
 * with no swapBuffers, so nothing is vsync/refresh-bound.
 */

#include "gol/gui/GolBench.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions_3_3_Core>
#include <QSurfaceFormat>

#include "gol/Grid.hpp"
#include "gol/ISimEngine.hpp"
#include "gol/engines/CpuEngine.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"

#ifdef GOL_HAVE_CUDA
#include "gol/engines/CudaEngine.hpp"
#include "gol/render/CudaGlInterop.hpp"
#endif

namespace gol {
namespace {

using Clock = std::chrono::steady_clock;

/// @brief Cells/sec (in Gcells) for @p iters passes over @p cells in @p elapsedMs.
double gcellsPerSec(std::size_t cells, long long iters, double elapsedMs) {
  if (elapsedMs <= 0.0) return 0.0;
  const double total = static_cast<double>(cells) * static_cast<double>(iters);
  return total / (elapsedMs / 1000.0) / 1e9;
}

/// @brief Seed @p grid from the configured RLE (centred) or a deterministic random fill.
void seedGrid(Grid& grid, const Config& cfg) {
  grid.fill(0);
  if (cfg.rlePath) {
    try {
      Pattern p = RleLoader::load(*cfg.rlePath);
      const std::size_t ox = cfg.cols > p.width ? (cfg.cols - p.width) / 2 : 0;
      const std::size_t oy = cfg.rows > p.height ? (cfg.rows - p.height) / 2 : 0;
      p.applyTo(grid, ox, oy);
      return;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[gol_gui bench] could not load RLE '%s' (%s); using random seed\n",
                   cfg.rlePath->c_str(), e.what());
    }
  }
  grid.randomize(cfg.seed);
}

} // namespace

int runGuiBench(const Config& cfg, long long iters, long long warmup) {
  if (iters <= 0) return 0;
  if (warmup < 0) warmup = 0;

  // Offscreen GL context: same 3.3 core default format set in main, but no window and
  // no swapBuffers, so nothing here is capped by the display refresh.
  QOffscreenSurface surface;
  surface.setFormat(QSurfaceFormat::defaultFormat());
  surface.create();
  if (!surface.isValid()) {
    std::fprintf(stderr, "[gol_gui bench] could not create an offscreen surface\n");
    return 1;
  }

  QOpenGLContext ctx;
  ctx.setFormat(QSurfaceFormat::defaultFormat());
  if (!ctx.create() || !ctx.makeCurrent(&surface)) {
    std::fprintf(stderr, "[gol_gui bench] could not create/make-current a GL context\n");
    return 1;
  }

  QOpenGLFunctions_3_3_Core gl;
  if (!gl.initializeOpenGLFunctions()) {
    std::fprintf(stderr, "[gol_gui bench] could not load GL 3.3 core functions\n");
    return 1;
  }

  const char* vendor = reinterpret_cast<const char*>(gl.glGetString(GL_VENDOR));
  const char* renderer = reinterpret_cast<const char*>(gl.glGetString(GL_RENDERER));

  Grid grid(cfg.rows, cfg.cols);
  seedGrid(grid, cfg);
  const std::size_t cells = grid.size();
  const auto bytes = static_cast<GLsizeiptr>(cells);

  // GL resources: PBO (interop copy destination) + R8UI texture (upload destination),
  // identical to GolGlWidget::initializeGL.
  GLuint pbo = 0, tex = 0;
  gl.glGenBuffers(1, &pbo);
  gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  gl.glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, nullptr, GL_DYNAMIC_COPY);
  gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  gl.glGenTextures(1, &tex);
  gl.glBindTexture(GL_TEXTURE_2D, tex);
  gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, static_cast<GLsizei>(cfg.cols),
                  static_cast<GLsizei>(cfg.rows), 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
  gl.glBindTexture(GL_TEXTURE_2D, 0);

  // Build the engine (CUDA if requested and available, else CPU) and upload the seed.
  std::unique_ptr<ISimEngine> engine;
#ifdef GOL_HAVE_CUDA
  CudaEngine* cudaEngine = nullptr;
#endif
  try {
#ifdef GOL_HAVE_CUDA
    if (cfg.engine == EngineKind::Cuda) {
      auto e = std::make_unique<CudaEngine>(cfg.blockSize, cfg.wrap, /*useShared=*/false);
      cudaEngine = e.get();
      engine = std::move(e);
    } else
#endif
    {
      engine = std::make_unique<CpuEngine>(cfg.threads, cfg.wrap);
    }
    engine->upload(grid);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[gol_gui bench] engine unavailable: %s\n", e.what());
    return 1;
  }

  // Try to register the PBO for zero-copy interop (CUDA only; needs the GL context on
  // the same GPU as CUDA). On failure, fall back to reporting host-upload only.
  bool interopOk = false;
#ifdef GOL_HAVE_CUDA
  CudaGlInterop interop;
  if (cudaEngine) {
    try {
      interop.registerBuffer(pbo, static_cast<std::size_t>(bytes));
      interopOk = true;
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "[gol_gui bench] interop unavailable on '%s' (%s); host-upload only.\n"
                   "                For the zero-copy path, launch via scripts/run_gui.sh (NVIDIA GL context).\n",
                   renderer ? renderer : "unknown", e.what());
    }
  }
#endif

  // --- per-phase work (one iteration each) -------------------------------------
#ifdef GOL_HAVE_CUDA
  auto interopPresent = [&] {
    gl.glBindTexture(GL_TEXTURE_2D, tex);
    interop.copyFromDevice(cudaEngine->currentDeviceBuffer());  // device->device into PBO
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(cfg.cols),
                       static_cast<GLsizei>(cfg.rows), GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    engine->step();
  };
#endif
  auto hostUploadPresent = [&] {
    engine->download(grid);
    gl.glBindTexture(GL_TEXTURE_2D, tex);
    gl.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);  // source from client memory, not a PBO
    gl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(cfg.cols),
                       static_cast<GLsizei>(cfg.rows), GL_RED_INTEGER, GL_UNSIGNED_BYTE, grid.data());
    engine->step();
  };

  // Time N repetitions of @p body (after @p warmup untimed ones), returning ms elapsed.
  auto timePhase = [&](auto&& body) {
    engine->upload(grid);
    for (long long i = 0; i < warmup; ++i) body();
    gl.glFinish();
    const auto t0 = Clock::now();
    for (long long i = 0; i < iters; ++i) body();
    gl.glFinish();  // drain GL so the wall covers the whole present, not just the queue
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  };

  // --- report header -----------------------------------------------------------
  const char* pathName = interopOk ? "zero-copy interop" : "host-upload";
  std::printf("\n[gol_gui bench] %s / %s (offscreen, no vsync)\n", engine->name().c_str(), pathName);
  std::printf("  GL_VENDOR / RENDERER : %s / %s\n", vendor ? vendor : "?", renderer ? renderer : "?");
  std::printf("  board                : %zu x %zu  (%.2f Mcells)\n", cfg.cols, cfg.rows, cells / 1e6);
  std::printf("  warmup / iters       : %lld / %lld\n\n", warmup, iters);
  std::printf("  %-20s %12s %14s\n", "phase", "ms/iter", "Gcells/s");

  // --- kernel-only + step wall (same loop) -------------------------------------
  engine->upload(grid);
  for (long long i = 0; i < warmup; ++i) engine->step();
  double kernelMsSum = 0.0;
  const auto s0 = Clock::now();
  for (long long i = 0; i < iters; ++i) {
    engine->step();
    kernelMsSum += engine->lastKernelMillis();
  }
  const double stepWallMs = std::chrono::duration<double, std::milli>(Clock::now() - s0).count();
  std::printf("  %-20s %12.4f %14.3f\n", "kernel-only", kernelMsSum / iters,
              gcellsPerSec(cells, iters, kernelMsSum));
  std::printf("  %-20s %12.4f %14.3f\n", "step wall", stepWallMs / iters,
              gcellsPerSec(cells, iters, stepWallMs));

  // --- interop present (zero-copy) ---------------------------------------------
#ifdef GOL_HAVE_CUDA
  if (interopOk) {
    const double ms = timePhase(interopPresent);
    std::printf("  %-20s %12.4f %14.3f   (zero-copy, no PCIe)\n", "interop present",
                ms / iters, gcellsPerSec(cells, iters, ms));
  }
#endif

  // --- host-upload present (PCIe download each frame) --------------------------
  {
    const double ms = timePhase(hostUploadPresent);
    std::printf("  %-20s %12.4f %14.3f   (PCIe download each frame)\n", "host-upload present",
                ms / iters, gcellsPerSec(cells, iters, ms));
  }

  std::printf("\n  For reference, a 60 FPS vsync display would cap presentation at %.4f Gcells/s.\n\n",
              60.0 * static_cast<double>(cells) / 1e9);

  // --- cleanup (GL context still current) --------------------------------------
#ifdef GOL_HAVE_CUDA
  interop.unregister();
#endif
  engine.reset();
  gl.glDeleteBuffers(1, &pbo);
  gl.glDeleteTextures(1, &tex);
  ctx.doneCurrent();
  return 0;
}

} // namespace gol
