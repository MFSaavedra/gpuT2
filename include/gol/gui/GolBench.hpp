#pragma once

#include "gol/Config.hpp"

/**
 * @file GolBench.hpp
 * @brief Headless throughput benchmark for gol_gui's present paths (no window, no vsync).
 *
 * The interactive viewer is refresh-bound: a QOpenGLWidget swaps buffers under vsync
 * (~60 Hz), so it can only present ~60 boards/s regardless of how fast the GPU computes
 * and moves them into the texture. This bench measures the present paths uncapped: it
 * creates an *offscreen* GL context (no window, no swapBuffers) and times the same
 * per-frame work the widget does, back to back.
 *
 * It reports, as Gcells/s:
 *   - kernel-only        : sum of ISimEngine::lastKernelMillis() (matches the headless
 *                          `gol` --csv kernel column: cudaEvent, excludes everything else);
 *   - step wall          : chrono around N step()s (adds launch overhead, no present);
 *   - interop present     : N x (step + CUDA D2D into the PBO + PBO->texture), the
 *                          zero-copy display path, decoupled from the display refresh;
 *   - host-upload present : N x (step + download to host + texture upload from client
 *                          memory), the PCIe round trip the interop path avoids.
 */

namespace gol {

/**
 * @brief Run the headless present-path benchmark and print the results to stdout.
 *
 * Requires a QGuiApplication/QApplication to already exist (for the offscreen GL
 * context). For the zero-copy interop path the GL context must land on the same GPU
 * as CUDA (run via scripts/run_gui.sh's PRIME-offload env on this Optimus box); if
 * interop registration fails it falls back to reporting host-upload only.
 *
 * @param cfg    Run configuration (grid size, engine, block size, wrap, seed, RLE).
 * @param iters  Number of timed steps per phase.
 * @param warmup Untimed warmup steps before each timed phase.
 * @return Process exit code (0 on success).
 */
int runGuiBench(const Config& cfg, long long iters, long long warmup);

} // namespace gol
