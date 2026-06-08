/**
 * @file main.cpp
 * @brief Application entry point: parses config, wires an engine + renderer,
 *        runs the generation loop, and prints the cells/sec metric.
 */

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "gol/Config.hpp"
#include "gol/Grid.hpp"
#include "gol/IRenderer.hpp"
#include "gol/ISimEngine.hpp"
#include "gol/Timer.hpp"
#include "gol/engines/CpuEngine.hpp"
#ifdef GOL_HAVE_CUDA
#include "gol/engines/CudaEngine.hpp"
#endif
#ifdef GOL_HAVE_OPENCL
#include "gol/engines/OpenCLEngine.hpp"
#endif
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"
#ifndef _WIN32
#include "gol/render/AnsiRenderer.hpp"
#endif

#include "gol/render/NullRenderer.hpp"
#include "gol/render/TextRenderer.hpp"

using namespace gol;

namespace {

/**
 * @brief Factory mapping a RendererKind to a concrete renderer.
 * @param kind Which renderer to build.
 * @return Owning pointer to the renderer (defaults to NullRenderer).
 */
  std::unique_ptr<IRenderer> makeRenderer(RendererKind kind) {
    switch (kind) {
      case RendererKind::Text:
        return std::make_unique<TextRenderer>();

#ifndef _WIN32
      case RendererKind::Ansi:
        return std::make_unique<AnsiRenderer>();
#else
      case RendererKind::Ansi:
        return std::make_unique<TextRenderer>();
#endif

      case RendererKind::Null:
      default:
        return std::make_unique<NullRenderer>();
    }
  }

/**
 * @brief Seed the grid either from an RLE pattern (centred) or a deterministic
 *        random fill.
 * @param[in,out] grid Board to seed in place.
 * @param cfg          Run configuration (rlePath, seed, dimensions).
 */
void seed(Grid& grid, const Config& cfg) {
  if (cfg.rlePath) {
    Pattern p = RleLoader::load(*cfg.rlePath);
    const std::size_t ox = cfg.cols > p.width ? (cfg.cols - p.width) / 2 : 0;
    const std::size_t oy = cfg.rows > p.height ? (cfg.rows - p.height) / 2 : 0;
    p.applyTo(grid, ox, oy);
  } else {
    grid.randomize(cfg.seed);
  }
}

/**
 * @brief Factory mapping the configured EngineKind to a concrete ISimEngine.
 * @param cfg Run configuration.
 * @return Owning pointer to the engine, or nullptr if that backend is not built.
 */
std::unique_ptr<ISimEngine> makeEngine(const Config& cfg) {
  switch (cfg.engine) {
    case EngineKind::Cpu:
      return std::make_unique<CpuEngine>(cfg.threads, cfg.wrap);
#ifdef GOL_HAVE_CUDA
    case EngineKind::Cuda:
      return std::make_unique<CudaEngine>(cfg.blockSize, cfg.wrap, cfg.useShared);
#endif
#ifdef GOL_HAVE_OPENCL
    case EngineKind::OpenCL:
      return std::make_unique<OpenCLEngine>(cfg.blockSize, cfg.wrap, cfg.useShared);
#endif
    default:
      return nullptr; // backend not compiled into this build
  }
}

/// @brief Backend-specific config suffix for the report line (threads / block+shared).
std::string backendConfig(const Config& cfg) {
  switch (cfg.engine) {
    case EngineKind::Cpu:
      return "threads=" + std::to_string(cfg.threads);
    case EngineKind::Cuda:
      return "block=" + std::to_string(cfg.blockSize) +
             (cfg.useShared ? " shared" : " no-shared");
    case EngineKind::OpenCL:
      return "block=" + std::to_string(cfg.blockSize) +
             (cfg.useShared ? " local" : " no-local");
    default:
      return "";
  }
}
  std::size_t firstMismatchIndex(const Grid& a, const Grid& b) {
    for (std::size_t y = 0; y < a.rows(); ++y) {
      for (std::size_t x = 0; x < a.cols(); ++x) {
        if (a.at(x, y) != b.at(x, y)) {
          return y * a.cols() + x;
        }
      }
    }
    return a.size();
  }

  int verifyAgainstCpu(const Config& cfg) {
    Grid initial(cfg.rows, cfg.cols);
    seed(initial, cfg);

    Grid cpuOut(cfg.rows, cfg.cols);
    Grid backendOut(cfg.rows, cfg.cols);

    CpuEngine cpu(1, cfg.wrap);
    cpu.upload(initial);

    for (std::uint64_t gen = 0; gen < cfg.generations; ++gen) {
      cpu.step();
    }

    cpu.download(cpuOut);

    auto engine = makeEngine(cfg);
    if (!engine) {
      std::cerr << "error: requested engine is not available in this build.\n";
      return 1;
    }

    engine->upload(initial);

    for (std::uint64_t gen = 0; gen < cfg.generations; ++gen) {
      engine->step();
    }

    engine->download(backendOut);

    if (cpuOut == backendOut) {
      std::cout << "verify: OK — " << engine->name()
                << " matches sequential CPU after "
                << cfg.generations << " generations.\n";
      return 0;
    }

    const std::size_t idx = firstMismatchIndex(cpuOut, backendOut);
    const std::size_t y = idx / cfg.cols;
    const std::size_t x = idx % cfg.cols;

    std::cerr << "verify: FAILED — first mismatch at "
              << "(x=" << x << ", y=" << y << ")\n"
              << "cpu=" << static_cast<int>(cpuOut.at(x, y))
              << " backend=" << static_cast<int>(backendOut.at(x, y))
              << "\n";

    return 1;
  }
/// @brief CSV header line. Keep the columns in sync with the data row in main().
const char* csvHeaderLine() {
  return "backend,rows,cols,generations,threads,block,shared,wrap,"
         "kernel_ms,wall_ms,mcells_kernel,mcells_wall";
}

} // namespace

/**
 * @brief Program entry point.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on error (unsupported engine or a thrown exception).
 */
int main(int argc, char** argv) {
  const Config cfg = parse(argc, argv);
  if (cfg.verify) {
    return verifyAgainstCpu(cfg);
  }
  if (cfg.csvHeader) { // single source of truth for the CSV columns
    std::cout << csvHeaderLine() << "\n";
    return 0;
  }

  try {
    Grid grid(cfg.rows, cfg.cols);
    seed(grid, cfg);

    auto engine = makeEngine(cfg);
    if (!engine) {
      std::cerr << "error: the requested engine is not available in this build "
                   "(rebuild with -DBUILD_CUDA=ON or -DBUILD_OPENCL=ON).\n";
      return 1;
    }
    auto renderer = makeRenderer(cfg.renderer);
    const bool rendering = cfg.renderer != RendererKind::Null;

    Timer totalTimer;

    Timer uploadTimer;
    engine->upload(grid);
    const double uploadMs = uploadTimer.elapsedMillis();

    if (rendering) renderer->render(grid, 0);

    Timer loopTimer;
    double kernelMs = 0.0;
    for (std::uint64_t gen = 0; gen < cfg.generations; ++gen) {
      engine->step();
      kernelMs += engine->lastKernelMillis();

      if (rendering) {
        engine->download(grid);
        renderer->render(grid, gen + 1);
        if (renderer->shouldClose()) break;
      }
    }
    const double wallMs = loopTimer.elapsedMillis();

    double downloadMs = 0.0;
    if (cfg.profile && !rendering) {
      Timer downloadTimer;
      engine->download(grid);
      downloadMs = downloadTimer.elapsedMillis();
    }

    const double totalMs = totalTimer.elapsedMillis();

    // Headline metric: cells evaluated per second = rows * cols * gens / time.
    const double cells = static_cast<double>(cfg.rows) * static_cast<double>(cfg.cols) *
                         static_cast<double>(cfg.generations);
    auto mcellsPerSec = [cells](double ms) {
      return ms > 0.0 ? cells / (ms / 1000.0) / 1e6 : 0.0;
    };
    if (cfg.profile) {
      const double launchAndSyncMs = wallMs - kernelMs;

      std::cout << "profile:    " << engine->name() << " (" << backendConfig(cfg) << ")\n"
                << "grid:       " << cfg.rows << "x" << cfg.cols
                << "  generations=" << cfg.generations
                << "  edges=" << (cfg.wrap ? "toroidal" : "bounded") << "\n"
                << "upload:     " << uploadMs << " ms\n"
                << "steps wall: " << wallMs << " ms\n"
                << "kernels:    " << kernelMs << " ms\n"
                << "overhead:   " << launchAndSyncMs << " ms  (launch/sync inside steps)\n"
                << "download:   " << downloadMs << " ms\n"
                << "total:      " << totalMs << " ms\n";

      return 0;
    }
    if (cfg.csv) {
      // One data row; columns match csvHeaderLine(). booleans as 0/1.
      std::cout << std::setprecision(10)
                << engine->name() << ',' << cfg.rows << ',' << cfg.cols << ','
                << cfg.generations << ',' << cfg.threads << ',' << cfg.blockSize << ','
                << (cfg.useShared ? 1 : 0) << ',' << (cfg.wrap ? 1 : 0) << ','
                << kernelMs << ',' << wallMs << ','
                << mcellsPerSec(kernelMs) << ',' << mcellsPerSec(wallMs) << '\n';
    } else {
      std::cout << "backend:    " << engine->name() << " (" << backendConfig(cfg) << ")\n"
                << "grid:       " << cfg.rows << "x" << cfg.cols
                << "  generations=" << cfg.generations
                << "  edges=" << (cfg.wrap ? "toroidal" : "bounded") << "\n"
                << "kernel:     " << kernelMs << " ms  -> "
                << mcellsPerSec(kernelMs) << " Mcells/s\n"
                << "wall:       " << wallMs << " ms  -> "
                << mcellsPerSec(wallMs) << " Mcells/s\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
