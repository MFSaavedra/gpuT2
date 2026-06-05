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
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"
#include "gol/render/AnsiRenderer.hpp"
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
    case RendererKind::Text: return std::make_unique<TextRenderer>();
    case RendererKind::Ansi: return std::make_unique<AnsiRenderer>();
    case RendererKind::Null:
    default: return std::make_unique<NullRenderer>();
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
    default:
      return "";
  }
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

    engine->upload(grid);

    // Generation 0 is the seed itself: render it before the first step so the
    // displayed frames are seed, step 1, step 2, ... with honest labels. The
    // benchmark path (NullRenderer) skips this, so timing is unaffected.
    if (rendering) renderer->render(grid, 0);

    Timer wall;
    double kernelMs = 0.0;
    for (std::uint64_t gen = 0; gen < cfg.generations; ++gen) {
      engine->step();
      kernelMs += engine->lastKernelMillis();
      if (rendering) {
        engine->download(grid);
        renderer->render(grid, gen + 1); // state after `gen + 1` steps
        if (renderer->shouldClose()) break;
      }
    }
    const double wallMs = wall.elapsedMillis();

    // Headline metric: cells evaluated per second = rows * cols * gens / time.
    const double cells = static_cast<double>(cfg.rows) * static_cast<double>(cfg.cols) *
                         static_cast<double>(cfg.generations);
    auto mcellsPerSec = [cells](double ms) {
      return ms > 0.0 ? cells / (ms / 1000.0) / 1e6 : 0.0;
    };

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
