#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "gol/Config.hpp"
#include "gol/Grid.hpp"
#include "gol/IRenderer.hpp"
#include "gol/Timer.hpp"
#include "gol/engines/CpuEngine.hpp"
#include "gol/patterns/Pattern.hpp"
#include "gol/patterns/RleLoader.hpp"
#include "gol/render/NullRenderer.hpp"
#include "gol/render/TextRenderer.hpp"

using namespace gol;

namespace {

std::unique_ptr<IRenderer> makeRenderer(RendererKind kind) {
  switch (kind) {
    case RendererKind::Text: return std::make_unique<TextRenderer>();
    case RendererKind::Null:
    default: return std::make_unique<NullRenderer>();
  }
}

// Seed the grid either from an RLE pattern (centred) or a deterministic random fill.
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

} // namespace

int main(int argc, char** argv) {
  const Config cfg = parse(argc, argv);

  if (cfg.engine != EngineKind::Cpu) {
    std::cerr << "error: only the CPU engine is implemented so far.\n";
    return 1;
  }

  try {
    Grid grid(cfg.rows, cfg.cols);
    seed(grid, cfg);

    CpuEngine engine(cfg.threads, cfg.wrap);
    auto renderer = makeRenderer(cfg.renderer);
    const bool rendering = cfg.renderer != RendererKind::Null;

    engine.upload(grid);

    // Generation 0 is the seed itself: render it before the first step so the
    // displayed frames are seed, step 1, step 2, ... with honest labels. The
    // benchmark path (NullRenderer) skips this, so timing is unaffected.
    if (rendering) renderer->render(grid, 0);

    Timer wall;
    double kernelMs = 0.0;
    for (std::uint64_t gen = 0; gen < cfg.generations; ++gen) {
      engine.step();
      kernelMs += engine.lastKernelMillis();
      if (rendering) {
        engine.download(grid);
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

    std::cout << "backend:    " << engine.name() << " (threads=" << engine.threads() << ")\n"
              << "grid:       " << cfg.rows << "x" << cfg.cols
              << "  generations=" << cfg.generations
              << "  edges=" << (cfg.wrap ? "toroidal" : "bounded") << "\n"
              << "kernel:     " << kernelMs << " ms  -> "
              << mcellsPerSec(kernelMs) << " Mcells/s\n"
              << "wall:       " << wallMs << " ms  -> "
              << mcellsPerSec(wallMs) << " Mcells/s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
