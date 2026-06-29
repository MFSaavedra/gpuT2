/**
 * @file main_gui.cpp
 * @brief Entry point for the Qt + CUDA/OpenGL interop Game of Life viewer.
 *
 * A separate front-end from the headless `gol` benchmark binary: Qt owns the event
 * loop (QApplication::exec), so this cannot reuse main.cpp's generation loop. It
 * requests a 4.3 core GL context, builds the window, and hands control to Qt.
 *
 * Usage: gol_gui [COLSxROWS] [--engine cpu|cuda] [--threads N] [--wrap] [--seed N] [--block N]
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include <QApplication>
#include <QSurfaceFormat>

#include "gol/Config.hpp"
#include "gol/gui/MainWindow.hpp"

namespace {
/// @brief Print usage (flags mirror the headless `gol`, plus the GUI controls) and exit.
[[noreturn]] void usage(int code) {
  std::cout <<
      "Game of Life -- Qt + OpenGL viewer (variant B36/S23)\n\n"
      "Usage: gol_gui [COLSxROWS] [options]\n\n"
      "  COLSxROWS            board-size shorthand, e.g. 1024x768\n"
      "  -r, --rows N         grid height             (default 512)\n"
      "  -c, --cols N         grid width              (default 512)\n"
      "  -g, --gens N         auto-pause at generation N (default: run freely)\n"
      "  -t, --threads N      CPU cores: 1 = sequential, 0 = all (default 0)\n"
      "      --engine NAME    cpu | cuda              (default: cuda if built, else cpu)\n"
      "      --wrap           toroidal edges (default: bounded)\n"
      "      --seed N         RNG seed for random fill (default 1)\n"
      "      --rle PATH       seed from an RLE pattern instead of random\n"
      "      --block N        CUDA threads per block  (default 128)\n"
      "  -h, --help           show this help\n\n"
      "Controls: space play/pause | S step | R reseed | C clear | F fit\n"
      "          wheel zoom | middle/right-drag pan | left-drag paint (Shift erases)\n";
  std::exit(code);
}

/// @brief Parse a non-negative integer or exit with a usage error.
unsigned long long toULL(const std::string& s, const char* flag) {
  try {
    return std::stoull(s);
  } catch (const std::exception&) {
    std::cerr << "error: " << flag << " expects an integer, got '" << s << "'\n";
    usage(2);
  }
}

/// @brief Parse argv into a Config. Mirrors the headless `gol` flags (the GUI runs
///        interactively, so --gens is an optional auto-pause and renderer/csv/verify
///        do not apply).
gol::Config parseArgs(int argc, char** argv) {
  gol::Config cfg;
  cfg.rows = 512;
  cfg.cols = 512;
  cfg.blockSize = 128;
  cfg.threads = 0;       // CPU engine: all hardware cores by default
  cfg.generations = 0;   // 0 = run freely (no auto-pause)
#ifdef GOL_HAVE_CUDA
  cfg.engine = gol::EngineKind::Cuda;
#else
  cfg.engine = gol::EngineKind::Cpu;
#endif

  auto need = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) { std::cerr << "error: " << flag << " requires a value\n"; usage(2); }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(0);
    } else if (a == "-r" || a == "--rows") {
      cfg.rows = static_cast<std::size_t>(toULL(need(i, "--rows"), "--rows"));
    } else if (a == "-c" || a == "--cols") {
      cfg.cols = static_cast<std::size_t>(toULL(need(i, "--cols"), "--cols"));
    } else if (a == "-g" || a == "--gens" || a == "--generations") {
      cfg.generations = toULL(need(i, "--gens"), "--gens");
    } else if (a == "-t" || a == "--threads") {
      cfg.threads = static_cast<unsigned>(toULL(need(i, "--threads"), "--threads"));
    } else if (a == "--wrap") {
      cfg.wrap = true;
    } else if (a == "--seed") {
      cfg.seed = toULL(need(i, "--seed"), "--seed");
    } else if (a == "--rle") {
      cfg.rlePath = need(i, "--rle");
    } else if (a == "--block") {
      cfg.blockSize = static_cast<int>(toULL(need(i, "--block"), "--block"));
    } else if (a == "--engine") {
      const std::string e = need(i, "--engine");
      if (e == "cpu") {
        cfg.engine = gol::EngineKind::Cpu;
      } else if (e == "cuda") {
#ifdef GOL_HAVE_CUDA
        cfg.engine = gol::EngineKind::Cuda;
#else
        std::cerr << "warning: this build has no CUDA engine; using cpu.\n";
        cfg.engine = gol::EngineKind::Cpu;
#endif
      } else {
        std::cerr << "error: unknown engine '" << e << "' (use cpu or cuda)\n";
        usage(2);
      }
    } else if (auto x = a.find('x'); x != std::string::npos &&
               a.find_first_not_of("0123456789x") == std::string::npos) {
      cfg.cols = static_cast<std::size_t>(toULL(a.substr(0, x), "COLSxROWS"));
      cfg.rows = static_cast<std::size_t>(toULL(a.substr(x + 1), "COLSxROWS"));
    } else {
      std::cerr << "error: unknown option '" << a << "'\n";
      usage(2);
    }
  }
  return cfg;
}
} // namespace

int main(int argc, char** argv) {
  // Request a core-profile 3.3 context before any GL widget is created. The viewer
  // uses nothing past GL 3.3 (integer textures, texelFetch, VAOs), so 3.3 maximises
  // the range of drivers/hardware it runs on.
  QSurfaceFormat fmt;
  fmt.setVersion(3, 3);
  fmt.setProfile(QSurfaceFormat::CoreProfile);
  fmt.setSwapInterval(1); // vsync
  QSurfaceFormat::setDefaultFormat(fmt);

  QApplication app(argc, argv);
  const gol::Config cfg = parseArgs(argc, argv);

  MainWindow window(cfg);
  window.show();
  return app.exec();
}
