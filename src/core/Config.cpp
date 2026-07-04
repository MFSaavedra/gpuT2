/**
 * @file Config.cpp
 * @brief Implementation of the argv -> Config parser.
 */

#include "gol/Config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace gol {

namespace {

/**
 * @brief Print the usage/help text and terminate the process.
 * @param code Process exit code (0 for `--help`, 2 for a usage error).
 */
[[noreturn]] void usageAndExit(int code) {
  std::cout <<
      "Conway's Game of Life (variant: born on 3 OR 6, survive on 2 or 3)\n"
      "\n"
      "Usage: gol [options]\n"
      "\n"
      "  -r, --rows N         grid height           (default 256)\n"
      "  -c, --cols N         grid width            (default 256)\n"
      "  -g, --gens N         generations to run    (default 100)\n"
      "  -t, --threads N      CPU cores: 1 = sequential, N = parallel,\n"
      "                       0 = all hardware cores (default 1)\n"
      "      --wrap           toroidal edges (default: bounded)\n"
      "      --seed N         RNG seed for random fill (default 1)\n"
      "      --rle PATH       seed from an RLE pattern instead of random\n"
      "      --engine NAME    cpu | cuda | opencl | hybrid  (default cpu)\n"
      "      --renderer NAME  null | text | ansi     (default null)\n"
      "                       text = scrolling ASCII dump; ansi = in-place animation\n"
      "      --block N        GPU threads per block  (default 256)\n"
      "      --shared         GPU shared-memory tiling\n"
      "      --cpu-frac F     hybrid: fixed CPU row fraction in [0,1]\n"
      "                       (default: auto-calibrate the split)\n"
      "      --gpu-backend N  hybrid: auto | cuda | opencl  (default auto; legacy)\n"
      "      --calib-steps N  hybrid: calibration steps per node (default 10)\n"
      "      --nodes LIST     hybrid: ordered nodes cpu|dgpu|dgpu-ocl|igpu\n"
      "                       (default igpu,dgpu; supersedes --gpu-backend/--cpu-frac)\n"
      "      --fracs F,F,...  hybrid: explicit per-node row fractions (matches --nodes)\n"
      "      --ocl-device SUB opencl device name/vendor substring (or GOL_OCL_DEVICE)\n"
      "      --profile        print upload/compute/download timing breakdown\n"
      "      --csv            print one CSV data row instead of the human summary\n"
      "      --csv-header     print the CSV header line and exit (for sweep scripts)\n"
      "      --verify         compare selected backend against sequential CPU\n"
      "  -h, --help           show this help\n";
  std::exit(code);
}

/**
 * @brief Read the value that follows a flag; errors out if it is missing.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param[in,out] i Index of the flag; advanced past the consumed value.
 * @param flag Flag name, for error messages.
 * @return The value token following the flag.
 */
std::string nextValue(int argc, char** argv, int& i, const char* flag) {
  if (i + 1 >= argc) {
    std::cerr << "error: " << flag << " requires a value\n";
    usageAndExit(2);
  }
  return argv[++i];
}

/**
 * @brief Parse a string to unsigned long long, erroring out on bad input.
 * @param s    String to parse.
 * @param flag Flag name, for error messages.
 * @return The parsed value. Exits the process if @p s is not an integer.
 */
unsigned long long toULL(const std::string& s, const char* flag) {
  try {
    return std::stoull(s);
  } catch (const std::exception&) {
    std::cerr << "error: " << flag << " expects an integer, got '" << s << "'\n";
    usageAndExit(2);
  }
}

/**
 * @brief Parse a string to double, erroring out on bad input.
 * @param s    String to parse.
 * @param flag Flag name, for error messages.
 * @return The parsed value. Exits the process if @p s is not a number.
 */
double toDouble(const std::string& s, const char* flag) {
  try {
    return std::stod(s);
  } catch (const std::exception&) {
    std::cerr << "error: " << flag << " expects a number, got '" << s << "'\n";
    usageAndExit(2);
  }
}

} // namespace

Config parse(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      usageAndExit(0);
    } else if (arg == "-r" || arg == "--rows") {
      cfg.rows = static_cast<std::size_t>(toULL(nextValue(argc, argv, i, "--rows"), "--rows"));
    } else if (arg == "-c" || arg == "--cols") {
      cfg.cols = static_cast<std::size_t>(toULL(nextValue(argc, argv, i, "--cols"), "--cols"));
    } else if (arg == "-g" || arg == "--gens" || arg == "--generations") {
      cfg.generations = toULL(nextValue(argc, argv, i, "--gens"), "--gens");
    } else if (arg == "-t" || arg == "--threads" || arg == "--cores") {
      cfg.threads = static_cast<unsigned>(toULL(nextValue(argc, argv, i, "--threads"), "--threads"));
    } else if (arg == "--wrap") {
      cfg.wrap = true;
    } else if (arg == "--seed") {
      cfg.seed = toULL(nextValue(argc, argv, i, "--seed"), "--seed");
    } else if (arg == "--rle") {
      cfg.rlePath = nextValue(argc, argv, i, "--rle");
    } else if (arg == "--engine") {
      const std::string v = nextValue(argc, argv, i, "--engine");
      if (v == "cpu") cfg.engine = EngineKind::Cpu;
      else if (v == "cuda") cfg.engine = EngineKind::Cuda;
      else if (v == "opencl") cfg.engine = EngineKind::OpenCL;
      else if (v == "hybrid") cfg.engine = EngineKind::Hybrid;
      else { std::cerr << "error: unknown engine '" << v << "'\n"; usageAndExit(2); }
    } else if (arg == "--renderer") {
      const std::string v = nextValue(argc, argv, i, "--renderer");
      if (v == "null") cfg.renderer = RendererKind::Null;
      else if (v == "text") cfg.renderer = RendererKind::Text;
      else if (v == "ansi") cfg.renderer = RendererKind::Ansi;
      else { std::cerr << "error: unknown renderer '" << v << "'\n"; usageAndExit(2); }
    } else if (arg == "--block") {
      cfg.blockSize = static_cast<int>(toULL(nextValue(argc, argv, i, "--block"), "--block"));
    } else if (arg == "--shared") {
      cfg.useShared = true;
    } else if (arg == "--cpu-frac") {
      cfg.cpuFrac = toDouble(nextValue(argc, argv, i, "--cpu-frac"), "--cpu-frac");
    } else if (arg == "--gpu-backend") {
      const std::string v = nextValue(argc, argv, i, "--gpu-backend");
      if (v == "auto" || v == "cuda" || v == "opencl") cfg.hybridGpu = v;
      else { std::cerr << "error: unknown gpu-backend '" << v << "'\n"; usageAndExit(2); }
    } else if (arg == "--calib-steps") {
      cfg.calibSteps = static_cast<unsigned>(toULL(nextValue(argc, argv, i, "--calib-steps"), "--calib-steps"));
    } else if (arg == "--nodes") {
      cfg.hybridNodes = nextValue(argc, argv, i, "--nodes");
    } else if (arg == "--ocl-device") {
      cfg.oclDevice = nextValue(argc, argv, i, "--ocl-device");
    } else if (arg == "--fracs") {
      const std::string v = nextValue(argc, argv, i, "--fracs");
      std::vector<double> fracs;
      std::size_t start = 0;
      while (start <= v.size()) {
        const std::size_t comma = v.find(',', start);
        const std::string tok =
            v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) fracs.push_back(toDouble(tok, "--fracs"));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      cfg.hybridFracs = std::move(fracs);
    } else if (arg == "--verify") {
      cfg.verify = true;
    } else if (arg == "--csv") {
      cfg.csv = true;
    } else if (arg == "--profile") {
        cfg.profile = true;
    } else if (arg == "--csv-header") {
      cfg.csvHeader = true;
    } else {
      std::cerr << "error: unknown option '" << arg << "'\n";
      usageAndExit(2);
    }
  }
  return cfg;
}

} // namespace gol
