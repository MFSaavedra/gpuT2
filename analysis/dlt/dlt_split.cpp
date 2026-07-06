/**
 * dlt_split.cpp -- cross-check the gol HybridEngine's static split against the
 * reference DLT library (DLTlib, Barlas "Multicore and GPU Programming, 2e",
 * Appendix G) shipped with the book.
 *
 * The HybridEngine calibrates each node's raw throughput R_i (cells/s) and splits
 * the board by part_i = R_i / sum_j R_j -- the communication-free Divisible Load
 * Theory optimum for a single-level tree (Barlas 11.3). This tool feeds the same
 * measured rates into the *library's* solver and prints its optimum.
 *
 * DLTlib model: an idle "LON" root (huge power -> ~0 load, a pure distributor) with
 * one compute child per device. The child's execution-time model is
 *     time_i = power_i * (part_i * L + e0_i),
 * i.e. e0_i is a FIXED amount of load-equivalent overhead. So a physical fixed
 * per-step overhead of tau_i seconds maps to  e0_i = tau_i * R_i  (power_i = 1/R_i).
 * With e0 = 0 (and link = 0) the optimum is exactly R_i / sum R_j.
 *
 * Two modes:
 *   Manual:  ./dlt_split [--L N] [--link L] [--e0 NAME=cells]... NAME=RATE ...
 *   Fit:     ./dlt_split --fit overhead.csv --target-N N [--target-N N]...
 *            reads (device,cells,t_step_s) rows, OLS-fits t = tau + cells/R per
 *            device, converts (R, tau) -> (power, e0), and solves at L = N*N.
 *
 * Build:  DLTLIB=/path/to/DLTlib ./build.sh
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// DLTlib needs this global defined by the including program (see DLTdemo.cpp).
long global_random_seed = 1;
#include "dltlib.cpp"

struct Dev {
  std::string name;
  double rate = 0.0;   // R_i, cells/s
  double e0 = 0.0;     // DLTlib load-units overhead
  double tau = 0.0;    // physical per-step overhead, seconds (fit mode; for display)
  int npts = 0;        // fit points (fit mode)
};

// ---- one-shot solve: idle root + a compute child per device; print the split ----
static int solveAndPrint(std::vector<Dev>& devs, double L, double link) {
  double sumR = 0.0;
  for (auto& d : devs) sumR += d.rate;

  Network net;
  net.InsertNode((char*)"LON", 1e12, 0.0, (char*)NULL, link, /*fe=*/true, /*thru=*/false);
  for (auto& d : devs)
    net.InsertNode((char*)d.name.c_str(), 1.0 / d.rate, d.e0,
                   (char*)"LON", link, /*fe=*/true, /*thru=*/false);

  net.SolveImage((long)L, /*a=*/1.0, /*c=*/1.0);

  printf("L=%.0f  (N=%.0f)  link=%.4g\n", L, (L > 0 ? std::sqrt(L) : 0.0), link);
  printf("%-6s %12s %10s %11s %13s %12s\n",
         "node", "R(Gc/s)", "tau(us)", "e0(cells)", "DLTlib part", "naive R/SumR");
  if (net.valid != 1) { printf("  (DLTlib: no valid solution for these params)\n"); return 1; }
  for (Node* h = net.head; h; h = h->next_n) {
    if (strcmp(h->name, "LON") == 0) continue;
    Dev* d = nullptr;
    for (auto& x : devs) if (x.name == h->name) d = &x;
    double naive = d ? d->rate / sumR : 0.0;
    printf("%-6s %12.4g %10.3g %11.4g %13.6f %12.6f\n",
           h->name, (d ? d->rate : 1.0/h->power) / 1e9,
           d ? d->tau * 1e6 : 0.0, h->e0, h->part, naive);
  }
  return 0;
}

// ---- OLS fit of t = tau + cells * (1/R) for each device in overhead.csv ----
static std::vector<Dev> fitDevices(const char* path) {
  FILE* f = std::fopen(path, "r");
  if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); std::exit(2); }
  std::vector<std::string> order;                 // preserve first-seen device order
  std::map<std::string, std::vector<std::pair<double,double>>> pts; // name -> (cells,t)
  char line[512];
  bool first = true;
  while (std::fgets(line, sizeof line, f)) {
    if (first) { first = false; if (std::strstr(line, "device")) continue; }
    char name[128]; double cells, t;
    if (std::sscanf(line, "%127[^,],%lf,%lf", name, &cells, &t) == 3) {
      if (!pts.count(name)) order.push_back(name);
      pts[name].push_back({cells, t});
    }
  }
  std::fclose(f);

  // Robust decoupled estimator (plain OLS of t on cells is leverage-biased by the
  // largest board, which inflates the intercept tau by orders of magnitude):
  //   R   = peak rate from the LARGEST board  (compute-dominated, fully boosted)
  //   tau = mean residual t - cells/R over the SMALL boards (cells <= cells_max/256),
  //         i.e. only where the fixed overhead is a resolvable fraction of t_step.
  std::vector<Dev> devs;
  for (const auto& nm : order) {
    auto v = pts[nm];
    std::sort(v.begin(), v.end());                 // ascending by cells
    const double cmax = v.back().first, tmax = v.back().second;
    const double R = tmax > 0 ? cmax / tmax : 0.0;
    const double thresh = cmax / 256.0;
    double sres = 0.0; int k = 0;
    for (auto& p : v)
      if (p.first <= thresh) { sres += p.second - p.first / R; ++k; }
    if (k == 0) { sres = v.front().second - v.front().first / R; k = 1; } // fallback
    double tau = sres / k;
    Dev d;
    d.name = nm;
    d.rate = R;
    d.tau  = tau;
    d.e0   = (tau > 0 ? tau : 0.0) * R;            // clamp negative (noise) to 0
    d.npts = (int)v.size();
    devs.push_back(d);
  }
  return devs;
}

int main(int argc, char** argv) {
  double L = 1e6, link = 0.0;
  const char* fitPath = nullptr;
  std::vector<double> targetN;
  std::vector<Dev> manual;
  std::vector<std::pair<std::string,double>> e0s;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--L" && i+1 < argc)         { L = std::atof(argv[++i]); continue; }
    if (a == "--link" && i+1 < argc)      { link = std::atof(argv[++i]); continue; }
    if (a == "--fit" && i+1 < argc)       { fitPath = argv[++i]; continue; }
    if (a == "--target-N" && i+1 < argc)  { targetN.push_back(std::atof(argv[++i])); continue; }
    if (a == "--e0" && i+1 < argc) {
      std::string kv = argv[++i]; auto e = kv.find('=');
      if (e != std::string::npos) e0s.push_back({kv.substr(0,e), std::atof(kv.c_str()+e+1)});
      continue;
    }
    auto eq = a.find('=');
    if (eq != std::string::npos) { Dev d; d.name=a.substr(0,eq); d.rate=std::atof(a.c_str()+eq+1); manual.push_back(d); continue; }
    fprintf(stderr, "ignoring unrecognized arg '%s'\n", a.c_str());
  }

  if (fitPath) {
    std::vector<Dev> devs = fitDevices(fitPath);
    printf("== fitted affine timing  t_step = tau + cells/R  (from %s) ==\n", fitPath);
    for (auto& d : devs)
      printf("   %-6s R=%.4g Gcells/s  tau=%.3g us  e0=%.4g cells  (%d pts)\n",
             d.name.c_str(), d.rate/1e9, d.tau*1e6, d.e0, d.npts);
    printf("\n");
    if (targetN.empty()) targetN.push_back(16384);
    for (double N : targetN) {
      solveAndPrint(devs, N*N, link);
      printf("\n");
    }
    return 0;
  }

  if (manual.empty()) {
    fprintf(stderr,
      "usage:\n"
      "  %s [--L N] [--link L] [--e0 NAME=cells]... NAME=RATE [NAME=RATE...]\n"
      "  %s --fit overhead.csv --target-N 16384 [--target-N 32768]...\n", argv[0], argv[0]);
    return 2;
  }
  for (auto& d : manual)
    for (auto& kv : e0s) if (kv.first == d.name) d.e0 = kv.second;
  return solveAndPrint(manual, L, link);
}
