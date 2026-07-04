/**
 * @file HybridEngine.cpp
 * @brief Implementation of the hybrid multi-device backend with static load
 *        balancing (Divisible Load Theory, Barlas chapter 11.3).
 *
 * Row-wise domain decomposition over an ordered list of nodes (top to bottom).
 * Each node owns a contiguous band of rows sized once -- from explicit fractions
 * or an a-priori calibration phase -- then frozen. Every generation the nodes
 * compute concurrently and exchange a one-row ghost halo across each adjacent
 * seam (and the far edges under --wrap); the bulk bands never move until
 * download(). Inter-node seams stage through host memory (no cross-vendor peer
 * copy), which is the only per-step host<->device traffic.
 */

#include "gol/engines/HybridEngine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "gol/LifeRules.hpp"
#include "gol/Timer.hpp"

#ifdef GOL_HAVE_CUDA
#include "gol/engines/cuda/CudaHaloPartition.hpp"
#endif
#ifdef GOL_HAVE_OPENCL
#include "gol/engines/opencl/OpenCLHaloPartition.hpp"
#endif

namespace gol {

namespace {
unsigned resolveThreads(unsigned requested) {
  if (requested != 0) return requested;
  unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1u : hw;
}

// Resolve GpuBackend::Auto to a concrete node kind for the legacy CPU+GPU ctor.
NodeKind resolveGpuKind(GpuBackend backend) {
  GpuBackend b = backend;
  if (b == GpuBackend::Auto) {
#if defined(GOL_HAVE_CUDA)
    b = GpuBackend::Cuda;
#elif defined(GOL_HAVE_OPENCL)
    b = GpuBackend::OpenCL;
#endif
  }
  return (b == GpuBackend::OpenCL) ? NodeKind::OpenCL : NodeKind::Cuda;
}

// The legacy two-node [CPU, GPU] composition.
std::vector<HybridNode> legacyNodeList(GpuBackend backend) {
  HybridNode cpu;
  cpu.kind = NodeKind::Cpu;
  cpu.label = "cpu";
  HybridNode gpu;
  gpu.kind = resolveGpuKind(backend);
  gpu.label = "gpu";
  return {cpu, gpu};
}

// Map one --nodes token to a HybridNode; throws if its backend is not compiled.
HybridNode makeNodeToken(const std::string& tok) {
  HybridNode n;
  if (tok == "cpu") {
    n.kind = NodeKind::Cpu;
    n.label = "cpu";
  } else if (tok == "dgpu") {
#if defined(GOL_HAVE_CUDA)
    n.kind = NodeKind::Cuda;
    n.label = "dgpu";
#else
    throw std::invalid_argument("hybrid: node 'dgpu' needs CUDA (build -DBUILD_CUDA=ON)");
#endif
  } else if (tok == "dgpu-ocl") {
#if defined(GOL_HAVE_OPENCL)
    n.kind = NodeKind::OpenCL;
    n.deviceHint = "nvidia";
    n.label = "dgpu-ocl";
#else
    throw std::invalid_argument("hybrid: node 'dgpu-ocl' needs OpenCL (build -DBUILD_OPENCL=ON)");
#endif
  } else if (tok == "igpu") {
#if defined(GOL_HAVE_OPENCL)
    n.kind = NodeKind::OpenCL;
    n.deviceHint = "intel";
    n.label = "igpu";
#else
    throw std::invalid_argument("hybrid: node 'igpu' needs OpenCL (build -DBUILD_OPENCL=ON)");
#endif
  } else {
    throw std::invalid_argument("hybrid: unknown node '" + tok +
                                "' (use cpu|dgpu|dgpu-ocl|igpu)");
  }
  return n;
}
} // namespace

std::vector<HybridNode> defaultHybridNodes() {
#if defined(GOL_HAVE_CUDA) && defined(GOL_HAVE_OPENCL)
  return {makeNodeToken("igpu"), makeNodeToken("dgpu")};   // iGPU(OpenCL) + dGPU(CUDA)
#elif defined(GOL_HAVE_CUDA)
  return {makeNodeToken("dgpu")};                          // dGPU only (no iGPU reachable)
#elif defined(GOL_HAVE_OPENCL)
  return {makeNodeToken("igpu"), makeNodeToken("dgpu-ocl")}; // both via OpenCL
#else
  return {};
#endif
}

std::vector<HybridNode> parseHybridNodes(const std::string& spec) {
  std::vector<HybridNode> nodes;
  std::size_t start = 0;
  while (start <= spec.size()) {
    const std::size_t comma = spec.find(',', start);
    std::string tok = spec.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.erase(tok.begin());
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
    if (!tok.empty()) nodes.push_back(makeNodeToken(tok));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  if (nodes.empty()) throw std::invalid_argument("hybrid: empty --nodes spec");
  return nodes;
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------
HybridEngine::HybridEngine(unsigned threads, int blockSize, bool wrap,
                           std::vector<HybridNode> nodes,
                           std::optional<std::vector<double>> fracs,
                           unsigned calibSteps)
    : threads_(resolveThreads(threads)),
      blockSize_(blockSize < 32 ? 32 : blockSize),
      wrap_(wrap),
      calibSteps_(calibSteps == 0 ? 1u : calibSteps),
      nodeSpec_(std::move(nodes)),
      fracsOverride_(std::move(fracs)) {
  if (nodeSpec_.empty()) {
    HybridNode gpu;
    gpu.kind = resolveGpuKind(GpuBackend::Auto);
    gpu.label = "gpu";
    nodeSpec_.push_back(gpu);
  }
  if (fracsOverride_ && fracsOverride_->size() != nodeSpec_.size()) {
    throw std::invalid_argument("hybrid: --fracs count does not match the node count");
  }
  for (const HybridNode& n : nodeSpec_) {
    if (n.kind == NodeKind::Cpu) hasCpuNode_ = true;
  }
  if (hasCpuNode_ && threads_ >= 2) startPool();
}

HybridEngine::HybridEngine(unsigned threads, int blockSize, bool wrap,
                           GpuBackend backend, std::optional<double> cpuFracOverride,
                           unsigned calibSteps)
    : HybridEngine(threads, blockSize, wrap, legacyNodeList(backend),
                   cpuFracOverride
                       ? std::optional<std::vector<double>>(std::vector<double>{
                             *cpuFracOverride, 1.0 - *cpuFracOverride})
                       : std::nullopt,
                   calibSteps) {}

HybridEngine::~HybridEngine() {
  if (!workers_.empty()) {
    stop_.store(true, std::memory_order_relaxed);
    barrier_->arrive_and_wait(); // workers observe stop_ and return
    for (auto& w : workers_) w.join();
  }
}

// ---------------------------------------------------------------------------
// Node / partition construction
// ---------------------------------------------------------------------------
std::unique_ptr<IHaloPartition> HybridEngine::makePartition(const HybridNode& n) const {
  switch (n.kind) {
    case NodeKind::Cuda:
#if defined(GOL_HAVE_CUDA)
      return makeCudaHaloPartition(blockSize_);
#else
      throw std::runtime_error("hybrid: CUDA node requested but CUDA is not built");
#endif
    case NodeKind::OpenCL:
#if defined(GOL_HAVE_OPENCL)
      return makeOpenCLHaloPartition(blockSize_, n.deviceHint);
#else
      throw std::runtime_error("hybrid: OpenCL node requested but OpenCL is not built");
#endif
    case NodeKind::Cpu:
      return nullptr;
  }
  return nullptr;
}

void HybridEngine::buildNodes() {
  nodes_.clear();
  nodes_.reserve(nodeSpec_.size()); // reserved once -> Node* into nodes_ stay valid
  for (const HybridNode& s : nodeSpec_) {
    Node node;
    node.spec = s;
    nodes_.push_back(std::move(node));
  }
}

std::string HybridEngine::gpuName() const {
  for (const Node* n : active_) {
    if (n->part) return n->part->name();
  }
  return "none";
}

std::vector<HybridEngine::NodeReport> HybridEngine::nodeReports() const {
  std::vector<NodeReport> reports;
  reports.reserve(active_.size());
  for (const Node* n : active_) reports.push_back({n->spec.label, n->rows, n->rate});
  return reports;
}

// ---------------------------------------------------------------------------
// Calibration (the a-priori "initialization phase" of DLT): measure each node's
// pure compute throughput independently, on the full board. Ghost contents do not
// affect kernel/compute cost, so the loops run with whatever ghosts are present.
// ---------------------------------------------------------------------------
double HybridEngine::measureCpuRate(const Grid& initial) {
  const std::size_t R = rows_;
  const std::size_t C = cols_;
  const double cells = static_cast<double>(R) * static_cast<double>(C) *
                       static_cast<double>(calibSteps_);
  std::vector<unsigned char> a((R + 2) * C, 0u);
  std::vector<unsigned char> b((R + 2) * C, 0u);
  std::copy(initial.data(), initial.data() + R * C, a.data() + C);
  for (unsigned i = 0; i < 2; ++i) { // warmup
    runCpuParallel(a.data(), b.data(), R);
    std::swap(a, b);
  }
  Timer t;
  for (unsigned i = 0; i < calibSteps_; ++i) {
    runCpuParallel(a.data(), b.data(), R);
    std::swap(a, b);
  }
  const double ms = t.elapsedMillis();
  return ms > 0.0 ? cells / (ms / 1000.0) : 0.0;
}

double HybridEngine::measureGpuRate(IHaloPartition& part, const Grid& initial) {
  const std::size_t R = rows_;
  const std::size_t C = cols_;
  const double cells = static_cast<double>(R) * static_cast<double>(C) *
                       static_cast<double>(calibSteps_);
  part.uploadRegion(initial.data(), R, C, wrap_);
  for (unsigned i = 0; i < 2; ++i) { part.launchStep(); part.finishStep(); } // warmup
  Timer t;
  for (unsigned i = 0; i < calibSteps_; ++i) { part.launchStep(); part.finishStep(); }
  const double ms = t.elapsedMillis();
  return ms > 0.0 ? cells / (ms / 1000.0) : 0.0;
}

void HybridEngine::calibrate(const Grid& initial) {
  for (Node& n : nodes_) {
    if (n.spec.kind == NodeKind::Cpu) {
      n.rate = measureCpuRate(initial);
    } else {
      n.rate = measureGpuRate(*n.part, initial);
    }
  }
}

// ---------------------------------------------------------------------------
// Row apportionment: fractions (explicit or R_i / sum R_j) -> integer row counts
// summing to R (largest-remainder), then build the active (non-empty) node list.
// ---------------------------------------------------------------------------
void HybridEngine::assignRows() {
  const std::size_t N = nodeSpec_.size();
  std::vector<double> frac(N, 0.0);
  if (fracsOverride_) {
    double sum = 0.0;
    for (double f : *fracsOverride_) sum += std::max(0.0, f);
    for (std::size_t i = 0; i < N; ++i) {
      frac[i] = sum > 0.0 ? std::max(0.0, (*fracsOverride_)[i]) / sum
                          : 1.0 / static_cast<double>(N);
    }
  } else {
    double sum = 0.0;
    for (const Node& n : nodes_) sum += n.rate;
    for (std::size_t i = 0; i < N; ++i) {
      frac[i] = sum > 0.0 ? nodes_[i].rate / sum : 1.0 / static_cast<double>(N);
    }
  }

  // Largest-remainder apportionment so the row counts sum to exactly R.
  std::vector<std::size_t> base(N, 0);
  std::vector<double> rem(N, 0.0);
  std::size_t assigned = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const double exact = frac[i] * static_cast<double>(rows_);
    base[i] = static_cast<std::size_t>(std::floor(exact));
    rem[i] = exact - static_cast<double>(base[i]);
    assigned += base[i];
  }
  std::size_t leftover = rows_ >= assigned ? rows_ - assigned : 0;
  std::vector<std::size_t> order(N);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return rem[a] > rem[b]; });
  for (std::size_t k = 0; k < leftover && k < N; ++k) base[order[k]] += 1;

  for (std::size_t i = 0; i < N; ++i) nodes_[i].rows = base[i];

  // Build the active list (non-empty nodes) + cached legacy figures.
  active_.clear();
  cpuNode_ = nullptr;
  cpuRows_ = 0;
  gpuRows_ = 0;
  cpuRate_ = 0.0;
  gpuRate_ = 0.0;
  bool firstGpu = true;
  for (Node& n : nodes_) {
    if (n.rows == 0) continue;
    active_.push_back(&n);
    if (n.spec.kind == NodeKind::Cpu) {
      cpuNode_ = &n;
      cpuRows_ = n.rows;
      cpuRate_ = n.rate;
    } else {
      gpuRows_ += n.rows;
      if (firstGpu) {
        gpuRate_ = n.rate;
        firstGpu = false;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// upload(): build nodes, decide the split (calibrate or fractions), size + seed.
// ---------------------------------------------------------------------------
void HybridEngine::upload(const Grid& initial) {
  rows_ = initial.rows();
  cols_ = initial.cols();
  rowBuf_.assign(cols_, 0u);

  buildNodes();

  if (fracsOverride_) {
    calibrated_ = false;
    assignRows();
    // Create partitions only for the GPU nodes that ended up non-empty.
    for (Node& n : nodes_) {
      if (n.rows > 0 && n.spec.kind != NodeKind::Cpu && !n.part) {
        n.part = makePartition(n.spec);
      }
    }
  } else {
    // Calibration needs every GPU node's device, so create them all first.
    for (Node& n : nodes_) {
      if (n.spec.kind != NodeKind::Cpu) n.part = makePartition(n.spec);
    }
    calibrate(initial);
    calibrated_ = true;
    assignRows();
  }

  setupBuffers(initial);
}

void HybridEngine::setupBuffers(const Grid& initial) {
  const std::size_t C = cols_;
  std::size_t off = 0; // running board-row offset of the current node
  for (Node* n : active_) {
    if (n->spec.kind == NodeKind::Cpu) {
      cpuCur_.assign((n->rows + 2) * C, 0u);
      cpuNxt_.assign((n->rows + 2) * C, 0u);
      // Board rows [off, off + rows) -> buffer rows [1, rows].
      std::copy(initial.data() + off * C, initial.data() + (off + n->rows) * C,
                cpuCur_.data() + C);
    } else {
      n->part->uploadRegion(initial.data() + off * C, n->rows, C, wrap_);
    }
    off += n->rows;
  }
  if (!cpuNode_) {
    cpuCur_.clear();
    cpuNxt_.clear();
  }
}

// ---------------------------------------------------------------------------
// Uniform per-node boundary access (host memcpy for CPU, IHaloPartition for GPU).
// The CPU node's buffer is cpuCur_ (height rows+2; real rows at [1, rows]).
// ---------------------------------------------------------------------------
void HybridEngine::readTopRealRow(const Node& n, unsigned char* out) {
  if (n.part) {
    n.part->readTopRow(out);
  } else {
    std::copy(cpuCur_.data() + cols_, cpuCur_.data() + 2 * cols_, out); // buffer row 1
  }
}

void HybridEngine::readBottomRealRow(const Node& n, unsigned char* out) {
  if (n.part) {
    n.part->readBottomRow(out);
  } else {
    std::copy(cpuCur_.data() + n.rows * cols_, cpuCur_.data() + (n.rows + 1) * cols_,
              out); // buffer row n.rows
  }
}

void HybridEngine::setTopGhost(const Node& n, const unsigned char* in) {
  if (n.part) {
    n.part->setTopGhost(in);
  } else {
    std::copy(in, in + cols_, cpuCur_.data()); // buffer row 0
  }
}

void HybridEngine::setBottomGhost(const Node& n, const unsigned char* in) {
  if (n.part) {
    n.part->setBottomGhost(in);
  } else {
    std::copy(in, in + cols_, cpuCur_.data() + (n.rows + 1) * cols_); // buffer row rows+1
  }
}

// ---------------------------------------------------------------------------
// Ghost exchange: for each adjacent pair, swap the two touching boundary rows;
// under --wrap the first/last nodes also wrap onto each other. Bounded far ghosts
// stay zero (set at upload, never written by the kernel/pool).
// ---------------------------------------------------------------------------
void HybridEngine::exchangeGhosts() {
  const std::size_t N = active_.size();
  if (N == 0) return;

  for (std::size_t i = 0; i + 1 < N; ++i) {
    readBottomRealRow(*active_[i], rowBuf_.data());   // node i last real row
    setTopGhost(*active_[i + 1], rowBuf_.data());     //   -> node i+1 top ghost
    readTopRealRow(*active_[i + 1], rowBuf_.data());  // node i+1 first real row
    setBottomGhost(*active_[i], rowBuf_.data());      //   -> node i bottom ghost
  }

  if (wrap_) {
    readBottomRealRow(*active_[N - 1], rowBuf_.data()); // last node's last real row
    setTopGhost(*active_[0], rowBuf_.data());           //   -> first node's top ghost
    readTopRealRow(*active_[0], rowBuf_.data());        // first node's first real row
    setBottomGhost(*active_[N - 1], rowBuf_.data());    //   -> last node's bottom ghost
  }
}

// ---------------------------------------------------------------------------
// step(): ghosts -> launch every GPU node async -> compute the CPU node on the
// calling thread concurrently -> finish every GPU node -> swap.
// ---------------------------------------------------------------------------
void HybridEngine::step() {
  Timer t;
  exchangeGhosts();
  for (Node* n : active_) {
    if (n->part) n->part->launchStep(); // async: returns before the kernel finishes
  }
  if (cpuNode_) runCpuParallel(cpuCur_.data(), cpuNxt_.data(), cpuNode_->rows);
  for (Node* n : active_) {
    if (n->part) n->part->finishStep(); // wait + swap that device's buffers
  }
  if (cpuNode_) std::swap(cpuCur_, cpuNxt_);
  lastMs_ = t.elapsedMillis();
}

void HybridEngine::download(Grid& out) {
  const std::size_t C = cols_;
  std::size_t off = 0;
  for (Node* n : active_) {
    if (n->part) {
      n->part->downloadRegion(out.data() + off * C);
    } else {
      std::copy(cpuCur_.data() + C, cpuCur_.data() + (n->rows + 1) * C,
                out.data() + off * C);
    }
    off += n->rows;
  }
}

// ---------------------------------------------------------------------------
// CPU worker pool (persistent Barrier pool, mirrors CpuEngine).
// ---------------------------------------------------------------------------
void HybridEngine::startPool() {
  barrier_ = std::make_unique<Barrier>(static_cast<std::ptrdiff_t>(threads_));
  workers_.reserve(threads_ - 1);
  for (unsigned id = 1; id < threads_; ++id) {
    workers_.emplace_back([this, id] { workerLoop(id); });
  }
}

void HybridEngine::workerLoop(unsigned id) {
  while (true) {
    barrier_->arrive_and_wait(); // wait for runCpuParallel to publish work
    if (stop_.load(std::memory_order_relaxed)) return;
    auto [b, e] = bufferRangeFor(id);
    computeCpuRows(poolSrc_, poolDst_, b, e);
    barrier_->arrive_and_wait(); // signal completion
  }
}

std::pair<std::size_t, std::size_t> HybridEngine::bufferRangeFor(unsigned id) const {
  const std::size_t base = poolRows_ / threads_;
  const std::size_t rem = poolRows_ % threads_;
  const std::size_t begin = id * base + std::min<std::size_t>(id, rem);
  const std::size_t count = base + (id < rem ? 1u : 0u);
  // Real rows map to buffer rows [1 + begin, 1 + begin + count).
  return {1 + begin, 1 + begin + count};
}

void HybridEngine::runCpuParallel(const unsigned char* src, unsigned char* dst,
                                  std::size_t nReal) {
  poolSrc_ = src;
  poolDst_ = dst;
  poolRows_ = nReal;
  if (workers_.empty()) {
    computeCpuRows(src, dst, 1, nReal + 1); // sequential: all real rows
  } else {
    barrier_->arrive_and_wait();            // release workers
    auto [b, e] = bufferRangeFor(0);
    computeCpuRows(src, dst, b, e);         // main thread owns partition 0
    barrier_->arrive_and_wait();            // wait for every partition
  }
}

void HybridEngine::computeCpuRows(const unsigned char* src, unsigned char* dst,
                                  std::size_t brBegin, std::size_t brEnd) const {
  const long C = static_cast<long>(cols_);
  for (std::size_t br = brBegin; br < brEnd; ++br) {
    const std::size_t rowOff = br * cols_;
    for (long x = 0; x < C; ++x) {
      int n = 0;
      for (long dy = -1; dy <= 1; ++dy) {
        for (long dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          long nx = x + dx;
          if (wrap_) {
            nx = (nx + C) % C;
          } else if (nx < 0 || nx >= C) {
            continue; // bounded: out-of-range column is dead
          }
          // Vertical neighbour is always a valid buffer row (ghost rows present).
          const long nbrRow = static_cast<long>(br) + dy;
          n += src[static_cast<std::size_t>(nbrRow) * cols_ +
                   static_cast<std::size_t>(nx)];
        }
      }
      dst[rowOff + static_cast<std::size_t>(x)] =
          nextState(src[rowOff + static_cast<std::size_t>(x)], n);
    }
  }
}

} // namespace gol
