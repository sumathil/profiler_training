#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
constexpr uint32_t kColorMemcpy = 0xFF1E88E5;         // Blue
constexpr uint32_t kColorKernelNaive = 0xFFE53935;    // Red
constexpr uint32_t kColorKernelTiled = 0xFFFDD835;    // Yellow
constexpr uint32_t kColorKernelWarmup = 0xFF8E24AA;   // Purple
constexpr uint32_t kColorKernelTimed = 0xFFFB8C00;    // Orange
constexpr uint32_t kColorMarker = 0xFFFF00FF;         // Magenta
constexpr uint32_t kColorRange = 0xFF00FFFF;          // Cyan
constexpr uint32_t kColorCPU = 0xFF4CAF50;            // Green

// NVTX Push/Pop RAII wrapper
class NvtxScopedRange {
 public:
  NvtxScopedRange(const char *name, uint32_t color) {
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
  }

  ~NvtxScopedRange() {
    nvtxRangePop();
  }
};

// NVTX Marker helper
void nvtxMarker(const char *name, uint32_t color) {
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.colorType = NVTX_COLOR_ARGB;
  attr.color = color;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = name;
  nvtxMarkEx(&attr);
}

// NVTX Range (start/end) helper class
class NvtxRange {
 public:
  NvtxRange() : rangeId_(0) {}

  void start(const char *name, uint32_t color) {
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    rangeId_ = nvtxRangeStartEx(&attr);
  }

  void end() {
    if (rangeId_ != 0) {
      nvtxRangeEnd(rangeId_);
      rangeId_ = 0;
    }
  }

  ~NvtxRange() {
    end();
  }

 private:
  nvtxRangeId_t rangeId_;
};

}  // namespace

// Type aliases for Kokkos views
using ViewMatrixType = Kokkos::View<float**, Kokkos::LayoutLeft>;
using ViewMatrixHost = Kokkos::View<float**, Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Naive matrix multiplication kernel using Kokkos
struct MatMulNaive {
  ViewMatrixType a;
  ViewMatrixType b;
  ViewMatrixType c;
  int n;

  MatMulNaive(ViewMatrixType a_, ViewMatrixType b_, ViewMatrixType c_, int n_)
      : a(a_), b(b_), c(c_), n(n_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int row, const int col) const {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
      sum += a(row, k) * b(k, col);
    }
    c(row, col) = sum;
  }
};

// Tiled matrix multiplication using Kokkos TeamPolicy
template <int TILE_SIZE>
struct MatMulTiled {
  ViewMatrixType a;
  ViewMatrixType b;
  ViewMatrixType c;
  int n;

  MatMulTiled(ViewMatrixType a_, ViewMatrixType b_, ViewMatrixType c_, int n_)
      : a(a_), b(b_), c(c_), n(n_) {}

  using team_policy = Kokkos::TeamPolicy<>;
  using team_member = typename team_policy::member_type;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &team) const {
    const int team_row = team.league_rank() / ((n + TILE_SIZE - 1) / TILE_SIZE);
    const int team_col = team.league_rank() % ((n + TILE_SIZE - 1) / TILE_SIZE);

    const int row_start = team_row * TILE_SIZE;
    const int col_start = team_col * TILE_SIZE;

    // Shared memory for tiles
    Kokkos::View<float[TILE_SIZE][TILE_SIZE],
                 Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> tileA(team.team_scratch(1));
    Kokkos::View<float[TILE_SIZE][TILE_SIZE],
                 Kokkos::DefaultExecutionSpace::scratch_memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> tileB(team.team_scratch(1));

    float sum = 0.0f;
    const int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
      // Load tiles in parallel
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, TILE_SIZE * TILE_SIZE),
                          [&](const int idx) {
        const int ty = idx / TILE_SIZE;
        const int tx = idx % TILE_SIZE;

        const int aRow = row_start + ty;
        const int aCol = t * TILE_SIZE + tx;
        const int bRow = t * TILE_SIZE + ty;
        const int bCol = col_start + tx;

        tileA(ty, tx) = (aRow < n && aCol < n) ? a(aRow, aCol) : 0.0f;
        tileB(ty, tx) = (bRow < n && bCol < n) ? b(bRow, bCol) : 0.0f;
      });

      team.team_barrier();

      // Compute partial sum for this thread
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, TILE_SIZE),
                             [&](const int k, float &lsum) {
        const int local_row = team.team_rank() / TILE_SIZE;
        const int local_col = team.team_rank() % TILE_SIZE;
        lsum += tileA(local_row, k) * tileB(k, local_col);
      }, sum);

      team.team_barrier();
    }

    // Write result
    const int row = row_start + team.team_rank() / TILE_SIZE;
    const int col = col_start + team.team_rank() % TILE_SIZE;
    if (row < n && col < n) {
      Kokkos::single(Kokkos::PerThread(team), [&]() {
        c(row, col) = sum;
      });
    }
  }
};

// Benchmark naive Kokkos kernel
double benchmarkKokkosNaive(ViewMatrixType d_a, ViewMatrixType d_b,
                            ViewMatrixType d_c, int n, int iterations) {
  NvtxScopedRange warmupRange("kokkos_naive_warmup", kColorKernelWarmup);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    char markerName[64];
    snprintf(markerName, sizeof(markerName), "naive_warmup_iteration_%d", i);
    nvtxMarker(markerName, kColorMarker);

    NvtxScopedRange launchRange("kokkos_naive_launch", kColorKernelNaive);
    Kokkos::parallel_for("MatMulNaive",
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}),
                        MatMulNaive(d_a, d_b, d_c, n));
  }
  Kokkos::fence();

  warmupRange.~NvtxScopedRange();

  // Timed runs
  NvtxRange timingRange;
  timingRange.start("naive_timing_section", kColorRange);

  Kokkos::Timer timer;
  {
    NvtxScopedRange timedRange("kokkos_naive_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      if (i % 10 == 0) {
        char markerName[64];
        snprintf(markerName, sizeof(markerName), "naive_timed_iter_%d", i);
        nvtxMarker(markerName, kColorMarker);
      }

      NvtxScopedRange launchRange("kokkos_naive_launch", kColorKernelNaive);
      Kokkos::parallel_for("MatMulNaive",
                          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}),
                          MatMulNaive(d_a, d_b, d_c, n));
    }
  }
  Kokkos::fence();
  double elapsedMs = timer.seconds() * 1000.0;

  timingRange.end();

  return elapsedMs / static_cast<double>(iterations);
}

// Benchmark tiled Kokkos kernel
template <int TILE_SIZE>
double benchmarkKokkosTiled(ViewMatrixType d_a, ViewMatrixType d_b,
                           ViewMatrixType d_c, int n, int iterations) {
  NvtxScopedRange warmupRange("kokkos_tiled_warmup", kColorKernelWarmup);

  const int num_teams = ((n + TILE_SIZE - 1) / TILE_SIZE) *
                        ((n + TILE_SIZE - 1) / TILE_SIZE);
  const int team_size = TILE_SIZE * TILE_SIZE;
  const int scratch_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

  using policy_type = Kokkos::TeamPolicy<>;
  policy_type policy(num_teams, team_size);
  policy = policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size));

  // Warmup
  for (int i = 0; i < 1; ++i) {
    nvtxMarker("tiled_warmup_start", kColorMarker);

    NvtxScopedRange launchRange("kokkos_tiled_launch", kColorKernelTiled);
    Kokkos::parallel_for("MatMulTiled", policy,
                        MatMulTiled<TILE_SIZE>(d_a, d_b, d_c, n));
  }
  Kokkos::fence();

  warmupRange.~NvtxScopedRange();

  // Timed runs
  NvtxRange timingRange;
  timingRange.start("tiled_timing_section", kColorRange);

  Kokkos::Timer timer;
  {
    NvtxScopedRange timedRange("kokkos_tiled_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      if (i % 10 == 0) {
        char markerName[64];
        snprintf(markerName, sizeof(markerName), "tiled_timed_iter_%d", i);
        nvtxMarker(markerName, kColorMarker);
      }

      NvtxScopedRange launchRange("kokkos_tiled_launch", kColorKernelTiled);
      Kokkos::parallel_for("MatMulTiled", policy,
                          MatMulTiled<TILE_SIZE>(d_a, d_b, d_c, n));
    }
  }
  Kokkos::fence();
  double elapsedMs = timer.seconds() * 1000.0;

  timingRange.end();

  return elapsedMs / static_cast<double>(iterations);
}

// CPU reference implementation
void matmulCpu(const std::vector<float> &a, const std::vector<float> &b,
               std::vector<float> &c, int n) {
  for (int row = 0; row < n; ++row) {
    int rowBase = row * n;
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int k = 0; k < n; ++k) {
        sum += a[rowBase + k] * b[k * n + col];
      }
      c[rowBase + col] = sum;
    }
  }
}

double benchmarkCpu(const std::vector<float> &a, const std::vector<float> &b,
                    std::vector<float> &c, int n, int iterations) {
  // Warmup
  matmulCpu(a, b, c, n);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    matmulCpu(a, b, c, n);
  }
  auto stop = std::chrono::high_resolution_clock::now();

  double elapsedMs = std::chrono::duration<double, std::milli>(stop - start).count();
  return elapsedMs / static_cast<double>(iterations);
}

double gflopsFromMs(int n, double avgMs) {
  double seconds = avgMs / 1e3;
  return (2.0 * static_cast<double>(n) * n * n) / (seconds * 1e9);
}

bool validateOutput(const std::vector<float> &out, float expected) {
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::fabs(out[i] - expected) > 1e-2f) {
      std::cerr << "Validation failed at index " << i
                << ": got " << out[i]
                << ", expected " << expected << std::endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  nvtxMarker("Program Start", kColorMarker);

  Kokkos::initialize(argc, argv);
  {
    int n = 1024;
    int iterations = 50;

    if (argc > 1) {
      n = std::atoi(argv[1]);
    }
    if (argc > 2) {
      iterations = std::atoi(argv[2]);
    }

    if (n <= 0 || iterations <= 0) {
      std::cerr << "Usage: ./matrix_kokkos [matrix_size] [iterations]" << std::endl;
      Kokkos::finalize();
      return EXIT_FAILURE;
    }

    size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);

    std::cout << "Kokkos Configuration:\n";
    std::cout << "  Execution Space: " << typeid(Kokkos::DefaultExecutionSpace).name() << "\n";
    std::cout << "  Memory Space: " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << "\n";
    std::cout << "Matrix size: " << n << "x" << n
              << ", iterations: " << iterations << "\n\n";

    // CPU benchmark
    std::vector<float> h_a(elements, 1.0f);
    std::vector<float> h_b(elements, 2.0f);
    std::vector<float> h_c_cpu(elements, 0.0f);

    NvtxRange cpuRange;
    cpuRange.start("CPU_Benchmark", kColorCPU);

    double cpuMs = benchmarkCpu(h_a, h_b, h_c_cpu, n, iterations);
    if (!validateOutput(h_c_cpu, 2.0f * static_cast<float>(n))) {
      Kokkos::finalize();
      return EXIT_FAILURE;
    }
    double cpuGflops = gflopsFromMs(n, cpuMs);

    cpuRange.end();
    nvtxMarker("CPU Benchmark Complete", kColorMarker);

    std::cout << "CPU Execution: " << std::fixed << std::setprecision(3)
              << cpuMs << " ms, " << cpuGflops << " GFLOPS\n\n";

    // Allocate Kokkos views
    NvtxScopedRange allocRange("Kokkos_View_Allocation", kColorMemcpy);
    ViewMatrixType d_a("d_a", n, n);
    ViewMatrixType d_b("d_b", n, n);
    ViewMatrixType d_c("d_c", n, n);

    ViewMatrixHost h_a_view("h_a", n, n);
    ViewMatrixHost h_b_view("h_b", n, n);
    ViewMatrixHost h_c_view("h_c", n, n);

    // Initialize host views
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        h_a_view(i, j) = 1.0f;
        h_b_view(i, j) = 2.0f;
      }
    }

    allocRange.~NvtxScopedRange();

    // Copy data to device
    {
      NvtxScopedRange copyRange("Kokkos_DeepCopy_H2D", kColorMemcpy);
      Kokkos::deep_copy(d_a, h_a_view);
      Kokkos::deep_copy(d_b, h_b_view);
    }

    nvtxMarker("Data Transfer Complete", kColorMarker);

    std::cout << std::string(90, '-') << "\n";
    std::cout << "\t\t\tKokkos GPU Execution\n";
    std::cout << std::string(90, '-') << "\n";
    std::cout << std::left
              << std::setw(12) << "TileSize"
              << std::setw(18) << "Naive Time(ms)"
              << std::setw(18) << "Tiled Time(ms)"
              << std::setw(12) << "Naive/CPU"
              << std::setw(18) << "Tiled/CPU"
              << std::setw(12) << "Tiled/Naive"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    NvtxRange gpuBenchmarkRange;
    gpuBenchmarkRange.start("All_Kokkos_Benchmarks", kColorRange);

    // Naive implementation
    {
      char markerName[64];
      snprintf(markerName, sizeof(markerName), "Benchmarking Naive");
      nvtxMarker(markerName, kColorMarker);

      double naiveTime = benchmarkKokkosNaive(d_a, d_b, d_c, n, iterations);
      double speedupNaiveVsCpu = cpuMs / naiveTime;

      std::cout << std::left
                << std::setw(12) << "Naive"
                << std::fixed << std::setprecision(3)
                << std::setw(18) << naiveTime
                << std::setw(18) << "-"
                << std::setw(12) << speedupNaiveVsCpu
                << std::setw(18) << "-"
                << std::setw(12) << "-"
                << "\n";

      // Validate naive
      {
        NvtxScopedRange copyRange("Kokkos_DeepCopy_D2H_Naive", kColorMemcpy);
        Kokkos::deep_copy(h_c_view, d_c);
      }

      std::vector<float> h_c_naive(elements);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          h_c_naive[i * n + j] = h_c_view(i, j);
        }
      }

      if (!validateOutput(h_c_naive, 2.0f * static_cast<float>(n))) {
        std::cerr << "Naive validation failed!\n";
        Kokkos::finalize();
        return EXIT_FAILURE;
      }
    }

    // Tiled implementations
    const int tileSizes[] = {8, 16, 32};
    for (int tileSize : tileSizes) {
      char markerName[64];
      snprintf(markerName, sizeof(markerName), "Benchmarking TileSize=%d", tileSize);
      nvtxMarker(markerName, kColorMarker);

      double naiveTime = benchmarkKokkosNaive(d_a, d_b, d_c, n, iterations);

      double tiledTime = 0.0;
      if (tileSize == 8) {
        tiledTime = benchmarkKokkosTiled<8>(d_a, d_b, d_c, n, iterations);
      } else if (tileSize == 16) {
        tiledTime = benchmarkKokkosTiled<16>(d_a, d_b, d_c, n, iterations);
      } else if (tileSize == 32) {
        tiledTime = benchmarkKokkosTiled<32>(d_a, d_b, d_c, n, iterations);
      }

      double speedupNaiveVsCpu = cpuMs / naiveTime;
      double speedupTiledVsCpu = cpuMs / tiledTime;
      double speedupTiledVsNaive = naiveTime / tiledTime;

      std::cout << std::left
                << std::setw(12) << tileSize
                << std::fixed << std::setprecision(3)
                << std::setw(18) << naiveTime
                << std::setw(18) << tiledTime
                << std::setw(12) << speedupNaiveVsCpu
                << std::setw(18) << speedupTiledVsCpu
                << std::setw(12) << speedupTiledVsNaive
                << "\n";

      // Validate tiled
      {
        NvtxScopedRange copyRange("Kokkos_DeepCopy_D2H_Tiled", kColorMemcpy);
        Kokkos::deep_copy(h_c_view, d_c);
      }

      std::vector<float> h_c_tiled(elements);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          h_c_tiled[i * n + j] = h_c_view(i, j);
        }
      }

      if (!validateOutput(h_c_tiled, 2.0f * static_cast<float>(n))) {
        std::cerr << "Tiled validation failed for tile size " << tileSize << "!\n";
        Kokkos::finalize();
        return EXIT_FAILURE;
      }
    }

    gpuBenchmarkRange.end();
    nvtxMarker("All Kokkos Benchmarks Complete", kColorMarker);

    std::cout << std::string(90, '-') << "\n";
  }

  nvtxMarker("Program End", kColorMarker);
  Kokkos::finalize();

  return EXIT_SUCCESS;
}
