#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {
constexpr uint32_t kColorMemcpyH2D = 0xFF1E88E5;      // Blue
constexpr uint32_t kColorMemcpyD2H = 0xFF43A047;      // Green
constexpr uint32_t kColorKernelNaive = 0xFFE53935;    // Red
constexpr uint32_t kColorKernelTiled = 0xFFFDD835;    // Yellow
constexpr uint32_t kColorKernelWarmup = 0xFF8E24AA;   // Purple
constexpr uint32_t kColorKernelTimed = 0xFFFB8C00;    // Orange
constexpr uint32_t kColorMarker = 0xFFFF00FF;         // Magenta
constexpr uint32_t kColorRange = 0xFF00FFFF;          // Cyan

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

__global__ void matmulNaive(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.0f;
    int rowBase = row * n;
    for (int k = 0; k < n; ++k) {
      sum += a[rowBase + k] * b[k * n + col];
    }
    c[rowBase + col] = sum;
  }
}

template <int TILE>
__global__ void matmulTiled(const float *a, const float *b, float *c, int n) {
  __shared__ float tileA[TILE][TILE];
  __shared__ float tileB[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float sum = 0.0f;
  int numTiles = (n + TILE - 1) / TILE;

  for (int t = 0; t < numTiles; ++t) {
    int aRow = row;
    int aCol = t * TILE + tx;
    int bRow = t * TILE + ty;
    int bCol = col;

    tileA[ty][tx] =
        (aRow < n && aCol < n) ? a[aRow * n + aCol] : 0.0f;
    tileB[ty][tx] =
        (bRow < n && bCol < n) ? b[bRow * n + bCol] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += tileA[ty][k] * tileB[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    c[row * n + col] = sum;
  }
}

double gpuMM(const float *d_a, const float *d_b, float *d_c, int n,
                       int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);

  // NVTX Push/Pop for warmup
  {
    NvtxScopedRange warmupRange("kernel_naive_warmup", kColorKernelWarmup);
    for (int i = 0; i < 3; ++i) {
      // NVTX Marker for iteration start
      char markerName[64];
      snprintf(markerName, sizeof(markerName), "naive_warmup_iteration_%d", i);
      nvtxMarker(markerName, kColorMarker);
      
      NvtxScopedRange launchRange("matmul_naive_launch", kColorKernelNaive);
      matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // NVTX Range for timing section
  NvtxRange timingRange;
  timingRange.start("naive_timing_section", kColorRange);
  
  CHECK_CUDA(cudaEventRecord(start));
  {
    NvtxScopedRange timedRange("kernel_naive_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      // NVTX Marker every 10 iterations
      if (i % 10 == 0) {
        char markerName[64];
        snprintf(markerName, sizeof(markerName), "naive_timed_iter_%d", i);
        nvtxMarker(markerName, kColorMarker);
      }
      
      NvtxScopedRange launchRange("matmul_naive_launch", kColorKernelNaive);
      matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));
  
  timingRange.end();

  float elapsedMs = elapsed(start, stop);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

double gpuMM_shared(const float *d_a, const float *d_b, float *d_c, int n,
                       int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);
  
  // NVTX Push/Pop for warmup
  {
    NvtxScopedRange warmupRange("kernel_tiled_warmup", kColorKernelWarmup);
    for (int i = 0; i < 1; ++i) {
      nvtxMarker("tiled_warmup_start", kColorMarker);
      
      NvtxScopedRange launchRange("matmul_tiled_launch", kColorKernelTiled);
      if (tileSize == 8) {
        matmulTiled<8><<<grid, block>>>(d_a, d_b, d_c, n);
      } else if (tileSize == 16) {
        matmulTiled<16><<<grid, block>>>(d_a, d_b, d_c, n);
      } else if (tileSize == 32) {
        matmulTiled<32><<<grid, block>>>(d_a, d_b, d_c, n);
      }
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // NVTX Range for timing section
  NvtxRange timingRange;
  timingRange.start("tiled_timing_section", kColorRange);
  
  CHECK_CUDA(cudaEventRecord(start));
  {
    NvtxScopedRange timedRange("kernel_tiled_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      // NVTX Marker every 10 iterations
      if (i % 10 == 0) {
        char markerName[64];
        snprintf(markerName, sizeof(markerName), "tiled_timed_iter_%d", i);
        nvtxMarker(markerName, kColorMarker);
      }
      
      NvtxScopedRange launchRange("matmul_tiled_launch", kColorKernelTiled);
      if (tileSize == 8) {
        matmulTiled<8><<<grid, block>>>(d_a, d_b, d_c, n);
      } else if (tileSize == 16) {
        matmulTiled<16><<<grid, block>>>(d_a, d_b, d_c, n);
      } else if (tileSize == 32) {
        matmulTiled<32><<<grid, block>>>(d_a, d_b, d_c, n);
      }
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));
  
  timingRange.end();

  float elapsedMs = elapsed(start, stop);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

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
  matmulCpu(a, b, c, n);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1; ++i) {
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
  // NVTX Marker for program start
  nvtxMarker("Program Start", kColorMarker);
  
  int n = 1024;
  int iterations = 50;

  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iterations = std::atoi(argv[2]);
  }

  if (n <= 0 || iterations <= 0) {
    std::cerr << "Usage: ./matmul_benchmark [matrix_size] [iterations]" << std::endl;
    return EXIT_FAILURE;
  }

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c_cpu(elements, 0.0f);
  std::vector<float> h_c_naive(elements, 0.0f);
  std::vector<float> h_c_tiled(elements, 0.0f);

  // NVTX Range for CPU benchmark
  NvtxRange cpuRange;
  cpuRange.start("CPU_Benchmark", kColorRange);
  
  double cpuMs = benchmarkCpu(h_a, h_b, h_c_cpu, n, iterations);
  if (!validateOutput(h_c_cpu, 2.0f * static_cast<float>(n))) {
    return EXIT_FAILURE;
  }
  double cpuGflops = gflopsFromMs(n, cpuMs);
  
  cpuRange.end();
  nvtxMarker("CPU Benchmark Complete", kColorMarker);

  // NVTX Push/Pop for GPU memory allocation
  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  
  {
    NvtxScopedRange allocRange("GPU_Memory_Allocation", kColorMemcpyH2D);
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
  }

  {
    NvtxScopedRange range("memcpy_h2d_a", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  }
  {
    NvtxScopedRange range("memcpy_h2d_b", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  }

  nvtxMarker("Memory Transfer Complete", kColorMarker);

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Matrix size: " << n << "x" << n
            << ", iterations: " << iterations << "\n\n";

  std::cout << "CPU Execution: " << std::fixed << std::setprecision(3) 
            << cpuMs << " ms\n\n";
  std::cout << std::string(90, '-') << "\n";
  std::cout << "\t\t\tGPU Execution\n";
  std::cout << std::string(90, '-') << "\n";
  std::cout << std::left 
            << std::setw(12) << "TileSize"
            << std::setw(18) << "Naive Time(ms)"
            << std::setw(18) << "Shared Time(ms)"
            << std::setw(12) << "Naive/CPU"
            << std::setw(18) << "Shared/CPU"
            << std::setw(12) << "Shared/Naive"
            << "\n";
  std::cout << std::string(90, '-') << "\n";

  // NVTX Range for all GPU benchmarks
  NvtxRange gpuBenchmarkRange;
  gpuBenchmarkRange.start("All_GPU_Benchmarks", kColorRange);
  
  const int tileSizes[] = {8, 16, 32};
  for (int tileSize : tileSizes) {
    if (tileSize * tileSize > prop.maxThreadsPerBlock) {
      continue;
    }

    // NVTX Marker for each tile size
    char markerName[64];
    snprintf(markerName, sizeof(markerName), "Benchmarking TileSize=%d", tileSize);
    nvtxMarker(markerName, kColorMarker);
    
    // NVTX Push/Pop for each tile size benchmark
    NvtxScopedRange tileBenchmark("Tile_Size_Benchmark", kColorKernelWarmup);

    double executiontime = gpuMM(d_a, d_b, d_c, n, tileSize, iterations);
    double executiontime_shared = gpuMM_shared(d_a, d_b, d_c, n, tileSize, iterations);
    double speedupNaiveVsCpu = cpuMs / executiontime;
    double speedupSharedVsCpu = cpuMs / executiontime_shared;
    double speeduptiledvsnaive = executiontime / executiontime_shared;
    double seconds = executiontime / 1e3;
    double seconds_shared = executiontime_shared / 1e3;
    double gflops = gflopsFromMs(n, executiontime);
    double gflops_shared = gflopsFromMs(n, executiontime_shared);
    
    std::cout << std::left
              << std::setw(12) << tileSize
              << std::fixed << std::setprecision(3)
              << std::setw(18) << executiontime
              << std::fixed << std::setprecision(3)
              << std::setw(18) << executiontime_shared
              << std::setw(12) << speedupNaiveVsCpu
              << std::setw(18) << speedupSharedVsCpu
              << std::setw(12) << speeduptiledvsnaive
              << "\n";
  }
  
  gpuBenchmarkRange.end();
  nvtxMarker("All GPU Benchmarks Complete", kColorMarker);

  {
    NvtxScopedRange range("memcpy_d2h_c_naive", kColorMemcpyD2H);
    CHECK_CUDA(cudaMemcpy(h_c_naive.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }

  if (!validateOutput(h_c_naive, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  {
    NvtxScopedRange range("memcpy_d2h_c_tiled", kColorMemcpyD2H);
    CHECK_CUDA(cudaMemcpy(h_c_tiled.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }

  if (!validateOutput(h_c_tiled, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  // NVTX Push/Pop for cleanup
  {
    NvtxScopedRange cleanupRange("GPU_Memory_Cleanup", kColorMemcpyD2H);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
  }

  nvtxMarker("Program End", kColorMarker);
  
  return EXIT_SUCCESS;
}
