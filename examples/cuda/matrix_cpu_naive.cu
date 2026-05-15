#include <cuda_runtime.h>
#include <nvToolsExt.h>

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
constexpr uint32_t kColorKernelWarmup = 0xFF8E24AA;   // Purple
constexpr uint32_t kColorKernelTimed = 0xFFFB8C00;    // Orange
constexpr uint32_t kColorMarker = 0xFFFF00FF;         // Magenta
constexpr uint32_t kColorRange = 0xFF00FFFF;          // Cyan
constexpr uint32_t kColorCPU = 0xFFFFFFFF;            // White

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

void matmulCpu(const std::vector<float> &a, const std::vector<float> &b,
               std::vector<float> &c, int n) {
  // NVTX Push/Pop for CPU computation
  NvtxScopedRange cpuCompute("CPU_Matrix_Multiply", kColorCPU);
  
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

double gpuMM(const float *d_a, const float *d_b, float *d_c, int n,
             int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);

  // NVTX Push/Pop for warmup
  {
    NvtxScopedRange warmupRange("kernel_naive_warmup", kColorKernelWarmup);
    for (int i = 0; i < 3; ++i) {
      // NVTX Marker for each warmup iteration
      char markerName[64];
      snprintf(markerName, sizeof(markerName), "warmup_iteration_%d", i);
      nvtxMarker(markerName, kColorMarker);
      
      NvtxScopedRange launchRange("matmul_naive_launch", kColorKernelNaive);
      matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  nvtxMarker("Warmup Complete", kColorMarker);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // NVTX Range for timing section
  NvtxRange timingRange;
  timingRange.start("GPU_Timing_Section", kColorRange);

  CHECK_CUDA(cudaEventRecord(start));
  {
    NvtxScopedRange timedRange("kernel_naive_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      // NVTX Marker every 10 iterations
      if (i % 10 == 0) {
        char markerName[64];
        snprintf(markerName, sizeof(markerName), "timed_iteration_%d", i);
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

double benchmarkCpu(const std::vector<float> &a, const std::vector<float> &b,
                    std::vector<float> &c, int n, int iterations) {
  // NVTX Push/Pop for initial CPU run
  {
    NvtxScopedRange initialRun("CPU_Initial_Run", kColorCPU);
    matmulCpu(a, b, c, n);
  }
  
  nvtxMarker("CPU Initial Run Complete", kColorMarker);

  // NVTX Range for CPU timing
  NvtxRange cpuTimingRange;
  cpuTimingRange.start("CPU_Timing_Section", kColorRange);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1; ++i) {
    matmulCpu(a, b, c, n);
  }
  auto stop = std::chrono::high_resolution_clock::now();
  
  cpuTimingRange.end();

  double elapsedMs = std::chrono::duration<double, std::milli>(stop - start).count();
  return elapsedMs / static_cast<double>(iterations);
}

double gflopsFromMs(int n, double avgMs) {
  double seconds = avgMs / 1e3;
  return (2.0 * static_cast<double>(n) * n * n) / (seconds * 1e9);
}

bool validateOutput(const std::vector<float> &out, float expected) {
  // NVTX Push/Pop for validation
  NvtxScopedRange validationRange("Output_Validation", kColorRange);
  
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
  
  // NVTX Push/Pop for initialization
  NvtxScopedRange initRange("Program_Initialization", kColorRange);
  
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
  
  // End initialization
  initRange.~NvtxScopedRange();
  nvtxMarker("Initialization Complete", kColorMarker);

  // NVTX Range for CPU benchmark
  NvtxRange cpuBenchmarkRange;
  cpuBenchmarkRange.start("CPU_Benchmark", kColorRange);
  
  double cpuMs = benchmarkCpu(h_a, h_b, h_c_cpu, n, iterations);
  if (!validateOutput(h_c_cpu, 2.0f * static_cast<float>(n))) {
    return EXIT_FAILURE;
  }
  double cpuGflops = gflopsFromMs(n, cpuMs);
  
  cpuBenchmarkRange.end();
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
  
  nvtxMarker("GPU Memory Allocated", kColorMarker);

  {
    NvtxScopedRange range("memcpy_h2d_a", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  }
  {
    NvtxScopedRange range("memcpy_h2d_b", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  }
  
  nvtxMarker("Host to Device Transfer Complete", kColorMarker);

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Matrix size: " << n << "x" << n
            << ", iterations: " << iterations << "\n\n";

  std::cout << "CPU Execution " << cpuMs << " ms\n\n";
  std::cout << std::string(42, '-') << "\n";
  std::cout << "\t\tGPU Execution\n";
  std::cout << std::string(42, '-') << "\n";
  std::cout << std::left 
            << std::setw(12) << "TileSize"
            << std::setw(18) << "Exec_time(ms)"
            << std::setw(12) << "SpeedUp"
            << "\n";
  std::cout << std::string(42, '-') << "\n";

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
    
    // NVTX Push/Pop for individual tile benchmark
    NvtxScopedRange tileBenchmark("Tile_Size_Benchmark", kColorKernelWarmup);

    double executiontime = gpuMM(d_a, d_b, d_c, n, tileSize, iterations);
    double speedupNaiveVsCpu = cpuMs / executiontime;
    double seconds = executiontime / 1e3;
    double gflops = (2.0 * static_cast<double>(n) * n * n) / (seconds * 1e9);
    
    std::cout << std::left
              << std::setw(12) << tileSize
              << std::fixed << std::setprecision(3)
              << std::setw(18) << executiontime
              << std::setw(12) << speedupNaiveVsCpu
              << "\n";
  }
  
  gpuBenchmarkRange.end();
  nvtxMarker("All GPU Benchmarks Complete", kColorMarker);

  {
    NvtxScopedRange range("memcpy_d2h_c", kColorMemcpyD2H);
    CHECK_CUDA(cudaMemcpy(h_c_naive.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }
  
  nvtxMarker("Device to Host Transfer Complete", kColorMarker);
  
  if (!validateOutput(h_c_naive, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }
  
  nvtxMarker("Validation Complete", kColorMarker);

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