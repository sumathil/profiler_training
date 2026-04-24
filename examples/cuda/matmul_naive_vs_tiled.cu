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
constexpr uint32_t kColorMemcpyH2D = 0xFF1E88E5;     // Blue
constexpr uint32_t kColorMemcpyD2H = 0xFF43A047;     // Green
constexpr uint32_t kColorKernelNaive = 0xFFE53935;   // Red
constexpr uint32_t kColorKernelTiled = 0xFF00ACC1;   // Cyan
constexpr uint32_t kColorKernelWarmup = 0xFF8E24AA;  // Purple
constexpr uint32_t kColorKernelTimed = 0xFFFB8C00;   // Orange

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
}  // namespace

__global__ void matmulNaive(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n || col >= n) {
    return;
  }

  float sum = 0.0f;
  int rowBase = row * n;
  for (int k = 0; k < n; ++k) {
    sum += a[rowBase + k] * b[k * n + col];
  }
  c[rowBase + col] = sum;
}

template <int TILE>
__global__ void matmulTiled(const float *a, const float *b, float *c, int n) {
  __shared__ float tileA[TILE][TILE];
  __shared__ float tileB[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  int numTiles = (n + TILE - 1) / TILE;
  float sum = 0.0f;

  for (int t = 0; t < numTiles; ++t) {
    int aCol = t * TILE + tx;
    int bRow = t * TILE + ty;
    tileA[ty][tx] = (row < n && aCol < n) ? a[row * n + aCol] : 0.0f;
    tileB[ty][tx] = (bRow < n && col < n) ? b[bRow * n + col] : 0.0f;

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

double benchmarkNaive(const float *d_a, const float *d_b, float *d_c, int n,
                      int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);

  {
    NvtxScopedRange warmupRange("naive_warmup", kColorKernelWarmup);
    for (int i = 0; i < 3; ++i) {
      NvtxScopedRange launchRange("matmul_naive_launch", kColorKernelNaive);
      matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  {
    NvtxScopedRange timedRange("naive_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      NvtxScopedRange launchRange("matmul_naive_launch", kColorKernelNaive);
      matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsedMs = elapsed(start, stop);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

template <int TILE>
double benchmarkTiledImpl(const float *d_a, const float *d_b, float *d_c, int n, int iterations) {
  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

  {
    NvtxScopedRange warmupRange("tiled_warmup", kColorKernelWarmup);
    for (int i = 0; i < 3; ++i) {
      NvtxScopedRange launchRange("matmul_tiled_launch", kColorKernelTiled);
      matmulTiled<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  {
    NvtxScopedRange timedRange("tiled_timed", kColorKernelTimed);
    for (int i = 0; i < iterations; ++i) {
      NvtxScopedRange launchRange("matmul_tiled_launch", kColorKernelTiled);
      matmulTiled<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsedMs = elapsed(start, stop);
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

double benchmarkTiled(const float *d_a, const float *d_b, float *d_c, int n,
                      int tileSize, int iterations) {
  if (tileSize == 8) {
    return benchmarkTiledImpl<8>(d_a, d_b, d_c, n, iterations);
  }
  if (tileSize == 16) {
    return benchmarkTiledImpl<16>(d_a, d_b, d_c, n, iterations);
  }
  if (tileSize == 32) {
    return benchmarkTiledImpl<32>(d_a, d_b, d_c, n, iterations);
  }
  std::cerr << "Unsupported tile size: " << tileSize
            << " (supported: 8, 16, 32)" << std::endl;
  return -1.0;
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

int main(int argc, char **argv) {
  int n = 2048;
  int iterations = 50;
  int tileSize = 16;
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iterations = std::atoi(argv[2]);
  }
  if (argc > 3) {
    tileSize = std::atoi(argv[3]);
  }

  if (n <= 0 || iterations <= 0 || tileSize <= 0) {
    std::cerr << "Usage: ./matmul_naive_vs_tiled [matrix_size] [iterations] [tile_size]\n"
              << "Example: ./matmul_naive_vs_tiled 2048 50 16" << std::endl;
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  if (tileSize * tileSize > prop.maxThreadsPerBlock) {
    std::cerr << "tile_size " << tileSize
              << " exceeds max threads per block on this GPU ("
              << prop.maxThreadsPerBlock << ")" << std::endl;
    return EXIT_FAILURE;
  }

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);
  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c_cpu(elements, 0.0f);
  std::vector<float> h_c_naive(elements, 0.0f);
  std::vector<float> h_c_tiled(elements, 0.0f);

  double cpuMs = benchmarkCpu(h_a, h_b, h_c_cpu, n, iterations);
  if (!validateOutput(h_c_cpu, 2.0f * static_cast<float>(n))) {
    return EXIT_FAILURE;
  }

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));
  {
    NvtxScopedRange range("memcpy_h2d_a", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  }
  {
    NvtxScopedRange range("memcpy_h2d_b", kColorMemcpyH2D);
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
  }

  double naiveMs = benchmarkNaive(d_a, d_b, d_c, n, tileSize, iterations);
  {
    NvtxScopedRange range("memcpy_d2h_naive", kColorMemcpyD2H);
    CHECK_CUDA(cudaMemcpy(h_c_naive.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }
  if (!validateOutput(h_c_naive, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  double tiledMs = benchmarkTiled(d_a, d_b, d_c, n, tileSize, iterations);
  if (tiledMs < 0.0) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }
  {
    NvtxScopedRange range("memcpy_d2h_tiled", kColorMemcpyD2H);
    CHECK_CUDA(cudaMemcpy(h_c_tiled.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }
  if (!validateOutput(h_c_tiled, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  float maxDiff = 0.0f;
  for (size_t i = 0; i < elements; ++i) {
    maxDiff = std::max(maxDiff, std::fabs(h_c_naive[i] - h_c_tiled[i]));
  }

  double cpuGflops = gflopsFromMs(n, cpuMs);
  double naiveGflops = gflopsFromMs(n, naiveMs);
  double tiledGflops = gflopsFromMs(n, tiledMs);
  double speedupTiledVsNaive = naiveMs / tiledMs;
  double speedupNaiveVsCpu = cpuMs / naiveMs;
  double speedupTiledVsCpu = cpuMs / tiledMs;

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "N=" << n << ", iterations=" << iterations
            << ", tile=" << tileSize << "\n";
  std::cout << "Kernel,AvgKernelMs,EstimatedGFLOPS\n";
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "cpu," << cpuMs << "," << cpuGflops << "\n";
  std::cout << "naive," << naiveMs << "," << naiveGflops << "\n";
  std::cout << "tiled," << tiledMs << "," << tiledGflops << "\n";
  std::cout << "speedup_naive_vs_cpu," << speedupNaiveVsCpu << "\n";
  std::cout << "speedup_tiled_vs_cpu," << speedupTiledVsCpu << "\n";
  std::cout << "speedup_tiled_vs_naive," << speedupTiledVsNaive << "\n";
  std::cout << "max_abs_diff," << std::setprecision(8) << maxDiff << "\n";

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return EXIT_SUCCESS;
}
