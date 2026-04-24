#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t err = (call);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                << static_cast<int>(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

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
  __shared__ float tileA[TILE][TILE + 1];  // +1 for padding to avoid bank conflicts
  __shared__ float tileB[TILE][TILE + 1];

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

bool validateOutput(const std::vector<float> &out, float expected) {
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::fabs(out[i] - expected) > 1e-2f) {
      std::cerr << "Validation failed at index " << i << ": got " << out[i]
                << ", expected " << expected << std::endl;
      return false;
    }
  }
  return true;
}

double benchmarkNaive(const float *d_a, const float *d_b, float *d_c, int n,
                      int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
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
double benchmarkTiledImpl(const float *d_a, const float *d_b, float *d_c, int n,
                          int iterations) {
  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    matmulTiled<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    matmulTiled<TILE><<<grid, block>>>(d_a, d_b, d_c, n);
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

double benchmarkCublasAsync(int n, int iterations, int streams,
                            std::vector<float> &h_c_out) {
  if (streams < 2) {
    std::cerr << "cublas_async requires streams >= 2 to show overlap." << std::endl;
    return -1.0;
  }

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float *> h_a_pinned(streams, nullptr);
  std::vector<float *> h_b_pinned(streams, nullptr);
  std::vector<float *> h_c_pinned(streams, nullptr);
  std::vector<float *> d_a(streams, nullptr);
  std::vector<float *> d_b(streams, nullptr);
  std::vector<float *> d_c(streams, nullptr);
  std::vector<cudaStream_t> stream(streams);
  std::vector<cublasHandle_t> handle(streams);

  for (int s = 0; s < streams; ++s) {
    CHECK_CUDA(cudaHostAlloc(&h_a_pinned[s], bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_b_pinned[s], bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_c_pinned[s], bytes, cudaHostAllocDefault));
    std::fill(h_a_pinned[s], h_a_pinned[s] + elements, 1.0f);
    std::fill(h_b_pinned[s], h_b_pinned[s] + elements, 2.0f);

    CHECK_CUDA(cudaMalloc(&d_a[s], bytes));
    CHECK_CUDA(cudaMalloc(&d_b[s], bytes));
    CHECK_CUDA(cudaMalloc(&d_c[s], bytes));
    CHECK_CUDA(cudaStreamCreate(&stream[s]));
    CUBLAS_CHECK(cublasCreate(&handle[s]));
    CUBLAS_CHECK(cublasSetStream(handle[s], stream[s]));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Warmup
  for (int i = 0; i < 3; ++i) {
    for (int s = 0; s < streams; ++s) {
      CHECK_CUDA(cudaMemcpyAsync(d_a[s], h_a_pinned[s], bytes, cudaMemcpyHostToDevice,
                                 stream[s]));
      CHECK_CUDA(cudaMemcpyAsync(d_b[s], h_b_pinned[s], bytes, cudaMemcpyHostToDevice,
                                 stream[s]));
      CUBLAS_CHECK(cublasSgemm(handle[s], CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                               d_b[s], n, d_a[s], n, &beta, d_c[s], n));
      CHECK_CUDA(cudaMemcpyAsync(h_c_pinned[s], d_c[s], bytes, cudaMemcpyDeviceToHost,
                                 stream[s]));
    }
  }
  for (int s = 0; s < streams; ++s) {
    CHECK_CUDA(cudaStreamSynchronize(stream[s]));
  }

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start, stream[0]));
  for (int i = 0; i < iterations; ++i) {
    for (int s = 0; s < streams; ++s) {
      CHECK_CUDA(cudaMemcpyAsync(d_a[s], h_a_pinned[s], bytes, cudaMemcpyHostToDevice,
                                 stream[s]));
      CHECK_CUDA(cudaMemcpyAsync(d_b[s], h_b_pinned[s], bytes, cudaMemcpyHostToDevice,
                                 stream[s]));
      CUBLAS_CHECK(cublasSgemm(handle[s], CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                               d_b[s], n, d_a[s], n, &beta, d_c[s], n));
      CHECK_CUDA(cudaMemcpyAsync(h_c_pinned[s], d_c[s], bytes, cudaMemcpyDeviceToHost,
                                 stream[s]));
    }
  }
  CHECK_CUDA(cudaEventRecord(stop, stream[0]));
  for (int s = 0; s < streams; ++s) {
    CHECK_CUDA(cudaStreamSynchronize(stream[s]));
  }

  float elapsedMs = elapsed(start, stop);
  h_c_out.assign(h_c_pinned[0], h_c_pinned[0] + elements);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  for (int s = 0; s < streams; ++s) {
    CUBLAS_CHECK(cublasDestroy(handle[s]));
    CHECK_CUDA(cudaStreamDestroy(stream[s]));
    CHECK_CUDA(cudaFree(d_a[s]));
    CHECK_CUDA(cudaFree(d_b[s]));
    CHECK_CUDA(cudaFree(d_c[s]));
    CHECK_CUDA(cudaFreeHost(h_a_pinned[s]));
    CHECK_CUDA(cudaFreeHost(h_b_pinned[s]));
    CHECK_CUDA(cudaFreeHost(h_c_pinned[s]));
  }

  return static_cast<double>(elapsedMs) / static_cast<double>(iterations);
}

double gflopsFromMs(int n, double avgMs) {
  double seconds = avgMs / 1e3;
  return (2.0 * static_cast<double>(n) * n * n) / (seconds * 1e9);
}

int main(int argc, char **argv) {
  int n = 2048;
  int iterations = 10;
  int tileSize = 16;
  int asyncStreams = 2;
  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iterations = std::atoi(argv[2]);
  }
  if (argc > 3) {
    tileSize = std::atoi(argv[3]);
  }
  if (argc > 4) {
    asyncStreams = std::atoi(argv[4]);
  }

  if (n <= 0 || iterations <= 0 || tileSize <= 0 || asyncStreams <= 0) {
    std::cerr << "Usage: ./matmul_4way_complex [matrix_size] "
              << "[iterations] [tile_size] [async_streams]\n"
              << "Example: ./matmul_4way_complex 2048 30 16 2"
              << std::endl;
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp prop{};
  //CHECK_CUDA(cudaGetDevice(&device));
  /*CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  if (tileSize * tileSize > prop.maxThreadsPerBlock) {
    std::cerr << "tile_size " << tileSize
              << " exceeds max threads per block on this GPU ("
              << prop.maxThreadsPerBlock << ")" << std::endl;
    return EXIT_FAILURE;
  }*/

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);
  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c_naive(elements, 0.0f);
  std::vector<float> h_c_tiled(elements, 0.0f);
  std::vector<float> h_c_cublas_async(elements, 0.0f);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  // Benchmark Naive
  double naiveMs = benchmarkNaive(d_a, d_b, d_c, n, tileSize, iterations);
  CHECK_CUDA(cudaMemcpy(h_c_naive.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  if (!validateOutput(h_c_naive, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  // Benchmark Tiled
  double tiledMs = benchmarkTiled(d_a, d_b, d_c, n, tileSize, iterations);
  if (tiledMs < 0.0) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }
  CHECK_CUDA(cudaMemcpy(h_c_tiled.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  if (!validateOutput(h_c_tiled, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  // Benchmark cuBLAS Async
  double cublasAsyncMs = benchmarkCublasAsync(n, iterations, asyncStreams, h_c_cublas_async);
  if (cublasAsyncMs < 0.0) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }
  if (!validateOutput(h_c_cublas_async, 2.0f * static_cast<float>(n))) {
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return EXIT_FAILURE;
  }

  // Calculate differences
  float maxDiffNaiveTiled = 0.0f;
  float maxDiffNaiveCublasAsync = 0.0f;
  for (size_t i = 0; i < elements; ++i) {
    maxDiffNaiveTiled =
        std::max(maxDiffNaiveTiled, std::fabs(h_c_naive[i] - h_c_tiled[i]));
    maxDiffNaiveCublasAsync =
        std::max(maxDiffNaiveCublasAsync, std::fabs(h_c_naive[i] - h_c_cublas_async[i]));
  }

  double naiveGflops = gflopsFromMs(n, naiveMs);
  double tiledGflops = gflopsFromMs(n, tiledMs);
  double cublasAsyncGflops = static_cast<double>(asyncStreams) * gflopsFromMs(n, cublasAsyncMs);

  // Output results
  std::cout << "Device: " << prop.name << "\n";
  std::cout << "N=" << n << ", iterations=" << iterations << ", tile=" << tileSize
            << ", async_streams=" << asyncStreams << "\n\n";
  std::cout << "Type,AvgMs,EstimatedGFLOPS\n";
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "naive," << naiveMs << "," << naiveGflops << "\n";
  std::cout << "tiled," << tiledMs << "," << tiledGflops << "\n";
  std::cout << "cublas_async_h2d_gemm_d2h," << cublasAsyncMs << ","
            << cublasAsyncGflops << "\n\n";
  std::cout << "speedup_tiled_vs_naive," << (naiveMs / tiledMs) << "\n";
  std::cout << "speedup_cublas_async_vs_naive," << (naiveMs / cublasAsyncMs) << "\n";
  std::cout << "speedup_cublas_async_vs_tiled," << (tiledMs / cublasAsyncMs) << "\n\n";
  std::cout << "max_abs_diff_naive_vs_tiled," << std::setprecision(8)
            << maxDiffNaiveTiled << "\n";
  std::cout << "max_abs_diff_naive_vs_cublas_async," << std::setprecision(8)
            << maxDiffNaiveCublasAsync << "\n";

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return EXIT_SUCCESS;
}