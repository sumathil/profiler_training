#include <cuda_runtime.h>

#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

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

double benchmarkKernel(const float *d_a, const float *d_b, float *d_c, int n,
                       int tileSize, int iterations) {
  dim3 block(tileSize, tileSize);
  dim3 grid((n + tileSize - 1) / tileSize, (n + tileSize - 1) / tileSize);
  // This is for warm up
  for (int i = 0; i < 1; ++i) {
    if (tileSize == 16) {
      matmulTiled<16><<<grid, block>>>(d_a, d_b, d_c, n);
    } else if (tileSize == 32) {
      matmulTiled<32><<<grid, block>>>(d_a, d_b, d_c, n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    if (tileSize == 16) {
      matmulTiled<16><<<grid, block>>>(d_a, d_b, d_c, n);
    } else if (tileSize == 32) {
      matmulTiled<32><<<grid, block>>>(d_a, d_b, d_c, n);
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

int main(int argc, char **argv) {
  int n = 1024;
  int iterations = 50;

  if (argc > 1) {
    n = std::atoi(argv[1]);
  }
  if (argc > 2) {
    iterations = std::atoi(argv[2]);
  }

  if (n <= 0 || iterations <= 0) {
    std::cerr << "Usage: ./matmul_tiled_benchmark [matrix_size] [iterations]"
              << std::endl;
    return EXIT_FAILURE;
  }

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c(elements, 0.0f);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Matrix size: " << n << "x" << n
            << ", iterations: " << iterations << "\n\n";
  std::cout << "TileSize,AvgKernelMs,EstimatedGFLOPS\n";

  const int tileSizes[] = {16, 32};
  for (int tileSize : tileSizes) {
    if (tileSize * tileSize > prop.maxThreadsPerBlock) {
      continue;
    }

    double avgMs = benchmarkKernel(d_a, d_b, d_c, n, tileSize, iterations);
    double seconds = avgMs / 1e3;
    double gflops = (2.0 * static_cast<double>(n) * n * n) / (seconds * 1e9);

    std::cout << tileSize << "," << avgMs << "," << gflops << "\n";
  }

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  float expected = 2.0f * static_cast<float>(n);
  for (int i = 0; i < 5 && i < static_cast<int>(elements); ++i) {
    if (std::fabs(h_c[i] - expected) > 1e-3f) {
      std::cerr << "Validation failed at index " << i << std::endl;
      CHECK_CUDA(cudaFree(d_a));
      CHECK_CUDA(cudaFree(d_b));
      CHECK_CUDA(cudaFree(d_c));
      return EXIT_FAILURE;
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return EXIT_SUCCESS;
}
