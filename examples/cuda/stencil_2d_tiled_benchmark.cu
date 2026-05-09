#include <cuda_runtime.h>

#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void stencil5PointTiledKernel(const float *in, float *out, int nx, int ny) {
  extern __shared__ float tile[];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * blockDim.x + tx;
  const int gy = blockIdx.y * blockDim.y + ty;

  const int tileWidth = static_cast<int>(blockDim.x) + 2;
  const int lx = tx + 1;
  const int ly = ty + 1;
  const int sidx = ly * tileWidth + lx;

  const bool inBounds = (gx < nx && gy < ny);
  float center = 0.0f;
  if (inBounds) {
    center = in[gy * nx + gx];
  }
  tile[sidx] = center;

  if (tx == 0) {
    tile[ly * tileWidth] = (inBounds && gx > 0) ? in[gy * nx + (gx - 1)] : center;
  }
  if (tx == static_cast<int>(blockDim.x) - 1) {
    tile[ly * tileWidth + (lx + 1)] =
        (inBounds && gx + 1 < nx) ? in[gy * nx + (gx + 1)] : center;
  }
  if (ty == 0) {
    tile[lx] = (inBounds && gy > 0) ? in[(gy - 1) * nx + gx] : center;
  }
  if (ty == static_cast<int>(blockDim.y) - 1) {
    tile[(ly + 1) * tileWidth + lx] =
        (inBounds && gy + 1 < ny) ? in[(gy + 1) * nx + gx] : center;
  }

  __syncthreads();

  if (!inBounds) {
    return;
  }

  const int idx = gy * nx + gx;
  if (gx == 0 || gy == 0 || gx == nx - 1 || gy == ny - 1) {
    out[idx] = center;
    return;
  }

  const float north = tile[(ly - 1) * tileWidth + lx];
  const float south = tile[(ly + 1) * tileWidth + lx];
  const float west = tile[ly * tileWidth + (lx - 1)];
  const float east = tile[ly * tileWidth + (lx + 1)];
  out[idx] = 0.5f * center + 0.125f * (north + south + west + east);
}

double benchmarkKernel(float *d_in, float *d_out, int nx, int ny, int blockDimXY,
                       int iterations) {
  dim3 block(blockDimXY, blockDimXY);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  size_t sharedBytes = static_cast<size_t>(block.x + 2) * static_cast<size_t>(block.y + 2) *
                       sizeof(float);

  float *inPtr = d_in;
  float *outPtr = d_out;
  for (int i = 0; i < 5; ++i) {
    stencil5PointTiledKernel<<<grid, block, sharedBytes>>>(inPtr, outPtr, nx, ny);
    float *tmpIn = inPtr;
    inPtr = outPtr;
    outPtr = tmpIn;
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  inPtr = d_in;
  outPtr = d_out;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    stencil5PointTiledKernel<<<grid, block, sharedBytes>>>(inPtr, outPtr, nx, ny);
    float *tmpIn = inPtr;
    inPtr = outPtr;
    outPtr = tmpIn;
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
  int nx = 4096;
  int ny = 4096;
  int iterations = 200;

  if (argc > 1) {
    nx = std::atoi(argv[1]);
  }
  if (argc > 2) {
    ny = std::atoi(argv[2]);
  }
  if (argc > 3) {
    iterations = std::atoi(argv[3]);
  }

  if (nx < 3 || ny < 3 || iterations <= 0) {
    std::cerr << "Usage: ./stencil_2d_tiled_benchmark [width>=3] [height>=3] [iterations]"
              << std::endl;
    return EXIT_FAILURE;
  }

  size_t elements = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_in(elements, 0.0f);
  std::vector<float> h_out(elements, 0.0f);
  std::vector<float> h_ref(elements, 0.0f);

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      h_in[y * nx + x] =
          std::sin(static_cast<float>(x) * 0.01f) + std::cos(static_cast<float>(y) * 0.02f);
    }
  }

  float *d_in = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Grid: " << ny << "x" << nx << ", iterations: " << iterations << "\n\n";
  std::cout << "TileDim,AvgKernelMs,EstimatedGBps\n";

  const int tileDims[] = {8, 16, 32};
  for (int tileDim : tileDims) {
    if (tileDim * tileDim > prop.maxThreadsPerBlock) {
      continue;
    }
    if (tileDim > prop.maxThreadsDim[0] || tileDim > prop.maxThreadsDim[1]) {
      continue;
    }

    size_t sharedBytes =
        static_cast<size_t>(tileDim + 2) * static_cast<size_t>(tileDim + 2) * sizeof(float);
    if (sharedBytes > prop.sharedMemPerBlock) {
      continue;
    }

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    double avgMs = benchmarkKernel(d_in, d_out, nx, ny, tileDim, iterations);
    double seconds = avgMs / 1e3;
    double bytesMoved = static_cast<double>(elements) * 6.0 * sizeof(float);
    double gbps = (bytesMoved / 1e9) / seconds;
    std::cout << tileDim << "x" << tileDim << "," << avgMs << "," << gbps << "\n";
  }

  // Reset input for validation since benchmarkKernel mutates device buffers.
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
  dim3 block(16, 16);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  size_t sharedBytes = static_cast<size_t>(block.x + 2) * static_cast<size_t>(block.y + 2) *
                       sizeof(float);
  stencil5PointTiledKernel<<<grid, block, sharedBytes>>>(d_in, d_out, nx, ny);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      int idx = y * nx + x;
      if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        h_ref[idx] = h_in[idx];
      } else {
        float center = h_in[idx];
        float north = h_in[(y - 1) * nx + x];
        float south = h_in[(y + 1) * nx + x];
        float west = h_in[idx - 1];
        float east = h_in[idx + 1];
        h_ref[idx] = 0.5f * center + 0.125f * (north + south + west + east);
      }
    }
  }

  for (size_t i = 0; i < elements; ++i) {
    if (std::fabs(h_out[i] - h_ref[i]) > 1e-5f) {
      std::cerr << "Validation failed at index " << i << ": got " << h_out[i]
                << ", expected " << h_ref[i] << std::endl;
      CHECK_CUDA(cudaFree(d_in));
      CHECK_CUDA(cudaFree(d_out));
      return EXIT_FAILURE;
    }
  }

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return EXIT_SUCCESS;
}
