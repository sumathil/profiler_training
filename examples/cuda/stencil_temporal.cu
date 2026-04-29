#include <cuda_runtime.h>
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// Temporal blocking: performs multiple iterations in shared memory
__global__ void stencil5PointTemporalKernel(const float *in, float *out, 
                                             int nx, int ny, int timeSteps) {
  extern __shared__ float smem[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x * blockDim.x;
  const int by = blockIdx.y * blockDim.y;
  const int gx = bx + tx;
  const int gy = by + ty;
  
  // We need halo of size 'timeSteps' on each side
  const int halo = timeSteps;
  const int tileWidth = blockDim.x + 2 * halo;
  const int tileHeight = blockDim.y + 2 * halo;
  const int tileSize = tileWidth * tileHeight;
  
  // Split shared memory into two buffers for ping-pong
  float *tile0 = smem;
  float *tile1 = smem + tileSize;
  
  // Cooperatively load entire tile including halo
  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int loadsPerThread = (tileSize + threadsPerBlock - 1) / threadsPerBlock;
  
  for (int i = 0; i < loadsPerThread; ++i) {
    int flatIdx = ty * blockDim.x + tx + i * threadsPerBlock;
    if (flatIdx < tileSize) {
      int sy = flatIdx / tileWidth;
      int sx = flatIdx % tileWidth;
      
      int gx_load = bx + sx - halo;
      int gy_load = by + sy - halo;
      
      float val = 0.0f;
      if (gx_load >= 0 && gx_load < nx && gy_load >= 0 && gy_load < ny) {
        val = in[gy_load * nx + gx_load];
      } else {
        // Out of bounds - use boundary value if available
        if (gx_load < 0) gx_load = 0;
        if (gx_load >= nx) gx_load = nx - 1;
        if (gy_load < 0) gy_load = 0;
        if (gy_load >= ny) gy_load = ny - 1;
        val = in[gy_load * nx + gx_load];
      }
      tile0[flatIdx] = val;
    }
  }
  
  __syncthreads();
  
  // Perform timeSteps iterations in shared memory
  float *readBuf = tile0;
  float *writeBuf = tile1;
  
  for (int t = 0; t < timeSteps; ++t) {
    // Each thread computes its stencil result
    // Account for shrinking valid region as we iterate
    int lx = tx + halo - t;
    int ly = ty + halo - t;
    
    // Check if this thread's location is still valid for computation
    bool isValid = (lx > 0 && ly > 0 && lx < tileWidth - 1 && ly < tileHeight - 1);
    
    if (isValid) {
      int idx = ly * tileWidth + lx;
      float center = readBuf[idx];
      float north = readBuf[idx - tileWidth];
      float south = readBuf[idx + tileWidth];
      float west = readBuf[idx - 1];
      float east = readBuf[idx + 1];
      
      writeBuf[idx] = 0.5f * center + 0.125f * (north + south + west + east);
    }
    
    __syncthreads();
    
    // Swap buffers for next iteration
    float *tmp = readBuf;
    readBuf = writeBuf;
    writeBuf = tmp;
  }
  
  // Write result back to global memory
  if (gx < nx && gy < ny) {
    int lx = tx + halo;
    int ly = ty + halo;
    
    // Handle boundaries
    if (gx == 0 || gy == 0 || gx == nx - 1 || gy == ny - 1) {
      out[gy * nx + gx] = in[gy * nx + gx];
    } else {
      out[gy * nx + gx] = readBuf[ly * tileWidth + lx];
    }
  }
}

// Original global memory kernel for comparison
__global__ void stencil5PointKernel(const float *in, float *out, int nx, int ny) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) return;

  int idx = y * nx + x;
  if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
    out[idx] = in[idx];
    return;
  }

  float center = in[idx];
  float north = in[(y - 1) * nx + x];
  float south = in[(y + 1) * nx + x];
  float west = in[idx - 1];
  float east = in[idx + 1];
  out[idx] = 0.5f * center + 0.125f * (north + south + west + east);
}

double benchmarkGlobal(float *d_in, float *d_out, int nx, int ny, 
                       int blockDimXY, int iterations) {
  dim3 block(blockDimXY, blockDimXY);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // Warmup
  float *inPtr = d_in;
  float *outPtr = d_out;
  for (int i = 0; i < 5; ++i) {
    stencil5PointKernel<<<grid, block>>>(inPtr, outPtr, nx, ny);
    std::swap(inPtr, outPtr);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  inPtr = d_in;
  outPtr = d_out;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    stencil5PointKernel<<<grid, block>>>(inPtr, outPtr, nx, ny);
    std::swap(inPtr, outPtr);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsedMs;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return static_cast<double>(elapsedMs) / iterations;
}

double benchmarkTemporal(float *d_in, float *d_out, int nx, int ny, 
                         int blockDimXY, int timeSteps, int outerIterations) {
  dim3 block(blockDimXY, blockDimXY);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  
  int halo = timeSteps;
  int tileWidth = blockDimXY + 2 * halo;
  int tileHeight = blockDimXY + 2 * halo;
  size_t sharedBytes = 2 * tileWidth * tileHeight * sizeof(float); // Two buffers

  // Warmup
  float *inPtr = d_in;
  float *outPtr = d_out;
  for (int i = 0; i < 2; ++i) {
    stencil5PointTemporalKernel<<<grid, block, sharedBytes>>>(inPtr, outPtr, nx, ny, timeSteps);
    std::swap(inPtr, outPtr);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  inPtr = d_in;
  outPtr = d_out;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < outerIterations; ++i) {
    stencil5PointTemporalKernel<<<grid, block, sharedBytes>>>(inPtr, outPtr, nx, ny, timeSteps);
    std::swap(inPtr, outPtr);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsedMs;
  CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // Return average time per single stencil iteration
  return static_cast<double>(elapsedMs) / (outerIterations * timeSteps);
}

int main(int argc, char **argv) {
  int nx = 4096;
  int ny = 4096;
  int totalIterations = 200;

  if (argc > 1) nx = std::atoi(argv[1]);
  if (argc > 2) ny = std::atoi(argv[2]);
  if (argc > 3) totalIterations = std::atoi(argv[3]);

  if (nx < 3 || ny < 3 || totalIterations <= 0) {
    std::cerr << "Usage: " << argv[0] << " [width>=3] [height>=3] [iterations]\n";
    return EXIT_FAILURE;
  }

  size_t elements = static_cast<size_t>(nx) * static_cast<size_t>(ny);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_in(elements);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      h_in[y * nx + x] = std::sin(x * 0.01f) + std::cos(y * 0.02f);
    }
  }

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  std::cout << "Device: " << prop.name << "\n";
  std::cout << "Grid: " << ny << "x" << nx << "\n";
  std::cout << "Total iterations: " << totalIterations << "\n\n";

  // Test global memory baseline
  std::cout << "=== Global Memory Baseline ===\n";
  int blockSizes[] = {16, 32};
  
  for (int bs : blockSizes) {
    if (bs * bs > prop.maxThreadsPerBlock) continue;
    
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    double avgMs = benchmarkGlobal(d_in, d_out, nx, ny, bs, totalIterations);
    double seconds = avgMs / 1e3;
    double bytesPerIter = elements * 6.0 * sizeof(float);
    double gbps = (bytesPerIter / 1e9) / seconds;
    std::cout << "Block " << bs << "x" << bs << ": " << avgMs << " ms, " 
              << gbps << " GB/s\n";
  }

  std::cout << "\n=== Temporal Blocking (Shared Memory) ===\n";
  std::cout << "TimeSteps | BlockSize | AvgMs | GB/s | Speedup\n";
  std::cout << "----------|-----------|-------|------|--------\n";

  // Test different temporal blocking factors
  int timeStepConfigs[] = {1, 2, 4, 8};
  double baselineMs = benchmarkGlobal(d_in, d_out, nx, ny, 16, totalIterations);

  for (int timeSteps : timeStepConfigs) {
    for (int bs : blockSizes) {
      if (bs * bs > prop.maxThreadsPerBlock) continue;
      
      int halo = timeSteps;
      int tileWidth = bs + 2 * halo;
      int tileHeight = bs + 2 * halo;
      size_t sharedBytes = 2 * tileWidth * tileHeight * sizeof(float);
      
      if (sharedBytes > prop.sharedMemPerBlock) {
        std::cout << timeSteps << "         | " << bs << "x" << bs 
                  << "       | SKIP (too much shared mem)\n";
        continue;
      }

      if (totalIterations % timeSteps != 0) continue;
      
      int outerIterations = totalIterations / timeSteps;
      
      CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
      double avgMs = benchmarkTemporal(d_in, d_out, nx, ny, bs, timeSteps, outerIterations);
      double seconds = avgMs / 1e3;
      double bytesPerIter = elements * 6.0 * sizeof(float);
      double gbps = (bytesPerIter / 1e9) / seconds;
      double speedup = baselineMs / avgMs;
      
      printf("%9d | %4dx%-4d | %5.3f | %4.1f | %.2fx\n", 
             timeSteps, bs, bs, avgMs, gbps, speedup);
    }
  }

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return EXIT_SUCCESS;
}
