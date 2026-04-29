#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

// ===== CPU TIMER =====
double cpuTimer() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ===== CUDA EVENT TIMER =====
class CudaTimer {
private:
  cudaEvent_t start, stop;
  
public:
  CudaTimer() {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }
  
  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  
  void startTimer() {
    CHECK_CUDA(cudaEventRecord(start, 0));
  }
  
  float stopTimer() {
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ===== CPU REFERENCE IMPLEMENTATION =====
void stencil5PointCPU(const float *in, float *out, int nx, int ny) {
  // Interior points
  for (int y = 1; y < ny - 1; y++) {
    for (int x = 1; x < nx - 1; x++) {
      int idx = y * nx + x;
      float center = in[idx];
      float north = in[idx - nx];
      float south = in[idx + nx];
      float west = in[idx - 1];
      float east = in[idx + 1];
      
      out[idx] = 0.5f * center + 0.125f * (north + south + west + east);
    }
  }
  
  // Boundaries remain unchanged
  for (int x = 0; x < nx; x++) {
    out[x] = in[x];
    out[(ny-1) * nx + x] = in[(ny-1) * nx + x];
  }
  for (int y = 0; y < ny; y++) {
    out[y * nx] = in[y * nx];
    out[y * nx + nx - 1] = in[y * nx + nx - 1];
  }
}

void stencil5PointCPUMultiStep(const float *in, float *out, int nx, int ny, int timeSteps) {
  float *temp1 = (float*)malloc(nx * ny * sizeof(float));
  float *temp2 = (float*)malloc(nx * ny * sizeof(float));
  
  memcpy(temp1, in, nx * ny * sizeof(float));
  
  for (int t = 0; t < timeSteps; t++) {
    stencil5PointCPU(temp1, temp2, nx, ny);
    float *swap = temp1;
    temp1 = temp2;
    temp2 = swap;
  }
  
  memcpy(out, temp1, nx * ny * sizeof(float));
  
  free(temp1);
  free(temp2);
}

// ===== TEMPORAL BLOCKING KERNEL =====
__global__ void stencil5PointTemporalKernel(const float *in, float *out, 
                                             int nx, int ny, int timeSteps) {
  extern __shared__ float smem[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * blockDim.x + tx;
  const int gy = blockIdx.y * blockDim.y + ty;
  
  const int halo = timeSteps;
  const int tileWidth = blockDim.x + 2 * halo;
  const int tileHeight = blockDim.y + 2 * halo;
  const int tileSize = tileWidth * tileHeight;
  
  float *tile0 = smem;
  float *tile1 = smem + tileSize;
  
  // ===== LOAD TILE WITH HALO =====
  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int threadId = ty * blockDim.x + tx;
  
  for (int offset = threadId; offset < tileSize; offset += threadsPerBlock) {
    int sy = offset / tileWidth;
    int sx = offset % tileWidth;
    
    int load_gx = blockIdx.x * blockDim.x + sx - halo;
    int load_gy = blockIdx.y * blockDim.y + sy - halo;
    
    int clamped_gx = max(0, min(nx - 1, load_gx));
    int clamped_gy = max(0, min(ny - 1, load_gy));
    
    tile0[offset] = in[clamped_gy * nx + clamped_gx];
  }
  
  __syncthreads();
  
  // ===== PERFORM TEMPORAL ITERATIONS =====
  float *readBuf = tile0;
  float *writeBuf = tile1;
  
  for (int t = 0; t < timeSteps; t++) {
    int margin = t + 1;
    
    for (int offset = threadId; offset < tileSize; offset += threadsPerBlock) {
      int sy = offset / tileWidth;
      int sx = offset % tileWidth;
      
      int curr_gx = blockIdx.x * blockDim.x + sx - halo;
      int curr_gy = blockIdx.y * blockDim.y + sy - halo;
      
      bool inComputeRegion = (sx >= margin && sx < tileWidth - margin &&
                              sy >= margin && sy < tileHeight - margin);
      bool isGlobalInterior = (curr_gx > 0 && curr_gx < nx - 1 &&
                               curr_gy > 0 && curr_gy < ny - 1);
      
      if (inComputeRegion && isGlobalInterior) {
        float center = readBuf[offset];
        float north = readBuf[offset - tileWidth];
        float south = readBuf[offset + tileWidth];
        float west = readBuf[offset - 1];
        float east = readBuf[offset + 1];
        
        writeBuf[offset] = 0.5f * center + 0.125f * (north + south + west + east);
      } else {
        writeBuf[offset] = readBuf[offset];
      }
    }
    
    __syncthreads();
    
    float *tmp = readBuf;
    readBuf = writeBuf;
    writeBuf = tmp;
  }
  
  // ===== WRITE RESULT =====
  if (gx < nx && gy < ny) {
    int sx = tx + halo;
    int sy = ty + halo;
    int tileIdx = sy * tileWidth + sx;
    int globalIdx = gy * nx + gx;
    
    out[globalIdx] = readBuf[tileIdx];
  }
}

// ===== SIMPLE KERNEL (NO TEMPORAL BLOCKING) =====
__global__ void stencil5PointSimpleKernel(const float *in, float *out, int nx, int ny) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
    int idx = y * nx + x;
    float center = in[idx];
    float north = in[idx - nx];
    float south = in[idx + nx];
    float west = in[idx - 1];
    float east = in[idx + 1];
    
    out[idx] = 0.5f * center + 0.125f * (north + south + west + east);
  }
}

// ===== VALIDATION =====
bool validateResults(const float *gpu, const float *cpu, int nx, int ny, 
                     float tolerance, int &errorCount, float &maxError) {
  errorCount = 0;
  maxError = 0.0f;
  bool passed = true;
  
  for (int i = 0; i < nx * ny; i++) {
    float diff = fabsf(gpu[i] - cpu[i]);
    if (diff > tolerance) {
      if (errorCount < 5) {
        int y = i / nx;
        int x = i % nx;
        printf("      Error at (%d,%d): CPU=%.6f, GPU=%.6f, diff=%.6e\n",
               x, y, cpu[i], gpu[i], diff);
      }
      errorCount++;
      passed = false;
    }
    if (diff > maxError) {
      maxError = diff;
    }
  }
  
  return passed;
}

// ===== PERFORMANCE METRICS =====
void printPerformanceMetrics(int nx, int ny, int timeSteps, 
                            double cpuTime, float gpuTime) {
  long long totalOps = (long long)(nx - 2) * (ny - 2) * timeSteps * 5; // 5-point stencil
  double cpuGFlops = (totalOps / 1e9) / cpuTime;
  double gpuGFlops = (totalOps / 1e9) / (gpuTime / 1000.0);
  
  long long bytesAccessed = (long long)nx * ny * sizeof(float) * (timeSteps + 1) * 2; // Read + write
  double cpuBandwidth = (bytesAccessed / 1e9) / cpuTime;
  double gpuBandwidth = (bytesAccessed / 1e9) / (gpuTime / 1000.0);
  
  printf("      Operations: %.2f M\n", totalOps / 1e6);
  printf("      CPU: %.2f ms (%.2f GFlop/s, %.2f GB/s)\n", 
         cpuTime * 1000, cpuGFlops, cpuBandwidth);
  printf("      GPU: %.2f ms (%.2f GFlop/s, %.2f GB/s)\n", 
         gpuTime, gpuGFlops, gpuBandwidth);
  printf("      Speedup: %.2fx\n", cpuTime * 1000.0 / gpuTime);
}

// ===== TEST FUNCTION =====
void testTemporalBlocking(dim3 blockDim, int nx, int ny, int timeSteps,
                          float *d_in, float *d_out, float *h_in, 
                          float *h_out_cpu, float *h_out_gpu,
                          int warmupRuns, int timingRuns) {
  
  dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
               (ny + blockDim.y - 1) / blockDim.y);
  
  int halo = timeSteps;
  int tileWidth = blockDim.x + 2 * halo;
  int tileHeight = blockDim.y + 2 * halo;
  int smemSize = 2 * tileWidth * tileHeight * sizeof(float);
  
  // Check shared memory limit
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (smemSize > prop.sharedMemPerBlock) {
    printf("  T=%d: ⚠️  SKIPPED (smem: %.1f KB > %.1f KB)\n",
           timeSteps, smemSize / 1024.0, prop.sharedMemPerBlock / 1024.0);
    return;
  }
  
  printf("  T=%d (grid=%dx%d, smem=%.1f KB):\n", 
         timeSteps, gridDim.x, gridDim.y, smemSize / 1024.0);
  
  // Warmup
  for (int i = 0; i < warmupRuns; i++) {
    stencil5PointTemporalKernel<<<gridDim, blockDim, smemSize>>>(
      d_in, d_out, nx, ny, timeSteps);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  
  // GPU timing with CUDA events
  CudaTimer timer;
  timer.startTimer();
  for (int i = 0; i < timingRuns; i++) {
    stencil5PointTemporalKernel<<<gridDim, blockDim, smemSize>>>(
      d_in, d_out, nx, ny, timeSteps);
  }
  float gpuTimeMs = timer.stopTimer() / timingRuns;
  CHECK_CUDA(cudaGetLastError());
  
  // Copy result back
  CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, nx * ny * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  
  // CPU timing
  double cpuStart = cpuTimer();
  stencil5PointCPUMultiStep(h_in, h_out_cpu, nx, ny, timeSteps);
  double cpuTime = cpuTimer() - cpuStart;
  
  // Validate
  int errorCount;
  float maxError;
  bool passed = validateResults(h_out_gpu, h_out_cpu, nx, ny, 1e-4f, 
                                 errorCount, maxError);
  
  if (passed) {
    printf("      ✓ PASSED (maxError=%.2e)\n", maxError);
  } else {
    printf("      ✗ FAILED (%d errors, maxError=%.2e)\n", errorCount, maxError);
  }
  
  printPerformanceMetrics(nx, ny, timeSteps, cpuTime, gpuTimeMs);
  printf("\n");
}

// ===== BASELINE COMPARISON =====
void testBaseline(dim3 blockDim, int nx, int ny, int timeSteps,
                  float *d_in, float *d_out, float *d_temp,
                  int warmupRuns, int timingRuns) {
  
  dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
               (ny + blockDim.y - 1) / blockDim.y);
  
  // Warmup
  for (int i = 0; i < warmupRuns; i++) {
    float *src = d_in;
    float *dst = d_temp;
    for (int t = 0; t < timeSteps; t++) {
      stencil5PointSimpleKernel<<<gridDim, blockDim>>>(src, dst, nx, ny);
      float *swap = src;
      src = dst;
      dst = swap;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  
  // Timing
  CudaTimer timer;
  timer.startTimer();
  for (int i = 0; i < timingRuns; i++) {
    float *src = d_in;
    float *dst = d_temp;
    for (int t = 0; t < timeSteps; t++) {
      stencil5PointSimpleKernel<<<gridDim, blockDim>>>(src, dst, nx, ny);
      float *swap = src;
      src = dst;
      dst = swap;
    }
  }
  float gpuTimeMs = timer.stopTimer() / timingRuns;
  
  // Copy final result
  float *finalBuf = (timeSteps % 2 == 0) ? d_in : d_temp;
  CHECK_CUDA(cudaMemcpy(d_out, finalBuf, nx * ny * sizeof(float), 
                        cudaMemcpyDeviceToDevice));
  
  printf("  Baseline (no temporal blocking): %.2f ms\n", gpuTimeMs);
  
  long long totalOps = (long long)(nx - 2) * (ny - 2) * timeSteps * 5;
  double gpuGFlops = (totalOps / 1e9) / (gpuTimeMs / 1000.0);
  printf("    %.2f GFlop/s\n\n", gpuGFlops);
}

// ===== MAIN =====
int main(int argc, char **argv) {
  int nx = 2048;
  int ny = 2048;
  
  if (argc > 1) nx = atoi(argv[1]);
  if (argc > 2) ny = atoi(argv[2]);
  
  const int size = nx * ny;
  const int warmupRuns = 3;
  const int timingRuns = 10;
  
  printf("╔════════════════════════════════════════════════════════╗\n");
  printf("║     Temporal Blocking Stencil Performance Test        ║\n");
  printf("╚════════════════════════════════════════════════════════╝\n\n");
  printf("Grid size: %d x %d (%.2f M points)\n", nx, ny, size / 1e6);
  printf("Warmup runs: %d, Timing runs: %d\n\n", warmupRuns, timingRuns);
  
  // GPU info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Shared memory per block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
  printf("Memory bandwidth: %.1f GB/s\n\n", 
         2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
  
  // Allocate memory
  float *h_in = (float*)malloc(size * sizeof(float));
  float *h_out_cpu = (float*)malloc(size * sizeof(float));
  float *h_out_gpu = (float*)malloc(size * sizeof(float));
  
  // Initialize
  for (int i = 0; i < size; i++) {
    h_in[i] = (float)(rand() % 100) / 100.0f;
  }
  
  float *d_in, *d_out, *d_temp;
  CHECK_CUDA(cudaMalloc(&d_in, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_temp, size * sizeof(float)));
  
  // Copy to device
  double transferStart = cpuTimer();
  CHECK_CUDA(cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice));
  double transferTime = cpuTimer() - transferStart;
  printf("Host→Device transfer: %.2f ms (%.2f GB/s)\n\n", 
         transferTime * 1000, (size * sizeof(float) / 1e9) / transferTime);
  
  // Test configurations
  dim3 blockSizes[] = {
    dim3(16, 16),
    dim3(32, 8),
    dim3(32, 16),
  };
  
  int timeStepsArray[] = {1, 2, 4, 8};
  
  printf("═══════════════════════════════════════════════════════\n");
  printf("PERFORMANCE TESTS\n");
  printf("═══════════════════════════════════════════════════════\n\n");
  
  for (int b = 0; b < 3; b++) {
    printf("┌─ Block size: %dx%d (%d threads) ─────────────────────\n", 
           blockSizes[b].x, blockSizes[b].y, 
           blockSizes[b].x * blockSizes[b].y);
    
    for (int t = 0; t < 4; t++) {
      testTemporalBlocking(blockSizes[b], nx, ny, timeStepsArray[t],
                          d_in, d_out, h_in, h_out_cpu, h_out_gpu,
                          warmupRuns, timingRuns);
    }
    
    // Baseline for T=4
    printf("  Baseline comparison (T=4):\n");
    testBaseline(blockSizes[b], nx, ny, 4, d_in, d_out, d_temp, 
                 warmupRuns, timingRuns);
    
    printf("└───────────────────────────────────────────────────────\n\n");
  }
  
  // Cleanup
  free(h_in);
  free(h_out_cpu);
  free(h_out_gpu);
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_temp));
  
  printf("Tests complete!\n");
  return 0;
}
