#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CHECK_CUDA(call) { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
}

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
    out[x] = in[x];                           // Top
    out[(ny-1) * nx + x] = in[(ny-1) * nx + x];  // Bottom
  }
  for (int y = 0; y < ny; y++) {
    out[y * nx] = in[y * nx];                 // Left
    out[y * nx + nx - 1] = in[y * nx + nx - 1];  // Right
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

// ===== FIXED TEMPORAL BLOCKING KERNEL =====
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
  // Each thread loads multiple elements
  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int threadId = ty * blockDim.x + tx;
  
  for (int offset = threadId; offset < tileSize; offset += threadsPerBlock) {
    int sy = offset / tileWidth;
    int sx = offset % tileWidth;
    
    // Global coordinates (with halo offset)
    int load_gx = blockIdx.x * blockDim.x + sx - halo;
    int load_gy = blockIdx.y * blockDim.y + sy - halo;
    
    // Clamp to valid range
    int clamped_gx = max(0, min(nx - 1, load_gx));
    int clamped_gy = max(0, min(ny - 1, load_gy));
    
    tile0[offset] = in[clamped_gy * nx + clamped_gx];
  }
  
  __syncthreads();
  
  // ===== PERFORM TEMPORAL ITERATIONS =====
  float *readBuf = tile0;
  float *writeBuf = tile1;
  
  for (int t = 0; t < timeSteps; t++) {
    // For time step t, we need margin of (t+1) on each side
    // because cells further than that won't be used in final result
    int margin = t + 1;
    
    for (int offset = threadId; offset < tileSize; offset += threadsPerBlock) {
      int sy = offset / tileWidth;
      int sx = offset % tileWidth;
      
      // Global coordinates
      int curr_gx = blockIdx.x * blockDim.x + sx - halo;
      int curr_gy = blockIdx.y * blockDim.y + sy - halo;
      
      // Check if this is an interior point that should be computed
      bool inComputeRegion = (sx >= margin && sx < tileWidth - margin &&
                              sy >= margin && sy < tileHeight - margin);
      bool isGlobalInterior = (curr_gx > 0 && curr_gx < nx - 1 &&
                               curr_gy > 0 && curr_gy < ny - 1);
      
      if (inComputeRegion && isGlobalInterior) {
        // Compute stencil
        float center = readBuf[offset];
        float north = readBuf[offset - tileWidth];
        float south = readBuf[offset + tileWidth];
        float west = readBuf[offset - 1];
        float east = readBuf[offset + 1];
        
        writeBuf[offset] = 0.5f * center + 0.125f * (north + south + west + east);
      } else {
        // Copy unchanged (boundaries or margin region)
        writeBuf[offset] = readBuf[offset];
      }
    }
    
    __syncthreads();
    
    // Swap buffers
    float *tmp = readBuf;
    readBuf = writeBuf;
    writeBuf = tmp;
  }
  
  // ===== WRITE RESULT (only interior of block, not halo) =====
  // Only write the actual block region, not the halo
  if (gx < nx && gy < ny) {
    int sx = tx + halo;
    int sy = ty + halo;
    int tileIdx = sy * tileWidth + sx;
    int globalIdx = gy * nx + gx;
    
    out[globalIdx] = readBuf[tileIdx];
  }
}

// ===== VALIDATION =====
bool validateResults(const float *gpu, const float *cpu, int nx, int ny, 
                     float tolerance, int &errorCount, float &maxError, int &maxErrorIdx) {
  errorCount = 0;
  maxError = 0.0f;
  maxErrorIdx = -1;
  bool passed = true;
  
  for (int i = 0; i < nx * ny; i++) {
    float diff = fabsf(gpu[i] - cpu[i]);
    if (diff > tolerance) {
      if (errorCount < 10) {  // Print first 10 errors
        int y = i / nx;
        int x = i % nx;
        printf("  Error at (%d,%d) [idx %d]: CPU=%.6f, GPU=%.6f, diff=%.6e\n",
               x, y, i, cpu[i], gpu[i], diff);
      }
      errorCount++;
      passed = false;
    }
    if (diff > maxError) {
      maxError = diff;
      maxErrorIdx = i;
    }
  }
  
  return passed;
}

// ===== TEST FUNCTION =====
void testTemporalBlocking(dim3 blockDim, int nx, int ny, int timeSteps,
                          float *d_in, float *d_out, float *h_in, 
                          float *h_out_cpu, float *h_out_gpu) {
  
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
    printf("  timeSteps=%d: ⚠️  SKIPPED (smem: %d > %d bytes)\n",
           timeSteps, smemSize, prop.sharedMemPerBlock);
    return;
  }
  
  // Run GPU kernel
  stencil5PointTemporalKernel<<<gridDim, blockDim, smemSize>>>(
    d_in, d_out, nx, ny, timeSteps);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  
  // Copy result back
  CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, nx * ny * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  
  // Run CPU reference
  stencil5PointCPUMultiStep(h_in, h_out_cpu, nx, ny, timeSteps);
  
  // Validate
  int errorCount;
  float maxError;
  int maxErrorIdx;
  bool passed = validateResults(h_out_gpu, h_out_cpu, nx, ny, 1e-4f, 
                                 errorCount, maxError, maxErrorIdx);
  
  if (passed) {
    printf("  timeSteps=%d: ✓ PASSED (maxError=%.2e)\n", 
           timeSteps, maxError);
  } else {
    printf("  timeSteps=%d: ✗ FAILED (%d errors, maxError=%.2e at idx %d)\n", 
           timeSteps, errorCount, maxError, maxErrorIdx);
    
    // Print some context around max error
    if (maxErrorIdx >= 0) {
      int y = maxErrorIdx / nx;
      int x = maxErrorIdx % nx;
      printf("    Location: (%d, %d) - ", x, y);
      if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
        printf("BOUNDARY\n");
      } else {
        printf("INTERIOR\n");
      }
    }
  }
}

// ===== MAIN =====
int main(int argc, char **argv) {
  const int nx = 8192;  // Start smaller for debugging
  const int ny = 8192;
  const int size = nx * ny;
  
  printf("=== Temporal Blocking Stencil Test ===\n");
  printf("Grid: %dx%d\n\n", nx, ny);
  
  // GPU info
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Compute: %d.%d\n", prop.major, prop.minor);
  printf("Shared mem/block: %zu bytes (%.1f KB)\n\n", 
         prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
  
  // Allocate memory
  float *h_in = (float*)malloc(size * sizeof(float));
  float *h_out_cpu = (float*)malloc(size * sizeof(float));
  float *h_out_gpu = (float*)malloc(size * sizeof(float));
  
  // Initialize with simple pattern for easier debugging
  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      h_in[y * nx + x] = (float)((x + y) % 10) / 10.0f;
    }
  }
  
  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, size * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice));
  
  // Test configurations
  dim3 blockSizes[] = {
    dim3(16, 16),
    dim3(32, 8),
  };
  
  int timeStepsArray[] = {1, 2, 4};
  
  printf("=== VALIDATION ===\n\n");
  
  for (int b = 0; b < 2; b++) {
    printf("Block: %dx%d\n", blockSizes[b].x, blockSizes[b].y);
    
    for (int t = 0; t < 3; t++) {
      testTemporalBlocking(blockSizes[b], nx, ny, timeStepsArray[t],
                          d_in, d_out, h_in, h_out_cpu, h_out_gpu);
    }
    printf("\n");
  }
  
  // Cleanup
  free(h_in);
  free(h_out_cpu);
  free(h_out_gpu);
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  
  printf("Tests complete!\n");
  return 0;
}
