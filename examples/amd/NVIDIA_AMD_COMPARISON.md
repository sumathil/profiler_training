# NVIDIA CUDA vs AMD ROCm Profiling - Quick Reference

This document provides a side-by-side comparison of profiling tools and workflows between NVIDIA CUDA and AMD ROCm ecosystems.

## Tool Equivalents

| NVIDIA CUDA | AMD ROCm | Purpose |
|-------------|----------|---------|
| Nsight Systems | rocprof --sys-trace | System-wide timeline profiling |
| Nsight Compute | rocprof -i metrics.txt | Detailed kernel metrics and analysis |
| nvprof (deprecated) | rocprof | Command-line profiler |
| NVTX | roctx | Code annotation API |
| cuBLAS | rocBLAS | Optimized BLAS library |
| cuDNN | MIOpen | Deep learning primitives |
| CUDA | HIP | GPU programming API |
| nvcc | hipcc | Compiler |

## Command Comparison

### Basic Profiling

| Task | NVIDIA | AMD |
|------|--------|-----|
| Profile with basic stats | `nsys profile -o output ./app` | `rocprof --stats ./app` |
| Profile with metrics | `ncu --set full -o output ./app` | `rocprof -i metrics.txt ./app` |
| Profile kernel execution | `nsys profile -t cuda ./app` | `rocprof --hip-trace ./app` |
| System trace | `nsys profile -t cuda,nvtx,osrt ./app` | `rocprof --sys-trace --hip-trace ./app` |
| Profile Python | `nsys profile python script.py` | `rocprof --stats --hip-trace python script.py` |

### Advanced Profiling

| Task | NVIDIA | AMD |
|------|--------|-----|
| Kernel filtering | `ncu -k kernel_name ./app` | `rocprof --kernel-name kernel_name ./app` |
| Source correlation | `ncu --import-source on ./app` | Build with `-ggdb` flag |
| Memory profiling | `ncu --section MemoryWorkloadAnalysis` | `rocprof -i mem_metrics.txt ./app` |
| Timeline visualization | Open .nsys-rep in Nsight Systems GUI | Open results.json in chrome://tracing |
| API trace | `nsys profile -t cuda ./app` | `rocprof --hip-trace --hsa-trace ./app` |

## Programming Model Comparison

### Code Translation

CUDA code can often be ported to HIP with minimal changes:

```cpp
// CUDA
#include <cuda_runtime.h>
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
kernel<<<grid, block>>>(args);
cudaDeviceSynchronize();

// HIP (AMD)
#include <hip/hip_runtime.h>
hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
hipLaunchKernelGGL(kernel, grid, block, 0, 0, args);
hipDeviceSynchronize();
```

### Annotation API

```cpp
// CUDA with NVTX
#include <nvtx3/nvToolsExt.h>
nvtxRangePush("Forward Pass");
// ... code ...
nvtxRangePop();

// HIP with roctx
#include <roctracer/roctx.h>
roctxRangePush("Forward Pass");
// ... code ...
roctxRangePop();
```

## Performance Metrics Comparison

### Hardware Metrics

| Metric | NVIDIA Term | AMD Term | Description |
|--------|-------------|----------|-------------|
| Execution units | CUDA Cores | Stream Processors | Scalar processing units |
| Vector unit | - | VALU | Vector ALU operations |
| Compute group | Warp (32 threads) | Wavefront (64 threads) | SIMT execution group |
| Shared memory | Shared Memory | LDS (Local Data Share) | On-chip scratchpad |
| Register file | Register File | VGPR/SGPR | Vector/Scalar registers |
| L2 Cache | L2 Cache | L2 Cache | Last-level cache |
| Occupancy | Occupancy | Wave Occupancy | Active warps/wavefronts |

### Common Metrics

| Metric Type | NVIDIA (ncu) | AMD (rocprof/omniperf) |
|-------------|--------------|------------------------|
| Compute utilization | `sm__throughput.avg.pct_of_peak_sustained_active` | `VALUUtilization` |
| Memory bandwidth | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | `FetchSize + WriteSize` |
| L2 cache hit rate | `lts__t_sector_hit_rate.pct` | `L2CacheHit` |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | `Wavefront Occupancy` |
| Memory stalls | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | `MemUnitStalled` |

## Profiling Workflow Comparison

### Workflow 1: Find Bottleneck Kernel

**NVIDIA:**
```bash
# Step 1: System profile
nsys profile -o output ./app
# Open output.nsys-rep in Nsight Systems GUI

# Step 2: Identify slow kernel
# Look at CUDA timeline

# Step 3: Detailed kernel analysis
ncu --set full -o kernel_analysis ./app
# Open kernel_analysis.ncu-rep in Nsight Compute
```

**AMD:**
```bash
# Step 1: System profile
rocprof --stats --sys-trace --hip-trace ./app
# View results.stats.csv and results.json

# Step 2: Identify slow kernel
cat results.stats.csv | sort -t',' -k3 -rn

# Step 3: Detailed kernel analysis
rocprof -i metrics.txt ./app
# View results.csv for detailed hardware metrics

# Step 4: Timeline analysis
# Open results.json in chrome://tracing
```

### Workflow 2: Memory Access Analysis

**NVIDIA:**
```bash
ncu --set full \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    -o mem_analysis ./app
# Check "Memory Workload Analysis" in GUI
```

**AMD:**
```bash
# Create memory-focused metrics file
cat > mem_metrics.txt << EOF
pmc : FetchSize WriteSize L2CacheHit MemUnitStalled
pmc : L2CacheMiss MemUnitBusy
EOF

rocprof -i mem_metrics.txt ./app
# Analyze results.csv for memory patterns
```

### Workflow 3: Optimization Iteration

**NVIDIA:**
```bash
# Baseline
ncu --set full -o v1 ./app_v1
# Optimize and compare
ncu --set full -o v2 ./app_v2
# Use GUI to compare v1 vs v2
```

**AMD:**
```bash
# Baseline
rocprof -i metrics.txt -o v1.csv ./app_v1
# Optimize and compare
rocprof -i metrics.txt -o v2.csv ./app_v2
# Manual comparison or use custom script
python compare_metrics.py v1.csv v2.csv
```

## PyTorch Profiling Comparison

### NVIDIA with NVTX

```python
import torch
import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx

# Training loop
for epoch in range(epochs):
    nvtx.range_push(f"Epoch {epoch}")
    
    nvtx.range_push("Forward")
    output = model(input)
    nvtx.range_pop()
    
    nvtx.range_push("Backward")
    loss.backward()
    nvtx.range_pop()
    
    nvtx.range_pop()  # Epoch
```

Profile with:
```bash
nsys profile -t cuda,nvtx,cudnn -o pytorch_profile python train.py
```

### AMD with ROCm

```python
import torch

# PyTorch on ROCm automatically uses HIP backend
device = torch.device('cuda')  # Still uses 'cuda' name but runs on ROCm

# Training loop (same code)
for epoch in range(epochs):
    output = model(input)
    loss.backward()
    optimizer.step()
```

Profile with:
```bash
# rocprof with timeline tracing
rocprof --stats --hip-trace --sys-trace python train.py

# View kernel stats
cat results.stats.csv

# View timeline
# Load results.json in chrome://tracing
```

## Library Performance Comparison

### Matrix Multiplication

**NVIDIA cuBLAS:**
```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);

cublasDestroy(handle);
```

**AMD rocBLAS:**
```cpp
#include <rocblas/rocblas.h>

rocblas_handle handle;
rocblas_create_handle(&handle);

rocblas_sgemm(handle,
              rocblas_operation_none, rocblas_operation_none,
              N, M, K,
              &alpha,
              d_B, N,
              d_A, K,
              &beta,
              d_C, N);

rocblas_destroy_handle(handle);
```

## Architecture Differences

### Thread Hierarchy

**NVIDIA:**
- Thread → Warp (32 threads) → Block → Grid
- Warp size: 32 threads (fixed)
- Max threads per block: 1024
- Shared memory per block: 48-164 KB (architecture dependent)

**AMD:**
- Thread → Wavefront (64 threads) → Workgroup → Grid
- Wavefront size: 64 threads (32 on RDNA3+)
- Max threads per workgroup: 1024
- LDS per workgroup: 64 KB (typical)

### Memory Hierarchy

| Level | NVIDIA | AMD |
|-------|--------|-----|
| L1 Cache/Shared | Combined 128 KB | Separate: 16 KB L1 + 64 KB LDS |
| L2 Cache | 6-50 MB (architecture dependent) | 4-8 MB (architecture dependent) |
| Register File | 256 KB per SM | 256 KB per CU (VGPR + SGPR) |

## Best Practices

### NVIDIA CUDA

1. **Occupancy**: Target 50-100% occupancy
2. **Warp divergence**: Minimize branching within warps
3. **Shared memory**: Use to reduce global memory traffic
4. **Coalescing**: Align memory access to 128-byte boundaries
5. **PTX inspection**: Use `cuobjdump -ptx` for low-level analysis

### AMD ROCm

1. **Wavefront occupancy**: Target 50-100% occupancy
2. **Vector ALU utilization**: Maximize VALU usage
3. **LDS usage**: Balance LDS vs occupancy
4. **Memory coalescing**: Align to 256-byte cache lines
5. **ISA inspection**: Use `llvm-objdump -d` for low-level analysis

## Common Optimization Patterns

### Shared Memory/LDS Tiling

Both platforms benefit from similar tiling strategies:

```cpp
// Tile size should be multiple of warp/wavefront
#define TILE_SIZE 16  // Both NVIDIA and AMD

__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // NVIDIA
    // or
    __shared__ float As[TILE_SIZE][TILE_SIZE];  // AMD (LDS)
    
    // Same tiling logic works on both platforms
    // ...
}
```

### Register Blocking

```cpp
// NVIDIA: Optimize for warp size 32
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// AMD: Optimize for wavefront size 64
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16  // Total: 256 threads = 4 wavefronts
```

## Performance Debugging Checklist

| Issue | NVIDIA Check | AMD Check |
|-------|--------------|-----------|
| Low performance | SM Utilization in ncu | VALU Utilization in Omniperf |
| Memory bottleneck | Memory Workload Analysis | FetchSize/WriteSize metrics |
| Low occupancy | Occupancy limiter in ncu | Wavefront occupancy in Omniperf |
| Cache misses | L2 hit rate | L2CacheHit percentage |
| Sync overhead | CUDA API trace in nsys | HIP API trace in rocprof |

## Migration Tips

### CUDA to HIP

1. **Use hipify**: AMD provides automated conversion tool
   ```bash
   hipify-perl cuda_code.cu > hip_code.hip
   ```

2. **Common replacements**:
   - `cuda` → `hip` in function names
   - `<<<grid, block>>>` → `hipLaunchKernelGGL(kernel, grid, block, 0, 0, args)`
   - `__syncthreads()` → `__syncthreads()` (same)

3. **Verify with profiling**: Compare performance metrics between CUDA and HIP versions

### Profiling Workflow Migration

1. **Timeline profiling**: nsys → omnitrace
2. **Kernel analysis**: ncu → omniperf
3. **Metrics**: Map NVIDIA metrics to AMD equivalents (see table above)
4. **Optimization targets**: Same principles apply (occupancy, memory coalescing, etc.)

## Additional Resources

### NVIDIA
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### AMD
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Omniperf User Guide](https://amdresearch.github.io/omniperf/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
