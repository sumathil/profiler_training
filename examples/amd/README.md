# AMD Profiling Tools Training

This directory contains examples and instructions for profiling HIP and ROCm applications using AMD's profiling tools.

## AMD Profiling Tools Overview

AMD provides several profiling tools for HIP/ROCm applications:

1. **rocprof** - Command-line profiler for collecting kernel metrics and performance counters
2. **rocm-systems** - System-wide profiler (similar to NVIDIA Nsight Systems) for timeline tracing
3. **rocTracer** - Low-level API tracing library (used internally by rocprof)

## Requirements

- AMD GPU (RDNA, CDNA architectures)
- ROCm 5.0+ installed
- HIP runtime and compiler
- Python 3.8+ (for PyTorch examples)

### Install ROCm

Follow the official ROCm installation guide:
```bash
# Ubuntu/Debian
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/debian jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-sdk rocprofiler-dev rocm-smi-lib
```

Install profiling tools:
```bash
# Install rocprof and rocm-systems
sudo apt install rocprofiler-dev roctracer-dev
```

Add ROCm to your PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

### Verify Installation

```bash
rocm-smi  # Check GPU info
hipconfig --version  # Check HIP version
rocprof --version  # Check rocprof version
```

## AMD Profiling Tools

### 1. rocprof - Performance Metrics Collection

`rocprof` is the primary command-line profiler for ROCm applications. It collects kernel execution times, memory operations, and hardware performance counters.

#### Basic Usage

Profile kernel execution:
```bash
rocprof --stats ./hip_application
```

Collect specific metrics:
```bash
rocprof --stats --hsa-trace ./hip_application
```

Generate detailed trace with timestamps:
```bash
rocprof --sys-trace ./hip_application
```

#### Advanced Features

**Collect Hardware Metrics:**
```bash
rocprof -i metrics.txt ./hip_application
```

Create a `metrics.txt` file with desired counters:
```
pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts GDSInsts VALUUtilization FetchSize WriteSize
```

**Common Performance Metrics:**
- `Wavefronts` - Total wavefronts dispatched
- `VALUInsts` - Vector ALU instructions
- `VALUUtilization` - Vector ALU utilization percentage
- `MemUnitStalled` - Memory unit stall cycles
- `WriteSize` - Total bytes written to memory
- `FetchSize` - Total bytes read from memory

**Profile Specific Kernels:**
```bash
rocprof --kernel-trace ./hip_application
```

#### Output Files

rocprof generates several output files:
- `results.csv` - Kernel execution statistics
- `results.stats.csv` - Summary statistics
- `results.json` - JSON format results for programmatic analysis

### 2. rocm-systems - System Timeline Profiling

rocm-systems (AMD's equivalent to NVIDIA Nsight Systems) provides system-wide timeline profiling with CPU and GPU activity visualization.

#### Installation

rocm-systems is typically included with ROCm SDK. Verify installation:
```bash
which rocprof  # rocm-systems uses rocprof backend
```

#### Basic Usage - Timeline Tracing

**Generate timeline trace:**
```bash
rocprof --sys-trace ./hip_application
```

This generates JSON trace files that can be visualized.

**Trace with HIP API calls:**
```bash
rocprof --hip-trace ./hip_application
```

**Trace with HSA operations:**
```bash
rocprof --hsa-trace ./hip_application
```

**Combined tracing:**
```bash
rocprof --sys-trace --hip-trace --hsa-trace ./hip_application
```

#### Visualization

Timeline traces can be viewed using:

1. **Chrome Tracing** (`chrome://tracing`):
   ```bash
   # rocprof generates results.json
   # Open Chrome browser
   # Navigate to chrome://tracing
   # Load results.json
   ```

2. **Perfetto** (https://ui.perfetto.dev/):
   ```bash
   # Convert to perfetto format if needed
   # Upload to https://ui.perfetto.dev/
   ```

#### What Timeline Traces Show

- Kernel execution timeline
- HIP API calls (hipMemcpy, hipLaunchKernel, etc.)
- HSA operations
- CPU-GPU synchronization points
- Memory transfers
- Idle periods and bottlenecks

### 3. rocTracer - API Call Tracing

rocTracer provides low-level API call tracing for HIP and HSA runtimes.

#### Basic Usage

```bash
roctracer ./hip_application
```

Enable specific API domains:
```bash
roctracer -d hip,hsa ./hip_application
```

#### Output

Generates timeline traces showing:
- HIP API calls (hipMemcpy, hipLaunchKernel, etc.)
- HSA operations
- Timestamps and durations
- Correlation between CPU and GPU activities

## Example Programs

### 1. HIP Matrix Multiplication

A basic matrix multiplication example using HIP:
- `matrix_multiply.hip` - HIP implementation with various optimizations
- Demonstrates global memory, shared memory, and register tiling

### 2. HIP GEMM Benchmark

Comprehensive GEMM benchmark comparing:
- Naive implementation
- Tiled/shared memory version
- rocBLAS library call

### 3. PyTorch on ROCm

Training examples with profiling annotations:
- MNIST classifier with roctracer integration
- Transformer model training
- Demonstrates PyTorch profiler on ROCm backend

## Profiling Workflow Examples

### Example 1: Profile and Optimize Matrix Multiply

```bash
# Build the example
hipcc -O3 matrix_multiply.hip -o matrix_multiply

# Profile with rocprof
rocprof --stats --hsa-trace ./matrix_multiply

# View results
cat results.stats.csv

# Collect detailed metrics
rocprof -i metrics.txt ./matrix_multiply
cat results.csv
```

### Example 2: Timeline Analysis with rocm-systems

```bash
# Generate system timeline
rocprof --sys-trace --hip-trace ./matrix_multiply

# View results.json in Chrome
# 1. Open Chrome browser
# 2. Navigate to chrome://tracing
# 3. Load results.json
```

Look for:
- Kernel execution gaps (idle GPU)
- Memory transfer overhead
- CPU-GPU synchronization bottlenecks
- Kernel launch overhead

### Example 3: Python Application Profiling

```bash
cd ../pytorch

# Profile PyTorch training with rocprof
rocprof --stats --hip-trace python mnist_rocm.py

# View kernel statistics
cat results.stats.csv

# View timeline in Chrome
# Load results.json in chrome://tracing
```

### Example 4: Compare Multiple Implementations

```bash
# Profile naive version
rocprof --stats -o naive ./matmul_naive

# Profile optimized version
rocprof --stats -o optimized ./matmul_optimized

# Profile library version
rocprof --stats -o rocblas ./matmul_rocblas

# Compare results from .stats.csv files
diff naive.stats.csv optimized.stats.csv
```

## Common Profiling Metrics Explained

### Kernel Execution Metrics
- **Duration** - Total kernel execution time
- **Grid Size** - Number of workgroups launched
- **Workgroup Size** - Threads per workgroup
- **VGPR/SGPR Usage** - Register usage per wavefront

### Memory Metrics
- **FetchSize** - Bytes read from global memory
- **WriteSize** - Bytes written to global memory
- **L2 Cache Hit Rate** - Percentage of L2 cache hits
- **Memory Bandwidth** - Achieved memory bandwidth (GB/s)

### Compute Metrics
- **VALUUtilization** - Vector ALU utilization (%)
- **SALUUtilization** - Scalar ALU utilization (%)
- **Occupancy** - Wavefront occupancy (active wavefronts / max wavefronts)
- **IPC** - Instructions per cycle

## Optimization Tips

### Memory Optimization
1. **Coalesce memory accesses** - Align consecutive threads to consecutive memory
2. **Use shared memory** - Reduce global memory traffic
3. **Minimize bank conflicts** - Distribute LDS accesses across banks
4. **Increase cache hit rate** - Improve temporal/spatial locality

### Compute Optimization
1. **Maximize occupancy** - Balance register usage and shared memory
2. **Reduce divergence** - Minimize branch divergence within wavefronts
3. **Increase instruction-level parallelism** - Unroll loops, interleave operations
4. **Use appropriate data types** - FP16/BF16 for ML workloads when possible

### Launch Configuration
1. **Workgroup size** - Typically multiples of 64 (wavefront size)
2. **Grid size** - Ensure enough workgroups to saturate GPU
3. **Resource usage** - Monitor VGPR/SGPR/LDS usage per wavefront

## Troubleshooting

### rocprof Issues

**Error: "rocprofiler: roctracer is not initialized"**
```bash
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
```

**Permission denied accessing performance counters:**
```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

### Timeline Visualization Issues

**Chrome tracing not loading:**
- Make sure you're using results.json (not results.csv)
- Try opening in Chromium or Edge browser
- Convert to Perfetto format if needed

**Missing GPU activity in timeline:**
Ensure HSA tracing is enabled:
```bash
rocprof --sys-trace --hsa-trace --hip-trace ./app
```

## Additional Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [rocprof User Guide](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [roctracer Documentation](https://rocm.docs.amd.com/projects/roctracer/en/latest/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD GPU Hardware Documentation](https://www.amd.com/en/technologies/cdna)
- [Chrome Tracing Viewer](chrome://tracing)

## Quick Reference

### Most Common Commands

```bash
# Quick kernel profiling
rocprof --stats ./app

# Detailed system trace with timeline
rocprof --sys-trace --hip-trace --hsa-trace ./app

# Collect hardware metrics
rocprof -i metrics.txt ./app

# Profile Python applications
rocprof --stats --hip-trace python script.py

# View GPU info
rocm-smi
rocminfo

# View timeline (load results.json)
# Open chrome://tracing in Chrome browser
```

### Useful Environment Variables

```bash
# rocprof output directory
export ROCPROF_OUTPUT_DIR=./profiling_results

# AMD GPU device selection
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Enable verbose output
export ROCPROFILER_LOG=1

# Set rocprof metrics file
export ROCPROF_METRICS_FILE=./metrics.txt
```
