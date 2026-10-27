# AMD Profiling Quick Start Guide

This guide will help you quickly get started with AMD profiling tools.

## Prerequisites Check

```bash
# Check ROCm installation
rocminfo | grep "Name:"

# Check HIP version
hipconfig --version

# Check GPU info
rocm-smi

# Verify profiling tools are installed
which rocprof
rocprof --version
```

## Quick Build and Profile

### Step 1: Build the Examples

```bash
# Navigate to AMD examples directory
cd examples/amd

# Build with CMake
mkdir build && cd build
cmake ..
cmake --build . -j

# Or build with line info for detailed profiling
cmake .. -DENABLE_LINEINFO=ON
cmake --build . -j
```

### Step 2: Basic Profiling with rocprof

```bash
# Run application normally
./matrix_multiply_hip

# Profile with basic stats
rocprof --stats ./matrix_multiply_hip

# View results
cat results.stats.csv

# Profile with HSA trace (includes memory operations)
rocprof --hsa-trace ./matrix_multiply_hip

# Profile with system trace (CPU + GPU timeline)
rocprof --sys-trace ./matrix_multiply_hip
```

### Step 3: Detailed Metrics Collection

```bash
# Use the provided metrics configuration
rocprof -i ../metrics.txt --timestamp on ./matrix_multiply_hip

# View detailed metrics
cat results.csv

# Look for:
# - High MemUnitStalled -> memory bottleneck
# - Low VALUUtilization -> compute underutilization
# - High L2CacheMiss -> poor data locality
```

### Step 4: Profile PyTorch on ROCm

```bash
# First, install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# Run normally
cd examples/amd
python mnist_rocm.py

# Profile with rocprof
rocprof --stats --hip-trace python mnist_rocm.py

# View kernel statistics
cat results.stats.csv

# View timeline trace
# Open Chrome browser: chrome://tracing
# Load results.json file
```

## Understanding the Output

### rocprof Output Files

After running `rocprof`, you'll see:

1. **results.stats.csv** - Summary statistics
   ```
   Name,Calls,TotalDurationNs,AverageNs,Percentage
   matmul_tiled,10,45231234,4523123,100.00
   ```

2. **results.csv** - Detailed per-kernel metrics
   ```
   Kernel,VALUUtilization,MemUnitStalled,L2CacheHit,WriteSize
   matmul_tiled,85.2,12.3,78.5,16777216
   ```

3. **results.json** - JSON format for scripting

### Key Metrics to Monitor

| Metric | Good Range | What It Means |
|--------|-----------|---------------|
| VALUUtilization | >70% | Vector units are busy (high compute) |
| MemUnitStalled | <20% | Memory system not bottlenecked |
| L2CacheHit | >80% | Good data locality |
| Occupancy | >50% | Sufficient wavefronts to hide latency |
| WriteSize/FetchSize | - | Memory traffic (lower is better) |

## Common Profiling Workflows

### Workflow 1: Find Performance Bottleneck

```bash
# Step 1: Get overall statistics
rocprof --stats ./matrix_multiply_hip > profile.txt

# Step 2: Identify slow kernels
# Look at "TotalDurationNs" in results.stats.csv

# Step 3: Profile the slow kernel with metrics
rocprof -i metrics.txt ./matrix_multiply_hip

# Step 4: Analyze metrics
# High MemUnitStalled -> optimize memory access patterns
# Low VALUUtilization -> increase work per thread
# Low Occupancy -> reduce register/LDS usage
```

### Workflow 2: Compare Implementations

```bash
# Profile multiple versions
rocprof --stats -o naive.csv ./matrix_multiply_naive
rocprof --stats -o tiled.csv ./matrix_multiply_tiled  
rocprof --stats -o rocblas.csv ./matrix_multiply_rocblas

# Compare kernel times and metrics
# Look for improvements in:
# - Total execution time
# - Memory traffic (FetchSize + WriteSize)
# - Cache hit rates
```

### Workflow 3: Timeline Analysis with rocm-systems

```bash
# Generate system timeline
rocprof --sys-trace --hip-trace --hsa-trace ./matrix_multiply_hip

# View timeline in Chrome
# 1. Open Chrome: chrome://tracing
# 2. Load results.json

# Look for:
# - Kernel launch overhead
# - Memory transfer times
# - Idle GPU periods
# - CPU-GPU synchronization points
# - HIP API call durations
```

### Workflow 4: Python Application Timeline

```bash
# Trace Python application
rocprof --sys-trace --hip-trace python mnist_rocm.py

# View in Chrome tracing
# Load results.json in chrome://tracing

# Analyze:
# - Training iteration timeline
# - Forward/backward pass timing
# - Data loading bottlenecks
# - GPU utilization gaps
```

## Optimization Checklist

Based on profiling results, apply these optimizations:

### Memory-Bound Kernels (High MemUnitStalled)
- ✓ Use shared memory to cache frequently accessed data
- ✓ Coalesce global memory accesses
- ✓ Increase arithmetic intensity (compute per memory access)
- ✓ Use appropriate data types (FP16 where possible)

### Compute-Bound Kernels (High VALUUtilization)
- ✓ Already well-optimized for compute
- ✓ Consider algorithmic improvements
- ✓ Ensure you're using optimal library routines (rocBLAS, rocFFT, etc.)

### Low Occupancy
- ✓ Reduce VGPR usage (simplify register-heavy code)
- ✓ Reduce LDS usage (shared memory)
- ✓ Adjust workgroup size
- ✓ Use `__launch_bounds__` hint

### Poor Cache Utilization (Low L2CacheHit)
- ✓ Improve data access patterns (spatial/temporal locality)
- ✓ Use tiling/blocking strategies
- ✓ Reorder computations to reuse cached data

## Troubleshooting

### "rocprofiler: roctracer is not initialized"
```bash
export HSA_TOOLS_LIB=/opt/rocm/lib/libroctracer64.so
```

### "Permission denied" on performance counters
```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'
```

### Omniperf web interface won't start
```bash
# Install dependencies
pip install pandas plotly dash werkzeug

# Specify port explicitly
omniperf analyze -p workloads/my_workload/ --gui --port 8050
```

### Application runs slower when profiled
This is normal - profiling adds overhead. Key points:
- Use `--stats` for minimal overhead (10-20%)
- Full metrics collection adds more overhead (2-3x slower)
- Timeline tracing can significantly slow execution

## Next Steps

1. **Read the full README** - `README.md` has comprehensive tool documentation
2. **Master timeline analysis** - Use Chrome tracing to visualize performance
3. **Try different workloads** - Profile your own HIP/ROCm applications
4. **Compare with libraries** - rocBLAS, rocFFT, etc. are highly optimized
5. **Iterate** - Profile → Optimize → Profile again

## Quick Reference Card

```bash
# Basic profiling
rocprof --stats ./app

# With metrics
rocprof -i metrics.txt ./app

# System trace
rocprof --sys-trace ./app

# Omniperf full analysis
omniperf profile -n name -- ./app
omniperf analyze -p workloads/name/ --gui

# Python tracing
omnitrace-python -- python script.py

# Check GPU
rocm-smi

# List available metrics
rocprof --list-basic
rocprof --list-derived
```

## Getting Help

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [Omniperf Guide](https://amdresearch.github.io/omniperf/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- AMD Developer Forums: [https://community.amd.com/](https://community.amd.com/)
