# AMD Profiling Examples - File Index

This directory contains comprehensive examples and documentation for profiling AMD ROCm applications.

## Documentation Files

### 📚 Getting Started

- **[README.md](README.md)** - Complete guide to AMD profiling tools
  - Overview of rocprof and rocm-systems
  - Installation instructions
  - Detailed tool usage and examples
  - Timeline visualization guide
  - Troubleshooting guide
  - ~11 KB, comprehensive reference

- **[QUICKSTART.md](QUICKSTART.md)** - Fast-track guide for beginners
  - Prerequisites check
  - Quick build and profile steps
  - Understanding output
  - Common workflows
  - ~7 KB, perfect for getting started quickly

- **[NVIDIA_AMD_COMPARISON.md](NVIDIA_AMD_COMPARISON.md)** - Cross-platform reference
  - Tool equivalents (CUDA ↔ HIP)
  - Command comparisons
  - Metrics mapping
  - Migration tips
  - ~11 KB, essential for CUDA developers

### 📋 Configuration Files

- **[CMakeLists.txt](CMakeLists.txt)** - Build configuration
  - HIP compilation setup
  - Architecture targeting
  - rocBLAS integration
  - Line info options

- **[metrics.txt](metrics.txt)** - rocprof metrics configuration
  - Hardware counter definitions
  - Memory metrics
  - Compute utilization metrics
  - Ready to use with `rocprof -i metrics.txt`

- **[.gitignore](.gitignore)** - Version control exclusions
  - Build artifacts
  - Profiling outputs
  - Data files

## Example Programs

### 🔧 HIP/ROCm Examples

- **[matrix_multiply.hip](matrix_multiply.hip)** - Matrix multiplication kernels
  - Naive implementation
  - Shared memory (LDS) tiled version
  - Vectorized version with loop unrolling
  - Performance benchmarking
  - ~8 KB, well-commented

- **[matrix_multiply_rocblas.hip](matrix_multiply_rocblas.hip)** - Library comparison
  - Custom kernel vs rocBLAS SGEMM
  - Performance comparison
  - Result verification
  - ~7 KB

### 🐍 PyTorch/ML Examples

- **[mnist_rocm.py](mnist_rocm.py)** - MNIST training on ROCm
  - PyTorch with ROCm backend
  - Multi-layer neural network
  - Training and inference profiling
  - Device information reporting
  - ~7 KB, production-ready template

## Automation Scripts

- **[profile_all.sh](profile_all.sh)** - Automated profiling suite
  - Multi-mode rocprof profiling (stats, metrics, timeline)
  - Automated report generation
  - Color-coded output
  - Error handling
  - ~7 KB, executable bash script

## Quick Reference

### For First-Time Users
1. Start with **[QUICKSTART.md](QUICKSTART.md)**
2. Build examples using **[CMakeLists.txt](CMakeLists.txt)**
3. Run **[profile_all.sh](profile_all.sh)** for automated profiling

### For CUDA Developers
1. Read **[NVIDIA_AMD_COMPARISON.md](NVIDIA_AMD_COMPARISON.md)**
2. Compare implementations in **[matrix_multiply.hip](matrix_multiply.hip)**
3. Reference **[README.md](README.md)** for tool details

### For Deep Learning Engineers
1. Check **[mnist_rocm.py](mnist_rocm.py)** for PyTorch example
2. Follow profiling instructions in **[README.md](README.md)**
3. Use **[profile_all.sh](profile_all.sh)** with Python scripts

### For Performance Engineers
1. Study **[README.md](README.md)** for all profiling tools
2. Customize **[metrics.txt](metrics.txt)** for specific metrics
3. Use **[matrix_multiply_rocblas.hip](matrix_multiply_rocblas.hip)** for library comparisons

## Building the Examples

```bash
mkdir build && cd build
cmake ..
cmake --build . -j

# Or with line info for profiling
cmake .. -DENABLE_LINEINFO=ON
cmake --build . -j

# For specific GPU architecture
cmake .. -DGPU_ARCH=gfx90a
cmake --build . -j
```

## Running Examples

### HIP Examples
```bash
# Run directly
./build/matrix_multiply_hip

# Profile with rocprof
rocprof --stats ./build/matrix_multiply_hip

# Detailed metrics
rocprof -i metrics.txt ./build/matrix_multiply_hip

# Compare with rocBLAS
./build/matrix_multiply_rocblas
```

### Python Examples
```bash
# Install PyTorch for ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

# Run training
python mnist_rocm.py

# Profile with Omnitrace
omnitrace-python -- python mnist_rocm.py
```

### Automated Profiling
```bash
# Profile all built examples
./profile_all.sh

# Results saved in profiling_results/
```

## File Statistics

| File | Type | Size | Purpose |
|------|------|------|---------|
| README.md | Documentation | 11 KB | Main guide |
| QUICKSTART.md | Documentation | 7 KB | Quick start |
| NVIDIA_AMD_COMPARISON.md | Documentation | 11 KB | Platform comparison |
| matrix_multiply.hip | Source code | 8 KB | HIP kernels |
| matrix_multiply_rocblas.hip | Source code | 7 KB | Library comparison |
| mnist_rocm.py | Python | 7 KB | PyTorch example |
| profile_all.sh | Bash script | 9 KB | Automation |
| CMakeLists.txt | Build config | 2 KB | CMake setup |
| metrics.txt | Config | 1 KB | Profiling metrics |
| .gitignore | Config | <1 KB | Version control |

**Total documentation**: ~29 KB  
**Total code**: ~22 KB  
**Total**: ~51 KB of comprehensive examples and documentation

## Tools Covered

### Profilers
- ✅ rocprof - Command-line profiler with metrics and timeline tracing
- ✅ rocm-systems - System-wide profiler (rocprof backend)
- ✅ rocTracer - Low-level API tracing library
- ✅ Chrome Tracing - Timeline visualization (chrome://tracing)

### Libraries
- ✅ rocBLAS - Linear algebra
- ✅ PyTorch - Deep learning framework

### Languages
- ✅ HIP/C++ - GPU kernels
- ✅ Python - Machine learning

## Support Matrix

| GPU Architecture | Status | Notes |
|------------------|--------|-------|
| CDNA (MI100, MI200) | ✅ Fully supported | Server/HPC GPUs |
| CDNA2 (MI210, MI250) | ✅ Fully supported | Latest datacenter GPUs |
| CDNA3 (MI300) | ✅ Fully supported | Next-gen datacenter |
| RDNA2 (RX 6000) | ⚠️ Partial support | Gaming GPUs, limited profiling |
| RDNA3 (RX 7000) | ⚠️ Partial support | Wavefront size = 32 |

## Getting Help

1. **Installation issues**: See [README.md](README.md) troubleshooting section
2. **Build errors**: Check [CMakeLists.txt](CMakeLists.txt) requirements
3. **Profiling problems**: Refer to [QUICKSTART.md](QUICKSTART.md)
4. **CUDA migration**: See [NVIDIA_AMD_COMPARISON.md](NVIDIA_AMD_COMPARISON.md)

## Contributing

To add new examples:
1. Follow the existing code structure
2. Include detailed comments
3. Add profiling instructions in README.md
4. Test with `profile_all.sh`

## License

These examples are part of the profiler training materials and follow the same license as the parent repository.

---

**Last Updated**: 2026-09-30  
**ROCm Version Tested**: 5.7+  
**PyTorch Version Tested**: 2.0+
