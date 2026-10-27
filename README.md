# Profiler Training Materials

Training materials to demonstrate the use of GPU profilers using CUDA and AMD ROCm examples.

## Overview

This repository contains comprehensive examples and tutorials for:
- **NVIDIA CUDA**: Nsight Systems, Nsight Compute profiling
- **AMD ROCm**: rocprof, Omniperf, Omnitrace profiling
- **AI/ML Examples**: PyTorch profiling on both NVIDIA and AMD GPUs

The CUDA examples include matrix multiplication (naive, shared memory, cuBLAS), Kokkos matrix multiplication, and 2D stencil operations. The AI examples demonstrate PyTorch profiling with MNIST training and transformer language models.

## Requirements

### For NVIDIA CUDA:
- NVIDIA GPU with CUDA support
- CUDA Toolkit (for `nvcc`)
- CMake 3.18+
- Python 3.10+ (for PyTorch examples)

### For AMD ROCm:
- AMD GPU with ROCm support
- ROCm 5.0+ with HIP compiler
- CMake 3.18+
- Python 3.10+ (for PyTorch examples)

Create a virtual environment and install Python packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch/TorchVision wheels (NVIDIA GPU setup):

```bash
python3.11 -m pip install -r requirements.txt
```

## Build

### NVIDIA CUDA Build

Basic build (default: V100/sm_70):

```bash
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build -j
```

Build with architecture options:

For A100 (sm_80):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCUDA_ARCH_SM70=OFF -DCUDA_ARCH_SM80=ON
cmake --build build -j
```

For H100 (sm_90):
```bash
cmake -S . -B build -DUSE_CUDA=ON -DCUDA_ARCH_SM70=OFF -DCUDA_ARCH_SM90=ON
cmake --build build -j
```

Enable lineinfo for profiling:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DENABLE_LINEINFO=ON
cmake --build build -j
```

Enable Kokkos examples:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DENABLE_KOKKOS=ON -DKOKKOS_INSTALL_DIR=/path/to/kokkos
cmake --build build -j
```

### AMD ROCm Build

Basic build (default: MI210/MI250 - gfx90a):

```bash
cmake -S . -B build_amd -DUSE_AMD=ON -DUSE_CUDA=OFF
cmake --build build_amd -j
```

Or use the convenience script:

```bash
./build_amd.sh
```

Build with architecture options:

For MI100 (gfx908):
```bash
cmake -S . -B build_amd -DUSE_AMD=ON -DAMD_ARCH_GFX908=ON -DAMD_ARCH_GFX90A=OFF
cmake --build build_amd -j
```

For MI300 (gfx942):
```bash
cmake -S . -B build_amd -DUSE_AMD=ON -DAMD_ARCH_GFX90A=OFF -DAMD_ARCH_GFX942=ON
cmake --build build_amd -j
```

Enable lineinfo for profiling:

```bash
cmake -S . -B build_amd -DUSE_AMD=ON -DENABLE_LINEINFO=ON
cmake --build build_amd -j
```

### Platform Detection

The CMake configuration automatically detects which platform to build for:
- Use `-DUSE_CUDA=ON` for NVIDIA platforms (default)
- Use `-DUSE_AMD=ON -DUSE_CUDA=OFF` for AMD platforms

## Run CUDA Examples

### Matrix multiply (CPU naive)

```bash
./build/matmul_cpu_naive
```

### Matrix multiply (shared memory)

```bash
./build/matmul_cpu_ns
```

### Matrix multiply (shared memory + cuBLAS comparison)

```bash
./build/matmul_cpu_nsc
```

### Matrix multiply (Kokkos)

Requires `-DENABLE_KOKKOS=ON` during build:

```bash
./build/matmul_kokkos
```

### 2D 5-point stencil

Default run:

```bash
./build/stencil_2d_benchmark
```

Custom width, height, and iterations:

```bash
./build/stencil_2d_benchmark 4096 4096 200
```

Arguments:
- `width` (default: `4096`)
- `height` (default: `4096`)
- `iterations` (default: `200`)

### 2D 5-point stencil (tiled/shared memory)

Default run:

```bash
./build/stencil_2d_tiled_benchmark
```

Custom width, height, and iterations:

```bash
./build/stencil_2d_tiled_benchmark 4096 4096 200
```

Arguments:
- `width` (default: `4096`)
- `height` (default: `4096`)
- `iterations` (default: `200`)

### 2D stencil comparison (optimized variants)

```bash
./build/stencil_2d_compare
```

## Run PyTorch Examples

### MNIST training with NVTX annotations

```bash
cd examples/pytorch
python pytorch_profiler_cudart.py
```

Profile with Nsight Systems:
```bash
nsys profile -t cuda,nvtx,cublas,osrt -o mnist_profile python pytorch_profiler_cudart.py
```

### Python Matrix multiplication profiling

```bash
cd examples/pytorch
python cuda_profile_matmul.py --m 4096 --n 4096 --k 4096 --iters 200
```

Profile with Nsight Systems:
```bash
nsys profile -o matmul_profile python cuda_profile_matmul.py --m 4096 --n 4096 --k 4096 --iters 200
```

Options:
- `--m`: Rows of matrix A and C (default: 4096)
- `--n`: Columns of matrix B and C (default: 4096)
- `--k`: Columns of A / rows of B (default: 4096)
- `--batch`: Batch size for batched matmul (default: 1)
- `--iters`: Total iterations (default: 200)
- `--warmup`: Warmup iterations (default: 20)
- `--dtype`: Data type - fp16, bf16, or fp32 (default: fp16)
- `--use-tf32`: Enable TF32 for fp32 matmul (Ampere+ GPUs)

### Transformer language model with AMP (Automatic Mixed)

```bash
cd examples/pytorch
python transformer_lm_amp.py --steps 300 --batch-size 64
```

Profile with Nsight Systems:
```bash
nsys profile -o transformer_profile python transformer_lm_amp.py --steps 300
```

## Profiling

### NVIDIA Profiling (Nsight Systems & Nsight Compute)

**Nsight Systems - Timeline profiling:**
```bash
# Basic profiling
nsys profile -o output ./build/matmul_cpu_naive

# Profile with CUDA API trace
nsys profile -t cuda,nvtx,cublas,osrt -o output ./build/matmul_cpu_nsc

# Profile PyTorch code
nsys profile -t cuda,nvtx,cublas,cudnn --capture-range=cudaProfilerApi -o output python app.py
```

**Nsight Compute - Kernel analysis:**
```bash
# Profile specific kernels with source code import
ncu --set full --import-source on -o output ./build/matmul_cpu_naive

# Profile Kokkos with kernel filter
ncu --set full --import-source on -k "regex:MatMulNaive" --kernel-name-base demangled -o kokkos_ncu ./matmul_kokkos
```

### AMD Profiling

See **[AMD_PROFILER_GUIDE.md](AMD_PROFILER_GUIDE.md)** for complete AMD profiling guide.

Quick commands:

**rocprof (always available):**
```bash
# Quick stats
rocprof --stats ./build_amd/matmul_hip

# Timeline trace
rocprof --sys-trace --hip-trace ./build_amd/matmul_hip

# Detailed metrics
rocprof -i metrics.txt ./build_amd/matmul_hip
```

**rocprofiler-systems (advanced, requires install):**
```bash
# Call-stack sampling
rocprof-sys-sample -f 1000 -- ./build_amd/matmul_hip

# Python profiling
rocprof-sys-python -- train.py
```

**Convenience script:**
```bash
./profile_amd.sh ./build_amd/matmul_hip
```

**View results:**
- Perfetto: https://ui.perfetto.dev → Upload `.proto` or `.json` file
- Chrome Tracing: `chrome://tracing` → Load `.json` file

## Run AMD Examples

After building with `-DUSE_AMD=ON`, run the AMD executables:

### Matrix multiply (HIP)

```bash
./build_amd/matmul_hip
```

### Matrix multiply (rocBLAS)

```bash
./build_amd/matmul_rocblas
```







