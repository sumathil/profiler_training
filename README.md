# Profiler Training Materials

Training materials to demonstrate the use of Nsight profilers using CUDA and AI examples.
The CUDA examples include matrix multiplication (naive, shared memory, cuBLAS), Kokkos matrix multiplication, and 2D stencil operations. The AI examples demonstrate PyTorch profiling with MNIST training and transformer language models.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (for `nvcc`)
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

### Basic build (default: V100/sm_70)

```bash
cmake -S . -B build
cmake --build build -j
```

### Build with architecture options

For A100 (sm_80):
```bash
cmake -S . -B build -DCUDA_ARCH_SM70=OFF -DCUDA_ARCH_SM80=ON
cmake --build build -j
```

For H100 (sm_90):
```bash
cmake -S . -B build -DCUDA_ARCH_SM70=OFF -DCUDA_ARCH_SM90=ON
cmake --build build -j
```

### Enable lineinfo for profiling

```bash
cmake -S . -B build -DENABLE_LINEINFO=ON
cmake --build build -j
```

### Enable Kokkos examples

```bash
cmake -S . -B build -DENABLE_KOKKOS=ON -DKOKKOS_INSTALL_DIR=/path/to/kokkos
cmake --build build -j
```

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

## Profiling with Nsight Systems

Basic profiling:
```bash
nsys profile -o output ./build/matmul_cpu_naive
```

Profile with CUDA API trace:
```bash
nsys profile -t cuda,nvtx,cublas,osrt -o output ./build/matmul_cpu_nsc
```


Profile PyTorch code:
```bash
nsys profile –t cuda,nvtx,cublas, cudnn --capture-range=cudaProfilerApi -o output_file python app.py
```

## Profiling with Nsight Compute

Profile specific kernels with source code import:
```bash
ncu --set full --import-source on -o output ./build/matmul_cpu_naive
```

Profile Kokkos with kernel filter:
```bash
ncu --set full --import-source on -k "regex:MatMulNaive" --kernel-name-base demangled -o kokkos_ncu ./matmul_kokkos
```

### Profiling using NSIGHT tools

#### Nsight systems

Creates nsys reports that gives system level analysis report

```
nsys profile -o naive ./build/matmul_cpu_naive 1024 10
```

To trace cublas library, pass `-t cublas` to the `nsys` CLI
 
```
nsys profile -t cuda,nvtx,osrt,cublas -o naive_shared_cublas ./build/matmul_cpu_nsc 1024 10
```

#### Nsight compute

Provide kernel level details to analyze memory throughput, warp occupancy, roofline 

```
ncu --import-source on --set full --call-stack --nvtx -o ncu_naive  ./build/matmul_cpu_naive 1024 5
```

### Profiling Python applications

```
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o mnist python3.11 examples/pytorch/mnist.py
```





