# Profiler Training Materials

Training materials to demonstrate the use of nsight profilers using CUDA and AI examples.
The CUDA examples are matrix multiplication and 2D stencil operation. The AI examples contains CNN image classification for mnist dataset.

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
python3.11 -m pip install -r requirements-gpu.txt
```

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

### Matrix multiply (naive)

Default run:

```bash
./build/matmul_cpu_naive
```

Custom matrix size and iterations:

```bash
./build/matmul_benchmark 1024 50
```

Arguments:
- `matrix_size` (default: `1024`)
- `iterations` (default: `50`)


### Matrix multiply (naive vs shared memory)

Default run:

```bash
./build/matmul_cpu_ns
```

Custom matrix size and iterations:

```bash
./build/matmul_cpu_ns 1024 50
```

Output format:

```text
Tilesize Execution Time Speedup
```

### Matrix multiply (cublas vs shared and naive)

Default run:

```bash
./build/matmul_cpu_nsc
```

Custom matrix size and iterations:

```bash
./build/matmul_cpu_nsc 1024 50
```

Output format:

```text
Tilesize Execution Time Speedup
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

Output format:

```text
BlockDim,AvgKernelMs,EstimatedGBps
```

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

Output format:

```text
TileDim,AvgKernelMs,EstimatedGBps
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





