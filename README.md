# Profiler Training Materials

Training materials to demonstrate the use of nsight profilers using CUDA and AI examples.
The CUDA examples are matrix multiplication and 2D stencil operation. The AI examples contains CNN image classification for mnist and CIFAR-10 datasets.

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
python3.11 -m pip install -r requirements.txt
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

### Matrix multiply

Default run:

```bash
./build/matmul_benchmark
```

Custom matrix size and iterations:

```bash
./build/matmul_benchmark 1024 50
```

Arguments:
- `matrix_size` (default: `1024`)
- `iterations` (default: `50`)


### Matrix multiply (shared memory)

Default run:

```bash
./build/matmul_tiled_benchmark
```

Custom matrix size and iterations:

```bash
./build/matmul_tiled_benchmark 1024 50
```

Output format:

```text
TileSize,AvgKernelMs,EstimatedGFLOPS
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






