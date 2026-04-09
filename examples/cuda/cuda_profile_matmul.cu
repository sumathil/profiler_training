#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvtx3.hpp>
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t err = (call);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                << static_cast<int>(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void matmulNaive(const float *a, const float *b, float *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float sum = 0.0f;
    int rowBase = row * n;
    for (int k = 0; k < n; ++k) {
      sum += a[rowBase + k] * b[k * n + col];
    }
    c[rowBase + col] = sum;
  }
}

struct Options {
  int n = 2048;
  int warmup = 5;
  int iterations = 50;
  bool use_cublas = false;
  bool use_streams = false;
  int streams = 2;
};

Options parse_args(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--n") == 0 && i + 1 < argc) {
      opt.n = std::atoi(argv[++i]);
    } else if (strcmp(arg, "--iterations") == 0 && i + 1 < argc) {
      opt.iterations = std::atoi(argv[++i]);
    } else if (strcmp(arg, "--warmup") == 0 && i + 1 < argc) {
      opt.warmup = std::atoi(argv[++i]);
    } else if (strcmp(arg, "--use-cublas") == 0) {
      opt.use_cublas = true;
    } else if (strcmp(arg, "--use-streams") == 0) {
      opt.use_streams = true;
    } else if (strcmp(arg, "--streams") == 0 && i + 1 < argc) {
      opt.streams = std::atoi(argv[++i]);
    } else if (strcmp(arg, "--help") == 0) {
      std::cout << "Usage: ./cuda_profile_matmul [--n N] [--iterations I] "
                   "[--warmup W] [--use-cublas] [--use-streams] [--streams S]\n";
      std::exit(EXIT_SUCCESS);
    }
  }
  return opt;
}

void run_naive(int n, int iterations, int warmup) {
  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c(elements, 0.0f);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  for (int i = 0; i < warmup; ++i) {
    matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "Profiling naive kernel: " << iterations << " iterations of "
            << n << "x" << n << " matmul (warmup=" << warmup << ")\n";

  CHECK_CUDA(cudaProfilerStart());
  nvtxRangePushA("matmul_profile_window");

  for (int i = 0; i < iterations; ++i) {
    char range_name[64];
    std::snprintf(range_name, sizeof(range_name), "iter_%d", i);
    nvtxRangePushA(range_name);
    matmulNaive<<<grid, block>>>(d_a, d_b, d_c, n);
    nvtxRangePop();
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  nvtxRangePop();
  CHECK_CUDA(cudaProfilerStop());

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  float expected = 2.0f * static_cast<float>(n);
  for (int i = 0; i < 5 && i < static_cast<int>(elements); ++i) {
    if (std::fabs(h_c[i] - expected) > 1e-3f) {
      std::cerr << "Validation failed at index " << i << std::endl;
      CHECK_CUDA(cudaFree(d_a));
      CHECK_CUDA(cudaFree(d_b));
      CHECK_CUDA(cudaFree(d_c));
      std::exit(EXIT_FAILURE);
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
}

void run_cublas(int n, int iterations, int warmup) {
  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);
  std::vector<float> h_c(elements, 0.0f);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, bytes));
  CHECK_CUDA(cudaMalloc(&d_b, bytes));
  CHECK_CUDA(cudaMalloc(&d_c, bytes));

  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha, d_b, n,
                             d_a, n,
                             &beta, d_c, n));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "Profiling cuBLAS SGEMM: " << iterations << " iterations of "
            << n << "x" << n << " matmul (warmup=" << warmup << ")\n";

  CHECK_CUDA(cudaProfilerStart());
  nvtxRangePushA("cublas_profile_window");

  for (int i = 0; i < iterations; ++i) {
    char range_name[64];
    std::snprintf(range_name, sizeof(range_name), "iter_%d", i);
    nvtxRangePushA(range_name);
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha, d_b, n,
                             d_a, n,
                             &beta, d_c, n));
    nvtxRangePop();
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  nvtxRangePop();
  CHECK_CUDA(cudaProfilerStop());

  CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  float expected = 2.0f * static_cast<float>(n);
  for (int i = 0; i < 5 && i < static_cast<int>(elements); ++i) {
    if (std::fabs(h_c[i] - expected) > 1e-3f) {
      std::cerr << "Validation failed at index " << i << std::endl;
      CUBLAS_CHECK(cublasDestroy(handle));
      CHECK_CUDA(cudaFree(d_a));
      CHECK_CUDA(cudaFree(d_b));
      CHECK_CUDA(cudaFree(d_c));
      std::exit(EXIT_FAILURE);
    }
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
}

void run_streams(int n, int iterations, int warmup, int streams) {
  if (streams < 2) {
    std::cerr << "--streams must be >= 2 for stream overlap example\n";
    std::exit(EXIT_FAILURE);
  }

  size_t elements = static_cast<size_t>(n) * static_cast<size_t>(n);
  size_t bytes = elements * sizeof(float);

  std::vector<float> h_a(elements, 1.0f);
  std::vector<float> h_b(elements, 2.0f);

  std::vector<float *> d_a(streams, nullptr);
  std::vector<float *> d_b(streams, nullptr);
  std::vector<float *> d_c(streams, nullptr);
  std::vector<cudaStream_t> stream(streams);

  for (int s = 0; s < streams; ++s) {
    CHECK_CUDA(cudaStreamCreate(&stream[s]));
    CHECK_CUDA(cudaMalloc(&d_a[s], bytes));
    CHECK_CUDA(cudaMalloc(&d_b[s], bytes));
    CHECK_CUDA(cudaMalloc(&d_c[s], bytes));
    CHECK_CUDA(cudaMemcpyAsync(d_a[s], h_a.data(), bytes, cudaMemcpyHostToDevice, stream[s]));
    CHECK_CUDA(cudaMemcpyAsync(d_b[s], h_b.data(), bytes, cudaMemcpyHostToDevice, stream[s]));
  }

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

  for (int i = 0; i < warmup; ++i) {
    for (int s = 0; s < streams; ++s) {
      matmulNaive<<<grid, block, 0, stream[s]>>>(d_a[s], d_b[s], d_c[s], n);
    }
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::cout << "Profiling stream overlap: " << iterations << " iterations of "
            << n << "x" << n << " matmul across " << streams
            << " streams (warmup=" << warmup << ")\n";

  CHECK_CUDA(cudaProfilerStart());
  nvtxRangePushA("streams_profile_window");

  for (int i = 0; i < iterations; ++i) {
    char range_name[64];
    std::snprintf(range_name, sizeof(range_name), "iter_%d", i);
    nvtxRangePushA(range_name);
    for (int s = 0; s < streams; ++s) {
      char stream_name[64];
      std::snprintf(stream_name, sizeof(stream_name), "stream_%d", s);
      nvtxRangePushA(stream_name);
      matmulNaive<<<grid, block, 0, stream[s]>>>(d_a[s], d_b[s], d_c[s], n);
      nvtxRangePop();
    }
    nvtxRangePop();
  }

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  nvtxRangePop();
  CHECK_CUDA(cudaProfilerStop());

  for (int s = 0; s < streams; ++s) {
    CHECK_CUDA(cudaFree(d_a[s]));
    CHECK_CUDA(cudaFree(d_b[s]));
    CHECK_CUDA(cudaFree(d_c[s]));
    CHECK_CUDA(cudaStreamDestroy(stream[s]));
  }
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);

  if (opt.n <= 0 || opt.iterations <= 0 || opt.warmup < 0) {
    std::cerr << "Invalid arguments. Use --help for usage.\n";
    return EXIT_FAILURE;
  }

  if (opt.use_streams) {
    run_streams(opt.n, opt.iterations, opt.warmup, opt.streams);
  } else if (opt.use_cublas) {
    run_cublas(opt.n, opt.iterations, opt.warmup);
  } else {
    run_naive(opt.n, opt.iterations, opt.warmup);
  }

  return EXIT_SUCCESS;
}

/*
Build (example):
  nvcc -O2 -lineinfo -o cuda_profile_matmul cuda_profile_matmul.cu -lnvToolsExt -lcublas

Profile with Nsight Systems:
  nsys profile -o output_profile ./cuda_profile_matmul --n 2048 --iterations 50 --warmup 5

cuBLAS GEMM mode:
  ./cuda_profile_matmul --use-cublas --n 4096 --iterations 50

Stream overlap mode:
  ./cuda_profile_matmul --use-streams --streams 4 --n 2048 --iterations 50

Profile with Nsight Compute (kernel-focused):
  ncu --set full ./cuda_profile_matmul --n 2048 --iterations 10 --warmup 2
*/
