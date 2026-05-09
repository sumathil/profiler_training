#pragma once
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                << " -> " << cudaGetErrorString(err__)                         \
                << " (" << static_cast<int>(err__) << ")" << std::endl;        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

inline float elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t err = (call);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                << static_cast<int>(err) << std::endl;                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
