#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

#define CHECK_HIP(call) \
    { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error: " << hipGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Naive matrix multiplication kernel
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared memory tiled matrix multiplication
#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tiles into shared memory
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Vectorized load version with float4
__global__ void matmul_vectorized(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Verify results
bool verifyResult(const float* C_host, const float* C_device, int M, int N,
                  float tolerance = 1e-3) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_host[i] - C_device[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": "
                      << C_host[i] << " vs " << C_device[i] << std::endl;
            return false;
        }
    }
    return true;
}

// CPU reference implementation
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    // Matrix dimensions
    int M = 2048;  // Rows of A and C
    int N = 2048;  // Columns of B and C
    int K = 2048;  // Columns of A, rows of B

    if (argc > 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    std::cout << "Matrix Multiplication: C(" << M << "x" << N << ") = "
              << "A(" << M << "x" << K << ") * B(" << K << "x" << N << ")\n";

    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize matrices
    srand(42);
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_HIP(hipMalloc(&d_A, sizeA));
    CHECK_HIP(hipMalloc(&d_B, sizeB));
    CHECK_HIP(hipMalloc(&d_C, sizeC));

    // Copy data to device
    CHECK_HIP(hipMemcpy(d_A, h_A, sizeA, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, sizeB, hipMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    hipLaunchKernelGGL(matmul_naive, gridDim, blockDim, 0, 0,
                       d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark naive kernel
    int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(matmul_naive, gridDim, blockDim, 0, 0,
                           d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Copy result back
    CHECK_HIP(hipMemcpy(h_C, d_C, sizeC, hipMemcpyDeviceToHost));

    // Benchmark tiled kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(matmul_tiled, gridDim, blockDim, 0, 0,
                           d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double time_tiled = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Copy tiled result
    float* h_C_tiled = (float*)malloc(sizeC);
    CHECK_HIP(hipMemcpy(h_C_tiled, d_C, sizeC, hipMemcpyDeviceToHost));

    // Benchmark vectorized kernel
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(matmul_vectorized, gridDim, blockDim, 0, 0,
                           d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double time_vectorized = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Calculate performance metrics
    double gflops = (2.0 * M * N * K * 1e-9);

    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Naive kernel:      " << time_naive << " ms, "
              << gflops / (time_naive / 1000.0) << " GFLOPS\n";
    std::cout << "Tiled kernel:      " << time_tiled << " ms, "
              << gflops / (time_tiled / 1000.0) << " GFLOPS\n";
    std::cout << "Vectorized kernel: " << time_vectorized << " ms, "
              << gflops / (time_vectorized / 1000.0) << " GFLOPS\n";
    std::cout << "Speedup (tiled vs naive): " << time_naive / time_tiled << "x\n";
    std::cout << "Speedup (vectorized vs naive): " << time_naive / time_vectorized << "x\n";

    // Verify results (optional, computationally expensive)
    // std::cout << "\nComputing CPU reference...\n";
    // matmul_cpu(h_A, h_B, h_C_ref, M, N, K);
    // if (verifyResult(h_C_ref, h_C, M, N)) {
    //     std::cout << "Verification PASSED!\n";
    // } else {
    //     std::cout << "Verification FAILED!\n";
    // }

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_C_tiled);
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    return 0;
}
