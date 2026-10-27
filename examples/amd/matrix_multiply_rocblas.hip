#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
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

#define CHECK_ROCBLAS(call) \
    { \
        rocblas_status status = call; \
        if (status != rocblas_status_success) { \
            std::cerr << "rocBLAS Error: " << rocblas_status_to_string(status) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

// Tiled matrix multiplication kernel (same as before)
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

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main(int argc, char** argv) {
    int M = 2048;
    int N = 2048;
    int K = 2048;

    if (argc > 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    std::cout << "Matrix Multiplication Comparison: Custom Kernel vs rocBLAS\n";
    std::cout << "C(" << M << "x" << N << ") = A(" << M << "x" << K
              << ") * B(" << K << "x" << N << ")\n\n";

    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_custom = (float*)malloc(sizeC);
    float* h_C_rocblas = (float*)malloc(sizeC);

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

    // Setup for custom kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up custom kernel
    hipLaunchKernelGGL(matmul_tiled, gridDim, blockDim, 0, 0,
                       d_A, d_B, d_C, M, N, K);
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark custom kernel
    int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(matmul_tiled, gridDim, blockDim, 0, 0,
                           d_A, d_B, d_C, M, N, K);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double time_custom = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Copy custom kernel result
    CHECK_HIP(hipMemcpy(h_C_custom, d_C, sizeC, hipMemcpyDeviceToHost));

    // rocBLAS setup
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));

    // rocBLAS parameters
    // rocblas_sgemm computes: C = alpha * A * B + beta * C
    // Note: rocBLAS assumes column-major, so we compute B^T * A^T = C^T
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up rocBLAS
    CHECK_ROCBLAS(rocblas_sgemm(handle,
                                rocblas_operation_none, rocblas_operation_none,
                                N, M, K,
                                &alpha,
                                d_B, N,
                                d_A, K,
                                &beta,
                                d_C, N));
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark rocBLAS
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        CHECK_ROCBLAS(rocblas_sgemm(handle,
                                    rocblas_operation_none, rocblas_operation_none,
                                    N, M, K,
                                    &alpha,
                                    d_B, N,
                                    d_A, K,
                                    &beta,
                                    d_C, N));
    }
    CHECK_HIP(hipDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double time_rocblas = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Copy rocBLAS result
    CHECK_HIP(hipMemcpy(h_C_rocblas, d_C, sizeC, hipMemcpyDeviceToHost));

    // Calculate performance metrics
    double gflops = (2.0 * M * N * K * 1e-9);

    std::cout << "\n=== Performance Results ===\n";
    std::cout << "Custom Tiled Kernel: " << time_custom << " ms, "
              << gflops / (time_custom / 1000.0) << " GFLOPS\n";
    std::cout << "rocBLAS (SGEMM):     " << time_rocblas << " ms, "
              << gflops / (time_rocblas / 1000.0) << " GFLOPS\n";
    std::cout << "rocBLAS speedup:     " << time_custom / time_rocblas << "x\n";

    // Verify results match
    std::cout << "\n=== Verification ===\n";
    double max_diff = 0.0;
    int mismatch_count = 0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(h_C_custom[i] - h_C_rocblas[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3) {
            mismatch_count++;
        }
    }
    std::cout << "Max difference: " << max_diff << "\n";
    std::cout << "Mismatches: " << mismatch_count << " / " << M * N << "\n";
    if (mismatch_count == 0) {
        std::cout << "Results match! ✓\n";
    } else {
        std::cout << "Results differ (expected due to FP precision)\n";
    }

    // Cleanup
    rocblas_destroy_handle(handle);
    free(h_A);
    free(h_B);
    free(h_C_custom);
    free(h_C_rocblas);
    CHECK_HIP(hipFree(d_A));
    CHECK_HIP(hipFree(d_B));
    CHECK_HIP(hipFree(d_C));

    return 0;
}
