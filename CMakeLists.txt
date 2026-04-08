cmake_minimum_required(VERSION 3.18)
project(simple_benchmarking_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

add_executable(benchmark examples/cuda/benchmark.cu)
add_executable(matmul_benchmark examples/cuda/matmul_benchmark.cu)
add_executable(matmul_tiled_benchmark examples/cuda/matmul_tiled_benchmark.cu)
add_executable(stencil_2d_benchmark examples/cuda/stencil_2d_benchmark.cu)
add_executable(stencil_2d_tiled_benchmark examples/cuda/stencil_2d_tiled_benchmark.cu)
add_executable(matmul_demo examples/cuda/matrix_demo.cu)
add_executable(matmul examples/cuda/matrix_multiplication.cu)

target_compile_options(benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(matmul_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(matmul_tiled_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(stencil_2d_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(stencil_2d_tiled_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
