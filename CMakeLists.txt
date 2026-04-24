cmake_minimum_required(VERSION 3.18)
project(simple_benchmarking_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

add_executable(matmul_benchmark examples/cuda/matmul_benchmark.cu)
add_executable(matmul_tiled_benchmark examples/cuda/matmul_tiled_benchmark.cu)
add_executable(matmul_naive_vs_tiled examples/cuda/matmul_naive_vs_tiled.cu)
add_executable(mm_cuda examples/cuda/MM_3_methods.cu)
add_executable(matmul_4way_complex examples/cuda/matmul_cpu_naive_tiled_async_cublas.cu)
add_executable(stencil_2d_benchmark examples/cuda/stencil_2d_benchmark.cu)
add_executable(stencil_2d_tiled_benchmark examples/cuda/stencil_2d_tiled_benchmark.cu)
add_executable(cuda_profile_matmul examples/cuda/cuda_profile_matmul.cu)

target_compile_options(matmul_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(matmul_tiled_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(matmul_naive_vs_tiled PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(matmul_4way_complex PRIVATE
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

find_package(CUDAToolkit REQUIRED)

# NVTX3 is header-only, no linking needed!
# Just add the include directory if needed
target_include_directories(matmul_benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(matmul_naive_vs_tiled PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(matmul_4way_complex PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(mm_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Link cuBLAS
if(TARGET CUDA::cublas)
  target_link_libraries(matmul_4way_complex PRIVATE CUDA::cublas)
  target_link_libraries(cuda_profile_matmul PRIVATE CUDA::cublas)
  target_link_libraries(mm_cuda PRIVATE CUDA::cublas)
else()
  target_link_libraries(matmul_4way_complex PRIVATE cublas)
  target_link_libraries(cuda_profile_matmul PRIVATE cublas)
  target_link_directories(matmul_4way_complex PRIVATE ${CUDAToolkit_LIBRARY_DIR})
  target_link_directories(cuda_profile_matmul PRIVATE ${CUDAToolkit_LIBRARY_DIR})
  target_include_directories(mm_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()