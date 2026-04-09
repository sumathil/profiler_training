cmake_minimum_required(VERSION 3.18)
project(profiler_training LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

add_executable(matmul_benchmark examples/cuda/matmul_benchmark.cu)
add_executable(matmul_tiled_benchmark examples/cuda/matmul_tiled_benchmark.cu)
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

target_compile_options(stencil_2d_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(stencil_2d_tiled_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_compile_options(cuda_profile_matmul PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

if(TARGET CUDA::cublas)
  target_link_libraries(cuda_profile_matmul PRIVATE CUDA::cublas)
else()
  target_link_libraries(cuda_profile_matmul PRIVATE cublas)
endif()

if(NOT TARGET CUDA::cublas)
  target_link_directories(cuda_profile_matmul PRIVATE
    ${CUDAToolkit_LIBRARY_DIR}
  )
endif()
