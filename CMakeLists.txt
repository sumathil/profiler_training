cmake_minimum_required(VERSION 3.18)
project(training LANGUAGES CXX CUDA)

# Set default build type to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type (default: RelWithDebInfo)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake options for CUDA architecture and lineinfo
option(CUDA_ARCH_SM70 "Enable sm_70 architecture" ON)
option(CUDA_ARCH_SM80 "Enable sm_80 architecture" OFF)
option(ENABLE_LINEINFO "Enable CUDA lineinfo for profiling" OFF)
option(ENABLE_KOKKOS "Enable Kokkos examples" OFF)

# Build CUDA architecture list based on options
set(CUDA_ARCH_LIST "")
if(CUDA_ARCH_SM70)
  list(APPEND CUDA_ARCH_LIST 70)
endif()
if(CUDA_ARCH_SM80)
  list(APPEND CUDA_ARCH_LIST 80)
endif()

# If no architecture is selected, default to sm_70
if(NOT CUDA_ARCH_LIST)
  set(CUDA_ARCH_LIST 70)
  message(WARNING "No CUDA architecture selected, defaulting to sm_70")
endif()

# Apply CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST} CACHE STRING "CUDA architectures" FORCE)

# Find CUDA toolkit for libraries like cuBLAS
find_package(CUDAToolkit REQUIRED)

# Convert list to readable format for display
list(JOIN CUDA_ARCH_LIST ", sm_" CUDA_ARCH_DISPLAY)
set(CUDA_ARCH_DISPLAY "sm_${CUDA_ARCH_DISPLAY}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CUDA_ARCH_DISPLAY}")
message(STATUS "CUDA lineinfo: ${ENABLE_LINEINFO}")

# Set build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")

set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")

# Add lineinfo for CUDA if enabled (useful for profiling in Release/RelWithDebInfo)
if(ENABLE_LINEINFO)
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --generate-line-info")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
  message(STATUS "CUDA line info enabled for Release and RelWithDebInfo builds")
endif()

# Define all executables

add_executable(stencil_2d_benchmark examples/cuda/stencil_2d_benchmark.cu)
add_executable(stencil_2d_tiled_benchmark examples/cuda/stencil_2d_tiled_benchmark.cu)
add_executable(stencil_2d_compare examples/cuda/stencil_2d_tiled_optimized.cu)
add_executable(matmul_cpu_naive examples/cuda/matrix_cpu_naive.cu)
add_executable(matmul_cpu_ns examples/cuda/matrix_cpu_naive_shared.cu)
add_executable(matmul_cpu_nsc examples/cuda/matrix_cpu_naive_shared_cublas.cu)

# Link libraries
#target_link_libraries(matmul_cpu_naive PRIVATE nvToolsExt)
#target_link_libraries(matmul_cpu_ns PRIVATE nvToolsExt)
#target_link_libraries(matmul_cpu_nsc PRIVATE nvToolsExt cublas)
target_link_libraries(matmul_cpu_nsc PRIVATE CUDA::cublas)

# Kokkos examples (optional)
if(ENABLE_KOKKOS)
  # Use Kokkos_DIR if provided, otherwise search in default locations
  if(DEFINED Kokkos_DIR)
    message(STATUS "Using Kokkos installation at: ${Kokkos_DIR}")
  endif()
  find_package(Kokkos REQUIRED)
  message(STATUS "Kokkos found, building Kokkos examples")

  add_executable(matmul_kokkos examples/cuda/matrix_kokkos.cpp)
  target_link_libraries(matmul_kokkos PRIVATE Kokkos::kokkos)
endif()