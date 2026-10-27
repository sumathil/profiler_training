cmake_minimum_required(VERSION 3.18)

# Platform detection option
option(USE_AMD "Build for AMD/ROCm platform" OFF)
option(USE_CUDA "Build for NVIDIA/CUDA platform" ON)

# Set project languages based on platform
if(USE_AMD)
  project(training LANGUAGES CXX HIP)
  set(USE_CUDA OFF)
elseif(USE_CUDA)
  project(training LANGUAGES CXX CUDA)
else()
  message(FATAL_ERROR "Either USE_CUDA or USE_AMD must be enabled")
endif()

# Set default build type to RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type (default: RelWithDebInfo)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific options
option(ENABLE_LINEINFO "Enable lineinfo for profiling" OFF)
option(ENABLE_KOKKOS "Enable Kokkos examples" OFF)

# CUDA-specific options
if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  option(CUDA_ARCH_SM70 "Enable sm_70 architecture (V100)" ON)
  option(CUDA_ARCH_SM80 "Enable sm_80 architecture (A100)" OFF)
  option(CUDA_ARCH_SM90 "Enable sm_90 architecture (H100)" OFF)
endif()

# AMD-specific options
if(USE_AMD)
  option(AMD_ARCH_GFX908 "Enable gfx908 architecture (MI100)" OFF)
  option(AMD_ARCH_GFX90A "Enable gfx90a architecture (MI210/MI250)" ON)
  option(AMD_ARCH_GFX942 "Enable gfx942 architecture (MI300)" OFF)
endif()

# CUDA-specific setup
if(USE_CUDA)
  # Build CUDA architecture list based on options
  set(CUDA_ARCH_LIST "")
  if(CUDA_ARCH_SM70)
    list(APPEND CUDA_ARCH_LIST 70)
  endif()
  if(CUDA_ARCH_SM80)
    list(APPEND CUDA_ARCH_LIST 80)
  endif()
  if(CUDA_ARCH_SM90)
    list(APPEND CUDA_ARCH_LIST 90)
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

  message(STATUS "Platform: NVIDIA CUDA")
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
  message(STATUS "CUDA architectures: ${CUDA_ARCH_DISPLAY}")
  message(STATUS "CUDA lineinfo: ${ENABLE_LINEINFO}")
endif()

# AMD-specific setup
if(USE_AMD)
  # Find ROCm/HIP
  find_package(hip REQUIRED)
  find_package(rocblas REQUIRED)

  # Build AMD architecture list based on options
  set(AMD_ARCH_LIST "")
  if(AMD_ARCH_GFX908)
    list(APPEND AMD_ARCH_LIST "gfx908")
  endif()
  if(AMD_ARCH_GFX90A)
    list(APPEND AMD_ARCH_LIST "gfx90a")
  endif()
  if(AMD_ARCH_GFX942)
    list(APPEND AMD_ARCH_LIST "gfx942")
  endif()

  # If no architecture is selected, default to gfx90a
  if(NOT AMD_ARCH_LIST)
    set(AMD_ARCH_LIST "gfx90a")
    message(WARNING "No AMD architecture selected, defaulting to gfx90a")
  endif()

  # Apply AMD architecture settings
  list(JOIN AMD_ARCH_LIST "," AMD_ARCH_STRING)
  set(CMAKE_HIP_ARCHITECTURES ${AMD_ARCH_STRING} CACHE STRING "AMD GPU architectures" FORCE)

  # Convert list to readable format for display
  list(JOIN AMD_ARCH_LIST ", " AMD_ARCH_DISPLAY)

  message(STATUS "Platform: AMD ROCm/HIP")
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
  message(STATUS "AMD architectures: ${AMD_ARCH_DISPLAY}")
  message(STATUS "HIP lineinfo: ${ENABLE_LINEINFO}")
endif()

# Set build-type specific flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")

if(USE_CUDA)
  set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")

  # Add lineinfo for CUDA if enabled (useful for profiling in Release/RelWithDebInfo)
  if(ENABLE_LINEINFO)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --generate-line-info")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} --generate-line-info")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -lineinfo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -lineinfo")
    message(STATUS "CUDA/CXX line info enabled for Release and RelWithDebInfo builds")
  endif()
endif()

if(USE_AMD)
  set(CMAKE_HIP_FLAGS_DEBUG "-g -O0")
  set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-g -O2 -DNDEBUG")

  # Add lineinfo for HIP if enabled (useful for profiling in Release/RelWithDebInfo)
  if(ENABLE_LINEINFO)
    set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -gline-tables-only")
    set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "${CMAKE_HIP_FLAGS_RELWITHDEBINFO} -gline-tables-only")
    message(STATUS "HIP line info enabled for Release and RelWithDebInfo builds")
  endif()
endif()

# Define all executables based on platform

if(USE_CUDA)
  add_executable(stencil_2d_benchmark examples/cuda/stencil_2d_benchmark.cu)
  add_executable(stencil_2d_tiled_benchmark examples/cuda/stencil_2d_tiled_benchmark.cu)
  add_executable(stencil_2d_compare examples/cuda/stencil_2d_tiled_optimized.cu)
  add_executable(matmul_cpu_naive examples/cuda/matrix_cpu_naive.cu)
  add_executable(matmul_cpu_ns examples/cuda/matrix_cpu_naive_shared.cu)
  add_executable(matmul_cpu_nsc examples/cuda/matrix_cpu_naive_shared_cublas.cu)

  # Link CUDA libraries
  target_link_libraries(matmul_cpu_nsc PRIVATE CUDA::cublas)
endif()

if(USE_AMD)
  add_executable(matmul_hip examples/amd/matrix_multiply.hip)
  add_executable(matmul_rocblas examples/amd/matrix_multiply_rocblas.hip)

  # Link ROCm/HIP libraries
  target_link_libraries(matmul_hip PRIVATE hip::device)
  target_link_libraries(matmul_rocblas PRIVATE hip::device roc::rocblas)
endif()

# Kokkos examples (optional) - works with both CUDA and AMD
if(ENABLE_KOKKOS)
  # If KOKKOS_INSTALL_DIR is provided (installation root), construct the CMake config path
  if(DEFINED KOKKOS_INSTALL_DIR)
    message(STATUS "Using Kokkos installation root: ${KOKKOS_INSTALL_DIR}")
    # Try common locations for the CMake config file
    if(EXISTS "${KOKKOS_INSTALL_DIR}/lib64/cmake/Kokkos/KokkosConfig.cmake")
      set(Kokkos_DIR "${KOKKOS_INSTALL_DIR}/lib64/cmake/Kokkos")
    elseif(EXISTS "${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos/KokkosConfig.cmake")
      set(Kokkos_DIR "${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos")
    else()
      message(FATAL_ERROR "Could not find KokkosConfig.cmake in ${KOKKOS_INSTALL_DIR}/lib64/cmake/Kokkos or ${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos")
    endif()
    message(STATUS "Found Kokkos CMake config at: ${Kokkos_DIR}")
  elseif(DEFINED Kokkos_DIR)
    # Kokkos_DIR was provided directly (backwards compatibility)
    message(STATUS "Using Kokkos CMake config at: ${Kokkos_DIR}")
  endif()

  find_package(Kokkos REQUIRED)
  message(STATUS "Kokkos found, building Kokkos examples (platform-agnostic)")

  add_executable(matmul_kokkos examples/cuda/matrix_kokkos.cpp)
  target_link_libraries(matmul_kokkos PRIVATE Kokkos::kokkos)
endif()