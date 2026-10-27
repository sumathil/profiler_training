cmake_minimum_required(VERSION 3.18)
project(AMD_Profiling_Examples LANGUAGES CXX)

# Find HIP
find_package(HIP REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# HIP compilation flags
set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -O3")

# Optional: Enable line info for profiling (similar to CUDA's -lineinfo)
option(ENABLE_LINEINFO "Enable line info for profiling" OFF)
if(ENABLE_LINEINFO)
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -ggdb")
endif()

# Architecture-specific flags
# Note: ROCm automatically targets the installed GPU architecture
# You can specify explicitly with: --amdgpu-target=gfx90a (MI200)
option(GPU_ARCH "Target GPU architecture (e.g., gfx90a, gfx908, gfx1030)" "")
if(GPU_ARCH)
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} --amdgpu-target=${GPU_ARCH}")
endif()

# Matrix multiplication example
hip_add_executable(matrix_multiply_hip matrix_multiply.hip)

# Find rocBLAS if available
find_package(rocblas QUIET)
if(rocblas_FOUND)
    message(STATUS "rocBLAS found, enabling rocBLAS comparison")

    # Build version with rocBLAS
    hip_add_executable(matrix_multiply_rocblas matrix_multiply_rocblas.hip)
    target_link_libraries(matrix_multiply_rocblas PRIVATE roc::rocblas)

    install(TARGETS matrix_multiply_rocblas DESTINATION bin)
else()
    message(STATUS "rocBLAS not found, skipping rocBLAS examples")
endif()

# Installation
install(TARGETS matrix_multiply_hip DESTINATION bin)

# Print build configuration
message(STATUS "")
message(STATUS "=== AMD Profiling Examples Build Configuration ===")
message(STATUS "HIP Version: ${HIP_VERSION}")
message(STATUS "HIP Compiler: ${HIP_HIPCC_EXECUTABLE}")
message(STATUS "HIP Flags: ${HIP_HIPCC_FLAGS}")
message(STATUS "Enable Line Info: ${ENABLE_LINEINFO}")
if(GPU_ARCH)
    message(STATUS "Target GPU Architecture: ${GPU_ARCH}")
else()
    message(STATUS "Target GPU Architecture: Auto-detected")
endif()
message(STATUS "================================================")
message(STATUS "")
