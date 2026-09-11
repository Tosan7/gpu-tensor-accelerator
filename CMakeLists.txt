cmake_minimum_required(VERSION 3.18)
project(gpu_tensor_accelerator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target NVIDIA GPU Architectures:
# sm_70 (Volta - V100)
# sm_80 (Ampere - A100)
# sm_89 (Ada Lovelace - RTX 4090 / L40)
# sm_90 (Hopper - H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

# Compiler flags for maximum optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)

# NVTX (NVIDIA Tools Extension) for Nsight Systems Tracing
find_package(CUDA REQUIRED)
find_library(NVTX_LIBRARY nvToolsExt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

# Kernel library
add_library(cuda_tensor_kernels STATIC
    src/kernels/fused_gelu.cu
    src/kernels/tiled_gemm.cu
    src/runtime/tensor_buffer.cpp
)

target_link_libraries(cuda_tensor_kernels
    ${CUDA_LIBRARIES}
    ${NVTX_LIBRARY}
)

# Standalone C++ benchmark binary
add_executable(cuda_benchmark src/main.cpp)
target_link_libraries(cuda_benchmark cuda_tensor_kernels)
