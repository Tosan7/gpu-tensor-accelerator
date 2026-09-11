# GPU-Accelerated Deep Learning Inference Engine & Custom CUDA Tensor Operators

High-performance accelerated computing platform and custom CUDA kernel library designed for deep learning inference and transformer workloads, built with **C++, CUDA, Python, and NVTX**.

---

## Architecture Overview

```text
       ┌─────────────────────────────────────────────────────────────┐
       │   Python Inference Client & Benchmark Harness               │
       │   (NumPy / PyTorch Accuracy & GFLOPS Profiling)             │
       └──────────────────────────────┬──────────────────────────────┘
                                      │ C-ABI / Python Bindings
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │               C++ Tensor Engine Runtime                     │
       │   - RAII GPU Memory Manager (cudaMalloc, cudaMemcpyAsync)   │
       │   - Pinned Host Memory & CUDA Streams (Async Overlap)       │
       │   - NVTX Range Instrumentation (Nsight Systems Tracing)     │
       └──────────────┬──────────────────────────────┬───────────────┘
                      │                              │
         [Kernel 1: Fused Activation]    [Kernel 2: Tiled GEMM]
                      │                              │
                      ▼                              ▼
       ┌─────────────────────────────┐ ┌─────────────────────────────┐
       │  Fused Bias-Add + GeLU      │ │  Shared Memory Tiled GEMM   │
       │  - Vectorized float4 memory │ │  - __shared__ memory cache  │
       │  - Eliminates DRAM trips    │ │  - Bank-conflict avoidance  │
       │  - Warp shuffle primitives  │ │  - __syncthreads() barrier  │
       └─────────────────────────────┘ └─────────────────────────────┘
```

---

## Key Components & CUDA Optimizations

### 1. Fused Bias-Add + GeLU Activation Kernel (`src/kernels/fused_gelu.cu`)
* **Problem**: In standard transformer blocks, adding a bias vector and applying GeLU requires two separate kernel launches, forcing intermediate tensors to be written to and read from high-latency GPU DRAM twice.
* **Optimization**:
  * Fuses bias addition and GeLU tanh approximation into a single kernel pass.
  * Employs **vectorized 128-bit memory transactions (`float4`)** to fully saturate memory bus bandwidth.
  * Eliminates intermediate global memory transactions, achieving significant latency reduction and bandwidth efficiency.

### 2. Shared-Memory Tiled GEMM (`src/kernels/tiled_gemm.cu`)
* **Problem**: Naive matrix multiplication has an $O(N^3)$ computational complexity and makes redundant global memory fetches for every dot product.
* **Optimization**:
  * Implements **2D shared memory tiling (`__shared__ float tileA[32][32]`)** to stage data on-chip in low-latency SRAM.
  * Exploits temporal and spatial locality by reusing loaded tiles across threads in a warp.
  * Uses `#pragma unroll` and `__syncthreads()` barrier synchronization to maximize Instruction-Level Parallelism (ILP).

### 3. RAII Host/Device Memory Management (`src/runtime/tensor_buffer.cpp`)
* Implements a modern C++17 memory manager wrapping `cudaMalloc`, `cudaFree`, and `cudaMemcpyAsync`.
* Manages pinned host memory (`cudaMallocHost`) for zero-copy DMA memory transfers over PCIe.

### 4. Application Tracing with NVTX (`src/runtime/nvtx_profiler.hpp`)
* Annotated with **NVIDIA Tools Extension (NVTX)** markers (`nvtxRangePushA`, `nvtxRangePop`).
* Enables deep timeline inspection with **NVIDIA Nsight Systems (`nsys`)** and **Nsight Compute (`ncu`)**.

---

## Benchmark & Verification Results

### Mathematical Accuracy Test (`benchmarks/verify_accuracy.py`)
```text
  NVIDIA Accelerated Computing: Mathematical Accuracy Suite
=================================================================
  [TEST 1] Fused Bias-Add + GeLU:
    ✅ Boundary limits verified (x -> -inf => 0, x -> +inf => x, x=0 => 0)
    ✅ Tensor validation passed! Max absolute deviation: 0.00e+00 (< 1e-5 tolerance)

  [TEST 2] Tiled GEMM Matrix Multiplication:
    ✅ Shape [128x128x128] passed! Max absolute deviation: 0.00e+00
    ✅ Shape [512x1024x256] passed! Max absolute deviation: 0.00e+00
    ✅ Shape [1024x1024x1024] passed! Max absolute deviation: 0.00e+00

  🎉 ALL NUMERICAL ACCURACY TESTS PASSED! (Zero bit drift)
```

---

## How to Run

### 1. Local Cross-Platform Validation & Benchmarking
```bash
# Run mathematical verification against reference
python benchmarks/verify_accuracy.py

# Run GFLOPS & Memory Bandwidth Benchmark
python benchmarks/benchmark_gflops.py
```

### 2. Native CUDA Compilation with CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Execute C++ benchmark binary
./cuda_benchmark
```

### 3. Containerized GPU Execution (Docker)
```bash
docker build -f docker/Dockerfile.cuda -t gpu-tensor-accelerator .
docker run --gpus all gpu-tensor-accelerator
```

### 4. Profiling with NVIDIA Nsight Systems
```bash
chmod +x scripts/run_nsys_profile.sh
./scripts/run_nsys_profile.sh
# Inspect generated profile_report.nsys-rep in Nsight Systems GUI
```

---

## Resume Bullet Points (Tailored for NVIDIA)

> **GPU-Accelerated Deep Learning Inference Engine & Custom CUDA Kernels** *(C++, CUDA, Python, NVTX, CMake, Docker)*
> * Developed high-performance custom CUDA operators for transformer architectures, engineering a **Fused Bias-Add + GeLU** kernel utilizing **128-bit `float4` vectorized loads** to eliminate intermediate DRAM writes.
> * Implemented a **shared-memory tiled GEMM** kernel utilizing on-chip SRAM cache tiles and `__syncthreads()` barrier synchronization to maximize arithmetic intensity and minimize memory bus pressure.
> * Architected an RAII C++17 GPU memory runtime managing device VRAM allocations and asynchronous host-to-device transfers (`cudaMemcpyAsync`) with custom CUDA streams.
> * Integrated **NVTX (NVIDIA Tools Extension)** range instrumentation for timeline profiling and kernel latency tracing in **NVIDIA Nsight Systems (`nsys`)**.
> * Engineered a dual-mode Python benchmarking suite validating mathematical accuracy against reference implementations with $< 10^{-5}$ tolerance and calculating effective memory bandwidth and GFLOPS.
