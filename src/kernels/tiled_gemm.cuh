#pragma once
#include <cuda_runtime.h>

/**
 * @brief High-Performance Shared-Memory Tiled GEMM (General Matrix Multiply)
 * 
 * Computes: C = alpha * (A x B) + beta * C
 * Matrix dimensions: A is [M x K], B is [K x N], C is [M x N]
 * 
 * Architecture Optimizations:
 * 1. 2D Tiling with on-chip shared memory (__shared__) to reduce global memory DRAM traffic by TILE_SIZE factor.
 * 2. Loop unrolling (#pragma unroll) for instruction-level parallelism (ILP).
 * 3. Thread coarsening and bank-conflict avoidance.
 */

constexpr int TILE_DIM = 32; // 32x32 = 1024 threads per block (maximum hardware occupancy)

cudaError_t launch_tiled_gemm(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
);
