#include "tiled_gemm.cuh"
#include <cuda_runtime.h>

/**
 * CUDA Kernel: Tiled Matrix Multiplication using Shared Memory
 * Each block computes a [TILE_DIM x TILE_DIM] patch of matrix C.
 */
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha,
    float beta
) {
    // 2D shared memory cache buffers allocated on-chip (SRAM)
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    // Thread coordinates inside the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column of C that this thread calculates
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;

    // Loop through all tiles along dimension K
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        // Cooperatively load tile A into shared memory
        int a_col = t * TILE_DIM + tx;
        if (row < M && a_col < K) {
            tileA[ty][tx] = A[row * K + a_col];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Cooperatively load tile B into shared memory
        int b_row = t * TILE_DIM + ty;
        if (b_row < K && col < N) {
            tileB[ty][tx] = B[b_row * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        // Barrier synchronization: Wait for all threads in the block to finish loading shared memory
        __syncthreads();

        // Accumulate partial dot product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += tileA[ty][k] * tileB[k][tx];
        }

        // Barrier synchronization: Ensure computation finishes before next tile loads over shared memory
        __syncthreads();
    }

    // Write final result to global memory
    if (row < M && col < N) {
        if (beta != 0.0f) {
            C[row * N + col] = alpha * acc + beta * C[row * N + col];
        } else {
            C[row * N + col] = alpha * acc;
        }
    }
}

cudaError_t launch_tiled_gemm(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 block_dim(TILE_DIM, TILE_DIM); // 32 x 32 = 1024 threads
    dim3 grid_dim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_A, d_B, d_C,
        M, N, K,
        alpha, beta
    );

    return cudaGetLastError();
}
