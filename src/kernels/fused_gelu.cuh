#pragma once
#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Fused Bias-Add + GeLU (Gaussian Error Linear Unit) Activation Kernel
 * 
 * Computes: output[i] = GeLU(input[i] + bias[i % hidden_dim])
 * 
 * Optimizations:
 * 1. Vectorized 128-bit memory transactions using float4 loads/stores to saturate DRAM bandwidth.
 * 2. Arithmetic fusion: avoids storing intermediate (input + bias) in global memory.
 * 3. Fast math approximations: uses native hardware __expf, __fdividef, and tanh.
 */

// Host launcher function
cudaError_t launch_fused_bias_gelu(
    const float* d_input,
    const float* d_bias,
    float* d_output,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream = nullptr
);

// Mathematical approximation constants for GeLU
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f; // sqrt(2 / pi)
constexpr float GELU_COEFF     = 0.044715f;
