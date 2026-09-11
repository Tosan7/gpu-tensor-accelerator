#include "fused_gelu.cuh"
#include <cuda_runtime.h>
#include <math.h>

/**
 * Device helper: Scalar GeLU calculation using fast tanh approximation
 * GeLU(x) = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * (x + GELU_COEFF * x^3)))
 */
__device__ __forceinline__ float compute_gelu(float x) {
    float x_cubed = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * CUDA Kernel: Vectorized float4 Fused Bias-Add + GeLU
 * Each thread processes 4 consecutive float elements (16 bytes = 128-bit transaction).
 * Squeezes maximum memory bandwidth from NVIDIA memory controllers.
 */
__global__ void fused_bias_gelu_vectorized_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int num_vec_elements,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vec_elements) {
        float4 in_val = input[idx];
        
        // Element indices within the hidden dimension for bias broadcasting
        int base_col = (idx * 4) % hidden_dim;

        float b0 = bias[(base_col + 0) % hidden_dim];
        float b1 = bias[(base_col + 1) % hidden_dim];
        float b2 = bias[(base_col + 2) % hidden_dim];
        float b3 = bias[(base_col + 3) % hidden_dim];

        float4 out_val;
        out_val.x = compute_gelu(in_val.x + b0);
        out_val.y = compute_gelu(in_val.y + b1);
        out_val.z = compute_gelu(in_val.z + b2);
        out_val.w = compute_gelu(in_val.w + b3);

        output[idx] = out_val;
    }
}

/**
 * CUDA Kernel: Scalar fallback for tail elements
 */
__global__ void fused_bias_gelu_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int col = idx % hidden_dim;
        output[idx] = compute_gelu(input[idx] + bias[col]);
    }
}

cudaError_t launch_fused_bias_gelu(
    const float* d_input,
    const float* d_bias,
    float* d_output,
    int total_elements,
    int hidden_dim,
    cudaStream_t stream
) {
    // If total_elements is a multiple of 4 and pointers are 16-byte aligned, use vectorized float4
    if (total_elements % 4 == 0 &&
        reinterpret_cast<uintptr_t>(d_input) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(d_output) % 16 == 0) 
    {
        int vec_elements = total_elements / 4;
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        fused_bias_gelu_vectorized_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const float4*>(d_input),
            d_bias,
            reinterpret_cast<float4*>(d_output),
            vec_elements,
            hidden_dim
        );
    } else {
        constexpr int BLOCK_SIZE = 256;
        int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

        fused_bias_gelu_scalar_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            d_input,
            d_bias,
            d_output,
            total_elements,
            hidden_dim
        );
    }

    return cudaGetLastError();
}
