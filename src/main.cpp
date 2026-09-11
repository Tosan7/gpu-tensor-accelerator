#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cuda_runtime.h>
#include "kernels/fused_gelu.cuh"
#include "kernels/tiled_gemm.cuh"
#include "runtime/tensor_buffer.hpp"
#include "runtime/nvtx_profiler.hpp"

void benchmark_fused_gelu(int batch_size, int seq_len, int hidden_dim) {
    int total_elements = batch_size * seq_len * hidden_dim;
    size_t tensor_bytes = total_elements * sizeof(float);
    size_t bias_bytes = hidden_dim * sizeof(float);

    std::cout << "\n=======================================================\n";
    std::cout << "  Benchmark 1: Fused Bias-Add + GeLU Activation\n";
    std::cout << "=======================================================\n";
    std::cout << "Tensor Shape: [" << batch_size << "x" << seq_len << "x" << hidden_dim << "]\n";
    std::cout << "Total Elements: " << total_elements << " (" << tensor_bytes / (1024.0 * 1024.0) << " MB)\n";

    // Allocate GPU buffers via RAII
    GpuTensorBuffer d_input(total_elements);
    GpuTensorBuffer d_bias(hidden_dim);
    GpuTensorBuffer d_output(total_elements);

    // Host initialization
    std::vector<float> h_input(total_elements, 1.0f);
    std::vector<float> h_bias(hidden_dim, 0.5f);
    d_input.copy_from_host(h_input.data());
    d_bias.copy_from_host(h_bias.data());

    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < 5; ++i) {
        launch_fused_bias_gelu(d_input.data(), d_bias.data(), d_output.data(), total_elements, hidden_dim);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    const int ITERS = 100;
    cudaEventRecord(start);
    {
        ScopedNvtxRange range("FusedGeLU_100x_Benchmark");
        for (int i = 0; i < ITERS; ++i) {
            launch_fused_bias_gelu(d_input.data(), d_bias.data(), d_output.data(), total_elements, hidden_dim);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / ITERS;

    // Memory throughput: Read input + Read bias + Write output
    // (Total bytes moved: 2 * total_elements * sizeof(float) + bias_bytes)
    double bytes_moved = (2.0 * tensor_bytes + bias_bytes);
    double bandwidth_gb_s = (bytes_moved / (avg_time_ms / 1000.0)) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg Kernel Execution Time: " << avg_time_ms << " ms\n";
    std::cout << "Effective Memory Bandwidth: " << bandwidth_gb_s << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void benchmark_tiled_gemm(int M, int N, int K) {
    std::cout << "\n=======================================================\n";
    std::cout << "  Benchmark 2: Shared-Memory Tiled GEMM (Matrix Multiply)\n";
    std::cout << "=======================================================\n";
    std::cout << "Matrix Dimensions: [" << M << " x " << K << "] * [" << K << " x " << N << "]\n";

    GpuTensorBuffer d_A(M * K);
    GpuTensorBuffer d_B(K * N);
    GpuTensorBuffer d_C(M * N);

    std::vector<float> h_A(M * K, 0.05f);
    std::vector<float> h_B(K * N, 0.02f);
    d_A.copy_from_host(h_A.data());
    d_B.copy_from_host(h_B.data());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        launch_tiled_gemm(d_A.data(), d_B.data(), d_C.data(), M, N, K);
    }
    cudaDeviceSynchronize();

    const int ITERS = 50;
    cudaEventRecord(start);
    {
        ScopedNvtxRange range("TiledGEMM_50x_Benchmark");
        for (int i = 0; i < ITERS; ++i) {
            launch_tiled_gemm(d_A.data(), d_B.data(), d_C.data(), M, N, K);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_ms = elapsed_ms / ITERS;

    // FLOPs calculation: 2 * M * N * K (multiply-accumulate)
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double gflops = (flops / (avg_time_ms / 1000.0)) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Avg GEMM Execution Time:   " << avg_time_ms << " ms\n";
    std::cout << "Compute Throughput:        " << gflops << " GFLOPS\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "  NVIDIA GPU Tensor Accelerator C++ Benchmark Runner\n";
    std::cout << "=======================================================\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU detected. Please run on an NVIDIA GPU environment.\n";
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Target Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

    // Run benchmarks
    benchmark_fused_gelu(16, 512, 4096); // LLM Transformer layer dimension
    benchmark_tiled_gemm(2048, 2048, 2048);

    std::cout << "\n[SUCCESS] C++ Benchmark Suite Completed.\n";
    return 0;
}
