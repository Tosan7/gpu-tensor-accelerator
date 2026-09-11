#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "nvtx_profiler.hpp"

/**
 * @brief RAII GPU Device & Pinned Host Tensor Memory Manager
 * 
 * Manages allocation and lifecycle of:
 * - Device VRAM memory (cudaMalloc / cudaFree)
 * - Page-locked / Pinned Host RAM (cudaMallocHost / cudaFreeHost) for DMA transfers
 * - Asynchronous H2D (Host-to-Device) and D2H (Device-to-Host) stream copies
 */
class GpuTensorBuffer {
public:
    explicit GpuTensorBuffer(size_t num_elements);
    ~GpuTensorBuffer();

    // Prevent accidental copying (resource safety)
    GpuTensorBuffer(const GpuTensorBuffer&) = delete;
    GpuTensorBuffer& operator=(const GpuTensorBuffer&) = delete;

    // Allow move semantics
    GpuTensorBuffer(GpuTensorBuffer&& other) noexcept;
    GpuTensorBuffer& operator=(GpuTensorBuffer&& other) noexcept;

    // Asynchronous memory transfers
    void copy_from_host_async(const float* host_ptr, cudaStream_t stream = nullptr);
    void copy_to_host_async(float* host_ptr, cudaStream_t stream = nullptr) const;

    // Synchronous memory transfers
    void copy_from_host(const float* host_ptr);
    void copy_to_host(float* host_ptr) const;

    // Accessors
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }
    size_t size() const { return num_elements_; }
    size_t bytes() const { return num_elements_ * sizeof(float); }

private:
    float* d_data_ = nullptr;
    size_t num_elements_ = 0;
};
