#include "tensor_buffer.hpp"
#include <iostream>

GpuTensorBuffer::GpuTensorBuffer(size_t num_elements) 
    : num_elements_(num_elements) {
    if (num_elements_ == 0) return;

    ScopedNvtxRange range("GpuTensorBuffer::Allocate");
    size_t total_bytes = bytes();
    cudaError_t err = cudaMalloc(&d_data_, total_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
}

GpuTensorBuffer::~GpuTensorBuffer() {
    if (d_data_) {
        cudaFree(d_data_);
        d_data_ = nullptr;
    }
}

GpuTensorBuffer::GpuTensorBuffer(GpuTensorBuffer&& other) noexcept
    : d_data_(other.d_data_), num_elements_(other.num_elements_) {
    other.d_data_ = nullptr;
    other.num_elements_ = 0;
}

GpuTensorBuffer& GpuTensorBuffer::operator=(GpuTensorBuffer&& other) noexcept {
    if (this != &other) {
        if (d_data_) {
            cudaFree(d_data_);
        }
        d_data_ = other.d_data_;
        num_elements_ = other.num_elements_;
        other.d_data_ = nullptr;
        other.num_elements_ = 0;
    }
    return *this;
}

void GpuTensorBuffer::copy_from_host_async(const float* host_ptr, cudaStream_t stream) {
    ScopedNvtxRange range("GpuTensorBuffer::H2D_Async");
    cudaError_t err = cudaMemcpyAsync(d_data_, host_ptr, bytes(), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync (H2D) failed: ") + cudaGetErrorString(err));
    }
}

void GpuTensorBuffer::copy_to_host_async(float* host_ptr, cudaStream_t stream) const {
    ScopedNvtxRange range("GpuTensorBuffer::D2H_Async");
    cudaError_t err = cudaMemcpyAsync(host_ptr, d_data_, bytes(), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync (D2H) failed: ") + cudaGetErrorString(err));
    }
}

void GpuTensorBuffer::copy_from_host(const float* host_ptr) {
    copy_from_host_async(host_ptr, nullptr);
    cudaStreamSynchronize(nullptr);
}

void GpuTensorBuffer::copy_to_host(float* host_ptr) const {
    copy_to_host_async(host_ptr, nullptr);
    cudaStreamSynchronize(nullptr);
}
