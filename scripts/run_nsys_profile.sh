#!/usr/bin/env bash
# ==============================================================================
# NVIDIA Nsight Systems (nsys) Timeline Profiling Script
# ==============================================================================
# Captures kernel execution timelines, memory transfers (H2D, D2H), and NVTX ranges.
#
# Generates:
#   - profile_report.nsys-rep (Open with NVIDIA Nsight Systems GUI)
#   - profile_stats.txt (Command-line summary of top GPU kernels)
# ==============================================================================

set -e

echo "=== Launching NVIDIA Nsight Systems Profiler ==="

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: 'nsys' command not found in PATH."
    echo "Ensure NVIDIA CUDA Toolkit and Nsight Systems are installed."
    exit 1
fi

OUTPUT_PREFIX="profile_report"

# Profile C++ benchmark executable with CUDA, NVTX, and OS runtime tracing
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="${OUTPUT_PREFIX}" \
    --force-overwrite=true \
    --stats=true \
    ./build/cuda_benchmark

echo "=== Profiling Complete ==="
echo "Timeline Report: ${OUTPUT_PREFIX}.nsys-rep"
