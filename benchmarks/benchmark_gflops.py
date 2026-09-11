"""
GFLOPS & Memory Bandwidth Performance Benchmark Harness
======================================================
Measures:
1. Arithmetic Intensity & GFLOPS for GEMM (2 * M * N * K / time).
2. Effective Memory Bandwidth (GB/s) and Fused Speedup for Activation Operators.
3. Saves a machine-readable 'gflops_benchmark_report.json' artifact.
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python.engine import TensorEngine


def benchmark_fused_speedup():
    print("\n[BENCHMARK 1] Fused Bias-Add + GeLU vs. Un-fused Memory-Bound Baseline")
    print("-" * 65)

    engine = TensorEngine()
    batch_size = 16
    seq_len = 512
    hidden_dim = 4096  # LLaMA / Mistral layer dimension
    total_elements = batch_size * seq_len * hidden_dim
    total_mb = (total_elements * 4) / (1024 * 1024)

    print(f"Tensor Shape: [{batch_size} x {seq_len} x {hidden_dim}] ({total_mb:.1f} MB)")

    np.random.seed(42)
    inp = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    bias = np.random.randn(hidden_dim).astype(np.float32)

    # Warm-up
    for _ in range(3):
        _ = engine.fused_bias_gelu(inp, bias)
        _ = engine.unfused_bias_gelu_baseline(inp, bias)

    ITERS = 20

    # 1. Measure Unfused Baseline
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = engine.unfused_bias_gelu_baseline(inp, bias)
    unfused_time_ms = ((time.perf_counter() - t0) / ITERS) * 1000

    # 2. Measure Fused Operator
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = engine.fused_bias_gelu(inp, bias)
    fused_time_ms = ((time.perf_counter() - t0) / ITERS) * 1000

    speedup = unfused_time_ms / fused_time_ms

    # Memory moved: 1 read input + 1 write output = 2 * total_elements * 4 bytes
    bytes_moved = 2.0 * total_elements * 4.0
    effective_bandwidth_gb_s = (bytes_moved / (fused_time_ms / 1000.0)) / 1e9

    print(f"  Un-fused Baseline Latency: {unfused_time_ms:.2f} ms (Multiple DRAM roundtrips)")
    print(f"  Fused Operator Latency:    {fused_time_ms:.2f} ms (Single pass)")
    print(f"  [RESULT] Kernel Fusion Speedup:   {speedup:.2f}x faster")
    print(f"  [RESULT] Effective Bandwidth:     {effective_bandwidth_gb_s:.2f} GB/s")

    return {
        "tensor_shape": [batch_size, seq_len, hidden_dim],
        "tensor_mb": round(total_mb, 2),
        "unfused_latency_ms": round(unfused_time_ms, 2),
        "fused_latency_ms": round(fused_time_ms, 2),
        "fusion_speedup": round(speedup, 2),
        "effective_bandwidth_gb_s": round(effective_bandwidth_gb_s, 2)
    }


def benchmark_gemm_gflops():
    print("\n[BENCHMARK 2] GEMM Compute Throughput (GFLOPS)")
    print("-" * 65)

    engine = TensorEngine()
    test_dims = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    results = []

    for M, N, K in test_dims:
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32) * 0.1
        B = np.random.randn(K, N).astype(np.float32) * 0.1

        # Warm-up
        _ = engine.gemm(A, B)

        ITERS = 10
        t0 = time.perf_counter()
        for _ in range(ITERS):
            _ = engine.gemm(A, B)
        avg_time_ms = ((time.perf_counter() - t0) / ITERS) * 1000

        # FLOPs = 2 * M * N * K (multiply + add for each element)
        flops = 2.0 * M * N * K
        gflops = (flops / (avg_time_ms / 1000.0)) / 1e9

        print(f"  Matrix [{M}x{K}] * [{K}x{N}] -> Time: {avg_time_ms:6.2f} ms | Compute: {gflops:7.2f} GFLOPS")

        results.append({
            "dimension": f"{M}x{N}x{K}",
            "latency_ms": round(avg_time_ms, 2),
            "throughput_gflops": round(gflops, 2)
        })

    return results


def main():
    print("=" * 65)
    print("  NVIDIA GPU Tensor Accelerator: Performance Benchmark Suite")
    print("=" * 65)

    fusion_metrics = benchmark_fused_speedup()
    gemm_metrics = benchmark_gemm_gflops()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fused_operator_benchmark": fusion_metrics,
        "gemm_throughput_benchmark": gemm_metrics
    }

    report_path = "gflops_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  [SUCCESS] BENCHMARK COMPLETE: Saved metrics to '{report_path}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
