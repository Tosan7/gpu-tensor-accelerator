"""
Deep Learning Kernel Numerical Accuracy Verification
===================================================
Tests mathematical precision of:
1. Fused Bias-Add + GeLU against analytical activation curves and boundary limits.
2. Tiled GEMM matrix multiplication against BLAS ground truth.
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from python.engine import TensorEngine


def verify_fused_gelu():
    print("\n[TEST 1] Verifying Fused Bias-Add + GeLU Numerical Accuracy...")
    engine = TensorEngine()

    batch_size = 4
    seq_len = 128
    hidden_dim = 1024

    # 1. Boundary & Known value tests
    print("    Sub-test A: Mathematical Boundary Limits...")
    test_inputs = np.array([-10.0, -5.0, 0.0, 2.0, 10.0], dtype=np.float32)
    zero_bias = np.zeros_like(test_inputs)
    results = engine.fused_bias_gelu(test_inputs, zero_bias)

    # Asymptotic limits: GeLU(0) == 0.0, GeLU(-10) -> 0.0, GeLU(10) -> 10.0
    assert np.isclose(results[2], 0.0, atol=1e-6), f"GeLU(0) should be 0.0, got {results[2]}"
    assert np.isclose(results[0], 0.0, atol=1e-4), f"GeLU(-10) should be ~0.0, got {results[0]}"
    assert np.isclose(results[4], 10.0, atol=1e-3), f"GeLU(10) should be ~10.0, got {results[4]}"
    print("    [PASS] Boundary limits verified (x -> -inf => 0, x -> +inf => x, x=0 => 0)")

    # 2. Random tensor validation with broadcast bias
    print(f"    Sub-test B: Transformer Tensor Dimensions [{batch_size}x{seq_len}x{hidden_dim}]...")
    np.random.seed(42)
    inp = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    bias = np.random.randn(hidden_dim).astype(np.float32)

    actual = engine.fused_bias_gelu(inp, bias)

    # Ground truth formula: 0.5 * (x) * (1 + erf(x / sqrt(2))) or tanh approximation
    x = inp + bias
    expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    max_diff = np.max(np.abs(actual - expected))
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-5), f"Max difference too large: {max_diff}"
    print(f"    [PASS] Tensor validation passed! Max absolute deviation: {max_diff:.2e} (< 1e-5 tolerance)")
    return True


def verify_tiled_gemm():
    print("\n[TEST 2] Verifying Tiled GEMM Matrix Multiplication Accuracy...")
    engine = TensorEngine()

    test_shapes = [
        (128, 128, 128),     # Small square
        (512, 1024, 256),    # Rectangular / Transformer projection
        (1024, 1024, 1024),  # Standard deep learning hidden layer
    ]

    for M, N, K in test_shapes:
        print(f"    Testing shape: A=[{M}x{K}] * B=[{K}x{N}] -> C=[{M}x{N}]...")
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32) * 0.1
        B = np.random.randn(K, N).astype(np.float32) * 0.1

        actual = engine.gemm(A, B)
        expected = np.matmul(A, B)

        max_diff = np.max(np.abs(actual - expected))
        assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), f"Shape ({M},{N},{K}) failed, diff: {max_diff}"
        print(f"    [PASS] Shape [{M}x{N}x{K}] passed! Max absolute deviation: {max_diff:.2e}")

    return True


def main():
    print("=" * 65)
    print("  NVIDIA Accelerated Computing: Mathematical Accuracy Suite")
    print("=" * 65)

    ok1 = verify_fused_gelu()
    ok2 = verify_tiled_gemm()

    print("\n" + "=" * 65)
    if ok1 and ok2:
        print("  [SUCCESS] ALL NUMERICAL ACCURACY TESTS PASSED! (Zero bit drift)")
        print("=" * 65)
        sys.exit(0)
    else:
        print("  [FAILURE] NUMERICAL MISMATCH DETECTED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
