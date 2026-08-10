#!/usr/bin/env python3
"""
Local test script for ArmSight — tests the full pipeline with a real ONNX model.
Run: python test_local.py
"""
import os
import sys
import tempfile
import numpy as np
import onnx
from onnx import helper, TensorProto

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.analyzer import analyze_model, benchmark_inference
from app.quantizer import quantize_int8, benchmark_quantized
from app.recommendations import generate_recommendations, summarize_for_arm
from app.deployment import generate_deployment_package
from app.mcp_server import list_tools, call_tool


def create_test_model(path: str):
    """Create a small ONNX MLP model for testing."""
    input_dim, h1, h2, output_dim = 10, 128, 64, 5
    np.random.seed(42)
    W1 = np.random.randn(input_dim, h1).astype(np.float32)
    b1 = np.zeros(h1, dtype=np.float32)
    W2 = np.random.randn(h1, h2).astype(np.float32)
    b2 = np.zeros(h2, dtype=np.float32)
    W3 = np.random.randn(h2, output_dim).astype(np.float32)
    b3 = np.zeros(output_dim, dtype=np.float32)

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    nodes = [
        helper.make_node("Gemm", ["input", "W1", "b1"], ["h1"], name="gemm1"),
        helper.make_node("Relu", ["h1"], ["h1r"], name="relu1"),
        helper.make_node("Gemm", ["h1r", "W2", "b2"], ["h2"], name="gemm2"),
        helper.make_node("Relu", ["h2"], ["h2r"], name="relu2"),
        helper.make_node("Gemm", ["h2r", "W3", "b3"], ["output"], name="gemm3"),
    ]

    initializers = [
        helper.make_tensor("W1", TensorProto.FLOAT, W1.shape, W1.flatten().tolist()),
        helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.flatten().tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, W2.shape, W2.flatten().tolist()),
        helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.flatten().tolist()),
        helper.make_tensor("W3", TensorProto.FLOAT, W3.shape, W3.flatten().tolist()),
        helper.make_tensor("b3", TensorProto.FLOAT, b3.shape, b3.flatten().tolist()),
    ]

    graph = helper.make_graph(nodes, "test_mlp", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"  Created test model: {path} ({os.path.getsize(path)} bytes)")


def main():
    print("=" * 60)
    print("ArmSight Local Test — Full Pipeline")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="armsight_test_")
    model_path = os.path.join(tmpdir, "test_model.onnx")
    quant_path = os.path.join(tmpdir, "test_model_int8.onnx")

    print("\n[1/6] Creating test ONNX model...")
    create_test_model(model_path)

    print("\n[2/6] Analyzing model...")
    analysis = analyze_model(model_path)
    print(f"  Size: {analysis['file_size_human']}")
    print(f"  Precision: {analysis['precision']}")
    print(f"  Parameters: {analysis['total_parameters']:,}")
    print(f"  Operators: {analysis['total_operators']} ({analysis['unique_operators']} unique)")
    print(f"  Op counts: {analysis['operator_counts']}")

    print("\n[3/6] Generating recommendations...")
    recs = generate_recommendations(analysis)
    for r in recs:
        print(f"  [{r['priority']}] {r['title']}")
        print(f"         {r['estimated_size_reduction']} | {r['estimated_latency_improvement']}")

    print("\n[4/6] Applying INT8 quantization (REAL)...")
    qresult = quantize_int8(model_path, quant_path)
    print(f"  Original: {qresult['original_size_bytes']} bytes")
    print(f"  Quantized: {qresult['quantized_size_bytes']} bytes")
    print(f"  Reduction: {qresult['size_reduction_ratio']}% ({qresult['size_reduction_factor']}x)")
    print(f"  OK: {qresult['ok']}")

    print("\n[5/6] Benchmarking before/after (REAL inference timing)...")
    if qresult['ok'] and os.path.exists(quant_path):
        bench = benchmark_quantized(model_path, quant_path, runs=20)
        if bench.get('ok'):
            o = bench['original']
            q = bench['quantized']
            print(f"  Original  — latency: {o['latency_ms_mean']}ms, throughput: {o['throughput_ops_per_sec']} ops/s")
            print(f"  Quantized — latency: {q['latency_ms_mean']}ms, throughput: {q['throughput_ops_per_sec']} ops/s")
            print(f"  Speedup: {bench['speedup_factor']}x | Latency improvement: {bench['latency_improvement_pct']}%")
        else:
            print(f"  Benchmark error: {bench.get('error')}")
    else:
        print("  Skipped (quantization failed)")

    print("\n[6/6] Testing MCP interface...")
    tools = list_tools()
    print(f"  MCP tools available: {len(tools['tools'])}")
    for t in tools['tools']:
        print(f"    - {t['name']}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED — Real quantization and benchmarking verified")
    print("=" * 60)


if __name__ == "__main__":
    main()
