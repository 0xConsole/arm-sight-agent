"""
ArmSight — OnnxModelAnalyzer
Analyzes ONNX model architecture: operators, layers, precision, parameter count.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def _human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def analyze_model(model_path: str) -> dict[str, Any]:
    """Analyze an ONNX model and return a structured report."""
    report: dict[str, Any] = {"model_path": model_path}

    # --- File size ---
    file_size = os.path.getsize(model_path)
    report["file_size_bytes"] = file_size
    report["file_size_human"] = _human_bytes(file_size)

    # --- onnx graph analysis ---
    try:
        model = onnx.load(model_path)
        report["ir_version"] = model.ir_version
        report["producer_name"] = model.producer_name
        report["opset"] = [o.version for o in model.opset_import]

        graph = model.graph
        report["input_names"] = [i.name for i in graph.input]
        report["output_names"] = [o.name for o in graph.output]

        # Count operators
        op_counts: dict[str, int] = {}
        total_params = 0
        tensor_sizes: list[dict[str, Any]] = []

        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        for init in graph.initializer:
            dims = list(init.dims)
            param_count = int(np.prod(dims)) if dims else 0
            total_params += param_count
            dtype_map = {
                1: "FLOAT", 2: "UINT8", 3: "INT8", 6: "INT32",
                7: "INT64", 11: "DOUBLE", 16: "BFLOAT16",
            }
            tensor_sizes.append({
                "name": init.name,
                "dims": dims,
                "dtype": dtype_map.get(init.data_type, f"UNKNOWN({init.data_type})"),
                "param_count": param_count,
            })

        report["operator_counts"] = dict(sorted(op_counts.items(), key=lambda x: -x[1]))
        report["total_operators"] = sum(op_counts.values())
        report["unique_operators"] = len(op_counts)
        report["total_parameters"] = total_params
        report["initializer_count"] = len(graph.initializer)
        report["top_tensors_by_size"] = sorted(
            tensor_sizes, key=lambda t: -t["param_count"]
        )[:10]

        # Precision analysis
        dtype_names = {t["dtype"] for t in tensor_sizes}
        if dtype_names <= {"FLOAT", "DOUBLE"}:
            report["precision"] = "FP32 (Float32)"
        elif dtype_names <= {"FLOAT", "DOUBLE", "BFLOAT16"}:
            report["precision"] = "Mixed FP32/BF16"
        else:
            report["precision"] = f"Mixed ({', '.join(sorted(dtype_names))})"

        report["onnx_parse_ok"] = True
        report["onnx_error"] = None
    except Exception as exc:  # pragma: no cover
        report["onnx_parse_ok"] = False
        report["onnx_error"] = str(exc)

    # --- onnxruntime session for runtime info ---
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3  # suppress warnings
        session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        report["providers"] = ort.get_available_providers()
        report["inputs"] = [
            {
                "name": i.name,
                "shape": [d if isinstance(d, int) else "dynamic" for d in i.shape],
                "type": str(i.type),
            }
            for i in session.get_inputs()
        ]
        report["outputs"] = [
            {
                "name": o.name,
                "shape": [d if isinstance(d, int) else "dynamic" for d in o.shape],
                "type": str(o.type),
            }
            for o in session.get_outputs()
        ]
        report["runtime_ok"] = True
        report["runtime_error"] = None
    except Exception as exc:
        report["runtime_ok"] = False
        report["runtime_error"] = str(exc)

    return report


def benchmark_inference(model_path: str, runs: int = 20) -> dict[str, Any]:
    """Run a quick latency/throughput benchmark on CPU."""
    import time

    result: dict[str, Any] = {}
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])

        inputs_info = session.get_inputs()
        feeds = {}
        for inp in inputs_info:
            shape = []
            for d in inp.shape:
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                else:
                    shape.append(1)
            dtype = np.float32
            if "int64" in str(inp.type).lower():
                dtype = np.int64
            elif "int32" in str(inp.type).lower():
                dtype = np.int32
            feeds[inp.name] = np.random.randn(*shape).astype(dtype)

        # Warmup
        for _ in range(3):
            session.run(None, feeds)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            session.run(None, feeds)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times_np = np.array(times)
        result["latency_ms_mean"] = round(float(np.mean(times_np)), 4)
        result["latency_ms_p50"] = round(float(np.percentile(times_np, 50)), 4)
        result["latency_ms_p95"] = round(float(np.percentile(times_np, 95)), 4)
        result["latency_ms_min"] = round(float(np.min(times_np)), 4)
        result["latency_ms_max"] = round(float(np.max(times_np)), 4)
        result["throughput_ops_per_sec"] = round(1000.0 / result["latency_ms_mean"], 2) if result["latency_ms_mean"] > 0 else 0
        result["runs"] = runs
        result["ok"] = True
        result["error"] = None
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
    return result
