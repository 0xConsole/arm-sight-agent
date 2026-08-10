"""
ArmSight — Quantizer
Applies REAL INT8 dynamic quantization to ONNX models using onnxruntime.quantization.
Produces REAL, measurable size reduction (typically ~4x for FP32→INT8).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.quantize import quantize_dynamic


def quantize_int8(input_model_path: str, output_model_path: str | None = None) -> dict[str, Any]:
    """
    Apply dynamic INT8 quantization to an ONNX model.
    This is REAL quantization — weights are quantized to INT8, producing
    a genuinely smaller model file with actual size reduction.
    """
    if output_model_path is None:
        base, ext = os.path.splitext(input_model_path)
        output_model_path = f"{base}_int8{ext}"

    result: dict[str, Any] = {
        "input_model": input_model_path,
        "output_model": output_model_path,
    }

    original_size = os.path.getsize(input_model_path)
    result["original_size_bytes"] = original_size

    try:
        # Dynamic quantization quantizes weights from FP32 to INT8/UINT8.
        # weight_type=QuantType.QInt8 produces INT8 quantized weights.
        from onnxruntime.quantization.quantize import quantize_dynamic
        from onnxruntime.quantization.calibrate import CalibrationMethod  # noqa: F401

        quantize_dynamic(
            input_model_path,
            output_model_path,
            weight_type=onnxruntime.quantization.QuantType.QInt8,
        )

        if not os.path.exists(output_model_path):
            raise RuntimeError("Quantization did not produce output file")

        quantized_size = os.path.getsize(output_model_path)
        result["quantized_size_bytes"] = quantized_size
        result["size_reduction_bytes"] = original_size - quantized_size
        result["size_reduction_ratio"] = round(
            (1 - quantized_size / original_size) * 100, 2
        ) if original_size > 0 else 0
        result["size_reduction_factor"] = round(
            original_size / quantized_size, 2
        ) if quantized_size > 0 else 0
        result["ok"] = True
        result["error"] = None

    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        # Fallback: try the simplest form
        try:
            quantize_dynamic(input_model_path, output_model_path)
            quantized_size = os.path.getsize(output_model_path)
            result["quantized_size_bytes"] = quantized_size
            result["size_reduction_bytes"] = original_size - quantized_size
            result["size_reduction_ratio"] = round(
                (1 - quantized_size / original_size) * 100, 2
            ) if original_size > 0 else 0
            result["size_reduction_factor"] = round(
                original_size / quantized_size, 2
            ) if quantized_size > 0 else 0
            result["ok"] = True
            result["error"] = None
            result["note"] = "Quantized with default weight type (fallback)"
        except Exception as exc2:
            result["ok"] = False
            result["error"] = f"Primary: {exc}; Fallback: {exc2}"

    return result


def benchmark_quantized(
    original_path: str,
    quantized_path: str,
    runs: int = 20,
) -> dict[str, Any]:
    """
    Benchmark original vs quantized model and return a comparison.
    Uses real inference timing on CPU.
    """
    from app.analyzer import benchmark_inference

    result: dict[str, Any] = {
        "original": benchmark_inference(original_path, runs),
        "quantized": benchmark_inference(quantized_path, runs),
    }

    orig = result["original"]
    quant = result["quantized"]

    if orig.get("ok") and quant.get("ok"):
        result["speedup_factor"] = round(
            orig["latency_ms_mean"] / quant["latency_ms_mean"], 4
        ) if quant["latency_ms_mean"] > 0 else 0
        result["latency_improvement_pct"] = round(
            (1 - quant["latency_ms_mean"] / orig["latency_ms_mean"]) * 100, 2
        ) if orig["latency_ms_mean"] > 0 else 0
        result["throughput_improvement_pct"] = round(
            (quant["throughput_ops_per_sec"] / orig["throughput_ops_per_sec"] - 1) * 100, 2
        ) if orig["throughput_ops_per_sec"] > 0 else 0
        result["ok"] = True
        result["error"] = None
    else:
        result["ok"] = False
        result["error"] = f"original_ok={orig.get('ok')}, quantized_ok={quant.get('ok')}"

    return result
