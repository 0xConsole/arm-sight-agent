"""
ArmSight — MCP (Model Context Protocol) Server Interface
Exposes analysis/optimization tools as MCP-compatible tools that an AI agent
can call autonomously. This is ArmSight's "unfair advantage" — the agent
calls these tools to analyze and optimize models without human intervention.

The MCP tool definitions follow the Model Context Protocol specification:
https://modelcontextprotocol.io/docs/concepts/tools
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

from app.analyzer import analyze_model, benchmark_inference
from app.quantizer import quantize_int8, benchmark_quantized
from app.recommendations import generate_recommendations, summarize_for_arm
from app.deployment import generate_deployment_package


# ---------------------------------------------------------------------------
# MCP Tool Registry
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "analyze_model",
        "description": (
            "Analyze an ONNX model's architecture: operators, layers, precision, "
            "parameter count, input/output shapes. Returns a structured report."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {
                    "type": "string",
                    "description": "Path to the ONNX model file",
                },
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "optimize_model",
        "description": (
            "Apply INT8 dynamic quantization to an ONNX model. Produces a real, "
            "measurably smaller model file (typically ~4x size reduction). Returns "
            "before/after size comparison."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to ONNX model"},
                "output_path": {"type": "string", "description": "Output path (optional)"},
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "benchmark_model",
        "description": (
            "Benchmark model inference latency and throughput on CPU. Returns "
            "mean/p50/p95 latency in ms and ops/sec throughput."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to ONNX model"},
                "runs": {"type": "integer", "description": "Number of benchmark runs", "default": 20},
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "recommend_optimizations",
        "description": (
            "Generate Arm64-specific optimization recommendations based on model "
            "analysis: INT8 quantization, NEON SIMD fusion, thread parallelism, "
            "memory layout, and Arm Compute Library provider."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to ONNX model"},
            },
            "required": ["model_path"],
        },
    },
    {
        "name": "generate_deployment",
        "description": (
            "Generate a complete Arm64-optimized deployment package: Dockerfile "
            "(linux/arm64), FastAPI inference server, benchmark script, and "
            "docker-compose.yml."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to ONNX model"},
                "output_dir": {"type": "string", "description": "Output directory for package"},
                "image_name": {"type": "string", "description": "Docker image name", "default": "armsight-inference"},
            },
            "required": ["model_path", "output_dir"],
        },
    },
    {
        "name": "full_pipeline",
        "description": (
            "Run the full ArmSight pipeline autonomously: analyze → recommend → "
            "quantize → benchmark before/after → generate deployment package. "
            "Returns the complete optimization report with measurable improvements."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Path to ONNX model"},
            },
            "required": ["model_path"],
        },
    },
]


def list_tools() -> dict[str, Any]:
    """Return MCP tool definitions (MCP tools/list method)."""
    return {"tools": MCP_TOOLS}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute an MCP tool by name with the given arguments.
    This is the core dispatch — an AI agent calls this to use ArmSight's tools.
    """
    model_path = arguments.get("model_path", "")

    if name == "analyze_model":
        return {"tool": name, "result": analyze_model(model_path)}

    elif name == "optimize_model":
        output_path = arguments.get("output_path")
        return {"tool": name, "result": quantize_int8(model_path, output_path)}

    elif name == "benchmark_model":
        runs = arguments.get("runs", 20)
        return {"tool": name, "result": benchmark_inference(model_path, runs)}

    elif name == "recommend_optimizations":
        analysis = analyze_model(model_path)
        recs = generate_recommendations(analysis)
        return {"tool": name, "result": {"analysis_summary": summarize_for_arm(analysis, recs), "recommendations": recs}}

    elif name == "generate_deployment":
        output_dir = arguments.get("output_dir", "/tmp/armsight_deploy")
        image_name = arguments.get("image_name", "armsight-inference")
        return {"tool": name, "result": generate_deployment_package(model_path, output_dir, image_name)}

    elif name == "full_pipeline":
        return _full_pipeline(model_path)

    else:
        return {"tool": name, "error": f"Unknown tool: {name}", "available_tools": [t["name"] for t in MCP_TOOLS]}


def _full_pipeline(model_path: str) -> dict[str, Any]:
    """The autonomous full pipeline: analyze → recommend → quantize → benchmark → deploy."""
    result: dict[str, Any] = {"tool": "full_pipeline", "steps": []}

    # Step 1: Analyze
    analysis = analyze_model(model_path)
    result["steps"].append({"step": "analyze", "status": "ok" if analysis.get("onnx_parse_ok") else "error"})

    # Step 2: Recommend
    recs = generate_recommendations(analysis)
    result["steps"].append({"step": "recommend", "status": "ok", "recommendation_count": len(recs)})

    # Step 3: Quantize
    base, ext = os.path.splitext(model_path)
    quantized_path = f"{base}_int8{ext}"
    quant_result = quantize_int8(model_path, quantized_path)
    result["steps"].append({
        "step": "quantize",
        "status": "ok" if quant_result.get("ok") else "error",
        "size_reduction_ratio": quant_result.get("size_reduction_ratio", 0),
    })

    # Step 4: Benchmark before/after
    if quant_result.get("ok") and os.path.exists(quantized_path):
        bench = benchmark_quantized(model_path, quantized_path, runs=20)
        result["steps"].append({"step": "benchmark", "status": "ok" if bench.get("ok") else "error"})
        result["benchmark_comparison"] = bench
    else:
        result["steps"].append({"step": "benchmark", "status": "skipped"})

    # Step 5: Generate deployment package
    deploy_dir = os.path.join(tempfile.gettempdir(), "armsight_deploy")
    deploy = generate_deployment_package(quantized_path if quant_result.get("ok") else model_path, deploy_dir)
    # Read generated files for display (so the UI can show the package contents)
    files_content: dict[str, str] = {}
    if deploy.get("ok"):
        for fname in ["Dockerfile", "server.py", "benchmark.py", "docker-compose.yml"]:
            fpath = os.path.join(deploy_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as fh:
                    files_content[fname] = fh.read()
    deploy["files_content"] = files_content
    result["steps"].append({"step": "generate_deployment", "status": "ok" if deploy.get("ok") else "error"})
    result["deployment_package"] = deploy

    # Summary with measurable improvements
    result["summary"] = {
        "analysis": summarize_for_arm(analysis, recs),
        "recommendations": recs,
        "quantization": quant_result,
    }
    if quant_result.get("ok"):
        result["summary"]["measurable_improvements"] = {
            "size_before_bytes": quant_result.get("original_size_bytes", 0),
            "size_after_bytes": quant_result.get("quantized_size_bytes", 0),
            "size_reduction_pct": quant_result.get("size_reduction_ratio", 0),
            "size_reduction_factor": quant_result.get("size_reduction_factor", 0),
        }
    if "benchmark_comparison" in result and result["benchmark_comparison"].get("ok"):
        bc = result["benchmark_comparison"]
        result["summary"]["measurable_improvements"].update({
            "latency_before_ms": bc["original"].get("latency_ms_mean", 0),
            "latency_after_ms": bc["quantized"].get("latency_ms_mean", 0),
            "speedup_factor": bc.get("speedup_factor", 0),
        })

    return result
