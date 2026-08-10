"""
ArmSight — Main FastAPI Application
Autonomous AI agent for Arm64 ML model inference optimization.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.analyzer import analyze_model, benchmark_inference
from app.quantizer import quantize_int8, benchmark_quantized
from app.recommendations import generate_recommendations, summarize_for_arm
from app.deployment import generate_deployment_package
from app.mcp_server import list_tools, call_tool, MCP_TOOLS

app = FastAPI(
    title="ArmSight",
    description="Autonomous AI agent for Arm64 ML model inference optimization. Analyzes ONNX models, applies INT8 quantization, and generates Arm64-optimized deployment packages.",
    version="1.0.0",
)

# Serve static files (UI)
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Temp storage for uploaded models
UPLOAD_DIR = Path(tempfile.gettempdir()) / "armsight_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main UI."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ArmSight", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    """Analyze an uploaded ONNX model."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    model_path = await _save_upload(file)
    try:
        result = analyze_model(str(model_path))
        return result
    finally:
        _cleanup(model_path)


@app.post("/api/recommend")
async def api_recommend(file: UploadFile = File(...)):
    """Generate Arm64 optimization recommendations for a model."""
    model_path = await _save_upload(file)
    try:
        analysis = analyze_model(str(model_path))
        recs = generate_recommendations(analysis)
        return {
            "analysis_summary": summarize_for_arm(analysis, recs),
            "recommendations": recs,
        }
    finally:
        _cleanup(model_path)


@app.post("/api/quantize")
async def api_quantize(file: UploadFile = File(...)):
    """Apply INT8 quantization and return before/after benchmarks."""
    model_path = await _save_upload(file)
    try:
        # Quantize
        base = model_path.stem
        quantized_path = UPLOAD_DIR / f"{base}_int8.onnx"
        quant_result = quantize_int8(str(model_path), str(quantized_path))

        response: dict[str, Any] = {"quantization": quant_result}

        # Benchmark if quantization succeeded
        if quant_result.get("ok") and quantized_path.exists():
            bench = benchmark_quantized(str(model_path), str(quantized_path), runs=20)
            response["benchmark_comparison"] = bench

        return response
    finally:
        _cleanup(model_path)
        if quantized_path.exists():
            _cleanup(quantized_path)


@app.post("/api/full")
async def api_full(file: UploadFile = File(...)):
    """Run the full pipeline: analyze → recommend → quantize → benchmark → deploy."""
    model_path = await _save_upload(file)
    try:
        result = call_tool("full_pipeline", {"model_path": str(model_path)})
        return result
    finally:
        _cleanup(model_path)
        # Clean quantized model too
        base = model_path.stem
        quant = UPLOAD_DIR / f"{base}_int8.onnx"
        if quant.exists():
            _cleanup(quant)


@app.post("/api/benchmark")
async def api_benchmark(file: UploadFile = File(...), runs: int = Query(default=20, ge=1, le=100)):
    """Benchmark model inference latency."""
    model_path = await _save_upload(file)
    try:
        return benchmark_inference(str(model_path), runs)
    finally:
        _cleanup(model_path)


@app.post("/api/deployment")
async def api_deployment(file: UploadFile = File(...), image_name: str = "armsight-inference"):
    """Generate a deployment package for the uploaded model."""
    model_path = await _save_upload(file)
    try:
        deploy_dir = UPLOAD_DIR / f"{model_path.stem}_deploy"
        result = generate_deployment_package(str(model_path), str(deploy_dir), image_name)
        # Read key files for display
        files_content = {}
        for fname in ["Dockerfile", "server.py", "benchmark.py", "docker-compose.yml"]:
            fpath = deploy_dir / fname
            if fpath.exists():
                files_content[fname] = fpath.read_text()
        result["files_content"] = files_content
        return result
    finally:
        _cleanup(model_path)


@app.get("/api/example-model")
async def api_example_model():
    """Generate a small example ONNX model for users to try without their own file."""
    import io
    from fastapi.responses import StreamingResponse
    model_bytes = _create_example_model()
    return StreamingResponse(
        io.BytesIO(model_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=example_model.onnx"},
    )


def _create_example_model() -> bytes:
    """Create a small ONNX model (MLP) for testing/demo purposes."""
    import numpy as np
    import onnx
    from onnx import helper, TensorProto

    # 4-layer MLP with biases: input(256) -> 1024 -> 512 -> 256 -> 10
    # ~920K params — large enough to show REAL quantization speedup (not just size).
    input_dim, hidden1, hidden2, hidden3, output_dim = 256, 1024, 512, 256, 10

    # Weights and biases (FP32) — large enough for quantization to show real speedup
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden1).astype(np.float32)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = np.random.randn(hidden1, hidden2).astype(np.float32)
    b2 = np.zeros(hidden2, dtype=np.float32)
    W3 = np.random.randn(hidden2, hidden3).astype(np.float32)
    b3 = np.zeros(hidden3, dtype=np.float32)
    W4 = np.random.randn(hidden3, output_dim).astype(np.float32)
    b4 = np.zeros(output_dim, dtype=np.float32)

    # Build graph
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    nodes = [
        helper.make_node("Gemm", ["input", "W1", "b1"], ["h1"], name="gemm1"),
        helper.make_node("Relu", ["h1"], ["h1r"], name="relu1"),
        helper.make_node("Gemm", ["h1r", "W2", "b2"], ["h2"], name="gemm2"),
        helper.make_node("Relu", ["h2"], ["h2r"], name="relu2"),
        helper.make_node("Gemm", ["h2r", "W3", "b3"], ["h3"], name="gemm3"),
        helper.make_node("Relu", ["h3"], ["h3r"], name="relu3"),
        helper.make_node("Gemm", ["h3r", "W4", "b4"], ["output"], name="gemm4"),
    ]

    initializers = [
        helper.make_tensor("W1", TensorProto.FLOAT, W1.shape, W1.flatten().tolist()),
        helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.flatten().tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, W2.shape, W2.flatten().tolist()),
        helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.flatten().tolist()),
        helper.make_tensor("W3", TensorProto.FLOAT, W3.shape, W3.flatten().tolist()),
        helper.make_tensor("b3", TensorProto.FLOAT, b3.shape, b3.flatten().tolist()),
        helper.make_tensor("W4", TensorProto.FLOAT, W4.shape, W4.flatten().tolist()),
        helper.make_tensor("b4", TensorProto.FLOAT, b4.shape, b4.flatten().tolist()),
    ]

    graph = helper.make_graph(nodes, "armsight_example_mlp", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.checker.check_model(model)
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# Routes — MCP Interface (the "unfair advantage")
# ---------------------------------------------------------------------------

@app.get("/mcp/tools")
async def mcp_list_tools():
    """List available MCP tools (MCP tools/list)."""
    return list_tools()


@app.post("/mcp/call")
async def mcp_call_tool(payload: dict[str, Any]):
    """Call an MCP tool (MCP tools/call). Body: {"name": "...", "arguments": {...}}"""
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'name' field")
    return call_tool(name, arguments)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_upload(file: UploadFile) -> Path:
    """Save uploaded file to temp dir and return path."""
    if not file.filename or not file.filename.endswith(".onnx"):
        raise HTTPException(status_code=400, detail="File must be an ONNX model (.onnx extension)")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(content)
    return dest


def _cleanup(path: Path | str):
    """Remove temp file silently."""
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass
