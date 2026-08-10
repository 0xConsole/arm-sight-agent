# ArmSight

> **Autonomous AI agent for Arm64 ML model inference optimization.**
> Analyzes ONNX models, applies REAL INT8 quantization, and generates Arm64-optimized deployment packages — all through an MCP-compatible tool interface that an AI agent can call autonomously.

Built for the **[Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)** (Cloud AI track).

---

## 🎯 The Problem

Deploying ML models on Arm64 (AWS Graviton, Cortex-A, Neoverse) requires platform-specific knowledge: which quantization scheme to use, how to tune thread parallelism for multi-core, when to leverage NEON SIMD, and how to package everything into an Arm64-optimized container. Most developers ship unoptimized FP32 models and leave significant performance on the table.

## 💡 The Solution

**ArmSight** is an autonomous AI agent that:
1. **Analyzes** any ONNX model — operators, layers, precision, parameter count, input/output shapes
2. **Recommends** Arm64-specific optimizations — INT8 quantization, NEON SIMD fusion, thread parallelism, memory layout, ACL provider
3. **Applies** real INT8 dynamic quantization using `onnxruntime.quantization` — producing **measurably smaller** models (typically ~4x size reduction)
4. **Benchmarks** before/after — real inference latency, throughput, and speedup measurements
5. **Generates** a complete Arm64-optimized deployment package — Dockerfile (`linux/arm64`), FastAPI inference server, benchmark script

## 🏆 Unique Angle

> *Unlike generic model optimizers, ArmSight exposes its capabilities as **MCP (Model Context Protocol) tools** that an AI agent can call autonomously — `analyze_model`, `optimize_model`, `benchmark_model`, `recommend_optimizations`, `generate_deployment`, `full_pipeline`. This makes ArmSight not just a tool, but an **agent-native** optimization platform.*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Web UI (HTML/CSS/JS)                  │
│  Upload ONNX → Analyze → Recommend → Quantize → Benchmark    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend (Python)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Analyzer │  │Quantizer │  │  Recomm. │  │  Deployment │  │
│  │ (onnx)   │  │(onnxrt)  │  │  Engine  │  │  Generator  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │
│       └──────────┬───┴─────────────┴───────────────┘         │
│                  ▼                                           │
│         ┌──────────────────┐                                 │
│         │   MCP Server     │ ← AI agent calls these tools     │
│         │  (tool registry) │   autonomously via MCP protocol  │
│         └──────────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Vercel (Serverless)                       │
│         FastAPI on Python runtime — free tier               │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- An ONNX model file (or use the built-in example model generator)

### Setup (< 5 commands)

```bash
# 1. Clone
git clone https://github.com/0xConsole/arm-sight-agent.git
cd arm-sight-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
uvicorn app.main:app --reload --port 8000

# 4. Open the UI
open http://localhost:8000
```

### Use via API / MCP

```bash
# List MCP tools (what an AI agent sees)
curl http://localhost:8000/mcp/tools | python -m json.tool

# Call the full pipeline autonomously (analyze → quantize → benchmark → deploy)
curl -X POST http://localhost:8000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"name": "analyze_model", "arguments": {"model_path": "examples/example_model.onnx"}}'
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python + FastAPI |
| Model analysis | `onnx` + `onnxruntime` |
| Quantization | `onnxruntime.quantization.quantize_dynamic` (REAL INT8) |
| Agent interface | MCP (Model Context Protocol) tool pattern |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Deployment | Vercel serverless (Python runtime) |
| Target platform | linux/arm64 (AWS Graviton, Cortex-A, Neoverse) |

## ✅ What's Real vs. Mocked

| Feature | Status | Notes |
|---|---|---|
| ONNX model analysis | ✅ **REAL** | Uses `onnx` + `onnxruntime` to parse graph, count operators/parameters |
| INT8 quantization | ✅ **REAL** | `onnxruntime.quantization.quantize_dynamic` — produces genuinely smaller ONNX files |
| Size measurement | ✅ **REAL** | Byte-level before/after file size comparison |
| Inference benchmarking | ✅ **REAL** | Actual `session.run()` timing on CPU (mean/p50/p95 latency, throughput) |
| Arm64 recommendations | ✅ **REAL** | Based on actual model architecture (operators, precision, param count) |
| Deployment package | ✅ **REAL** | Generates working Dockerfile targeting `linux/arm64` + FastAPI server + benchmark script |
| MCP tool interface | ✅ **REAL** | Tools are callable via `POST /mcp/call` — any MCP client can invoke them |

**Nothing is mocked.** Every measurement comes from real ONNX runtime operations.

## 📊 Measurable Improvements (Example)

For a typical FP32 ONNX model:

| Metric | Before (FP32) | After (INT8) | Improvement |
|---|---|---|---|
| Model size | ~4.2 MB | ~1.1 MB | **4.0x reduction** |
| Inference latency | ~2.5 ms | ~1.8 ms | **~28% faster** |
| Throughput | ~400 ops/s | ~550 ops/s | **~37% higher** |

*Actual numbers vary by model. The quantization and benchmarking are real — run it on your model to see your results.*

## 🐳 Generated Deployment Package

The `generate_deployment` tool produces:

```
deploy_package/
├── Dockerfile          # linux/arm64 target, ONNX Runtime with NEON
├── server.py           # FastAPI inference server (optimized session options)
├── model.onnx          # Your (optionally quantized) model
├── benchmark.py        # Latency/throughput benchmark script
├── docker-compose.yml  # One-command deployment
└── README.md           # Usage instructions
```

```bash
# Build and run on Arm64
docker buildx build --platform linux/arm64 -t armsight-inference .
docker run --rm -p 8000:8000 armsight-inference
python benchmark.py http://localhost:8000
```

## 🔌 MCP Tool Reference

ArmSight exposes 6 tools via the MCP interface:

| Tool | Description |
|---|---|
| `analyze_model` | Analyze ONNX architecture: operators, precision, params |
| `optimize_model` | Apply INT8 dynamic quantization (real size reduction) |
| `benchmark_model` | Measure inference latency and throughput |
| `recommend_optimizations` | Generate Arm64-specific recommendations |
| `generate_deployment` | Create Arm64 Docker + FastAPI deployment package |
| `full_pipeline` | Run all of the above autonomously |

## 📁 Project Structure

```
arm-sight-agent/
├── api/
│   └── index.py          # Vercel serverless entry point
├── app/
│   ├── main.py           # FastAPI app + routes
│   ├── analyzer.py       # ONNX model analysis
│   ├── quantizer.py      # INT8 quantization (REAL)
│   ├── recommendations.py # Arm64 optimization recommendations
│   ├── deployment.py     # Deployment package generator
│   └── mcp_server.py     # MCP tool registry + dispatch
├── static/
│   └── index.html        # Web UI
├── requirements.txt
├── vercel.json
└── README.md
```

## 📜 License

Apache License 2.0 — see [LICENSE](LICENSE).

## 🔗 Links

- **Live Demo:** [Deployed on Vercel](https://arm-sight-agent.vercel.app)
- **GitHub:** [github.com/0xConsole/arm-sight-agent](https://github.com/0xConsole/arm-sight-agent)
- **Challenge:** [Arm Create: AI Optimization Challenge](https://arm-ai-optimization-challenge.devpost.com/)
