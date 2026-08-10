"""
ArmSight — Recommendations
Generates Arm64-specific optimization recommendations based on model analysis.
"""
from __future__ import annotations

from typing import Any


def generate_recommendations(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate Arm64-specific optimization recommendations from analysis."""
    recs: list[dict[str, Any]] = []

    precision = analysis.get("precision", "")
    total_params = analysis.get("total_parameters", 0)
    op_counts = analysis.get("operator_counts", {})
    file_size = analysis.get("file_size_bytes", 0)

    # 1. INT8 Quantization (always recommend for FP32 models)
    if "FP32" in precision or "Float32" in precision:
        est_factor = 4.0  # typical FP32->INT8
        est_new_size = file_size / est_factor if file_size else 0
        recs.append({
            "id": "int8_quantization",
            "title": "INT8 Dynamic Quantization",
            "category": "Precision",
            "priority": "HIGH",
            "description": (
                "Quantize FP32 weights to INT8 using onnxruntime dynamic quantization. "
                "Reduces model size by ~4x with minimal accuracy loss for most workloads."
            ),
            "arm_benefit": (
                "Arm Cortex-A and Graviton cores benefit significantly from INT8 — "
                "the dot-product instructions (SDOT/UDOT on Cortex-A78/A710 and Graviton3+) "
                "accelerate INT8 GEMM operations, reducing both memory bandwidth and compute cycles."
            ),
            "estimated_size_reduction": f"~{est_factor:.0f}x (from {_human_bytes_local(file_size)} to ~{_human_bytes_local(est_new_size)})",
            "estimated_latency_improvement": "10-40% on Arm64 (memory-bandwidth bound models benefit most)",
            "actionable": True,
            "applies_to": "All Arm64 CPUs (Cortex-A, Graviton, Neoverse)",
        })
    else:
        recs.append({
            "id": "precision_already_optimized",
            "title": "Precision Already Optimized",
            "category": "Precision",
            "priority": "INFO",
            "description": f"Model is already using {precision}. No quantization needed.",
            "arm_benefit": "N/A",
            "estimated_size_reduction": "N/A",
            "estimated_latency_improvement": "N/A",
            "actionable": False,
            "applies_to": "N/A",
        })

    # 2. NEON SIMD Operator Fusion
    conv_ops = op_counts.get("Conv", 0) + op_counts.get("ConvInteger", 0)
    gemm_ops = op_counts.get("Gemm", 0) + op_counts.get("MatMul", 0)
    if conv_ops > 0 or gemm_ops > 0:
        recs.append({
            "id": "neon_simd_fusion",
            "title": "NEON SIMD Operator Fusion",
            "category": "Compute",
            "priority": "MEDIUM",
            "description": (
                f"Model has {conv_ops} convolution and {gemm_ops} GEMM operations. "
                "ONNX Runtime on Arm64 already fuses Conv+Activation and Gemm+Add patterns, "
                "but manual fusion of consecutive element-wise ops can further reduce kernel launch overhead."
            ),
            "arm_benefit": (
                "Arm NEON SIMD processes 128-bit vectors per instruction — 4x FP32 or 8x INT16 or 16x INT8 per cycle. "
                "Fused operators keep data in NEON registers, avoiding L1 cache round-trips. "
                "Graviton3's V1 cores add 4x128-bit SVE for even wider SIMD throughput."
            ),
            "estimated_size_reduction": "No size change",
            "estimated_latency_improvement": "5-15% (operator-fusion-bound models)",
            "actionable": True,
            "applies_to": "All Arm64 with NEON (universal on Armv8+)",
        })

    # 3. Thread Parallelism for Multi-core
    if total_params > 1_000_000:
        recs.append({
            "id": "thread_parallelism",
            "title": "Multi-core Thread Parallelism (Graviton/Neoverse)",
            "category": "Parallelism",
            "priority": "HIGH",
            "description": (
                f"Model has {total_params:,} parameters — large enough to benefit from "
                "inter-op thread parallelism. Set ort.SessionOptions intra_op_num_threads to "
                "match core count and enable execution_mode=ORT_SEQUENTIAL for small batches "
                "or ORT_PARALLEL for large batches."
            ),
            "arm_benefit": (
                "AWS Graviton3 has 64 vCPUs; Graviton4 has up to 96. Neoverse N2/V2 designs scale "
                "linearly with cores for compute-bound inference. Thread parallelism on Arm64 "
                "is especially effective because each core has private L1/L2 caches, unlike "
                "some x86 designs with shared L2."
            ),
            "estimated_size_reduction": "No size change",
            "estimated_latency_improvement": "2-8x on multi-core (scales with core count)",
            "actionable": True,
            "applies_to": "Multi-core Arm64: Graviton2/3/4, Neoverse N1/N2/V1/V2",
        })

    # 4. Memory Layout Optimization
    if op_counts.get("Conv", 0) > 5 or op_counts.get("MatMul", 0) > 5:
        recs.append({
            "id": "memory_layout",
            "title": "Memory Layout Optimization (NCHW → NHWC)",
            "category": "Memory",
            "priority": "MEDIUM",
            "description": (
                "ONNX Runtime on Arm64 prefers channels-last (NHWC) memory layout for convolutions. "
                "The runtime auto-converts NCHW→NHWC internally, but pre-converting avoids runtime overhead."
            ),
            "arm_benefit": (
                "Arm Mali GPUs and the onnxruntime CPU EP on Arm64 both favor NHWC for convolutions — "
                "it enables contiguous memory access patterns that match NEON SIMD load/store widths. "
                "Graviton's L2 cache line prefetcher works best with sequential NHWC access."
            ),
            "estimated_size_reduction": "No size change",
            "estimated_latency_improvement": "3-10% for CNN-heavy models",
            "actionable": True,
            "applies_to": "CNN models on Arm64 CPU and Mali GPU",
        })

    # 5. Provider selection
    recs.append({
        "id": "arm_execution_provider",
        "title": "Arm Compute Library (ACL) Execution Provider",
        "category": "Runtime",
        "priority": "MEDIUM",
        "description": (
            "ONNX Runtime supports the Arm Compute Library (ACL) execution provider, "
            "which is hand-tuned for Mali GPUs and Arm CPUs with NEON optimizations. "
            "Build onnxruntime from source with --use_acl to enable."
        ),
        "arm_benefit": (
            "ACL provides hand-optimized NEON kernels for convolution, GEMM, and pooling "
            "that outperform the generic CPU EP by 10-30% on Cortex-A78/A715 and Graviton. "
            "For Mali GPUs, ACL enables GPU offload entirely."
        ),
        "estimated_size_reduction": "No size change",
        "estimated_latency_improvement": "10-30% over generic CPU EP",
        "actionable": True,
        "applies_to": "Arm CPUs (ACL-CPU) and Mali GPUs (ACL-GPU)",
    })

    return recs


def _human_bytes_local(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def summarize_for_arm(analysis: dict[str, Any], recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a concise Arm-targeted summary for display."""
    return {
        "model_precision": analysis.get("precision", "unknown"),
        "total_parameters": analysis.get("total_parameters", 0),
        "model_size": analysis.get("file_size_human", "unknown"),
        "operator_count": analysis.get("total_operators", 0),
        "arm64_optimizations_available": len([r for r in recommendations if r.get("actionable")]),
        "high_priority_count": len([r for r in recommendations if r.get("priority") == "HIGH"]),
        "estimated_total_size_reduction": "Up to 4x with INT8 quantization",
        "estimated_total_latency_improvement": "Up to 40% on Arm64 multi-core",
    }
