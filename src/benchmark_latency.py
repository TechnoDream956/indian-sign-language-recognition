#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline & Component Latency Benchmarking Suite.
===============================================
Measures real-world inference latencies and throughput across all components
on local hardware (Apple Silicon / CPU):
- Section 1: Component & Inference-Only Benchmarks
  - Single-Hand Landmark MLP (63D)
  - Dual-Hand 144D Feature Extraction & MLP
  - RGB CNN Classifier (128x128x3)
  - Multi-Modal Fusion Engine Arbitration
  - AlphabetEngine Landmark Arbitration (`process_dual_landmarks`)
- Section 2: True Full End-to-End Pipeline (`InferenceEngine.process_frame`)
  - Sensor Ingestion -> Orientation/Unmirroring -> MediaPipe Detection ->
    Landmark Normalization -> Drawing -> JPEG Encoding -> Model Inference ->
    Multi-Modal Fusion -> WordBuffer State Machine.

Reports: Median (p50), Mean, p90, p95, p99, Min, Max, and Effective FPS.
Also provides mathematical explanation of Mean vs. p95 right-tail skew.

Usage:
    ./venv/bin/python src/benchmark_latency.py [--iterations 50] [--json]

Author: Antigravity Benchmarking Suite
Date: 2026-11-16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.alphabet_engine import AlphabetEngine
from src.cnn_engine import CNNEngine
from src.dual_hand_engine import DualHandEngine, compute_dual_hand_features
from src.fusion_engine import FusionEngine
from src.inference_engine import InferenceEngine


def profile_callable(fn, *args, iterations: int = 50, warmup: int = 10, **kwargs) -> Dict[str, float]:
    """Profile latency of a callable over warmup and benchmark iterations."""
    for _ in range(warmup):
        _ = fn(*args, **kwargs)

    latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "effective_fps": round(float(1000.0 / np.mean(arr)), 1) if np.mean(arr) > 0 else 0.0,
    }


def load_sample_hand_frame() -> np.ndarray:
    """Load a sample image with hands from Dataset.zip if available, or generate a realistic frame."""
    zip_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "staging_downloads" / "realsign" / "Dataset.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                candidates = [n for n in z.namelist() if n.startswith("Testing/") and (n.endswith(".jpg") or n.endswith(".png"))]
                if candidates:
                    data = z.read(candidates[0])
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return cv2.resize(img, (640, 480))
        except Exception:
            pass

    # Fallback synthetic frame (640x480 BGR)
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)


def run_benchmark(iterations: int = 50) -> Dict[str, Any]:
    print(f"Initializing ISL Recognition Pipeline for Latency Benchmark ({iterations} iterations)...")
    alpha_engine = AlphabetEngine()

    # Generate realistic dummy landmarks
    rng = np.random.default_rng(42)
    c1 = rng.uniform(0.1, 0.4, (21, 3)).astype(np.float32)
    c2 = rng.uniform(0.6, 0.9, (21, 3)).astype(np.float32)
    single_c = rng.uniform(0.3, 0.7, (21, 3)).astype(np.float32)

    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class MockHand:
        def __init__(self, coords):
            self.landmark = [MockLandmark(x, y, z) for x, y, z in coords]

    mock_hands_dual = [MockHand(c1), MockHand(c2)]
    mock_hands_single = [MockHand(single_c)]

    class MockClassification:
        def __init__(self, label, score):
            self.label = label
            self.score = score

    class MockHandedness:
        def __init__(self, label, score):
            self.classification = [MockClassification(label, score)]

    mock_handedness_dual = [MockHandedness("Left", 0.98), MockHandedness("Right", 0.99)]
    mock_handedness_single = [MockHandedness("Right", 0.99)]

    rgb_dummy = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    results: Dict[str, Any] = {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "iterations": iterations,
        "inference_only_components": {},
        "full_pipeline_components": {},
        "notes": (
            "Right-tail skew: Mean latency can be higher than or near p95 when occasional OS thread context "
            "switches, Apple Silicon frequency scaling, or Python memory allocations cause rare spike frames "
            "(reflected in p99 and max). Median represents typical sustained throughput."
        ),
    }

    # =========================================================================
    # Section 1: Component & Inference-Only Benchmarks
    # =========================================================================
    print("\n--- Benchmarking Component & Inference-Only Execution ---")

    print("Profiling: Dual-Hand 144D Feature Extraction...")
    results["inference_only_components"]["dual_hand_feature_extraction_144d"] = profile_callable(
        compute_dual_hand_features, c1, c2, "config_d_ablation_144d", iterations=iterations
    )

    print("Profiling: Dual-Hand Engine (Features + MLP Inference)...")
    results["inference_only_components"]["dual_hand_engine_predict"] = profile_callable(
        alpha_engine.dual_hand_engine.predict, c1, c2, iterations=iterations
    )

    if alpha_engine.single_model is not None:
        print("Profiling: Single-Hand MLP (63D) Inference...")
        single_input = rng.uniform(-1.0, 1.0, (1, 63)).astype(np.float32)
        results["inference_only_components"]["single_hand_mlp_predict"] = profile_callable(
            lambda x: alpha_engine.single_model(x, training=False), single_input, iterations=iterations
        )

    print("Profiling: RGB CNN Engine (Crop + 128x128 Inference)...")
    results["inference_only_components"]["cnn_engine_predict"] = profile_callable(
        alpha_engine.cnn_engine.predict, rgb_dummy, mock_hands_dual, iterations=iterations
    )

    print("Profiling: Fusion Engine Arbitration...")
    dummy_lm_res = {"pred_label": "A", "confidence": 0.88, "status": "OK", "model_name": "TestLM"}
    dummy_cnn_res = {"pred_label": "A", "confidence": 0.82, "status": "OK", "model_name": "TestCNN"}
    results["inference_only_components"]["fusion_engine_arbitration"] = profile_callable(
        alpha_engine.fusion_engine.fuse, dummy_lm_res, dummy_cnn_res, True, iterations=iterations
    )

    print("Profiling: AlphabetEngine - Single Hand (Landmarks -> Decision)...")
    results["inference_only_components"]["alphabet_engine_single_hand"] = profile_callable(
        alpha_engine.process_dual_landmarks, mock_hands_single, mock_handedness_single, rgb_dummy, iterations=iterations
    )

    print("Profiling: AlphabetEngine - Dual Hand (Landmarks -> Decision)...")
    results["inference_only_components"]["alphabet_engine_dual_hand"] = profile_callable(
        alpha_engine.process_dual_landmarks, mock_hands_dual, mock_handedness_dual, rgb_dummy, iterations=iterations
    )

    # =========================================================================
    # Section 2: Full End-to-End Pipeline (InferenceEngine.process_frame)
    # =========================================================================
    print("\n--- Benchmarking Full End-to-End Pipeline (Capture -> Output) ---")
    inf_engine = InferenceEngine(camera_index=-1)
    inf_engine._is_cloud = False  # Ensure drawing & JPEG encoding are enabled
    sample_frame = load_sample_hand_frame()

    print("Profiling: Full Pipeline - Empty / Background Frame (No Hands)...")
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results["full_pipeline_components"]["full_pipeline_empty_frame"] = profile_callable(
        inf_engine.process_frame, empty_frame, is_mirrored=False, iterations=iterations
    )

    print("Profiling: Full Pipeline - Active Hand Frame (MediaPipe + Drawing + JPEG + Models)...")
    results["full_pipeline_components"]["full_pipeline_active_frame"] = profile_callable(
        inf_engine.process_frame, sample_frame, is_mirrored=False, iterations=iterations
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Latency & Throughput Benchmark for ISL Recognition.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    benchmark_data = run_benchmark(iterations=args.iterations)

    if args.json:
        print(json.dumps(benchmark_data, indent=2))
    else:
        print("\n" + "=" * 80)
        print("          ISL PIPELINE INFERENCE & END-TO-END LATENCY BENCHMARK         ")
        print("=" * 80)
        hw = benchmark_data["hardware"]
        print(f"Hardware:    {hw['machine']} ({hw['platform']})")
        print(f"Processor:   {hw['processor']}")
        print(f"Python:      {hw['python_version']}")
        print(f"Iterations:  {benchmark_data['iterations']}\n")

        print("1. INFERENCE-ONLY & COMPONENT LATENCIES (Excludes Sensor & OpenCV Drawing)")
        print(f"{'Component':<36} | {'Median':<8} | {'Mean':<8} | {'p95':<8} | {'Max':<8} | {'FPS':<6}")
        print("-" * 80)
        for name, m in benchmark_data["inference_only_components"].items():
            print(
                f"{name:<36} | "
                f"{m['median_ms']:>6.2f} ms | "
                f"{m['mean_ms']:>6.2f} ms | "
                f"{m['p95_ms']:>6.2f} ms | "
                f"{m['max_ms']:>6.2f} ms | "
                f"{m['effective_fps']:>5.1f}"
            )

        print("\n2. TRUE FULL END-TO-END PIPELINE (InferenceEngine.process_frame)")
        print("(Ingestion -> Unmirror -> MediaPipe Tracking -> Landmarks -> Models -> Fusion -> Drawing -> JPEG)")
        print(f"{'Pipeline Mode':<36} | {'Median':<8} | {'Mean':<8} | {'p95':<8} | {'Max':<8} | {'FPS':<6}")
        print("-" * 80)
        for name, m in benchmark_data["full_pipeline_components"].items():
            print(
                f"{name:<36} | "
                f"{m['median_ms']:>6.2f} ms | "
                f"{m['mean_ms']:>6.2f} ms | "
                f"{m['p95_ms']:>6.2f} ms | "
                f"{m['max_ms']:>6.2f} ms | "
                f"{m['effective_fps']:>5.1f}"
            )

        print("\n3. TAIL SKEW & STATISTICAL EXPLANATION:")
        print(f"   {benchmark_data['notes']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
