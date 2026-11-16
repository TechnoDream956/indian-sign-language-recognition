#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Benchmark Sample Recording & Evaluation Tool.
=================================================
Records verified live webcam samples with ground truth labels and signer metadata
to evaluate real-world pipeline accuracy and cross-signer generalization.

Features:
1. Interactive Webcam Recording Mode:
   - Extracts MediaPipe landmarks from camera.
   - Saves unmirrored landmarks matching model coordinate convention.
   - Stores labeled samples to git-ignored `data/live_benchmark_samples/<signer_id>/`.
2. Evaluation Mode (`--eval`):
   - Evaluates active pipeline (AlphabetEngine) on recorded live samples.
   - Reports accuracy, per-signer breakdown, and error analysis.
3. Headless/Synthetic Test Mode (`--synthetic`):
   - Generates synthetic benchmark samples for automated headless testing.

Usage:
    # Record live samples for a signer:
    ./venv/bin/python tools/collection/record_live_benchmark_samples.py --record --signer-id arpit --letter A --count 10

    # Evaluate active pipeline on all recorded live samples:
    ./venv/bin/python tools/collection/record_live_benchmark_samples.py --eval

    # Run synthetic sample test in headless environments:
    ./venv/bin/python tools/collection/record_live_benchmark_samples.py --synthetic --count 5

Author: Sachin (Data Collection & Preprocessing) & Arpit (Model Evaluation)
Date: 2026-11-16
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.alphabet_engine import AlphabetEngine, extract_dual_hand_landmarks

DATA_DIR = WORKSPACE_ROOT / "data" / "live_benchmark_samples"


def save_sample(
    output_dir: Path,
    signer_id: str,
    label: str,
    left_coords: np.ndarray,
    right_coords: np.ndarray,
    hand_count: int,
    rgb_crop: Optional[np.ndarray] = None,
) -> Path:
    """Save recorded sample to an .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_id = f"smp_{int(time.time() * 1000)}_{label}"
    filepath = output_dir / f"{sample_id}.npz"

    save_dict = {
        "signer_id": signer_id,
        "label": label,
        "hand_count": hand_count,
        "left_landmarks": left_coords.astype(np.float32),
        "right_landmarks": right_coords.astype(np.float32),
        "timestamp": time.time(),
    }
    if rgb_crop is not None:
        save_dict["rgb_crop"] = rgb_crop

    np.savez_compressed(filepath, **save_dict)
    return filepath


def record_live_samples(
    signer_id: str,
    target_letter: str,
    target_count: int = 10,
    camera_id: int = 0,
) -> None:
    """Record live samples using webcam with MediaPipe Hands."""
    try:
        import mediapipe as mp
    except ImportError:
        print("[Error] MediaPipe is not installed. Install it or run in synthetic mode.")
        return

    signer_dir = DATA_DIR / signer_id
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {camera_id}.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    print(f"\n=======================================================")
    print(f" LIVE RECORDING: Letter '{target_letter}' | Signer '{signer_id}'")
    print(f" Target: {target_count} samples | Press [SPACE] to capture, [Q] to quit")
    print(f"=======================================================\n")

    captured = 0
    try:
        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame is BGR, unmirrored
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Left/Right feature extraction (Unmirrored convention)
            left_feats, right_feats, combined_126, meta = extract_dual_hand_landmarks(
                results.multi_hand_landmarks, results.multi_handedness
            )

            # Display feed (visual flip for intuitive user feedback)
            display_frame = cv2.flip(frame, 1)
            status_text = f"Letter: {target_letter} | Captured: {captured}/{target_count} | Hands: {meta['hand_count']}"
            cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, Q to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Live ISL Benchmark Recorder", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Recording aborted by user.")
                break
            elif key == 32:  # Spacebar
                if meta["hand_count"] == 0:
                    print("[Warning] No hands detected! Hold your sign clearly in view.")
                    continue

                fp = save_sample(
                    output_dir=signer_dir,
                    signer_id=signer_id,
                    label=target_letter,
                    left_coords=meta["c1"],
                    right_coords=meta["c2"],
                    hand_count=meta["hand_count"],
                )
                captured += 1
                print(f"[{captured}/{target_count}] Captured sample -> {fp.name}")
                time.sleep(0.3)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

    print(f"\nFinished recording: {captured} samples saved to {signer_dir}")


def generate_synthetic_samples(count: int = 5) -> None:
    """Generate synthetic samples for headless testing and verification."""
    test_dir = DATA_DIR / "synthetic_test_signer"
    letters = ["A", "B", "C", "D", "K", "T"]
    rng = np.random.default_rng(12345)

    print(f"Generating synthetic benchmark samples in: {test_dir}...")
    for letter in letters:
        for i in range(count):
            c1 = rng.uniform(0.1, 0.4, (21, 3)).astype(np.float32)
            c2 = rng.uniform(0.6, 0.9, (21, 3)).astype(np.float32)
            save_sample(
                output_dir=test_dir,
                signer_id="synthetic_test_signer",
                label=letter,
                left_coords=c1,
                right_coords=c2,
                hand_count=2,
            )
    print(f"Generated {len(letters) * count} synthetic samples successfully.")


def evaluate_recorded_samples() -> Dict[str, Any]:
    """Evaluate pipeline on all recorded live benchmark samples."""
    sample_files = sorted(glob.glob(str(DATA_DIR / "**" / "*.npz"), recursive=True))
    if not sample_files:
        print(f"[Warning] No benchmark samples found in {DATA_DIR}.")
        print("Use --record or --synthetic to create samples first.")
        return {"error": "No samples found"}

    print(f"\nEvaluating pipeline on {len(sample_files)} benchmark samples...")
    engine = AlphabetEngine()

    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class MockHand:
        def __init__(self, coords):
            self.landmark = [MockLandmark(x, y, z) for x, y, z in coords]

    by_signer = defaultdict(lambda: {"total": 0, "correct": 0, "by_letter": defaultdict(lambda: {"total": 0, "correct": 0})})
    overall_total = 0
    overall_correct = 0

    for f in sample_files:
        data = np.load(f, allow_pickle=True)
        signer_id = str(data["signer_id"])
        true_label = str(data["label"]).upper()
        hand_count = int(data["hand_count"])
        left_coords = data["left_landmarks"]
        right_coords = data["right_landmarks"]

        if hand_count == 2:
            mock_hands = [MockHand(left_coords), MockHand(right_coords)]
        elif hand_count == 1:
            active_coords = left_coords if not np.allclose(left_coords, 0.0) else right_coords
            mock_hands = [MockHand(active_coords)]
        else:
            mock_hands = []

        res = engine.process_dual_landmarks(mock_hands)
        pred_label = res.get("active_letter") or res.get("landmark_prediction") or res.get("fused_prediction")

        is_correct = (pred_label == true_label)
        overall_total += 1
        if is_correct:
            overall_correct += 1

        signer_stats = by_signer[signer_id]
        signer_stats["total"] += 1
        if is_correct:
            signer_stats["correct"] += 1
        signer_stats["by_letter"][true_label]["total"] += 1
        if is_correct:
            signer_stats["by_letter"][true_label]["correct"] += 1

    overall_acc = (overall_correct / overall_total) if overall_total > 0 else 0.0
    print("\n=======================================================")
    print(f" LIVE BENCHMARK EVALUATION RESULTS ({overall_total} samples)")
    print(f" Overall Accuracy: {overall_acc * 100:.2f}% ({overall_correct}/{overall_total})")
    print("=======================================================")

    for s_id, s_stats in by_signer.items():
        s_acc = (s_stats["correct"] / s_stats["total"]) if s_stats["total"] > 0 else 0.0
        print(f"\nSigner: {s_id} | Accuracy: {s_acc * 100:.2f}% ({s_stats['correct']}/{s_stats['total']})")
        for ltr, l_stats in sorted(s_stats["by_letter"].items()):
            l_acc = (l_stats["correct"] / l_stats["total"]) if l_stats["total"] > 0 else 0.0
            print(f"  Letter '{ltr}': {l_acc * 100:.1f}% ({l_stats['correct']}/{l_stats['total']})")

    return {
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": round(overall_acc, 4),
        "signers": {s: {"total": st["total"], "correct": st["correct"], "accuracy": round(st["correct"] / st["total"], 4)} for s, st in by_signer.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Live Benchmark Sample Recorder and Evaluator.")
    parser.add_argument("--record", action="store_true", help="Launch interactive webcam recorder")
    parser.add_argument("--eval", action="store_true", help="Evaluate active pipeline on recorded samples")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test samples (headless)")
    parser.add_argument("--signer-id", type=str, default="user_01", help="Signer ID for recording")
    parser.add_argument("--letter", type=str, default="A", help="Target letter (A-Z) to record")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to capture/generate")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default: 0)")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_samples(count=args.count)
    elif args.record:
        record_live_samples(
            signer_id=args.signer_id,
            target_letter=args.letter.upper(),
            target_count=args.count,
            camera_id=args.camera_id,
        )
    elif args.eval:
        evaluate_recorded_samples()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
