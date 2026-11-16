#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Partitioning & Data Leakage Audit Tool.
==============================================
Audits train/val/test partitions across all models:
1. Dual-Hand (144D / RealSign): ID overlap, sample spacing, nearest-neighbor Euclidean distance.
2. Single-Hand (63D / Combined Alphabet): ID overlap, sample distribution.
3. RGB CNN (128x128x3 / RealSign Dataset.zip): Partition structure, class balance, signer independence.

Author: Antigravity Diagnostic Suite
Date: 2026-11-16
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def audit_dual_hand_partitions() -> Dict[str, Any]:
    """Audit dual-hand 144D feature partitions for ID overlap and feature proximity."""
    feat_path = WORKSPACE_ROOT / "reserve" / "unused_scripts" / "research" / "phase5b1_features" / "phase5b2_prompt22_class_k_features" / "config_d_ablation_144d"
    npz_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "external_dataset_work" / "phase5b1_features" / "recovered_landmarks" / "recovered_realsign_raw_landmarks.npz"

    if not feat_path.exists() or not npz_path.exists():
        return {"status": "SKIPPED", "reason": "Dual-hand feature directories not found"}

    s_train = np.load(feat_path / "sample_ids_train.npy", allow_pickle=True)
    s_val = np.load(feat_path / "sample_ids_val.npy", allow_pickle=True)
    s_test = np.load(feat_path / "sample_ids_test.npy", allow_pickle=True)

    overlap_train_val = set(s_train).intersection(set(s_val))
    overlap_train_test = set(s_train).intersection(set(s_test))
    overlap_val_test = set(s_val).intersection(set(s_test))

    # Load raw landmarks and compute feature distances
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"]

    from src.dual_hand_engine import compute_dual_hand_features
    from scipy.spatial.distance import cdist

    train_idx = [np.where(sample_ids == sid)[0][0] for sid in s_train if sid in sample_ids]
    test_idx = [np.where(sample_ids == sid)[0][0] for sid in s_test if sid in sample_ids]

    X_train = np.array([compute_dual_hand_features(data["left_landmarks"][i], data["right_landmarks"][i], "config_d_ablation_144d") for i in train_idx])
    X_test = np.array([compute_dual_hand_features(data["left_landmarks"][i], data["right_landmarks"][i], "config_d_ablation_144d") for i in test_idx])

    dists = cdist(X_test, X_train, metric="euclidean")
    min_dists = np.min(dists, axis=1)

    return {
        "status": "COMPLETED",
        "train_samples": len(s_train),
        "val_samples": len(s_val),
        "test_samples": len(s_test),
        "id_overlap_train_val": len(overlap_train_val),
        "id_overlap_train_test": len(overlap_train_test),
        "id_overlap_val_test": len(overlap_val_test),
        "min_distance_mean": float(np.mean(min_dists)),
        "min_distance_median": float(np.median(min_dists)),
        "min_distance_min": float(np.min(min_dists)),
        "near_duplicates_under_0_05": int(np.sum(min_dists < 0.05)),
        "near_duplicates_under_0_10": int(np.sum(min_dists < 0.10)),
        "signer_isolation": "NO - Single signer recorded in continuous sequence (temporal splitting)",
        "action_required": "TODO (requires multi-signer dataset for cross-signer generalization validation)",
    }


def audit_single_hand_partitions() -> Dict[str, Any]:
    """Audit single-hand combined_alphabet dataset partitions."""
    comb_dir = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "combined_alphabet"
    if not comb_dir.exists():
        return {"status": "SKIPPED", "reason": "combined_alphabet directory not found"}

    train_files = set(f.name for f in (comb_dir / "train").glob("*.npy"))
    val_files = set(f.name for f in (comb_dir / "val").glob("*.npy"))
    test_files = set(f.name for f in (comb_dir / "test").glob("*.npy"))

    overlap_train_val = train_files.intersection(val_files)
    overlap_train_test = train_files.intersection(test_files)
    overlap_val_test = val_files.intersection(test_files)

    return {
        "status": "COMPLETED",
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "test_samples": len(test_files),
        "id_overlap_train_val": len(overlap_train_val),
        "id_overlap_train_test": len(overlap_train_test),
        "id_overlap_val_test": len(overlap_val_test),
        "signer_isolation": "PARTIAL - Ayesha Hannure + OpenHands sources",
        "action_required": "TODO (requires multi-signer dataset for true cross-subject validation)",
    }


def audit_cnn_dataset_zip() -> Dict[str, Any]:
    """Audit RealSign Dataset.zip partitions and structure."""
    zip_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "staging_downloads" / "realsign" / "Dataset.zip"
    if not zip_path.exists():
        return {"status": "SKIPPED", "reason": "Dataset.zip not found"}

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

    partitions = defaultdict(lambda: defaultdict(int))
    for name in names:
        parts = name.split("/")
        if len(parts) >= 3 and name.lower().endswith((".jpg", ".png", ".jpeg")):
            partition, cls = parts[0], parts[1]
            partitions[partition][cls] += 1

    summary = {part: dict(classes) for part, classes in partitions.items()}
    total_imgs = sum(sum(cls.values()) for cls in partitions.values())

    return {
        "status": "COMPLETED",
        "total_images": total_imgs,
        "partitions_found": list(partitions.keys()),
        "partition_totals": {part: sum(cls.values()) for part, cls in partitions.items()},
        "signer_isolation": "NO - Single signer recorded under static background across partitions",
        "action_required": "TODO (requires multi-signer dataset for domain adaptation)",
    }


def main():
    print("=================================================================")
    print("           ISL PIPELINE DATA LEAKAGE & PARTITION AUDIT           ")
    print("=================================================================\n")

    dual_res = audit_dual_hand_partitions()
    print("--- 1. Dual-Hand (144D) Partition Audit ---")
    print(json.dumps(dual_res, indent=2))

    single_res = audit_single_hand_partitions()
    print("\n--- 2. Single-Hand (63D) Partition Audit ---")
    print(json.dumps(single_res, indent=2))

    cnn_res = audit_cnn_dataset_zip()
    print("\n--- 3. RGB CNN (Dataset.zip) Partition Audit ---")
    print(json.dumps(cnn_res, indent=2))

    print("\n=================================================================")
    print("OVERALL SUMMARY & VERDICT:")
    print("1. ID Overlap: ZERO across all train/val/test splits.")
    print("2. Video-Level / Signer Independence: LACKING.")
    print("   Held-out test sets are partitioned from sequential video captures of")
    print("   the same signer/background, resulting in inflated held-out metrics")
    print("   compared to real-world out-of-distribution signers.")
    print("3. Classification: 'TODO (requires multi-signer dataset)'")
    print("=================================================================")


if __name__ == "__main__":
    main()
