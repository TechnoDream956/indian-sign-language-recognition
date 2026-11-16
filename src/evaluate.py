#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Model Evaluation & Empirical Benchmarking Tool.
=========================================================
Evaluates ISL Recognition Models on held-out test partitions:
1. Single-Hand 26-Class MLP (63D) - Same-Signer Held-Out
2. Dual-Hand 20-Class MLP (144D, Prompt 23 Config D Ablation) - Same-Signer Held-Out
3. RGB CNN 26-Class Classifier (128x128x3) - Same-Signer Held-Out
4. True Paired Multi-Modal Fusion Ablation (Geometry-Only vs CNN-Only vs Fused on 243 paired samples)
5. Comprehensive Mirror Parity Test across all models (Dual-Hand, Single-Hand, RGB CNN)
6. McNemar Significance Test for Heuristic Overrides (K vs T)
7. 95% Empirical Bootstrap Confidence Intervals
8. Confusion Matrix Plotting (saves to assets/confusion_matrix.png)

Usage:
    ./venv/bin/python src/evaluate.py [--model all|dual|single|cnn|ablation|fusion|mirror] [--output-matrix assets/confusion_matrix.png] [--json]

Authors:
    Arpit: Recognition Model Development
    Sachin: Data Collection, Preprocessing & Backend Support
    Abhu: Multilingual Support & TTS
    Yash: Frontend & Deployment
Date: 2026-11-16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import tensorflow as tf

# Workspace imports
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.dual_hand_engine import DualHandEngine, compute_dual_hand_features
from src.fusion_engine import FusionEngine

DUAL_CLASSES_20 = [
    "A", "B", "D", "E", "F", "G", "H", "I", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
]
ALPHABET_CLASSES_26 = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for accuracy and macro F1."""
    accs = []
    f1s = []
    n = len(y_true)
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

    acc_ci = (float(np.percentile(accs, 100 * (alpha / 2))), float(np.percentile(accs, 100 * (1 - alpha / 2))))
    f1_ci = (float(np.percentile(f1s, 100 * (alpha / 2))), float(np.percentile(f1s, 100 * (1 - alpha / 2))))
    return acc_ci, f1_ci


def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, Any]:
    """Perform McNemar's test with continuity correction and exact binomial test."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(correct_a & (~correct_b)))  # correct in A, wrong in B
    c = int(np.sum((~correct_a) & correct_b))  # wrong in A, correct in B
    d = int(np.sum((~correct_a) & (~correct_b)))

    p_val_chi2 = 1.0
    chi2_stat = 0.0
    if (b + c) > 0:
        chi2_stat = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val_chi2 = float(1.0 - chi2.cdf(chi2_stat, df=1))
        binom_res = binomtest(b, b + c, p=0.5)
        p_val_exact = float(binom_res.pvalue)
    else:
        p_val_exact = 1.0

    return {
        "contingency_table": {"both_correct": a, "a_only": b, "b_only": c, "both_wrong": d},
        "chi2_stat": round(chi2_stat, 4),
        "p_value_chi2": round(p_val_chi2, 6),
        "p_value_exact_binomial": round(p_val_exact, 6),
        "is_significant_at_05": p_val_exact < 0.05,
        "is_significant_at_01": p_val_exact < 0.01,
    }


def evaluate_dual_hand_model(plot_cm_path: Optional[Path] = None) -> Dict[str, Any]:
    """Evaluate dual-hand 144D model on held-out test partitions."""
    model_path = WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k" / "config_d_ablation_144d" / "best_val_model.h5"
    npz_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "external_dataset_work" / "phase5b1_features" / "recovered_landmarks" / "recovered_realsign_raw_landmarks.npz"
    feat_path = WORKSPACE_ROOT / "reserve" / "unused_scripts" / "research" / "phase5b1_features" / "phase5b2_prompt22_class_k_features" / "config_d_ablation_144d"

    if not model_path.exists() or not npz_path.exists() or not feat_path.exists():
        return {"error": "Required dual-hand model or dataset files not found"}

    model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)

    # 1. Full 252 Test Features (RealSign 243 + OpenHands 9)
    X_test_all = np.load(feat_path / "X_test.npy", allow_pickle=True)
    y_test_all_raw = np.load(feat_path / "y_test.npy", allow_pickle=True)
    sids_test = np.load(feat_path / "sample_ids_test.npy", allow_pickle=True)

    class_to_idx = {c: i for i, c in enumerate(DUAL_CLASSES_20)}
    idx_to_class = {i: c for i, c in enumerate(DUAL_CLASSES_20)}
    y_true_all = np.array([class_to_idx[y] for y in y_test_all_raw], dtype=np.int32)

    probs_all = model.predict(X_test_all, verbose=0)
    preds_all = np.argmax(probs_all, axis=1)

    acc_all = float(accuracy_score(y_true_all, preds_all))
    f1_all = float(f1_score(y_true_all, preds_all, average="macro", zero_division=0))

    # 2. RealSign 243 Same-Signer Held-Out Partition (with Raw Landmarks)
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"]

    test_indices = [np.where(sample_ids == sid)[0][0] for sid in sids_test if sid in sample_ids]
    raw_left = data["left_landmarks"][test_indices]
    raw_right = data["right_landmarks"][test_indices]
    labels_raw = data["labels"][test_indices]
    y_true_rs = np.array([class_to_idx[l] for l in labels_raw], dtype=np.int32)

    X_feats_rs = np.array([compute_dual_hand_features(l, r, "config_d_ablation_144d") for l, r in zip(raw_left, raw_right)])
    probs_rs = model.predict(X_feats_rs, verbose=0)
    raw_preds_rs = np.argmax(probs_rs, axis=1)

    acc_rs_base = float(accuracy_score(y_true_rs, raw_preds_rs))
    f1_rs_base = float(f1_score(y_true_rs, raw_preds_rs, average="macro", zero_division=0))
    acc_ci_base, f1_ci_base = compute_bootstrap_ci(y_true_rs, raw_preds_rs)

    top3_preds = np.argsort(probs_rs, axis=1)[:, -3:]
    top3_acc_base = float(np.mean([y_true_rs[i] in top3_preds[i] for i in range(len(y_true_rs))]))

    one_hot_y = tf.keras.utils.to_categorical(y_true_rs, num_classes=len(DUAL_CLASSES_20))
    brier_base = float(np.mean(np.sum((probs_rs - one_hot_y) ** 2, axis=1)))

    # K vs T Override
    preds_kt = raw_preds_rs.copy()
    for i in range(len(preds_kt)):
        lbl = idx_to_class[preds_kt[i]]
        c1 = raw_left[i]
        if lbl == "T" and len(c1) >= 21:
            dy = c1[5, 1] - c1[8, 1]
            dx = abs(c1[8, 0] - c1[5, 0])
            if dy > 0.08 and dy > 1.2 * dx:
                preds_kt[i] = class_to_idx["K"]

    acc_kt = float(accuracy_score(y_true_rs, preds_kt))
    f1_kt = float(f1_score(y_true_rs, preds_kt, average="macro", zero_division=0))
    acc_ci_kt, f1_ci_kt = compute_bootstrap_ci(y_true_rs, preds_kt)

    # McNemar Paired Test for K/T Override
    mcnemar_kt = run_mcnemar_test(y_true_rs, raw_preds_rs, preds_kt)

    # B/H Clamp
    preds_bh = raw_preds_rs.copy()
    for i in range(len(preds_bh)):
        lbl = idx_to_class[preds_bh[i]]
        c1 = raw_left[i]
        c2 = raw_right[i]
        if lbl in ["B", "H"]:
            wrist_dist = float(np.linalg.norm(c1[0] - c2[0]))
            if wrist_dist > 0.42:
                preds_bh[i] = -1
            elif lbl == "B":
                a1 = float(np.linalg.norm(c1[4] - c1[8]))
                a2 = float(np.linalg.norm(c2[4] - c2[8]))
                if a1 > 0.16 or a2 > 0.16:
                    preds_bh[i] = -1

    acc_bh = float(np.mean(preds_bh == y_true_rs))
    f1_bh = float(f1_score(y_true_rs, np.where(preds_bh == -1, 999, preds_bh), average="macro", labels=list(range(20)), zero_division=0))

    if plot_cm_path is not None:
        plot_cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true_rs, preds_kt, labels=list(range(len(DUAL_CLASSES_20))))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DUAL_CLASSES_20, yticklabels=DUAL_CLASSES_20)
        plt.title(f"Dual-Hand 144D Confusion Matrix - Same-Signer Held-Out (Acc: {acc_kt*100:.2f}%)")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(plot_cm_path, dpi=300)
        plt.close()

    return {
        "model_name": "ISL Dual-Hand MLP (144D, Config D Ablation)",
        "num_classes": 20,
        "partition_reconciliation": {
            "full_test_samples": len(y_true_all),
            "full_test_accuracy": round(acc_all, 4),
            "full_test_macro_f1": round(f1_all, 4),
            "realsign_samples": len(y_true_rs),
            "openhands_samples": len(y_true_all) - len(y_true_rs),
            "openhands_note": "9 OpenHands samples suffer from scale-normalization divergence (L-inf vs wrist-to-MCP).",
        },
        "same_signer_held_out_realsign_243": {
            "baseline_model_only": {
                "top1_accuracy": round(acc_rs_base, 4),
                "top3_accuracy": round(top3_acc_base, 4),
                "macro_f1": round(f1_rs_base, 4),
                "brier_score": round(brier_base, 4),
                "top1_accuracy_95ci": [round(acc_ci_base[0], 4), round(acc_ci_base[1], 4)],
                "macro_f1_95ci": [round(f1_ci_base[0], 4), round(f1_ci_base[1], 4)],
            },
            "with_k_vs_t_override": {
                "top1_accuracy": round(acc_kt, 4),
                "macro_f1": round(f1_kt, 4),
                "delta_accuracy": round(acc_kt - acc_rs_base, 4),
                "delta_f1": round(f1_kt - f1_rs_base, 4),
                "top1_accuracy_95ci": [round(acc_ci_kt[0], 4), round(acc_ci_kt[1], 4)],
                "macro_f1_95ci": [round(f1_ci_kt[0], 4), round(f1_ci_kt[1], 4)],
                "mcnemar_test": mcnemar_kt,
            },
            "with_bh_clamp": {
                "top1_accuracy": round(acc_bh, 4),
                "macro_f1": round(f1_bh, 4),
                "delta_accuracy": round(acc_bh - acc_rs_base, 4),
                "note": "Degrades test accuracy by rejecting valid poses. Default set to inactive behind config flag.",
            },
        },
    }


def evaluate_paired_fusion() -> Dict[str, Any]:
    """Run true paired multi-modal fusion ablation on the exact same 243 RealSign samples."""
    dh_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k" / "config_d_ablation_144d" / "best_val_model.h5", compile=False, safe_mode=False)
    cnn_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "cnn_sign_model.h5", compile=False)
    fusion_engine = FusionEngine()

    feat_path = WORKSPACE_ROOT / "reserve" / "unused_scripts" / "research" / "phase5b1_features" / "phase5b2_prompt22_class_k_features" / "config_d_ablation_144d"
    sids_test = np.load(feat_path / "sample_ids_test.npy", allow_pickle=True)
    manifest_df = pd.read_csv(WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "external_dataset_work" / "phase5b1_features" / "phase5b2_expansion_v2" / "manifest_phase5b2_v1.csv")
    manifest_map = {row["sample_id"]: (row["label"], row["original_identifier"]) for _, row in manifest_df.iterrows()}

    npz_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "external_dataset_work" / "phase5b1_features" / "recovered_landmarks" / "recovered_realsign_raw_landmarks.npz"
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"]

    rs_sids = [s for s in sids_test if "realsign" in s]
    test_indices = [np.where(sample_ids == sid)[0][0] for sid in rs_sids]
    raw_left = data["left_landmarks"][test_indices]
    raw_right = data["right_landmarks"][test_indices]
    labels = data["labels"][test_indices]

    zip_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "staging_downloads" / "realsign" / "Dataset.zip"
    imgs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for sid in rs_sids:
            lbl, orig_id = manifest_map[sid]
            for p in [f"Testing/{lbl}/{orig_id}", f"Training/{lbl}/{orig_id}", f"Validation/{lbl}/{orig_id}"]:
                if p in z.namelist():
                    raw_bytes = z.read(p)
                    arr = np.frombuffer(raw_bytes, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    norm = (resized.astype(np.float32) / 127.5) - 1.0
                    imgs.append(norm)
                    break
    X_imgs = np.stack(imgs, axis=0)

    # 1. Geometry-only
    CLASSES_20 = DUAL_CLASSES_20
    i2c_20 = {i: c for i, c in enumerate(CLASSES_20)}
    X_feats = np.array([compute_dual_hand_features(l, r, "config_d_ablation_144d") for l, r in zip(raw_left, raw_right)])
    probs_geom = dh_model.predict(X_feats, verbose=0)
    preds_geom_raw = [i2c_20[i] for i in np.argmax(probs_geom, axis=1)]
    confs_geom = np.max(probs_geom, axis=1)

    preds_geom_kt = list(preds_geom_raw)
    for i in range(len(preds_geom_kt)):
        if preds_geom_kt[i] == "T" and len(raw_left[i]) >= 21:
            dy = raw_left[i][5, 1] - raw_left[i][8, 1]
            dx = abs(raw_left[i][8, 0] - raw_left[i][5, 0])
            if dy > 0.08 and dy > 1.2 * dx:
                preds_geom_kt[i] = "K"

    # 2. CNN-only
    CLASSES_26 = ALPHABET_CLASSES_26
    probs_cnn = cnn_model.predict(X_imgs, verbose=0)
    preds_cnn = [CLASSES_26[i] for i in np.argmax(probs_cnn, axis=1)]
    confs_cnn = np.max(probs_cnn, axis=1)

    # 3. Fused
    fused_labels = []
    for i in range(len(labels)):
        lm_res = {"pred_label": preds_geom_kt[i], "confidence": float(confs_geom[i]), "status": "OK", "model_name": "DualHand-144D"}
        cnn_res = {"pred_label": preds_cnn[i], "confidence": float(confs_cnn[i]), "status": "OK", "model_name": "CNN-128x128"}
        fused_res = fusion_engine.fuse(lm_res, cnn_res, hand_detected=True)
        fused_labels.append(fused_res.get("fused_prediction") or preds_geom_kt[i])

    y_true = np.array(labels)
    acc_geom_raw = float(accuracy_score(y_true, preds_geom_raw))
    f1_geom_raw = float(f1_score(y_true, preds_geom_raw, average="macro", zero_division=0))

    acc_geom_kt = float(accuracy_score(y_true, preds_geom_kt))
    f1_geom_kt = float(f1_score(y_true, preds_geom_kt, average="macro", zero_division=0))

    acc_cnn = float(accuracy_score(y_true, preds_cnn))
    f1_cnn = float(f1_score(y_true, preds_cnn, average="macro", zero_division=0))

    acc_fused = float(accuracy_score(y_true, fused_labels))
    f1_fused = float(f1_score(y_true, fused_labels, average="macro", zero_division=0))

    return {
        "paired_samples": len(y_true),
        "geometry_only_baseline": {"accuracy": round(acc_geom_raw, 4), "macro_f1": round(f1_geom_raw, 4)},
        "geometry_only_with_kt": {"accuracy": round(acc_geom_kt, 4), "macro_f1": round(f1_geom_kt, 4)},
        "cnn_only": {"accuracy": round(acc_cnn, 4), "macro_f1": round(f1_cnn, 4)},
        "multi_modal_fused": {"accuracy": round(acc_fused, 4), "macro_f1": round(f1_fused, 4)},
    }


def evaluate_mirror_parity_all() -> Dict[str, Any]:
    """Evaluate mirrored vs unmirrored input across all 3 models."""
    # 1. Dual-Hand (243 samples)
    dh_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k" / "config_d_ablation_144d" / "best_val_model.h5", compile=False, safe_mode=False)
    npz_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "external_dataset_work" / "phase5b1_features" / "recovered_landmarks" / "recovered_realsign_raw_landmarks.npz"
    feat_path = WORKSPACE_ROOT / "reserve" / "unused_scripts" / "research" / "phase5b1_features" / "phase5b2_prompt22_class_k_features" / "config_d_ablation_144d"
    sids_test = np.load(feat_path / "sample_ids_test.npy", allow_pickle=True)
    data = np.load(npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"]

    rs_sids = [s for s in sids_test if "realsign" in s]
    test_indices = [np.where(sample_ids == sid)[0][0] for sid in rs_sids]
    raw_left = data["left_landmarks"][test_indices]
    raw_right = data["right_landmarks"][test_indices]
    y_true_dh = np.array([DUAL_CLASSES_20.index(l) for l in data["labels"][test_indices]], dtype=np.int32)

    X_dh_unmirrored = np.array([compute_dual_hand_features(l, r, "config_d_ablation_144d") for l, r in zip(raw_left, raw_right)])
    acc_dh_unmirrored = float(accuracy_score(y_true_dh, np.argmax(dh_model.predict(X_dh_unmirrored, verbose=0), axis=1)))

    raw_left_m = raw_left.copy()
    raw_left_m[:, :, 0] = 1.0 - raw_left_m[:, :, 0]
    raw_right_m = raw_right.copy()
    raw_right_m[:, :, 0] = 1.0 - raw_right_m[:, :, 0]
    X_dh_mirrored = np.array([compute_dual_hand_features(raw_right_m[i], raw_left_m[i], "config_d_ablation_144d") for i in range(len(raw_left_m))])
    acc_dh_mirrored = float(accuracy_score(y_true_dh, np.argmax(dh_model.predict(X_dh_mirrored, verbose=0), axis=1)))

    # 2. Single-Hand (536 samples)
    sh_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "alphabet_model_26class.h5", compile=False)
    comb_test = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "combined_alphabet" / "test"
    test_files = sorted(comb_test.glob("*single*.npy"))
    X_sh_list, y_sh_list = [], []
    for f in test_files:
        lbl = f.stem.split("_")[3].upper()
        if len(lbl) == 1 and lbl in ALPHABET_CLASSES_26:
            arr = np.load(f).flatten().astype(np.float32)
            if len(arr) == 63:
                X_sh_list.append(arr)
                y_sh_list.append(ord(lbl) - ord("A"))
    X_sh = np.stack(X_sh_list, axis=0)
    y_sh = np.array(y_sh_list, dtype=np.int32)

    acc_sh_unmirrored = float(accuracy_score(y_sh, np.argmax(sh_model.predict(X_sh, verbose=0), axis=1)))
    X_sh_m = X_sh.copy().reshape(-1, 21, 3)
    X_sh_m[:, :, 0] = -X_sh_m[:, :, 0]
    X_sh_m = X_sh_m.reshape(-1, 63)
    acc_sh_mirrored = float(accuracy_score(y_sh, np.argmax(sh_model.predict(X_sh_m, verbose=0), axis=1)))

    # 3. RGB CNN (780 samples)
    cnn_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "cnn_sign_model.h5", compile=False)
    zip_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "staging_downloads" / "realsign" / "Dataset.zip"
    imgs_u, imgs_m, y_cnn_list = [], [], []
    with zipfile.ZipFile(zip_path, "r") as z:
        test_files_by_class = defaultdict(list)
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "Testing" and name.lower().endswith((".jpg", ".png", ".jpeg")):
                cls = parts[1]
                if cls in ALPHABET_CLASSES_26:
                    test_files_by_class[cls].append(name)
        for cls in ALPHABET_CLASSES_26:
            for name in test_files_by_class[cls][:30]:
                raw_bytes = z.read(name)
                arr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                res = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                imgs_u.append((res.astype(np.float32) / 127.5) - 1.0)
                imgs_m.append((cv2.flip(res, 1).astype(np.float32) / 127.5) - 1.0)
                y_cnn_list.append(ord(cls) - ord("A"))
    y_cnn = np.array(y_cnn_list, dtype=np.int32)
    acc_cnn_unmirrored = float(accuracy_score(y_cnn, np.argmax(cnn_model.predict(np.stack(imgs_u, axis=0), verbose=0), axis=1)))
    acc_cnn_mirrored = float(accuracy_score(y_cnn, np.argmax(cnn_model.predict(np.stack(imgs_m, axis=0), verbose=0), axis=1)))

    return {
        "dual_hand_144d": {
            "unmirrored_accuracy": round(acc_dh_unmirrored, 4),
            "mirrored_accuracy": round(acc_dh_mirrored, 4),
            "delta_drop": round(acc_dh_mirrored - acc_dh_unmirrored, 4),
        },
        "single_hand_63d": {
            "unmirrored_accuracy": round(acc_sh_unmirrored, 4),
            "mirrored_accuracy": round(acc_sh_mirrored, 4),
            "delta_drop": round(acc_sh_mirrored - acc_sh_unmirrored, 4),
        },
        "rgb_cnn_128x128": {
            "unmirrored_accuracy": round(acc_cnn_unmirrored, 4),
            "mirrored_accuracy": round(acc_cnn_mirrored, 4),
            "delta_drop": round(acc_cnn_mirrored - acc_cnn_unmirrored, 4),
        },
    }


def evaluate_single_hand_model() -> Dict[str, Any]:
    """Evaluate canonical single-hand 26-class MLP (63D) on held-out single test set."""
    model_path = WORKSPACE_ROOT / "models" / "alphabet_model_26class.h5"
    comb_test = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "combined_alphabet" / "test"

    if not model_path.exists() or not comb_test.exists():
        return {"error": "Single hand model or combined_alphabet test directory not found"}

    model = tf.keras.models.load_model(str(model_path), compile=False)
    test_files = sorted(comb_test.glob("*single*.npy"))

    X_list, y_list = [], []
    for f in test_files:
        lbl = f.stem.split("_")[3].upper()
        if len(lbl) == 1 and lbl in ALPHABET_CLASSES_26:
            arr = np.load(f).flatten().astype(np.float32)
            if len(arr) == 63:
                X_list.append(arr)
                y_list.append(ord(lbl) - ord("A"))

    X = np.stack(X_list, axis=0)
    y_true = np.array(y_list, dtype=np.int32)

    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    acc_ci, f1_ci = compute_bootstrap_ci(y_true, preds)

    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))]))

    one_hot_y = tf.keras.utils.to_categorical(y_true, num_classes=26)
    brier = float(np.mean(np.sum((probs - one_hot_y) ** 2, axis=1)))

    return {
        "model_name": "ISL Canonical Single-Hand MLP (63D)",
        "num_classes": 26,
        "test_samples": len(y_true),
        "same_signer_held_out": {
            "top1_accuracy": round(acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "macro_f1": round(f1, 4),
            "brier_score": round(brier, 4),
            "top1_accuracy_95ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "macro_f1_95ci": [round(f1_ci[0], 4), round(f1_ci[1], 4)],
        },
    }


def evaluate_cnn_model(samples_per_class: int = 30) -> Dict[str, Any]:
    """Evaluate 26-class RGB CNN on held-out test partition in Dataset.zip."""
    model_path = WORKSPACE_ROOT / "models" / "cnn_sign_model.h5"
    zip_path = WORKSPACE_ROOT / "reserve" / "old_datasets_and_manifests" / "staging_downloads" / "realsign" / "Dataset.zip"

    if not model_path.exists() or not zip_path.exists():
        return {"error": "CNN model or Dataset.zip not found"}

    model = tf.keras.models.load_model(str(model_path), compile=False)

    with zipfile.ZipFile(zip_path, "r") as z:
        test_files_by_class: Dict[str, List[str]] = defaultdict(list)
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == "Testing" and name.lower().endswith((".jpg", ".png", ".jpeg")):
                cls = parts[1]
                if cls in ALPHABET_CLASSES_26:
                    test_files_by_class[cls].append(name)

        X_imgs, y_labels = [], []
        for cls in ALPHABET_CLASSES_26:
            chosen = test_files_by_class[cls][:samples_per_class]
            for name in chosen:
                raw_bytes = z.read(name)
                arr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                normalized = (resized.astype(np.float32) / 127.5) - 1.0
                X_imgs.append(normalized)
                y_labels.append(ord(cls) - ord("A"))

    X = np.stack(X_imgs, axis=0)
    y_true = np.array(y_labels, dtype=np.int32)

    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    acc_ci, f1_ci = compute_bootstrap_ci(y_true, preds)

    top3_preds = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = float(np.mean([y_true[i] in top3_preds[i] for i in range(len(y_true))]))

    return {
        "model_name": "ISL RGB CNN Hand Classifier (128x128x3)",
        "num_classes": 26,
        "test_samples": len(y_true),
        "same_signer_held_out": {
            "top1_accuracy": round(acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "macro_f1": round(f1, 4),
            "top1_accuracy_95ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
            "macro_f1_95ci": [round(f1_ci[0], 4), round(f1_ci[1], 4)],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL Recognition Models on Held-Out Test Data.")
    parser.add_argument("--model", choices=["all", "dual", "single", "cnn", "ablation", "fusion", "mirror"], default="all", help="Model to evaluate")
    parser.add_argument("--output-matrix", type=str, default="assets/confusion_matrix.png", help="Path to save confusion matrix image")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    cm_path = WORKSPACE_ROOT / args.output_matrix if args.output_matrix else None

    if args.model in ["all", "dual", "ablation"]:
        results["dual_hand"] = evaluate_dual_hand_model(plot_cm_path=cm_path)

    if args.model in ["all", "single"]:
        results["single_hand"] = evaluate_single_hand_model()

    if args.model in ["all", "cnn"]:
        results["cnn"] = evaluate_cnn_model()

    if args.model in ["all", "fusion"]:
        results["paired_fusion"] = evaluate_paired_fusion()

    if args.model in ["all", "mirror"]:
        results["mirror_parity"] = evaluate_mirror_parity_all()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=================================================================")
        print("          INDIAN SIGN LANGUAGE RECOGNITION - EVALUATION          ")
        print("=================================================================\n")

        if "dual_hand" in results:
            dh = results["dual_hand"]
            rec = dh.get("partition_reconciliation", {})
            print("--- 1. Dual-Hand Model (144D, 20 Classes) ---")
            print(f"Full Test Set: {rec.get('full_test_samples')} samples (Acc: {rec.get('full_test_accuracy')*100:.2f}%, F1: {rec.get('full_test_macro_f1'):.4f})")
            print(f"Breakdown: {rec.get('realsign_samples')} RealSign + {rec.get('openhands_samples')} OpenHands")

            rs = dh.get("same_signer_held_out_realsign_243", {})
            base = rs.get("baseline_model_only", {})
            print(f"\nRealSign Same-Signer Held-Out (243 samples):")
            print(f"Model-Only Accuracy: {base.get('top1_accuracy', 0)*100:.2f}%  [95% CI: {base.get('top1_accuracy_95ci')[0]*100:.2f}% - {base.get('top1_accuracy_95ci')[1]*100:.2f}%]")
            print(f"Model-Only Macro F1: {base.get('macro_f1', 0):.4f}  [95% CI: {base.get('macro_f1_95ci')[0]:.4f} - {base.get('macro_f1_95ci')[1]:.4f}]")
            print(f"Model-Only Top-3:    {base.get('top3_accuracy', 0)*100:.2f}%")
            print(f"Brier Score:         {base.get('brier_score', 0):.4f}")

            kt = rs.get("with_k_vs_t_override", {})
            mc = kt.get("mcnemar_test", {})
            print(f"\nWith K vs T Override:")
            print(f"Accuracy: {kt.get('top1_accuracy', 0)*100:.2f}% ({kt.get('delta_accuracy', 0)*100:+.2f}%)  [95% CI: {kt.get('top1_accuracy_95ci')[0]*100:.2f}% - {kt.get('top1_accuracy_95ci')[1]*100:.2f}%]")
            print(f"Macro F1: {kt.get('macro_f1', 0):.4f} ({kt.get('delta_f1', 0):+.4f})  [95% CI: {kt.get('macro_f1_95ci')[0]:.4f} - {kt.get('macro_f1_95ci')[1]:.4f}]")
            print(f"McNemar Paired Test: chi2 = {mc.get('chi2_stat')}, exact p = {mc.get('p_value_exact_binomial')} (Significant at p < 0.01: {mc.get('is_significant_at_01')})")

        if "single_hand" in results:
            sh = results["single_hand"]
            st = sh.get("same_signer_held_out", {})
            print("\n--- 2. Single-Hand Baseline Model (63D, 26 Classes) ---")
            print(f"Same-Signer Held-Out Samples: {sh.get('test_samples')}")
            print(f"Accuracy:  {st.get('top1_accuracy', 0)*100:.2f}%  [95% CI: {st.get('top1_accuracy_95ci')[0]*100:.2f}% - {st.get('top1_accuracy_95ci')[1]*100:.2f}%]")
            print(f"Top-3 Acc: {st.get('top3_accuracy', 0)*100:.2f}%")
            print(f"Macro F1:  {st.get('macro_f1', 0):.4f}  [95% CI: {st.get('macro_f1_95ci')[0]:.4f} - {st.get('macro_f1_95ci')[1]:.4f}]")

        if "cnn" in results:
            cnn = results["cnn"]
            ct = cnn.get("same_signer_held_out", {})
            print("\n--- 3. RGB CNN Classifier (128x128x3, 26 Classes) ---")
            print(f"Same-Signer Held-Out Samples: {cnn.get('test_samples')}")
            print(f"Accuracy:  {ct.get('top1_accuracy', 0)*100:.2f}%  [95% CI: {ct.get('top1_accuracy_95ci')[0]*100:.2f}% - {ct.get('top1_accuracy_95ci')[1]*100:.2f}%]")
            print(f"Top-3 Acc: {ct.get('top3_accuracy', 0)*100:.2f}%")
            print(f"Macro F1:  {ct.get('macro_f1', 0):.4f}  [95% CI: {ct.get('macro_f1_95ci')[0]:.4f} - {ct.get('macro_f1_95ci')[1]:.4f}]")

        if "paired_fusion" in results:
            pf = results["paired_fusion"]
            print("\n--- 4. Paired Multi-Modal Fusion Ablation (243 Paired Samples) ---")
            print(f"Geometry-Only (Baseline): {pf['geometry_only_baseline']['accuracy']*100:.2f}% (F1: {pf['geometry_only_baseline']['macro_f1']})")
            print(f"Geometry-Only (+ K/T):    {pf['geometry_only_with_kt']['accuracy']*100:.2f}% (F1: {pf['geometry_only_with_kt']['macro_f1']})")
            print(f"CNN-Only (128x128x3):     {pf['cnn_only']['accuracy']*100:.2f}% (F1: {pf['cnn_only']['macro_f1']})")
            print(f"Multi-Modal Fused:        {pf['multi_modal_fused']['accuracy']*100:.2f}% (F1: {pf['multi_modal_fused']['macro_f1']})")

        if "mirror_parity" in results:
            mp = results["mirror_parity"]
            print("\n--- 5. Mirror Parity Evaluation Across Models ---")
            for m_name, m_data in mp.items():
                print(f"{m_name:<20}: Unmirrored = {m_data['unmirrored_accuracy']*100:.2f}% | Mirrored = {m_data['mirrored_accuracy']*100:.2f}% (Drop: {m_data['delta_drop']*100:+.2f}%)")

        if cm_path and cm_path.exists():
            print(f"\nConfusion matrix saved to: {cm_path}")
        print("=================================================================\n")


if __name__ == "__main__":
    main()
