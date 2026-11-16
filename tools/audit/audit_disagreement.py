#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Disagreement Numbers Between Geometry and CNN Classifiers.
Calculates exact agreement and disagreement statistics on paired test samples (N=243 and N=232).
"""

from __future__ import annotations
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.dual_hand_engine import compute_dual_hand_features
from src.fusion_engine import FusionEngine

DUAL_CLASSES_20 = [
    "A", "B", "D", "E", "F", "G", "H", "I", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
]
ALPHABET_CLASSES_26 = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def run_audit():
    dh_model = tf.keras.models.load_model(
        WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k" / "config_d_ablation_144d" / "best_val_model.h5",
        compile=False,
        safe_mode=False
    )
    cnn_model = tf.keras.models.load_model(WORKSPACE_ROOT / "models" / "cnn_sign_model.h5", compile=False)

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

    # 1. Predictions
    i2c_20 = {i: c for i, c in enumerate(DUAL_CLASSES_20)}
    X_feats = np.array([compute_dual_hand_features(l, r, "config_d_ablation_144d") for l, r in zip(raw_left, raw_right)])
    probs_geom = dh_model.predict(X_feats, verbose=0)
    preds_geom_clean = np.array([i2c_20[i] for i in np.argmax(probs_geom, axis=1)])

    preds_geom_kt = np.array(preds_geom_clean)
    for i in range(len(preds_geom_kt)):
        if preds_geom_kt[i] == "T" and len(raw_left[i]) >= 21:
            dy = raw_left[i][5, 1] - raw_left[i][8, 1]
            dx = abs(raw_left[i][8, 0] - raw_left[i][5, 0])
            if dy > 0.08 and dy > 1.2 * dx:
                preds_geom_kt[i] = "K"

    probs_cnn = cnn_model.predict(X_imgs, verbose=0)
    preds_cnn = np.array([ALPHABET_CLASSES_26[i] for i in np.argmax(probs_cnn, axis=1)])
    y_true = np.array(labels)

    def analyze(name: str, y_geom: np.ndarray, y_cnn: np.ndarray, y_t: np.ndarray):
        n = len(y_t)
        agree_mask = (y_geom == y_cnn)
        disagree_mask = ~agree_mask

        n_agree = int(np.sum(agree_mask))
        agree_correct = int(np.sum((y_geom == y_t) & agree_mask))
        agree_acc = agree_correct / n_agree if n_agree > 0 else 0.0

        n_disagree = int(np.sum(disagree_mask))
        geom_right = int(np.sum((y_geom == y_t) & disagree_mask))
        cnn_right = int(np.sum((y_cnn == y_t) & disagree_mask))
        neither_right = int(np.sum((y_geom != y_t) & (y_cnn != y_t) & disagree_mask))

        total_geom_correct = int(np.sum(y_geom == y_t))
        total_cnn_correct = int(np.sum(y_cnn == y_t))

        print(f"==================================================")
        print(f"ANALYSIS: {name} (N = {n})")
        print(f"==================================================")
        print(f"Overall Geometry Correct: {total_geom_correct}/{n} ({total_geom_correct/n*100:.2f}%)")
        print(f"Overall CNN Correct:      {total_cnn_correct}/{n} ({total_cnn_correct/n*100:.2f}%)")
        print(f"--- Agreement ---")
        print(f"Agree Count:              {n_agree} / {n} ({n_agree/n*100:.2f}%)")
        print(f"Agree Correct:            {agree_correct} / {n_agree} ({agree_acc*100:.2f}%)")
        print(f"--- Disagreement ---")
        print(f"Disagree Count:           {n_disagree} / {n} ({n_disagree/n*100:.2f}%)")
        print(f"  Geometry Right:         {geom_right} ({geom_right/n_disagree*100:.2f}% of disagree)")
        print(f"  CNN Right:              {cnn_right} ({cnn_right/n_disagree*100:.2f}% of disagree)")
        print(f"  Neither Right:          {neither_right} ({neither_right/n_disagree*100:.2f}% of disagree)")
        print(f"--- Identity Checks ---")
        print(f"  agree_correct + geom_right = {agree_correct} + {geom_right} = {agree_correct + geom_right} (Expected: {total_geom_correct}) -> {'MATCH' if agree_correct + geom_right == total_geom_correct else 'MISMATCH'}")
        print(f"  agree_correct + cnn_right  = {agree_correct} + {cnn_right} = {agree_correct + cnn_right} (Expected: {total_cnn_correct}) -> {'MATCH' if agree_correct + cnn_right == total_cnn_correct else 'MISMATCH'}")
        print(f"  geom_right + cnn_right + neither = {geom_right} + {cnn_right} + {neither_right} = {geom_right + cnn_right + neither_right} (Expected: {n_disagree}) -> {'MATCH' if geom_right + cnn_right + neither_right == n_disagree else 'MISMATCH'}")
        print()

    # (A) Full N=243: Clean Geometry vs CNN
    analyze("(A) Clean Geometry vs CNN (N=243)", preds_geom_clean, preds_cnn, y_true)

    # (B) Full N=243: K/T-Overridden Geometry vs CNN
    analyze("(B) K/T-Overridden Geometry vs CNN (N=243, *possibly tuned on test data)", preds_geom_kt, preds_cnn, y_true)

    # Near-duplicate filtering (1-NN distance < 0.10)
    # The 11 near duplicates were identified in Phase 5B:
    # 232 clean samples:
    X_train_raw = np.load(WORKSPACE_ROOT / "reserve" / "unused_scripts" / "research" / "phase5b1_features" / "phase5b2_prompt22_class_k_features" / "config_d_ablation_144d" / "X_train.npy")
    from scipy.spatial.distance import cdist
    dists = cdist(X_feats, X_train_raw, metric="euclidean")
    min_dists = np.min(dists, axis=1)
    clean_232_mask = (min_dists >= 0.10)

    if np.sum(clean_232_mask) == 232:
        analyze("(A2) Clean Geometry vs CNN (Deduplicated N=232)", preds_geom_clean[clean_232_mask], preds_cnn[clean_232_mask], y_true[clean_232_mask])
        analyze("(B2) K/T-Overridden Geometry vs CNN (Deduplicated N=232, *possibly tuned)", preds_geom_kt[clean_232_mask], preds_cnn[clean_232_mask], y_true[clean_232_mask])



if __name__ == "__main__":
    run_audit()
