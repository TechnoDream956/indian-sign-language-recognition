#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DualHandEngine: High-Accuracy Geometric Classifier for Bimanual ISL Alphabet.
=============================================================================
Integrates the validated Prompt 23 (and Prompt 17B) dual-hand neural models
into the production inference pipeline.

Supported Configurations:
- Config D-Ablation (144D, Default Best): 129D base + 3D aperture + 4D palm prox + 8D finger extension
- Config D-Minimal  (140D): 129D base + 3D aperture + 4D palm prox + 4D min finger extension
- Config C          (132D): 129D base + 3D aperture
- Config A          (129D): 126D shape + 3D relative wrist vector

Classes (20 Bimanual ISL Letters):
['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z']
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Default best checkpoint: Prompt 23 Config D-Ablation (144D)
DEFAULT_P23_DIR = WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k"
DEFAULT_P17_DIR = WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt17b"

CLASSES_20 = [
    "A", "B", "D", "E", "F", "G", "H", "I", "K", "M",
    "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
]


def normalize_single_hand_coords(coords: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Scale and translation normalization for a single hand (21, 3).
    Wrist (P_0) becomes origin (0, 0, 0).
    Scale is Euclidean distance to middle finger MCP (landmark 9).
    """
    if coords is None or len(coords) < 21:
        return np.zeros(63, dtype=np.float32), 1.0, np.zeros(3, dtype=np.float32)

    coords = np.array(coords, dtype=np.float32)
    wrist = coords[0:1, :].copy()
    translated = coords - wrist
    middle_mcp = translated[9]
    scale = float(np.linalg.norm(middle_mcp))
    if scale < 1e-6:
        scale = float(np.max(np.abs(translated))) + 1e-6
    scale = max(scale, 1e-4)

    normalized = translated / scale
    return normalized.flatten().astype(np.float32), scale, wrist.flatten()


def compute_dual_hand_features(
    c1: np.ndarray,
    c2: np.ndarray,
    config_key: str = "config_d_ablation_144d"
) -> np.ndarray:
    """
    Extract exact geometric features matching training specifications:
    c1: Screen-Left Hand coordinates (21, 3)
    c2: Screen-Right Hand coordinates (21, 3)
    """
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)

    has_c1 = not np.allclose(c1, 0.0)
    has_c2 = not np.allclose(c2, 0.0)
    has_both = has_c1 and has_c2

    # 1. Base 126D Shape + 3D Relative Wrist
    c1_63, s1, wrist_1 = normalize_single_hand_coords(c1)
    c2_63, s2, wrist_2 = normalize_single_hand_coords(c2)

    s_ref = max((s1 + s2) / 2.0, 1e-4) if has_both else (s1 if has_c1 else s2)
    rel_wrist_3 = ((wrist_2 - wrist_1) / s_ref).astype(np.float32) if has_both else np.zeros(3, dtype=np.float32)

    shape_126 = np.concatenate([c1_63, c2_63]).astype(np.float32)
    feat_129 = np.concatenate([shape_126, rel_wrist_3]).astype(np.float32)

    if config_key == "config_a_129d":
        return feat_129

    # 2. 3D Aperture Features:
    # a1: Thumb(4) to Index(8) distance in c1
    # a2: Thumb(4) to Index(8) distance in c2
    # a_cross: Min cross-hand thumb-to-index distance in screen plane
    a1 = float(np.linalg.norm(c1[4] - c1[8])) / s1 if has_c1 else 0.0
    a2 = float(np.linalg.norm(c2[4] - c2[8])) / s2 if has_c2 else 0.0
    if has_both:
        d_12 = float(np.linalg.norm(c1[4, :2] - c2[8, :2])) / s_ref
        d_21 = float(np.linalg.norm(c2[4, :2] - c1[8, :2])) / s_ref
        a_cross = min(d_12, d_21)
    else:
        a_cross = 0.0
    aperture_3 = np.array([a1, a2, a_cross], dtype=np.float32)
    feat_132 = np.concatenate([feat_129, aperture_3]).astype(np.float32)

    if config_key == "config_c_132d":
        return feat_132

    # 3. Supplemental D Features:
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    palm_landmarks = [0, 5, 9, 17]  # Wrist, Index MCP, Middle MCP, Pinky MCP

    c1_palm_2d = np.mean(c1[palm_landmarks, :2], axis=0) if has_c1 else np.zeros(2, dtype=np.float32)
    c2_palm_2d = np.mean(c2[palm_landmarks, :2], axis=0) if has_c2 else np.zeros(2, dtype=np.float32)

    # 4D Symmetric Palm Proximity
    palm_prox_4d = np.zeros(4, dtype=np.float32)
    if has_both:
        for t_idx, t in enumerate(tips):
            d1 = float(np.linalg.norm(c1[t, :2] - c2_palm_2d))
            d2 = float(np.linalg.norm(c2[t, :2] - c1_palm_2d))
            palm_prox_4d[t_idx] = min(d1, d2) / s_ref

    # Finger Extension relative to Middle MCP (landmark 9)
    s1_clamped = max(s1, 1e-4)
    s2_clamped = max(s2, 1e-4)
    ext1_4d = (np.linalg.norm(c1[tips] - c1[9], axis=1) / s1_clamped).astype(np.float32) if has_c1 else np.zeros(4, dtype=np.float32)
    ext2_4d = (np.linalg.norm(c2[tips] - c2[9], axis=1) / s2_clamped).astype(np.float32) if has_c2 else np.zeros(4, dtype=np.float32)

    min_ext_4d = np.minimum(ext1_4d, ext2_4d).astype(np.float32)

    if config_key == "config_d_minimal_140d":
        return np.concatenate([feat_132, palm_prox_4d, min_ext_4d]).astype(np.float32)

    # 8D Sorted Finger Extension for config_d_ablation_144d
    sorted_ext_8d = np.zeros(8, dtype=np.float32)
    for t_idx in range(4):
        sorted_ext_8d[2 * t_idx] = min(ext1_4d[t_idx], ext2_4d[t_idx])
        sorted_ext_8d[2 * t_idx + 1] = max(ext1_4d[t_idx], ext2_4d[t_idx])

    return np.concatenate([feat_132, palm_prox_4d, sorted_ext_8d]).astype(np.float32)


class DualHandEngine:
    """
    Manages dual-hand neural network inference using validated checkpoints.
    Default: Prompt 23 Config D-Ablation 144D model (EXP-K-D-ABL).
    """

    def __init__(
        self,
        config_key: str = "config_d_ablation_144d",
        use_prompt23: bool = True,
        custom_model_path: Optional[Path] = None,
    ):
        self.config_key = config_key
        self.use_prompt23 = use_prompt23
        self.classes = CLASSES_20
        self.num_classes = len(self.classes)
        self.inv_label_map = {i: c for i, c in enumerate(self.classes)}
        self.model = None
        self.is_loaded = False
        self.expected_dim = {
            "config_a_129d": 129,
            "config_c_132d": 132,
            "config_d_minimal_140d": 140,
            "config_d_ablation_144d": 144,
        }.get(config_key, 144)

        # Resolve model path
        if custom_model_path and Path(custom_model_path).exists():
            self.model_path = Path(custom_model_path)
        else:
            base_dir = DEFAULT_P23_DIR if use_prompt23 else DEFAULT_P17_DIR
            self.model_path = base_dir / config_key / "best_val_model.h5"

        self._load_model()

    def _load_model(self) -> None:
        """Load the Keras model with safe_mode=False for Lambda layer compatibility."""
        if not self.model_path.exists():
            print(f"[DualHandEngine Warning] Model file not found at: {self.model_path}")
            self.is_loaded = False
            return

        try:
            print(f"[DualHandEngine] Loading dual-hand model from: {self.model_path}")
            self.model = load_model(self.model_path, compile=False, safe_mode=False)
            self.is_loaded = True
            print(f"[DualHandEngine] Successfully loaded {self.config_key} ({self.expected_dim}D -> {self.num_classes} classes).")
        except Exception as e:
            print(f"[DualHandEngine Error] Failed to load dual-hand model: {e}")
            self.is_loaded = False

    def predict(
        self,
        c1: Optional[np.ndarray],
        c2: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run forward inference on two detected hands.
        c1: Screen-Left Hand (21, 3)
        c2: Screen-Right Hand (21, 3)

        Returns dictionary with:
        - pred_label: Predicted character (e.g. 'K', 'T', 'B') or None
        - confidence: Float confidence [0.0, 1.0]
        - probabilities: Dict mapping each of the 20 classes to its probability
        - feature_vector: Numpy array of extracted features
        - latency_ms: Execution time in milliseconds
        - is_valid: True if model is loaded and input valid
        """
        t0 = time.time()

        if not self.is_loaded or self.model is None:
            return {
                "pred_label": None,
                "confidence": 0.0,
                "probabilities": {},
                "latency_ms": 0.0,
                "is_valid": False,
                "status": "MODEL_NOT_LOADED",
            }

        if c1 is None or c2 is None or len(c1) < 21 or len(c2) < 21:
            return {
                "pred_label": None,
                "confidence": 0.0,
                "probabilities": {},
                "latency_ms": (time.time() - t0) * 1000.0,
                "is_valid": False,
                "status": "MISSING_HANDS",
            }

        # Extract features
        features = compute_dual_hand_features(c1, c2, self.config_key)
        if features.shape[0] != self.expected_dim:
            return {
                "pred_label": None,
                "confidence": 0.0,
                "probabilities": {},
                "latency_ms": (time.time() - t0) * 1000.0,
                "is_valid": False,
                "status": f"DIM_MISMATCH (got {features.shape[0]}, expected {self.expected_dim})",
            }

        # Forward pass
        feats_batch = features.reshape(1, -1).astype(np.float32)
        logits_or_probs = self.model(feats_batch, training=False).numpy()[0]

        # Ensure probabilities sum to 1
        exp_vals = np.exp(logits_or_probs - np.max(logits_or_probs))
        probs = exp_vals / np.sum(exp_vals) if np.max(logits_or_probs) > 1.01 or np.min(logits_or_probs) < -0.01 else logits_or_probs

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.inv_label_map.get(pred_idx, "--")

        prob_dict = {self.inv_label_map[i]: float(probs[i]) for i in range(len(self.classes))}
        top5_indices = np.argsort(probs)[::-1][:5]
        top5 = [
            {"class": self.inv_label_map[i], "confidence": round(float(probs[i]), 4)}
            for i in top5_indices
        ]
        latency_ms = (time.time() - t0) * 1000.0

        return {
            "pred_label": pred_label,
            "confidence": round(confidence, 4),
            "top5": top5,
            "probabilities": prob_dict,
            "feature_vector": features,
            "latency_ms": round(latency_ms, 2),
            "is_valid": True,
            "status": "OK",
            "model_config": self.config_key,
            "classes_supported": self.num_classes,
        }
