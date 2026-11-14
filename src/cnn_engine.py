#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNNEngine: Image-Based Convolutional Neural Network Recognition Pathway.
=======================================================================
Implements the RGB image preprocessing and CNN inference pathway for ISL signs.

Features:
- Extracts square, aspect-ratio-preserved hand ROI bounding boxes from MediaPipe landmarks
- Preprocesses RGB crops (center padding, resize to 128x128, [-1.0, 1.0] normalization)
- Runs CNN inference, extracting top-5 predictions and probabilities for live debugging
- Transparent status reporting: If CNN weights are not present, reports OFFLINE honestly
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
POTENTIAL_CNN_PATHS = [
    WORKSPACE_ROOT / "models" / "cnn_sign_model.h5",
    WORKSPACE_ROOT / "models" / "experimental" / "cnn_sign_model.h5",
    WORKSPACE_ROOT / "models" / "cnn_model.h5",
]


class CNNEngine:
    """
    Manages the RGB image-based CNN inference pathway.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        confidence_threshold: float = 0.50,
        model_path: Optional[Path] = None,
    ):
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
        self.model_path = None
        self.classes: List[str] = [chr(65 + i) for i in range(26)]
        self.status = "OFFLINE"
        self.status_message = "No trained CNN checkpoint found"

        # Check for model path
        if model_path and Path(model_path).exists():
            self._load_model(Path(model_path))
        else:
            for p in POTENTIAL_CNN_PATHS:
                if p.exists():
                    self._load_model(p)
                    break

        if not self.is_loaded:
            self.status = "OFFLINE"
            self.status_message = (
                "CNN weights checkpoint not found on disk. "
                "Image preprocessing pipeline is operational; awaiting trained model weights."
            )

    def _load_model(self, path: Path) -> None:
        """Attempt to load a Keras CNN model."""
        try:
            from tensorflow.keras.models import load_model
            print(f"[CNNEngine] Loading CNN checkpoint from: {path}")
            self.model = load_model(path, compile=False, safe_mode=False)
            self.model_path = path
            self.is_loaded = True
            self.status = "ONLINE"
            self.status_message = f"CNN model active ({path.name})"

            # Try to infer classes from output shape
            if hasattr(self.model, "output_shape"):
                out_dim = self.model.output_shape[-1]
                if out_dim == 26:
                    self.classes = [chr(65 + i) for i in range(26)]
                elif out_dim == 20:
                    self.classes = [
                        "A", "B", "D", "E", "F", "G", "H", "I", "K", "M",
                        "N", "P", "Q", "R", "S", "T", "W", "X", "Y", "Z"
                    ]
                else:
                    self.classes = [f"Class_{i}" for i in range(out_dim)]

            # Warm up graph execution with dummy forward pass
            dummy_warmup = np.zeros((1, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            _ = self.model(dummy_warmup, training=False)
            print(f"[CNNEngine] Successfully initialized and warmed up CNN model with {len(self.classes)} classes.")
        except Exception as e:
            print(f"[CNNEngine Warning] Could not load CNN model: {e}")
            self.is_loaded = False
            self.status = "ERROR"
            self.status_message = f"Model load error: {e}"

    def extract_hand_roi(
        self,
        rgb_frame: np.ndarray,
        multi_hand_landmarks: Optional[List[Any]] = None,
        padding_ratio: float = 0.20
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Crop the hand region of interest from the RGB frame with aspect-ratio preservation.
        Ensures the extracted bounding box is made square before cropping to avoid squashing.
        """
        h, w = rgb_frame.shape[:2]

        if multi_hand_landmarks and len(multi_hand_landmarks) > 0:
            all_x, all_y = [], []
            for hand in multi_hand_landmarks:
                lms = hand.landmark if hasattr(hand, "landmark") else hand
                for lm in lms:
                    x = lm.x if hasattr(lm, "x") else lm[0]
                    y = lm.y if hasattr(lm, "y") else lm[1]
                    all_x.append(x * w)
                    all_y.append(y * h)

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            box_w = max_x - min_x
            box_h = max_y - min_y

            # Square aspect ratio preservation: use the larger dimension
            side = max(box_w, box_h)
            pad = side * padding_ratio
            half_side = (side + 2 * pad) / 2.0

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            x1 = max(0, int(center_x - half_side))
            y1 = max(0, int(center_y - half_side))
            x2 = min(w, int(center_x + half_side))
            y2 = min(h, int(center_y + half_side))

            if x2 > x1 and y2 > y1:
                crop = rgb_frame[y1:y2, x1:x2]
                return crop, (x1, y1, x2, y2)

        # Fallback: central 60% square region
        min_dim = min(h, w)
        cx, cy = w // 2, h // 2
        half = int(min_dim * 0.35)
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        crop = rgb_frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def preprocess_image(self, roi: np.ndarray) -> np.ndarray:
        """Resize and normalize ROI for CNN inference."""
        resized = cv2.resize(roi, self.target_size, interpolation=cv2.INTER_AREA)
        # Normalize to [-1.0, 1.0] range matching train_rgb_cnn_model.py
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return normalized

    def predict(
        self,
        rgb_frame: np.ndarray,
        multi_hand_landmarks: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute CNN prediction on the provided RGB frame.
        Returns top prediction, confidence, and top-5 probability distribution for diagnostics.
        """
        t0 = time.time()

        if rgb_frame is None or rgb_frame.size == 0:
            return {
                "pred_label": None,
                "confidence": 0.0,
                "top5": [],
                "status": "NO_FRAME",
                "is_available": False,
                "latency_ms": 0.0,
                "message": "RGB frame is empty",
            }

        roi, bbox = self.extract_hand_roi(rgb_frame, multi_hand_landmarks)
        preprocessed = self.preprocess_image(roi)

        if not self.is_loaded or self.model is None:
            latency_ms = (time.time() - t0) * 1000.0
            return {
                "pred_label": None,
                "confidence": 0.0,
                "top5": [],
                "status": "OFFLINE",
                "is_available": False,
                "roi_shape": roi.shape,
                "bbox": bbox,
                "latency_ms": round(latency_ms, 2),
                "message": self.status_message,
            }

        try:
            batch = np.expand_dims(preprocessed, axis=0)
            preds = self.model(batch, training=False).numpy()[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            pred_label = self.classes[pred_idx] if pred_idx < len(self.classes) else f"C_{pred_idx}"

            # Extract Top-5 predictions for live telemetry & diagnostics
            top5_indices = np.argsort(preds)[::-1][:5]
            top5 = [
                {
                    "class": self.classes[idx] if idx < len(self.classes) else f"C_{idx}",
                    "confidence": round(float(preds[idx]), 4)
                }
                for idx in top5_indices
            ]

            latency_ms = (time.time() - t0) * 1000.0

            return {
                "pred_label": pred_label,
                "confidence": round(confidence, 4),
                "top5": top5,
                "status": "ONLINE",
                "is_available": True,
                "roi_shape": roi.shape,
                "bbox": bbox,
                "latency_ms": round(latency_ms, 2),
                "message": "CNN inference successful",
            }
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            return {
                "pred_label": None,
                "confidence": 0.0,
                "top5": [],
                "status": "ERROR",
                "is_available": False,
                "latency_ms": round(latency_ms, 2),
                "message": f"Inference execution failed: {e}",
            }
