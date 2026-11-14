#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FusionEngine: Confidence-Aware Multi-Modal Decision Fusion.
==========================================================
Combines the RGB CNN recognition pathway with the geometric Landmark pathway.

Key Architecture:
- Calibrated confidence evaluation across both streams.
- Landmark-First Geometric Prior: MediaPipe landmarks provide coordinate
  invariance to background, skin tone, clothing, and room lighting.
  When landmarks have high tracking certainty (confidence >= 0.60), they serve
  as the primary ground truth.
- Multi-Modal Agreement Bonus: When both CNN and Landmark agree on the sign,
  confidence is boosted: C_fused = min(1.0, 0.65 * C_landmark + 0.35 * C_cnn + 0.10).
- Robust Disagreement Arbitration:
  * When Landmark is confident (>= 0.60), Landmark is preserved as dominant
    even if the CNN predicts a different class (preventing an inaccurate CNN from corrupting correct landmarks).
  * Disagreements are clearly recorded in telemetry for transparent live debugging.
  * When both streams are below threshold (< 0.50), emission is rejected (status: REJECTED).
"""

from typing import Any, Dict, List, Optional, Tuple


class FusionEngine:
    """
    Confidence-aware, multi-modal decision fuser for Indian Sign Language recognition.
    """

    def __init__(
        self,
        landmark_weight: float = 0.65,
        cnn_weight: float = 0.35,
        min_confidence_threshold: float = 0.50,
        agreement_bonus: float = 0.10,
    ):
        self.landmark_weight = landmark_weight
        self.cnn_weight = cnn_weight
        self.min_confidence_threshold = min_confidence_threshold
        self.agreement_bonus = agreement_bonus

    def fuse(
        self,
        landmark_result: Dict[str, Any],
        cnn_result: Optional[Dict[str, Any]] = None,
        hand_detected: bool = False,
    ) -> Dict[str, Any]:
        """
        Fuse predictions from Landmark pathway and CNN pathway.

        landmark_result: Dict with 'pred_label', 'confidence', 'status', 'hand_count', 'top5'
        cnn_result: Dict with 'pred_label', 'confidence', 'status', 'is_available', 'top5'
        hand_detected: Bool indicating if MediaPipe tracking locked onto hands.
        """
        if not hand_detected:
            return {
                "fused_label": None,
                "confidence": 0.0,
                "fusion_mode": "NO_HAND",
                "status": "NO_HAND",
                "is_accepted": False,
                "reason": "No hands detected in camera frame",
                "landmark_label": None,
                "landmark_confidence": 0.0,
                "landmark_top5": [],
                "cnn_label": None,
                "cnn_confidence": 0.0,
                "cnn_top5": [],
                "agreement": False,
            }

        lm_label = landmark_result.get("pred_label")
        lm_conf = float(landmark_result.get("confidence", 0.0))
        lm_top5 = landmark_result.get("top5", [])

        cnn_available = cnn_result is not None and cnn_result.get("is_available", False)
        cnn_label = cnn_result.get("pred_label") if cnn_available else None
        cnn_conf = float(cnn_result.get("confidence", 0.0)) if cnn_available else 0.0
        cnn_top5 = cnn_result.get("top5", []) if cnn_available else []

        # Case 1: CNN is OFFLINE / Unavailable -> Transparent Landmark-Only Mode
        if not cnn_available:
            accepted = lm_label is not None and lm_conf >= self.min_confidence_threshold
            return {
                "fused_label": lm_label if accepted else None,
                "confidence": lm_conf,
                "fusion_mode": "LANDMARK_ONLY (CNN Offline)",
                "status": "ACCEPTED" if accepted else "REJECTED_LOW_CONFIDENCE",
                "is_accepted": accepted,
                "reason": "CNN weights offline; relying on geometric landmark classifier",
                "landmark_label": lm_label,
                "landmark_confidence": lm_conf,
                "landmark_top5": lm_top5,
                "cnn_label": None,
                "cnn_confidence": 0.0,
                "cnn_top5": [],
                "agreement": None,
            }

        # Case 2: Multi-Modal Agreement
        if lm_label is not None and lm_label == cnn_label:
            fused_conf = min(
                1.0,
                self.landmark_weight * lm_conf + self.cnn_weight * cnn_conf + self.agreement_bonus
            )
            accepted = fused_conf >= self.min_confidence_threshold
            return {
                "fused_label": lm_label if accepted else None,
                "confidence": round(fused_conf, 4),
                "fusion_mode": "MULTI_MODAL_AGREEMENT",
                "status": "AGREEMENT" if accepted else "REJECTED_LOW_CONFIDENCE",
                "is_accepted": accepted,
                "reason": f"Both landmark ({lm_conf:.2f}) and CNN ({cnn_conf:.2f}) agree on '{lm_label}'",
                "landmark_label": lm_label,
                "landmark_confidence": lm_conf,
                "landmark_top5": lm_top5,
                "cnn_label": cnn_label,
                "cnn_confidence": cnn_conf,
                "cnn_top5": cnn_top5,
                "agreement": True,
            }

        # Case 3: Disagreement Handling with Landmark Geometric Authority
        lm_is_confident = lm_label is not None and lm_conf >= self.min_confidence_threshold
        cnn_is_confident = cnn_label is not None and cnn_conf >= 0.85

        if lm_is_confident:
            # Landmark is confident: prioritize landmark physical geometry
            # Do NOT allow an inaccurate CNN to override or block a confident landmark!
            return {
                "fused_label": lm_label,
                "confidence": lm_conf,
                "fusion_mode": "LANDMARK_PRIORITIZED",
                "status": "ACCEPTED",
                "is_accepted": True,
                "reason": f"Geometric landmark confident ('{lm_label}' {lm_conf:.2f}); CNN suggests '{cnn_label}' ({cnn_conf:.2f})",
                "landmark_label": lm_label,
                "landmark_confidence": lm_conf,
                "landmark_top5": lm_top5,
                "cnn_label": cnn_label,
                "cnn_confidence": cnn_conf,
                "cnn_top5": cnn_top5,
                "agreement": False,
            }

        if cnn_is_confident and not lm_is_confident:
            # Landmark is uncertain, but CNN has very high confidence (>= 0.85)
            return {
                "fused_label": cnn_label,
                "confidence": cnn_conf,
                "fusion_mode": "CNN_PRIORITIZED",
                "status": "ACCEPTED",
                "is_accepted": True,
                "reason": f"CNN strongly confident ('{cnn_label}' {cnn_conf:.2f}) while landmark uncertain ({lm_conf:.2f})",
                "landmark_label": lm_label,
                "landmark_confidence": lm_conf,
                "landmark_top5": lm_top5,
                "cnn_label": cnn_label,
                "cnn_confidence": cnn_conf,
                "cnn_top5": cnn_top5,
                "agreement": False,
            }

        # Case 4: Both streams are uncertain -> Reject prediction cleanly
        return {
            "fused_label": None,
            "confidence": max(lm_conf, cnn_conf),
            "fusion_mode": "REJECTED_UNCERTAIN",
            "status": "REJECTED",
            "is_accepted": False,
            "reason": f"Uncertain input: Landmark ({lm_conf:.2f}) and CNN ({cnn_conf:.2f}) both below threshold",
            "landmark_label": lm_label,
            "landmark_confidence": lm_conf,
            "landmark_top5": lm_top5,
            "cnn_label": cnn_label,
            "cnn_confidence": cnn_conf,
            "cnn_top5": cnn_top5,
            "agreement": False,
        }
