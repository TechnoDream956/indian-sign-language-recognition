"""
Quality Filter and Landmark Validation.
========================================
Validates landmark dimensionality, numerical integrity, coordinate bounds,
tracking confidence, label correctness, and duplicate detection.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    FEATURE_SCHEMA_SINGLE_HAND,
    FEATURE_SCHEMA_DUAL_HAND,
    MIN_LANDMARK_SCORE,
    MAX_NORMALIZED_COORD_VALUE,
    MIN_SAMPLE_QUALITY_SCORE,
    VALID_ALPHABET_CLASSES,
    DUPLICATE_DISTANCE_EPSILON,
)


class QualityAssessment:
    """Result of quality validation checks."""

    def __init__(
        self,
        is_valid: bool,
        quality_score: float,
        reasons: List[str],
        schema: str,
        is_duplicate: bool = False,
    ):
        self.is_valid = is_valid
        self.quality_score = float(quality_score)
        self.reasons = reasons
        self.schema = schema
        self.is_duplicate = is_duplicate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 4),
            "reasons": self.reasons,
            "schema": self.schema,
            "is_duplicate": self.is_duplicate,
        }


class QualityFilter:
    """Rigorous gatekeeper for landmark and correction validation."""

    def __init__(
        self,
        min_score: float = MIN_LANDMARK_SCORE,
        max_coord: float = MAX_NORMALIZED_COORD_VALUE,
        min_quality: float = MIN_SAMPLE_QUALITY_SCORE,
        duplicate_epsilon: float = DUPLICATE_DISTANCE_EPSILON,
    ):
        self.min_score = min_score
        self.max_coord = max_coord
        self.min_quality = min_quality
        self.duplicate_epsilon = duplicate_epsilon

    def validate_label(self, label: str) -> Tuple[bool, str]:
        """Validate if label is a valid Indian Sign Language alphabet (A-Z)."""
        clean_lbl = (label or "").strip().upper()
        if not clean_lbl:
            return False, "Label is empty"
        if len(clean_lbl) != 1 or clean_lbl not in VALID_ALPHABET_CLASSES:
            return False, f"Invalid alphabet label '{label}'. Must be single letter A-Z."
        return True, "OK"

    def assess_landmarks(
        self,
        features: np.ndarray,
        schema: str = FEATURE_SCHEMA_SINGLE_HAND,
        raw_landmarks: Optional[List[Dict[str, Any]]] = None,
        tracking_quality: Optional[Dict[str, Any]] = None,
        existing_samples: Optional[List[np.ndarray]] = None,
    ) -> QualityAssessment:
        """
        Comprehensive quality assessment of a landmark sample:
        1. Dimension and shape validation
        2. Finiteness check (no NaN / Inf)
        3. Coordinate bounds check
        4. Geometry variance check (not all zeroes or flat)
        5. Tracking score check
        6. Near-duplicate detection
        """
        reasons: List[str] = []
        is_valid = True
        quality_score = 1.0

        if not isinstance(features, np.ndarray):
            try:
                features = np.array(features, dtype=np.float32)
            except Exception as e:
                return QualityAssessment(
                    is_valid=False,
                    quality_score=0.0,
                    reasons=[f"Features could not be converted to numpy array: {e}"],
                    schema=schema,
                )

        # 1. Dimension check
        if schema == FEATURE_SCHEMA_SINGLE_HAND:
            if features.shape != (63,):
                return QualityAssessment(
                    is_valid=False,
                    quality_score=0.0,
                    reasons=[f"Expected shape (63,) for single-hand, got {features.shape}"],
                    schema=schema,
                )
        elif schema == FEATURE_SCHEMA_DUAL_HAND:
            if features.shape != (126,):
                return QualityAssessment(
                    is_valid=False,
                    quality_score=0.0,
                    reasons=[f"Expected shape (126,) for dual-hand, got {features.shape}"],
                    schema=schema,
                )
        else:
            return QualityAssessment(
                is_valid=False,
                quality_score=0.0,
                reasons=[f"Unsupported schema '{schema}'"],
                schema=schema,
            )

        # 2. Finiteness check
        if not np.all(np.isfinite(features)):
            return QualityAssessment(
                is_valid=False,
                quality_score=0.0,
                reasons=["Feature vector contains NaN or Infinite values"],
                schema=schema,
            )

        # 3. Flat / all zeros check
        norm_val = float(np.linalg.norm(features))
        if norm_val < 1e-4:
            return QualityAssessment(
                is_valid=False,
                quality_score=0.0,
                reasons=["Feature vector is empty or all zeros"],
                schema=schema,
            )

        # 4. Extreme outlier bounds check
        max_abs = float(np.max(np.abs(features)))
        if max_abs > self.max_coord:
            quality_score -= 0.3
            reasons.append(f"Extreme coordinate excursion detected: max={max_abs:.2f} > {self.max_coord}")
            if max_abs > self.max_coord * 2.0:
                is_valid = False

        # 5. Tracking quality check
        if tracking_quality:
            t_score = tracking_quality.get("score")
            if t_score is not None:
                t_score = float(t_score)
                if t_score < self.min_score:
                    quality_score -= 0.25
                    reasons.append(f"Tracking confidence {t_score:.2f} below threshold {self.min_score}")
                    if t_score < 0.2:
                        is_valid = False

        # 6. Anatomical sanity check (wrist at 0,0,0 relative to first landmark)
        if schema == FEATURE_SCHEMA_SINGLE_HAND:
            wrist = features[:3]
            if np.linalg.norm(wrist) > 1e-3:
                # In standard normalized hand features, wrist point is origin (0,0,0)
                quality_score -= 0.15
                reasons.append("Wrist origin deviation from reference anchor")

        # 7. Duplicate / near-duplicate check against existing samples
        is_duplicate = False
        if existing_samples and len(existing_samples) > 0:
            for s in existing_samples:
                if s.shape == features.shape:
                    dist = float(np.linalg.norm(features - s))
                    if dist < self.duplicate_epsilon:
                        is_duplicate = True
                        is_valid = False
                        quality_score = 0.0
                        reasons.append(f"Near-duplicate sample detected (L2 distance {dist:.4f} < {self.duplicate_epsilon})")
                        break

        # Final quality threshold
        quality_score = max(0.0, min(1.0, quality_score))
        if quality_score < self.min_quality:
            is_valid = False
            reasons.append(f"Overall quality score {quality_score:.2f} below minimum {self.min_quality}")

        return QualityAssessment(
            is_valid=is_valid,
            quality_score=quality_score,
            reasons=reasons if reasons else ["Quality checks passed"],
            schema=schema,
            is_duplicate=is_duplicate,
        )
