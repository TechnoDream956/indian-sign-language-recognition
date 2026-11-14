#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial 129-Dimensional Dual-Hand Feature Extractor for ISL Recognition.

Phase 4 Innovation:
Extends the 126-dimensional internal landmark representation to 129 dimensions
by appending the normalized 3D relative wrist displacement vector:
    X_129 = [ LeftHand_63, RightHand_63, RelativeWrist_3 ]

Where:
    RelativeWrist = (RightWrist - LeftWrist) / ReferenceScale
    ReferenceScale = (Scale_Left + Scale_Right) / 2.0

Key Properties Enforced:
1. Scale Invariance: Divided by average hand dimension (wrist-to-middle-MCP),
   yielding a dimensionless hand-length distance vector invariant to camera zoom.
2. Consistent Hand Ordering: Left hand at [0:63], Right hand at [63:126],
   Relative displacement (from Left to Right) at [126:129].
3. Deterministic Spatial Fallback: Handles MediaPipe classification ambiguity
   or identity swapping gracefully via horizontal wrist coordinate sorting.
4. Robust Single-Hand / Missing-Hand Behavior: Deterministically zero-pads
   absent hands and displacement without producing NaNs or Infinities.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def _get_landmark_list(hand_obj: Any) -> List[Any]:
    """Retrieve iterable of 21 landmark items from either NormalizedLandmarkList or array/list."""
    if hasattr(hand_obj, "landmark"):
        return list(hand_obj.landmark)
    return list(hand_obj)

def _get_wrist_coords(hand_obj: Any) -> np.ndarray:
    """Extract (x, y, z) coordinates of the wrist landmark (index 0)."""
    lms = _get_landmark_list(hand_obj)
    if not lms:
        return np.zeros(3, dtype=np.float32)
    first = lms[0]
    if hasattr(first, "x"):
        return np.array([first.x, first.y, getattr(first, "z", 0.0)], dtype=np.float32)
    elif isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 3:
        return np.array([first[0], first[1], first[2]], dtype=np.float32)
    elif isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2:
        return np.array([first[0], first[1], 0.0], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)

def _get_handedness_label_and_score(handedness_obj: Any) -> Tuple[str, float]:
    """Extract ('Left'|'Right'|'Unknown', score) from MediaPipe ClassificationList or dict."""
    if handedness_obj is None:
        return "Unknown", 0.0
    if hasattr(handedness_obj, "classification") and handedness_obj.classification:
        c = handedness_obj.classification[0]
        label = str(getattr(c, "label", "Unknown"))
        score = float(getattr(c, "score", 0.0))
        return label, score
    elif isinstance(handedness_obj, dict):
        return str(handedness_obj.get("label", "Unknown")), float(handedness_obj.get("score", 0.0))
    elif isinstance(handedness_obj, str):
        return handedness_obj, 1.0
    return "Unknown", 0.0

def normalize_single_hand(landmarks: Any) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Normalizes a single hand's 21 landmarks:
    1. Translates relative to wrist: P'_i = P_i - P_wrist
    2. Computes hand scale: s = ||P'_9 (middle MCP)||_2
    3. Normalizes coords: P''_i = P'_i / s
    
    Returns:
    - normalized_63: (63,) float32 array
    - scale: float scale factor
    - raw_wrist: (3,) float32 raw coordinate array
    """
    lms = _get_landmark_list(landmarks)
    if hasattr(lms[0], "x"):
        coords = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms], dtype=np.float32)
    else:
        coords = np.array(lms, dtype=np.float32)
        if coords.shape == (21, 2):
            coords = np.hstack([coords, np.zeros((21, 1), dtype=np.float32)])
            
    raw_wrist = coords[0].copy()
    translated = coords - raw_wrist
    
    middle_mcp = translated[9]
    scale = float(np.linalg.norm(middle_mcp))
    if scale < 1e-6:
        scale = float(np.max(np.abs(translated))) + 1e-6
    if scale < 1e-6:
        scale = 1.0
        
    normalized = translated / scale
    return normalized.flatten().astype(np.float32), scale, raw_wrist

def extract_spatial_129d_landmarks(
    multi_hand_landmarks: Optional[List[Any]],
    multi_handedness: Optional[List[Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract deterministic 129-dimensional spatial-aware dual-hand representations.
    
    Returns:
    - left_features: (63,) float32 array
    - right_features: (63,) float32 array
    - relative_wrist: (3,) float32 array: (Wrist_Right - Wrist_Left) / RefScale
    - combined_129: (129,) float32 array: [Left(63), Right(63), RelativeWrist(3)]
    - metadata: dict with hand_count, has_left, has_right, is_ambiguous, scale_left, scale_right
    """
    if not multi_hand_landmarks or len(multi_hand_landmarks) == 0:
        left_feats = np.zeros(63, dtype=np.float32)
        right_feats = np.zeros(63, dtype=np.float32)
        rel_wrist = np.zeros(3, dtype=np.float32)
        combined_129 = np.zeros(129, dtype=np.float32)
        meta = {
            "hand_count": 0,
            "has_left": False,
            "has_right": False,
            "is_ambiguous": False,
            "scale_left": 0.0,
            "scale_right": 0.0,
            "reference_scale": 1.0
        }
        return left_feats, right_feats, rel_wrist, combined_129, meta

    if len(multi_hand_landmarks) == 1:
        feats, scale, wrist = normalize_single_hand(multi_hand_landmarks[0])
        lbl, score = ("Unknown", 0.0)
        if multi_handedness and len(multi_handedness) > 0:
            lbl, score = _get_handedness_label_and_score(multi_handedness[0])

        rel_wrist = np.zeros(3, dtype=np.float32)
        if lbl == "Left":
            left_feats = feats
            right_feats = np.zeros(63, dtype=np.float32)
            meta = {
                "hand_count": 1,
                "has_left": True,
                "has_right": False,
                "is_ambiguous": False,
                "scale_left": scale,
                "scale_right": 0.0,
                "reference_scale": scale
            }
        else:
            left_feats = np.zeros(63, dtype=np.float32)
            right_feats = feats
            meta = {
                "hand_count": 1,
                "has_left": False,
                "has_right": True,
                "is_ambiguous": False,
                "scale_left": 0.0,
                "scale_right": scale,
                "reference_scale": scale
            }
            
        combined_129 = np.concatenate([left_feats, right_feats, rel_wrist]).astype(np.float32)
        return left_feats, right_feats, rel_wrist, combined_129, meta

    # Two or more hands detected
    hand_0 = multi_hand_landmarks[0]
    hand_1 = multi_hand_landmarks[1]

    lbl_0, scr_0 = _get_handedness_label_and_score(multi_handedness[0]) if (multi_handedness and len(multi_handedness) > 0) else ("Unknown", 0.0)
    lbl_1, scr_1 = _get_handedness_label_and_score(multi_handedness[1]) if (multi_handedness and len(multi_handedness) > 1) else ("Unknown", 0.0)

    is_ambiguous = False
    if lbl_0 == "Left" and lbl_1 == "Right":
        left_hand, right_hand = hand_0, hand_1
    elif lbl_0 == "Right" and lbl_1 == "Left":
        left_hand, right_hand = hand_1, hand_0
    else:
        # Ambiguous handedness or identical labels
        # Fallback to horizontal spatial ordering: smaller wrist x is viewer-left (Left)
        is_ambiguous = True
        w0 = _get_wrist_coords(hand_0)
        w1 = _get_wrist_coords(hand_1)
        if w0[0] <= w1[0]:
            left_hand, right_hand = hand_0, hand_1
        else:
            left_hand, right_hand = hand_1, hand_0

    left_feats, scale_l, wrist_l = normalize_single_hand(left_hand)
    right_feats, scale_r, wrist_r = normalize_single_hand(right_hand)

    ref_scale = max((scale_l + scale_r) / 2.0, 1e-4)
    rel_wrist = ((wrist_r - wrist_l) / ref_scale).astype(np.float32)

    combined_129 = np.concatenate([left_feats, right_feats, rel_wrist]).astype(np.float32)
    meta = {
        "hand_count": 2,
        "has_left": True,
        "has_right": True,
        "is_ambiguous": is_ambiguous,
        "scale_left": scale_l,
        "scale_right": scale_r,
        "reference_scale": ref_scale
    }
    return left_feats, right_feats, rel_wrist, combined_129, meta

def validate_spatial_129d_features(
    combined_129: np.ndarray,
    meta: Dict[str, Any]
) -> Tuple[bool, str]:
    """Validate dimensionality, finiteness, and consistency of 129D features."""
    if not isinstance(combined_129, np.ndarray):
        return False, "Feature vector is not a numpy array"
    if combined_129.shape != (129,):
        return False, f"Expected shape (129,), got {combined_129.shape}"
    if not np.all(np.isfinite(combined_129)):
        return False, "Feature vector contains non-finite values (NaN or Inf)"

    has_left = meta.get("has_left", False)
    has_right = meta.get("has_right", False)
    hand_count = meta.get("hand_count", 0)

    left_part = combined_129[:63]
    right_part = combined_129[63:126]
    rel_wrist = combined_129[126:129]

    if hand_count == 0:
        if not np.allclose(combined_129, 0.0):
            return False, "0 hands detected but feature vector is non-zero"
    elif hand_count == 1:
        if has_left and not np.allclose(right_part, 0.0):
            return False, "Right hand absent but right feature vector is non-zero"
        if has_right and not np.allclose(left_part, 0.0):
            return False, "Left hand absent but left feature vector is non-zero"
        if not np.allclose(rel_wrist, 0.0):
            return False, "Single hand present but relative wrist displacement is non-zero"
    else:  # hand_count >= 2
        if np.allclose(left_part, 0.0) or np.allclose(right_part, 0.0):
            return False, "2 hands detected but one hand vector is all zeros"

    return True, "OK"

