#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphabetEngine: Integrated Single & Dual-Hand Multi-Modal ISL Recognition.
==========================================================================
Coordinates:
- Single-Hand Recognition: Canonical 26-class MLP (A–Z)
- Dual-Hand Recognition: Prompt 23 20-class Geometric MLP (EXP-K-D-ABL 144D)
- Image-based RGB CNN Recognition: CNNEngine
- Decision Fusion: Confidence-aware FusionEngine
- Stateful Temporal Debouncing: 5-frame consistency filter, 0.8s hold cooldown
- WordBuffer: Live composition of fingerspelled letters into readable sentences
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
DEFAULT_SINGLE_MODEL_PATH = WORKSPACE_ROOT / "models" / "alphabet_model_26class.h5"
DEFAULT_SINGLE_LABEL_MAP_PATH = WORKSPACE_ROOT / "models" / "alphabet_label_map.json"

try:
    from cnn_engine import CNNEngine
    from dual_hand_engine import DualHandEngine
    from fusion_engine import FusionEngine
except ImportError:
    from src.cnn_engine import CNNEngine
    from src.dual_hand_engine import DualHandEngine
    from src.fusion_engine import FusionEngine


def validate_single_hand_features(feats: np.ndarray) -> Tuple[bool, str]:
    """Validate dimension and numerical finiteness of single-hand features."""
    if not isinstance(feats, np.ndarray) or feats.shape != (63,):
        return False, f"Expected shape (63,), got {getattr(feats, 'shape', None)}"
    if not np.all(np.isfinite(feats)):
        return False, "Contains non-finite values (NaN or Inf)"
    return True, "OK"


def validate_dual_hand_features(feats: np.ndarray, meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate dimension, numerical finiteness, and hand presence consistency of dual-hand features."""
    if not isinstance(feats, np.ndarray) or feats.shape != (126,):
        return False, f"Expected shape (126,), got {getattr(feats, 'shape', None)}"
    if not np.all(np.isfinite(feats)):
        return False, "Contains non-finite values (NaN or Inf)"
    if not meta.get("has_left", False) and not np.allclose(feats[:63], 0.0):
        return False, "Left hand flagged absent but has non-zero features"
    if not meta.get("has_right", False) and not np.allclose(feats[63:], 0.0):
        return False, "Right hand flagged absent but has non-zero features"
    return True, "OK"


def normalize_hand_landmarks(landmarks: Any) -> np.ndarray:
    """
    Exact mathematical normalization identical to Phase 5B / 5D / 5E:
    1. Translate relative to wrist: P'_i = P_i - P_0  (P'_0 becomes (0,0,0))
    2. Scale by Euclidean distance to middle finger MCP: s = ||P'_9||_2 + 1e-6
    3. Scale normalized: P''_i = P'_i / s
    4. Flatten into (63,) float32 vector.
    """
    if isinstance(landmarks, np.ndarray):
        coords = landmarks.copy().astype(np.float32)
    else:
        coords = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in landmarks], dtype=np.float32)
    wrist = coords[0:1, :].copy()
    translated = coords - wrist

    middle_mcp = translated[9]
    scale = float(np.linalg.norm(middle_mcp))
    if scale < 1e-6:
        scale = float(np.max(np.abs(translated))) + 1e-6
    scale = max(scale, 1e-4)

    normalized = translated / scale
    return normalized.flatten().astype(np.float32)


def _get_landmark_list(hand_obj: Any) -> List[Any]:
    """Retrieve iterable of 21 landmark items from NormalizedLandmarkList or list."""
    if hasattr(hand_obj, "landmark"):
        return list(hand_obj.landmark)
    return list(hand_obj)


def _get_coords_array(hand_obj: Any) -> np.ndarray:
    """Extract (21, 3) float32 array from hand landmark object."""
    if hand_obj is None:
        return np.zeros((21, 3), dtype=np.float32)
    lms = _get_landmark_list(hand_obj)
    if not lms or len(lms) < 21:
        return np.zeros((21, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in lms[:21]], dtype=np.float32)


def _get_wrist_x(hand_obj: Any) -> float:
    """Extract wrist x-coordinate for horizontal spatial ordering fallback."""
    lms = _get_landmark_list(hand_obj)
    if lms:
        first = lms[0]
        if hasattr(first, "x"):
            return float(first.x)
        elif isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0:
            return float(first[0])
    return 0.5


def _get_handedness_label_and_score(handedness_obj: Any) -> Tuple[str, float]:
    """Extract ('Left'|'Right'|'Unknown', score) from MediaPipe ClassificationList or dict."""
    if handedness_obj is None:
        return "Unknown", 0.0
    if hasattr(handedness_obj, "classification") and handedness_obj.classification:
        c = handedness_obj.classification[0]
        return str(getattr(c, "label", "Unknown")), float(getattr(c, "score", 0.0))
    elif isinstance(handedness_obj, dict):
        return str(handedness_obj.get("label", "Unknown")), float(handedness_obj.get("score", 0.0))
    elif isinstance(handedness_obj, str):
        return handedness_obj, 1.0
    return "Unknown", 0.0


def extract_dual_hand_landmarks(
    multi_hand_landmarks: Optional[List[Any]],
    multi_handedness: Optional[List[Any]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract deterministic dual-hand representations from MediaPipe Hands.
    Returns:
    - left_feats: (63,) normalized single-hand representation
    - right_feats: (63,) normalized single-hand representation
    - combined_126: (126,) concatenated representation
    - metadata: dict with hand_count, c1 (21,3), c2 (21,3), left_score, right_score
    """
    if not multi_hand_landmarks or len(multi_hand_landmarks) == 0:
        return (
            np.zeros(63, dtype=np.float32),
            np.zeros(63, dtype=np.float32),
            np.zeros(126, dtype=np.float32),
            {
                "hand_count": 0,
                "has_left": False,
                "has_right": False,
                "c1": np.zeros((21, 3), dtype=np.float32),
                "c2": np.zeros((21, 3), dtype=np.float32),
                "left_score": 0.0,
                "right_score": 0.0,
                "is_ambiguous": False
            }
        )

    if len(multi_hand_landmarks) == 1:
        lms = _get_landmark_list(multi_hand_landmarks[0])
        feats = normalize_hand_landmarks(lms)
        coords = _get_coords_array(multi_hand_landmarks[0])

        lbl, score = ("Unknown", 0.0)
        if multi_handedness and len(multi_handedness) > 0:
            lbl, score = _get_handedness_label_and_score(multi_handedness[0])

        if lbl == "Left":
            left_feats = feats
            right_feats = np.zeros(63, dtype=np.float32)
            meta = {
                "hand_count": 1,
                "has_left": True,
                "has_right": False,
                "c1": coords,
                "c2": np.zeros((21, 3), dtype=np.float32),
                "left_score": score,
                "right_score": 0.0,
                "is_ambiguous": False
            }
        else:
            left_feats = np.zeros(63, dtype=np.float32)
            right_feats = feats
            meta = {
                "hand_count": 1,
                "has_left": False,
                "has_right": True,
                "c1": np.zeros((21, 3), dtype=np.float32),
                "c2": coords,
                "left_score": 0.0,
                "right_score": score,
                "is_ambiguous": False
            }
        combined_126 = np.concatenate([left_feats, right_feats])
        return left_feats, right_feats, combined_126, meta

    # Exactly 2 hands detected
    hand_0 = multi_hand_landmarks[0]
    hand_1 = multi_hand_landmarks[1]

    # In mirrored camera view and Prompt 22/23 feature contracts:
    # Channel 1 (c1) is strictly the screen-left hand (wrist x smaller)
    # Channel 2 (c2) is strictly the screen-right hand (wrist x larger)
    x_0 = _get_wrist_x(hand_0)
    x_1 = _get_wrist_x(hand_1)
    if x_0 <= x_1:
        left_hand, right_hand = hand_0, hand_1
        left_idx, right_idx = 0, 1
    else:
        left_hand, right_hand = hand_1, hand_0
        left_idx, right_idx = 1, 0

    left_lbl, left_score = (
        _get_handedness_label_and_score(multi_handedness[left_idx])
        if (multi_handedness and len(multi_handedness) > left_idx)
        else ("Unknown", 0.0)
    )
    right_lbl, right_score = (
        _get_handedness_label_and_score(multi_handedness[right_idx])
        if (multi_handedness and len(multi_handedness) > right_idx)
        else ("Unknown", 0.0)
    )

    c1 = _get_coords_array(left_hand)
    c2 = _get_coords_array(right_hand)
    left_feats = normalize_hand_landmarks(_get_landmark_list(left_hand))
    right_feats = normalize_hand_landmarks(_get_landmark_list(right_hand))
    combined_126 = np.concatenate([left_feats, right_feats])

    is_ambiguous = (left_lbl == right_lbl or left_lbl == "Unknown" or right_lbl == "Unknown")

    meta = {
        "hand_count": 2,
        "has_left": True,
        "has_right": True,
        "c1": c1,
        "c2": c2,
        "left_score": left_score,
        "right_score": right_score,
        "is_ambiguous": is_ambiguous
    }
    return left_feats, right_feats, combined_126, meta


class WordBuffer:
    """
    Stateful Word and Sentence buffer for fingerspelling text construction.
    """

    def __init__(self):
        self.current_word: str = ""
        self.sentence: str = ""

    def add_letter(self, letter: str) -> None:
        """Append a valid uppercase character to current word."""
        if letter and len(letter) == 1 and letter.isalpha():
            self.current_word += letter.upper()

    def add_space(self) -> None:
        """Commit current word and add space to sentence."""
        if self.current_word:
            self.commit_word()
        elif self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "

    def backspace(self) -> None:
        """Delete last character from current word or sentence."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence:
            self.sentence = self.sentence[:-1]

    def clear_word(self) -> None:
        """Clear current word buffer."""
        self.current_word = ""

    def clear_sentence(self) -> None:
        """Clear accumulated sentence text."""
        self.sentence = ""

    def clear_all(self) -> None:
        """Clear both current word and sentence."""
        self.current_word = ""
        self.sentence = ""

    def get_current_word(self) -> str:
        return self.current_word

    def get_sentence(self) -> str:
        return self.sentence

    def get_full_text(self) -> str:
        if self.sentence and self.current_word:
            if self.sentence.endswith(" "):
                return self.sentence + self.current_word
            return f"{self.sentence} {self.current_word}"
        elif self.sentence:
            return self.sentence
        return self.current_word

    def commit_word(self) -> None:
        if self.current_word:
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self.sentence += self.current_word + " "
            self.current_word = ""


class AlphabetEngine:
    """
    Comprehensive Real-Time Recognition Engine for Indian Sign Language Alphabet.
    Seamlessly dispatches single-hand vs dual-hand gestures, executes multi-modal
    CNN + Landmark fusion, debounces predictions, and accumulates fingerspelled words.
    """

    def __init__(
        self,
        single_model_path: Path = DEFAULT_SINGLE_MODEL_PATH,
        single_label_map_path: Path = DEFAULT_SINGLE_LABEL_MAP_PATH,
        confidence_threshold: float = 0.65,
        stability_frames: int = 5,
        cooldown_seconds: float = 0.8,
        word_buffer: Optional[WordBuffer] = None,
        tts_engine: Optional[Any] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_seconds = cooldown_seconds

        self.word_buffer = word_buffer if word_buffer is not None else WordBuffer()
        self.tts_engine = tts_engine
        self.target_language = "English"

        # 1. Single-Hand Landmark Model (26 Classes A-Z)
        self.single_model = None
        self.single_label_map = {}
        self.inv_single_label_map = {}
        self.model = None
        self.label_map = {}
        self.inv_label_map = {}
        self._load_single_model(single_model_path, single_label_map_path)

        # 2. Dual-Hand Landmark Engine (20 Classes, Prompt 23 144D model)
        self.dual_hand_engine = DualHandEngine(config_key="config_d_ablation_144d", use_prompt23=True)

        # 3. Image-Based CNN Engine
        self.cnn_engine = CNNEngine()

        # 4. Multi-Modal Fusion Engine
        self.fusion_engine = FusionEngine()

        # State Machine Variables
        self.raw_prediction: Optional[str] = None
        self.raw_confidence: float = 0.0
        self.stable_letter: Optional[str] = None
        self.last_stable_letter: Optional[str] = None
        self.emitted_letter: Optional[str] = None
        self.state: str = "IDLE"

        self.consecutive_candidate: Optional[str] = None
        self.consecutive_count: int = 0
        self.last_emitted_letter: Optional[str] = None
        self.last_emitted_time: float = 0.0
        self.has_released_same_letter: bool = True
        self.emission_counter: int = 0

        # Detailed Telemetry Cache
        self._last_telemetry: Dict[str, Any] = {}

    def _load_single_model(self, model_path: Path, label_map_path: Path) -> None:
        """Load canonical 26-class single-hand model."""
        if not Path(model_path).exists():
            print(f"[AlphabetEngine Warning] Single-hand model missing at: {model_path}")
            return
        try:
            print(f"[AlphabetEngine] Loading 26-class single-hand model from: {model_path}")
            self.single_model = load_model(model_path, compile=False, safe_mode=False)
            if Path(label_map_path).exists():
                with open(label_map_path, "r") as f:
                    self.single_label_map = json.load(f)
                self.inv_single_label_map = {v: k for k, v in self.single_label_map.items()}
            else:
                self.inv_single_label_map = {i: chr(65 + i) for i in range(26)}
            self.model = self.single_model
            self.label_map = self.single_label_map
            self.inv_label_map = self.inv_single_label_map
            print(f"[AlphabetEngine] Single-hand model ready ({len(self.inv_single_label_map)} classes).")
        except Exception as e:
            print(f"[AlphabetEngine Error] Could not load single-hand model: {e}")

    def reset_debounce_state(self) -> None:
        """Reset debouncing counters without modifying word buffer."""
        self.raw_prediction = None
        self.raw_confidence = 0.0
        self.stable_letter = None
        self.emitted_letter = None
        self.consecutive_candidate = None
        self.consecutive_count = 0
        self.has_released_same_letter = True
        self.state = "RESET"

    def reset_all(self) -> None:
        """Reset debounce state and clear word buffer."""
        self.reset_debounce_state()
        self.word_buffer.clear_all()
        self.last_emitted_letter = None
        self.last_stable_letter = None
        self.last_emitted_time = 0.0
        self.state = "IDLE"

    def set_target_language(self, language: str) -> None:
        self.target_language = language

    def space(self) -> Dict[str, Any]:
        self.word_buffer.add_space()
        return self.get_state_snapshot()

    def backspace(self) -> Dict[str, Any]:
        self.word_buffer.backspace()
        return self.get_state_snapshot()

    def commit_word(self) -> Dict[str, Any]:
        self.word_buffer.commit_word()
        return self.get_state_snapshot()

    def clear_word(self) -> Dict[str, Any]:
        self.word_buffer.clear_word()
        return self.get_state_snapshot()

    def clear_sentence(self) -> Dict[str, Any]:
        self.word_buffer.clear_sentence()
        return self.get_state_snapshot()

    def clear_all(self) -> Dict[str, Any]:
        self.word_buffer.clear_all()
        return self.get_state_snapshot()

    def reset(self) -> Dict[str, Any]:
        self.reset_debounce_state()
        return self.get_state_snapshot()

    def process_dual_landmarks(
        self,
        multi_hand_landmarks: Optional[List[Any]],
        multi_handedness: Optional[List[Any]] = None,
        rgb_frame: Optional[np.ndarray] = None,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process incoming frame, extract landmarks, run single/dual landmark & CNN pathways,
        fuse outputs, debounce predictions, and update word buffer.
        """
        now = current_time if current_time is not None else time.time()
        self.emitted_letter = None

        # 1. Feature Extraction from MediaPipe Hands
        left_feats, right_feats, combined_126, meta = extract_dual_hand_landmarks(
            multi_hand_landmarks, multi_handedness
        )
        hand_count = meta["hand_count"]

        # Case A: 0 Hands Detected
        if hand_count == 0:
            self.reset_debounce_state()
            self.state = "NO_HAND"
            self._last_telemetry = {
                "hand_detected": False,
                "hand_count": 0,
                "hand_status": "NO HAND",
                "active_pipeline": "NONE",
                "landmark_prediction": None,
                "landmark_confidence": 0.0,
                "cnn_prediction": None,
                "cnn_confidence": 0.0,
                "cnn_status": self.cnn_engine.status,
                "fused_prediction": None,
                "fused_confidence": 0.0,
                "fusion_mode": "NO_HAND",
                "fusion_reason": "No hands visible",
            }
            return self._build_status(hand_detected=False)

        # Case B: 1 Hand Detected -> Single-Hand 26-Class Pipeline
        elif hand_count == 1:
            single_feats = left_feats if meta["has_left"] else right_feats
            single_coords = meta["c1"] if meta["has_left"] else meta["c2"]
            landmark_pred_label = None
            landmark_conf = 0.0
            lm_top5 = []

            if self.single_model is not None:
                batch = single_feats.reshape(1, 63).astype(np.float32)
                probs = self.single_model(batch, training=False).numpy()[0]
                idx = int(np.argmax(probs))
                landmark_conf = float(probs[idx])
                landmark_pred_label = self.inv_single_label_map.get(idx, "--")

                top5_idx = np.argsort(probs)[::-1][:5]
                lm_top5 = [
                    {"class": self.inv_single_label_map.get(i, f"C_{i}"), "confidence": round(float(probs[i]), 4)}
                    for i in top5_idx
                ]

                # Physical Hand Geometry Checks (I/Y and U/V)
                if len(single_coords) >= 21 and not np.allclose(single_coords, 0.0):
                    wrist = single_coords[0]
                    scale = float(np.linalg.norm(single_coords[9] - wrist))
                    if scale > 1e-4:
                        pinky_ext = float(np.linalg.norm(single_coords[20] - single_coords[17])) / scale
                        thumb_dist = float(np.linalg.norm(single_coords[4] - single_coords[5])) / scale
                        index_ext = float(np.linalg.norm(single_coords[8] - single_coords[5])) / scale
                        middle_ext = float(np.linalg.norm(single_coords[12] - single_coords[9])) / scale

                        # 1. I vs Y: Pinky up + folded index & middle
                        if pinky_ext > 0.8 and index_ext < 0.6 and middle_ext < 0.6:
                            if thumb_dist < 0.85:
                                landmark_pred_label = "I"
                                landmark_conf = max(landmark_conf, 0.88)
                            elif thumb_dist >= 1.15:
                                landmark_pred_label = "Y"
                                landmark_conf = max(landmark_conf, 0.88)

                        # 2. U vs V: Index + Middle up
                        if index_ext > 0.8 and middle_ext > 0.8:
                            tip_dist = float(np.linalg.norm(single_coords[8] - single_coords[12])) / scale
                            if tip_dist < 0.38 and landmark_pred_label in ["V", "H", "B", "U"]:
                                landmark_pred_label = "U"
                                landmark_conf = max(landmark_conf, 0.88)
                            elif tip_dist >= 0.42 and landmark_pred_label in ["U", "H", "B", "V"]:
                                landmark_pred_label = "V"
                                landmark_conf = max(landmark_conf, 0.88)

            landmark_res = {
                "pred_label": landmark_pred_label,
                "confidence": landmark_conf,
                "top5": lm_top5,
                "status": "OK",
                "model_name": "26-Class Single-Hand MLP",
            }

            # Run CNN Pathway if RGB frame is available
            cnn_res = None
            if rgb_frame is not None:
                cnn_res = self.cnn_engine.predict(rgb_frame, multi_hand_landmarks)

            # Fuse Landmark & CNN Pathways
            fusion_res = self.fusion_engine.fuse(landmark_res, cnn_res, hand_detected=True)

            active_pipeline = "Single-Hand (26 Classes)"
            hand_status_str = "1 HAND"

        # Case C: 2 Hands Detected -> Dual-Hand 20-Class Pipeline (EXP-K-D-ABL 144D)
        else:
            if self.dual_hand_engine is None:
                self.consecutive_count = 0
                self.consecutive_candidate = None
                self.stable_letter = None
                self.raw_prediction = None
                self.raw_confidence = 0.0
                self.state = "2 HANDS — DUAL-HAND MODEL UNAVAILABLE"
                self._last_telemetry = {
                    "hand_detected": True,
                    "hand_count": 2,
                    "hand_status": "2 HANDS",
                    "active_pipeline": "Dual-Hand (Unavailable)",
                    "landmark_prediction": None,
                    "landmark_confidence": 0.0,
                    "landmark_top5": [],
                    "cnn_prediction": None,
                    "cnn_confidence": 0.0,
                    "cnn_top5": [],
                    "fused_prediction": None,
                    "fused_confidence": 0.0,
                    "fusion_mode": "NONE",
                    "is_accepted": False,
                }
                return self._build_status(hand_detected=True)

            c1 = meta["c1"]  # Screen-left hand (21, 3)
            c2 = meta["c2"]  # Screen-right hand (21, 3)

            landmark_res = self.dual_hand_engine.predict(c1, c2)
            landmark_pred_label = landmark_res.get("pred_label")
            landmark_conf = float(landmark_res.get("confidence", 0.0))
            lm_top5 = landmark_res.get("top5", [])

            # 1. Unimanual Override:
            # Check if one hand is actively gesturing a strictly unimanual letter (C, I, L, O, U, V)
            unimanual_signs = {"C", "I", "L", "O", "U", "V"}
            best_single_label = None
            best_single_conf = 0.0

            if self.single_model is not None:
                for hand_coords in [c1, c2]:
                    if not np.allclose(hand_coords, 0.0):
                        feats = normalize_hand_landmarks(hand_coords)
                        batch = feats.reshape(1, 63).astype(np.float32)
                        probs = self.single_model(batch, training=False).numpy()[0]
                        s_idx = int(np.argmax(probs))
                        s_conf = float(probs[s_idx])
                        s_lbl = self.inv_single_label_map.get(s_idx, "")

                        # Apply single hand physical geometry
                        wrist = hand_coords[0]
                        scale = float(np.linalg.norm(hand_coords[9] - wrist))
                        if scale > 1e-4:
                            pinky_ext = float(np.linalg.norm(hand_coords[20] - hand_coords[17])) / scale
                            thumb_dist = float(np.linalg.norm(hand_coords[4] - hand_coords[5])) / scale
                            index_ext = float(np.linalg.norm(hand_coords[8] - hand_coords[5])) / scale
                            middle_ext = float(np.linalg.norm(hand_coords[12] - hand_coords[9])) / scale

                            if pinky_ext > 0.8 and index_ext < 0.6 and middle_ext < 0.6:
                                if thumb_dist < 0.85:
                                    s_lbl = "I"
                                    s_conf = max(s_conf, 0.85)
                                elif thumb_dist >= 1.15:
                                    s_lbl = "Y"
                                    s_conf = max(s_conf, 0.85)

                            if index_ext > 0.8 and middle_ext > 0.8:
                                tip_dist = float(np.linalg.norm(hand_coords[8] - hand_coords[12])) / scale
                                if tip_dist < 0.38 and s_lbl in ["V", "H", "B", "U"]:
                                    s_lbl = "U"
                                    s_conf = max(s_conf, 0.85)
                                elif tip_dist >= 0.42 and s_lbl in ["U", "H", "B", "V"]:
                                    s_lbl = "V"
                                    s_conf = max(s_conf, 0.85)

                        if s_lbl in unimanual_signs and s_conf > best_single_conf:
                            best_single_label = s_lbl
                            best_single_conf = s_conf

            if best_single_label and best_single_conf >= 0.70:
                if landmark_conf < 0.65 or landmark_pred_label in ["B", "H", "A", "T"]:
                    landmark_res = {
                        "pred_label": best_single_label,
                        "confidence": round(best_single_conf, 4),
                        "top5": [{"class": best_single_label, "confidence": round(best_single_conf, 4)}],
                        "status": "OK",
                        "model_name": "26-Class Single-Hand (Unimanual Override)",
                    }
                    landmark_pred_label = best_single_label
                    landmark_conf = best_single_conf

            # 2. ISL Class K vs Class T Disambiguation:
            # In ISL, K has passive index finger pointing vertically upward (stem).
            # In T, passive index is horizontal.
            if landmark_pred_label == "T" and len(c1) >= 21:
                idx_tip_y = c1[8, 1]
                idx_mcp_y = c1[5, 1]
                idx_tip_x = c1[8, 0]
                idx_mcp_x = c1[5, 0]
                dy = idx_mcp_y - idx_tip_y  # positive when pointing up
                dx = abs(idx_tip_x - idx_mcp_x)
                if dy > 0.08 and dy > 1.2 * dx:
                    landmark_pred_label = "K"
                    landmark_res["pred_label"] = "K"
                    landmark_res["confidence"] = max(landmark_conf, 0.88)
                    landmark_conf = landmark_res["confidence"]

            # 3. Physical validation for B and H to prevent spurious open-hand collapses:
            if landmark_pred_label in ["B", "H"]:
                wrist_dist = float(np.linalg.norm(c1[0] - c2[0]))
                if wrist_dist > 0.42:
                    landmark_res["confidence"] = round(landmark_conf * 0.3, 4)
                    landmark_conf = float(landmark_res["confidence"])
                    if landmark_conf < self.confidence_threshold:
                        landmark_res["pred_label"] = None
                        landmark_pred_label = None

                # For B, hands must form loops (thumb and index touching)
                if landmark_pred_label == "B":
                    a1 = float(np.linalg.norm(c1[4] - c1[8]))
                    a2 = float(np.linalg.norm(c2[4] - c2[8]))
                    if a1 > 0.16 or a2 > 0.16:
                        landmark_res["confidence"] = round(landmark_conf * 0.3, 4)
                        landmark_conf = float(landmark_res["confidence"])
                        if landmark_conf < self.confidence_threshold:
                            landmark_res["pred_label"] = None
                            landmark_pred_label = None

            # Run CNN Pathway if RGB frame is available
            cnn_res = None
            if rgb_frame is not None:
                cnn_res = self.cnn_engine.predict(rgb_frame, multi_hand_landmarks)

            # Fuse Landmark & CNN Pathways
            fusion_res = self.fusion_engine.fuse(landmark_res, cnn_res, hand_detected=True)

            active_pipeline = f"Dual-Hand (20 Classes, {self.dual_hand_engine.config_key})"
            hand_status_str = "2 HANDS"

        # 2. Update Telemetry Cache
        self._last_telemetry = {
            "hand_detected": True,
            "hand_count": hand_count,
            "hand_status": hand_status_str,
            "active_pipeline": active_pipeline,
            "landmark_prediction": landmark_res.get("pred_label"),
            "landmark_confidence": landmark_res.get("confidence", 0.0),
            "landmark_top5": landmark_res.get("top5", []),
            "cnn_prediction": cnn_res.get("pred_label") if cnn_res else None,
            "cnn_confidence": cnn_res.get("confidence", 0.0) if cnn_res else 0.0,
            "cnn_top5": cnn_res.get("top5", []) if cnn_res else [],
            "cnn_status": cnn_res.get("status") if cnn_res else self.cnn_engine.status,
            "fused_prediction": fusion_res.get("fused_label"),
            "fused_confidence": fusion_res.get("confidence", 0.0),
            "fusion_mode": fusion_res.get("fusion_mode"),
            "fusion_reason": fusion_res.get("reason"),
            "is_accepted": fusion_res.get("is_accepted", False),
        }

        # 3. Feed Fused Prediction to Debounce State Machine
        self.raw_prediction = fusion_res.get("fused_label")
        self.raw_confidence = float(fusion_res.get("confidence", 0.0))

        if self.raw_prediction and self.raw_confidence >= self.confidence_threshold:
            if self.raw_prediction == self.consecutive_candidate:
                self.consecutive_count += 1
            else:
                self.consecutive_candidate = self.raw_prediction
                self.consecutive_count = 1
                if self.consecutive_candidate != self.last_emitted_letter:
                    self.has_released_same_letter = True

            if self.consecutive_count >= self.stability_frames:
                self.stable_letter = self.consecutive_candidate
                self.last_stable_letter = self.consecutive_candidate
                time_since_last_emission = now - self.last_emitted_time
                can_emit = False

                if self.stable_letter == self.last_emitted_letter:
                    if self.has_released_same_letter and (time_since_last_emission >= self.cooldown_seconds):
                        can_emit = True
                else:
                    if time_since_last_emission >= self.cooldown_seconds:
                        can_emit = True

                if can_emit:
                    self.emission_counter += 1
                    self.word_buffer.add_letter(self.stable_letter)
                    self.emitted_letter = self.stable_letter
                    self.last_emitted_letter = self.stable_letter
                    self.last_emitted_time = now
                    self.has_released_same_letter = False
                    self.state = f"EMITTED({self.stable_letter})"
                else:
                    if self.stable_letter == self.last_emitted_letter and not self.has_released_same_letter:
                        self.state = f"HOLDING({self.stable_letter})"
                    else:
                        remaining_cooldown = max(0.0, self.cooldown_seconds - time_since_last_emission)
                        self.state = f"COOLDOWN({remaining_cooldown:.1f}s)"
            else:
                self.stable_letter = None
                self.state = f"STABILIZING({self.consecutive_count}/{self.stability_frames})"
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)
            if self.consecutive_count == 0:
                self.consecutive_candidate = None
                self.stable_letter = None
                self.has_released_same_letter = True
            if hand_count == 2:
                self.state = "HOLDING_DUAL_HAND (2 HANDS)"
                self.raw_prediction = None
            else:
                self.state = "LOW_CONFIDENCE" if self.raw_prediction else "NEUTRAL"

        return self._build_status(hand_detected=True)

    def process_features(
        self,
        features: Optional[np.ndarray],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Direct 63D feature vector evaluation matching unit test harnesses."""
        now = current_time if current_time is not None else time.time()
        self.emitted_letter = None

        if features is None:
            self.reset_debounce_state()
            self.state = "NO_HAND"
            return self._build_status(hand_detected=False)

        model_to_use = getattr(self, "model", None) or self.single_model
        inv_map = getattr(self, "inv_label_map", None) or self.inv_single_label_map

        if model_to_use is not None and features is not None:
            batch = features.reshape(1, -1).astype(np.float32)
            probs = model_to_use(batch, training=False).numpy()[0]
            idx = int(np.argmax(probs))
            self.raw_confidence = float(probs[idx])
            self.raw_prediction = inv_map.get(idx, None)
        else:
            self.raw_prediction = None
            self.raw_confidence = 0.0

        if self.raw_prediction and self.raw_confidence >= self.confidence_threshold:
            if self.raw_prediction == self.consecutive_candidate:
                self.consecutive_count += 1
            else:
                self.consecutive_candidate = self.raw_prediction
                self.consecutive_count = 1
                if self.consecutive_candidate != self.last_emitted_letter:
                    self.has_released_same_letter = True

            if self.consecutive_count >= self.stability_frames:
                self.stable_letter = self.consecutive_candidate
                self.last_stable_letter = self.consecutive_candidate
                time_since_last_emission = now - self.last_emitted_time
                can_emit = False
                if self.stable_letter == self.last_emitted_letter:
                    if self.has_released_same_letter and (time_since_last_emission >= self.cooldown_seconds):
                        can_emit = True
                else:
                    if time_since_last_emission >= self.cooldown_seconds:
                        can_emit = True

                if can_emit:
                    self.emission_counter += 1
                    self.word_buffer.add_letter(self.stable_letter)
                    self.emitted_letter = self.stable_letter
                    self.last_emitted_letter = self.stable_letter
                    self.last_emitted_time = now
                    self.has_released_same_letter = False
                    self.state = f"EMITTED({self.stable_letter})"
                else:
                    if self.stable_letter == self.last_emitted_letter and not self.has_released_same_letter:
                        self.state = f"HOLDING({self.stable_letter})"
                    else:
                        rem = max(0.0, self.cooldown_seconds - time_since_last_emission)
                        self.state = f"COOLDOWN({rem:.1f}s)"
            else:
                self.stable_letter = None
                self.state = f"STABILIZING({self.consecutive_count}/{self.stability_frames})"
        else:
            self.consecutive_count = max(0, self.consecutive_count - 1)
            if self.consecutive_count == 0:
                self.consecutive_candidate = None
                self.stable_letter = None
                self.has_released_same_letter = True
            self.state = "LOW_CONFIDENCE" if self.raw_prediction else "NEUTRAL"

        return self._build_status(hand_detected=True)

    def process_landmarks(
        self,
        landmarks: Any,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Backward-compatible single-hand landmark interface."""
        if landmarks is None:
            return self.process_dual_landmarks(None, None, current_time=current_time)
        return self.process_dual_landmarks([landmarks], None, current_time=current_time)

    def speak_sentence(self, language: str = "English") -> bool:
        """Speak composed sentence via configured TTS engine."""
        if self.tts_engine is None:
            return False
        sentence = self.word_buffer.get_sentence().strip()
        if not sentence:
            return False
        return self.tts_engine.speak(sentence, language)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return full snapshot including buffer and telemetry."""
        return self._build_status(hand_detected=(self.state != "NO_HAND"))

    def _build_status(self, hand_detected: bool) -> Dict[str, Any]:
        """Assemble comprehensive engine status dictionary."""
        status = {
            "hand_detected": hand_detected,
            "hand_count": self._last_telemetry.get("hand_count", 1 if hand_detected else 0),
            "hand_status": self._last_telemetry.get("hand_status", "1 HAND" if hand_detected else "NO HAND"),
            "active_pipeline": self._last_telemetry.get("active_pipeline", "Landmark"),
            "raw_prediction": self.raw_prediction,
            "raw_letter": self.raw_prediction,
            "raw_confidence": self.raw_confidence,
            "stable_letter": self.stable_letter,
            "last_stable_letter": self.last_stable_letter,
            "stability_progress": round(min(1.0, self.consecutive_count / max(1, self.stability_frames)), 2),
            "consecutive_count": self.consecutive_count,
            "stability_frames": self.stability_frames,
            "emitted_letter": self.emitted_letter,
            "emission_id": self.emission_counter,
            "state": self.state,
            "current_word": self.word_buffer.get_current_word(),
            "sentence": self.word_buffer.get_sentence(),
            "full_text": self.word_buffer.get_full_text(),
            # Multi-Modal Pathway Telemetry & Live Diagnostics
            "landmark_prediction": self._last_telemetry.get("landmark_prediction"),
            "landmark_confidence": self._last_telemetry.get("landmark_confidence", 0.0),
            "landmark_top5": self._last_telemetry.get("landmark_top5", []),
            "top5": self._last_telemetry.get("landmark_top5", []),
            "cnn_prediction": self._last_telemetry.get("cnn_prediction"),
            "cnn_confidence": self._last_telemetry.get("cnn_confidence", 0.0),
            "cnn_top5": self._last_telemetry.get("cnn_top5", []),
            "cnn_status": self._last_telemetry.get("cnn_status", self.cnn_engine.status),
            "fused_prediction": self._last_telemetry.get("fused_prediction"),
            "fused_confidence": self._last_telemetry.get("fused_confidence", 0.0),
            "fusion_mode": self._last_telemetry.get("fusion_mode", "LANDMARK_ONLY"),
            "fusion_reason": self._last_telemetry.get("fusion_reason", ""),
            "is_accepted": self._last_telemetry.get("is_accepted", False),
        }
        return status
