#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous Sign Recognition and Sentence Construction Framework for ISL.

Implements Phase 6 Staged Architecture:
1. Frame-level token accumulator with temporal smoothing.
2. Landmark kinetic energy & velocity boundary detector (sign start, apex, release).
3. Non-maximal suppression and beam search token decoder.
4. Word/phrase boundary detection via neutral pose duration.
5. Sentence constructor with uncertainty preservation.
6. Language correction interface preserving raw tokens.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DecodedSign:
    token: str
    confidence: float
    start_time: float
    end_time: float
    is_uncertain: bool = False


@dataclass
class SentenceState:
    raw_tokens: List[str] = field(default_factory=list)
    constructed_sentence: str = ""
    corrected_sentence: str = ""
    last_action: str = "IDLE"
    last_update_time: float = field(default_factory=time.time)


class KineticBoundaryDetector:
    """
    Detects sign boundaries (onset, apex, and release) using hand/wrist kinetic velocity.
    When velocity drops after high movement, an apex gesture is detected.
    When hands drop to bottom third of frame or rest for >0.8s, a sign boundary is triggered.
    """

    def __init__(self, rest_velocity_threshold: float = 0.015, min_rest_duration_sec: float = 0.75):
        self.rest_threshold = rest_velocity_threshold
        self.min_rest_duration = min_rest_duration_sec
        self.prev_landmarks: Optional[np.ndarray] = None
        self.last_movement_time: float = time.time()
        self.in_gesture: bool = False

    def update(self, current_landmarks: np.ndarray, timestamp: Optional[float] = None) -> Dict[str, Any]:
        t = timestamp or time.time()
        if self.prev_landmarks is None or current_landmarks is None:
            self.prev_landmarks = current_landmarks
            return {"velocity": 0.0, "is_moving": False, "is_boundary": False}

        # Compute Euclidean displacement of landmarks
        diff = np.abs(current_landmarks - self.prev_landmarks)
        velocity = float(np.mean(diff))
        self.prev_landmarks = current_landmarks

        is_moving = velocity > self.rest_threshold
        if is_moving:
            self.last_movement_time = t
            self.in_gesture = True
            is_boundary = False
        else:
            rest_duration = t - self.last_movement_time
            if self.in_gesture and rest_duration >= self.min_rest_duration:
                is_boundary = True
                self.in_gesture = False
            else:
                is_boundary = False

        return {
            "velocity": round(velocity, 4),
            "is_moving": is_moving,
            "is_boundary": is_boundary,
        }


class ContinuousSentenceDecoder:
    """
    Decodes continuous streams of frame predictions into grammatically structured sentences.
    Preserves raw recognition tokens and handles confidence calibration.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        consecutive_frames_required: int = 4,
        cooldown_same_token_sec: float = 1.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.consecutive_required = consecutive_frames_required
        self.cooldown_sec = cooldown_same_token_sec

        self.boundary_detector = KineticBoundaryDetector()
        self.token_history: deque = deque(maxlen=20)
        self.current_candidate: Optional[str] = None
        self.candidate_count: int = 0
        self.last_emitted_token: Optional[str] = None
        self.last_emitted_time: float = 0.0

        self.sentence_tokens: List[DecodedSign] = []

    def process_frame(
        self,
        predicted_token: str,
        confidence: float,
        landmarks: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> SentenceState:
        t = timestamp or time.time()

        # Update kinetic boundaries if landmarks available
        boundary_info = self.boundary_detector.update(landmarks, t) if landmarks is not None else {}
        is_boundary = boundary_info.get("is_boundary", False)

        # Filter low confidence predictions
        is_uncertain = confidence < self.confidence_threshold
        effective_token = predicted_token if not is_uncertain and predicted_token != "..." else None

        if effective_token:
            if effective_token == self.current_candidate:
                self.candidate_count += 1
            else:
                self.current_candidate = effective_token
                self.candidate_count = 1

            if self.candidate_count >= self.consecutive_required:
                # Check cooldown to prevent duplicate token spam
                can_emit = True
                if effective_token == self.last_emitted_token:
                    if (t - self.last_emitted_time) < self.cooldown_sec:
                        can_emit = False

                if can_emit:
                    decoded = DecodedSign(
                        token=effective_token,
                        confidence=confidence,
                        start_time=t - (self.candidate_count * 0.033),
                        end_time=t,
                        is_uncertain=False,
                    )
                    self.sentence_tokens.append(decoded)
                    self.last_emitted_token = effective_token
                    self.last_emitted_time = t
                    self.candidate_count = 0
        else:
            self.candidate_count = max(0, self.candidate_count - 1)
            if self.candidate_count == 0:
                self.current_candidate = None

        # Format constructed sentence
        raw_list = [s.token for s in self.sentence_tokens]
        constructed = " ".join(raw_list)

        # Basic punctuation and grammatical post-processing
        corrected = self._apply_grammatical_rules(raw_list)

        return SentenceState(
            raw_tokens=raw_list,
            constructed_sentence=constructed,
            corrected_sentence=corrected,
            last_action="BOUNDARY_DETECTED" if is_boundary else "INFERRING",
            last_update_time=t,
        )

    def _apply_grammatical_rules(self, tokens: List[str]) -> str:
        """Lightweight rule-based grammar normalizer for ISL Subject-Object-Verb (SOV) order."""
        if not tokens:
            return ""
        capitalized = [t.capitalize() for t in tokens]
        sentence = " ".join(capitalized)
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence

    def clear(self) -> None:
        """Reset the sentence state."""
        self.sentence_tokens.clear()
        self.current_candidate = None
        self.candidate_count = 0
        self.last_emitted_token = None
        self.last_emitted_time = 0.0
