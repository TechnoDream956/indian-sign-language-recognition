#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5P: Dual-Hand ISL Alphabet Data Collection Tool
=====================================================
Isolated data acquisition tool for capturing Indian Sign Language (ISL)
alphabet landmark features.

Modes:
1. SOLO MODE (--solo):
   - Hands-free auto-capture: automatically detects when the expected number
     of hands are visible and stable for the required consecutive frames.
   - Auto-saves verified samples without requiring keyboard input.
   - Enforces a mandatory neutral/release gesture between captures so only ONE
     sample is recorded per gesture.
   - Shows on-screen save confirmations: "SAVED: <LETTER> — X/Y samples".
   - Automatic letter progression after target samples are met, with a configurable
     countdown timer so the signer can prepare for the next sign.
   - Quality checks: validates (126,) dimension, finiteness, and rejects duplicate poses.
   - Direct letter jump via keyboard (press 'A'–'Z' anytime hands are released).
   - Undo/Delete last sample via 'D'.
   - Pause/Resume via 'SPACE'.

2. MANUAL MODE:
   - Review and stage workflow with [SPACE] capture, [SPACE] confirm, and [D] discard.

ISL Hand Configuration:
- Recognizes canonical ISL reference where 24 letters are two-handed and letter 'C'
  is standardly single-handed. Supports --expected-hands auto (default), 2, or 1.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import cv2
import mediapipe as mp
import numpy as np

# Ensure workspace root is importable
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.alphabet_engine import (
    extract_dual_hand_landmarks,
    validate_dual_hand_features,
)

# Standard ISL 26-letter alphabet target vocabulary
DEFAULT_ALPHABET_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Centralized ISL Alphabet Hand Configuration:
# - Single-handed letters (5): C, L, O, U, V
# - Two-handed letters (21): A, B, D, E, F, G, H, I, J, K, M, N, P, Q, R, S, T, W, X, Y, Z
ISL_ALPHABET_HAND_CONFIG: Dict[str, int] = {
    # Single-handed (5 letters)
    "C": 1,
    "L": 1,
    "O": 1,
    "U": 1,
    "V": 1,
    # Two-handed (21 letters)
    "A": 2,
    "B": 2,
    "D": 2,
    "E": 2,
    "F": 2,
    "G": 2,
    "H": 2,
    "I": 2,
    "J": 2,
    "K": 2,
    "M": 2,
    "N": 2,
    "P": 2,
    "Q": 2,
    "R": 2,
    "S": 2,
    "T": 2,
    "W": 2,
    "X": 2,
    "Y": 2,
    "Z": 2,
}


def check_is_duplicate(
    candidate: np.ndarray,
    recent_samples: List[np.ndarray],
    threshold: float = 1e-3
) -> Tuple[bool, float]:
    """
    Check if a candidate 126-feature vector is an obvious duplicate of recent samples.
    
    Returns:
    - (is_dup, min_distance)
    """
    if not recent_samples or len(recent_samples) == 0:
        return False, float("inf")
    
    min_dist = float("inf")
    for prev in recent_samples:
        dist = float(np.linalg.norm(candidate - prev))
        if dist < min_dist:
            min_dist = dist
        if dist < threshold:
            return True, dist
            
    return False, min_dist


class DualHandCollector:
    """
    Standalone collector that records stabilized 126-feature dual-hand vectors.
    Supports both Manual Review Mode and Hands-Free Solo Mode.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = WORKSPACE_ROOT / "data" / "phase5p_dual_hand",
        target_labels: Optional[List[str]] = None,
        expected_hands: Union[int, str] = "auto",
        stability_threshold: int = 5,
        signer_id: str = "signer_01",
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        solo_mode: bool = False,
        samples_per_label_target: int = 5,
        countdown_duration: float = 5.0,
        wait_for_user_start: bool = False,
    ):
        self.run_id = run_id
        self.session_id = session_id or (run_id if run_id else datetime.datetime.now().strftime("sess_%Y%m%d_%H%M%S"))
        
        # Safe run isolation: If a run_id is supplied, isolate into data/phase5p_dual_hand/runs/<run_id>
        p_out = Path(output_dir)
        if run_id:
            if p_out.name != run_id and "runs" not in str(p_out):
                self.output_dir = p_out / "runs" / run_id
            else:
                self.output_dir = p_out
        else:
            self.output_dir = p_out

        self.samples_dir = self.output_dir / "samples"
        self.metadata_csv = self.output_dir / "metadata.csv"
        self.report_json = self.output_dir / "collection_report.json"

        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.target_labels = target_labels or DEFAULT_ALPHABET_CLASSES
        self.current_label_idx = 0
        self.expected_hands_mode = expected_hands
        self.stability_threshold = stability_threshold
        self.signer_id = signer_id

        # Solo Mode configuration
        self.solo_mode = solo_mode
        self.samples_per_label_target = samples_per_label_target
        self.countdown_duration = countdown_duration
        self.wait_for_user_start = wait_for_user_start
        self.solo_state = "WAITING_TO_START" if (solo_mode and wait_for_user_start) else "READY_FOR_SIGN"
        self.countdown_start_time = 0.0
        self.has_released_after_save = True
        self.saved_feedback_timer = 0.0
        self.saved_feedback_msg = ""
        self.rejected_samples_count = 0
        self.last_rejection_reason = ""

        # General state tracking
        self.consecutive_valid_frames = 0
        self.recent_samples: List[np.ndarray] = []
        self.last_saved_sample: Optional[Dict[str, Any]] = None
        self.samples_per_label: Dict[str, int] = {lbl: 0 for lbl in self.target_labels}
        self.total_saved = 0

        # Review state (manual mode)
        self.in_review_mode = False
        self.pending_sample: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None

        self._init_metadata_file()
        self._load_existing_counts()

    def _init_metadata_file(self) -> None:
        """Create metadata.csv with headers if it does not already exist."""
        if not self.metadata_csv.exists():
            with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sample_id",
                    "label",
                    "hand_count",
                    "has_left",
                    "has_right",
                    "is_ambiguous",
                    "signer_id",
                    "session_id",
                    "timestamp",
                    "quality_status",
                    "filepath",
                ])

    def _load_existing_counts(self) -> None:
        """Scan existing metadata to resume sample counts accurately."""
        if not self.metadata_csv.exists():
            return

        try:
            with open(self.metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = row.get("label")
                    if lbl in self.samples_per_label:
                        self.samples_per_label[lbl] += 1
                    self.total_saved += 1
        except Exception as e:
            print(f"[Collector Warning] Error reading metadata.csv: {e}")

    @property
    def current_label(self) -> str:
        return self.target_labels[self.current_label_idx]

    def get_expected_hands(self, label: Optional[str] = None) -> int:
        """Return expected hand count for the specified letter based on configuration."""
        target_lbl = label or self.current_label
        if str(self.expected_hands_mode).lower() == "auto":
            return ISL_ALPHABET_HAND_CONFIG.get(target_lbl, 2)
        try:
            return int(self.expected_hands_mode)
        except (ValueError, TypeError):
            return 2

    def next_label(self) -> str:
        self.current_label_idx = (self.current_label_idx + 1) % len(self.target_labels)
        self.consecutive_valid_frames = 0
        self.in_review_mode = False
        self.pending_sample = None
        self.has_released_after_save = True
        self.solo_state = "READY_FOR_SIGN"
        return self.current_label

    def prev_label(self) -> str:
        self.current_label_idx = (self.current_label_idx - 1) % len(self.target_labels)
        self.consecutive_valid_frames = 0
        self.in_review_mode = False
        self.pending_sample = None
        self.has_released_after_save = True
        self.solo_state = "READY_FOR_SIGN"
        return self.current_label

    def set_label(self, label: str) -> bool:
        """Select a target label directly via keyboard."""
        upper_lbl = label.upper()
        if upper_lbl in self.target_labels:
            self.current_label_idx = self.target_labels.index(upper_lbl)
            self.consecutive_valid_frames = 0
            self.in_review_mode = False
            self.pending_sample = None
            self.has_released_after_save = True
            self.solo_state = "READY_FOR_SIGN"
            return True
        return False

    def start_collection(self) -> None:
        """Explicit user start for collection run."""
        self.solo_state = "READY_FOR_SIGN"
        self.has_released_after_save = True
        self.consecutive_valid_frames = 0

    def validate_sample(
        self,
        features: np.ndarray,
        meta: Dict[str, Any],
        label: str
    ) -> Tuple[bool, str]:
        """
        Validate quality, dimensions, hand count, coordinate finiteness,
        non-empty features, and duplicate thresholds before saving.
        """
        if label not in self.target_labels:
            return False, f"Label '{label}' is not in target vocabulary"

        expected = self.get_expected_hands(label)
        actual_hands = meta.get("hand_count", 0)
        if actual_hands != expected:
            return False, f"Expected {expected} hands for '{label}', got {actual_hands}"

        if not isinstance(features, np.ndarray) or features.shape != (126,):
            return False, f"Invalid feature shape {getattr(features, 'shape', None)}, expected (126,)"

        if not np.all(np.isfinite(features)):
            return False, "Features contain NaN or Infinite values"

        # Check non-empty features
        left_norm = float(np.linalg.norm(features[:63]))
        right_norm = float(np.linalg.norm(features[63:]))
        if expected == 1:
            has_left_feats = (left_norm > 1e-4)
            has_right_feats = (right_norm > 1e-4)
            if not (has_left_feats ^ has_right_feats):
                return False, "Single-hand features must have exactly one populated hand and one zero-padded hand"
        elif expected == 2:
            if left_norm < 1e-4 or right_norm < 1e-4:
                return False, "Two-handed features require non-zero coordinates for both hands"

        # Check duplicate
        is_dup, dist = check_is_duplicate(features, self.recent_samples, threshold=1e-3)
        if is_dup:
            return False, f"Duplicate pose detected (distance {dist:.5f} < 0.001)"

        return True, "Valid"

    def process_frame_landmarks(
        self,
        multi_hand_landmarks: Optional[List[Any]],
        multi_handedness: Optional[List[Any]] = None,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process extracted landmarks and evaluate stability & quality readiness.
        In Solo Mode, automatically coordinates state transitions, countdowns, and auto-saving.
        """
        now = current_time if current_time is not None else time.time()
        left_feats, right_feats, combined_126, meta = extract_dual_hand_landmarks(
            multi_hand_landmarks, multi_handedness
        )
        hand_count = meta["hand_count"]
        expected_hands = self.get_expected_hands()

        # Waiting for explicit user start
        if self.solo_mode and self.solo_state == "WAITING_TO_START":
            return {
                "hand_count": hand_count,
                "expected_hands": expected_hands,
                "has_left": meta["has_left"],
                "has_right": meta["has_right"],
                "is_ambiguous": meta["is_ambiguous"],
                "features": combined_126,
                "is_stable": False,
                "consecutive_valid_frames": 0,
                "is_valid": False,
                "validation_msg": "Waiting for user to press [SPACE] to start run",
                "is_duplicate": False,
                "duplicate_dist": float("inf"),
                "ready_to_record": False,
                "auto_saved": False,
                "auto_save_msg": "",
                "solo_state": self.solo_state,
                "countdown_remaining": 0.0,
            }

        # Completed-label guard: If current label already reached target, stop accepting further samples
        cur_label_count = self.samples_per_label.get(self.current_label, 0)
        if self.solo_mode and self.solo_state not in ("PAUSED", "COUNTDOWN_NEXT_LETTER", "RUN_COMPLETE"):
            if cur_label_count >= self.samples_per_label_target:
                if self.current_label_idx + 1 < len(self.target_labels):
                    self.solo_state = "COUNTDOWN_NEXT_LETTER"
                    self.countdown_start_time = now
                    self.has_released_after_save = False
                    self.consecutive_valid_frames = 0
                else:
                    self.solo_state = "RUN_COMPLETE"
                    self.consecutive_valid_frames = 0

        # Handle Solo Countdown state
        countdown_remaining = 0.0
        if self.solo_mode and self.solo_state == "COUNTDOWN_NEXT_LETTER":
            elapsed = now - self.countdown_start_time
            countdown_remaining = max(0.0, self.countdown_duration - elapsed)
            self.consecutive_valid_frames = 0  # Do not accumulate stability during countdown!
            if countdown_remaining <= 0.0:
                if self.current_label_idx + 1 < len(self.target_labels):
                    self.next_label()
                    # Advance automatically, but REQUIRE neutral/release state before collecting next letter
                    self.solo_state = "WAITING_FOR_RELEASE"
                    self.has_released_after_save = False
                    self.consecutive_valid_frames = 0
                else:
                    self.solo_state = "RUN_COMPLETE"
                    self.consecutive_valid_frames = 0

        # Quality validation checks
        is_expected = (hand_count == expected_hands)
        is_valid, validation_msg = validate_dual_hand_features(combined_126, meta)
        is_duplicate, min_dist = check_is_duplicate(combined_126, self.recent_samples)

        # Solo Mode: Check gesture release (one sample per gesture rule)
        if self.solo_mode and not self.has_released_after_save:
            released = False
            if hand_count == 0 or hand_count < expected_hands:
                released = True
            elif self.recent_samples:
                dist_from_last = float(np.linalg.norm(combined_126 - self.recent_samples[-1]))
                if dist_from_last > 0.05:
                    released = True

            if released:
                self.has_released_after_save = True
                if self.solo_state not in ("COUNTDOWN_NEXT_LETTER", "RUN_COMPLETE"):
                    self.solo_state = "READY_FOR_SIGN"
                self.consecutive_valid_frames = 0
            else:
                if self.solo_state not in ("COUNTDOWN_NEXT_LETTER", "RUN_COMPLETE"):
                    self.solo_state = "WAITING_FOR_RELEASE"
                self.consecutive_valid_frames = 0

        # Stability counter tracking
        if is_expected and (not self.solo_mode or self.has_released_after_save) and (self.solo_state not in ("PAUSED", "COUNTDOWN_NEXT_LETTER", "RUN_COMPLETE", "WAITING_TO_START")):
            self.consecutive_valid_frames += 1
            if self.solo_mode and self.solo_state != "COUNTDOWN_NEXT_LETTER":
                self.solo_state = "STABILIZING"
        else:
            self.consecutive_valid_frames = 0
            if self.solo_mode and self.has_released_after_save and (self.solo_state not in ("PAUSED", "COUNTDOWN_NEXT_LETTER", "WAITING_FOR_RELEASE", "RUN_COMPLETE", "WAITING_TO_START")):
                self.solo_state = "READY_FOR_SIGN"

        is_stable = (self.consecutive_valid_frames >= self.stability_threshold)
        ready_to_record = is_stable and is_valid and (not is_duplicate)

        # Solo Mode Autonomous Capture
        auto_saved = False
        auto_save_msg = ""
        if self.solo_mode and self.solo_state not in ("PAUSED", "COUNTDOWN_NEXT_LETTER", "WAITING_TO_START", "WAITING_FOR_RELEASE", "RUN_COMPLETE"):
            if ready_to_record and self.has_released_after_save:
                # Pre-check target before saving to prevent extra samples
                if self.samples_per_label.get(self.current_label, 0) < self.samples_per_label_target:
                    self.pending_sample = (combined_126, meta)
                    self.in_review_mode = True
                    ok, save_msg = self.commit_pending_sample()
                    if ok:
                        auto_saved = True
                        auto_save_msg = save_msg
                        cur_cnt = self.samples_per_label.get(self.current_label, 0)
                        self.has_released_after_save = False

                        # Check if target reached
                        if cur_cnt >= self.samples_per_label_target:
                            self.saved_feedback_msg = f"SAVED: {self.current_label} — {cur_cnt}/{self.samples_per_label_target} (TARGET REACHED!)"
                            self.saved_feedback_timer = now + 4.0
                            if self.current_label_idx + 1 < len(self.target_labels):
                                self.solo_state = "COUNTDOWN_NEXT_LETTER"
                                self.countdown_start_time = now
                                self.has_released_after_save = False
                                self.consecutive_valid_frames = 0
                            else:
                                self.solo_state = "RUN_COMPLETE"
                                self.consecutive_valid_frames = 0
                        else:
                            self.saved_feedback_msg = f"SAVED: {self.current_label} — {cur_cnt}/{self.samples_per_label_target} samples"
                            self.saved_feedback_timer = now + 2.5
                            self.solo_state = "WAITING_FOR_RELEASE"
                    else:
                        auto_save_msg = save_msg

        return {
            "hand_count": hand_count,
            "expected_hands": expected_hands,
            "has_left": meta["has_left"],
            "has_right": meta["has_right"],
            "is_ambiguous": meta["is_ambiguous"],
            "features": combined_126,
            "is_stable": is_stable,
            "consecutive_valid_frames": self.consecutive_valid_frames,
            "is_valid": is_valid,
            "validation_msg": validation_msg,
            "is_duplicate": is_duplicate,
            "duplicate_dist": min_dist,
            "ready_to_record": ready_to_record,
            "auto_saved": auto_saved,
            "auto_save_msg": auto_save_msg,
            "solo_state": self.solo_state,
            "countdown_remaining": countdown_remaining,
        }

    def stage_sample_for_review(self, frame_info: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Stage a verified sample for user review (manual mode).
        """
        if not frame_info["is_valid"]:
            return False, f"Invalid features: {frame_info['validation_msg']}"
        expected = self.get_expected_hands()
        if frame_info["hand_count"] != expected:
            return False, f"Expected {expected} hands, got {frame_info['hand_count']}"
        if frame_info["is_duplicate"]:
            return False, f"Duplicate sample detected (dist={frame_info['duplicate_dist']:.4f})"

        self.pending_sample = (frame_info["features"], frame_info)
        self.in_review_mode = True
        return True, "Sample captured! Press [SPACE] to Confirm & Save, or [D] to Discard."

    def commit_pending_sample(self) -> Tuple[bool, str]:
        """
        Commit and write pending sample to disk and metadata.csv after validation.
        """
        if not self.in_review_mode or self.pending_sample is None:
            return False, "No pending sample to commit."

        features, meta = self.pending_sample
        label = self.current_label

        # Strict quality validation
        is_ok, val_reason = self.validate_sample(features, meta, label)
        if not is_ok:
            self.rejected_samples_count += 1
            self.last_rejection_reason = val_reason
            self.in_review_mode = False
            self.pending_sample = None
            self.consecutive_valid_frames = 0
            return False, f"REJECTED: {val_reason}"

        # Prevent extra samples if target already reached
        current_cnt = self.samples_per_label.get(label, 0)
        if current_cnt >= self.samples_per_label_target:
            self.in_review_mode = False
            self.pending_sample = None
            self.consecutive_valid_frames = 0
            return False, f"Target already reached for '{label}' ({current_cnt}/{self.samples_per_label_target})"

        count = current_cnt + 1
        sample_id = f"sample_{label}_{count:04d}"
        filename = f"{sample_id}.npy"
        filepath = self.samples_dir / filename

        # 1. Save .npy array
        np.save(filepath, features.astype(np.float32))

        # 2. Append metadata row
        iso_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        rel_path = f"samples/{filename}"
        with open(self.metadata_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                sample_id,
                label,
                meta["hand_count"],
                meta["has_left"],
                meta["has_right"],
                meta["is_ambiguous"],
                self.signer_id,
                self.session_id,
                iso_timestamp,
                "VALID",
                rel_path,
            ])

        # 3. Update memory state
        self.samples_per_label[label] = count
        self.total_saved += 1
        self.recent_samples.append(features)
        if len(self.recent_samples) > 20:
            self.recent_samples.pop(0)

        self.last_saved_sample = {
            "sample_id": sample_id,
            "label": label,
            "filepath": str(filepath),
        }

        self.in_review_mode = False
        self.pending_sample = None
        self.consecutive_valid_frames = 0

        self.write_summary_report()
        return True, f"Saved {sample_id} ({label}) successfully!"

    def redo_current_label(self, label: Optional[str] = None) -> Tuple[bool, str]:
        """
        Redo / Clear all samples recorded for the given label in this run.
        Deletes physical files, cleans metadata.csv, and resets count to 0.
        """
        target_lbl = label or self.current_label
        count = self.samples_per_label.get(target_lbl, 0)
        if count <= 0:
            return False, f"No samples recorded for letter '{target_lbl}' to redo."

        # 1. Delete all physical files for this label
        for i in range(1, count + 1):
            sample_file = self.samples_dir / f"sample_{target_lbl}_{i:04d}.npy"
            if sample_file.exists():
                try:
                    sample_file.unlink()
                except Exception as e:
                    print(f"[Collector Warning] Failed to delete {sample_file.name}: {e}")

        # 2. Rewrite metadata.csv omitting this label
        if self.metadata_csv.exists():
            try:
                with open(self.metadata_csv, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                fieldnames = [
                    "sample_id", "label", "hand_count", "has_left", "has_right",
                    "is_ambiguous", "signer_id", "session_id", "timestamp",
                    "quality_status", "filepath"
                ]
                filtered = [r for r in reader if r.get("label") != target_lbl]
                with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(filtered)
            except Exception as e:
                return False, f"Failed updating metadata.csv: {e}"

        # 3. Update memory state
        self.total_saved = max(0, self.total_saved - count)
        self.samples_per_label[target_lbl] = 0
        self.consecutive_valid_frames = 0
        self.has_released_after_save = True
        self.solo_state = "READY_FOR_SIGN"
        self.write_summary_report()
        return True, f"Reset letter '{target_lbl}' (0/{self.samples_per_label_target} samples). Ready to re-record."

    def discard_pending_sample(self) -> Tuple[bool, str]:
        """
        Discard the currently staged sample.
        """
        if not self.in_review_mode or self.pending_sample is None:
            return False, "No sample pending discard."

        self.in_review_mode = False
        self.pending_sample = None
        self.consecutive_valid_frames = 0
        return True, "Sample discarded. Ready to re-record."

    def delete_last_sample_for_label(self, label: Optional[str] = None) -> Tuple[bool, str]:
        """
        Undo / Delete the most recently saved sample for a given label.
        Removes the physical file and updates metadata.csv.
        """
        target_lbl = label or self.current_label
        count = self.samples_per_label.get(target_lbl, 0)
        if count <= 0:
            return False, f"No recorded samples for letter '{target_lbl}' to delete."

        sample_id = f"sample_{target_lbl}_{count:04d}"
        target_file = self.samples_dir / f"{sample_id}.npy"

        # 1. Remove physical file
        if target_file.exists():
            try:
                target_file.unlink()
            except Exception as e:
                return False, f"Failed to delete file {target_file.name}: {e}"

        # 2. Rewrite metadata.csv omitting this sample_id
        if self.metadata_csv.exists():
            try:
                with open(self.metadata_csv, "r", encoding="utf-8") as f:
                    reader = list(csv.DictReader(f))
                fieldnames = [
                    "sample_id", "label", "hand_count", "has_left", "has_right",
                    "is_ambiguous", "signer_id", "session_id", "timestamp",
                    "quality_status", "filepath"
                ]
                filtered = [r for r in reader if r.get("sample_id") != sample_id]
                with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(filtered)
            except Exception as e:
                return False, f"Failed updating metadata.csv: {e}"

        # 3. Update memory state
        self.samples_per_label[target_lbl] = count - 1
        self.total_saved = max(0, self.total_saved - 1)
        if self.recent_samples:
            self.recent_samples.pop()

        self.consecutive_valid_frames = 0
        self.has_released_after_save = True
        self.solo_state = "READY_FOR_SIGN"
        self.write_summary_report()
        return True, f"Deleted {sample_id} ({target_lbl}). Count: {self.samples_per_label[target_lbl]}/{self.samples_per_label_target}"

    def write_summary_report(self) -> Dict[str, Any]:
        """
        Generate and persist collection_report.json summary.
        """
        report = {
            "session_id": self.session_id,
            "signer_id": self.signer_id,
            "solo_mode": self.solo_mode,
            "total_samples": self.total_saved,
            "expected_hands_mode": self.expected_hands_mode,
            "samples_per_label": self.samples_per_label,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            with open(self.report_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"[Collector Warning] Failed to write report.json: {e}")
        return report


def draw_hud(
    frame: np.ndarray,
    collector: DualHandCollector,
    frame_info: Dict[str, Any],
    status_msg: str
) -> np.ndarray:
    """
    Render clean, informative HUD over the OpenCV window.
    """
    h, w, _ = frame.shape
    now = time.time()
    overlay = frame.copy()

    # Top banner bar
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    # Bottom control bar
    cv2.rectangle(overlay, (0, h - 75), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 1. Mode Badge & Target Label
    lbl = collector.current_label
    count = collector.samples_per_label.get(lbl, 0)
    target = collector.samples_per_label_target if collector.solo_mode else "∞"
    exp_h = collector.get_expected_hands(lbl)

    if collector.solo_mode:
        mode_text = "[SOLO AUTO-COLLECT]"
        mode_color = (255, 200, 0)
    else:
        mode_text = "[MANUAL MODE]"
        mode_color = (200, 200, 200)

    cv2.putText(frame, mode_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1)

    run_tag = f" | RUN: {collector.run_id}" if collector.run_id else ""
    lbl_text = f"LETTER: {lbl} [{collector.current_label_idx + 1}/{len(collector.target_labels)}]{run_tag}"
    cv2.putText(frame, lbl_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    progress_text = f"Samples: {count}/{target} | Needs: {exp_h} Hand{'s' if exp_h > 1 else ''}"
    cv2.putText(frame, progress_text, (20, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # 2. Hand Status Tag
    hand_count = frame_info["hand_count"]
    if hand_count == exp_h:
        tag_color = (0, 220, 0)
        tag_text = f"{hand_count} HAND{'S' if hand_count > 1 else ''} DETECTED"
    elif hand_count > 0:
        tag_color = (0, 215, 255)
        tag_text = f"{hand_count} HAND{'S' if hand_count > 1 else ''} (NEED {exp_h})"
    else:
        tag_color = (0, 0, 255)
        tag_text = "NO HANDS"

    cv2.putText(frame, tag_text, (w - 300, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tag_color, 2)

    # 3. Solo State & Stability Indicator
    thresh = collector.stability_threshold
    stable_count = frame_info["consecutive_valid_frames"]

    if collector.solo_mode:
        state = collector.solo_state
        if state == "WAITING_TO_START":
            stab_text = "[READY TO START — PRESS SPACE]"
            stab_color = (0, 255, 255)
        elif state == "PAUSED":
            stab_text = "[PAUSED — PRESS SPACE TO RESUME]"
            stab_color = (0, 0, 255)
        elif state == "COUNTDOWN_NEXT_LETTER":
            rem = frame_info.get("countdown_remaining", 0.0)
            next_idx = (collector.current_label_idx + 1) % len(collector.target_labels)
            next_lbl = collector.target_labels[next_idx]
            stab_text = f"NEXT LETTER IN {rem:.1f}s: GET READY FOR '{next_lbl}'"
            stab_color = (0, 165, 255)
        elif state == "WAITING_FOR_RELEASE":
            stab_text = "[PLEASE RELEASE HANDS FOR NEXT SAMPLE]"
            stab_color = (0, 215, 255)
        elif state == "RUN_COMPLETE":
            stab_text = "[A–Z RUN COMPLETE! PRESS Q TO EXIT]"
            stab_color = (0, 255, 100)
        elif stable_count >= thresh:
            stab_text = f"STABILITY: {thresh}/{thresh} [SAVING...]"
            stab_color = (0, 255, 0)
        else:
            stab_text = f"STABILITY: {stable_count}/{thresh} [HOLD STEADY]"
            stab_color = (180, 180, 180)
    else:
        if collector.in_review_mode:
            stab_text = "[REVIEW MODE: CONFIRM OR DISCARD]"
            stab_color = (0, 165, 255)
        elif stable_count >= thresh:
            stab_text = f"STABILITY: {thresh}/{thresh} [READY]"
            stab_color = (0, 255, 0)
        else:
            stab_text = f"STABILITY: {stable_count}/{thresh} [HOLD STEADY]"
            stab_color = (180, 180, 180)

    cv2.putText(frame, stab_text, (w - 470, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, stab_color, 2)

    # Center Prominent Banners for Countdown / Start / Completion
    if collector.solo_mode:
        state = collector.solo_state
        if state == "WAITING_TO_START":
            center_overlay = frame.copy()
            cv2.rectangle(center_overlay, (40, h // 2 - 55), (w - 40, h // 2 + 55), (20, 40, 20), -1)
            cv2.addWeighted(center_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, "[PRESS SPACE TO START COLLECTION]", (w // 2 - 250, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(frame, f"Starting at Letter '{lbl}' ({exp_h} Hand{'s' if exp_h > 1 else ''}) | Target: {target} samples", (w // 2 - 240, h // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        elif state == "COUNTDOWN_NEXT_LETTER":
            rem = frame_info.get("countdown_remaining", 0.0)
            countdown_num = int(np.ceil(rem))
            next_idx = (collector.current_label_idx + 1) % len(collector.target_labels)
            next_lbl = collector.target_labels[next_idx]
            next_exp = collector.get_expected_hands(next_lbl)
            center_overlay = frame.copy()
            cv2.rectangle(center_overlay, (40, h // 2 - 65), (w - 40, h // 2 + 65), (35, 20, 65), -1)
            cv2.addWeighted(center_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, f"NEXT LETTER IN {countdown_num}...", (w // 2 - 170, h // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            cv2.putText(frame, f"PREPARE FOR '{next_lbl}' ({next_exp} Hand{'s' if next_exp > 1 else ''})", (w // 2 - 210, h // 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif state == "RUN_COMPLETE":
            center_overlay = frame.copy()
            cv2.rectangle(center_overlay, (40, h // 2 - 50), (w - 40, h // 2 + 50), (20, 60, 20), -1)
            cv2.addWeighted(center_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, "[RUN COMPLETE: ALL LETTERS SAVED]", (w // 2 - 240, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
            cv2.putText(frame, "Press [Q] or [ESC] to write final report & exit.", (w // 2 - 220, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # 4. Prominent Saved Feedback Toast or Status Message
    if collector.saved_feedback_timer > now:
        cv2.putText(frame, collector.saved_feedback_msg, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    elif status_msg:
        cv2.putText(frame, status_msg, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)

    # 5. Instructions Bar
    if collector.solo_mode:
        ctrl_text = "[SPACE] Start/Pause | [R] Redo Letter | [D] Undo Sample | [N/P] Skip | [Q] Exit"
    else:
        ctrl_text = "[SPACE] Record/Confirm | [D] Discard | [N/P] Next/Prev Letter | [Q] Exit"
    cv2.putText(frame, ctrl_text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


def run_collector(
    camera_id: int = 0,
    output_dir: Path = WORKSPACE_ROOT / "data" / "phase5p_dual_hand",
    signer_id: str = "signer_01",
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    target_labels: Optional[List[str]] = None,
    expected_hands: Union[int, str] = "auto",
    stability_threshold: int = 5,
    solo_mode: bool = False,
    samples_per_label: int = 5,
    countdown_duration: float = 5.0,
    wait_for_user_start: bool = True,
) -> None:
    """
    Main interactive collector loop supporting Solo Mode and Manual Mode.
    """
    collector = DualHandCollector(
        output_dir=output_dir,
        target_labels=target_labels,
        expected_hands=expected_hands,
        stability_threshold=stability_threshold,
        signer_id=signer_id,
        session_id=session_id,
        run_id=run_id,
        solo_mode=solo_mode,
        samples_per_label_target=samples_per_label,
        countdown_duration=countdown_duration,
        wait_for_user_start=wait_for_user_start if solo_mode else False,
    )

    mode_title = "SOLO HANDS-FREE MODE" if solo_mode else "MANUAL REVIEW MODE"
    print("\n" + "=" * 68)
    print(f"  ISL ALPHABET COLLECTOR (Phase 5P) — {mode_title}")
    print("=" * 68)
    print(f"Output Directory:    {collector.output_dir}")
    print(f"Run Identifier:      {collector.run_id or 'none (default)'}")
    print(f"Signer Identifier:   {collector.signer_id}")
    print(f"Session Identifier:  {collector.session_id}")
    print(f"Target Vocabulary:   {collector.target_labels}")
    print(f"Target Per Label:    {collector.samples_per_label_target if solo_mode else 'Manual'}")
    print(f"Countdown Duration:  {collector.countdown_duration}s")
    print(f"Expected Hands:      {collector.expected_hands_mode} (ISL C,L,O,U,V=1, Others=2)")

    if solo_mode:
        print("\nSolo Mode Controls:")
        print("  [SPACE]      - Start Run / Pause / Resume collection")
        print("  [R]          - Redo current letter (clears samples for letter and resets)")
        print("  [D]          - Undo / Delete last saved sample for current letter")
        print("  [N] / [P]    - Skip to next / previous letter")
        print("  [A–Z]        - Jump directly to any letter")
        print("  [Q] / [ESC]  - Save report and exit")
    else:
        print("\nManual Mode Controls:")
        print("  [SPACE]      - Capture current stable pose / Confirm in review mode")
        print("  [D]          - Discard sample in review mode")
        print("  [N] / [P]    - Next / previous letter")
        print("  [Q] / [ESC]  - Save report and exit")
    print("=" * 68 + "\n")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Collector Error] Could not open camera {camera_id}.")
        hands_detector.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if solo_mode and collector.solo_state == "WAITING_TO_START":
        status_msg = f"Ready. Press [SPACE] to start collecting letter '{collector.current_label}'."
    else:
        status_msg = f"Ready. Show {collector.get_expected_hands()} hand(s) for letter '{collector.current_label}'."

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands_detector.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Evaluate frame
            frame_info = collector.process_frame_landmarks(
                results.multi_hand_landmarks, results.multi_handedness, current_time=time.time()
            )

            if frame_info.get("auto_saved"):
                status_msg = frame_info["auto_save_msg"]

            # Draw HUD
            hud_frame = draw_hud(frame, collector, frame_info, status_msg)
            cv2.imshow("ISL Alphabet Data Collector (Press 'Q' to Exit)", hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # Q or ESC
                print("[Collector] Clean exit requested by user.")
                break

            # Solo Mode Keyboard Handling
            if collector.solo_mode:
                if key == ord(" "):  # SPACE: Start / Pause / Resume
                    if collector.solo_state == "WAITING_TO_START":
                        collector.start_collection()
                        status_msg = f"Collection started! Show {collector.get_expected_hands()} hand(s) for letter '{collector.current_label}'."
                    elif collector.solo_state == "PAUSED":
                        collector.solo_state = "READY_FOR_SIGN"
                        status_msg = "Resumed collection."
                    else:
                        collector.solo_state = "PAUSED"
                        status_msg = "PAUSED. Press [SPACE] to resume."
                elif key == ord("r"):  # R: Redo current letter
                    ok, msg = collector.redo_current_label()
                    status_msg = msg
                elif key == ord("d"):  # D: Delete / Undo last sample
                    ok, msg = collector.delete_last_sample_for_label()
                    status_msg = msg
                elif key == ord("n") or key == ord("]"):  # N or ]: Next letter
                    lbl = collector.next_label()
                    status_msg = f"Switched to letter '{lbl}'."
                elif key == ord("p") or key == ord("["):  # P or [: Previous letter
                    lbl = collector.prev_label()
                    status_msg = f"Switched to letter '{lbl}'."
                elif (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
                    letter = chr(key).upper()
                    if collector.set_label(letter):
                        status_msg = f"Jumped to letter '{letter}'."

            # Manual Mode Keyboard Handling
            else:
                if key == ord(" "):  # SPACE: Stage / Confirm
                    if collector.in_review_mode:
                        ok, msg = collector.commit_pending_sample()
                        status_msg = msg
                    else:
                        if frame_info["ready_to_record"]:
                            ok, msg = collector.stage_sample_for_review(frame_info)
                            status_msg = msg
                        else:
                            exp = collector.get_expected_hands()
                            if not frame_info["is_stable"]:
                                status_msg = f"Hold pose steady ({frame_info['consecutive_valid_frames']}/{collector.stability_threshold} frames)."
                            elif frame_info["is_duplicate"]:
                                status_msg = "Duplicate pose! Shift hands slightly."
                            elif frame_info["hand_count"] != exp:
                                status_msg = f"Need {exp} hand(s) (currently {frame_info['hand_count']})."
                elif key == ord("d"):  # D: Discard
                    if collector.in_review_mode:
                        ok, msg = collector.discard_pending_sample()
                        status_msg = msg
                    else:
                        status_msg = "No pending sample to discard."
                elif key == ord("n") or key == ord("]"):  # N or ]: Next letter
                    lbl = collector.next_label()
                    status_msg = f"Switched to letter '{lbl}'."
                elif key == ord("p") or key == ord("["):  # P or [: Previous letter
                    lbl = collector.prev_label()
                    status_msg = f"Switched to letter '{lbl}'."
                elif (ord("a") <= key <= ord("z")) or (ord("A") <= key <= ord("Z")):
                    letter = chr(key).upper()
                    if collector.set_label(letter):
                        status_msg = f"Jumped to letter '{letter}'."

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()
        report = collector.write_summary_report()
        print("\n" + "=" * 68)
        print("  COLLECTION SESSION COMPLETE")
        print("=" * 68)
        print(f"Total Samples Saved: {report['total_samples']}")
        print(f"Metadata File:       {collector.metadata_csv}")
        print(f"Report File:         {collector.report_json}")
        print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Dual-Hand Alphabet Data Collector")
    parser.add_argument("--solo", action="store_true", help="Enable hands-free solo recording mode")
    parser.add_argument("--run-id", type=str, default=None, help="Unique identifier for an isolated collection run (e.g. 'run_az_solo_01')")
    parser.add_argument("--samples-per-label", type=int, default=5, help="Target samples per letter in solo mode (default: 5)")
    parser.add_argument("--countdown", type=float, default=5.0, help="Countdown seconds before advancing to next letter (default: 5.0)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--output-dir", type=str, default="data/phase5p_dual_hand", help="Directory for dataset storage")
    parser.add_argument("--signer-id", type=str, default="signer_01", help="Signer identifier string")
    parser.add_argument("--session-id", type=str, default=None, help="Session identifier override")
    parser.add_argument("--expected-hands", default="auto", help="Expected hands: 'auto' (ISL reference: C,L,O,U,V=1, others=2), '2', or '1'")
    parser.add_argument("--stability", type=int, default=5, help="Consecutive frames required for stability (default: 5)")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_ALPHABET_CLASSES, help="Subset of alphabet labels to collect (default: A through Z)")

    args = parser.parse_args()

    run_collector(
        camera_id=args.camera,
        output_dir=Path(args.output_dir),
        signer_id=args.signer_id,
        session_id=args.session_id,
        run_id=args.run_id,
        target_labels=args.labels,
        expected_hands=args.expected_hands,
        stability_threshold=args.stability,
        solo_mode=args.solo,
        samples_per_label=args.samples_per_label,
        countdown_duration=args.countdown,
        wait_for_user_start=True,
    )
