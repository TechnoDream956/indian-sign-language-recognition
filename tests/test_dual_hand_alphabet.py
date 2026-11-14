#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Phase 5O: Dual-Hand ISL Alphabet Recognition
======================================================
Comprehensive test suite validating:
1. Single-hand feature extraction (left only, right only)
2. Dual-hand feature extraction (left + right -> 126 features)
3. Stable left/right ordering (handedness preservation & reverse order resilience)
4. Ambiguous handedness fallback (spatial x-coordinate sorting)
5. Missing-hand handling (clean zero representation & presence flags)
6. Feature validation (dimension, finiteness, NaN/Inf rejection)
7. Hand-count transitions (0 -> 1 -> 2 -> 1 -> 0)
8. Prevention of false letter emission on two-hand gestures
9. Preservation of single-hand model inference & debounce state machine
10. Backward compatibility of process_landmarks()
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from src.alphabet_engine import (
    WordBuffer,
    AlphabetEngine,
    normalize_hand_landmarks,
    extract_dual_hand_landmarks,
    validate_single_hand_features,
    validate_dual_hand_features,
)


class MockLandmark:
    """Mock MediaPipe NormalizedLandmark."""
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockClassification:
    def __init__(self, label="Left", score=0.95):
        self.label = label
        self.score = score


class MockHandedness:
    """Mock MediaPipe ClassificationList."""
    def __init__(self, label="Left", score=0.95):
        self.classification = [MockClassification(label, score)]


def create_mock_hand_landmarks(wrist_x=0.5, wrist_y=0.5, scale=0.1):
    """Generate a realistic set of 21 hand landmarks."""
    lms = []
    for i in range(21):
        if i == 0:
            lms.append(MockLandmark(wrist_x, wrist_y, 0.0))
        elif i == 9:  # Middle finger MCP
            lms.append(MockLandmark(wrist_x, wrist_y - scale, 0.0))
        else:
            lms.append(MockLandmark(wrist_x + (i - 10) * 0.005, wrist_y - (scale * (i % 5 + 1) / 5), 0.001 * i))
    return lms


class TestDualHandExtraction(unittest.TestCase):
    """Test suite for dual-hand landmark extraction and feature representation."""

    def setUp(self):
        self.left_lms = create_mock_hand_landmarks(wrist_x=0.3, wrist_y=0.6, scale=0.12)
        self.right_lms = create_mock_hand_landmarks(wrist_x=0.7, wrist_y=0.6, scale=0.12)
        self.left_handedness = MockHandedness("Left", score=0.97)
        self.right_handedness = MockHandedness("Right", score=0.96)

    def test_missing_hands_returns_zeros(self):
        """0 hands detected produces zeros with valid metadata."""
        left, right, combined, meta = extract_dual_hand_landmarks([], [])
        self.assertEqual(meta["hand_count"], 0)
        self.assertFalse(meta["has_left"])
        self.assertFalse(meta["has_right"])
        self.assertEqual(left.shape, (63,))
        self.assertEqual(right.shape, (63,))
        self.assertEqual(combined.shape, (126,))
        self.assertTrue(np.allclose(left, 0.0))
        self.assertTrue(np.allclose(right, 0.0))
        self.assertTrue(np.allclose(combined, 0.0))

        valid, msg = validate_dual_hand_features(combined, meta)
        self.assertTrue(valid, msg)

    def test_single_left_hand_extraction(self):
        """Single left hand populates left half of 126-feature vector."""
        left, right, combined, meta = extract_dual_hand_landmarks(
            [self.left_lms], [self.left_handedness]
        )
        self.assertEqual(meta["hand_count"], 1)
        self.assertTrue(meta["has_left"])
        self.assertFalse(meta["has_right"])
        self.assertFalse(np.allclose(left, 0.0))
        self.assertTrue(np.allclose(right, 0.0))
        self.assertTrue(np.allclose(combined[:63], left))
        self.assertTrue(np.allclose(combined[63:], 0.0))

        valid, msg = validate_dual_hand_features(combined, meta)
        self.assertTrue(valid, msg)

    def test_single_right_hand_extraction(self):
        """Single right hand populates right half of 126-feature vector."""
        left, right, combined, meta = extract_dual_hand_landmarks(
            [self.right_lms], [self.right_handedness]
        )
        self.assertEqual(meta["hand_count"], 1)
        self.assertFalse(meta["has_left"])
        self.assertTrue(meta["has_right"])
        self.assertTrue(np.allclose(left, 0.0))
        self.assertFalse(np.allclose(right, 0.0))
        self.assertTrue(np.allclose(combined[:63], 0.0))
        self.assertTrue(np.allclose(combined[63:], right))

        valid, msg = validate_dual_hand_features(combined, meta)
        self.assertTrue(valid, msg)

    def test_dual_hands_correct_ordering(self):
        """Both hands extracted with deterministic [left, right] concatenation."""
        left, right, combined, meta = extract_dual_hand_landmarks(
            [self.left_lms, self.right_lms],
            [self.left_handedness, self.right_handedness]
        )
        self.assertEqual(meta["hand_count"], 2)
        self.assertTrue(meta["has_left"])
        self.assertTrue(meta["has_right"])
        self.assertFalse(meta["is_ambiguous"])
        self.assertEqual(combined.shape, (126,))
        self.assertTrue(np.allclose(combined[:63], left))
        self.assertTrue(np.allclose(combined[63:], right))

        valid, msg = validate_dual_hand_features(combined, meta)
        self.assertTrue(valid, msg)

    def test_dual_hands_reverse_input_resilience(self):
        """Passing [right, left] still produces deterministic [left, right] representation."""
        left, right, combined, meta = extract_dual_hand_landmarks(
            [self.right_lms, self.left_lms],
            [self.right_handedness, self.left_handedness]
        )
        self.assertEqual(meta["hand_count"], 2)
        self.assertTrue(meta["has_left"])
        self.assertTrue(meta["has_right"])
        self.assertFalse(meta["is_ambiguous"])

        # Compare directly with standard left normalization
        expected_left = normalize_hand_landmarks(self.left_lms)
        expected_right = normalize_hand_landmarks(self.right_lms)
        self.assertTrue(np.allclose(left, expected_left))
        self.assertTrue(np.allclose(right, expected_right))
        self.assertTrue(np.allclose(combined[:63], expected_left))
        self.assertTrue(np.allclose(combined[63:], expected_right))

    def test_ambiguous_handedness_spatial_fallback(self):
        """When MediaPipe labels are identical, falls back to x-coordinate sorting."""
        # Both labeled "Right"
        both_right_0 = MockHandedness("Right", score=0.8)
        both_right_1 = MockHandedness("Right", score=0.8)

        # left_lms has wrist_x=0.3, right_lms has wrist_x=0.7
        left, right, combined, meta = extract_dual_hand_landmarks(
            [self.right_lms, self.left_lms],  # passed in right, left order
            [both_right_0, both_right_1]
        )
        self.assertEqual(meta["hand_count"], 2)
        self.assertTrue(meta["is_ambiguous"])
        # The hand with smaller wrist x (0.3) must be assigned to left_feats
        expected_left = normalize_hand_landmarks(self.left_lms)
        expected_right = normalize_hand_landmarks(self.right_lms)
        self.assertTrue(np.allclose(left, expected_left))
        self.assertTrue(np.allclose(right, expected_right))


class TestFeatureValidation(unittest.TestCase):
    """Test validation of feature dimensions, finiteness, and NaN/Inf rejection."""

    def test_validate_single_hand_success(self):
        feats = np.ones(63, dtype=np.float32) * 0.5
        ok, msg = validate_single_hand_features(feats)
        self.assertTrue(ok)
        self.assertEqual(msg, "OK")

    def test_validate_single_hand_wrong_dimension(self):
        feats = np.ones(126, dtype=np.float32)
        ok, msg = validate_single_hand_features(feats)
        self.assertFalse(ok)
        self.assertIn("(63,)", msg)

    def test_validate_single_hand_nan_inf(self):
        feats = np.ones(63, dtype=np.float32)
        feats[5] = np.nan
        ok, msg = validate_single_hand_features(feats)
        self.assertFalse(ok)
        self.assertIn("non-finite", msg)

        feats[5] = np.inf
        ok, msg = validate_single_hand_features(feats)
        self.assertFalse(ok)
        self.assertIn("non-finite", msg)

    def test_validate_dual_hand_dimension_mismatch(self):
        feats = np.zeros(63, dtype=np.float32)
        ok, msg = validate_dual_hand_features(feats, {"has_left": False, "has_right": False})
        self.assertFalse(ok)
        self.assertIn("(126,)", msg)

    def test_validate_dual_hand_consistency(self):
        feats = np.zeros(126, dtype=np.float32)
        feats[10] = 1.0  # left hand has non-zero features but has_left is False
        ok, msg = validate_dual_hand_features(feats, {"has_left": False, "has_right": False})
        self.assertFalse(ok)
        self.assertIn("Left hand flagged absent", msg)


class TestAlphabetEngineDualHand(unittest.TestCase):
    """Test AlphabetEngine state machine and prevention of false emissions on dual hands."""

    def setUp(self):
        self.engine = AlphabetEngine()
        self.left_lms = create_mock_hand_landmarks(wrist_x=0.3, wrist_y=0.6, scale=0.1)
        self.right_lms = create_mock_hand_landmarks(wrist_x=0.7, wrist_y=0.6, scale=0.1)
        self.left_handedness = MockHandedness("Left", score=0.95)
        self.right_handedness = MockHandedness("Right", score=0.95)

    def test_zero_hands_resets_state(self):
        """Zero hands resets debounce and sets state to NO_HAND."""
        status = self.engine.process_dual_landmarks([], [])
        self.assertEqual(status["hand_count"], 0)
        self.assertEqual(status["hand_status"], "NO HAND")
        self.assertFalse(status["hand_detected"])
        self.assertEqual(status["state"], "NO_HAND")

    def test_two_hands_prevents_false_letter_emission(self):
        """Continuous two-hand detection holds safe state and emits NO letters."""
        self.engine.word_buffer.clear_all()

        for t in range(15):
            status = self.engine.process_dual_landmarks(
                [self.left_lms, self.right_lms],
                [self.left_handedness, self.right_handedness],
                current_time=float(t) * 0.05
            )
            self.assertEqual(status["hand_count"], 2)
            self.assertEqual(status["hand_status"], "2 HANDS")
            self.assertTrue(status["hand_detected"])
            self.assertIsNone(status["raw_prediction"])
            self.assertIsNone(status["stable_letter"])
            self.assertIsNone(status["emitted_letter"])
            self.assertIn("2 HANDS", status["state"])
            self.assertEqual(self.engine.word_buffer.get_current_word(), "")

        self.assertEqual(self.engine.word_buffer.get_current_word(), "")
        self.assertEqual(self.engine.word_buffer.get_sentence(), "")

    def test_hand_count_transitions_lifecycle(self):
        """0 hands -> 1 hand -> 2 hands -> 1 hand -> 0 hands transitions cleanly."""
        # 1. Start with 0 hands
        s0 = self.engine.process_dual_landmarks([], [], current_time=0.0)
        self.assertEqual(s0["hand_count"], 0)
        self.assertEqual(s0["state"], "NO_HAND")

        # 2. Transition to 1 hand
        s1 = self.engine.process_dual_landmarks([self.right_lms], [self.right_handedness], current_time=0.05)
        self.assertEqual(s1["hand_count"], 1)
        self.assertEqual(s1["hand_status"], "1 HAND")
        self.assertTrue(s1["hand_detected"])

        # 3. Transition to 2 hands
        s2 = self.engine.process_dual_landmarks(
            [self.left_lms, self.right_lms],
            [self.left_handedness, self.right_handedness],
            current_time=0.10
        )
        self.assertEqual(s2["hand_count"], 2)
        self.assertEqual(s2["hand_status"], "2 HANDS")
        self.assertIn("2 HANDS", s2["state"])

        # 4. Transition back to 1 hand
        s3 = self.engine.process_dual_landmarks([self.right_lms], [self.right_handedness], current_time=0.15)
        self.assertEqual(s3["hand_count"], 1)
        self.assertEqual(s3["hand_status"], "1 HAND")

        # 5. Transition back to 0 hands
        s4 = self.engine.process_dual_landmarks([], [], current_time=0.20)
        self.assertEqual(s4["hand_count"], 0)
        self.assertEqual(s4["state"], "NO_HAND")

    def test_backward_compatibility_process_landmarks(self):
        """Legacy process_landmarks interface remains functional and includes hand metadata."""
        status = self.engine.process_landmarks(self.right_lms, current_time=1.0)
        self.assertIn("hand_count", status)
        self.assertIn("hand_status", status)
        self.assertEqual(status["hand_count"], 1)
        self.assertEqual(status["hand_status"], "1 HAND")
        self.assertTrue(status["hand_detected"])

        status_none = self.engine.process_landmarks(None, current_time=1.1)
        self.assertEqual(status_none["hand_count"], 0)
        self.assertEqual(status_none["hand_status"], "NO HAND")
        self.assertFalse(status_none["hand_detected"])


if __name__ == "__main__":
    unittest.main()
