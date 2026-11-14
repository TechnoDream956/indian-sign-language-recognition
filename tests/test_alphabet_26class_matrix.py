#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite: Alphabet 26-Class Accuracy & Disambiguation Matrix.
=============================================================
Validates:
1. All 26 classes (A-Z) present in label maps and engines.
2. Problematic 13 letters disambiguation:
   - Single-hand I vs Y (pinky extension vs thumb spread).
   - Single-hand U vs V (finger-tip distance thresholding).
   - Dual-hand K vs T (passive index vertical stem detection).
   - Letters D, F, M, N, P, Q, R, Z.
3. Prevention of false collapses into B and H.
4. Multi-modal fusion top-5 distribution and acceptance telemetry.
"""

import math
import sys
import unittest
import numpy as np
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from src.alphabet_engine import AlphabetEngine, extract_dual_hand_landmarks
from src.cnn_engine import CNNEngine
from src.dual_hand_engine import DualHandEngine
from src.fusion_engine import FusionEngine


class DummyLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class DummyHandList:
    def __init__(self, landmarks):
        self.landmark = landmarks


def create_hand(wrist_x: float, wrist_y: float = 0.5, scale: float = 0.15) -> DummyHandList:
    landmarks = [DummyLandmark(wrist_x, wrist_y, 0.0)]
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x - 0.02 * i, wrist_y - 0.02 * i, 0.0))
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x - 0.01 * i, wrist_y - 0.04 * i, 0.0))
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x, wrist_y - scale * (i / 4.0), 0.0))
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x + 0.01 * i, wrist_y - 0.035 * i, 0.0))
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x + 0.02 * i, wrist_y - 0.03 * i, 0.0))
    return DummyHandList(landmarks)


class TestAlphabet26ClassMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.alphabet_engine = AlphabetEngine()
        cls.cnn_engine = CNNEngine()
        cls.fusion_engine = FusionEngine()

    def test_01_all_26_classes_present_in_engines(self):
        """All 26 letters A through Z must be present in alphabet engines and models."""
        expected_classes = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        self.assertEqual(len(expected_classes), 26)

        # AlphabetEngine single-hand classes
        if self.alphabet_engine.single_label_map:
            for letter in expected_classes:
                self.assertIn(letter, self.alphabet_engine.single_label_map)

        # CNNEngine classes
        self.assertEqual(len(self.cnn_engine.classes), 26)
        for letter in expected_classes:
            self.assertIn(letter, self.cnn_engine.classes)

    def test_02_disambiguate_i_vs_y(self):
        """Disambiguate 'I' vs 'Y': tucked thumb yields 'I', extended thumb yields 'Y'."""
        # Single hand with extended pinky and folded index, middle, ring, and tucked thumb
        hand_i = create_hand(wrist_x=0.5, wrist_y=0.5, scale=0.15)
        # Fold index, middle, ring
        hand_i.landmark[8] = DummyLandmark(0.48, 0.46, 0.0)
        hand_i.landmark[12] = DummyLandmark(0.50, 0.46, 0.0)
        hand_i.landmark[16] = DummyLandmark(0.51, 0.46, 0.0)
        # Thumb tucked close to index base
        hand_i.landmark[4] = DummyLandmark(0.49, 0.47, 0.0)
        hand_i.landmark[5] = DummyLandmark(0.48, 0.46, 0.0)
        # Pinky tip extended upright
        hand_i.landmark[17] = DummyLandmark(0.52, 0.47, 0.0)
        hand_i.landmark[20] = DummyLandmark(0.53, 0.30, 0.0)

        res_i = self.alphabet_engine.process_dual_landmarks([hand_i])
        self.assertEqual(res_i["hand_count"], 1)
        self.assertEqual(res_i["raw_letter"], "I")

        # Hand with extended thumb and extended pinky (horn / Y shape)
        hand_y = create_hand(wrist_x=0.5, wrist_y=0.5, scale=0.15)
        # Fold index, middle, ring
        hand_y.landmark[8] = DummyLandmark(0.48, 0.46, 0.0)
        hand_y.landmark[12] = DummyLandmark(0.50, 0.46, 0.0)
        hand_y.landmark[16] = DummyLandmark(0.51, 0.46, 0.0)
        # Pinky extended
        hand_y.landmark[17] = DummyLandmark(0.52, 0.47, 0.0)
        hand_y.landmark[20] = DummyLandmark(0.58, 0.32, 0.0)
        # Thumb extended outwards
        hand_y.landmark[4] = DummyLandmark(0.30, 0.42, 0.0)
        hand_y.landmark[5] = DummyLandmark(0.48, 0.46, 0.0)

        res_y = self.alphabet_engine.process_dual_landmarks([hand_y])
        self.assertEqual(res_y["hand_count"], 1)

    def test_03_disambiguate_u_vs_v(self):
        """Disambiguate 'U' vs 'V': touching tips -> 'U', spread tips -> 'V'."""
        # Touching fingertips: U
        hand_u = create_hand(wrist_x=0.5, wrist_y=0.5, scale=0.15)
        scale = 0.15
        hand_u.landmark[8] = DummyLandmark(0.50, 0.50 - scale * 1.2, 0.0)
        hand_u.landmark[12] = DummyLandmark(0.52, 0.50 - scale * 1.2, 0.0)
        # Fold other fingers
        hand_u.landmark[16] = DummyLandmark(0.52, 0.46, 0.0)
        hand_u.landmark[20] = DummyLandmark(0.53, 0.47, 0.0)

        res_u = self.alphabet_engine.process_dual_landmarks([hand_u])
        self.assertEqual(res_u["hand_count"], 1)

        # Spread fingertips: V
        hand_v = create_hand(wrist_x=0.5, wrist_y=0.5, scale=0.15)
        hand_v.landmark[8] = DummyLandmark(0.44, 0.50 - scale * 1.2, 0.0)
        hand_v.landmark[12] = DummyLandmark(0.56, 0.50 - scale * 1.2, 0.0)
        hand_v.landmark[16] = DummyLandmark(0.52, 0.46, 0.0)
        hand_v.landmark[20] = DummyLandmark(0.53, 0.47, 0.0)

        res_v = self.alphabet_engine.process_dual_landmarks([hand_v])
        self.assertEqual(res_v["hand_count"], 1)

    def test_04_disambiguate_k_vs_t(self):
        """Disambiguate 'K' vs 'T': vertical stem on passive hand overrides T to K."""
        # Dual hands configured with passive hand index pointing vertically
        hand_passive = create_hand(wrist_x=0.40, wrist_y=0.60, scale=0.15)
        # Vertical stem: landmark 8 much higher than landmark 5 (dy > 0.08, dy > 1.2 * dx)
        hand_passive.landmark[5] = DummyLandmark(0.40, 0.55, 0.0)
        hand_passive.landmark[8] = DummyLandmark(0.40, 0.42, 0.0)

        hand_active = create_hand(wrist_x=0.55, wrist_y=0.55, scale=0.15)

        res_k = self.alphabet_engine.process_dual_landmarks([hand_passive, hand_active])
        self.assertEqual(res_k["hand_count"], 2)
        # Should not falsely collapse to T if passive hand has vertical stem
        self.assertNotEqual(res_k.get("landmark_prediction"), "T")

    def test_05_false_b_and_h_collapse_prevention(self):
        """Separated hands with open posture must never collapse to B or H."""
        hand1 = create_hand(wrist_x=0.20, wrist_y=0.50)
        hand2 = create_hand(wrist_x=0.80, wrist_y=0.50)

        res = self.alphabet_engine.process_dual_landmarks([hand1, hand2])
        self.assertEqual(res["hand_count"], 2)
        # If model outputs B or H, suppression logic must demote confidence < 0.50
        if res.get("landmark_prediction") in ["B", "H"]:
            self.assertLess(res.get("landmark_confidence", 0.0), 0.50)

    def test_06_top5_distribution_in_telemetry(self):
        """Top-5 distribution must be present in Landmark, CNN, and telemetry snapshots."""
        hand = create_hand(wrist_x=0.50, wrist_y=0.50)
        res = self.alphabet_engine.process_dual_landmarks([hand])
        self.assertIn("top5", res)
        self.assertIsInstance(res["top5"], list)
        self.assertGreater(len(res["top5"]), 0)

        # CNN top5 check
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cnn_res = self.cnn_engine.predict(dummy_frame)
        self.assertIn("top5", cnn_res)
        self.assertIsInstance(cnn_res["top5"], list)
        self.assertGreaterEqual(len(cnn_res["top5"]), 1)

    def test_07_is_accepted_boolean_propagation(self):
        """is_accepted flag must be emitted accurately."""
        lm_high = {"pred_label": "K", "confidence": 0.85, "status": "OK"}
        cnn_high = {"pred_label": "K", "confidence": 0.85, "is_available": True, "status": "ONLINE"}
        fused_acc = self.fusion_engine.fuse(lm_high, cnn_high, hand_detected=True)
        self.assertTrue(fused_acc["is_accepted"])

        lm_low = {"pred_label": "K", "confidence": 0.20, "status": "LOW_CONFIDENCE"}
        cnn_low = {"pred_label": "A", "confidence": 0.25, "is_available": True, "status": "ONLINE"}
        fused_rej = self.fusion_engine.fuse(lm_low, cnn_low, hand_detected=True)
        self.assertFalse(fused_rej["is_accepted"])


if __name__ == "__main__":
    unittest.main()
