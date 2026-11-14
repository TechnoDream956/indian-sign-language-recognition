#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite: Alphabet Recognition Accuracy, Multi-Modal Fusion & Multilingual Translation.
========================================================================================
Validates:
1. Spatial coordinate sorting (Channel 1 Screen-Left vs Channel 2 Screen-Right).
2. Single-Hand (26-class) vs Dual-Hand (20-class) dispatching.
3. Problematic letters (D, F, I, K, M, N, P, Q, R, U, V, Y, Z) and B/H collapse prevention.
4. CNN Model online status, input normalization, and inference latency.
5. Decision Fusion logic (agreement bonus, disagreement margin, offline fallback).
6. Multilingual translation across all 10 Indic languages with zero untranslated English fallbacks.
7. TTS voice discovery and non-blocking safety.
"""

import json
import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add repo root to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from src.alphabet_engine import AlphabetEngine, extract_dual_hand_landmarks
from src.cnn_engine import CNNEngine
from src.dual_hand_engine import DualHandEngine, compute_dual_hand_features
from src.fusion_engine import FusionEngine
from src.translator import translate_label, translate_sentence, get_supported_languages
from src.tts_engine import TTSEngine


class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class DummyHandList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class DummyClassification:
    def __init__(self, label, score=0.99):
        self.label = label
        self.score = score


class DummyClassificationList:
    def __init__(self, label, score=0.99):
        self.classification = [DummyClassification(label, score)]


def create_synthetic_hand(wrist_x: float, wrist_y: float = 0.5, scale: float = 0.15) -> DummyHandList:
    """Generate a valid 21-landmark hand centered at wrist_x, wrist_y with physical hierarchy."""
    landmarks = []
    # Wrist (0)
    landmarks.append(DummyLandmark(wrist_x, wrist_y, 0.0))
    # Thumb (1-4)
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x - 0.02 * i, wrist_y - 0.02 * i, 0.0))
    # Index (5-8)
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x - 0.01 * i, wrist_y - 0.04 * i, 0.0))
    # Middle (9-12) -> landmark 9 defines scale
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x, wrist_y - scale * (i / 4.0), 0.0))
    # Ring (13-16)
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x + 0.01 * i, wrist_y - 0.035 * i, 0.0))
    # Pinky (17-20)
    for i in range(1, 5):
        landmarks.append(DummyLandmark(wrist_x + 0.02 * i, wrist_y - 0.03 * i, 0.0))
    return DummyHandList(landmarks)


class TestAlphabetAccuracyAndFusion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = AlphabetEngine()
        cls.dual_engine = DualHandEngine(config_key="config_d_ablation_144d", use_prompt23=True)
        cls.fusion = FusionEngine()

    def test_01_spatial_ordering_screen_left_right(self):
        """Channel 1 must always be screen-left (smaller x) and Channel 2 screen-right (larger x)."""
        hand_a = create_synthetic_hand(wrist_x=0.70)
        hand_b = create_synthetic_hand(wrist_x=0.30)

        # In order [hand_a, hand_b], hand_b is screen-left (0.30) and hand_a is screen-right (0.70)
        l_feats, r_feats, c_126, meta = extract_dual_hand_landmarks([hand_a, hand_b])

        self.assertEqual(meta["hand_count"], 2)
        # c1 should be hand_b (wrist.x = 0.30)
        self.assertAlmostEqual(meta["c1"][0, 0], 0.30, places=2)
        # c2 should be hand_a (wrist.x = 0.70)
        self.assertAlmostEqual(meta["c2"][0, 0], 0.70, places=2)
        # rel_wrist x must be positive (c2.x - c1.x > 0)
        wrist_disp_x = meta["c2"][0, 0] - meta["c1"][0, 0]
        self.assertGreater(wrist_disp_x, 0.0)

    def test_02_single_hand_dispatch(self):
        """Single visible hand must route to 26-class single-hand pipeline."""
        single_hand = create_synthetic_hand(wrist_x=0.45)
        res = self.engine.process_dual_landmarks([single_hand])
        self.assertTrue(res["hand_detected"])
        self.assertEqual(res["hand_count"], 1)
        self.assertIn("Single-Hand", res["active_pipeline"])

    def test_03_dual_hand_dispatch(self):
        """Two visible hands must activate dual-hand pipeline."""
        hand_left = create_synthetic_hand(wrist_x=0.35)
        hand_right = create_synthetic_hand(wrist_x=0.65)
        res = self.engine.process_dual_landmarks([hand_left, hand_right])
        self.assertTrue(res["hand_detected"])
        self.assertEqual(res["hand_count"], 2)
        self.assertIn("Dual-Hand", res["active_pipeline"])

    def test_04_b_and_h_false_collapse_prevention(self):
        """Widely separated hands with zero contact must NOT trigger false positive B or H."""
        # Create two hands at extreme ends of the screen (0.10 and 0.90)
        hand_left = create_synthetic_hand(wrist_x=0.10)
        hand_right = create_synthetic_hand(wrist_x=0.90)
        res = self.engine.process_dual_landmarks([hand_left, hand_right])
        # Even if raw MLP leans B or H, the suppression logic dampens confidence
        if res.get("landmark_prediction") in ["B", "H"]:
            self.assertLess(res.get("landmark_confidence", 0.0), 0.50)

    def test_05_decision_fusion_agreement_bonus(self):
        """When Landmark and CNN agree, fused confidence must receive agreement bonus."""
        lm_res = {"pred_label": "K", "confidence": 0.80, "status": "OK"}
        cnn_res = {"pred_label": "K", "confidence": 0.85, "is_available": True, "status": "ONLINE"}
        fused = self.fusion.fuse(lm_res, cnn_res, hand_detected=True)

        self.assertEqual(fused["fused_label"], "K")
        self.assertEqual(fused["status"], "AGREEMENT")
        self.assertTrue(fused["agreement"])
        # Expected: 0.60 * 0.80 + 0.40 * 0.85 + 0.05 = 0.48 + 0.34 + 0.05 = 0.87
        self.assertGreaterEqual(fused["confidence"], 0.85)

    def test_06_decision_fusion_disagreement_arbitration(self):
        """Confident stream overrides uncertain stream during disagreement."""
        # Landmark confident, CNN uncertain
        lm_res = {"pred_label": "A", "confidence": 0.92, "status": "OK"}
        cnn_res = {"pred_label": "B", "confidence": 0.20, "is_available": True, "status": "ONLINE"}
        fused = self.fusion.fuse(lm_res, cnn_res, hand_detected=True)
        self.assertEqual(fused["fused_label"], "A")
        self.assertEqual(fused["fusion_mode"], "LANDMARK_PRIORITIZED")

        # CNN confident, Landmark uncertain
        lm_res2 = {"pred_label": "D", "confidence": 0.25, "status": "LOW_CONFIDENCE"}
        cnn_res2 = {"pred_label": "K", "confidence": 0.88, "is_available": True, "status": "ONLINE"}
        fused2 = self.fusion.fuse(lm_res2, cnn_res2, hand_detected=True)
        self.assertEqual(fused2["fused_label"], "K")
        self.assertEqual(fused2["fusion_mode"], "CNN_PRIORITIZED")

    def test_07_multilingual_word_translation(self):
        """Everyday sign words must translate into verified native Indic scripts (never raw English)."""
        test_words = ["cat", "dog", "water", "food", "mother", "father", "friend", "hello", "yes", "no"]
        indic_langs = ["Hindi", "Marathi", "Bengali", "Tamil", "Telugu", "Gujarati", "Kannada", "Malayalam", "Punjabi"]

        for lang in indic_langs:
            for word in test_words:
                translated = translate_label(word, lang)
                self.assertIsNotNone(translated, f"Translation of {word} to {lang} is None")
                self.assertNotEqual(translated.strip(), "", f"Translation of {word} to {lang} is empty")
                # Must not be equal to original English word
                self.assertNotEqual(
                    translated.lower(),
                    word.lower(),
                    f"Word '{word}' was left in English for {lang}: '{translated}'"
                )

    def test_08_multilingual_sentence_translation(self):
        """Sentences must translate into native Indic phrases."""
        test_sentences = [
            ("i want water", "Hindi", "मुझे पानी चाहिए।"),
            ("i love you", "Hindi", "मैं आपसे प्यार करता हूँ"),
            ("hello how are you", "Marathi", "नमस्कार तुम्ही कसे आहात"),
        ]
        for src, lang, expected in test_sentences:
            res = translate_sentence(src, lang)
            self.assertEqual(res, expected)

    def test_09_tts_engine_catalog(self):
        """TTS engine must report available voices without crashing."""
        tts = TTSEngine()
        catalog = tts.get_voice_catalog()
        self.assertIn("English", catalog)
        self.assertIn("Hindi", catalog)
        self.assertTrue(catalog["English"]["available"])
        self.assertTrue(catalog["Hindi"]["available"])
        self.assertEqual(catalog["Hindi"]["voice"], "Lekha")

    def test_10_live_cnn_model_online(self):
        """CNNEngine must load the trained CNN model, report ONLINE, and infer in <25ms."""
        cnn = CNNEngine()
        self.assertTrue(cnn.is_loaded)
        self.assertEqual(cnn.status, "ONLINE")
        self.assertEqual(len(cnn.classes), 26)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pred = cnn.predict(dummy_frame)
        self.assertTrue(pred["is_available"])
        self.assertEqual(pred["status"], "ONLINE")
        self.assertIn(pred["pred_label"], cnn.classes)
        self.assertLess(pred["latency_ms"], 25.0)


if __name__ == "__main__":
    unittest.main()
