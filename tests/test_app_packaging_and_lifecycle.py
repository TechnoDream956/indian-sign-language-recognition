#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for App Packaging, Lifecycle Coordinator, Clean Camera Release, and Alphabet-Only Validation.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, engine, cancel_disconnect_timer
from inference_engine import InferenceEngine
from alphabet_engine import AlphabetEngine, WordBuffer


class TestWordModeAbsenceInUI(unittest.TestCase):
    """Verify that Word Mode and obsolete selectors are completely absent from the rendered UI."""

    def setUp(self):
        self.client = app.test_client()

    def test_rendered_index_has_no_word_mode(self):
        """Rendered index.html must contain no Word Mode tab or MoP-GRU selectors."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertNotIn("Word Mode", html)
        self.assertNotIn("Legacy 13", html)
        self.assertNotIn("Expanded 29", html)
        self.assertNotIn("mopgru", html.lower())
        self.assertIn("Alphabet Mode", html)
        self.assertIn("26 Letters", html)

    def test_rendered_index_has_quit_app_button(self):
        """Rendered index.html must contain the Quit App button and shutdown overlay."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        self.assertIn("btn-quit-app", html)
        self.assertIn("Quit App", html)
        self.assertIn("app-shutdown-overlay", html)


class TestHandCountAndDualHandSafety(unittest.TestCase):
    """Verify hand detection reports 0, 1, and 2 hands and suppresses emissions for 2 hands."""

    def setUp(self):
        self.alphabet_engine = AlphabetEngine()

    def test_zero_hands_state(self):
        """0 hands reports NO HAND and clears stale state."""
        status = self.alphabet_engine.process_dual_landmarks(None, None)
        self.assertEqual(status["hand_count"], 0)
        self.assertEqual(status["hand_status"], "NO HAND")
        self.assertFalse(status["hand_detected"])
        self.assertEqual(status["state"], "NO_HAND")
        self.assertIsNone(status["raw_prediction"])

    def test_two_hands_suppresses_letter_emission(self):
        """2 hands reports '2 HANDS — DUAL-HAND MODEL UNAVAILABLE' and suppresses predictions."""
        # Create mock 21 landmarks for 2 hands
        class MockLandmark:
            def __init__(self, x=0.5, y=0.5, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        class MockClassification:
            def __init__(self, label="Left", score=0.95):
                self.label = label
                self.score = score

        class MockHandedness:
            def __init__(self, label="Left", score=0.95):
                self.classification = [MockClassification(label, score)]

        left_lms = [MockLandmark(0.3, 0.5, 0.0) for _ in range(21)]
        right_lms = [MockLandmark(0.7, 0.5, 0.0) for _ in range(21)]
        handedness = [MockHandedness("Left"), MockHandedness("Right")]

        # Verify fallback when dual hand model is explicitly None
        orig_dual = self.alphabet_engine.dual_hand_engine
        self.alphabet_engine.dual_hand_engine = None
        try:
            status = self.alphabet_engine.process_dual_landmarks(
                [left_lms, right_lms], handedness
            )
            self.assertEqual(status["hand_count"], 2)
            self.assertEqual(status["hand_status"], "2 HANDS")
            self.assertEqual(status["state"], "2 HANDS — DUAL-HAND MODEL UNAVAILABLE")
            self.assertIsNone(status["raw_prediction"])
            self.assertIsNone(status["emitted_letter"])
        finally:
            self.alphabet_engine.dual_hand_engine = orig_dual


class TestLifecycleAndShutdownCoordinator(unittest.TestCase):
    """Test camera release, shutdown endpoints, and heartbeat handling."""

    def setUp(self):
        self.client = app.test_client()

    def test_shutdown_endpoint(self):
        """POST /api/shutdown returns 200 with shutdown confirmation."""
        with patch("app._perform_graceful_shutdown"):
            resp = self.client.post("/api/shutdown")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data["ok"])

    def test_disconnect_beacon_and_heartbeat(self):
        """POST /api/client-disconnect starts disconnect timer; /api/heartbeat cancels it."""
        resp1 = self.client.post("/api/client-disconnect")
        self.assertEqual(resp1.status_code, 200)

        # Heartbeat cancels the timer
        resp2 = self.client.post("/api/heartbeat")
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.get_json()["alive"])

    def test_buffer_aliases(self):
        """POST /api/buffer/* endpoints work identically to /api/alphabet/*."""
        engine.alphabet_clear_sentence()
        engine.alphabet_clear_word()

        resp = self.client.post("/api/buffer/space")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post("/api/buffer/backspace")
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post("/api/buffer/reset")
        self.assertEqual(resp.status_code, 200)

    def test_engine_stop_idempotent(self):
        """Calling engine.stop() multiple times must not raise errors."""
        test_engine = InferenceEngine(camera_index=-1)
        test_engine.stop()
        test_engine.stop()  # Second call must be clean
        self.assertFalse(test_engine._running)
        self.assertFalse(test_engine._state["camera_ok"])

    def test_camera_release_on_stop(self):
        """Mock cv2.VideoCapture release is guaranteed when engine stops."""
        test_engine = InferenceEngine(camera_index=-1)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        test_engine._cap = mock_cap

        test_engine.stop()
        mock_cap.release.assert_called()
        self.assertIsNone(test_engine._cap)


if __name__ == "__main__":
    unittest.main()
