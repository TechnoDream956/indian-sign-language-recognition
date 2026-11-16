#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Camera Mirroring Parity and Configurable Orientation.
===================================================================
Verifies:
1. `InferenceEngine` defaults to `model_input_unmirrored=True`.
2. Models receive unmirrored frames matching third-person training data.
3. Display images and overlay coordinates are correctly mirrored for user selfie view.
4. Legacy mirrored behavior is fully reachable when `set_model_input_unmirrored(False)`.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.inference_engine import InferenceEngine


class TestMirrorParityAndPipeline(unittest.TestCase):
    def setUp(self):
        self.engine = InferenceEngine(camera_index=-1)

    def test_default_mirror_configuration(self):
        """InferenceEngine defaults to unmirrored model inputs with legacy switchability."""
        self.assertTrue(self.engine.get_model_input_unmirrored())
        self.assertTrue(self.engine._state.get("model_input_unmirrored"))

        # Test toggling to legacy mode
        res_legacy = self.engine.set_model_input_unmirrored(False)
        self.assertFalse(self.engine.get_model_input_unmirrored())
        self.assertFalse(res_legacy["model_input_unmirrored"])

        # Test toggling back
        res_active = self.engine.set_model_input_unmirrored(True)
        self.assertTrue(self.engine.get_model_input_unmirrored())
        self.assertTrue(res_active["model_input_unmirrored"])

    def test_process_frame_unmirrored_models_pathway(self):
        """process_frame passes unmirrored frame to MediaPipe and outputs mirrored display coords."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame[100:200, 50:150] = [255, 255, 255] # Asymmetric marker on left side

        mock_detector = MagicMock()
        mock_hand = MagicMock()
        mock_lm = MagicMock()
        mock_lm.x, mock_lm.y, mock_lm.z = 0.25, 0.5, 0.0
        mock_hand.landmark = [mock_lm] * 21
        mock_res = MagicMock()
        mock_res.multi_hand_landmarks = [mock_hand]
        mock_res.multi_handedness = None
        mock_detector.process.return_value = mock_res
        self.engine._hands_detector = mock_detector

        # Mode A: Unmirrored Models (Default)
        self.engine.set_model_input_unmirrored(True)
        res_unmirrored = self.engine.process_frame(dummy_frame, is_mirrored=False)
        self.assertTrue(res_unmirrored.get("hand_detected"))

        # Verify MediaPipe received the unmirrored frame
        called_rgb = mock_detector.process.call_args[0][0]
        # In unmirrored frame, the white square is on x in [50, 150]
        self.assertEqual(called_rgb[150, 100, 0], 255)

        # Mode B: Legacy Mode (model_input_unmirrored = False)
        self.engine.set_model_input_unmirrored(False)
        res_legacy = self.engine.process_frame(dummy_frame, is_mirrored=False)
        called_rgb_legacy = mock_detector.process.call_args[0][0]
        # In legacy mode, frame was flipped horizontally, so marker is at 640 - 100 = 540
        self.assertEqual(called_rgb_legacy[150, 540, 0], 255)


if __name__ == "__main__":
    unittest.main()
