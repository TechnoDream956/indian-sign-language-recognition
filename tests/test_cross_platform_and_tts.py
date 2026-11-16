#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Cross-Platform Reliability, Multi-Tier TTS Fallback Chain,
and Translation Retry/Backoff Hardening.
====================================================================
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.tts_engine import TTSEngine, is_headless_environment, play_audio_file_cross_platform
from src.translator import translate_sentence, _translate_online_safe
from src.translation_adapter import TranslationAdapter
from src.inference_engine import InferenceEngine


class TestCrossPlatformAndTTS(unittest.TestCase):
    """Test suite for Phase 3 cross-platform reliability and speech/translation hardening."""

    def setUp(self):
        self.tts = TTSEngine()

    def test_voice_catalog_contains_all_10_languages(self):
        """Verify that voice catalog reports all 10 supported Indic languages with valid tiers."""
        catalog = self.tts.get_voice_catalog()
        expected = [
            "English", "Hindi", "Marathi", "Bengali", "Tamil",
            "Telugu", "Gujarati", "Kannada", "Malayalam", "Punjabi",
        ]
        for lang in expected:
            self.assertIn(lang, catalog)
            self.assertTrue(catalog[lang]["available"])
            self.assertIn("tier", catalog[lang])
            self.assertIn("voice", catalog[lang])

    def test_tts_speak_empty_text(self):
        """Verify that speaking empty text returns clean fallback result without crashing."""
        res = self.tts.speak_with_result("", "English")
        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["reason"], "empty_text")

    def test_tts_headless_environment_delegation(self):
        """Verify that in headless environment, TTS immediately delegates to browser Web Speech API."""
        with patch.dict(os.environ, {"ISL_HEADLESS": "1"}):
            headless_tts = TTSEngine()
            res = headless_tts.speak_with_result("Hello", "English")
            self.assertFalse(res["success"])
            self.assertEqual(res["fallback"], "browser")
            self.assertEqual(res["engine"], "browser")
            self.assertEqual(res["reason"], "headless_environment")

    def test_tts_is_voice_available(self):
        """Verify voice availability checks across tiers."""
        self.assertTrue(self.tts.is_voice_available("English"))
        self.assertTrue(self.tts.is_voice_available("Hindi"))
        self.assertTrue(self.tts.is_voice_available("Gujarati"))
        self.assertTrue(self.tts.is_voice_available("Malayalam"))

    def test_translation_retry_and_backoff_exhaustion(self):
        """Verify that online translation retries with backoff and falls back gracefully on network error."""
        with patch("time.sleep") as mock_sleep:
            with patch("deep_translator.MyMemoryTranslator.translate", side_effect=Exception("Connection refused")):
                res = _translate_online_safe("NON_EXISTENT_SAMPLE_PHRASE", "Hindi", max_retries=2)
                self.assertIsNone(res)
                # Should have slept twice (attempt 0 -> sleep 0.25, attempt 1 -> sleep 0.5)
                self.assertEqual(mock_sleep.call_count, 2)

    def test_translation_offline_fallback(self):
        """Verify that offline dictionary translations work without network dependency."""
        self.assertEqual(translate_sentence("HELLO", "Hindi"), "नमस्ते")
        self.assertEqual(translate_sentence("THANK YOU", "Hindi"), "धन्यवाद")
        self.assertEqual(translate_sentence("BALL", "Marathi"), "चेंडू")

    def test_camera_backend_priority_selection(self):
        """Verify camera backend probe configures appropriate platform backends."""
        inf = InferenceEngine(camera_index=-1)
        with patch("sys.platform", "darwin"):
            with patch("cv2.VideoCapture") as mock_cap:
                mock_instance = MagicMock()
                mock_instance.isOpened.return_value = True
                mock_cap.return_value = mock_instance
                cap = inf._open_camera_capture()
                self.assertIsNotNone(cap)
                self.assertTrue(mock_cap.called)

        with patch("sys.platform", "win32"):
            with patch("cv2.VideoCapture") as mock_cap:
                mock_instance = MagicMock()
                mock_instance.isOpened.return_value = True
                mock_cap.return_value = mock_instance
                cap = inf._open_camera_capture()
                self.assertIsNotNone(cap)
                self.assertTrue(mock_cap.called)

        with patch("sys.platform", "linux"):
            with patch("cv2.VideoCapture") as mock_cap:
                mock_instance = MagicMock()
                mock_instance.isOpened.return_value = True
                mock_cap.return_value = mock_instance
                cap = inf._open_camera_capture()
                self.assertIsNotNone(cap)
                self.assertTrue(mock_cap.called)


if __name__ == "__main__":
    unittest.main()
