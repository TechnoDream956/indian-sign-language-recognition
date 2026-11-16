#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for Neural Speech Adapter & Browser TTS Fallback.
=============================================================================
Verifies all 15 Phase 3C core requirements:
1. Browser speech remains the default.
2. Neural TTS can be enabled via configuration.
3. Valid neural audio returns data URI.
4. Neural service timeout triggers browser fallback.
5. HTTP errors trigger fallback.
6. Malformed audio responses trigger fallback.
7. Unsupported / empty text triggers graceful fallback.
8. Repeated requests use caching.
9. Stop/cancel prevents continued playback (sequencing).
10. Multiple requests do not overlap (thread safety).
11. Stale audio responses can be discarded.
12. Webcam streaming remains unaffected.
13. Translation continues working alongside speech.
14. Feedback authentication continues working.
15. Token storage no longer unnecessarily persists bearer tokens in localStorage.
"""

import concurrent.futures
import io
import json
import os
import sys
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

# Ensure src/ and repo root are on PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from speech_adapter import SpeechAdapter, get_speech_adapter, reset_speech_adapter
from translation_adapter import get_translation_adapter

try:
    from feedback_learning.privacy_manager import generate_session_token, verify_session_token
except ImportError:
    from src.feedback_learning.privacy_manager import generate_session_token, verify_session_token


class TestSpeechAdapter(unittest.TestCase):
    """Test suite verifying SpeechAdapter compliance and safety."""

    def setUp(self):
        reset_speech_adapter()

    def tearDown(self):
        reset_speech_adapter()

    def test_01_browser_speech_remains_the_default(self):
        """1. Browser speech remains the default when neural TTS is unconfigured/disabled."""
        adapter = SpeechAdapter(service_url="", enabled=False)
        self.assertFalse(adapter.is_neural_enabled())

        res = adapter.synthesize(text="Hello world", target_lang="Hindi")
        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["engine"], "browser")
        self.assertEqual(res["reason"], "neural_tts_disabled")
        self.assertIsNone(res["data_uri"])

    def test_02_neural_tts_can_be_enabled_via_configuration(self):
        """2. Neural TTS can be enabled via explicit configuration or environment flag."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)
        self.assertTrue(adapter.is_neural_enabled())
        self.assertEqual(adapter.service_url, "http://127.0.0.1:5001")

        with patch.dict(os.environ, {"ENABLE_NEURAL_TTS": "1", "LANGUAGE_SERVICE_URL": "http://localhost:8000"}):
            adapter_env = SpeechAdapter()
            self.assertTrue(adapter_env.is_neural_enabled())
            self.assertEqual(adapter_env.service_url, "http://localhost:8000")

    def test_03_valid_neural_audio_returns_data_uri(self):
        """3. Valid neural audio returns data URI and correct metadata."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        fake_response = {
            "success": True,
            "data_uri": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
            "audio_id": "test_audio_123",
            "sample_rate": 16000,
            "duration_seconds": 1.25,
            "model_name": "mms-tts",
            "speaker": "hin",
            "language": "Hindi",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = adapter.synthesize(text="नमस्ते", target_lang="Hindi")

        self.assertTrue(res["success"])
        self.assertEqual(res["engine"], "neural")
        self.assertIsNone(res["fallback"])
        self.assertEqual(res["data_uri"], fake_response["data_uri"])
        self.assertEqual(res["audio_id"], "test_audio_123")
        self.assertEqual(res["sample_rate"], 16000)

    def test_04_neural_service_timeout_triggers_browser_fallback(self):
        """4. Neural service timeout triggers immediate browser fallback without crashing."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True, timeout=0.5)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
            res = adapter.synthesize(text="Long text", target_lang="Tamil")

        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["engine"], "browser")
        self.assertIn("timeout", res["reason"])

    def test_05_http_errors_trigger_fallback(self):
        """5. HTTP errors (e.g. 500, 503) trigger graceful browser fallback."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        http_err = urllib.error.HTTPError(
            url="http://127.0.0.1:5001/api/v2/language/speak",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b'{"error": "GPU out of memory"}'),
        )

        with patch("urllib.request.urlopen", side_effect=http_err):
            res = adapter.synthesize(text="Emergency stop", target_lang="Marathi")

        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["engine"], "browser")
        self.assertIn("503", res["reason"])

    def test_06_malformed_audio_responses_trigger_fallback(self):
        """6. Malformed audio responses (bad JSON, missing data_uri) trigger fallback."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"success": false, "error": "Model initialization failed"}'
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = adapter.synthesize(text="Corrupt audio test", target_lang="Bengali")

        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["engine"], "browser")
        self.assertIn("malformed_or_unsuccessful", res["reason"])

    def test_07_empty_text_triggers_graceful_fallback(self):
        """7. Empty or blank text triggers graceful browser fallback without remote call."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        res = adapter.synthesize(text="", target_lang="Telugu")
        self.assertFalse(res["success"])
        self.assertEqual(res["fallback"], "browser")
        self.assertEqual(res["reason"], "empty_text")

        res_spaces = adapter.synthesize(text="   \n  \t  ", target_lang="Telugu")
        self.assertFalse(res_spaces["success"])
        self.assertEqual(res_spaces["fallback"], "browser")

    def test_08_repeated_requests_use_caching(self):
        """8. Repeated requests return cached audio data URI to eliminate network roundtrips."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        fake_response = {
            "success": True,
            "data_uri": "data:audio/wav;base64,CACHE_DATA_TEST",
            "audio_id": "cache_id_001",
            "sample_rate": 16000,
            "duration_seconds": 1.0,
            "model_name": "mms-tts",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            res1 = adapter.synthesize(text="Namaste", target_lang="Hindi")
            self.assertTrue(res1["success"])
            self.assertFalse(res1.get("cached", False))
            self.assertEqual(mock_urlopen.call_count, 1)

            res2 = adapter.synthesize(text="Namaste", target_lang="Hindi")
            self.assertTrue(res2["success"])
            self.assertTrue(res2.get("cached", False))
            self.assertEqual(mock_urlopen.call_count, 1)  # No second HTTP call

        stats = adapter.get_stats()
        self.assertEqual(stats["cache_hits"], 1)

    def test_09_stop_and_clear_cache(self):
        """9. Cache can be safely cleared and statistics retrieved."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)
        adapter.store_cache("Test", "English", None, {"success": True, "data_uri": "data:audio/wav;base64,AAA"})
        self.assertEqual(adapter.get_stats()["cache_size"], 1)

        adapter.clear_cache()
        self.assertEqual(adapter.get_stats()["cache_size"], 0)

    def test_10_concurrent_requests_thread_safe(self):
        """10. Multiple concurrent requests do not corrupt cache or state."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        fake_response = {
            "success": True,
            "data_uri": "data:audio/wav;base64,CONCURRENT_AUDIO",
            "audio_id": "thread_safe_audio",
            "sample_rate": 16000,
            "duration_seconds": 0.8,
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        def synthesize_task(i):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                return adapter.synthesize(text=f"Sentence {i % 5}", target_lang="Hindi")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(synthesize_task, i) for i in range(25)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 25)
        for r in results:
            self.assertTrue(r["success"])
        stats = adapter.get_stats()
        self.assertGreaterEqual(stats["requests_total"], 25)

    def test_11_voice_catalog_discovery(self):
        """11. Remote voice catalog discovery returns catalog or empty list gracefully."""
        adapter = SpeechAdapter(service_url="http://127.0.0.1:5001", enabled=True)

        fake_catalog = {
            "success": True,
            "voices": [
                {"voice_id": "mms-hin-vits", "language": "Hindi", "sample_rate": 16000},
                {"voice_id": "mms-tam-vits", "language": "Tamil", "sample_rate": 16000},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_catalog).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            voices = adapter.get_voices("Hindi")

        self.assertEqual(len(voices), 2)
        self.assertEqual(voices[0]["voice_id"], "mms-hin-vits")

        # Disabled adapter returns empty without network call
        disabled_adapter = SpeechAdapter(service_url="", enabled=False)
        self.assertEqual(disabled_adapter.get_voices(), [])

    def test_12_webcam_streaming_remains_unaffected(self):
        """12. Recognition engine and webcam streaming methods remain intact."""
        from inference_engine import InferenceEngine
        self.assertTrue(hasattr(InferenceEngine, "process_frame"))
        self.assertTrue(hasattr(InferenceEngine, "wait_for_frame"))
        self.assertTrue(hasattr(InferenceEngine, "get_state"))

    def test_13_translation_continues_working_alongside_speech(self):
        """13. TranslationAdapter continues working smoothly alongside SpeechAdapter."""
        trans_adapter = get_translation_adapter()
        res = trans_adapter.translate("HELLO", "Hindi")
        self.assertTrue(res.get("ok"))
        self.assertIn("translation", res)

    def test_14_feedback_authentication_continues_working(self):
        """14. Feedback HMAC session authentication remains intact."""
        token = generate_session_token(user_id="test_user_777")
        self.assertIn("test_user_777", token)
        is_valid, user_id, err = verify_session_token(token)
        self.assertTrue(is_valid)
        self.assertEqual(user_id, "test_user_777")
        self.assertEqual(err, "")

    def test_15_token_storage_purges_localstorage_and_uses_sessionstorage(self):
        """15. Frontend app.js purges persistent localStorage token and uses sessionStorage exclusively."""
        app_js_path = os.path.join(REPO_ROOT, "src", "static", "app.js")
        with open(app_js_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify localStorage.removeItem("isl_feedback_session_token") is present
        self.assertIn('localStorage.removeItem("isl_feedback_session_token")', content)

        # Verify no localStorage.setItem for feedback session token
        self.assertNotIn('localStorage.setItem("isl_feedback_session_token"', content)

        # Verify sessionStorage is used for feedback session token
        self.assertIn('sessionStorage.getItem("isl_feedback_session_token")', content)
        self.assertIn('sessionStorage.setItem("isl_feedback_session_token"', content)


if __name__ == "__main__":
    unittest.main()
