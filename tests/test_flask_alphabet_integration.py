"""
Automated Integration Tests for Phase 5G: Flask Alphabet Mode
============================================================
Tests:
1. Mode switch -> Alphabet Mode (POST /api/mode)
2. Verify active mode is 'alphabet' (GET /api/mode)
3. Alphabet state endpoint (GET /api/alphabet/state)
4. Add letter via engine and verify in state
5. Add space (POST /api/alphabet/space)
6. Backspace (POST /api/alphabet/backspace)
7. Clear word (POST /api/alphabet/clear-word)
8. Clear sentence (POST /api/alphabet/clear-sentence)
9. Commit word (POST /api/alphabet/commit)
10. Speak text (POST /api/alphabet/speak)
11. Reset debounce (POST /api/alphabet/reset)
12. Return to Word Mode (POST /api/mode)
"""

import unittest
from unittest.mock import MagicMock, patch
import json
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import app, engine


class TestFlaskAlphabetIntegration(unittest.TestCase):
    """Integration test suite for Flask Alphabet Mode APIs."""

    def setUp(self):
        self.client = app.test_client()
        # Mock TTSEngine to avoid actual audio output during test runs
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.word_buffer.clear_all()
            engine._alphabet_engine.reset_debounce_state()
            self.mock_tts = MagicMock()
            self.mock_tts.is_speaking = False
            self.mock_tts.speak.return_value = True
            engine._alphabet_engine.tts_engine = self.mock_tts
            engine._tts_engine = self.mock_tts

    def tearDown(self):
        engine._tts_engine = None
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.word_buffer.clear_all()
            engine._alphabet_engine.reset_debounce_state()
            engine._alphabet_engine.tts_engine = None
        # Always return to Word Mode after test
        engine.set_app_mode("word")

    def test_mode_switch_to_alphabet(self):
        """Test switching application mode to Alphabet Mode."""
        resp = self.client.post("/api/mode", json={"mode": "alphabet"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("app_mode"), "alphabet")

        # Verify via GET /api/mode
        get_resp = self.client.get("/api/mode")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.get_json().get("mode"), "alphabet")

    def test_alphabet_state_endpoint(self):
        """Test GET /api/alphabet/state returns valid schema."""
        self.client.post("/api/mode", json={"mode": "alphabet"})
        resp = self.client.get("/api/alphabet/state")
        self.assertEqual(resp.status_code, 200)
        state = resp.get_json()
        self.assertEqual(state.get("app_mode"), "alphabet")
        self.assertIn("raw_prediction", state)
        self.assertIn("confidence", state)
        self.assertIn("stable_letter", state)
        self.assertIn("current_word", state)
        self.assertIn("sentence", state)
        self.assertIn("hand_detected", state)

    def test_word_buffer_endpoints(self):
        """Test buffer manipulation APIs: space, backspace, clear, commit."""
        self.client.post("/api/mode", json={"mode": "alphabet"})

        # Manually populate word buffer to test API endpoints
        engine._alphabet_engine.word_buffer.current_word = "HELLO"
        engine._sync_alphabet_state_locked()

        # 1. Backspace: "HELLO" -> "HELL"
        resp = self.client.post("/api/alphabet/backspace")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("current_word"), "HELL")

        # 2. Add Space / Commit: "HELL" -> sentence "HELL ", word ""
        resp = self.client.post("/api/alphabet/space")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("current_word"), "")
        self.assertEqual(resp.get_json().get("sentence"), "HELL ")

        # 3. New word
        engine._alphabet_engine.word_buffer.current_word = "WORLD"
        engine._sync_alphabet_state_locked()

        # 4. Clear word
        resp = self.client.post("/api/alphabet/clear-word")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("current_word"), "")
        # Sentence should remain intact
        self.assertEqual(resp.get_json().get("sentence"), "HELL ")

        # 5. Clear sentence
        resp = self.client.post("/api/alphabet/clear-sentence")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("sentence"), "")

    def test_alphabet_commit_endpoint(self):
        """Test POST /api/alphabet/commit commits current word."""
        self.client.post("/api/mode", json={"mode": "alphabet"})
        engine._alphabet_engine.word_buffer.current_word = "SIGN"
        engine._sync_alphabet_state_locked()

        resp = self.client.post("/api/alphabet/commit")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("current_word"), "")
        self.assertEqual(data.get("sentence"), "SIGN ")

    def test_alphabet_speak_endpoint(self):
        """Test POST /api/alphabet/speak triggers TTS engine."""
        self.client.post("/api/mode", json={"mode": "alphabet"})
        engine._alphabet_engine.word_buffer.sentence = "HELLO WORLD "
        engine._sync_alphabet_state_locked()

        resp = self.client.post("/api/alphabet/speak")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertTrue(data.get("triggered"))
        self.mock_tts.speak.assert_called_with("HELLO WORLD", "English")

    def test_alphabet_reset_endpoint(self):
        """Test POST /api/alphabet/reset clears state machine without corrupting sentence."""
        self.client.post("/api/mode", json={"mode": "alphabet"})
        engine._alphabet_engine.word_buffer.sentence = "PRESERVED TEXT "
        engine._alphabet_engine.consecutive_count = 4
        engine._alphabet_engine.consecutive_candidate = "A"

        resp = self.client.post("/api/alphabet/reset")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("sentence"), "PRESERVED TEXT ")
        self.assertIsNone(engine._alphabet_engine.consecutive_candidate)
        self.assertEqual(engine._alphabet_engine.consecutive_count, 0)

    def test_return_to_word_mode(self):
        """Test returning from Alphabet Mode to Word Mode."""
        self.client.post("/api/mode", json={"mode": "alphabet"})
        self.assertEqual(engine.get_app_mode(), "alphabet")

        resp = self.client.post("/api/mode", json={"mode": "word"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json().get("app_mode"), "word")
        self.assertEqual(engine.get_app_mode(), "word")


if __name__ == "__main__":
    unittest.main()
