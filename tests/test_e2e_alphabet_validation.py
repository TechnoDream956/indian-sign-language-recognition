#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5I: Comprehensive End-to-End Validation & Edge-Case Test Suite.

Validates:
1. Full sentence workflow: "HELLO HOW ARE YOU" -> 10 languages
2. Word-building reliability (single letter, held single-emission, repeated LL, words)
3. Sentence controls (SPACE, BACKSPACE, COMMIT, CLEAR WORD, CLEAR SENTENCE, RESET)
4. Translation state and caching across all 10 languages
5. Non-blocking TTS (English and translated speech, overlap prevention)
6. Performance latency measurements (inference, translation, buffer)
7. Error and edge cases (empty buffers, unknown words, proper nouns, numbers, rapid switching)
8. Rapid mode switching (Word <-> Alphabet)
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

import numpy as np
from translator import (
    translate_sentence,
    translate_label,
    get_supported_languages,
    _TRANSLATION_CACHE,
)
from alphabet_engine import AlphabetEngine, WordBuffer, normalize_hand_landmarks
from app import app, engine


class TestEndToEndAlphabetValidation(unittest.TestCase):
    """Exhaustive test suite for Phase 5I End-to-End Validation."""

    def setUp(self):
        self.client = app.test_client()
        engine.set_app_mode("alphabet")
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.reset_all()
        # Reset language to English
        engine.set_language("English")

    def tearDown(self):
        engine._tts_engine = None
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.reset_all()
        engine.set_app_mode("word")


    # ------------------------------------------------------------------ #
    # 1. Full E2E Scenario: "HELLO HOW ARE YOU" -> 10 Languages         #
    # ------------------------------------------------------------------ #
    def test_01_e2e_hello_how_are_you_multilingual(self):
        wb = engine._alphabet_engine.word_buffer

        # Step-by-step composition:
        # H E L L O
        for ch in "HELLO":
            wb.add_letter(ch)
        self.assertEqual(wb.get_current_word(), "HELLO")
        engine.alphabet_add_space()
        self.assertEqual(wb.get_sentence().strip(), "HELLO")
        self.assertEqual(wb.get_current_word(), "")

        # H O W
        for ch in "HOW":
            wb.add_letter(ch)
        engine.alphabet_add_space()
        self.assertEqual(wb.get_sentence().strip(), "HELLO HOW")

        # A R E
        for ch in "ARE":
            wb.add_letter(ch)
        engine.alphabet_add_space()
        self.assertEqual(wb.get_sentence().strip(), "HELLO HOW ARE")

        # Y O U
        for ch in "YOU":
            wb.add_letter(ch)
        engine.alphabet_commit_word()

        # Final English Source Sentence
        source = wb.get_sentence().strip()
        self.assertEqual(source, "HELLO HOW ARE YOU")

        # Expected translation matrix across all 10 languages
        expected_translations = {
            "English": "HELLO HOW ARE YOU",
            "Hindi": "नमस्ते आप कैसे हैं",
            "Marathi": "नमस्कार तुम्ही कसे आहात",
            "Bengali": "নমস্কার আপনি কেমন আছেন",
            "Tamil": "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்",
            "Telugu": "నమస్కారం మీరు ఎలా ఉన్నారు",
            "Gujarati": "નમસ્તે તમે કેમ છો",
            "Kannada": "ನಮಸ್ಕಾರ ನೀವು ಹೇಗಿದ್ದೀರಿ",
            "Malayalam": "നമസ്കാരം നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്",
            "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ",
        }

        for lang, expected_trans in expected_translations.items():
            resp = self.client.post("/api/language", json={"language": lang})
            self.assertEqual(resp.status_code, 200)

            state = engine.get_alphabet_state()
            # 1. English source must remain unchanged
            self.assertEqual(state["sentence"].strip(), "HELLO HOW ARE YOU")
            # 2. Target language must be updated
            self.assertEqual(state["target_language"], lang)
            # 3. Translation must match expected native text
            self.assertEqual(state["translation"], expected_trans)
            # 4. WordBuffer current word must remain intact
            self.assertEqual(state["current_word"], "")

    # ------------------------------------------------------------------ #
    # 2. Word-Building Reliability (Hold, Release, Repetition)          #
    # ------------------------------------------------------------------ #
    def test_02_word_building_reliability(self):
        alpha_engine = engine._alphabet_engine
        wb = alpha_engine.word_buffer
        wb.clear_all()
        alpha_engine.reset_debounce_state()

        # Dummy normalized feature vector (63,)
        dummy_feat = np.zeros((63,), dtype=np.float32)

        # Mock model to output 'A' with 95% confidence
        mock_probs = np.zeros((26,), dtype=np.float32)
        mock_probs[0] = 0.95  # 'A' = 0

        with patch.object(alpha_engine, "model") as mock_model:
            mock_tensor = MagicMock()
            mock_tensor.numpy.return_value = np.expand_dims(mock_probs, axis=0)
            mock_model.return_value = mock_tensor

            t = 100.0

            # 1. Feed 4 frames: below 5-frame threshold -> no emission
            for i in range(4):
                res = alpha_engine.process_features(dummy_feat, current_time=t + i * 0.05)
                self.assertIsNone(res["emitted_letter"])
                self.assertEqual(wb.get_current_word(), "")

            # 2. 5th frame: reached stability threshold -> emitted 'A'
            res = alpha_engine.process_features(dummy_feat, current_time=t + 0.25)
            self.assertEqual(res["emitted_letter"], "A")
            self.assertEqual(wb.get_current_word(), "A")

            # 3. Hold 'A' continuously for 20 more frames: single emission guarantee!
            for i in range(6, 26):
                res = alpha_engine.process_features(dummy_feat, current_time=t + i * 0.05)
                self.assertIsNone(res["emitted_letter"], f"Held letter emitted duplicate on frame {i}")
                self.assertEqual(wb.get_current_word(), "A")

            # 4. Release hand (process None)
            res_release = alpha_engine.process_features(None, current_time=t + 2.0)
            self.assertEqual(res_release["state"], "NO_HAND")
            self.assertEqual(wb.get_current_word(), "A")

            # 5. Repeat letter 'A' after release and cooldown (> 0.8s):
            # Feed 4 frames
            for i in range(4):
                alpha_engine.process_features(dummy_feat, current_time=t + 3.0 + i * 0.05)
            # 5th frame emits second 'A' -> "AA"
            res2 = alpha_engine.process_features(dummy_feat, current_time=t + 3.25)
            self.assertEqual(res2["emitted_letter"], "A")
            self.assertEqual(wb.get_current_word(), "AA")

    # ------------------------------------------------------------------ #
    # 3. Sentence Controls Verification                                  #
    # ------------------------------------------------------------------ #
    def test_03_sentence_controls_behavior(self):
        wb = engine._alphabet_engine.word_buffer
        wb.clear_all()

        # Type "HELLO"
        for c in "HELLO":
            wb.add_letter(c)
        engine._sync_alphabet_state_locked()
        self.assertEqual(engine.get_alphabet_state()["current_word"], "HELLO")

        # COMMIT
        res_commit = engine.alphabet_commit_word()
        self.assertEqual(res_commit["current_word"], "")
        self.assertEqual(res_commit["sentence"].strip(), "HELLO")

        # Type "WORLD"
        for c in "WORLD":
            wb.add_letter(c)
        engine._sync_alphabet_state_locked()
        self.assertEqual(engine.get_alphabet_state()["current_word"], "WORLD")

        # BACKSPACE on in-progress word deletes 'D'
        res_bk1 = engine.alphabet_backspace()
        self.assertEqual(res_bk1["current_word"], "WORL")

        # CLEAR WORD clears in-progress word without affecting sentence
        res_cw = engine.alphabet_clear_word()
        self.assertEqual(res_cw["current_word"], "")
        self.assertEqual(res_cw["sentence"].strip(), "HELLO")

        # BACKSPACE on empty word deletes trailing space then characters from sentence
        res_bk2 = engine.alphabet_backspace()  # Deletes space
        self.assertEqual(res_bk2["sentence"], "HELLO")
        res_bk3 = engine.alphabet_backspace()  # Deletes 'O'
        self.assertEqual(res_bk3["sentence"], "HELL")

        # RESET: Must reset debounce state WITHOUT clearing sentence or current word!
        wb.add_letter("O")
        engine._sync_alphabet_state_locked()
        self.assertEqual(engine.get_alphabet_state()["current_word"], "O")
        res_reset = engine.alphabet_reset()
        self.assertEqual(res_reset["sentence"], "HELL")
        self.assertEqual(res_reset["current_word"], "O")
        self.assertEqual(engine._alphabet_engine.consecutive_count, 0)

        # CLEAR SENTENCE clears accumulated sentence without affecting current word
        res_cs = engine.alphabet_clear_sentence()
        self.assertEqual(res_cs["sentence"], "")
        self.assertEqual(res_cs["current_word"], "O")

    # ------------------------------------------------------------------ #
    # 4. Translation Cache and State Matrix across 10 Languages         #
    # ------------------------------------------------------------------ #
    def test_04_translation_cache_and_state_matrix(self):
        sentence = "HELLO FRIEND"
        engine._alphabet_engine.word_buffer.sentence = sentence + " "
        engine._sync_alphabet_state_locked()

        languages = [
            ("English", "HELLO FRIEND"),
            ("Hindi", "नमस्ते दोस्त"),
            ("Tamil", "வணக்கம் நண்பரே"),
            ("Marathi", "नमस्कार मित्र"),
            ("Bengali", "নমস্কার বন্ধু"),
            ("Telugu", "నమస్కారం స్నేహితుడా"),
            ("Gujarati", "નમસ્તે મિત્ર"),
            ("Kannada", "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತ"),
            ("Malayalam", "നമസ്കാരം സുഹൃത്തേ"),
            ("Punjabi", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤ"),
        ]

        # First pass: calculates and caches (non-English entries are stored in _TRANSLATION_CACHE)
        for lang, expected in languages:
            engine.set_language(lang)
            trans_data = engine.alphabet_get_translation()
            self.assertEqual(trans_data["source"].strip(), sentence)
            self.assertEqual(trans_data["translation"], expected)
            if lang != "English":
                self.assertIn((sentence, lang), _TRANSLATION_CACHE)

        # Second pass: verify O(1) cache hits
        t0 = time.perf_counter()
        for lang, expected in languages:
            cached_val = translate_sentence(sentence, lang)
            self.assertEqual(cached_val, expected)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        # 10 cache hits should execute in under 1000 microseconds (1 ms)
        self.assertLess(elapsed_us, 1000.0, f"Cache lookup took {elapsed_us:.1f}us, expected <1000us")

    # ------------------------------------------------------------------ #
    # 5. Non-Blocking TTS Validation                                     #
    # ------------------------------------------------------------------ #
    def test_05_tts_non_blocking_and_overlap_prevention(self):
        engine._alphabet_engine.word_buffer.sentence = "HELLO "
        engine.set_language("Hindi")

        mock_tts = MagicMock()
        mock_tts.is_speaking = False
        mock_tts.speak.return_value = True
        engine._tts_engine = mock_tts

        # 1. First speak call triggers TTS asynchronously
        resp1 = self.client.post("/api/alphabet/speak-translation")
        self.assertEqual(resp1.status_code, 200)
        self.assertTrue(resp1.get_json()["triggered"])
        mock_tts.speak.assert_called_with("नमस्ते", language="Hindi")

        # 2. Simulate active speech in progress
        mock_tts.is_speaking = True
        resp2 = self.client.post("/api/alphabet/speak-translation")
        self.assertEqual(resp2.status_code, 200)
        # Must prevent overlapping speech
        self.assertFalse(resp2.get_json()["triggered"])

        # 3. Test English speak endpoint
        mock_tts.reset_mock()
        mock_tts.is_speaking = False
        resp3 = self.client.post("/api/alphabet/speak")
        self.assertEqual(resp3.status_code, 200)
        self.assertTrue(resp3.get_json()["triggered"])
        mock_tts.speak.assert_called_with("HELLO", "English")

    # ------------------------------------------------------------------ #
    # 6. Performance Latency Measurements                                #
    # ------------------------------------------------------------------ #
    def test_06_performance_benchmarks(self):
        # A. MLP Forward-pass latency
        alpha_model = engine._alphabet_engine.model
        dummy_input = np.random.randn(1, 63).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = alpha_model(dummy_input, training=False)

        latencies_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = alpha_model(dummy_input, training=False)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_mlp_lat = np.mean(latencies_ms)
        # MLP forward pass on Apple Silicon / CPU should be < 5.0 ms
        self.assertLess(mean_mlp_lat, 5.0, f"Alphabet MLP latency {mean_mlp_lat:.2f}ms exceeds 5.0ms threshold")

        # B. Translation Latency (Uncached vs Cached)
        uncached_t0 = time.perf_counter()
        _ = translate_sentence("TAKE CARE MY FRIEND", "Hindi")
        uncached_lat_ms = (time.perf_counter() - uncached_t0) * 1000.0

        cached_t0 = time.perf_counter()
        _ = translate_sentence("TAKE CARE MY FRIEND", "Hindi")
        cached_lat_ms = (time.perf_counter() - cached_t0) * 1000.0

        self.assertLess(uncached_lat_ms, 2.0, "Uncached sentence translation took > 2.0ms")
        self.assertLess(cached_lat_ms, 0.1, "Cached sentence translation took > 0.1ms")

    # ------------------------------------------------------------------ #
    # 7. Error and Edge Cases                                            #
    # ------------------------------------------------------------------ #
    def test_07_error_and_edge_cases(self):
        wb = engine._alphabet_engine.word_buffer
        wb.clear_all()

        # A. Empty buffer operations: should return cleanly without exception
        res_space = engine.alphabet_add_space()
        self.assertEqual(res_space["sentence"], "")

        res_bk = engine.alphabet_backspace()
        self.assertEqual(res_bk["sentence"], "")

        res_cw = engine.alphabet_clear_word()
        self.assertEqual(res_cw["current_word"], "")

        res_cs = engine.alphabet_clear_sentence()
        self.assertEqual(res_cs["sentence"], "")

        # B. Speak on empty buffer
        resp_spk = self.client.post("/api/alphabet/speak")
        self.assertEqual(resp_spk.status_code, 200)
        self.assertFalse(resp_spk.get_json()["triggered"])

        resp_spk_trans = self.client.post("/api/alphabet/speak-translation")
        self.assertEqual(resp_spk_trans.status_code, 200)
        self.assertFalse(resp_spk_trans.get_json()["triggered"])

        # C. Switch language with empty sentence
        resp_lang = self.client.post("/api/language", json={"language": "Tamil"})
        self.assertEqual(resp_lang.status_code, 200)
        self.assertEqual(engine.get_alphabet_state()["translation"], "")

        # D. Proper noun preservation: "HELLO ROHIT"
        trans_proper = translate_sentence("HELLO ROHIT", "Hindi")
        self.assertIn("ROHIT", trans_proper)
        self.assertIn("नमस्ते", trans_proper)

        # E. Numbers preservation: "HELLO 123"
        trans_nums = translate_sentence("HELLO 123", "Hindi")
        self.assertIn("123", trans_nums)
        self.assertIn("नमस्ते", trans_nums)

        # F. Rapid language switching (10 iterations)
        langs = get_supported_languages()
        for l in langs:
            res = self.client.post("/api/language", json={"language": l})
            self.assertEqual(res.status_code, 200)
        self.assertEqual(engine.get_alphabet_state()["target_language"], langs[-1])

    # ------------------------------------------------------------------ #
    # 8. Rapid Mode Switching: Word <-> Alphabet                        #
    # ------------------------------------------------------------------ #
    def test_08_rapid_mode_switching(self):
        wb = engine._alphabet_engine.word_buffer
        wb.sentence = "PRESERVED SENTENCE "
        engine._sync_alphabet_state_locked()

        # Switch modes rapidly 10 times
        for _ in range(5):
            # Word Mode
            r1 = self.client.post("/api/mode", json={"mode": "word"})
            self.assertEqual(r1.status_code, 200)
            self.assertEqual(engine.get_app_mode(), "word")

            # Alphabet Mode
            r2 = self.client.post("/api/mode", json={"mode": "alphabet"})
            self.assertEqual(r2.status_code, 200)
            self.assertEqual(engine.get_app_mode(), "alphabet")

        # Verify WordBuffer sentence was NOT corrupted or lost across switches
        final_state = engine.get_alphabet_state()
        self.assertEqual(final_state["sentence"].strip(), "PRESERVED SENTENCE")


if __name__ == "__main__":
    unittest.main()
