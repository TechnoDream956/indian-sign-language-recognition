#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5H: Comprehensive Automated Integration & Unit Tests for Multilingual Alphabet Translation.

Tests all 16 criteria specified in Phase 5H:
1. English source sentence remains unchanged.
2. English -> English returns the original sentence.
3. Known Hindi translation works.
4. Known Marathi translation works.
5. Known Tamil translation works.
6. Known Telugu translation works.
7. Known Gujarati translation works.
8. Known Kannada translation works.
9. Known Malayalam translation works.
10. Known Punjabi translation works.
11. Unknown words are preserved safely.
12. Changing language does not destroy source sentence.
13. Translation endpoint (/api/alphabet/translation) returns correct schema.
14. Speak translation calls existing TTSEngine.
15. Existing Word Mode translation still works.
16. Existing Alphabet Mode word buffer operations still work.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from translator import (
    translate_sentence,
    translate_label,
    get_supported_languages,
    LANGUAGE_DISPLAY_NAMES,
)
from app import app, engine
from alphabet_engine import WordBuffer


class TestMultilingualAlphabetTranslation(unittest.TestCase):
    """Test suite for Phase 5H Multilingual Translation and API endpoints."""

    def setUp(self):
        self.client = app.test_client()
        engine.set_app_mode("alphabet")
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.word_buffer.clear_all()

    def tearDown(self):
        engine._tts_engine = None
        if engine._alphabet_engine is not None:
            engine._alphabet_engine.word_buffer.clear_all()
        engine.set_app_mode("word")


    # ------------------------------------------------------------------ #
    # Test 1: English source sentence remains unchanged                   #
    # ------------------------------------------------------------------ #
    def test_01_english_source_sentence_remains_unchanged(self):
        source = "HELLO HOW ARE YOU"
        # Translating to Hindi
        res = translate_sentence(source, "Hindi")
        self.assertEqual(res, "नमस्ते आप कैसे हैं")
        # Ensure original input string variable was not mutated
        self.assertEqual(source, "HELLO HOW ARE YOU")

    # ------------------------------------------------------------------ #
    # Test 2: English -> English returns original sentence               #
    # ------------------------------------------------------------------ #
    def test_02_english_to_english_returns_original(self):
        source = "HELLO HOW ARE YOU"
        res = translate_sentence(source, "English")
        self.assertEqual(res, source)

    # ------------------------------------------------------------------ #
    # Tests 3-10: Known translations for 8 Indian languages              #
    # ------------------------------------------------------------------ #
    def test_03_known_hindi_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Hindi")
        self.assertEqual(res, "नमस्ते आप कैसे हैं")

    def test_04_known_marathi_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Marathi")
        self.assertEqual(res, "नमस्कार तुम्ही कसे आहात")

    def test_05_known_tamil_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Tamil")
        self.assertEqual(res, "வணக்கம் நீங்கள் எப்படி இருக்கிறீர்கள்")

    def test_06_known_telugu_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Telugu")
        self.assertEqual(res, "నమస్కారం మీరు ఎలా ఉన్నారు")

    def test_07_known_gujarati_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Gujarati")
        self.assertEqual(res, "નમસ્તે તમે કેમ છો")

    def test_08_known_kannada_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Kannada")
        self.assertEqual(res, "ನಮಸ್ಕಾರ ನೀವು ಹೇಗಿದ್ದೀರಿ")

    def test_09_known_malayalam_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Malayalam")
        self.assertEqual(res, "നമസ്കാരം നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്")

    def test_10_known_punjabi_translation(self):
        res = translate_sentence("HELLO HOW ARE YOU", "Punjabi")
        self.assertEqual(res, "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ")

    # ------------------------------------------------------------------ #
    # Test 11: Unknown words are preserved safely                        #
    # ------------------------------------------------------------------ #
    def test_11_unknown_words_preserved_safely(self):
        # "ROHIT" is an unknown proper noun, "HELLO" is known
        res = translate_sentence("HELLO ROHIT", "Hindi")
        self.assertIn("ROHIT", res)
        self.assertTrue(res.startswith("नमस्ते") or res.endswith("ROHIT"))

        # Purely unknown sentence
        unknown = "ZEBRA JUPITER MATRIX"
        res_unknown = translate_sentence(unknown, "Hindi")
        self.assertEqual(res_unknown, unknown)

    # ------------------------------------------------------------------ #
    # Test 12: Changing language does not destroy source sentence        #
    # ------------------------------------------------------------------ #
    def test_12_changing_language_preserves_source_sentence(self):
        # Populate buffer
        engine._alphabet_engine.word_buffer.current_word = "HELLO"
        engine.alphabet_commit_word()
        engine._alphabet_engine.word_buffer.current_word = "FRIEND"
        engine.alphabet_commit_word()

        initial_sentence = engine._alphabet_engine.word_buffer.get_sentence().strip()
        self.assertEqual(initial_sentence, "HELLO FRIEND")

        # Switch to Hindi
        resp1 = self.client.post("/api/language", json={"language": "Hindi"})
        self.assertEqual(resp1.status_code, 200)
        state_hi = engine.get_alphabet_state()
        self.assertEqual(state_hi["sentence"].strip(), "HELLO FRIEND")
        self.assertEqual(state_hi["target_language"], "Hindi")
        self.assertEqual(state_hi["translation"], "नमस्ते दोस्त")

        # Switch to Tamil
        resp2 = self.client.post("/api/language", json={"language": "Tamil"})
        self.assertEqual(resp2.status_code, 200)
        state_ta = engine.get_alphabet_state()
        self.assertEqual(state_ta["sentence"].strip(), "HELLO FRIEND")
        self.assertEqual(state_ta["target_language"], "Tamil")
        self.assertEqual(state_ta["translation"], "வணக்கம் நண்பரே")

        # Switch back to English
        resp3 = self.client.post("/api/language", json={"language": "English"})
        self.assertEqual(resp3.status_code, 200)
        state_en = engine.get_alphabet_state()
        self.assertEqual(state_en["sentence"].strip(), "HELLO FRIEND")
        self.assertEqual(state_en["translation"], "HELLO FRIEND")

    # ------------------------------------------------------------------ #
    # Test 13: Translation endpoint returns correct schema               #
    # ------------------------------------------------------------------ #
    def test_13_translation_endpoint_schema(self):
        engine._alphabet_engine.word_buffer.clear_all()
        engine._alphabet_engine.word_buffer.current_word = "HELLO"
        engine.alphabet_commit_word()
        engine.set_language("Hindi")

        resp = self.client.get("/api/alphabet/translation")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("source", data)
        self.assertIn("language", data)
        self.assertIn("translation", data)
        self.assertEqual(data["source"].strip(), "HELLO")
        self.assertEqual(data["language"], "Hindi")
        self.assertEqual(data["translation"], "नमस्ते")

    # ------------------------------------------------------------------ #
    # Test 14: Speak translation calls existing TTS                      #
    # ------------------------------------------------------------------ #
    def test_14_speak_translation_calls_existing_tts(self):
        engine._alphabet_engine.word_buffer.clear_all()
        engine._alphabet_engine.word_buffer.current_word = "THANK"
        engine.alphabet_commit_word()
        engine._alphabet_engine.word_buffer.current_word = "YOU"
        engine.alphabet_commit_word()
        engine.set_language("Hindi")

        # Ensure engine has a mock TTSEngine
        mock_tts = MagicMock()
        mock_tts.is_speaking = False
        mock_tts.speak.return_value = True
        engine._tts_engine = mock_tts

        resp = self.client.post("/api/alphabet/speak-translation")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertTrue(data.get("triggered"))
        mock_tts.speak.assert_called_once_with("धन्यवाद", language="Hindi")

    # ------------------------------------------------------------------ #
    # Test 15: Existing Word Mode translation still works                #
    # ------------------------------------------------------------------ #
    def test_15_existing_word_mode_translation_still_works(self):
        engine.set_app_mode("word")
        self.assertEqual(engine.get_app_mode(), "word")

        # Check translate_label across original 13 classes
        self.assertEqual(translate_label("hello", "Hindi"), "नमस्ते")
        self.assertEqual(translate_label("thanks", "Tamil"), "நன்றி")
        self.assertEqual(translate_label("friend", "Punjabi"), "ਦੋਸਤ")
        self.assertEqual(translate_label("good", "Marathi"), "छान")

        # Check 29-class expanded signs
        self.assertEqual(translate_label("hospital", "Hindi"), "अस्पताल")
        self.assertEqual(translate_label("medicine", "Telugu"), "మందు")

        # Word mode switch model check
        resp = self.client.post("/api/model", json={"model": "legacy_13"})
        self.assertEqual(resp.status_code, 200)

    # ------------------------------------------------------------------ #
    # Test 16: Existing Alphabet Mode word buffer still works            #
    # ------------------------------------------------------------------ #
    def test_16_existing_alphabet_mode_word_buffer_still_works(self):
        engine.set_app_mode("alphabet")
        wb = engine._alphabet_engine.word_buffer
        wb.clear_all()

        # Add letters
        wb.add_letter("T")
        wb.add_letter("E")
        wb.add_letter("S")
        wb.add_letter("T")
        self.assertEqual(wb.get_current_word(), "TEST")

        # Space commits word to sentence with trailing space
        wb.add_space()
        self.assertEqual(wb.get_current_word(), "")
        self.assertEqual(wb.get_sentence(), "TEST ")

        # First backspace deletes trailing space
        wb.backspace()
        self.assertEqual(wb.get_sentence(), "TEST")

        # Second backspace deletes last character 'T'
        wb.backspace()
        self.assertEqual(wb.get_sentence(), "TES")

        # Clear sentence
        wb.clear_sentence()
        self.assertEqual(wb.get_sentence(), "")



if __name__ == "__main__":
    unittest.main()
