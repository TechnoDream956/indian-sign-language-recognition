#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Test Suite for the Independent Multilingual Translation & TTS Subsystem.
Tests registry, translation engine, speech engine, audio manager, and versioned REST endpoints.
"""

import io
import json
import os
import sys
import unittest

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.language_services.language_registry import (
    LANGUAGES,
    get_all_voices,
    get_supported_languages,
    resolve_language,
)
from src.language_services.translation_engine import (
    TranslationResult,
    UnifiedTranslationEngine,
)
from src.language_services.speech_engine import (
    MMSSpeechEngine,
    SpeechResult,
    UnifiedSpeechEngine,
)
from src.language_services.audio_manager import AudioManager
from src.language_services.service import get_language_service
from src.language_services.api import create_standalone_app


class TestLanguageRegistry(unittest.TestCase):
    """Test language registry resolution, script tags, and voice mappings."""

    def test_required_languages_present(self):
        required = ["hi", "bn", "mr", "ta", "te", "gu", "kn", "ml"]
        for code in required:
            self.assertIn(code, LANGUAGES)
            spec = LANGUAGES[code]
            self.assertTrue(spec.native_name)
            self.assertTrue(spec.indictrans_code)
            self.assertTrue(spec.mms_tts_code)
            self.assertTrue(len(spec.voices) > 0)

    def test_resolve_language_variations(self):
        self.assertEqual(resolve_language("Hindi").code, "hi")
        self.assertEqual(resolve_language("hi").code, "hi")
        self.assertEqual(resolve_language("HINDI").code, "hi")
        self.assertEqual(resolve_language("hin_Deva").code, "hi")
        self.assertEqual(resolve_language("हिंदी").code, "hi")

        self.assertEqual(resolve_language("Bengali").code, "bn")
        self.assertEqual(resolve_language("Marathi").code, "mr")
        self.assertEqual(resolve_language("Tamil").code, "ta")
        self.assertEqual(resolve_language("Telugu").code, "te")
        self.assertEqual(resolve_language("Gujarati").code, "gu")
        self.assertEqual(resolve_language("Kannada").code, "kn")
        self.assertEqual(resolve_language("Malayalam").code, "ml")

    def test_get_all_voices(self):
        voices = get_all_voices()
        self.assertGreater(len(voices), 10)
        hindi_voices = [v for v in voices if v["language_code"] == "hi"]
        self.assertTrue(any(v["is_default"] for v in hindi_voices))


class TestAudioManager(unittest.TestCase):
    """Test audio header parsing, data URIs, and disk caching."""

    def setUp(self):
        self.audio_mgr = AudioManager()

    def test_inspect_wav_invalid(self):
        with self.assertRaises(ValueError):
            self.audio_mgr.inspect_wav(b"too_short")
        with self.assertRaises(ValueError):
            self.audio_mgr.inspect_wav(b"X" * 50)

    def test_base64_data_uri(self):
        sample = b"RIFF....WAVEfmt "
        uri = self.audio_mgr.to_base64_data_uri(sample)
        self.assertTrue(uri.startswith("data:audio/wav;base64,"))


class TestTranslationEngine(unittest.TestCase):
    """Test genuine neural machine translation with IndicTrans2 ONNX int8."""

    @classmethod
    def setUpClass(cls):
        cls.engine = UnifiedTranslationEngine()

    def test_benchmark_sentences_hindi(self):
        test_sentences = [
            ("Hello, how are you?", "Hindi"),
            ("I need some water.", "Hindi"),
            ("My name is Arpit.", "Hindi"),
            ("Can you please help me?", "Hindi"),
            ("I am going to college tomorrow.", "Hindi"),
            ("Thank you for your support.", "Hindi"),
        ]
        for src, lang in test_sentences:
            res = self.engine.translate(src, target_lang=lang)
            self.assertTrue(res.success, f"Failed translating '{src}' to {lang}: {res.error}")
            self.assertTrue(len(res.translated_text) > 0)
            self.assertNotEqual(res.translated_text.lower(), src.lower(), "Returned raw English instead of translation!")
            self.assertIn("indictrans2", res.model_name.lower())
            print(f"\n[Translation Check] '{src}' -> ({lang}) : {res.translated_text} (Time: {res.processing_duration}s)")

    def test_multilingual_coverage(self):
        sentence = "Can you please help me?"
        targets = ["Bengali", "Marathi", "Tamil", "Telugu", "Gujarati", "Kannada", "Malayalam"]
        for tgt in targets:
            res = self.engine.translate(sentence, target_lang=tgt)
            self.assertTrue(res.success, f"Failed for {tgt}: {res.error}")
            self.assertTrue(len(res.translated_text) > 0)
            self.assertNotEqual(res.translated_text.lower(), sentence.lower())
            print(f"[Multilingual Check] '{sentence}' -> ({tgt}) : {res.translated_text} (Time: {res.processing_duration}s)")

    def test_caching_layer(self):
        sentence = "Hello, how are you?"
        # First call (computed or cached)
        res1 = self.engine.translate(sentence, target_lang="Hindi", use_cache=True)
        self.assertTrue(res1.success)

        # Second call must be instant and cached
        res2 = self.engine.translate(sentence, target_lang="Hindi", use_cache=True)
        self.assertTrue(res2.success)
        self.assertTrue(res2.cached)
        self.assertLess(res2.processing_duration, 0.05)
        self.assertEqual(res1.translated_text, res2.translated_text)


class TestSpeechEngine(unittest.TestCase):
    """Test neural text-to-speech synthesis with Meta MMS-TTS (VITS)."""

    @classmethod
    def setUpClass(cls):
        cls.speech = UnifiedSpeechEngine()

    def test_hindi_synthesis(self):
        text = "नमस्ते, आप कैसे हैं?"
        res: SpeechResult = self.speech.synthesize(text, target_lang="Hindi")
        self.assertTrue(res.success, f"Synthesis failed: {res.error}")
        self.assertGreater(len(res.audio_bytes), 1000)
        self.assertEqual(res.sample_rate, 16000)
        self.assertGreater(res.duration_seconds, 0.5)
        self.assertIn("mms-tts-hin", res.model_name)
        print(f"\n[TTS Check] Hindi '{text}' -> Audio: {len(res.audio_bytes)} bytes, Duration: {res.duration_seconds}s, Gen Time: {res.processing_duration}s")

    def test_bengali_synthesis(self):
        text = "নমস্কার, আপনি কেমন আছেন?"
        res: SpeechResult = self.speech.synthesize(text, target_lang="Bengali")
        self.assertTrue(res.success, f"Synthesis failed: {res.error}")
        self.assertGreater(len(res.audio_bytes), 1000)
        self.assertGreater(res.duration_seconds, 0.5)
        print(f"[TTS Check] Bengali '{text}' -> Audio: {len(res.audio_bytes)} bytes, Duration: {res.duration_seconds}s, Gen Time: {res.processing_duration}s")

    def test_unsupported_language_error(self):
        res = self.speech.synthesize("test", target_lang="Klingon")
        self.assertFalse(res.success)
        self.assertIn("Unsupported", res.error)


class TestAPIEndpoints(unittest.TestCase):
    """Test versioned REST API v2 endpoints."""

    @classmethod
    def setUpClass(cls):
        app = create_standalone_app()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/api/v2/language/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("peak_memory_mb", data)
        self.assertGreater(data["peak_memory_mb"], 0)

    def test_languages_endpoint(self):
        resp = self.client.get("/api/v2/language/languages")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["total_languages"], 8)

    def test_voices_endpoint(self):
        resp = self.client.get("/api/v2/language/voices?lang=Hindi")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_voices"], 0)

    def test_translate_endpoint(self):
        payload = {"text": "Thank you for your support.", "target_lang": "Hindi"}
        resp = self.client.post("/api/v2/language/translate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["translated_text"])
        self.assertEqual(data["target_lang"], "hi")

    def test_speak_and_download_endpoint(self):
        payload = {"text": "नमस्ते", "target_lang": "Hindi"}
        resp = self.client.post("/api/v2/language/speak", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["data_uri"].startswith("data:audio/wav;base64,"))
        audio_id = data["audio_id"]

        # Fetch audio via audio stream endpoint
        audio_resp = self.client.get(f"/api/v2/language/audio/{audio_id}")
        self.assertEqual(audio_resp.status_code, 200)
        self.assertEqual(audio_resp.content_type, "audio/wav")
        self.assertGreater(len(audio_resp.data), 1000)

    def test_standalone_test_ui(self):
        resp = self.client.get("/language-test")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"ISL Multilingual Services", resp.data)
        self.assertIn(b"BrowserAudioManager", resp.data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
