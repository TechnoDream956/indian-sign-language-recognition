#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-blocking Text-to-Speech (TTS) Engine for macOS.
===================================================
Provides thread-safe asynchronous speech synthesis across all 10 supported languages:
- Native macOS `say` voice synthesis for installed voices (English, Hindi, Marathi, Bengali, Tamil, Telugu, Kannada).
- High-fidelity gTTS + `afplay` synthesis for languages without macOS native voices (Gujarati, Malayalam, Punjabi)
  or as an automatic fallback when an offline voice is missing.
"""

import os
import re
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional

# Supported language code mapping for gTTS
GTTS_LANG_CODES: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
}


class TTSEngine:
    """Thread-safe, non-blocking hybrid TTS engine wrapping macOS `say` and `afplay`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_process: Optional[subprocess.Popen] = None
        self._is_speaking = False
        self.available_voices = self._discover_voices()
        self.voice_map = self._build_voice_map()

    def _discover_voices(self) -> Dict[str, str]:
        """Parse macOS `say -v '?'` output to get installed voice names."""
        discovered = {}
        try:
            output = subprocess.check_output(["say", "-v", "?"], text=True)
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                match = re.match(r"^([^\t#]+?)\s+([a-z]{2}_[A-Z]{2}|\w+)", line.strip())
                if match:
                    raw_name = match.group(1).strip()
                    clean_name = raw_name.split("(")[0].strip()
                    discovered[clean_name.lower()] = clean_name
                    discovered[raw_name.lower()] = raw_name
        except Exception as e:
            print(f"[TTS Warning] Could not query system voices: {e}")
        return discovered

    def _build_voice_map(self) -> Dict[str, List[str]]:
        """Map supported languages to verified macOS native voice candidates."""
        return {
            "English": ["Rishi", "Aman", "Samantha", "Alex", "Karen", "Daniel", "Victoria"],
            "Hindi": ["Lekha"],
            "Marathi": ["Lekha"],  # Uses Devanagari script; verified fluent on macOS
            "Bengali": ["Piya"],
            "Tamil": ["Vani"],
            "Telugu": ["Geeta"],
            "Kannada": ["Soumya"],
            "Gujarati": [],   # Powered via gTTS
            "Malayalam": [],  # Powered via gTTS
            "Punjabi": [],    # Powered via gTTS
        }

    def get_best_voice(self, language: str) -> Optional[str]:
        """Find the verified native voice for a language, or None if unavailable."""
        candidates = self.voice_map.get(language, [])
        for candidate in candidates:
            cand_lower = candidate.lower()
            if cand_lower in self.available_voices:
                return self.available_voices[cand_lower]
            for voice_key, voice_name in self.available_voices.items():
                if cand_lower in voice_key:
                    return voice_name
        return None

    def is_voice_available(self, language: str) -> bool:
        """Check whether a voice (native macOS or gTTS) exists for the given language."""
        if language in GTTS_LANG_CODES:
            return True
        return self.get_best_voice(language) is not None

    def get_voice_catalog(self) -> Dict[str, dict]:
        """Return availability status and voice name for all 10 supported languages."""
        catalog = {}
        for lang in [
            "English", "Hindi", "Marathi", "Bengali", "Tamil",
            "Telugu", "Gujarati", "Kannada", "Malayalam", "Punjabi"
        ]:
            voice = self.get_best_voice(lang)
            if voice:
                note = f"Native macOS voice: {voice}" if lang != "Marathi" else f"Devanagari voice: {voice}"
            elif lang in GTTS_LANG_CODES:
                note = f"Multilingual Audio Engine (gTTS: {lang})"
            else:
                note = "Audio synthesis unavailable"
            catalog[lang] = {
                "available": True,
                "voice": voice or f"gTTS_{GTTS_LANG_CODES.get(lang, 'en')}",
                "note": note,
            }
        return catalog

    def _check_speaking_unlocked(self) -> bool:
        """Inspect speech status assuming caller already holds self._lock."""
        if self._current_process is not None:
            if self._current_process.poll() is None:
                return True
            else:
                self._current_process = None
                self._is_speaking = False
        return self._is_speaking

    @property
    def is_speaking(self) -> bool:
        """Check if speech is currently playing."""
        with self._lock:
            return self._check_speaking_unlocked()

    def speak(self, text: str, language: str = "English") -> bool:
        """
        Speak text asynchronously in a background thread.
        - Uses native macOS `say` when a native voice is installed.
        - Uses `gTTS` + `afplay` for Gujarati, Malayalam, Punjabi, or fallback.
        - Never silently speaks non-English text using an English voice.

        Returns:
            True if speech process started, False if busy or voice unavailable.
        """
        if not text or not text.strip():
            return False

        with self._lock:
            if self._check_speaking_unlocked():
                return False

            voice_name = self.get_best_voice(language)
            self._is_speaking = True

            def _run_speech():
                temp_audio_file = None
                try:
                    if voice_name:
                        # Native macOS voice with natural speaking rate (175 wpm)
                        cmd = ["say", "-v", voice_name, "-r", "175", text]
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        with self._lock:
                            self._current_process = proc
                        proc.wait()
                    else:
                        # Fallback or languages without macOS voice: gTTS + afplay
                        gtts_code = GTTS_LANG_CODES.get(language, "hi")
                        from gtts import gTTS
                        tts = gTTS(text=text, lang=gtts_code)
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                            temp_audio_file = f.name
                            tts.save(f.name)

                        # afplay is built-in on all macOS systems
                        proc = subprocess.Popen(
                            ["afplay", temp_audio_file],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        with self._lock:
                            self._current_process = proc
                        proc.wait()
                except Exception as e:
                    print(f"[TTS Error] Speech playback failed for {language}: {e}")
                finally:
                    if temp_audio_file and os.path.exists(temp_audio_file):
                        try:
                            os.remove(temp_audio_file)
                        except Exception:
                            pass
                    with self._lock:
                        self._current_process = None
                        self._is_speaking = False

            thread = threading.Thread(target=_run_speech, daemon=True)
            thread.start()
            return True

    def stop(self):
        """Immediately terminate any running speech process."""
        with self._lock:
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass
                self._current_process = None
            self._is_speaking = False


if __name__ == "__main__":
    import time
    tts = TTSEngine()
    print("Voice Catalog:")
    catalog = tts.get_voice_catalog()
    for lang, info in catalog.items():
        print(f"  {lang:<12}: {info['note']} (Voice: {info['voice']})")

    print("\nTesting speech synthesis across samples:")
    for lang, sample in [("Hindi", "नमस्ते"), ("Marathi", "चेंडूचा खेळ"), ("Gujarati", "નમસ્તે"), ("Punjabi", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ")]:
        print(f"Speaking {lang}: '{sample}'...")
        tts.speak(sample, lang)
        time.sleep(1.5)
