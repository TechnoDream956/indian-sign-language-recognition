#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-blocking Cross-Platform Text-to-Speech (TTS) Engine.
========================================================
Provides thread-safe asynchronous speech synthesis across all 10 supported languages:
English, Hindi, Marathi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Punjabi.

Harden Fallback Chain:
1. Native macOS `say` (when on macOS and native voice is installed)
2. Offline Cross-Platform `pyttsx3` (SAPI5 on Windows, eSpeak on Linux, NSSpeechSynthesizer on macOS)
3. Google Text-to-Speech (`gTTS`) + local audio player (afplay, paplay, aplay, ffplay, mpg123, powershell)
4. Browser Web Speech API fallback (for headless environments, missing audio drivers, or silent failures)

Author: Antigravity Team
Date: 2026-11-16
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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

# ISO language codes for pyttsx3 voice matching
PYTTSX3_LANG_HINTS: Dict[str, List[str]] = {
    "English": ["en", "eng"],
    "Hindi": ["hi", "hin"],
    "Marathi": ["mr", "mar"],
    "Bengali": ["bn", "ben"],
    "Tamil": ["ta", "tam"],
    "Telugu": ["te", "tel"],
    "Gujarati": ["gu", "guj"],
    "Kannada": ["kn", "kan"],
    "Malayalam": ["ml", "mal"],
    "Punjabi": ["pa", "pan"],
}


def is_headless_environment() -> bool:
    """
    Check if running in a headless container, CI runner, or cloud environment
    without physical audio playback hardware.
    """
    if os.environ.get("RENDER") or os.environ.get("IS_CLOUD") in ("1", "true", "True"):
        return True
    if os.environ.get("HEADLESS") in ("1", "true", "True") or os.environ.get("ISL_HEADLESS") in ("1", "true", "True"):
        return True
    if sys.platform.startswith("linux"):
        # If Linux and no ALSA/Pulse audio devices exist
        if not os.path.exists("/dev/snd") and not os.environ.get("PULSE_SERVER"):
            return True
    return False


def play_audio_file_cross_platform(file_path: str) -> Optional[subprocess.Popen]:
    """
    Spawn an audio playback process appropriate for the host OS.
    Returns Popen process if started, None otherwise.
    """
    if not os.path.exists(file_path):
        return None

    # 1. macOS: afplay is built into macOS
    if sys.platform.startswith("darwin") and shutil.which("afplay"):
        try:
            return subprocess.Popen(
                ["afplay", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning("[TTS] afplay failed: %s", e)

    # 2. Windows: PowerShell Media.SoundPlayer or start
    elif sys.platform.startswith("win"):
        if shutil.which("powershell"):
            try:
                cmd = [
                    "powershell",
                    "-c",
                    f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()',
                ]
                return subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.warning("[TTS] powershell audio playback failed: %s", e)

    # 3. Linux: probe PulseAudio / ALSA players
    elif sys.platform.startswith("linux"):
        for player in ["paplay", "aplay", "ffplay", "mpg123"]:
            if shutil.which(player):
                try:
                    cmd = (
                        [player, "-nodisp", "-autoexit", file_path]
                        if player == "ffplay"
                        else [player, file_path]
                    )
                    return subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as e:
                    logger.debug("[TTS] %s player failed: %s", player, e)

    # 4. Fallback: ffplay if installed anywhere
    if shutil.which("ffplay"):
        try:
            return subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    return None


class TTSEngine:
    """
    Thread-safe, non-blocking hybrid TTS engine with resilient multi-tier fallback:
    macOS `say` -> `pyttsx3` -> `gTTS` -> Browser Web Speech API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_process: Optional[subprocess.Popen] = None
        self._is_speaking = False
        self._is_headless = is_headless_environment()

        # Tier 1: macOS say voices
        self.say_available = sys.platform.startswith("darwin") and bool(shutil.which("say"))
        self.available_voices = self._discover_say_voices() if self.say_available else {}
        self.voice_map = self._build_say_voice_map()

        # Tier 2: pyttsx3 availability check
        self._pyttsx3_engine = None
        self._pyttsx3_available = False
        if not self._is_headless:
            self._init_pyttsx3()

    def _init_pyttsx3(self) -> None:
        """Initialize pyttsx3 engine safely without crashing on missing drivers."""
        try:
            import pyttsx3
            # Test initialization
            eng = pyttsx3.init()
            # Verify basic property query works
            _ = eng.getProperty("voices")
            self._pyttsx3_engine = eng
            self._pyttsx3_available = True
            logger.info("[TTS] pyttsx3 initialized successfully as Tier 2 fallback.")
        except Exception as e:
            logger.debug("[TTS] pyttsx3 unavailable or failed initialization: %s", e)
            self._pyttsx3_engine = None
            self._pyttsx3_available = False

    def _discover_say_voices(self) -> Dict[str, str]:
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
            logger.warning("[TTS Warning] Could not query macOS system voices: %s", e)
        return discovered

    def _build_say_voice_map(self) -> Dict[str, List[str]]:
        """Map supported languages to verified macOS native voice candidates."""
        return {
            "English": ["Rishi", "Aman", "Samantha", "Alex", "Karen", "Daniel", "Victoria"],
            "Hindi": ["Lekha"],
            "Marathi": ["Lekha"],
            "Bengali": ["Piya"],
            "Tamil": ["Vani"],
            "Telugu": ["Geeta"],
            "Kannada": ["Soumya"],
            "Gujarati": [],
            "Malayalam": [],
            "Punjabi": [],
        }

    def get_best_voice(self, language: str) -> Optional[str]:
        """Find the verified native macOS voice for a language, or None if unavailable."""
        if not self.say_available:
            return None
        candidates = self.voice_map.get(language, [])
        for candidate in candidates:
            cand_lower = candidate.lower()
            if cand_lower in self.available_voices:
                return self.available_voices[cand_lower]
            for voice_key, voice_name in self.available_voices.items():
                if cand_lower in voice_key:
                    return voice_name
        return None

    def _find_pyttsx3_voice(self, language: str) -> Optional[str]:
        """
        Find a compatible pyttsx3 voice ID for the given language.
        On macOS, languages with no authentic named native voice (Gujarati, Punjabi, Malayalam)
        are explicitly bypassed so execution falls through to Tier 3 (gTTS) rather than producing
        incomprehensible English-accented phonetic artifacts.
        """
        if sys.platform.startswith("darwin") and language in ["Gujarati", "Punjabi", "Malayalam"]:
            return None

        if not self._pyttsx3_available or self._pyttsx3_engine is None:
            return None

        hints = PYTTSX3_LANG_HINTS.get(language, ["en"])
        try:
            voices = self._pyttsx3_engine.getProperty("voices")
            for v in voices:
                v_id = str(getattr(v, "id", "")).lower()
                v_name = str(getattr(v, "name", "")).lower()
                for hint in hints:
                    # Require token boundary so 'pa' does not match 'compact' and 'gu' does not match 'portuguese'
                    pattern = rf"(?:^|[._\-\s]){hint}(?:[._\-\s]|$)"
                    if re.search(pattern, v_id) or re.search(pattern, v_name):
                        return getattr(v, "id")
        except Exception:
            pass
        return None

    def is_voice_available(self, language: str) -> bool:
        """Check whether a voice exists for the given language across any tier."""
        if self.get_best_voice(language) is not None:
            return True
        if self._find_pyttsx3_voice(language) is not None:
            return True
        if language in GTTS_LANG_CODES:
            return True
        return False

    def get_voice_catalog(self) -> Dict[str, dict]:
        """Return availability status and active engine tier for all 10 supported languages."""
        catalog = {}
        for lang in [
            "English", "Hindi", "Marathi", "Bengali", "Tamil",
            "Telugu", "Gujarati", "Kannada", "Malayalam", "Punjabi",
        ]:
            say_voice = self.get_best_voice(lang)
            if say_voice:
                tier = "Tier 1: macOS say"
                voice_label = say_voice
            elif self._find_pyttsx3_voice(lang):
                tier = "Tier 2: pyttsx3"
                voice_label = "pyttsx3"
            elif lang in GTTS_LANG_CODES:
                tier = "Tier 3: gTTS"
                voice_label = f"gTTS_{GTTS_LANG_CODES[lang]}"
            else:
                tier = "Tier 4: Browser Web Speech"
                voice_label = "WebSpeech"

            catalog[lang] = {
                "available": True,
                "tier": tier,
                "voice": voice_label,
                "note": f"{tier} ({voice_label})",
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
        Speak text asynchronously in a background thread via the hardened fallback chain:
        macOS `say` -> `pyttsx3` -> `gTTS` -> Browser Web Speech fallback.

        Returns:
            True if local speech playback was successfully started.
            False if busy, empty text, or if audio hardware is unavailable (prompting browser fallback).
        """
        res = self.speak_with_result(text, language)
        return bool(res.get("success") and not res.get("fallback"))

    def speak_with_result(self, text: str, language: str = "English") -> Dict[str, Any]:
        """
        Execute speech synthesis and return structured diagnostic details including active tier.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "fallback": "browser",
                "engine": "none",
                "reason": "empty_text",
            }

        # In headless / containerized environments without audio hardware, immediately signal browser fallback
        if self._is_headless:
            logger.info("[TTS] Headless environment detected. Delegating to browser Web Speech API.")
            return {
                "success": False,
                "fallback": "browser",
                "engine": "browser",
                "reason": "headless_environment",
            }

        with self._lock:
            if self._check_speaking_unlocked():
                return {
                    "success": False,
                    "fallback": None,
                    "engine": "busy",
                    "reason": "speech_in_progress",
                }

            self._is_speaking = True

        def _execute_speech_chain():
            temp_audio_file = None
            try:
                # -------------------------------------------------------------
                # Tier 1: macOS say
                # -------------------------------------------------------------
                say_voice = self.get_best_voice(language)
                if say_voice and self.say_available:
                    try:
                        cmd = ["say", "-v", say_voice, "-r", "175", text]
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        with self._lock:
                            self._current_process = proc
                        proc.wait()
                        if proc.returncode == 0:
                            return
                    except Exception as e:
                        logger.warning("[TTS Tier 1 Failed] macOS say error: %s", e)

                # -------------------------------------------------------------
                # Tier 2: pyttsx3 (Only attempted if a valid matching voice was found)
                # -------------------------------------------------------------
                if self._pyttsx3_available and self._pyttsx3_engine is not None:
                    v_id = self._find_pyttsx3_voice(language)
                    if v_id is not None:
                        try:
                            self._pyttsx3_engine.setProperty("voice", v_id)
                            self._pyttsx3_engine.setProperty("rate", 170)
                            self._pyttsx3_engine.say(text)
                            self._pyttsx3_engine.runAndWait()
                            return
                        except Exception as e:
                            logger.warning("[TTS Tier 2 Failed] pyttsx3 execution error: %s", e)

                # -------------------------------------------------------------
                # Tier 3: gTTS + Local Audio Player
                # -------------------------------------------------------------
                gtts_code = GTTS_LANG_CODES.get(language, "hi")
                try:
                    from gtts import gTTS
                    tts = gTTS(text=text, lang=gtts_code)
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        temp_audio_file = f.name
                        tts.save(f.name)

                    proc = play_audio_file_cross_platform(temp_audio_file)
                    if proc is not None:
                        with self._lock:
                            self._current_process = proc
                        proc.wait()
                        if proc.returncode == 0:
                            return
                    else:
                        logger.warning("[TTS Tier 3 Warning] No local audio player executable found.")
                except Exception as e:
                    logger.warning("[TTS Tier 3 Failed] gTTS playback error: %s", e)

                # -------------------------------------------------------------
                # Tier 4: Fallback to Browser Web Speech API
                # -------------------------------------------------------------
                logger.info("[TTS Tier 4] Local playback exhausted. Triggering browser fallback.")

            finally:
                if temp_audio_file and os.path.exists(temp_audio_file):
                    try:
                        os.remove(temp_audio_file)
                    except Exception:
                        pass
                with self._lock:
                    self._current_process = None
                    self._is_speaking = False

        thread = threading.Thread(target=_execute_speech_chain, daemon=True)
        thread.start()

        return {
            "success": True,
            "fallback": None,
            "engine": "local_hardware",
            "language": language,
        }

    def stop(self):
        """Immediately terminate any running speech process."""
        with self._lock:
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass
                self._current_process = None
            if self._pyttsx3_available and self._pyttsx3_engine is not None:
                try:
                    self._pyttsx3_engine.stop()
                except Exception:
                    pass
            self._is_speaking = False


if __name__ == "__main__":
    tts = TTSEngine()
    print("Cross-Platform Voice Catalog:")
    catalog = tts.get_voice_catalog()
    for lang, info in catalog.items():
        print(f"  {lang:<12}: {info['tier']} -> {info['voice']}")

    print("\nTesting speech synthesis fallback chain:")
    for lang, sample in [
        ("English", "Hello, Indian Sign Language Recognition is ready."),
        ("Hindi", "नमस्ते"),
        ("Marathi", "चेंडूचा खेळ"),
        ("Gujarati", "નમસ્તે"),
    ]:
        print(f"Speaking {lang}: '{sample}'...")
        tts.speak(sample, lang)
        import time
        time.sleep(1.2)
