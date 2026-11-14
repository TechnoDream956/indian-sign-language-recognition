#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Services Facade.
Combines translation, speech synthesis, and audio management
into an integrated, production-grade subsystem.
"""

import logging
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from language_services.audio_manager import AudioManager
    from language_services.config import default_config
    from language_services.language_registry import LANGUAGES, get_all_voices, get_supported_languages, resolve_language
    from language_services.speech_engine import SpeechResult, UnifiedSpeechEngine
    from language_services.translation_engine import TranslationResult, UnifiedTranslationEngine
else:
    from .audio_manager import AudioManager
    from .config import default_config
    from .language_registry import LANGUAGES, get_all_voices, get_supported_languages, resolve_language
    from .speech_engine import SpeechResult, UnifiedSpeechEngine
    from .translation_engine import TranslationResult, UnifiedTranslationEngine

logger = logging.getLogger(__name__)


class LanguageService:
    """Singleton service providing multilingual translation and neural TTS."""

    def __init__(self):
        self.config = default_config
        self.translator = UnifiedTranslationEngine()
        self.speaker = UnifiedSpeechEngine()
        self.audio_manager = AudioManager()
        self._start_time = time.time()
        logger.info("LanguageService initialized.")

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
        use_cache: bool = True,
    ) -> TranslationResult:
        """Translate text with genuine machine translation."""
        return self.translator.translate(
            text=text,
            target_lang=target_lang,
            source_lang=source_lang,
            use_cache=use_cache,
        )

    def speak(
        self,
        text: str,
        target_lang: str,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize neural speech, cache the resulting WAV, and return
        browser-ready audio metadata including data URI and audio_id.
        """
        t0 = time.time()
        result: SpeechResult = self.speaker.synthesize(
            text=text,
            target_lang=target_lang,
            voice_id=voice_id,
        )

        if not result.success or not result.audio_bytes:
            return {
                "success": False,
                "error": result.error or "Speech generation produced no audio",
                "audio_id": None,
                "data_uri": None,
                "sample_rate": result.sample_rate,
                "duration_seconds": 0.0,
                "model_name": result.model_name,
                "processing_duration": result.processing_duration,
                "speaker": result.speaker,
                "language": result.language,
            }

        audio_id = self.audio_manager.generate_audio_id(
            text=text,
            lang=result.language,
            voice_id=voice_id or result.speaker,
        )

        self.audio_manager.save_wav(
            audio_id=audio_id,
            audio_bytes=result.audio_bytes,
            sample_rate=result.sample_rate,
            metadata={
                "model_name": result.model_name,
                "speaker": result.speaker,
                "language": result.language,
            },
        )

        data_uri = self.audio_manager.to_base64_data_uri(result.audio_bytes)
        total_time = round(time.time() - t0, 3)

        return {
            "success": True,
            "error": None,
            "audio_id": audio_id,
            "data_uri": data_uri,
            "sample_rate": result.sample_rate,
            "duration_seconds": result.duration_seconds,
            "model_name": result.model_name,
            "processing_duration": total_time,
            "speaker": result.speaker,
            "language": result.language,
        }

    def get_supported_languages(self) -> List[Dict]:
        """List all supported Indian languages."""
        return get_supported_languages()

    def get_voices(self, target_lang: Optional[str] = None) -> List[Dict]:
        """List all available voices, optionally filtered by language."""
        all_voices = get_all_voices()
        if not target_lang:
            return all_voices
        spec = resolve_language(target_lang)
        if not spec:
            return []
        return [v for v in all_voices if v["language_code"] == spec.code]

    def get_health(self) -> Dict[str, Any]:
        """Telemetry check returning system health, peak memory, and model status."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes; on Linux in kilobytes
        if sys.platform == "darwin":
            peak_memory_mb = round(usage.ru_maxrss / (1024 * 1024), 2)
        else:
            peak_memory_mb = round(usage.ru_maxrss / 1024, 2)

        import torch
        device_name = "cpu"
        if torch.backends.mps.is_available():
            device_name = "mps (Apple Silicon Metal)"
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)

        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "peak_memory_mb": peak_memory_mb,
            "device": device_name,
            "translation_model": self.translator._engine.repo_id,
            "speech_backend": self.config.speech.primary_backend,
            "cached_translations": len(self.translator._memory_cache),
            "total_supported_languages": len(LANGUAGES),
        }


# Singleton service instance
_service_instance: Optional[LanguageService] = None


def get_language_service() -> LanguageService:
    global _service_instance
    if _service_instance is None:
        _service_instance = LanguageService()
    return _service_instance
