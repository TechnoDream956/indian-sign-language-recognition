#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Language Services Subsystem for Indian Sign Language Recognition.
Provides independent, zero-paid-API multilingual neural translation and speech synthesis.
"""

from .config import default_config, ServiceConfig
from .language_registry import (
    LANGUAGES,
    LanguageSpec,
    VoiceInfo,
    resolve_language,
    get_supported_languages,
    get_all_voices,
)
from .translation_engine import (
    TranslationResult,
    BaseTranslationEngine,
    IndicTrans2ONNXEngine,
    UnifiedTranslationEngine,
)
from .speech_engine import (
    SpeechResult,
    BaseSpeechEngine,
    MMSSpeechEngine,
    IndicF5SpeechEngine,
    UnifiedSpeechEngine,
)
from .audio_manager import AudioManager
from .service import LanguageService, get_language_service

__all__ = [
    "default_config",
    "ServiceConfig",
    "LANGUAGES",
    "LanguageSpec",
    "VoiceInfo",
    "resolve_language",
    "get_supported_languages",
    "get_all_voices",
    "TranslationResult",
    "BaseTranslationEngine",
    "IndicTrans2ONNXEngine",
    "UnifiedTranslationEngine",
    "SpeechResult",
    "BaseSpeechEngine",
    "MMSSpeechEngine",
    "IndicF5SpeechEngine",
    "UnifiedSpeechEngine",
    "AudioManager",
    "LanguageService",
    "get_language_service",
]
