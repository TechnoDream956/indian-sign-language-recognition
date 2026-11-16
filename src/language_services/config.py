#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for Multilingual Translation and Neural TTS Subsystem.
Zero-dependency dataclasses and environment-driven defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "language_services"
AUDIO_DIR = DATA_DIR / "audio_cache"
TRANSLATION_CACHE_FILE = DATA_DIR / "translations_v2_cache.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TranslationConfig:
    """Settings for the neural machine translation engine."""
    # Available backends: 'indictrans2', 'nllb', 'marian'
    primary_backend: str = os.getenv("ISL_TRANSLATION_BACKEND", "indictrans2")
    indictrans_repo: str = os.getenv(
        "ISL_INDICTRANS_REPO",
        "hari31416/indictrans2-en-indic-dist-200M-ONNX-int8",
    )
    nllb_repo: str = os.getenv(
        "ISL_NLLB_REPO",
        "facebook/nllb-200-distilled-600M",
    )
    hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    device: str = os.getenv("ISL_TORCH_DEVICE", "auto")  # 'auto', 'mps', 'cpu', 'cuda'
    max_length: int = int(os.getenv("ISL_TRANSLATION_MAX_LEN", "256"))
    num_beams: int = int(os.getenv("ISL_TRANSLATION_BEAMS", "4"))
    enable_cache: bool = True
    cache_path: Path = TRANSLATION_CACHE_FILE
    lazy_load: bool = True  # Do not load weights on startup


@dataclass
class SpeechConfig:
    """Settings for the multilingual neural text-to-speech engine."""
    # Available backends: 'mms', 'indicf5', 'edge'
    primary_backend: str = os.getenv("ISL_TTS_BACKEND", "mms")
    mms_repo_template: str = os.getenv(
        "ISL_MMS_REPO_TEMPLATE",
        "facebook/mms-tts-{lang}",
    )
    indicf5_repo: str = os.getenv(
        "ISL_INDICF5_REPO",
        "6Morpheus6/IndicF5",
    )
    device: str = os.getenv("ISL_TORCH_DEVICE", "auto")
    sample_rate: int = int(os.getenv("ISL_AUDIO_SAMPLE_RATE", "16000"))
    audio_format: str = "wav"
    audio_output_dir: Path = AUDIO_DIR
    audio_retention_hours: int = 6
    lazy_load: bool = True  # Singleton loaded on first request


@dataclass
class ServiceConfig:
    """Global configuration for the independent language service."""
    host: str = os.getenv("ISL_LANG_HOST", "127.0.0.1")
    port: int = int(os.getenv("ISL_LANG_PORT", "5005"))
    debug: bool = os.getenv("ISL_LANG_DEBUG", "0") == "1"
    cors_allowed_origins: str = os.getenv("ISL_CORS_ORIGINS", "*")
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)


# Global default configuration instance
default_config = ServiceConfig()
