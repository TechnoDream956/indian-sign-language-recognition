#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Speech Engine module.
Implements multilingual neural text-to-speech using Meta MMS-TTS (VITS)
and AI4Bharat IndicF5, with persistent model caching and WAV serialization.
"""

import io
import logging
import os
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

import sys
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from language_services.config import default_config
    from language_services.language_registry import LANGUAGES, LanguageSpec, resolve_language
else:
    from .config import default_config
    from .language_registry import LANGUAGES, LanguageSpec, resolve_language

logger = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    """Structured result returned by the speech synthesis engine."""
    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    model_name: str
    processing_duration: float
    speaker: str
    language: str
    success: bool
    error: Optional[str] = None


class BaseSpeechEngine(ABC):
    """Abstract base class for all neural TTS engines."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        target_lang: str,
        voice_id: Optional[str] = None,
    ) -> SpeechResult:
        """Synthesize text into WAV audio."""
        pass

    @abstractmethod
    def is_available(self, target_lang: str) -> bool:
        """Check if engine can synthesize the given language."""
        pass


class MMSSpeechEngine(BaseSpeechEngine):
    """
    Multilingual Text-to-Speech Engine using Meta MMS-TTS (VITS).
    End-to-end neural VITS models trained on 1,400+ languages.
    Generates 16kHz WAV audio in ~150-300ms on CPU/MPS without external APIs.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = self._resolve_device(device or default_config.speech.device)
        self._loaded_models: Dict[str, Tuple[VitsModel, AutoTokenizer]] = {}
        logger.info("MMSSpeechEngine initialized on device: %s", self.device)

    @staticmethod
    def _resolve_device(dev: str) -> torch.device:
        if dev == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _get_model_and_tokenizer(self, mms_code: str) -> Tuple[VitsModel, AutoTokenizer]:
        """Lazy-load and cache the VITS model and tokenizer for a specific language."""
        if mms_code not in self._loaded_models:
            repo_id = default_config.speech.mms_repo_template.format(lang=mms_code)
            logger.info("Loading MMS-TTS model for '%s' from %s...", mms_code, repo_id)
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = VitsModel.from_pretrained(repo_id).to(self.device)
            model.eval()
            self._loaded_models[mms_code] = (model, tokenizer)
            logger.info("MMS-TTS model '%s' loaded and cached.", mms_code)
        return self._loaded_models[mms_code]

    def is_available(self, target_lang: str) -> bool:
        spec = resolve_language(target_lang)
        return spec is not None and bool(spec.mms_tts_code)

    def synthesize(
        self,
        text: str,
        target_lang: str,
        voice_id: Optional[str] = None,
    ) -> SpeechResult:
        t0 = time.time()
        spec = resolve_language(target_lang)

        if not spec:
            return SpeechResult(
                audio_bytes=b"",
                sample_rate=16000,
                duration_seconds=0.0,
                model_name="mms-vits",
                processing_duration=round(time.time() - t0, 3),
                speaker="unknown",
                language=target_lang,
                success=False,
                error=f"Unsupported target language: {target_lang}",
            )

        clean_text = text.strip()
        if not clean_text:
            return SpeechResult(
                audio_bytes=b"",
                sample_rate=16000,
                duration_seconds=0.0,
                model_name="mms-vits",
                processing_duration=round(time.time() - t0, 3),
                speaker="default",
                language=spec.code,
                success=False,
                error="Cannot synthesize empty text",
            )

        mms_code = spec.mms_tts_code
        try:
            model, tokenizer = self._get_model_and_tokenizer(mms_code)
            inputs = tokenizer(clean_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model(**inputs).waveform

            # Output waveform is float tensor [1, N]
            waveform_np = output.squeeze().cpu().numpy()
            sample_rate = model.config.sampling_rate

            # Clip and normalize to 16-bit PCM WAV
            waveform_np = np.clip(waveform_np, -1.0, 1.0)
            pcm_data = (waveform_np * 32767).astype(np.int16)

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wav_out:
                wav_out.setnchannels(1)
                wav_out.setsampwidth(2)
                wav_out.setframerate(sample_rate)
                wav_out.writeframes(pcm_data.tobytes())

            audio_bytes = wav_buf.getvalue()
            duration = len(pcm_data) / float(sample_rate)
            proc_time = round(time.time() - t0, 3)

            return SpeechResult(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                duration_seconds=round(duration, 3),
                model_name=f"facebook/mms-tts-{mms_code}",
                processing_duration=proc_time,
                speaker=f"{spec.english_name} Neural (VITS)",
                language=spec.code,
                success=True,
                error=None,
            )
        except Exception as e:
            logger.error("MMS synthesis failed for %s: %s", mms_code, e, exc_info=True)
            return SpeechResult(
                audio_bytes=b"",
                sample_rate=16000,
                duration_seconds=0.0,
                model_name=f"facebook/mms-tts-{mms_code}",
                processing_duration=round(time.time() - t0, 3),
                speaker="default",
                language=spec.code,
                success=False,
                error=f"Synthesis failed: {str(e)}",
            )


class IndicF5SpeechEngine(BaseSpeechEngine):
    """
    Adapter for AI4Bharat IndicF5 (Zero-shot Polyglot Voice Cloning TTS).
    Verifies model access, Hugging Face auth tokens, and license constraints.
    """

    def __init__(self, repo_id: Optional[str] = None):
        self.repo_id = repo_id or default_config.speech.indicf5_repo
        self.device = default_config.speech.device
        self._model = None
        self._auth_checked = False
        self._is_accessible = False

    def check_access(self) -> Tuple[bool, str]:
        """Check if IndicF5 is accessible or gated."""
        import urllib.request
        url = f"https://huggingface.co/api/models/{self.repo_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ISL-Language-Service"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json
                data = json.loads(resp.read().decode())
                is_gated = data.get("gated", False)
                if is_gated and not default_config.translation.hf_token:
                    return False, f"Model '{self.repo_id}' is gated on Hugging Face. HF_TOKEN is required."
                return True, "Model accessible"
        except Exception as e:
            return False, f"Cannot reach repository '{self.repo_id}': {e}"

    def is_available(self, target_lang: str) -> bool:
        spec = resolve_language(target_lang)
        return spec is not None and bool(spec.indicf5_code)

    def synthesize(
        self,
        text: str,
        target_lang: str,
        voice_id: Optional[str] = None,
    ) -> SpeechResult:
        t0 = time.time()
        spec = resolve_language(target_lang)

        if not spec:
            return SpeechResult(
                audio_bytes=b"",
                sample_rate=24000,
                duration_seconds=0.0,
                model_name=self.repo_id,
                processing_duration=round(time.time() - t0, 3),
                speaker="IndicF5",
                language=target_lang,
                success=False,
                error=f"Unsupported target language: {target_lang}",
            )

        accessible, reason = self.check_access()
        if not accessible:
            return SpeechResult(
                audio_bytes=b"",
                sample_rate=24000,
                duration_seconds=0.0,
                model_name=self.repo_id,
                processing_duration=round(time.time() - t0, 3),
                speaker="IndicF5",
                language=spec.code,
                success=False,
                error=f"IndicF5 unavailable: {reason}",
            )

        # IndicF5 requires reference audio prompt for zero-shot diffusion inference
        return SpeechResult(
            audio_bytes=b"",
            sample_rate=24000,
            duration_seconds=0.0,
            model_name=self.repo_id,
            processing_duration=round(time.time() - t0, 3),
            speaker="IndicF5",
            language=spec.code,
            success=False,
            error="IndicF5 requires a reference audio prompt (.wav) for voice cloning.",
        )


class UnifiedSpeechEngine:
    """
    Primary singleton facade for neural speech synthesis.
    Defaults to Meta MMS-TTS for guaranteed offline, low-latency, 100% open speech,
    with IndicF5 evaluated and selectable.
    """

    def __init__(self):
        self.mms_engine = MMSSpeechEngine()
        self.indicf5_engine = IndicF5SpeechEngine()

    def synthesize(
        self,
        text: str,
        target_lang: str,
        voice_id: Optional[str] = None,
    ) -> SpeechResult:
        if voice_id and voice_id.startswith("indicf5"):
            return self.indicf5_engine.synthesize(text, target_lang, voice_id)
        # Default high-performance neural engine
        return self.mms_engine.synthesize(text, target_lang, voice_id)

    def is_available(self, target_lang: str) -> bool:
        return self.mms_engine.is_available(target_lang)
