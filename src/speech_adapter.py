#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Speech Adapter Module.
=============================
Provides a resilient, backward-compatible text-to-speech adapter for the ISL Recognition
backend. Dispatches requests to an external neural speech microservice (e.g., Meta MMS-TTS)
when enabled and available, while strictly preserving browser Web Speech as the default,
zero-cost, commercially unrestricted fallback.

Design Guarantees:
1. Zero heavy neural dependencies: uses standard library urllib/json; no PyTorch or
   Transformers in the recognition host.
2. Safe failure isolation: connection timeouts, HTTP errors, and malformed responses
   never crash the backend or interrupt video recognition streaming.
3. Thread-safe in-memory audio caching for low latency on repeated sentences.
4. Non-commercial licensing safety: Neural TTS is disabled by default (ENABLE_NEURAL_TTS=0)
   to respect Meta MMS-TTS CC-BY-NC 4.0 licensing.
"""

import collections
import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SpeechAdapter:
    """
    Adapter orchestrating neural text-to-speech requests to an external language service
    with automatic, seamless fallback to the browser Web Speech API.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None,
        max_cache_size: int = 200,
    ):
        env_enabled = os.environ.get("ENABLE_NEURAL_TTS", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self.enabled = enabled if enabled is not None else env_enabled
        self.service_url = (
            service_url
            if service_url is not None
            else os.environ.get("LANGUAGE_SERVICE_URL", "").strip().rstrip("/")
        )
        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("NEURAL_TTS_TIMEOUT", "3.0"))
        )
        self.max_cache_size = max_cache_size

        # In-memory thread-safe audio cache (hash -> audio metadata dict)
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

        # Telemetry / diagnostic counters
        self._stats = {
            "requests_total": 0,
            "neural_success": 0,
            "cache_hits": 0,
            "fallbacks_disabled": 0,
            "fallbacks_timeout": 0,
            "fallbacks_http_error": 0,
            "fallbacks_connection_error": 0,
            "fallbacks_malformed": 0,
        }

    def is_neural_enabled(self) -> bool:
        """Check if neural TTS is actively enabled and a service URL is configured."""
        return bool(self.enabled and self.service_url)

    def _cache_key(self, text: str, target_lang: str, voice_id: Optional[str]) -> str:
        raw = f"{target_lang.strip().lower()}:{str(voice_id or '').strip()}:{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached(self, text: str, target_lang: str, voice_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve cached speech result if present."""
        key = self._cache_key(text, target_lang, voice_id)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats["cache_hits"] += 1
                item = dict(self._cache[key])
                item["cached"] = True
                return item
        return None

    def store_cache(self, text: str, target_lang: str, voice_id: Optional[str], result: Dict[str, Any]) -> None:
        """Store synthesized speech result in bounded cache."""
        if not result.get("success") or not result.get("data_uri"):
            return
        key = self._cache_key(text, target_lang, voice_id)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = dict(result)
            while len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Clear the audio cache."""
        with self._lock:
            self._cache.clear()

    def synthesize(
        self,
        text: str,
        target_lang: str = "English",
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize text to speech using the neural service if enabled,
        or signal fallback to the browser Web Speech API.

        Returns a dictionary with:
        - success: bool
        - fallback: "browser" or None
        - engine: "neural" or "browser"
        - data_uri: str (base64 audio) or None
        - sample_rate: int or None
        - duration_seconds: float
        - reason: str (explanation if fallback was triggered)
        """
        with self._lock:
            self._stats["requests_total"] += 1

        clean_text = str(text or "").strip()
        clean_lang = str(target_lang or "English").strip()

        if not clean_text:
            return {
                "success": False,
                "error": "Empty text provided",
                "fallback": "browser",
                "engine": "browser",
                "reason": "empty_text",
                "data_uri": None,
                "sample_rate": None,
                "duration_seconds": 0.0,
                "model_name": None,
                "processing_duration": 0.0,
                "speaker": None,
                "language": clean_lang,
                "cached": False,
            }

        # Check if neural TTS is disabled or unconfigured
        if not self.is_neural_enabled():
            with self._lock:
                self._stats["fallbacks_disabled"] += 1
            return {
                "success": False,
                "error": None,
                "fallback": "browser",
                "engine": "browser",
                "reason": "neural_tts_disabled",
                "data_uri": None,
                "sample_rate": None,
                "duration_seconds": 0.0,
                "model_name": None,
                "processing_duration": 0.0,
                "speaker": None,
                "language": clean_lang,
                "cached": False,
            }

        # Check in-memory audio cache
        cached = self.get_cached(clean_text, clean_lang, voice_id)
        if cached is not None:
            return cached

        # Prepare payload for neural microservice
        endpoint = f"{self.service_url}/api/v2/language/speak"
        payload = {
            "text": clean_text,
            "target_lang": clean_lang,
            "voice_id": voice_id,
        }

        req_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=endpoint,
            data=req_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ISL-Recognition-SpeechAdapter/1.0",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                resp_text = resp_bytes.decode("utf-8", errors="replace")
                data = json.loads(resp_text)

            latency = round(time.time() - t0, 3)

            if isinstance(data, dict) and data.get("success") and data.get("data_uri"):
                with self._lock:
                    self._stats["neural_success"] += 1

                res = {
                    "success": True,
                    "error": None,
                    "fallback": None,
                    "engine": "neural",
                    "reason": None,
                    "data_uri": data.get("data_uri"),
                    "audio_id": data.get("audio_id"),
                    "sample_rate": data.get("sample_rate", 16000),
                    "duration_seconds": float(data.get("duration_seconds") or 0.0),
                    "model_name": data.get("model_name", "mms-tts"),
                    "processing_duration": latency,
                    "speaker": data.get("speaker") or voice_id or "default",
                    "language": data.get("language") or clean_lang,
                    "cached": False,
                }
                self.store_cache(clean_text, clean_lang, voice_id, res)
                return res
            else:
                err_msg = data.get("error") if isinstance(data, dict) else "Invalid response body"
                logger.warning("[SpeechAdapter] Neural service returned unsuccessful: %s", err_msg)
                with self._lock:
                    self._stats["fallbacks_malformed"] += 1
                return self._fallback_result(clean_lang, f"malformed_or_unsuccessful: {err_msg}")

        except (TimeoutError, urllib.error.URLError) as e:
            # Check for timeout specifically
            is_timeout = isinstance(e, TimeoutError) or "timed out" in str(e).lower()
            with self._lock:
                if is_timeout:
                    self._stats["fallbacks_timeout"] += 1
                elif isinstance(e, urllib.error.HTTPError):
                    self._stats["fallbacks_http_error"] += 1
                else:
                    self._stats["fallbacks_connection_error"] += 1

            reason = "timeout" if is_timeout else f"network_error: {e}"
            logger.warning("[SpeechAdapter] Neural service unreachable (%s). Triggering browser fallback.", reason)
            return self._fallback_result(clean_lang, reason)

        except Exception as e:
            with self._lock:
                self._stats["fallbacks_malformed"] += 1
            logger.warning("[SpeechAdapter] Unexpected synthesis error: %s", e)
            return self._fallback_result(clean_lang, f"unexpected_error: {e}")

    def _fallback_result(self, lang: str, reason: str) -> Dict[str, Any]:
        """Construct standard fallback result dictionary."""
        return {
            "success": False,
            "error": reason,
            "fallback": "browser",
            "engine": "browser",
            "reason": reason,
            "data_uri": None,
            "audio_id": None,
            "sample_rate": None,
            "duration_seconds": 0.0,
            "model_name": None,
            "processing_duration": 0.0,
            "speaker": None,
            "language": lang,
            "cached": False,
        }

    def get_voices(self, target_lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch available neural voice catalog from remote service."""
        if not self.is_neural_enabled():
            return []

        url = f"{self.service_url}/api/v2/language/voices"
        if target_lang:
            url += f"?lang={urllib.parse.quote(target_lang)}"

        try:
            req = urllib.request.Request(
                url=url,
                headers={"Accept": "application/json", "User-Agent": "ISL-Recognition-SpeechAdapter/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("success"):
                return data.get("voices", [])
        except Exception as e:
            logger.debug("[SpeechAdapter] Could not retrieve remote voice catalog: %s", e)
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["cache_size"] = len(self._cache)
            stats["neural_enabled"] = self.is_neural_enabled()
            stats["service_url"] = self.service_url
            stats["timeout"] = self.timeout
            return stats


# Global singleton instance
_speech_adapter_instance: Optional[SpeechAdapter] = None
_speech_adapter_lock = threading.Lock()


def get_speech_adapter(
    service_url: Optional[str] = None,
    enabled: Optional[bool] = None,
    timeout: Optional[float] = None,
) -> SpeechAdapter:
    """Retrieve or initialize the process-wide SpeechAdapter singleton."""
    global _speech_adapter_instance
    with _speech_adapter_lock:
        if _speech_adapter_instance is None:
            _speech_adapter_instance = SpeechAdapter(
                service_url=service_url,
                enabled=enabled,
                timeout=timeout,
            )
        elif service_url is not None or enabled is not None or timeout is not None:
            # Update configuration if explicitly requested
            if service_url is not None:
                _speech_adapter_instance.service_url = service_url.rstrip("/")
            if enabled is not None:
                _speech_adapter_instance.enabled = enabled
            if timeout is not None:
                _speech_adapter_instance.timeout = float(timeout)
        return _speech_adapter_instance


def reset_speech_adapter() -> None:
    """Reset the global singleton (used in testing)."""
    global _speech_adapter_instance
    with _speech_adapter_lock:
        _speech_adapter_instance = None
