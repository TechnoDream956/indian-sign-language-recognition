#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation Adapter Module.
===========================
Provides a resilient, backward-compatible translation adapter for the ISL Recognition
backend. Dispatches requests to an external IndicTrans2 neural translation microservice
when enabled and available, with automatic, transparent fallback to the local legacy
dictionary and rule-based translator.

Design Guarantees:
1. Zero heavy neural dependencies: uses standard library urllib/json; no PyTorch or Transformers in recognition host.
2. Safe failure isolation: connection timeouts, HTTP errors, and malformed responses never crash the backend.
3. Thread-safe, language-partitioned caching.
4. Non-blocking: recognition frame processing remains decoupled from remote network calls.
"""

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported 10 dashboard languages mapping
DASHBOARD_LANGUAGES: Dict[str, str] = {
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

# Reverse mapping from code to display name
CODE_TO_DASHBOARD_NAME: Dict[str, str] = {v: k for k, v in DASHBOARD_LANGUAGES.items()}


def normalize_target_language(lang: str) -> Tuple[str, str]:
    """
    Resolve language string to (canonical_name, lang_code).
    Defaults to ('English', 'en') if unrecognized.
    """
    if not lang or not str(lang).strip():
        return "English", "en"

    clean = str(lang).strip()
    # Check exact name match
    if clean in DASHBOARD_LANGUAGES:
        return clean, DASHBOARD_LANGUAGES[clean]

    # Check case-insensitive match
    lower = clean.lower()
    for name, code in DASHBOARD_LANGUAGES.items():
        if name.lower() == lower or code.lower() == lower:
            return name, code

    # Fallback to provided name
    return clean, clean.lower()[:3]


class TranslationAdapter:
    """
    Adapter orchestrating neural translation requests to an external language service
    with automatic local fallback.
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: Optional[float] = None,
        max_cache_size: int = 2000,
    ):
        env_enabled = os.environ.get("ENABLE_NEURAL_TRANSLATION", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self.enabled = enabled if enabled is not None else env_enabled
        self.service_url = (
            service_url
            if service_url is not None
            else os.environ.get("LANGUAGE_SERVICE_URL", "").strip()
        )
        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(os.environ.get("NEURAL_TRANSLATION_TIMEOUT", "2.5"))
        )
        self.max_cache_size = max_cache_size

        self._cache: Dict[str, Tuple[str, str]] = {}  # key -> (translated_text, engine_name)
        self._lock = threading.Lock()

    def is_neural_enabled(self) -> bool:
        """Check whether neural translation is enabled and has a valid service URL."""
        return bool(self.enabled and self.service_url and self.service_url.strip())

    def clear_cache(self) -> None:
        """Clear adapter memory cache."""
        with self._lock:
            self._cache.clear()

    def _make_cache_key(self, text: str, lang_code: str) -> str:
        """Generate normalized cache key partitioned by language and lowercased text."""
        norm_text = re.sub(r"\s+", " ", text.strip().lower())
        return f"{lang_code.lower()}:{norm_text}"

    def _validate_translated_text(self, original_text: str, translated_text: str, lang_code: str) -> bool:
        """
        Validate that translated text is valid and not an empty string or raw untranslated English.
        """
        if not translated_text or not translated_text.strip():
            return False

        clean_trans = translated_text.strip()
        clean_orig = original_text.strip()

        # If translating to an Indic language, the translation should not be identical to source English
        if lang_code.lower() not in ("en", "eng") and clean_trans.lower() == clean_orig.lower():
            return False

        return True

    def _call_neural_service(self, text: str, target_lang: str, lang_code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Perform HTTP request to the external language service.
        Returns (success, translated_text, error_message).
        """
        base_url = self.service_url.rstrip("/")
        endpoint = f"{base_url}/api/v2/language/translate"

        payload = json.dumps({
            "text": text,
            "target_lang": target_lang,
            "source_lang": "en",
        }).encode("utf-8")

        max_retries = 2
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.getcode()
                    body = resp.read().decode("utf-8")

                    if status != 200:
                        if status in (429, 502, 503, 504) and attempt < max_retries:
                            time.sleep(0.25 * (2 ** attempt))
                            continue
                        return False, None, f"HTTP {status}"

                    data = json.loads(body)
                    raw_translation = data.get("translated_text", "")
                    if self._validate_translated_text(text, raw_translation, lang_code):
                        return True, raw_translation.strip(), None
                    else:
                        return False, None, "Invalid or untranslated output from neural model"

            except urllib.error.HTTPError as e:
                if e.code in (429, 502, 503, 504) and attempt < max_retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                return False, None, f"HTTPError {e.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < max_retries:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                return False, None, f"NetworkError: {type(e).__name__}"
            except json.JSONDecodeError as e:
                return False, None, f"MalformedJSON: {e}"
            except Exception as e:
                return False, None, f"UnexpectedError: {type(e).__name__}"

        return False, None, "RetriesExhausted"

    def translate(self, text: Optional[str], target_lang: Optional[str]) -> Dict[str, Any]:
        """
        Translate input text into target language with fallback guarantees.
        """
        raw_text = text or ""
        clean_text = raw_text.strip()
        canonical_name, lang_code = normalize_target_language(target_lang or "English")

        # 1. Empty text check
        if not clean_text:
            return {
                "ok": True,
                "source": raw_text,
                "language": canonical_name,
                "translation": "",
                "engine": "none",
                "cached": False,
            }

        # 2. English passthrough
        if lang_code in ("en", "eng") or canonical_name.lower() == "english":
            return {
                "ok": True,
                "source": raw_text,
                "language": canonical_name,
                "translation": clean_text,
                "engine": "identity",
                "cached": False,
            }

        cache_key = self._make_cache_key(clean_text, lang_code)

        # 3. Cache lookup
        with self._lock:
            if cache_key in self._cache:
                cached_text, engine_used = self._cache[cache_key]
                return {
                    "ok": True,
                    "source": raw_text,
                    "language": canonical_name,
                    "translation": cached_text,
                    "engine": engine_used,
                    "cached": True,
                }

        # 4. Neural translation path (when enabled)
        if self.is_neural_enabled():
            success, translated_text, err = self._call_neural_service(
                clean_text, canonical_name, lang_code
            )
            if success and translated_text:
                with self._lock:
                    if len(self._cache) >= self.max_cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = (translated_text, "indictrans2")

                return {
                    "ok": True,
                    "source": raw_text,
                    "language": canonical_name,
                    "translation": translated_text,
                    "engine": "indictrans2",
                    "cached": False,
                }
            else:
                logger.debug(
                    "[TranslationAdapter] Neural translation unavailable (%s). Falling back to legacy.",
                    err,
                )

        # 5. Local legacy fallback path
        try:
            try:
                from translator import translate_sentence
            except ImportError:
                from src.translator import translate_sentence
            legacy_trans = translate_sentence(clean_text, canonical_name)
        except Exception as e:
            logger.warning("[TranslationAdapter] Legacy translator raised exception: %s", e)
            legacy_trans = clean_text

        with self._lock:
            if len(self._cache) >= self.max_cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = (legacy_trans, "legacy")

        return {
            "ok": True,
            "source": raw_text,
            "language": canonical_name,
            "translation": legacy_trans,
            "engine": "legacy",
            "cached": False,
        }


# Singleton instance
_global_adapter: Optional[TranslationAdapter] = None
_adapter_lock = threading.Lock()


def get_translation_adapter() -> TranslationAdapter:
    """Retrieve or initialize singleton TranslationAdapter instance."""
    global _global_adapter
    if _global_adapter is None:
        with _adapter_lock:
            if _global_adapter is None:
                _global_adapter = TranslationAdapter()
    return _global_adapter
