#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Manager module.
Handles WAV audio caching, disk persistence, base64 encoding,
WAV header verification, and automatic cleanup of expired audio.
"""

import base64
import hashlib
import io
import os
import struct
import time
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import default_config


class AudioManager:
    """Manages audio generation artifacts, metadata, and browser delivery."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or default_config.speech.audio_output_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Tuple[bytes, dict]] = {}

    def generate_audio_id(self, text: str, lang: str, voice_id: str) -> str:
        """Create a deterministic unique hash for the speech request."""
        raw_key = f"{lang}:{voice_id}:{text.strip()}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def save_wav(
        self,
        audio_id: str,
        audio_bytes: bytes,
        sample_rate: int,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Save WAV bytes to cache directory and update in-memory cache."""
        file_path = self.cache_dir / f"{audio_id}.wav"
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        meta = metadata or {}
        meta.update({
            "audio_id": audio_id,
            "file_path": str(file_path),
            "sample_rate": sample_rate,
            "size_bytes": len(audio_bytes),
            "timestamp": time.time(),
        })
        self._memory_cache[audio_id] = (audio_bytes, meta)
        return file_path

    def get_audio(self, audio_id: str) -> Optional[Tuple[bytes, dict]]:
        """Retrieve audio bytes and metadata by audio_id."""
        if audio_id in self._memory_cache:
            return self._memory_cache[audio_id]

        file_path = self.cache_dir / f"{audio_id}.wav"
        if file_path.exists():
            with open(file_path, "rb") as f:
                data = f.read()
            duration, sr = self.inspect_wav(data)
            meta = {
                "audio_id": audio_id,
                "file_path": str(file_path),
                "sample_rate": sr,
                "duration_seconds": duration,
                "size_bytes": len(data),
            }
            self._memory_cache[audio_id] = (data, meta)
            return data, meta
        return None

    def to_base64_data_uri(self, audio_bytes: bytes) -> str:
        """Convert WAV bytes to data:audio/wav;base64 URI for instant DOM playback."""
        b64 = base64.b64encode(audio_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"

    @staticmethod
    def inspect_wav(audio_bytes: bytes) -> Tuple[float, int]:
        """
        Inspect WAV bytes and return (duration_in_seconds, sample_rate).
        Raises ValueError if invalid WAV.
        """
        if len(audio_bytes) < 44:
            raise ValueError("Audio data too short to be a valid WAV file")

        # Verify RIFF / WAVE signature
        riff, _, wave_id = struct.unpack("<4sI4s", audio_bytes[:12])
        if riff != b"RIFF" or wave_id != b"WAVE":
            raise ValueError("Invalid WAV header format")

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
                return round(duration, 3), sample_rate
        except Exception as e:
            raise ValueError(f"Failed to parse WAV stream: {e}")

    def cleanup_old_files(self, max_age_hours: int = 6) -> int:
        """Delete audio files older than max_age_hours to prevent disk bloat."""
        now = time.time()
        max_age_sec = max_age_hours * 3600
        removed = 0
        for p in self.cache_dir.glob("*.wav"):
            try:
                if now - p.stat().st_mtime > max_age_sec:
                    p.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                pass
        return removed
