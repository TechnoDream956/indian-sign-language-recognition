"""
Prediction Snapshot Management.
================================
Captures, serializes, and manages immutability-preserved prediction snapshots.
Ensures user feedback is attached to the exact frame and landmarks that caused
the prediction, never the subsequent frame when the button was pressed.
"""

import time
import uuid
import threading
from typing import Any, Dict, List, Optional
import numpy as np

from .config import (
    PRODUCTION_MODEL_VERSION,
    PRODUCTION_MODEL_CHECKSUM,
    FEATURE_SCHEMA_SINGLE_HAND,
    SNAPSHOT_RING_BUFFER_MAX_SIZE,
    SNAPSHOT_TTL_SECONDS,
)


class PredictionSnapshot:
    """Immutable snapshot of an ISL inference event."""

    def __init__(
        self,
        predicted_alphabet: str,
        confidence_score: float,
        feature_vector: np.ndarray,
        feature_schema: str = FEATURE_SCHEMA_SINGLE_HAND,
        raw_landmarks: Optional[List[Dict[str, Any]]] = None,
        tracking_quality: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        prediction_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        production_model_version: str = PRODUCTION_MODEL_VERSION,
        model_checksum: str = PRODUCTION_MODEL_CHECKSUM,
    ):
        self.prediction_id = prediction_id or f"pred_{uuid.uuid4().hex[:12]}"
        self.predicted_alphabet = (predicted_alphabet or "").upper()
        self.confidence_score = float(confidence_score)
        self.timestamp = float(timestamp or time.time())
        self.production_model_version = production_model_version
        self.model_checksum = model_checksum
        self.feature_schema = feature_schema
        self.user_id = user_id or "anonymous_local"

        # Ensure feature vector is a float32 numpy array
        if isinstance(feature_vector, np.ndarray):
            self.feature_vector = feature_vector.copy().astype(np.float32)
        elif isinstance(feature_vector, (list, tuple)):
            self.feature_vector = np.array(feature_vector, dtype=np.float32)
        else:
            self.feature_vector = np.zeros(63, dtype=np.float32)

        self.raw_landmarks = raw_landmarks or []
        self.tracking_quality = tracking_quality or {}
        self.metadata = metadata or {}

    def to_dict(self, include_features: bool = True) -> Dict[str, Any]:
        """Convert snapshot to dictionary for API response or storage."""
        data = {
            "prediction_id": self.prediction_id,
            "predicted_alphabet": self.predicted_alphabet,
            "confidence_score": round(self.confidence_score, 4),
            "timestamp": self.timestamp,
            "production_model_version": self.production_model_version,
            "model_checksum": self.model_checksum,
            "feature_schema": self.feature_schema,
            "user_id": self.user_id,
            "tracking_quality": self.tracking_quality,
            "metadata": self.metadata,
        }
        if include_features:
            data["feature_vector"] = self.feature_vector.tolist()
            data["raw_landmarks"] = self.raw_landmarks
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionSnapshot":
        """Deserialize from dictionary."""
        feats = np.array(data.get("feature_vector", []), dtype=np.float32)
        return cls(
            predicted_alphabet=data.get("predicted_alphabet", ""),
            confidence_score=data.get("confidence_score", 0.0),
            feature_vector=feats,
            feature_schema=data.get("feature_schema", FEATURE_SCHEMA_SINGLE_HAND),
            raw_landmarks=data.get("raw_landmarks"),
            tracking_quality=data.get("tracking_quality"),
            metadata=data.get("metadata"),
            user_id=data.get("user_id"),
            prediction_id=data.get("prediction_id"),
            timestamp=data.get("timestamp"),
            production_model_version=data.get("production_model_version", PRODUCTION_MODEL_VERSION),
            model_checksum=data.get("model_checksum", PRODUCTION_MODEL_CHECKSUM),
        )


class SnapshotRingBuffer:
    """Thread-safe bounded ring buffer retaining recent prediction snapshots."""

    def __init__(
        self,
        max_size: int = SNAPSHOT_RING_BUFFER_MAX_SIZE,
        ttl_seconds: float = SNAPSHOT_TTL_SECONDS,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._snapshots: Dict[str, PredictionSnapshot] = {}
        self._order: List[str] = []
        self._last_snapshot: Optional[PredictionSnapshot] = None

    def add(self, snapshot: PredictionSnapshot) -> str:
        """Add a snapshot to the buffer and prune expired entries."""
        with self._lock:
            now = time.time()
            pid = snapshot.prediction_id

            # Prune by TTL
            self._prune_expired_locked(now)

            # Prune if max size reached
            while len(self._order) >= self.max_size:
                oldest_id = self._order.pop(0)
                self._snapshots.pop(oldest_id, None)

            self._snapshots[pid] = snapshot
            self._order.append(pid)
            self._last_snapshot = snapshot
            return pid

    def get(self, prediction_id: str) -> Optional[PredictionSnapshot]:
        """Retrieve snapshot by prediction_id if present and not expired."""
        with self._lock:
            snapshot = self._snapshots.get(prediction_id)
            if snapshot is None:
                return None
            if time.time() - snapshot.timestamp > self.ttl_seconds:
                # Expired
                self._snapshots.pop(prediction_id, None)
                if prediction_id in self._order:
                    self._order.remove(prediction_id)
                return None
            return snapshot

    def get_latest(self) -> Optional[PredictionSnapshot]:
        """Get the most recent prediction snapshot."""
        with self._lock:
            if self._last_snapshot is not None:
                if time.time() - self._last_snapshot.timestamp <= self.ttl_seconds:
                    return self._last_snapshot
            return None

    def _prune_expired_locked(self, current_time: float) -> None:
        """Internal prune of expired snapshots."""
        to_remove = []
        for pid, s in self._snapshots.items():
            if current_time - s.timestamp > self.ttl_seconds:
                to_remove.append(pid)
        for pid in to_remove:
            self._snapshots.pop(pid, None)
            if pid in self._order:
                self._order.remove(pid)

    def count(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()
            self._order.clear()
            self._last_snapshot = None
