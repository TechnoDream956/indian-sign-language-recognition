"""
Personal Learning Store.
========================
User-isolated store for personal landmark corrections and exemplars.
Guarantees absolute data isolation between different users.
"""

import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

from .sample_collector import SampleCollector


class PersonalStore:
    """Stores and retrieves in-memory and persistent personal exemplars per user."""

    def __init__(self, collector: Optional[SampleCollector] = None):
        self.collector = collector or SampleCollector()
        self._lock = threading.RLock()
        # In-memory exemplar cache: {user_id: {label: [array, array, ...]}}
        self._cache: Dict[str, Dict[str, List[np.ndarray]]] = {}

    def get_user_exemplars(self, user_id: str, reload_db: bool = False) -> Dict[str, List[np.ndarray]]:
        """Retrieve all verified landmark exemplars for a user, categorized by label."""
        with self._lock:
            if not reload_db and user_id in self._cache:
                return self._cache[user_id]

        # Load from collector storage
        user_exemplars: Dict[str, List[np.ndarray]] = {}
        with self.collector._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT label, feature_file FROM landmark_samples
                WHERE user_id = ?
                """,
                (user_id,),
            )
            rows = cur.fetchall()

        for r in rows:
            lbl = r["label"]
            fpath = self.collector.storage_dir / r["feature_file"]
            if fpath.exists():
                try:
                    data = np.load(str(fpath))
                    feat = data["features"].astype(np.float32)
                    if lbl not in user_exemplars:
                        user_exemplars[lbl] = []
                    user_exemplars[lbl].append(feat)
                except Exception:
                    pass

        with self._lock:
            self._cache[user_id] = user_exemplars
            return self._cache[user_id]

    def add_exemplar(self, user_id: str, label: str, features: np.ndarray) -> None:
        """Add a new in-memory exemplar for the user."""
        with self._lock:
            if user_id not in self._cache:
                self._cache[user_id] = {}
            if label not in self._cache[user_id]:
                self._cache[user_id][label] = []
            self._cache[user_id][label].append(features.copy().astype(np.float32))

    def clear_user_cache(self, user_id: str) -> None:
        """Evict user exemplars from in-memory cache."""
        with self._lock:
            self._cache.pop(user_id, None)

    def get_exemplar_stats(self, user_id: str) -> Dict[str, int]:
        """Return count of exemplars per class for the user."""
        exemplars = self.get_user_exemplars(user_id)
        return {lbl: len(samples) for lbl, samples in exemplars.items()}
