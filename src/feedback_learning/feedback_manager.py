"""
Feedback Manager Facade.
========================
High-level orchestrator coordinating prediction snapshots, feedback events,
personal adaptation, and global learning workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    PRODUCTION_MODEL_VERSION,
    PRODUCTION_MODEL_CHECKSUM,
    FEATURE_SCHEMA_SINGLE_HAND,
    GLOBAL_CONTRIBUTION_DEFAULT_ENABLED,
)
from .prediction_snapshot import PredictionSnapshot, SnapshotRingBuffer
from .sample_collector import SampleCollector
from .quality_filter import QualityFilter
from .personal_store import PersonalStore
from .personal_adapter import PersonalAdapter
from .global_contribution import GlobalContributionManager
from .global_dataset import GlobalDatasetManager
from .model_registry import ModelRegistry
from .evaluator import ModelEvaluator
from .trainer import OfflineTrainer
from .privacy_manager import PrivacyManager


class FeedbackManager:
    """Unified subsystem entry point for Continuous Human-in-the-Loop Learning."""

    def __init__(
        self,
        collector: Optional[SampleCollector] = None,
        ring_buffer: Optional[SnapshotRingBuffer] = None,
        privacy_manager: Optional[PrivacyManager] = None,
        global_contribution: Optional[GlobalContributionManager] = None,
        global_dataset: Optional[GlobalDatasetManager] = None,
        personal_store: Optional[PersonalStore] = None,
        personal_adapter: Optional[PersonalAdapter] = None,
    ):
        self.collector = collector or SampleCollector()
        self.ring_buffer = ring_buffer or SnapshotRingBuffer()
        self.quality_filter = QualityFilter()

        self.personal_store = personal_store or PersonalStore(self.collector)
        self.personal_adapter = personal_adapter or PersonalAdapter(self.personal_store)

        self.global_contribution = global_contribution or GlobalContributionManager(self.collector)
        self.global_dataset = global_dataset or GlobalDatasetManager(self.collector)

        self.model_registry = ModelRegistry(collector=self.collector)
        self.evaluator = ModelEvaluator()
        self.trainer = OfflineTrainer(self.global_dataset, self.model_registry)
        self.privacy_manager = privacy_manager or PrivacyManager(
            collector=self.collector,
            contribution_manager=self.global_contribution,
        )

    def capture_prediction_snapshot(
        self,
        predicted_alphabet: str,
        confidence_score: float,
        feature_vector: np.ndarray,
        feature_schema: str = FEATURE_SCHEMA_SINGLE_HAND,
        raw_landmarks: Optional[List[Dict[str, Any]]] = None,
        tracking_quality: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> PredictionSnapshot:
        """
        Create, buffer, and record a prediction snapshot.
        Returns the snapshot instance containing the unique prediction_id.
        """
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        snapshot = PredictionSnapshot(
            predicted_alphabet=predicted_alphabet,
            confidence_score=confidence_score,
            feature_vector=feature_vector,
            feature_schema=feature_schema,
            raw_landmarks=raw_landmarks,
            tracking_quality=tracking_quality,
            metadata=metadata,
            user_id=clean_uid,
            production_model_version=PRODUCTION_MODEL_VERSION,
            model_checksum=PRODUCTION_MODEL_CHECKSUM,
        )

        self.ring_buffer.add(snapshot)
        # Asynchronously or directly record event in DB
        self.collector.save_prediction_event(snapshot)
        return snapshot

    def reassign_snapshots(self, old_user_id: str, new_user_id: str) -> int:
        """
        Reassign buffered prediction snapshots from old_user_id to new_user_id.
        Used when the frontend establishes its authenticated session upon connection.
        """
        clean_old = self.privacy_manager.sanitize_user_id(old_user_id)
        clean_new = self.privacy_manager.sanitize_user_id(new_user_id)
        if not clean_old or not clean_new or clean_old == clean_new:
            return 0
        count = 0
        with self.ring_buffer._lock:
            for s in self.ring_buffer._snapshots.values():
                if s.user_id == clean_old:
                    s.user_id = clean_new
                    count += 1
        return count

    def confirm_prediction(
        self,
        prediction_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """User confirmed that prediction was correct ("Yes")."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        snapshot = self.ring_buffer.get(prediction_id)

        # Verify session ownership
        if snapshot is not None and snapshot.user_id and snapshot.user_id != clean_uid:
            return {
                "ok": False,
                "error": f"Forbidden: Prediction snapshot '{prediction_id}' was recorded by another session.",
            }

        res = self.collector.save_confirmation(
            prediction_id=prediction_id,
            user_id=clean_uid,
            snapshot=snapshot,
        )
        return res

    def correct_prediction(
        self,
        prediction_id: str,
        actual_label: str,
        user_id: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        User corrected an incorrect prediction ("No" -> actual letter).
        Must be attached to the exact original prediction snapshot!
        """
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        snapshot = self.ring_buffer.get(prediction_id)

        if snapshot is None:
            return False, {
                "ok": False,
                "error": f"Prediction snapshot '{prediction_id}' expired or not found. "
                         "Feedback must reference an active prediction snapshot.",
            }

        # Verify session ownership
        if snapshot.user_id and snapshot.user_id != clean_uid:
            return False, {
                "ok": False,
                "error": f"Forbidden: Prediction snapshot '{prediction_id}' was recorded by another session.",
            }

        success, res = self.collector.save_correction(
            snapshot=snapshot,
            actual_label=actual_label,
            user_id=clean_uid,
        )

        if success and res.get("is_valid"):
            # Update in-memory personal store for real-time personalization
            self.personal_store.add_exemplar(
                user_id=clean_uid,
                label=actual_label.upper(),
                features=snapshot.feature_vector,
            )

        return success, res

    def adapt_prediction(
        self,
        user_id: Optional[str],
        features: np.ndarray,
        production_prediction: Optional[str],
        production_confidence: float,
    ) -> Dict[str, Any]:
        """Apply user-specific conservative personalization."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        user_info = self.collector.get_or_create_user(clean_uid)
        pers_enabled = bool(user_info.get("personalization_enabled", True))

        return self.personal_adapter.adapt_prediction(
            user_id=clean_uid,
            features=features,
            production_prediction=production_prediction,
            production_confidence=production_confidence,
            personalization_enabled=pers_enabled,
        )

    def get_preferences(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Get user learning preferences."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        return self.collector.get_or_create_user(clean_uid)

    def update_preferences(
        self,
        user_id: Optional[str],
        personalization_enabled: Optional[bool] = None,
        global_contribution_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Update user preferences."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        return self.collector.update_user_preferences(
            user_id=clean_uid,
            personalization_enabled=personalization_enabled,
            global_contribution_enabled=global_contribution_enabled,
        )

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate feedback and learning statistics."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id) if user_id else None
        return self.collector.get_feedback_stats(clean_uid)

    def get_history(self, user_id: Optional[str], limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve feedback review history for user."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        return self.collector.get_feedback_history(clean_uid, limit=limit)

    def reset_personal_data(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Reset user's personal corrections and exemplars."""
        clean_uid = self.privacy_manager.sanitize_user_id(user_id)
        self.personal_store.clear_user_cache(clean_uid)
        return self.collector.reset_personal_data(clean_uid)
