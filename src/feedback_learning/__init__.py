"""
Human-in-the-Loop Continuous Learning for Indian Sign Language Recognition.
=============================================================================
Provides safe, decoupled personal and global feedback learning subsystems.
"""

from .config import (
    PRODUCTION_MODEL_VERSION,
    PRODUCTION_MODEL_CHECKSUM,
    FEATURE_SCHEMA_SINGLE_HAND,
    FEATURE_SCHEMA_DUAL_HAND,
)
from .prediction_snapshot import PredictionSnapshot, SnapshotRingBuffer
from .quality_filter import QualityFilter, QualityAssessment
from .sample_collector import SampleCollector
from .personal_store import PersonalStore
from .personal_adapter import PersonalAdapter
from .global_contribution import GlobalContributionManager
from .global_dataset import GlobalDatasetManager
from .model_registry import ModelRegistry
from .evaluator import ModelEvaluator, EvaluationReport
from .trainer import OfflineTrainer
from .privacy_manager import PrivacyManager
from .feedback_manager import FeedbackManager
from .feedback_api import feedback_bp, get_feedback_manager

__all__ = [
    "PRODUCTION_MODEL_VERSION",
    "PRODUCTION_MODEL_CHECKSUM",
    "FEATURE_SCHEMA_SINGLE_HAND",
    "FEATURE_SCHEMA_DUAL_HAND",
    "PredictionSnapshot",
    "SnapshotRingBuffer",
    "QualityFilter",
    "QualityAssessment",
    "SampleCollector",
    "PersonalStore",
    "PersonalAdapter",
    "GlobalContributionManager",
    "GlobalDatasetManager",
    "ModelRegistry",
    "ModelEvaluator",
    "EvaluationReport",
    "OfflineTrainer",
    "PrivacyManager",
    "FeedbackManager",
    "feedback_bp",
    "get_feedback_manager",
]
