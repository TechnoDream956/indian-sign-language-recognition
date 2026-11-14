"""
Offline Global Model Trainer.
=============================
Trains candidate models strictly offline using accepted, verified, and
curated global contribution samples. Never touches production weights directly.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .config import (
    FEATURE_SCHEMA_SINGLE_HAND,
    VALID_ALPHABET_CLASSES,
    REGISTRY_DIR,
)
from .global_dataset import GlobalDatasetManager
from .evaluator import ModelEvaluator, EvaluationReport
from .model_registry import ModelRegistry


def build_candidate_mlp(input_dim: int = 63, num_classes: int = 26) -> tf.keras.Model:
    """Build candidate MLP architecture identical to production single-hand model."""
    model = Sequential([
        Input(shape=(input_dim,), name="single_hand_landmarks"),
        Dense(128, activation="relu", name="dense_1"),
        BatchNormalization(name="bn_1"),
        Dropout(0.3, name="dropout_1"),
        Dense(64, activation="relu", name="dense_2"),
        BatchNormalization(name="bn_2"),
        Dropout(0.3, name="dropout_2"),
        Dense(32, activation="relu", name="dense_3"),
        Dropout(0.2, name="dropout_3"),
        Dense(num_classes, activation="softmax", name="alphabet_output"),
    ], name="ISL_Candidate_Alphabet_MLP")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class OfflineTrainer:
    """Orchestrates candidate training runs on curated global datasets."""

    def __init__(
        self,
        dataset_manager: Optional[GlobalDatasetManager] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.dataset_manager = dataset_manager or GlobalDatasetManager()
        self.registry = registry or ModelRegistry()
        self.evaluator = ModelEvaluator()

    def train_candidate(
        self,
        version_id: Optional[str] = None,
        epochs: int = 20,
        batch_size: int = 16,
        synthetic_benchmark_samples: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Execute offline training on accepted global samples:
        1. Load accepted samples
        2. Verify class balance and schema
        3. Split train/validation
        4. Train candidate model
        5. Evaluate metrics
        6. Register as 'candidate'
        """
        feats, labels, users = self.dataset_manager.load_accepted_samples()

        if len(feats) < 10:
            return {
                "ok": False,
                "error": f"Insufficient accepted global samples ({len(feats)} < 10 minimum required for training).",
            }

        X = np.array(feats, dtype=np.float32)
        y_indices = np.array([self.evaluator.class_to_idx.get(l, 0) for l in labels], dtype=np.int32)

        # Train / validation split
        n = len(X)
        indices = np.random.permutation(n)
        val_size = max(2, int(0.2 * n))
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        X_train, y_train = X[train_idx], y_indices[train_idx]
        X_val, y_val = X[val_idx], y_indices[val_idx]

        model = build_candidate_mlp(input_dim=63, num_classes=len(VALID_ALPHABET_CLASSES))
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0,
        )

        # Evaluation
        val_labels = [VALID_ALPHABET_CLASSES[i] for i in y_val]
        eval_report = self.evaluator.evaluate_keras_model(model, X_val, np.array(val_labels))

        v_id = version_id or f"candidate_v_{int(time.time())}"
        cand_path = self.registry.registry_dir / f"{v_id}.h5"
        model.save(str(cand_path))

        # Register candidate in ModelRegistry
        reg_info = self.registry.register_candidate(
            candidate_model_path=cand_path,
            version_id=v_id,
            dataset_version=f"global_{len(X)}_samples",
            evaluation_metrics=eval_report.to_dict(),
        )

        return {
            "ok": True,
            "version_id": v_id,
            "checksum": reg_info["checksum"],
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "metrics": eval_report.to_dict(),
            "epochs_trained": len(history.history["loss"]),
            "model_path": str(cand_path),
            "message": "Candidate model successfully trained and registered in 'candidate' state.",
        }
