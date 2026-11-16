"""
Curated Global Dataset Management.
==================================
Manages the lifecycle of contributed global samples: pending, accepted,
rejected, and withdrawn. Prepares leak-free, user-partitioned training datasets.
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    GLOBAL_DATASET_DIR,
    FEATURE_SCHEMA_SINGLE_HAND,
    VALID_ALPHABET_CLASSES,
)
from .sample_collector import SampleCollector
from .quality_filter import QualityFilter, QualityAssessment


class GlobalDatasetManager:
    """Oversees moderation, validation, curation, and export of global samples."""

    def __init__(
        self,
        collector: Optional[SampleCollector] = None,
        export_dir: Path = GLOBAL_DATASET_DIR,
    ):
        self.collector = collector or SampleCollector()
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.quality_filter = QualityFilter()

    def get_contributions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List contributions with optional status filtering."""
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                if status:
                    cur.execute(
                        """
                        SELECT gc.*, ls.feature_file, ls.quality_score, ls.feature_schema
                        FROM global_contributions gc
                        JOIN landmark_samples ls ON gc.sample_id = ls.sample_id
                        WHERE gc.contribution_status = ?
                        ORDER BY gc.created_at DESC LIMIT ?
                        """,
                        (status, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT gc.*, ls.feature_file, ls.quality_score, ls.feature_schema
                        FROM global_contributions gc
                        JOIN landmark_samples ls ON gc.sample_id = ls.sample_id
                        ORDER BY gc.created_at DESC LIMIT ?
                        """,
                        (limit,),
                    )
                return [dict(r) for r in cur.fetchall()]

    def moderate_contribution(
        self,
        contribution_id: str,
        new_status: str,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Manually or programmatically set moderation status ('accepted', 'rejected', etc.)."""
        valid_statuses = {"pending", "accepted", "rejected", "withdrawn"}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {valid_statuses}")

        now = time.time()
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE global_contributions
                    SET contribution_status = ?, review_notes = ?, updated_at = ?
                    WHERE contribution_id = ?
                    """,
                    (new_status, notes, now, contribution_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return {"ok": False, "error": f"Contribution '{contribution_id}' not found"}

        return {"ok": True, "contribution_id": contribution_id, "status": new_status, "notes": notes}

    def review_pending_contributions(self) -> Dict[str, int]:
        """
        Automated moderation pass for pending contributions:
        Validates schema, quality score, duplicate detection, and auto-promotes eligible samples.
        """
        pending = self.get_contributions(status="pending", limit=500)
        accepted_feats = self.load_accepted_samples()[0]  # List of existing accepted feature vectors

        accepted_count = 0
        rejected_count = 0

        for item in pending:
            cid = item["contribution_id"]
            lbl = item["label"]
            fpath = self.collector.storage_dir / item["feature_file"]

            if not fpath.exists():
                self.moderate_contribution(cid, "rejected", "Feature file missing on disk")
                rejected_count += 1
                continue

            try:
                data = np.load(str(fpath))
                feats = data["features"].astype(np.float32)
            except Exception as e:
                self.moderate_contribution(cid, "rejected", f"Corrupt feature file: {e}")
                rejected_count += 1
                continue

            # Check quality
            assessment = self.quality_filter.assess_landmarks(
                features=feats,
                schema=item["feature_schema"],
                existing_samples=accepted_feats,
            )

            if assessment.is_valid:
                self.moderate_contribution(cid, "accepted", "Automated quality pass: verified")
                accepted_feats.append(feats)
                accepted_count += 1
            else:
                self.moderate_contribution(cid, "rejected", f"Quality check failed: {'; '.join(assessment.reasons)}")
                rejected_count += 1

        return {
            "processed": len(pending),
            "accepted": accepted_count,
            "rejected": rejected_count,
        }

    def load_accepted_samples(self) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """
        Load all currently accepted samples for dataset curation or training.
        Returns (features_list, labels_list, user_ids_list).
        """
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT gc.user_id, gc.label, ls.feature_file, ls.feature_schema
                    FROM global_contributions gc
                    JOIN landmark_samples ls ON gc.sample_id = ls.sample_id
                    WHERE gc.contribution_status = 'accepted'
                    ORDER BY gc.created_at ASC
                    """
                )
                rows = cur.fetchall()

        feats_list = []
        labels_list = []
        user_ids_list = []

        for r in rows:
            fpath = self.collector.storage_dir / r["feature_file"]
            if fpath.exists():
                try:
                    data = np.load(str(fpath))
                    feats = data["features"].astype(np.float32)
                    feats_list.append(feats)
                    labels_list.append(r["label"])
                    user_ids_list.append(r["user_id"])
                except Exception:
                    pass

        return feats_list, labels_list, user_ids_list

    def export_dataset(self, dataset_version: str = "v1") -> Dict[str, Any]:
        """
        Export accepted global samples into a versioned training archive with user-partitioned splits.
        """
        feats_list, labels_list, user_ids_list = self.load_accepted_samples()
        if not feats_list:
            return {"ok": False, "error": "No accepted samples available for export", "sample_count": 0}

        X = np.array(feats_list, dtype=np.float32)
        y = np.array(labels_list)
        users = np.array(user_ids_list)

        # Split by user if multiple users exist to prevent leakage
        unique_users = np.unique(users)
        if len(unique_users) > 1:
            val_user = unique_users[-1]
            val_mask = (users == val_user)
            train_mask = ~val_mask
        else:
            # Single user fallback: stratify/random split
            n = len(X)
            val_size = max(1, int(0.2 * n))
            val_mask = np.zeros(n, dtype=bool)
            val_mask[-val_size:] = True
            train_mask = ~val_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        out_path = self.export_dir / f"global_dataset_{dataset_version}.npz"
        np.savez_compressed(
            str(out_path),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            classes=VALID_ALPHABET_CLASSES,
        )

        return {
            "ok": True,
            "dataset_version": dataset_version,
            "path": str(out_path),
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "unique_users": len(unique_users),
            "classes_represented": len(np.unique(y)),
        }
