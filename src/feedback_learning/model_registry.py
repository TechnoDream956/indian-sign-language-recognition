"""
Model Registry and Version Management.
======================================
Maintains explicit deployment states (production, candidate, rejected, archived)
for recognition models, with cryptographic checksums, metrics, and rollback guarantees.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    REGISTRY_DIR,
    PRODUCTION_MODEL_PATH,
    PRODUCTION_MODEL_CHECKSUM,
    PRODUCTION_MODEL_VERSION,
    DB_PATH,
)
from .sample_collector import SampleCollector


def compute_file_sha256(path: Path) -> str:
    """Compute cryptographic SHA256 checksum of model file."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    """
    Manages candidate vs production model artifacts and ensures the production
    model is never modified without explicit manual promotion.
    """

    def __init__(self, registry_dir: Path = REGISTRY_DIR, collector: Optional[SampleCollector] = None):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.collector = collector or SampleCollector()

        self._ensure_production_registered()

    def _ensure_production_registered(self) -> None:
        """Register the baseline production model on first initialization."""
        checksum = compute_file_sha256(PRODUCTION_MODEL_PATH)
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM model_versions WHERE version_id = ?", (PRODUCTION_MODEL_VERSION,))
                if not cur.fetchone():
                    cur.execute(
                        """
                        INSERT INTO model_versions (
                            version_id, model_checksum, training_dataset_version,
                            evaluation_metrics, deployment_status, created_at
                        )
                        VALUES (?, ?, 'baseline_v1', ?, 'production', ?)
                        """,
                        (
                            PRODUCTION_MODEL_VERSION,
                            checksum or PRODUCTION_MODEL_CHECKSUM,
                            json.dumps({"accuracy": 0.965, "classes": 26, "note": "Baseline production model"}),
                            time.time(),
                        ),
                    )
                    conn.commit()

    def register_candidate(
        self,
        candidate_model_path: Path,
        version_id: str,
        dataset_version: str,
        evaluation_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Register a newly trained offline candidate model."""
        p = Path(candidate_model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        checksum = compute_file_sha256(p)
        now = time.time()

        # Copy to registry dir
        dest = self.registry_dir / f"{version_id}.h5"
        shutil.copy2(str(p), str(dest))

        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO model_versions (
                        version_id, model_checksum, training_dataset_version,
                        evaluation_metrics, deployment_status, created_at
                    )
                    VALUES (?, ?, ?, ?, 'candidate', ?)
                    """,
                    (
                        version_id,
                        checksum,
                        dataset_version,
                        json.dumps(evaluation_metrics),
                        now,
                    ),
                )
                conn.commit()

        return {
            "ok": True,
            "version_id": version_id,
            "checksum": checksum,
            "status": "candidate",
            "path": str(dest),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in the registry."""
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
                models = []
                for r in cur.fetchall():
                    item = dict(r)
                    if item.get("evaluation_metrics"):
                        try:
                            item["evaluation_metrics"] = json.loads(item["evaluation_metrics"])
                        except Exception:
                            pass
                    models.append(item)
                return models

    def get_model(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get model details by version_id."""
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM model_versions WHERE version_id = ?", (version_id,))
                row = cur.fetchone()
                if not row:
                    return None
                item = dict(row)
                if item.get("evaluation_metrics"):
                    try:
                        item["evaluation_metrics"] = json.loads(item["evaluation_metrics"])
                    except Exception:
                        pass
                return item

    def update_model_status(self, version_id: str, new_status: str) -> Dict[str, Any]:
        """Update deployment status ('candidate', 'rejected', 'archived')."""
        valid = {"candidate", "rejected", "archived"}
        if new_status not in valid:
            raise ValueError(f"Status '{new_status}' invalid. Use promote_candidate for 'production'.")

        with self.collector._lock:
            with self.collector._get_connection() as conn:
                conn.execute(
                    "UPDATE model_versions SET deployment_status = ? WHERE version_id = ?",
                    (new_status, version_id),
                )
                conn.commit()

        return {"ok": True, "version_id": version_id, "status": new_status}

    def promote_candidate_to_production(self, candidate_version_id: str, authorized: bool = False) -> Dict[str, Any]:
        """
        Explicit, authorized manual promotion of a candidate model.
        Archives previous production model and sets candidate as production.
        """
        if not authorized:
            return {
                "ok": False,
                "error": "Manual authorization required to promote candidate model. Production model weights are protected.",
            }

        candidate = self.get_model(candidate_version_id)
        if not candidate:
            return {"ok": False, "error": f"Candidate version '{candidate_version_id}' not found"}

        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                # Archive currently active production models
                cur.execute(
                    "UPDATE model_versions SET deployment_status = 'archived' WHERE deployment_status = 'production'"
                )
                # Set candidate to production
                cur.execute(
                    "UPDATE model_versions SET deployment_status = 'production' WHERE version_id = ?",
                    (candidate_version_id,),
                )
                conn.commit()

        return {
            "ok": True,
            "promoted_version_id": candidate_version_id,
            "status": "production",
            "message": f"Candidate model '{candidate_version_id}' promoted to production state in registry.",
        }

    def rollback_to_version(self, target_version_id: str, authorized: bool = False) -> Dict[str, Any]:
        """Rollback active production state to a previously archived version."""
        if not authorized:
            return {"ok": False, "error": "Manual authorization required for rollback."}

        target = self.get_model(target_version_id)
        if not target:
            return {"ok": False, "error": f"Target version '{target_version_id}' not found"}

        with self.collector._lock:
            with self.collector._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE model_versions SET deployment_status = 'archived' WHERE deployment_status = 'production'"
                )
                cur.execute(
                    "UPDATE model_versions SET deployment_status = 'production' WHERE version_id = ?",
                    (target_version_id,),
                )
                conn.commit()

        return {
            "ok": True,
            "active_production_version": target_version_id,
            "message": f"Successfully rolled back production state to '{target_version_id}'.",
        }
