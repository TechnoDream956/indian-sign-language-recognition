"""
Sample Collector and Persistent Storage Manager.
================================================
Handles structured persistence for prediction events, corrections,
confirmations, landmark vectors, and global contribution references.
Uses SQLite for indexing/metadata and compressed numpy storage for vectors.
"""

import os
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    DB_PATH,
    FEEDBACK_DATA_DIR,
    LANDMARKS_STORAGE_DIR,
    GLOBAL_DATASET_DIR,
    FEATURE_SCHEMA_SINGLE_HAND,
    GLOBAL_CONTRIBUTION_DEFAULT_ENABLED,
    CURRENT_CONSENT_VERSION,
)
from .quality_filter import QualityFilter, QualityAssessment
from .prediction_snapshot import PredictionSnapshot


class ManagedConnection(sqlite3.Connection):
    """
    SQLite connection subclass that ensures connections are closed upon exiting a with-block.
    Preserves transaction commit on clean exit and rollback on exception.
    """

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


class SampleCollector:
    """Manages persistent database operations and landmark binary storage."""

    def __init__(self, db_path: Path = DB_PATH, storage_dir: Path = LANDMARKS_STORAGE_DIR):
        self.db_path = Path(db_path)
        self.storage_dir = Path(storage_dir)
        self._lock = threading.RLock()
        self.quality_filter = QualityFilter()

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, factory=ManagedConnection)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 15000;")
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Users
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        personalization_enabled INTEGER NOT NULL DEFAULT 1,
                        global_contribution_enabled INTEGER NOT NULL DEFAULT 0,
                        consent_version TEXT NOT NULL DEFAULT '1.0',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)

                # PredictionEvents
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_events (
                        prediction_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        predicted_label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        model_version TEXT NOT NULL,
                        model_checksum TEXT NOT NULL,
                        feature_schema TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        metadata_json TEXT
                    )
                """)

                # CorrectionEvents
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS correction_events (
                        correction_id TEXT PRIMARY KEY,
                        prediction_id TEXT NOT NULL,
                        actual_label TEXT,
                        user_id TEXT NOT NULL,
                        feedback_type TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        quality_status TEXT NOT NULL,
                        quality_score REAL NOT NULL,
                        FOREIGN KEY (prediction_id) REFERENCES prediction_events(prediction_id)
                    )
                """)

                # LandmarkSamples
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS landmark_samples (
                        sample_id TEXT PRIMARY KEY,
                        correction_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        feature_schema TEXT NOT NULL,
                        sequence_length INTEGER NOT NULL,
                        feature_file TEXT NOT NULL,
                        quality_score REAL NOT NULL,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (correction_id) REFERENCES correction_events(correction_id)
                    )
                """)

                # GlobalContributions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS global_contributions (
                        contribution_id TEXT PRIMARY KEY,
                        sample_id TEXT NOT NULL UNIQUE,
                        user_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        consent_version TEXT NOT NULL,
                        contribution_status TEXT NOT NULL,
                        review_notes TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        FOREIGN KEY (sample_id) REFERENCES landmark_samples(sample_id)
                    )
                """)

                # ModelVersions (Registry table)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        version_id TEXT PRIMARY KEY,
                        model_checksum TEXT NOT NULL,
                        training_dataset_version TEXT NOT NULL,
                        evaluation_metrics TEXT,
                        deployment_status TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                """)

                # Create helpful indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_corr_user ON correction_events(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_user ON landmark_samples(user_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON landmark_samples(label)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_global_status ON global_contributions(contribution_status)")
                conn.commit()

    # ------------------------------------------------------------------ #
    # User Management & Isolation                                        #
    # ------------------------------------------------------------------ #

    def get_or_create_user(self, user_id: str) -> Dict[str, Any]:
        """Ensure user profile exists with default safe preferences."""
        clean_uid = (user_id or "anonymous_local").strip()
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM users WHERE user_id = ?", (clean_uid,))
                row = cur.fetchone()
                if row:
                    return dict(row)
                cur.execute(
                    """
                    INSERT INTO users (user_id, personalization_enabled, global_contribution_enabled, consent_version, created_at, updated_at)
                    VALUES (?, 1, 0, ?, ?, ?)
                    """,
                    (clean_uid, CURRENT_CONSENT_VERSION, now, now),
                )
                conn.commit()
                return {
                    "user_id": clean_uid,
                    "personalization_enabled": 1,
                    "global_contribution_enabled": 0,
                    "consent_version": CURRENT_CONSENT_VERSION,
                    "created_at": now,
                    "updated_at": now,
                }

    def update_user_preferences(
        self,
        user_id: str,
        personalization_enabled: Optional[bool] = None,
        global_contribution_enabled: Optional[bool] = None,
        consent_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update user preferences."""
        user = self.get_or_create_user(user_id)
        now = time.time()
        new_pers = (
            int(personalization_enabled)
            if personalization_enabled is not None
            else user["personalization_enabled"]
        )
        new_glob = (
            int(global_contribution_enabled)
            if global_contribution_enabled is not None
            else user["global_contribution_enabled"]
        )
        new_consent = consent_version or user["consent_version"]

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE users
                    SET personalization_enabled = ?, global_contribution_enabled = ?, consent_version = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (new_pers, new_glob, new_consent, now, user["user_id"]),
                )
                conn.commit()

        return self.get_or_create_user(user_id)

    # ------------------------------------------------------------------ #
    # Prediction Events                                                  #
    # ------------------------------------------------------------------ #

    def save_prediction_event(self, snapshot: PredictionSnapshot) -> None:
        """Record prediction event asynchronously or directly."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prediction_events (
                        prediction_id, user_id, predicted_label, confidence,
                        model_version, model_checksum, feature_schema, timestamp, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.prediction_id,
                        snapshot.user_id,
                        snapshot.predicted_alphabet,
                        snapshot.confidence_score,
                        snapshot.production_model_version,
                        snapshot.model_checksum,
                        snapshot.feature_schema,
                        snapshot.timestamp,
                        json.dumps(snapshot.metadata),
                    ),
                )
                conn.commit()

    def get_prediction_event(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored prediction event by prediction_id."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM prediction_events WHERE prediction_id = ?", (prediction_id,))
                row = cur.fetchone()
                return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Confirmation Feedback ("Yes" was correct)                          #
    # ------------------------------------------------------------------ #

    def save_confirmation(
        self,
        prediction_id: str,
        user_id: str,
        snapshot: Optional[PredictionSnapshot] = None,
    ) -> Dict[str, Any]:
        """Record lightweight confirmation without storing landmark vector."""
        now = time.time()
        corr_id = f"conf_{uuid.uuid4().hex[:12]}"

        # If snapshot provided and prediction event doesn't exist yet, save it
        if snapshot is not None:
            self.save_prediction_event(snapshot)

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO correction_events (
                        correction_id, prediction_id, actual_label, user_id, feedback_type,
                        created_at, quality_status, quality_score
                    )
                    VALUES (?, ?, ?, ?, 'confirmation', ?, 'confirmed', 1.0)
                    """,
                    (
                        corr_id,
                        prediction_id,
                        snapshot.predicted_alphabet if snapshot else None,
                        user_id,
                        now,
                    ),
                )
                conn.commit()

        return {
            "ok": True,
            "confirmation_id": corr_id,
            "prediction_id": prediction_id,
            "feedback_type": "confirmation",
            "message": "Confirmation recorded.",
        }

    # ------------------------------------------------------------------ #
    # Correction Feedback ("No" was incorrect, provide actual sign)      #
    # ------------------------------------------------------------------ #

    def save_correction(
        self,
        snapshot: PredictionSnapshot,
        actual_label: str,
        user_id: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Record correction, validate quality, save landmark sample, and queue global contribution if opted-in.
        """
        now = time.time()
        actual_label = (actual_label or "").strip().upper()

        # 1. Label validation
        val_ok, msg = self.quality_filter.validate_label(actual_label)
        if not val_ok:
            return False, {"ok": False, "error": msg}

        # 2. Retrieve user preferences
        user_pref = self.get_or_create_user(user_id)

        # 3. Assess landmark sample quality
        existing_feats = self.get_user_landmark_arrays(user_id, label=actual_label)
        assessment: QualityAssessment = self.quality_filter.assess_landmarks(
            features=snapshot.feature_vector,
            schema=snapshot.feature_schema,
            raw_landmarks=snapshot.raw_landmarks,
            tracking_quality=snapshot.tracking_quality,
            existing_samples=existing_feats,
        )

        corr_id = f"corr_{uuid.uuid4().hex[:12]}"
        sample_id = f"smp_{uuid.uuid4().hex[:12]}"

        # First persist prediction event
        self.save_prediction_event(snapshot)

        # Quality status
        quality_status = "valid" if assessment.is_valid else ("duplicate" if assessment.is_duplicate else "rejected")

        # Save binary feature array
        sample_rel_path = f"{user_id}/{sample_id}.npz"
        sample_abs_path = self.storage_dir / sample_rel_path
        sample_abs_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(sample_abs_path),
            features=snapshot.feature_vector,
            label=actual_label,
            schema=snapshot.feature_schema,
            raw_landmarks=json.dumps(snapshot.raw_landmarks),
        )

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Insert correction event
                cur.execute(
                    """
                    INSERT INTO correction_events (
                        correction_id, prediction_id, actual_label, user_id, feedback_type,
                        created_at, quality_status, quality_score
                    )
                    VALUES (?, ?, ?, ?, 'correction', ?, ?, ?)
                    """,
                    (
                        corr_id,
                        snapshot.prediction_id,
                        actual_label,
                        user_id,
                        now,
                        quality_status,
                        assessment.quality_score,
                    ),
                )

                # Insert landmark sample
                cur.execute(
                    """
                    INSERT INTO landmark_samples (
                        sample_id, correction_id, user_id, label, feature_schema,
                        sequence_length, feature_file, quality_score, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample_id,
                        corr_id,
                        user_id,
                        actual_label,
                        snapshot.feature_schema,
                        1,
                        sample_rel_path,
                        assessment.quality_score,
                        now,
                    ),
                )

                # If global contribution is explicitly enabled AND sample is valid, add to global_contributions table
                global_queued = False
                if user_pref["global_contribution_enabled"] and assessment.is_valid:
                    contrib_id = f"gc_{uuid.uuid4().hex[:12]}"
                    cur.execute(
                        """
                        INSERT INTO global_contributions (
                            contribution_id, sample_id, user_id, label, consent_version,
                            contribution_status, review_notes, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, 'pending', 'Submitted via explicit opt-in', ?, ?)
                        """,
                        (
                            contrib_id,
                            sample_id,
                            user_id,
                            actual_label,
                            user_pref["consent_version"],
                            now,
                            now,
                        ),
                    )
                    global_queued = True

                conn.commit()

        return True, {
            "ok": True,
            "correction_id": corr_id,
            "sample_id": sample_id,
            "actual_label": actual_label,
            "original_label": snapshot.predicted_alphabet,
            "is_valid": assessment.is_valid,
            "quality_status": quality_status,
            "quality_score": assessment.quality_score,
            "global_contribution_queued": global_queued,
            "message": "Correction saved. Thank you for helping improve recognition.",
        }

    # ------------------------------------------------------------------ #
    # Retrieval Methods                                                  #
    # ------------------------------------------------------------------ #

    def get_user_landmark_arrays(self, user_id: str, label: Optional[str] = None) -> List[np.ndarray]:
        """Load stored feature vectors for a user."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                if label:
                    cur.execute(
                        "SELECT feature_file FROM landmark_samples WHERE user_id = ? AND label = ?",
                        (user_id, label),
                    )
                else:
                    cur.execute(
                        "SELECT feature_file FROM landmark_samples WHERE user_id = ?",
                        (user_id,),
                    )
                rows = cur.fetchall()

        arrays = []
        for r in rows:
            p = self.storage_dir / r["feature_file"]
            if p.exists():
                try:
                    data = np.load(str(p))
                    arrays.append(data["features"])
                except Exception:
                    pass
        return arrays

    def get_feedback_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve feedback history for a user."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT c.correction_id, c.prediction_id, c.actual_label, c.feedback_type,
                           c.created_at, c.quality_status, c.quality_score,
                           p.predicted_label, p.confidence, p.model_version
                    FROM correction_events c
                    LEFT JOIN prediction_events p ON c.prediction_id = p.prediction_id
                    WHERE c.user_id = ?
                    ORDER BY c.created_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def get_feedback_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate learning review and sample statistics."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # Filter by user if provided
                user_clause = "WHERE user_id = ?" if user_id else ""
                params = (user_id,) if user_id else ()

                # Total confirmations
                cur.execute(
                    f"SELECT COUNT(*) FROM correction_events {user_clause} {'AND' if user_clause else 'WHERE'} feedback_type = 'confirmation'",
                    params,
                )
                total_confirmations = cur.fetchone()[0]

                # Total corrections
                cur.execute(
                    f"SELECT COUNT(*) FROM correction_events {user_clause} {'AND' if user_clause else 'WHERE'} feedback_type = 'correction'",
                    params,
                )
                total_corrections = cur.fetchone()[0]

                # Total stored landmark samples
                cur.execute(f"SELECT COUNT(*) FROM landmark_samples {user_clause}", params)
                personal_samples = cur.fetchone()[0]

                # Global contribution metrics
                cur.execute("SELECT COUNT(*) FROM global_contributions")
                global_total = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM global_contributions WHERE contribution_status = 'pending'")
                global_pending = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM global_contributions WHERE contribution_status = 'accepted'")
                global_accepted = cur.fetchone()[0]

                # Total predictions reviewed
                total_reviewed = total_confirmations + total_corrections

                user_pref = self.get_or_create_user(user_id) if user_id else None

        return {
            "total_reviewed": total_reviewed,
            "correct_confirmations": total_confirmations,
            "incorrect_corrections": total_corrections,
            "personal_correction_samples": personal_samples,
            "global_contributions_total": global_total,
            "global_contributions_pending": global_pending,
            "global_contributions_accepted": global_accepted,
            "personalization_enabled": bool(user_pref["personalization_enabled"]) if user_pref else True,
            "global_contribution_enabled": bool(user_pref["global_contribution_enabled"]) if user_pref else False,
        }

    # ------------------------------------------------------------------ #
    # Personal Data Reset & Privacy                                      #
    # ------------------------------------------------------------------ #

    def reset_personal_data(self, user_id: str) -> Dict[str, Any]:
        """
        Completely delete user's personal corrections, samples, and prediction events,
        revoke global contribution consent, and atomically withdraw all pending and accepted
        global contributions for this user.
        """
        clean_uid = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_")).strip()[:64]
        if not clean_uid:
            return {
                "ok": True,
                "user_id": user_id,
                "removed_files": 0,
                "withdrawn_contributions_count": 0,
                "message": "Invalid user identifier or no data to erase.",
            }

        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.cursor()

                # 1. Get feature files to remove
                cur.execute("SELECT feature_file FROM landmark_samples WHERE user_id = ?", (clean_uid,))
                files_to_remove = [r["feature_file"] for r in cur.fetchall()]

                # 2. Delete personal landmark_samples, correction_events, prediction_events
                cur.execute("DELETE FROM landmark_samples WHERE user_id = ?", (clean_uid,))
                cur.execute("DELETE FROM correction_events WHERE user_id = ?", (clean_uid,))
                cur.execute("DELETE FROM prediction_events WHERE user_id = ?", (clean_uid,))

                # 3. Revoke global contribution consent in users table
                cur.execute(
                    "UPDATE users SET global_contribution_enabled = 0, updated_at = ? WHERE user_id = ?",
                    (now, clean_uid),
                )

                # 4. Atomically withdraw all pending and accepted global contributions
                cur.execute(
                    """
                    UPDATE global_contributions
                    SET contribution_status = 'withdrawn',
                        review_notes = 'Withdrawn via complete personal data reset',
                        updated_at = ?
                    WHERE user_id = ? AND contribution_status IN ('pending', 'accepted')
                    """,
                    (now, clean_uid),
                )
                withdrawn_count = cur.rowcount
                conn.commit()

        # Remove physical files from storage
        count_removed = 0
        for f in files_to_remove:
            p = self.storage_dir / f
            if p.exists():
                try:
                    p.unlink()
                    count_removed += 1
                except Exception:
                    pass

        # Remove user folder if empty
        user_folder = self.storage_dir / clean_uid
        if user_folder.exists():
            try:
                user_folder.rmdir()
            except Exception:
                pass

        return {
            "ok": True,
            "user_id": clean_uid,
            "removed_files": count_removed,
            "withdrawn_contributions_count": withdrawn_count,
            "message": "Personal learning data successfully erased and contributions withdrawn.",
        }
