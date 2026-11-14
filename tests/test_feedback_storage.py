"""
Tests for Persistent Feedback Storage & SQLite Concurrency Configuration.
========================================================================
Validates:
- FEEDBACK_DATA_DIR environment override resolution vs default fallback.
- Directory auto-creation on the configured path.
- SQLite initialization on configured persistent storage mount.
- SQLite Write-Ahead Logging (WAL) mode active.
- SQLite PRAGMA synchronous = NORMAL and busy_timeout = 15000.
- Safe concurrent read/write transactions across multiple worker threads.
- SQLite database reopening and data persistence across simulated restarts.
- Generation and handling of SQLite WAL / SHM files during transactions.
- Reliable connection closure upon context exit (no dangling descriptors).
- Failure fast on missing or unwritable configured mount in production.
- Preservation of historical data across connection cycles.
"""

import importlib
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.feedback_learning import config
from src.feedback_learning.sample_collector import SampleCollector


class TestFeedbackStorageConfiguration(unittest.TestCase):
    """Hermetic tests for feedback storage path resolution, permissions and SQLite engine settings."""

    def setUp(self):
        self.orig_env = os.environ.get("FEEDBACK_DATA_DIR")
        self.orig_prod_force = os.environ.get("ISL_FORCE_PRODUCTION_CHECK")
        self.temp_dir = tempfile.mkdtemp(prefix="isl_storage_test_")

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["FEEDBACK_DATA_DIR"] = self.orig_env
        else:
            os.environ.pop("FEEDBACK_DATA_DIR", None)

        if self.orig_prod_force is not None:
            os.environ["ISL_FORCE_PRODUCTION_CHECK"] = self.orig_prod_force
        else:
            os.environ.pop("ISL_FORCE_PRODUCTION_CHECK", None)

        importlib.reload(config)

        if os.path.exists(self.temp_dir):
            try:
                os.chmod(self.temp_dir, 0o755)
            except OSError:
                pass
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_feedback_dir_when_env_unset(self):
        """When FEEDBACK_DATA_DIR is unset, fallback to workspace data/feedback."""
        os.environ.pop("FEEDBACK_DATA_DIR", None)
        importlib.reload(config)
        expected_default = (config.WORKSPACE_ROOT / "data" / "feedback").resolve()
        self.assertEqual(config.FEEDBACK_DATA_DIR, expected_default)
        self.assertEqual(config.DB_PATH, expected_default / "feedback.db")
        self.assertEqual(config.LANDMARKS_STORAGE_DIR, expected_default / "landmarks")

    def test_feedback_data_dir_override(self):
        """When FEEDBACK_DATA_DIR is set, paths resolve to the overridden directory."""
        custom_path = Path(self.temp_dir) / "custom_render_disk"
        os.environ["FEEDBACK_DATA_DIR"] = str(custom_path)
        importlib.reload(config)

        self.assertEqual(config.FEEDBACK_DATA_DIR, custom_path.resolve())
        self.assertEqual(config.DB_PATH, (custom_path / "feedback.db").resolve())
        self.assertEqual(config.LANDMARKS_STORAGE_DIR, (custom_path / "landmarks").resolve())
        self.assertEqual(config.GLOBAL_DATASET_DIR, (custom_path / "global_dataset").resolve())
        self.assertEqual(config.REGISTRY_DIR, (custom_path / "model_registry").resolve())
        self.assertEqual(config.PERSONAL_STORES_DIR, (custom_path / "personal_stores").resolve())

    def test_directory_auto_creation(self):
        """ensure_feedback_directories safely creates mount base and subdirectories."""
        mount_dir = Path(self.temp_dir) / "nested" / "disk_mount"
        self.assertFalse(mount_dir.exists())

        config.ensure_feedback_directories(mount_dir)
        self.assertTrue(mount_dir.exists())
        self.assertTrue((mount_dir / "landmarks").exists())
        self.assertTrue((mount_dir / "global_dataset").exists())
        self.assertTrue((mount_dir / "model_registry").exists())
        self.assertTrue((mount_dir / "personal_stores").exists())

    def test_sqlite_wal_mode_and_pragmas(self):
        """SampleCollector connection enforces WAL journal mode, synchronous NORMAL, and 15s timeout."""
        db_path = Path(self.temp_dir) / "wal_test.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        collector = SampleCollector(db_path=db_path, storage_dir=storage_dir)

        with collector._get_connection() as conn:
            cur = conn.cursor()

            # Check journal_mode
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            self.assertEqual(journal_mode.lower(), "wal")

            # Check synchronous (1 corresponds to NORMAL in SQLite)
            cur.execute("PRAGMA synchronous;")
            sync_mode = cur.fetchone()[0]
            self.assertEqual(int(sync_mode), 1)

            # Check busy_timeout
            cur.execute("PRAGMA busy_timeout;")
            busy_timeout = cur.fetchone()[0]
            self.assertEqual(int(busy_timeout), 15000)

    def test_database_persistence_across_restart(self):
        """Data persists in SQLite database when connection is closed and reopened."""
        db_path = Path(self.temp_dir) / "persistence.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        # Session 1: Create user and record an event
        collector1 = SampleCollector(db_path=db_path, storage_dir=storage_dir)
        user1 = collector1.get_or_create_user("persist_user_1")
        self.assertEqual(user1["user_id"], "persist_user_1")

        # Simulate restart by instantiating a completely new SampleCollector on same DB
        del collector1

        collector2 = SampleCollector(db_path=db_path, storage_dir=storage_dir)
        user_after = collector2.get_or_create_user("persist_user_1")
        self.assertIsNotNone(user_after)
        self.assertEqual(user_after["user_id"], "persist_user_1")

    def test_wal_files_created_on_disk(self):
        """WAL mode writes produce database files on the configured storage mount."""
        db_path = Path(self.temp_dir) / "wal_files.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        collector = SampleCollector(db_path=db_path, storage_dir=storage_dir)

        # Execute write transactions
        for i in range(10):
            collector.get_or_create_user(f"wal_user_{i}")

        # In WAL mode, the main database file exists on disk
        self.assertTrue(db_path.exists())
        self.assertGreater(db_path.stat().st_size, 0)

        # Open connection and hold transaction to assert -wal and -shm files exist on mount
        conn = collector._get_connection()
        conn.execute("INSERT INTO users (user_id, personalization_enabled, global_contribution_enabled, consent_version, created_at, updated_at) VALUES ('wal_active_user', 1, 0, '1.0', 1.0, 1.0);")
        wal_file = Path(str(db_path) + "-wal")
        shm_file = Path(str(db_path) + "-shm")
        self.assertTrue(wal_file.exists() or shm_file.exists())
        conn.commit()

        # Perform checkpoint to verify WAL handling
        cur = conn.cursor()
        cur.execute("PRAGMA wal_checkpoint(PASSIVE);")
        result = cur.fetchone()
        self.assertIsNotNone(result)
        conn.close()

    def test_concurrent_read_write_wal_safety(self):
        """Multiple threads can concurrently write and read without locking collisions."""
        db_path = Path(self.temp_dir) / "concurrency.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        collector = SampleCollector(db_path=db_path, storage_dir=storage_dir)
        errors = []
        num_threads = 6
        ops_per_thread = 15

        def worker_task(thread_id: int):
            try:
                for j in range(ops_per_thread):
                    uid = f"worker_{thread_id}_user_{j}"
                    # Write/Read operation
                    u = collector.get_or_create_user(uid)
                    if u is None or u["user_id"] != uid:
                        errors.append(f"Mismatch in thread {thread_id} op {j}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        self.assertEqual(errors, [], f"Encountered concurrency errors: {errors}")

    def test_connection_closes_reliably_after_context_exit(self):
        """ManagedConnection automatically and reliably closes connection after context exit."""
        db_path = Path(self.temp_dir) / "managed_conn.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        collector = SampleCollector(db_path=db_path, storage_dir=storage_dir)

        captured_conn = None
        with collector._get_connection() as conn:
            captured_conn = conn
            conn.execute("SELECT 1;")

        # Connection should be closed immediately upon exiting context manager
        self.assertIsNotNone(captured_conn)
        with self.assertRaises(sqlite3.ProgrammingError):
            captured_conn.execute("SELECT 1;")

    def test_unwritable_feedback_directory_fails_fast(self):
        """ensure_feedback_directories fails fast with RuntimeError when mount is read-only."""
        mount_dir = Path(self.temp_dir) / "readonly_mount"
        mount_dir.mkdir(parents=True, exist_ok=True)
        config.ensure_feedback_directories(mount_dir)

        # Lock directory permissions to read-only
        os.chmod(mount_dir, 0o555)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                config.ensure_feedback_directories(mount_dir)
            self.assertIn("CRITICAL STORAGE ERROR", str(ctx.exception))
            # Verify no sensitive full host paths leaked
            self.assertNotIn(str(mount_dir), str(ctx.exception))
        finally:
            os.chmod(mount_dir, 0o755)

    def test_missing_uncreatable_mount_fails_fast(self):
        """ensure_feedback_directories fails fast when configured path cannot be created."""
        impossible_path = Path("/proc/nonexistent_root_mount/isl_storage")
        with self.assertRaises(RuntimeError) as ctx:
            config.ensure_feedback_directories(impossible_path)
        self.assertIn("CRITICAL STORAGE ERROR", str(ctx.exception))
        self.assertNotIn(str(impossible_path), str(ctx.exception))

    def test_production_startup_failure_when_storage_unwritable(self):
        """In production, unwritable storage raises critical RuntimeError immediately."""
        os.environ["ISL_FORCE_PRODUCTION_CHECK"] = "1"
        self.assertTrue(config.is_production_environment())

        mount_dir = Path(self.temp_dir) / "prod_readonly_mount"
        mount_dir.mkdir(parents=True, exist_ok=True)
        config.ensure_feedback_directories(mount_dir)

        os.chmod(mount_dir, 0o555)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                config.ensure_feedback_directories(mount_dir)
            self.assertIn("CRITICAL STORAGE ERROR", str(ctx.exception))
        finally:
            os.chmod(mount_dir, 0o755)

    def test_existing_data_preservation_across_connection_cycles(self):
        """Repeated connection open and close cycles preserve all user data intact."""
        db_path = Path(self.temp_dir) / "data_preserve.db"
        storage_dir = Path(self.temp_dir) / "landmarks"

        collector = SampleCollector(db_path=db_path, storage_dir=storage_dir)

        # Write 5 users across 5 separate connection contexts
        for i in range(5):
            collector.get_or_create_user(f"preserve_user_{i}")

        # Verify all 5 exist in a fresh connection
        with collector._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users;")
            count = cur.fetchone()[0]
            self.assertEqual(count, 5)


if __name__ == "__main__":
    unittest.main()
