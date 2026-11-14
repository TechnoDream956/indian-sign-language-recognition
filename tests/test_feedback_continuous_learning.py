"""
Hermetic Test Suite for Continuous Feedback Learning Subsystem.
==============================================================
Validates:
- Exact prediction snapshot association
- Landmark schema validation and corruption rejection
- Confirmation vs Correction handling
- Personal data isolation and conservative adaptation
- Explicit global opt-in default, consent enforcement, and withdrawal
- Curated global dataset and moderation lifecycle
- Candidate model evaluation and regression checks
- Model registry deployment states and rollback
- API v2 contracts and error handling
- Absolute protection and checksum preservation of production model weights
"""

import os
os.environ["ISL_CLOUD"] = "1"
os.environ["RENDER"] = "true"

import hashlib
import json
import shutil
import tempfile
import unittest
import uuid
import numpy as np

from src.feedback_learning.config import (
    PRODUCTION_MODEL_PATH,
    PRODUCTION_MODEL_CHECKSUM,
    CNN_MODEL_PATH,
    CNN_MODEL_CHECKSUM,
    DUAL_HAND_MODEL_PATH,
    DUAL_HAND_MODEL_CHECKSUM,
    FEATURE_SCHEMA_SINGLE_HAND,
    FEATURE_SCHEMA_DUAL_HAND,
    VALID_ALPHABET_CLASSES,
)
from src.feedback_learning.prediction_snapshot import PredictionSnapshot, SnapshotRingBuffer
from src.feedback_learning.quality_filter import QualityFilter
from src.feedback_learning.sample_collector import SampleCollector
from src.feedback_learning.personal_store import PersonalStore
from src.feedback_learning.personal_adapter import PersonalAdapter
from src.feedback_learning.global_contribution import GlobalContributionManager
from src.feedback_learning.global_dataset import GlobalDatasetManager
from src.feedback_learning.model_registry import ModelRegistry, compute_file_sha256
from src.feedback_learning.evaluator import ModelEvaluator, EvaluationReport
from src.feedback_learning.feedback_manager import FeedbackManager
from src.feedback_learning.privacy_manager import PrivacyManager
from src.feedback_learning.feedback_api import feedback_bp, get_feedback_manager
from flask import Flask


def compute_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_mock_landmarks(seed: int = 42) -> np.ndarray:
    """Generate realistic normalized single-hand 63D landmark vector."""
    np.random.seed(seed)
    coords = np.random.uniform(0.1, 0.9, (21, 3)).astype(np.float32)
    coords[0] = [0.0, 0.0, 0.0]  # Wrist anchor
    # Scale relative to middle MCP (point 9)
    scale = float(np.linalg.norm(coords[9])) + 1e-4
    normalized = coords / scale
    return normalized.flatten().astype(np.float32)


class TestPredictionSnapshotAssociation(unittest.TestCase):
    """Test prediction snapshot creation, exact association, and ring buffer TTL."""

    def test_snapshot_exact_attributes_and_serialization(self):
        feats = generate_mock_landmarks(10)
        snapshot = PredictionSnapshot(
            predicted_alphabet="B",
            confidence_score=0.88,
            feature_vector=feats,
            feature_schema=FEATURE_SCHEMA_SINGLE_HAND,
            metadata={"fusion_mode": "landmark_dominant"},
            user_id="user_test_1",
        )

        d = snapshot.to_dict()
        self.assertTrue(d["prediction_id"].startswith("pred_"))
        self.assertEqual(d["predicted_alphabet"], "B")
        self.assertEqual(d["confidence_score"], 0.88)
        self.assertEqual(d["feature_schema"], FEATURE_SCHEMA_SINGLE_HAND)
        self.assertEqual(d["user_id"], "user_test_1")
        self.assertEqual(len(d["feature_vector"]), 63)

        # Deserialization
        restored = PredictionSnapshot.from_dict(d)
        self.assertEqual(restored.prediction_id, snapshot.prediction_id)
        self.assertEqual(restored.predicted_alphabet, "B")
        np.testing.assert_array_almost_equal(restored.feature_vector, feats)

    def test_ring_buffer_bounds_and_lookup(self):
        buf = SnapshotRingBuffer(max_size=3, ttl_seconds=5.0)
        s1 = PredictionSnapshot("A", 0.9, generate_mock_landmarks(1))
        s2 = PredictionSnapshot("B", 0.85, generate_mock_landmarks(2))
        s3 = PredictionSnapshot("C", 0.75, generate_mock_landmarks(3))
        s4 = PredictionSnapshot("D", 0.95, generate_mock_landmarks(4))

        buf.add(s1)
        buf.add(s2)
        buf.add(s3)
        self.assertEqual(buf.count(), 3)
        self.assertIsNotNone(buf.get(s1.prediction_id))

        # Adding 4th drops oldest (s1)
        buf.add(s4)
        self.assertEqual(buf.count(), 3)
        self.assertIsNone(buf.get(s1.prediction_id))
        self.assertIsNotNone(buf.get(s4.prediction_id))
        self.assertEqual(buf.get_latest().prediction_id, s4.prediction_id)


class TestLandmarkQualityFilter(unittest.TestCase):
    """Validate dimensionality, numerical integrity, coordinate bounds, and duplicate detection."""

    def setUp(self):
        self.qf = QualityFilter()

    def test_valid_single_hand_sample(self):
        feats = generate_mock_landmarks(20)
        assessment = self.qf.assess_landmarks(feats, schema=FEATURE_SCHEMA_SINGLE_HAND)
        self.assertTrue(assessment.is_valid)
        self.assertGreaterEqual(assessment.quality_score, 0.60)
        self.assertFalse(assessment.is_duplicate)

    def test_reject_wrong_dimension(self):
        feats = np.zeros(50, dtype=np.float32)
        assessment = self.qf.assess_landmarks(feats, schema=FEATURE_SCHEMA_SINGLE_HAND)
        self.assertFalse(assessment.is_valid)
        self.assertIn("Expected shape (63,)", assessment.reasons[0])

    def test_reject_corrupted_nan_and_inf(self):
        feats = generate_mock_landmarks(21)
        feats[5] = np.nan
        assessment = self.qf.assess_landmarks(feats, schema=FEATURE_SCHEMA_SINGLE_HAND)
        self.assertFalse(assessment.is_valid)
        self.assertIn("NaN or Infinite", assessment.reasons[0])

        feats[5] = np.inf
        assessment2 = self.qf.assess_landmarks(feats, schema=FEATURE_SCHEMA_SINGLE_HAND)
        self.assertFalse(assessment2.is_valid)

    def test_reject_empty_or_all_zeros(self):
        feats = np.zeros(63, dtype=np.float32)
        assessment = self.qf.assess_landmarks(feats, schema=FEATURE_SCHEMA_SINGLE_HAND)
        self.assertFalse(assessment.is_valid)
        self.assertIn("empty or all zeros", assessment.reasons[0])

    def test_reject_near_duplicates(self):
        f1 = generate_mock_landmarks(30)
        f2 = f1 + 0.001  # very close (L2 distance < 0.015)
        assessment = self.qf.assess_landmarks(f2, existing_samples=[f1])
        self.assertTrue(assessment.is_duplicate)
        self.assertFalse(assessment.is_valid)

    def test_label_validation(self):
        valid, _ = self.qf.validate_label("A")
        self.assertTrue(valid)
        valid, _ = self.qf.validate_label("z")
        self.assertTrue(valid)

        invalid1, _ = self.qf.validate_label("1")
        self.assertFalse(invalid1)
        invalid2, _ = self.qf.validate_label("WORD")
        self.assertFalse(invalid2)
        invalid3, _ = self.qf.validate_label("")
        self.assertFalse(invalid3)


class TestFeedbackSubsystemHermetic(unittest.TestCase):
    """Test full feedback lifecycle in isolated temporary directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_feedback.db")
        self.storage_dir = os.path.join(self.temp_dir, "landmarks")
        self.collector = SampleCollector(db_path=self.db_path, storage_dir=self.storage_dir)
        self.manager = FeedbackManager(collector=self.collector)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_confirmation_feedback(self):
        """Confirming a prediction ('Yes') saves lightweight record without landmark array."""
        feats = generate_mock_landmarks(40)
        snapshot = self.manager.capture_prediction_snapshot(
            predicted_alphabet="A",
            confidence_score=0.91,
            feature_vector=feats,
            user_id="user_alice",
        )

        res = self.manager.confirm_prediction(snapshot.prediction_id, user_id="user_alice")
        self.assertTrue(res["ok"])
        self.assertEqual(res["feedback_type"], "confirmation")

        stats = self.manager.get_stats("user_alice")
        self.assertEqual(stats["correct_confirmations"], 1)
        self.assertEqual(stats["incorrect_corrections"], 0)
        self.assertEqual(stats["personal_correction_samples"], 0)  # No heavy sample saved

    def test_correction_feedback_and_snapshot_association(self):
        """Correcting a prediction ('No') saves correction against exact snapshot."""
        feats = generate_mock_landmarks(41)
        snapshot = self.manager.capture_prediction_snapshot(
            predicted_alphabet="M",
            confidence_score=0.62,
            feature_vector=feats,
            user_id="user_bob",
        )

        success, res = self.manager.correct_prediction(
            prediction_id=snapshot.prediction_id,
            actual_label="N",
            user_id="user_bob",
        )
        self.assertTrue(success)
        self.assertTrue(res["ok"])
        self.assertEqual(res["actual_label"], "N")
        self.assertEqual(res["original_label"], "M")
        self.assertTrue(res["is_valid"])

        # Sample should now be in personal store
        exemplars = self.manager.personal_store.get_user_exemplars("user_bob")
        self.assertIn("N", exemplars)
        self.assertEqual(len(exemplars["N"]), 1)

    def test_correction_fails_for_unknown_or_expired_snapshot(self):
        """Corrections for non-existent prediction IDs fail safely."""
        success, res = self.manager.correct_prediction(
            prediction_id="pred_nonexistent123",
            actual_label="A",
            user_id="user_bob",
        )
        self.assertFalse(success)
        self.assertIn("expired or not found", res["error"])

    def test_personal_data_isolation(self):
        """User Alice's corrections must never leak into User Bob's store."""
        f_alice = generate_mock_landmarks(50)
        s_alice = self.manager.capture_prediction_snapshot("A", 0.7, f_alice, user_id="alice")
        self.manager.correct_prediction(s_alice.prediction_id, "B", user_id="alice")

        bob_exemplars = self.manager.personal_store.get_user_exemplars("bob")
        self.assertEqual(len(bob_exemplars), 0)

        alice_exemplars = self.manager.personal_store.get_user_exemplars("alice")
        self.assertIn("B", alice_exemplars)

    def test_personal_adaptation_confidence_thresholds(self):
        """
        Personal adapter requires at least 2 independent samples and high similarity
        to consider an override. Single sample will NOT trigger override.
        """
        adapter = self.manager.personal_adapter
        f1 = generate_mock_landmarks(60)

        # Store 1 sample for class 'C'
        s1 = self.manager.capture_prediction_snapshot("D", 0.60, f1, user_id="user_charlie")
        self.manager.correct_prediction(s1.prediction_id, "C", user_id="user_charlie")

        # Query with similar features: 1 sample is insufficient -> no override
        test_f = f1 + 0.01
        res1 = adapter.adapt_prediction(
            user_id="user_charlie",
            features=test_f,
            production_prediction="D",
            production_confidence=0.60,
            personalization_enabled=True,
        )
        self.assertFalse(res1["personalization_applied"])
        self.assertEqual(res1["adjusted_prediction"], "D")

        # Now store a 2nd independent sample for 'C'
        f2 = f1 + 0.02
        s2 = self.manager.capture_prediction_snapshot("D", 0.55, f2, user_id="user_charlie")
        self.manager.correct_prediction(s2.prediction_id, "C", user_id="user_charlie")

        # Query again with similar features -> now has 2 samples, satisfies threshold -> overrides!
        res2 = adapter.adapt_prediction(
            user_id="user_charlie",
            features=test_f,
            production_prediction="D",
            production_confidence=0.60,
            personalization_enabled=True,
        )
        self.assertTrue(res2["personalization_applied"])
        self.assertEqual(res2["adjusted_prediction"], "C")
        self.assertGreaterEqual(res2["similarity_score"], 0.88)

    def test_personalization_disabled_behavior(self):
        """When personalization is disabled, production prediction is preserved verbatim."""
        adapter = self.manager.personal_adapter
        f = generate_mock_landmarks(70)

        # Store 2 samples for 'K'
        s1 = self.manager.capture_prediction_snapshot("T", 0.5, f, user_id="user_dave")
        self.manager.correct_prediction(s1.prediction_id, "K", user_id="user_dave")
        s2 = self.manager.capture_prediction_snapshot("T", 0.5, f + 0.02, user_id="user_dave")
        self.manager.correct_prediction(s2.prediction_id, "K", user_id="user_dave")

        # When disabled:
        res = adapter.adapt_prediction(
            user_id="user_dave",
            features=f,
            production_prediction="T",
            production_confidence=0.55,
            personalization_enabled=False,
        )
        self.assertFalse(res["personalization_applied"])
        self.assertEqual(res["adjusted_prediction"], "T")
        self.assertIn("Personalization disabled", res["reason"])

    def test_reset_personal_data(self):
        """Reset completely erases user's stored exemplars and files."""
        f = generate_mock_landmarks(80)
        s = self.manager.capture_prediction_snapshot("X", 0.6, f, user_id="user_eve")
        self.manager.correct_prediction(s.prediction_id, "Y", user_id="user_eve")

        self.assertIn("Y", self.manager.personal_store.get_user_exemplars("user_eve"))

        reset_res = self.manager.reset_personal_data("user_eve")
        self.assertTrue(reset_res["ok"])

        # Cache and store should be empty
        self.assertEqual(len(self.manager.personal_store.get_user_exemplars("user_eve", reload_db=True)), 0)
        stats = self.manager.get_stats("user_eve")
        self.assertEqual(stats["personal_correction_samples"], 0)

    def test_global_optin_default_and_consent_enforcement(self):
        """Global contribution must default to False and enforce strict opt-in."""
        pref = self.collector.get_or_create_user("user_frank")
        self.assertEqual(pref["global_contribution_enabled"], 0)  # Default disabled!

        # Correcting while not opted-in: sample is NOT queued for global contribution
        f = generate_mock_landmarks(90)
        s = self.manager.capture_prediction_snapshot("P", 0.7, f, user_id="user_frank")
        _, res = self.manager.correct_prediction(s.prediction_id, "Q", user_id="user_frank")
        self.assertFalse(res["global_contribution_queued"])

        # Now explicitly opt-in:
        self.manager.global_contribution.grant_consent("user_frank")
        pref_after = self.collector.get_or_create_user("user_frank")
        self.assertEqual(pref_after["global_contribution_enabled"], 1)

        # Subsequent valid correction is now queued!
        s2 = self.manager.capture_prediction_snapshot("P", 0.65, f + 0.05, user_id="user_frank")
        _, res2 = self.manager.correct_prediction(s2.prediction_id, "Q", user_id="user_frank")
        self.assertTrue(res2["global_contribution_queued"])

    def test_consent_withdrawal(self):
        """Withdrawing consent revokes opt-in and marks existing samples as 'withdrawn'."""
        self.manager.global_contribution.grant_consent("user_grace")
        f = generate_mock_landmarks(100)
        s = self.manager.capture_prediction_snapshot("A", 0.6, f, user_id="user_grace")
        self.manager.correct_prediction(s.prediction_id, "B", user_id="user_grace")

        # Verify queued in pending
        pending = self.manager.global_dataset.get_contributions(status="pending")
        self.assertTrue(any(p["user_id"] == "user_grace" for p in pending))

        # Withdraw consent
        w_res = self.manager.global_contribution.withdraw_consent("user_grace", withdraw_existing_samples=True)
        self.assertTrue(w_res["ok"])
        self.assertFalse(w_res["global_contribution_enabled"])
        self.assertGreaterEqual(w_res["withdrawn_contributions_count"], 1)

        # Contributions should now be marked 'withdrawn'
        pending_after = self.manager.global_dataset.get_contributions(status="pending")
        self.assertFalse(any(p["user_id"] == "user_grace" for p in pending_after))


class TestModelRegistryAndEvaluation(unittest.TestCase):
    """Test model registry states, rollback, and evaluation metrics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "registry.db")
        self.reg_dir = os.path.join(self.temp_dir, "models")
        self.collector = SampleCollector(db_path=self.db_path, storage_dir=self.temp_dir)
        self.registry = ModelRegistry(registry_dir=self.reg_dir, collector=self.collector)
        self.evaluator = ModelEvaluator()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_production_model_baseline_registered(self):
        """Production model must be registered in registry with baseline checksum."""
        prod = self.registry.get_model("1.0.0-alphabet-26class")
        self.assertIsNotNone(prod)
        self.assertEqual(prod["deployment_status"], "production")
        self.assertEqual(prod["model_checksum"], PRODUCTION_MODEL_CHECKSUM)

    def test_candidate_registration_and_promotion(self):
        """Registering candidate and explicit manual promotion."""
        # Create dummy candidate model file
        dummy_path = os.path.join(self.temp_dir, "cand.h5")
        with open(dummy_path, "wb") as f:
            f.write(b"MOCK_MODEL_WEIGHTS_CANDIDATE")

        reg_res = self.registry.register_candidate(
            candidate_model_path=dummy_path,
            version_id="candidate_v2",
            dataset_version="curated_v1",
            evaluation_metrics={"accuracy": 0.98},
        )
        self.assertTrue(reg_res["ok"])
        self.assertEqual(reg_res["status"], "candidate")

        # Unauthorized promotion fails
        unauth = self.registry.promote_candidate_to_production("candidate_v2", authorized=False)
        self.assertFalse(unauth["ok"])

        # Authorized promotion succeeds
        auth = self.registry.promote_candidate_to_production("candidate_v2", authorized=True)
        self.assertTrue(auth["ok"])
        self.assertEqual(self.registry.get_model("candidate_v2")["deployment_status"], "production")
        self.assertEqual(self.registry.get_model("1.0.0-alphabet-26class")["deployment_status"], "archived")

        # Rollback restores original
        rb = self.registry.rollback_to_version("1.0.0-alphabet-26class", authorized=True)
        self.assertTrue(rb["ok"])
        self.assertEqual(self.registry.get_model("1.0.0-alphabet-26class")["deployment_status"], "production")

    def test_evaluation_report_and_regression_detection(self):
        """Evaluator computes accuracy, precision, recall, and flags regressions."""
        y_true = np.array(["A", "B", "C", "D", "E"])
        y_pred = np.array(["A", "B", "C", "D", "E"])  # 100%

        baseline = self.evaluator.evaluate_predictions(y_true, y_pred)
        self.assertEqual(baseline.overall_accuracy, 1.0)
        self.assertEqual(baseline.f1_macro, 1.0)
        self.assertFalse(baseline.regression_detected)

        # Regressed candidate
        y_pred_bad = np.array(["A", "A", "A", "D", "E"])  # 60%
        cand_eval = self.evaluator.evaluate_predictions(y_true, y_pred_bad, baseline_report=baseline)
        self.assertEqual(cand_eval.overall_accuracy, 0.6)
        self.assertTrue(cand_eval.regression_detected)
        self.assertGreater(len(cand_eval.regression_details), 0)


class TestFeedbackApiEndpoints(unittest.TestCase):
    """Hermetic API testing of Flask Blueprint v2 routes."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test-secret"
        self.app.register_blueprint(feedback_bp)
        self.client = self.app.test_client()
        self.user_id = f"api_user_{uuid.uuid4().hex[:8]}"
        self.token = get_feedback_manager().privacy_manager.generate_session_token(self.user_id)
        self.auth_headers = {"X-User-ID": self.user_id, "X-Session-Token": self.token}

    def test_feedback_api_preferences_roundtrip(self):
        """GET & PUT /api/v2/learning/preferences."""
        headers = self.auth_headers
        resp = self.client.get("/api/v2/learning/preferences", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["personalization_enabled"])
        self.assertFalse(data["global_contribution_enabled"])

        # Update preferences
        resp2 = self.client.put(
            "/api/v2/learning/preferences",
            headers=headers,
            json={"personalization_enabled": False, "global_contribution_enabled": True},
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertFalse(data2["personalization_enabled"])
        self.assertTrue(data2["global_contribution_enabled"])

    def test_feedback_api_prediction_confirm_correct(self):
        """Register snapshot, confirm, and correct via API."""
        headers = self.auth_headers
        feats = generate_mock_landmarks(120).tolist()

        # 1. Register prediction snapshot
        resp_pred = self.client.post(
            "/api/v2/feedback/prediction",
            headers=headers,
            json={
                "predicted_alphabet": "H",
                "confidence_score": 0.85,
                "feature_vector": feats,
            },
        )
        self.assertEqual(resp_pred.status_code, 201)
        pred_data = resp_pred.get_json()
        pid = pred_data["prediction_id"]

        # 2. Confirm
        resp_conf = self.client.post(
            "/api/v2/feedback/confirm",
            headers=headers,
            json={"prediction_id": pid},
        )
        self.assertEqual(resp_conf.status_code, 200)
        self.assertTrue(resp_conf.get_json()["ok"])

        # 3. Correct (on new prediction)
        resp_pred2 = self.client.post(
            "/api/v2/feedback/prediction",
            headers=headers,
            json={
                "predicted_alphabet": "O",
                "confidence_score": 0.70,
                "feature_vector": feats,
            },
        )
        pid2 = resp_pred2.get_json()["prediction_id"]

        resp_corr = self.client.post(
            "/api/v2/feedback/correct",
            headers=headers,
            json={"prediction_id": pid2, "actual_label": "C"},
        )
        self.assertEqual(resp_corr.status_code, 200)
        self.assertTrue(resp_corr.get_json()["ok"])
        self.assertEqual(resp_corr.get_json()["actual_label"], "C")

        # 4. History and Stats
        resp_hist = self.client.get("/api/v2/feedback/history", headers=headers)
        self.assertEqual(resp_hist.status_code, 200)
        self.assertGreaterEqual(len(resp_hist.get_json()["history"]), 2)

        resp_stats = self.client.get("/api/v2/feedback/stats", headers=headers)
        self.assertEqual(resp_stats.status_code, 200)
        self.assertEqual(resp_stats.get_json()["stats"]["total_reviewed"], 2)


class TestNonNegotiableModelPreservation(unittest.TestCase):
    """
    NON-NEGOTIABLE VERIFICATION:
    Ensure production models and checksums have not been modified, replaced, or overwritten.
    """

    def test_production_single_hand_model_unchanged(self):
        """Single-hand model sha256 checksum match."""
        self.assertTrue(PRODUCTION_MODEL_PATH.exists())
        checksum = compute_sha256(PRODUCTION_MODEL_PATH)
        self.assertEqual(
            checksum,
            PRODUCTION_MODEL_CHECKSUM,
            "CRITICAL VIOLATION: Production single-hand model weights have been modified!",
        )

    def test_production_cnn_model_unchanged(self):
        """CNN model sha256 checksum match."""
        self.assertTrue(CNN_MODEL_PATH.exists())
        checksum = compute_sha256(CNN_MODEL_PATH)
        self.assertEqual(
            checksum,
            CNN_MODEL_CHECKSUM,
            "CRITICAL VIOLATION: Production CNN model weights have been modified!",
        )

    def test_production_dual_hand_model_unchanged(self):
        """Dual-hand model sha256 checksum match."""
        self.assertTrue(DUAL_HAND_MODEL_PATH.exists())
        checksum = compute_sha256(DUAL_HAND_MODEL_PATH)
        self.assertEqual(
            checksum,
            DUAL_HAND_MODEL_CHECKSUM,
            "CRITICAL VIOLATION: Production dual-hand model weights have been modified!",
        )


if __name__ == "__main__":
    unittest.main()
