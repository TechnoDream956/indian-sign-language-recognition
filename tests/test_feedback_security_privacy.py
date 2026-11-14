"""
Hermetic Security and Privacy Test Suite for Feedback & Continuous Learning.
=============================================================================
Validates:
1. Missing session token returns 401 on protected routes.
2. Invalid token signature rejected (401).
3. Expired token rejected (401).
4. Tampered token payload rejected (401).
5. Spoofed X-User-ID header/payload rejected (403).
6. User A cannot access User B's feedback history.
7. User A cannot access User B's feedback stats.
8. User A cannot delete User B's personal learning data.
9. Valid session token successfully accesses its own data.
10. Personal data reset completely removes personal records.
11. Personal data reset withdraws pending global contributions.
12. Personal data reset withdraws accepted global contributions.
13. Withdrawn contributions are excluded from dataset export.
14. Resetting User A does not affect User B's records or contributions.
15. Repeated reset requests are idempotent and return clean status.
"""

import os
os.environ["ISL_CLOUD"] = "1"
os.environ["RENDER"] = "true"
os.environ["ISL_TESTING"] = "1"
os.environ["FEEDBACK_SECRET_KEY"] = "test-isolated-feedback-security-key-32bytes-hex"

import time
import uuid
import json
import shutil
import tempfile
import unittest
import numpy as np
from flask import Flask

from src.feedback_learning.config import (
    FEATURE_SCHEMA_SINGLE_HAND,
    FEEDBACK_SECRET_KEY,
    FEEDBACK_TOKEN_TTL_SECONDS,
)
from src.feedback_learning.prediction_snapshot import PredictionSnapshot
from src.feedback_learning.sample_collector import SampleCollector
from src.feedback_learning.personal_store import PersonalStore
from src.feedback_learning.personal_adapter import PersonalAdapter
from src.feedback_learning.global_contribution import GlobalContributionManager
from src.feedback_learning.global_dataset import GlobalDatasetManager
from src.feedback_learning.privacy_manager import (
    PrivacyManager,
    generate_session_token,
    verify_session_token,
)
from src.feedback_learning.feedback_manager import FeedbackManager
import src.feedback_learning.feedback_api as fb_api_mod


class TestFeedbackSecurityAndPrivacy(unittest.TestCase):
    """Hermetic test suite verifying feedback security, authentication, and privacy guarantees."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_feedback.db")
        self.storage_dir = os.path.join(self.temp_dir, "landmarks")
        self.dataset_dir = os.path.join(self.temp_dir, "global_dataset")

        self.collector = SampleCollector(
            db_path=self.db_path,
            storage_dir=self.storage_dir,
        )
        self.contribution_mgr = GlobalContributionManager(self.collector)
        self.dataset_mgr = GlobalDatasetManager(self.collector, export_dir=self.dataset_dir)
        self.privacy_mgr = PrivacyManager(
            collector=self.collector,
            contribution_manager=self.contribution_mgr,
        )
        self.feedback_mgr = FeedbackManager(
            collector=self.collector,
            privacy_manager=self.privacy_mgr,
            global_contribution=self.contribution_mgr,
            global_dataset=self.dataset_mgr,
        )

        # Patch API blueprint singleton to use hermetic test instance
        self._orig_mgr = fb_api_mod._feedback_manager
        fb_api_mod._feedback_manager = self.feedback_mgr

        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "test-secret-key-2026"
        self.app.register_blueprint(fb_api_mod.feedback_bp)

        # Attach CORS handler matching production app.py
        from flask import Response
        @self.app.after_request
        def add_cors(resp):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-Session-Token"
            resp.headers["Access-Control-Expose-Headers"] = "X-Session-Token"
            return resp

        @self.app.route("/api/<path:subpath>", methods=["OPTIONS"])
        def cors_options(subpath):
            return Response(status=204)

        self.client = self.app.test_client()

        self.user_a = f"usr_alice_{uuid.uuid4().hex[:6]}"
        self.user_b = f"usr_bob_{uuid.uuid4().hex[:6]}"
        self.token_a = self.privacy_mgr.generate_session_token(self.user_a)
        self.token_b = self.privacy_mgr.generate_session_token(self.user_b)

    def tearDown(self):
        fb_api_mod._feedback_manager = self._orig_mgr
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_snapshot(self, user_id: str, label: str = "A") -> PredictionSnapshot:
        feats = np.random.uniform(0.1, 0.9, size=(63,)).astype(np.float32)
        return self.feedback_mgr.capture_prediction_snapshot(
            predicted_alphabet=label,
            confidence_score=0.88,
            feature_vector=feats,
            feature_schema=FEATURE_SCHEMA_SINGLE_HAND,
            user_id=user_id,
        )

    # 1. Missing session token returns 401 on protected routes
    def test_01_missing_session_token_returns_401(self):
        protected_endpoints = [
            ("GET", "/api/v2/feedback/history"),
            ("GET", "/api/v2/feedback/stats"),
            ("GET", "/api/v2/learning/preferences"),
            ("PUT", "/api/v2/learning/preferences", {"personalization_enabled": True}),
            ("DELETE", "/api/v2/learning/personal-data"),
            ("POST", "/api/v2/learning/global-consent"),
            ("DELETE", "/api/v2/learning/global-consent"),
            ("POST", "/api/v2/feedback/confirm", {"prediction_id": "test_id"}),
            ("POST", "/api/v2/feedback/correct", {"prediction_id": "test_id", "actual_label": "B"}),
        ]
        for method, url, *payload in protected_endpoints:
            kwargs = {}
            if payload:
                kwargs["json"] = payload[0]
            if method == "GET":
                resp = self.client.get(url, **kwargs)
            elif method == "POST":
                resp = self.client.post(url, **kwargs)
            elif method == "PUT":
                resp = self.client.put(url, **kwargs)
            elif method == "DELETE":
                resp = self.client.delete(url, **kwargs)

            self.assertEqual(
                resp.status_code,
                401,
                f"Expected 401 for {method} {url} without session token, got {resp.status_code}",
            )
            data = resp.get_json()
            self.assertFalse(data.get("ok"))
            self.assertIn("error", data)

    # 2. Invalid token signature rejected
    def test_02_invalid_token_signature_rejected(self):
        bad_token = f"{self.user_a}.{int(time.time())}.deadbeefcafebabe0123456789abcdef"
        headers = {"X-Session-Token": bad_token}
        resp = self.client.get("/api/v2/feedback/history", headers=headers)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid session token signature", resp.get_json()["error"])

    # 3. Expired token rejected
    def test_03_expired_token_rejected(self):
        expired_ts = int(time.time()) - (FEEDBACK_TOKEN_TTL_SECONDS + 3600)
        expired_token = self.privacy_mgr.generate_session_token(self.user_a, timestamp=expired_ts)
        headers = {"X-Session-Token": expired_token}
        resp = self.client.get("/api/v2/learning/preferences", headers=headers)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("expired", resp.get_json()["error"].lower())

    # 4. Tampered token payload rejected
    def test_04_tampered_token_payload_rejected(self):
        # Generate valid token for user_a, then modify payload to user_b
        valid_parts = self.token_a.split(".")
        tampered_token = f"{self.user_b}.{valid_parts[1]}.{valid_parts[2]}"
        headers = {"X-Session-Token": tampered_token}
        resp = self.client.get("/api/v2/feedback/stats", headers=headers)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid session token signature", resp.get_json()["error"])

    # 5. Spoofed X-User-ID header rejected / ignored when token does not match
    def test_05_spoofed_user_id_rejected(self):
        # User A's token with header claiming User B
        headers = {
            "X-Session-Token": self.token_a,
            "X-User-ID": self.user_b,
        }
        resp = self.client.get("/api/v2/feedback/history", headers=headers)
        self.assertEqual(resp.status_code, 403)
        err = resp.get_json()["error"]
        self.assertEqual(err, "Forbidden: User identity does not match authenticated session.")
        self.assertNotIn(self.user_a, err)
        self.assertNotIn(self.user_b, err)

        # Also query param spoofing
        resp2 = self.client.get(f"/api/v2/feedback/history?user_id={self.user_b}", headers={"X-Session-Token": self.token_a})
        self.assertEqual(resp2.status_code, 403)
        err2 = resp2.get_json()["error"]
        self.assertEqual(err2, "Forbidden: User identity does not match authenticated session.")
        self.assertNotIn(self.user_a, err2)
        self.assertNotIn(self.user_b, err2)

    # 6. User A cannot access User B's feedback history
    def test_06_user_a_cannot_access_user_b_history(self):
        # Create history for user B
        snap_b = self._create_mock_snapshot(self.user_b, label="B")
        self.feedback_mgr.confirm_prediction(snap_b.prediction_id, user_id=self.user_b)

        # User A queries their own history
        headers_a = {"X-Session-Token": self.token_a}
        resp_a = self.client.get("/api/v2/feedback/history", headers=headers_a)
        self.assertEqual(resp_a.status_code, 200)
        data_a = resp_a.get_json()
        self.assertEqual(data_a["user_id"], self.user_a)
        self.assertEqual(data_a["count"], 0)

        # User B queries their own history
        headers_b = {"X-Session-Token": self.token_b}
        resp_b = self.client.get("/api/v2/feedback/history", headers=headers_b)
        self.assertEqual(resp_b.status_code, 200)
        data_b = resp_b.get_json()
        self.assertEqual(data_b["user_id"], self.user_b)
        self.assertEqual(data_b["count"], 1)

    # 7. User A cannot access User B's feedback stats
    def test_07_user_a_cannot_access_user_b_stats(self):
        snap_b = self._create_mock_snapshot(self.user_b, label="B")
        self.feedback_mgr.confirm_prediction(snap_b.prediction_id, user_id=self.user_b)

        headers_a = {"X-Session-Token": self.token_a}
        resp_a = self.client.get("/api/v2/feedback/stats", headers=headers_a)
        self.assertEqual(resp_a.status_code, 200)
        stats_a = resp_a.get_json()["stats"]
        self.assertEqual(stats_a["total_reviewed"], 0)

        headers_b = {"X-Session-Token": self.token_b}
        resp_b = self.client.get("/api/v2/feedback/stats", headers=headers_b)
        self.assertEqual(resp_b.status_code, 200)
        stats_b = resp_b.get_json()["stats"]
        self.assertEqual(stats_b["total_reviewed"], 1)

    # 8. User A cannot delete User B's personal learning data
    def test_08_user_a_cannot_delete_user_b_personal_data(self):
        # Bob contributes a correction
        snap_b = self._create_mock_snapshot(self.user_b, label="B")
        self.feedback_mgr.correct_prediction(snap_b.prediction_id, actual_label="C", user_id=self.user_b)

        # Verify Bob has samples
        samples_b = self.collector.get_user_landmark_arrays(self.user_b)
        self.assertEqual(len(samples_b), 1)

        # Alice attempts deletion (even trying to spoof X-User-ID)
        resp_spoof = self.client.delete(
            "/api/v2/learning/personal-data",
            headers={"X-Session-Token": self.token_a, "X-User-ID": self.user_b},
        )
        self.assertEqual(resp_spoof.status_code, 403)

        # Alice performs legitimate deletion of her own data
        resp_alice = self.client.delete(
            "/api/v2/learning/personal-data",
            headers={"X-Session-Token": self.token_a},
        )
        self.assertEqual(resp_alice.status_code, 200)
        self.assertEqual(resp_alice.get_json()["user_id"], self.user_a)

        # Bob's data must remain intact
        samples_b_after = self.collector.get_user_landmark_arrays(self.user_b)
        self.assertEqual(len(samples_b_after), 1)

    # 9. Valid session token successfully accesses its own data
    def test_09_valid_session_token_accesses_own_data(self):
        headers_a = {"X-Session-Token": self.token_a}
        # 1. Register prediction snapshot
        feats = np.random.uniform(0.1, 0.9, size=(63,)).tolist()
        resp_pred = self.client.post(
            "/api/v2/feedback/prediction",
            headers=headers_a,
            json={
                "predicted_alphabet": "A",
                "confidence_score": 0.92,
                "feature_vector": feats,
            },
        )
        self.assertEqual(resp_pred.status_code, 201)
        pid = resp_pred.get_json()["prediction_id"]

        # 2. Confirm prediction
        resp_conf = self.client.post(
            "/api/v2/feedback/confirm",
            headers=headers_a,
            json={"prediction_id": pid},
        )
        self.assertEqual(resp_conf.status_code, 200)
        self.assertTrue(resp_conf.get_json()["ok"])

        # 3. Read history & stats
        resp_hist = self.client.get("/api/v2/feedback/history", headers=headers_a)
        self.assertEqual(resp_hist.status_code, 200)
        self.assertEqual(len(resp_hist.get_json()["history"]), 1)

        resp_stats = self.client.get("/api/v2/feedback/stats", headers=headers_a)
        self.assertEqual(resp_stats.status_code, 200)
        self.assertEqual(resp_stats.get_json()["stats"]["total_reviewed"], 1)

    # 10. Personal data reset completely removes personal records
    def test_10_personal_data_reset_removes_personal_records(self):
        # Alice enables opt-in and saves correction
        self.contribution_mgr.grant_consent(self.user_a)
        snap = self._create_mock_snapshot(self.user_a, label="K")
        self.feedback_mgr.correct_prediction(snap.prediction_id, actual_label="L", user_id=self.user_a)

        # Check records exist
        self.assertEqual(len(self.collector.get_user_landmark_arrays(self.user_a)), 1)
        self.assertGreaterEqual(len(self.collector.get_feedback_history(self.user_a)), 1)

        # Reset personal data
        res = self.collector.reset_personal_data(self.user_a)
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(res["removed_files"], 1)

        # Verify personal records purged
        self.assertEqual(len(self.collector.get_user_landmark_arrays(self.user_a)), 0)
        self.assertEqual(len(self.collector.get_feedback_history(self.user_a)), 0)

    # 11. Personal data reset withdraws pending global contributions
    def test_11_personal_data_reset_withdraws_pending_contributions(self):
        # Alice opts in to global contribution
        self.contribution_mgr.grant_consent(self.user_a)
        snap = self._create_mock_snapshot(self.user_a, label="M")
        self.feedback_mgr.correct_prediction(snap.prediction_id, actual_label="N", user_id=self.user_a)

        # Check pending contributions exist
        pending = self.dataset_mgr.get_contributions(status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["user_id"], self.user_a)
        self.assertEqual(pending[0]["contribution_status"], "pending")

        # Reset personal data via API
        resp = self.client.delete(
            "/api/v2/learning/personal-data",
            headers={"X-Session-Token": self.token_a},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertGreaterEqual(data["withdrawn_contributions_count"], 1)

        # Verify no pending contributions remain
        pending_after = self.dataset_mgr.get_contributions(status="pending")
        self.assertEqual(len(pending_after), 0)

        # Verify status in database is 'withdrawn'
        with self.collector._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT contribution_status, review_notes FROM global_contributions WHERE user_id = ?", (self.user_a,))
            rows = cur.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["contribution_status"], "withdrawn")
            self.assertIn("Withdrawn via complete personal data reset", rows[0]["review_notes"])

    # 12. Personal data reset withdraws accepted global contributions
    def test_12_personal_data_reset_withdraws_accepted_contributions(self):
        # Alice opts in and submits correction
        self.contribution_mgr.grant_consent(self.user_a)
        snap = self._create_mock_snapshot(self.user_a, label="P")
        self.feedback_mgr.correct_prediction(snap.prediction_id, actual_label="Q", user_id=self.user_a)

        pending = self.dataset_mgr.get_contributions(status="pending")
        self.assertEqual(len(pending), 1)
        contrib_id = pending[0]["contribution_id"]

        # Moderate and accept the sample
        mod_res = self.dataset_mgr.moderate_contribution(contrib_id, new_status="accepted", notes="audit")
        self.assertTrue(mod_res["ok"])
        self.assertEqual(mod_res["status"], "accepted")

        # Verify it is accepted in dataset
        accepted_feats, _, _ = self.dataset_mgr.load_accepted_samples()
        self.assertEqual(len(accepted_feats), 1)

        # Reset personal data
        res = self.collector.reset_personal_data(self.user_a)
        self.assertTrue(res["ok"])
        self.assertEqual(res["withdrawn_contributions_count"], 1)

        # Verify it is withdrawn and NOT accepted anymore
        accepted_after, _, _ = self.dataset_mgr.load_accepted_samples()
        self.assertEqual(len(accepted_after), 0)

        with self.collector._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT contribution_status FROM global_contributions WHERE contribution_id = ?", (contrib_id,))
            row = cur.fetchone()
            self.assertEqual(row["contribution_status"], "withdrawn")

    # 13. Withdrawn contributions are excluded from dataset export
    def test_13_withdrawn_contributions_excluded_from_export(self):
        self.contribution_mgr.grant_consent(self.user_a)
        snap = self._create_mock_snapshot(self.user_a, label="R")
        self.feedback_mgr.correct_prediction(snap.prediction_id, actual_label="S", user_id=self.user_a)

        pending = self.dataset_mgr.get_contributions(status="pending")
        contrib_id = pending[0]["contribution_id"]
        self.dataset_mgr.moderate_contribution(contrib_id, new_status="accepted")

        # Reset personal data to withdraw
        self.collector.reset_personal_data(self.user_a)

        # Export dataset manifest
        manifest = self.dataset_mgr.export_dataset(dataset_version="audit-v1.0")
        self.assertFalse(manifest.get("ok", False))
        self.assertEqual(manifest["sample_count"], 0)

    # 14. Resetting User A does not affect User B's records or contributions
    def test_14_resetting_user_a_does_not_affect_user_b(self):
        # Alice & Bob both contribute samples
        self.contribution_mgr.grant_consent(self.user_a)
        self.contribution_mgr.grant_consent(self.user_b)

        snap_a = self._create_mock_snapshot(self.user_a, label="A")
        self.feedback_mgr.correct_prediction(snap_a.prediction_id, actual_label="B", user_id=self.user_a)

        snap_b = self._create_mock_snapshot(self.user_b, label="C")
        self.feedback_mgr.correct_prediction(snap_b.prediction_id, actual_label="D", user_id=self.user_b)

        # Moderate Bob's sample to accepted
        pending = self.dataset_mgr.get_contributions(status="pending")
        bob_contrib = next(c for c in pending if c["user_id"] == self.user_b)
        self.dataset_mgr.moderate_contribution(bob_contrib["contribution_id"], new_status="accepted")

        # Alice resets her data
        self.collector.reset_personal_data(self.user_a)

        # Verify Alice is cleared
        self.assertEqual(len(self.collector.get_user_landmark_arrays(self.user_a)), 0)

        # Verify Bob is completely intact
        self.assertEqual(len(self.collector.get_user_landmark_arrays(self.user_b)), 1)
        accepted_feats, _, accepted_uids = self.dataset_mgr.load_accepted_samples()
        self.assertEqual(len(accepted_feats), 1)
        self.assertEqual(accepted_uids[0], self.user_b)

    # 15. Repeated reset requests are idempotent and return clean status
    def test_15_repeated_reset_requests_idempotent(self):
        self.contribution_mgr.grant_consent(self.user_a)
        snap = self._create_mock_snapshot(self.user_a, label="X")
        self.feedback_mgr.correct_prediction(snap.prediction_id, actual_label="Y", user_id=self.user_a)

        # First reset
        res1 = self.collector.reset_personal_data(self.user_a)
        self.assertTrue(res1["ok"])
        self.assertGreaterEqual(res1["removed_files"], 1)

        # Second reset (immediately after)
        res2 = self.collector.reset_personal_data(self.user_a)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["removed_files"], 0)
        self.assertEqual(res2["withdrawn_contributions_count"], 0)

        # Third reset via API
        resp = self.client.delete(
            "/api/v2/learning/personal-data",
            headers={"X-Session-Token": self.token_a},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()["ok"])

    # 16. Production secret management fails closed when missing or insecure
    def test_16_production_secret_fails_closed(self):
        from src.feedback_learning.config import resolve_feedback_secret_key
        from unittest.mock import patch

        # Production with missing secret fails closed
        with patch.dict(os.environ, {"ISL_FORCE_PRODUCTION_CHECK": "1", "ISL_TESTING": "0"}, clear=False):
            with patch.dict(os.environ, {"FEEDBACK_SECRET_KEY": "", "SECRET_KEY": ""}):
                with self.assertRaises(RuntimeError) as ctx:
                    resolve_feedback_secret_key()
                self.assertIn("CRITICAL STARTUP ERROR", str(ctx.exception))
                self.assertIn("not configured", str(ctx.exception))

            # Production with insecure default fails closed
            with patch.dict(os.environ, {"FEEDBACK_SECRET_KEY": "isl-feedback-secret-2026-auth-token"}):
                with self.assertRaises(RuntimeError) as ctx2:
                    resolve_feedback_secret_key()
                self.assertIn("CRITICAL STARTUP ERROR", str(ctx2.exception))
                self.assertIn("insecure", str(ctx2.exception))

            # Production with valid 32-char secret succeeds
            valid_secret = "a" * 32
            with patch.dict(os.environ, {"FEEDBACK_SECRET_KEY": valid_secret}):
                res = resolve_feedback_secret_key()
                self.assertEqual(res, valid_secret)

    # 17. CORS preflight permits X-Session-Token and exposes it
    def test_17_cors_preflight_allows_and_exposes_session_token(self):
        # Test OPTIONS preflight for feedback endpoint
        resp = self.client.open(
            "/api/v2/feedback/history",
            method="OPTIONS",
            headers={
                "Origin": "https://my-app.vercel.app",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type, X-Session-Token",
            },
        )
        self.assertIn(resp.status_code, [200, 204])
        allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
        self.assertIn("X-Session-Token", allow_headers)
        expose_headers = resp.headers.get("Access-Control-Expose-Headers", "")
        self.assertIn("X-Session-Token", expose_headers)

    # 18. Authorization errors never disclose internal user IDs or spoofed IDs
    def test_18_authorization_errors_never_disclose_user_ids(self):
        spoofed_payload = {
            "prediction_id": "test_pred_999",
            "actual_label": "B",
            "user_id": "attacker_spoofed_id",
        }
        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers={"X-Session-Token": self.token_a},
            data=json.dumps(spoofed_payload),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)
        err = resp.get_json()["error"]
        self.assertEqual(err, "Forbidden: User identity does not match authenticated session.")
        self.assertNotIn("attacker_spoofed_id", err)
        self.assertNotIn(self.user_a, err)


if __name__ == "__main__":
    unittest.main()
