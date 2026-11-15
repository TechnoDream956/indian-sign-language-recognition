"""
Hermetic Tests for UI Feedback Correction, Camera Controls, and Session Quit.
=============================================================================
Validates:
1. F -> V prediction correction flow via authenticated feedback API.
2. Missing session token rejection (401 Unauthorized).
3. Invalid prediction ID rejection.
4. Cross-user unauthorized prediction modification protection (403 Forbidden).
5. Feedback history and statistics update following correction.
6. Frontend deployment copies synchronization across src/ and public/.
7. UI element structure and accessibility contracts.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from flask import Flask

from src.feedback_learning.config import (
    FEATURE_SCHEMA_SINGLE_HAND,
    resolve_feedback_secret_key,
)
from src.feedback_learning.feedback_api import feedback_bp, get_feedback_manager
from src.feedback_learning.privacy_manager import PrivacyManager
from src.feedback_learning.sample_collector import SampleCollector


class TestFeedbackCorrectionAndControls(unittest.TestCase):
    """Hermetic unit tests for correction API and frontend contracts."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="isl_correction_test_")
        self.db_path = Path(self.temp_dir) / "feedback.db"
        self.landmarks_dir = Path(self.temp_dir) / "landmarks"

        self.collector = SampleCollector(db_path=self.db_path, storage_dir=self.landmarks_dir)
        self.privacy_mgr = PrivacyManager()

        # Build hermetic test Flask app
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test-isl-secret-key"
        self.app.register_blueprint(feedback_bp)
        self.client = self.app.test_client()

        # Create authenticated test user
        self.user_id = "usr_correction_tester_01"
        self.session_token = self.privacy_mgr.generate_session_token(self.user_id)
        self.headers = {
            "Content-Type": "application/json",
            "X-Session-Token": self.session_token,
        }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_f_to_v_correction_authenticated_flow(self):
        """
        Verify the complete F -> V correction flow:
        Model predicted F; user corrects to V; API updates continuous learning and history.
        """
        fb_mgr = get_feedback_manager()

        # 1. Register a prediction of 'F' with 63D dummy feature vector
        feats = np.random.randn(63).astype(np.float32)
        snapshot = fb_mgr.capture_prediction_snapshot(
            predicted_alphabet="F",
            confidence_score=0.88,
            feature_vector=feats,
            feature_schema=FEATURE_SCHEMA_SINGLE_HAND,
            user_id=self.user_id,
        )
        pid = snapshot.prediction_id
        self.assertIsNotNone(pid)

        # 2. Submit correction to 'V'
        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers=self.headers,
            json={"prediction_id": pid, "actual_label": "V"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["actual_label"], "V")
        self.assertEqual(data["original_label"], "F")
        self.assertIn("correction_id", data)

        # 3. Verify feedback history reflects F -> V
        hist_resp = self.client.get("/api/v2/feedback/history", headers=self.headers)
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.get_json()
        self.assertTrue(hist_data["ok"])
        self.assertGreaterEqual(len(hist_data["history"]), 1)

        event = hist_data["history"][0]
        self.assertEqual(event["prediction_id"], pid)
        self.assertEqual(event["actual_label"], "V")

        # 4. Verify statistics
        stats_resp = self.client.get("/api/v2/feedback/stats", headers=self.headers)
        self.assertEqual(stats_resp.status_code, 200)
        stats_data = stats_resp.get_json()["stats"]
        self.assertGreaterEqual(stats_data["incorrect_corrections"], 1)

    def test_correction_missing_session_token_rejected(self):
        """Requests without session token are rejected with 401."""
        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers={"Content-Type": "application/json"},
            json={"prediction_id": "pred_test_123", "actual_label": "V"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertFalse(resp.get_json()["ok"])
        self.assertIn("Authentication required", resp.get_json()["error"])

    def test_correction_invalid_prediction_id_rejected(self):
        """Submitting non-existent prediction ID returns an error."""
        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers=self.headers,
            json={"prediction_id": "pred_nonexistent_xyz", "actual_label": "V"},
        )
        self.assertIn(resp.status_code, (400, 404))
        self.assertFalse(resp.get_json()["ok"])

    def test_unauthorized_cross_user_prediction_ownership(self):
        """User B cannot correct User A's prediction (cross-session 403 Forbidden)."""
        fb_mgr = get_feedback_manager()

        # User A creates snapshot
        feats = np.random.randn(63).astype(np.float32)
        snapshot_a = fb_mgr.capture_prediction_snapshot(
            predicted_alphabet="F",
            confidence_score=0.91,
            feature_vector=feats,
            user_id="usr_user_a",
        )

        # User B attempts to correct User A's prediction
        user_b_token = self.privacy_mgr.generate_session_token("usr_user_b")
        headers_b = {
            "Content-Type": "application/json",
            "X-Session-Token": user_b_token,
        }

        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers=headers_b,
            json={"prediction_id": snapshot_a.prediction_id, "actual_label": "V"},
        )
        self.assertEqual(resp.status_code, 403)
        self.assertFalse(resp.get_json()["ok"])
        self.assertIn("recorded by another session", resp.get_json()["error"])

    def test_invalid_session_token_rejected_with_401(self):
        """Malformed or forged token returns 401 Unauthorized."""
        resp = self.client.post(
            "/api/v2/feedback/correct",
            headers={
                "Content-Type": "application/json",
                "X-Session-Token": "usr_fake_user.123456789.bad_hmac_signature",
            },
            json={"prediction_id": "pred_test_123", "actual_label": "V"},
        )
        self.assertEqual(resp.status_code, 401)
        self.assertFalse(resp.get_json()["ok"])
        self.assertIn("Unauthorized", resp.get_json()["error"])

    def test_same_session_can_correct_multiple_fresh_predictions(self):
        """The same authenticated session can record and correct multiple predictions."""
        fb_mgr = get_feedback_manager()

        # Correction 1: A -> B
        s1 = fb_mgr.capture_prediction_snapshot(
            predicted_alphabet="A",
            confidence_score=0.85,
            feature_vector=np.random.randn(63).astype(np.float32),
            user_id=self.user_id,
        )
        r1 = self.client.post(
            "/api/v2/feedback/correct",
            headers=self.headers,
            json={"prediction_id": s1.prediction_id, "actual_label": "B"},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.get_json()["ok"])

        # Correction 2: C -> D
        s2 = fb_mgr.capture_prediction_snapshot(
            predicted_alphabet="C",
            confidence_score=0.88,
            feature_vector=np.random.randn(63).astype(np.float32),
            user_id=self.user_id,
        )
        r2 = self.client.post(
            "/api/v2/feedback/correct",
            headers=self.headers,
            json={"prediction_id": s2.prediction_id, "actual_label": "D"},
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.get_json()["ok"])

    def test_session_initialization_and_persistence(self):
        """Verify /api/v2/session endpoint issues tokens and preserves identities across re-auth."""
        # 1. Fresh session with no token
        resp1 = self.client.post("/api/v2/session")
        self.assertEqual(resp1.status_code, 200)
        d1 = resp1.get_json()
        self.assertTrue(d1["ok"])
        self.assertTrue(d1["user_id"].startswith("usr_"))
        token1 = d1["session_token"]
        self.assertIsNotNone(token1)
        self.assertEqual(resp1.headers.get("X-Session-Token"), token1)

        # 2. Returning session (e.g. browser refresh sending stored token)
        resp2 = self.client.post(
            "/api/v2/session",
            headers={"X-Session-Token": token1},
        )
        self.assertEqual(resp2.status_code, 200)
        d2 = resp2.get_json()
        self.assertTrue(d2["ok"])
        self.assertEqual(d2["user_id"], d1["user_id"])  # Same user identity preserved!

    def test_redundant_pause_camera_button_absent(self):
        """Verify that obsolete #btn-toggle-camera button is removed from all templates."""
        root = Path(__file__).resolve().parent.parent
        template_paths = [root / "src" / "templates" / "index.html"]
        if (root / "public" / "index.html").exists():
            template_paths.append(root / "public" / "index.html")
        for template_path in template_paths:
            html = template_path.read_text(encoding="utf-8")
            self.assertNotIn('id="btn-toggle-camera"', html)
            self.assertNotIn("Pause Camera", html)

    def test_global_camera_toggle_present(self):
        """Verify that global #btn-session-toggle is present in all templates."""
        root = Path(__file__).resolve().parent.parent
        template_paths = [root / "src" / "templates" / "index.html"]
        if (root / "public" / "index.html").exists():
            template_paths.append(root / "public" / "index.html")
        for template_path in template_paths:
            html = template_path.read_text(encoding="utf-8")
            self.assertIn('id="btn-session-toggle"', html)

    def test_frontend_files_synchronization(self):
        """Verify that canonical frontend files are present and valid."""
        root = Path(__file__).resolve().parent.parent

        src_app_js = (root / "src" / "static" / "app.js").read_text(encoding="utf-8")
        src_css = (root / "src" / "static" / "style.css").read_text(encoding="utf-8")
        self.assertTrue(len(src_app_js) > 1000)
        self.assertTrue(len(src_css) > 1000)

        pub_app_js = root / "public" / "app.js"
        if pub_app_js.exists():
            pub_static_app_js = root / "public" / "static" / "app.js"
            self.assertEqual(src_app_js, pub_app_js.read_text(encoding="utf-8"))
            self.assertEqual(src_app_js, pub_static_app_js.read_text(encoding="utf-8"))
            pub_css = root / "public" / "style.css"
            pub_static_css = root / "public" / "static" / "style.css"
            self.assertEqual(src_css, pub_css.read_text(encoding="utf-8"))
            self.assertEqual(src_css, pub_static_css.read_text(encoding="utf-8"))

    def test_html_ui_elements_present(self):
        """Verify that correction modal, camera toggle, and exit screen exist in all HTML files."""
        root = Path(__file__).resolve().parent.parent

        required_ids = [
            "btn-session-toggle",
            "btn-quit-app",
            "btn-commit-sign",
            "btn-correct-prediction",
            "modal-correction",
            "correction-original-glyph",
            "correction-target-glyph",
            "correction-az-buttons",
            "correction-select-letter",
            "btn-submit-correction",
            "btn-cancel-correction",
            "modal-quit-confirm",
            "btn-confirm-quit",
            "btn-cancel-quit",
            "exit-screen",
            "btn-return-to-app",
        ]

        template_paths = [root / "src" / "templates" / "index.html"]
        if (root / "public" / "index.html").exists():
            template_paths.append(root / "public" / "index.html")
        for template_path in template_paths:
            html = template_path.read_text(encoding="utf-8")
            for elem_id in required_ids:
                self.assertIn(
                    f'id="{elem_id}"',
                    html,
                    f"Missing required element #{elem_id} in {template_path.name}",
                )

    def test_events_endpoint_authentication_hardening(self):
        """Verify that /events rejects unauthenticated requests and accepts X-Session-Token header."""
        from src.app import app as full_app
        client = full_app.test_client()

        # 1. Missing token -> 401 Unauthorized
        r_missing = client.get("/events")
        self.assertEqual(r_missing.status_code, 401)
        self.assertFalse(r_missing.get_json()["ok"])
        self.assertIn("Authentication required", r_missing.get_json()["error"])

        # 2. Invalid signature token -> 401 Unauthorized
        r_invalid = client.get("/events", headers={"X-Session-Token": "usr_fake.12345.invalidsig"})
        self.assertEqual(r_invalid.status_code, 401)
        self.assertFalse(r_invalid.get_json()["ok"])
        self.assertIn("Unauthorized", r_invalid.get_json()["error"])

        # 3. OPTIONS preflight -> 204 No Content with CORS
        r_options = client.open("/events", method="OPTIONS")
        self.assertEqual(r_options.status_code, 204)
        self.assertIn("X-Session-Token", r_options.headers.get("Access-Control-Allow-Headers", ""))

        # 4. Valid signed token in header -> 200 OK text/event-stream
        token = self.privacy_mgr.generate_session_token("usr_test_stream_auth")
        r_valid = client.get("/events", headers={"X-Session-Token": token}, buffered=False)
        self.assertEqual(r_valid.status_code, 200)
        self.assertEqual(r_valid.mimetype, "text/event-stream")
        r_valid.close()

    def test_frontend_sse_security_and_cancellation_contracts(self):
        """Verify that frontend JavaScript uses header-based Fetch SSE without URL tokens and handles cancellation."""
        root = Path(__file__).resolve().parent.parent

        candidate_paths = [
            root / "src" / "static" / "app.js",
            root / "public" / "app.js",
            root / "public" / "static" / "app.js",
        ]
        js_paths = [p for p in candidate_paths if p.exists()]
        self.assertTrue(len(js_paths) >= 1)

        for js_path in js_paths:
            content = js_path.read_text(encoding="utf-8")

            # Must NEVER place tokens in the SSE URL
            self.assertNotIn("?token=", content, f"URL token query string found in {js_path}")
            self.assertNotIn("/events?token=", content, f"URL token query string found in {js_path}")

            # Must use Fetch with X-Session-Token
            self.assertIn('getApiUrl("/events")', content, f"Missing getApiUrl(/events) in {js_path}")
            self.assertIn("sseAbortController", content, f"Missing sseAbortController in {js_path}")
            self.assertIn("disconnectSSE", content, f"Missing disconnectSSE in {js_path}")
            self.assertIn("scheduleSseReconnect", content, f"Missing scheduleSseReconnect in {js_path}")

            # Verify Stop Camera and Quit Website disconnect SSE
            self.assertIn("disconnectSSE();", content, f"Missing disconnectSSE call in {js_path}")

    def test_is_localhost_environment_helper_and_contracts(self):
        """Verify isLocalhostEnvironment is defined, referenced correctly, and evaluates accurately."""
        import subprocess

        root = Path(__file__).resolve().parent.parent

        candidate_paths = [
            root / "src" / "static" / "app.js",
            root / "public" / "app.js",
            root / "public" / "static" / "app.js",
        ]
        js_paths = [p for p in candidate_paths if p.exists()]
        self.assertTrue(len(js_paths) >= 1)

        for js_path in js_paths:
            content = js_path.read_text(encoding="utf-8")
            self.assertIn("function isLocalhostEnvironment()", content, f"Helper missing in {js_path}")
            self.assertIn("const isLocal = isLocalhostEnvironment();", content, f"Call missing in {js_path}")
            self.assertIn("isLocalhostEnvironment() ? \"\" : RENDER_BACKEND_URL", content)

        # Run Node execution to verify boolean outputs under different environments
        node_script = """
        const fs = require('fs');
        const code = fs.readFileSync('src/static/app.js', 'utf8');
        const mockDoc = { getElementById: () => null, querySelectorAll: () => [], addEventListener: () => {} };
        const mockStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

        function testEnv(location) {
          const win = { location, addEventListener: () => {} };
          const fn = new Function('window', 'document', 'localStorage', 'sessionStorage', code + '\\nreturn isLocalhostEnvironment();');
          return fn(win, mockDoc, mockStorage, mockStorage);
        }

        const results = {
          localhost: testEnv({ hostname: 'localhost', protocol: 'http:' }),
          loopback: testEnv({ hostname: '127.0.0.1', protocol: 'http:' }),
          fileProto: testEnv({ hostname: '', protocol: 'file:' }),
          vercel: testEnv({ hostname: 'indian-sign-language-recognition-4t6xeb63n.vercel.app', protocol: 'https:' }),
          render: testEnv({ hostname: 'isl-recognition-backend.onrender.com', protocol: 'https:' })
        };
        console.log(JSON.stringify(results));
        """
        proc = subprocess.run(["node", "-e", node_script], cwd=str(root), capture_output=True, text=True, check=True)
        results = json.loads(proc.stdout.strip())
        self.assertTrue(results["localhost"])
        self.assertTrue(results["loopback"])
        self.assertTrue(results["fileProto"])
        self.assertFalse(results["vercel"])
        self.assertFalse(results["render"])

    def test_cloud_webcam_session_activation_and_stream_loop_contracts(self):
        """Verify cloud webcam streaming activation, streamLoop kickoff, and error separation."""
        import subprocess

        root = Path(__file__).resolve().parent.parent

        candidate_paths = [
            root / "src" / "static" / "app.js",
            root / "public" / "app.js",
            root / "public" / "static" / "app.js",
        ]
        js_paths = [p for p in candidate_paths if p.exists()]
        self.assertTrue(len(js_paths) >= 1)

        for js_path in js_paths:
            content = js_path.read_text(encoding="utf-8")
            self.assertIn("streamLoop(videoEl)", content)
            self.assertIn("streamingActive = true;", content)
            self.assertIn("isPermissionError", content)
            self.assertIn("Camera session activation error:", content)
            self.assertIn("browserWebcamStream.getTracks().forEach", content)

        node_script = """
        const fs = require('fs');
        const code = fs.readFileSync('src/static/app.js', 'utf8');

        let toasts = [];
        let streamLoopCalls = 0;

        const mockDoc = {
          getElementById: (id) => ({
            id,
            style: {},
            classList: { add: () => {}, remove: () => {}, toggle: () => {} },
            addEventListener: () => {},
            appendChild: () => {},
            play: async () => {},
            pause: () => {},
            videoWidth: 640
          }),
          querySelectorAll: () => [],
          addEventListener: () => {},
          createElement: () => ({
            className: '',
            style: {},
            classList: { add: () => {} },
            appendChild: () => {},
            remove: () => {}
          }),
          body: { appendChild: () => {} }
        };
        const mockStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

        // Test 1: Successful cloud activation starts streamLoop exactly once
        const win1 = {
          location: { hostname: 'preview.vercel.app', protocol: 'https:' },
          addEventListener: () => {},
          navigator: {
            mediaDevices: {
              getUserMedia: async () => ({
                getTracks: () => [{ stop: () => {} }]
              })
            }
          }
        };

        const fn1 = new Function('window', 'document', 'localStorage', 'sessionStorage', 'navigator', 'onLoopCall', code + '\\n' + `
          streamLoop = function(v) { onLoopCall(); };
          return enableCloudWebcamMode().then(() => ({
            streamingActive,
            isCloudMode,
            hasStream: !!browserWebcamStream
          }));
        `);

        fn1(win1, mockDoc, mockStorage, mockStorage, win1.navigator, () => { streamLoopCalls++; })
          .then((res1) => {
            // Test 2: Error separation for permission denial
            let permissionToasts = [];
            const win2 = {
              location: { hostname: 'preview.vercel.app', protocol: 'https:' },
              addEventListener: () => {},
              navigator: {
                mediaDevices: {
                  getUserMedia: async () => {
                    const e = new Error('User blocked camera');
                    e.name = 'NotAllowedError';
                    throw e;
                  }
                }
              }
            };

            const fn2 = new Function('window', 'document', 'localStorage', 'sessionStorage', 'navigator', 'onToast', code + '\\n' + `
              showToast = function(msg, type) { onToast(msg, type); };
              return enableCloudWebcamMode();
            `);

            return fn2(win2, mockDoc, mockStorage, mockStorage, win2.navigator, (msg, type) => {
              permissionToasts.push({ msg, type });
            }).then(() => ({
              res1,
              streamLoopCalls,
              permissionToasts
            }));
          })
          .then((out) => {
            console.log(JSON.stringify(out));
            process.exit(0);
          });
        """
        proc = subprocess.run(["node", "-e", node_script], cwd=str(root), capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout.strip())
        self.assertTrue(data["res1"]["streamingActive"])
        self.assertTrue(data["res1"]["isCloudMode"])
        self.assertTrue(data["res1"]["hasStream"])
        self.assertEqual(data["streamLoopCalls"], 1)

        # Check permission error toast
        p_toasts = data["permissionToasts"]
        self.assertGreaterEqual(len(p_toasts), 1)
        self.assertIn("Webcam permission denied: User blocked camera", p_toasts[0]["msg"])


if __name__ == "__main__":
    unittest.main()
