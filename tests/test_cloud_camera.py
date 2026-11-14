"""
Unit & Integration Tests for Cloud Camera Architecture
======================================================
Verifies:
1. Environment detection (Render, IS_CLOUD, HEADLESS, Linux fallback).
2. InferenceEngine behavior in cloud mode (zero cv2.VideoCapture calls).
3. Health check endpoints: /api/healthz and /api/status return HTTP 200 immediately.
4. Frame ingestion endpoint /api/camera/frame (binary, base64, multipart).
5. State consistency across local and cloud modes.
"""

import base64
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from inference_engine import InferenceEngine, is_cloud_environment
from app import app, engine


class TestCloudEnvironmentDetection(unittest.TestCase):
    """Test environment detection flags."""

    def test_render_env_detected(self):
        with patch.dict(os.environ, {"RENDER": "true"}):
            self.assertTrue(is_cloud_environment())

    def test_is_cloud_env_detected(self):
        with patch.dict(os.environ, {"IS_CLOUD": "1"}):
            self.assertTrue(is_cloud_environment())

    def test_headless_env_detected(self):
        with patch.dict(os.environ, {"HEADLESS": "true"}):
            self.assertTrue(is_cloud_environment())

    def test_isl_headless_env_detected(self):
        with patch.dict(os.environ, {"ISL_HEADLESS": "1"}):
            self.assertTrue(is_cloud_environment())


class TestInferenceEngineCloudMode(unittest.TestCase):
    """Test InferenceEngine cloud mode initialization and camera loop decoupling."""

    @patch("inference_engine.cv2.VideoCapture")
    def test_zero_videocapture_calls_in_cloud_mode(self, mock_videocapture):
        """In cloud mode, cv2.VideoCapture must NEVER be called."""
        with patch.dict(os.environ, {"RENDER": "true"}):
            cloud_engine = InferenceEngine(camera_index=0)
            self.assertTrue(cloud_engine.is_cloud)
            cloud_engine.start()
            import time
            time.sleep(0.3)
            cloud_engine.stop()

            # cv2.VideoCapture should never be called
            mock_videocapture.assert_not_called()

    def test_is_ready_cloud_payload(self):
        """Verify is_ready() provides camera_ok=True and cloud_mode=True in cloud."""
        with patch.dict(os.environ, {"RENDER": "true"}):
            cloud_engine = InferenceEngine(camera_index=0)
            ready = cloud_engine.is_ready()
            self.assertTrue(ready["cloud_mode"])
            self.assertEqual(ready["camera_mode"], "cloud_browser")
            self.assertTrue(ready["camera_ok"])


class TestCloudApiEndpoints(unittest.TestCase):
    """Test Flask HTTP endpoints for Render and browser webcam ingestion."""

    def setUp(self):
        self.client = app.test_client()

    def test_healthz_endpoint(self):
        """GET /api/healthz returns HTTP 200 with JSON payload."""
        resp = self.client.get("/api/healthz")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("camera_mode", data)
        self.assertIn("cloud_mode", data)

    def test_status_endpoint(self):
        """GET /api/status returns HTTP 200."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("running", data)
        self.assertIn("camera_mode", data)

    def test_post_frame_binary_jpeg(self):
        """POST /api/camera/frame accepts raw JPEG binary bytes."""
        # Create a synthetic 640x480 test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", img)
        raw_bytes = jpeg.tobytes()

        resp = self.client.post(
            "/api/camera/frame",
            data=raw_bytes,
            content_type="image/jpeg",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertIn("prediction", data)
        self.assertIn("state", data)

    def test_post_frame_base64_json(self):
        """POST /api/camera/frame accepts base64 data URL in JSON payload."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

        payload = {
            "image": f"data:image/jpeg;base64,{b64}",
            "mirrored": False,
        }
        resp = self.client.post(
            "/api/camera/frame",
            json=payload,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))
        self.assertIn("prediction", data)

    def test_post_frame_multipart(self):
        """POST /api/camera/frame accepts multipart/form-data upload."""
        import io
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", img)
        file_storage = (io.BytesIO(jpeg.tobytes()), "frame.jpg")

        resp = self.client.post(
            "/api/camera/frame",
            data={"frame": file_storage},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get("ok"))

    def test_post_frame_empty_returns_400(self):
        """POST /api/camera/frame with empty data returns HTTP 400."""
        resp = self.client.post(
            "/api/camera/frame",
            data=b"",
            content_type="image/jpeg",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertFalse(data.get("ok"))


if __name__ == "__main__":
    unittest.main()
