#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InferenceEngine: High-Performance Multi-Modal ISL Recognition Engine.
=====================================================================
Pipeline:
  webcam frame → MediaPipe Hands (up to 2 hands) + Hand ROI Cropping →
  Landmark Pathway (Single 26-Class / Dual 20-Class) + RGB CNN Pathway →
  Confidence-Aware Decision Fusion →
  Temporal Debouncing & Hold-to-Emit State Machine →
  WordBuffer Assembly → Indic Multilingual Translation → Non-blocking TTSEngine.
"""

from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = WORKSPACE_DIR / "models"
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

from alphabet_engine import AlphabetEngine, WordBuffer, normalize_hand_landmarks
from dual_hand_engine import DualHandEngine
from translator import LANGUAGE_DISPLAY_NAMES, translate_sentence
from tts_engine import TTSEngine


def is_cloud_environment() -> bool:
    """
    Detect if running in Render, Cloud, Docker container or headless environment
    where physical server-side cameras do not exist.
    """
    if os.environ.get("RENDER"):
        return True
    if os.environ.get("IS_CLOUD") in ("1", "true", "True"):
        return True
    if os.environ.get("HEADLESS") in ("1", "true", "True") or os.environ.get("ISL_HEADLESS") in ("1", "true", "True"):
        return True
    # If on Linux and no video devices exist in /dev
    if sys.platform.startswith("linux"):
        video_devs = [f"/dev/video{i}" for i in range(4)]
        if not any(os.path.exists(v) for v in video_devs):
            return True
    return False


class InferenceEngine:
    """
    Thread-safe engine for Indian Sign Language Alphabet and Fingerspelling Translation.
    Supports camera start/stop, multi-modal CNN + Landmark fusion, and multilingual speech.
    """

    _active_instance: Optional["InferenceEngine"] = None
    _instance_count: int = 0

    @classmethod
    def get_active_instance(cls) -> Optional["InferenceEngine"]:
        """Return the currently registered active InferenceEngine singleton."""
        return cls._active_instance

    def __init__(
        self,
        camera_index: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
    ):
        InferenceEngine._active_instance = self
        InferenceEngine._instance_count += 1

        self._camera_index = camera_index
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._is_cloud = is_cloud_environment()
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._frame_lock)
        self._raw_frame_lock = threading.Lock()
        self._raw_frame_cond = threading.Condition(self._raw_frame_lock)
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._raw_frame_seq: int = 0
        self._frame_id: int = 0
        self._thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._camera_enabled = True
        self._speech_enabled = True
        self._session_user_id: Optional[str] = None

        # Pipeline Architecture
        self._app_mode = "alphabet"
        self._active_model_id = "hybrid_fusion"
        self._alphabet_engine: Optional[AlphabetEngine] = None
        self._tts_engine: Optional[TTSEngine] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._hands_detector: Any = None

        self._model_registry: Dict[str, Dict[str, Any]] = {
            "hybrid_fusion": {
                "id": "hybrid_fusion",
                "name": "Hybrid Multi-Modal Fusion (Single 26D + Dual 144D + CNN)",
                "classes": 26,
                "class_count": 26,
                "status": "Active (Default)",
                "experimental": False,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Dynamic dispatcher: 1 hand -> 26 classes, 2 hands -> 20 classes with CNN fusion.",
            },
            "dual_hand_p23_144d": {
                "id": "dual_hand_p23_144d",
                "name": "Prompt 23 Dual-Hand D-Ablation (144D)",
                "classes": 20,
                "class_count": 20,
                "status": "Experimental",
                "experimental": True,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Top-performing dual-hand checkpoint with recovered Class K features.",
            },
            "dual_hand_p23_140d": {
                "id": "dual_hand_p23_140d",
                "name": "Prompt 23 Dual-Hand D-Minimal (140D)",
                "classes": 20,
                "class_count": 20,
                "status": "Experimental",
                "experimental": True,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Compact dual-hand checkpoint with 14 geometric contact features.",
            },
            "alphabet_26": {
                "id": "alphabet_26",
                "name": "Production Single-Hand (26 Letters A–Z)",
                "classes": 26,
                "class_count": 26,
                "status": "Production Baseline",
                "experimental": False,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Original 63-feature single-hand landmark baseline classifier.",
            },
            "legacy_13": {
                "id": "legacy_13",
                "name": "Legacy 13-Sign Vocabulary",
                "classes": 13,
                "class_count": 13,
                "status": "Archived Legacy",
                "experimental": False,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Original 13 sign classes mapped into alphabet / word pipeline.",
            },
            "mopgru_29": {
                "id": "mopgru_29",
                "name": "Expanded 29-Sign Vocabulary",
                "classes": 29,
                "class_count": 29,
                "status": "Archived Legacy",
                "experimental": False,
                "display_threshold": 0.50,
                "speech_threshold": 0.70,
                "description": "Expanded 29 sign classes mapped into alphabet / word pipeline.",
            },
        }

        # Shared Telemetry State
        self._state: Dict[str, Any] = {
            "app_mode": "alphabet",
            "active_model": self._active_model_id,
            "model_display_name": self._model_registry[self._active_model_id]["name"],
            "model_experimental": False,
            "model_class_count": 26,
            "camera_ok": True if self._is_cloud else False,
            "camera_mode": "cloud_browser" if self._is_cloud else "local_hardware",
            "cloud_mode": self._is_cloud,
            "camera_enabled": True,
            "model_ok": False,
            "speech_enabled": True,
            "speech_status": "Ready",
            "language": "English",
            "fps": 0.0,
            "camera_fps": 0.0,
            "inference_fps": 0.0,
            "latency_ms": 0.0,
            "t_mediapipe_ms": 0.0,
            "t_infer_ms": 0.0,
            "hand_detected": False,
            "alphabet_hand_count": 0,
            "alphabet_hand_status": "NO HAND",
            # Predictions & Multi-Modal Pathways
            "pred_label": "--",
            "confidence": 0.0,
            "translated_label": "--",
            "landmark_prediction": None,
            "landmark_confidence": 0.0,
            "landmark_model": "--",
            "cnn_prediction": None,
            "cnn_confidence": 0.0,
            "cnn_status": "OFFLINE",
            "fused_prediction": None,
            "fused_confidence": 0.0,
            "fusion_mode": "INITIALIZING",
            "fusion_reason": "",
            # Latch & Debouncing
            "alphabet_raw_letter": "--",
            "alphabet_confidence": 0.0,
            "alphabet_stable_letter": "--",
            "alphabet_emitted_letter": None,
            "alphabet_emission_id": 0,
            "emission_id": 0,
            "alphabet_state": "IDLE",
            # WordBuffer & Translation
            "alphabet_current_word": "",
            "alphabet_sentence": "",
            "alphabet_full_text": "",
            "alphabet_target_language": "English",
            "alphabet_translation": "",
        }
        self._frame_bytes: Optional[bytes] = None

        # Preload models
        try:
            self._preload_alphabet_model()
            self._state["model_ok"] = True
        except Exception as e:
            print(f"[InferenceEngine Warning] Eager model preload deferred: {e}")

    # ------------------------------------------------------------------ #
    #  Model & Mode APIs                                                 #
    # ------------------------------------------------------------------ #

    def get_available_models(self) -> List[dict]:
        with self._lock:
            return list(self._model_registry.values())

    def get_active_model(self) -> dict:
        with self._lock:
            return self._model_registry.get(self._active_model_id, self._model_registry["hybrid_fusion"])

    def set_model(self, model_id: str) -> dict:
        with self._lock:
            if model_id not in self._model_registry:
                raise ValueError(f"Unknown model_id: {model_id}. Available: {list(self._model_registry.keys())}")
            self._active_model_id = model_id
            cfg = self._model_registry[model_id]
            self._state["active_model"] = model_id
            self._state["model_display_name"] = cfg["name"]
            self._state["model_experimental"] = cfg["experimental"]
            self._state["model_class_count"] = cfg["classes"]

            # Configure AlphabetEngine
            if self._alphabet_engine is not None:
                if model_id == "dual_hand_p23_140d":
                    self._alphabet_engine.dual_hand_engine = DualHandEngine(config_key="config_d_minimal_140d", use_prompt23=True)
                elif model_id in ("dual_hand_p23_144d", "hybrid_fusion"):
                    self._alphabet_engine.dual_hand_engine = DualHandEngine(config_key="config_d_ablation_144d", use_prompt23=True)
            return {"success": True, "active_model": cfg}

    def get_session_user_id(self) -> Optional[str]:
        with self._lock:
            return self._session_user_id

    def set_session_user_id(self, user_id: Optional[str]) -> None:
        with self._lock:
            self._session_user_id = str(user_id).strip() if user_id else None

    def get_app_mode(self) -> str:
        return self._app_mode

    def set_app_mode(self, mode: str) -> dict:
        with self._lock:
            self._app_mode = mode
            self._state["app_mode"] = mode
            return {"success": True, "app_mode": mode}

    # ------------------------------------------------------------------ #
    #  Camera Hardware Controls                                          #
    # ------------------------------------------------------------------ #

    def toggle_camera(self) -> dict:
        with self._lock:
            self._camera_enabled = not self._camera_enabled
            self._state["camera_enabled"] = self._camera_enabled
            enabled = self._camera_enabled
        return {"ok": True, "camera_enabled": enabled}

    def start_camera(self) -> dict:
        with self._lock:
            self._camera_enabled = True
            self._state["camera_enabled"] = True
        return {"ok": True, "camera_enabled": True}

    def stop_camera(self) -> dict:
        with self._lock:
            self._camera_enabled = False
            self._state["camera_enabled"] = False
        return {"ok": True, "camera_enabled": False}

    def toggle_speech(self) -> dict:
        with self._lock:
            self._speech_enabled = not self._speech_enabled
            self._state["speech_enabled"] = self._speech_enabled
            enabled = self._speech_enabled
        return {"ok": True, "speech_enabled": enabled}

    # ------------------------------------------------------------------ #
    #  Lifecycle & State Access                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        if self._is_cloud:
            self._worker_thread = threading.Thread(target=self._run_cloud, daemon=True, name="CloudWorker")
            self._thread = self._worker_thread
            self._worker_thread.start()
        else:
            self._capture_thread = threading.Thread(target=self._run_camera_capture, daemon=True, name="CameraCapture")
            self._worker_thread = threading.Thread(target=self._run_inference_worker, daemon=True, name="InferenceWorker")
            self._thread = self._worker_thread
            self._capture_thread.start()
            self._worker_thread.start()

    def stop(self) -> None:
        self._running = False

        # Wake all waiting threads immediately
        with self._raw_frame_lock:
            self._raw_frame_cond.notify_all()
        with self._frame_condition:
            self._frame_condition.notify_all()

        # Release camera immediately so blocking cap.read() unblocks promptly
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
            except Exception:
                pass

        for th in (getattr(self, "_capture_thread", None), getattr(self, "_worker_thread", None), self._thread):
            if th is not None and th.is_alive() and threading.current_thread() != th:
                th.join(timeout=1.0)

        if self._hands_detector is not None:
            try:
                self._hands_detector.close()
            except Exception:
                pass
            self._hands_detector = None

        with self._lock:
            self._state["camera_ok"] = False

    def get_latest_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def get_state(self) -> dict:
        """Alias for get_latest_state for backward-compatibility with tests."""
        return self.get_latest_state()

    @property
    def is_cloud(self) -> bool:
        return self._is_cloud

    def is_ready(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "running": self._running,
                "camera_ok": self._state.get("camera_ok", False),
                "camera_enabled": self._camera_enabled,
                "camera_mode": self._state.get("camera_mode", "cloud_browser" if self._is_cloud else "local_hardware"),
                "cloud_mode": self._is_cloud,
                "model_ok": self._state.get("model_ok", False),
                "app_mode": self._app_mode,
                "active_model": self._active_model_id,
                "camera_fps": self._state.get("camera_fps", 0.0),
                "inference_fps": self._state.get("inference_fps", 0.0),
                "fps": self._state.get("fps", 0.0),
                "latency_ms": self._state.get("latency_ms", 0.0),
                "t_mediapipe_ms": self._state.get("t_mediapipe_ms", 0.0),
                "t_infer_ms": self._state.get("t_infer_ms", 0.0),
                "active_engines": getattr(self.__class__, "_instance_count", 1),
            }

    def wait_for_frame(self, last_frame_id: int = -1, timeout: float = 0.1) -> Tuple[int, Optional[bytes]]:
        with self._frame_condition:
            if self._frame_id == last_frame_id and self._running:
                self._frame_condition.wait(timeout=timeout)
            return self._frame_id, self._frame_bytes

    def set_language(self, language: str) -> None:
        with self._lock:
            self._state["language"] = language
            self._state["alphabet_target_language"] = language
            if self._alphabet_engine is not None:
                self._alphabet_engine.set_target_language(language)

    # ------------------------------------------------------------------ #
    #  WordBuffer & Text Composition Actions                             #
    # ------------------------------------------------------------------ #

    def get_alphabet_state(self) -> dict:
        with self._lock:
            return self._get_alphabet_state_locked()

    def alphabet_space(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.space()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_add_letter(self, letter: str) -> dict:
        with self._lock:
            if self._alphabet_engine is not None and letter:
                clean_char = letter.strip().upper()[:1]
                if clean_char:
                    self._alphabet_engine.word_buffer.add_letter(clean_char)
                    self._sync_alphabet_state_locked()
                    return {
                        "ok": True,
                        "letter": clean_char,
                        "current_word": self._alphabet_engine.word_buffer.get_current_word(),
                        "sentence": self._alphabet_engine.word_buffer.get_sentence()
                    }
            return {"ok": False}

    def alphabet_add_space(self) -> dict:
        return self.alphabet_space()

    def alphabet_backspace(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.backspace()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_commit_word(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.commit_word()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_clear_word(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.clear_word()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_clear_sentence(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.clear_sentence()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_clear_all(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.clear_all()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_reset(self) -> dict:
        with self._lock:
            if self._alphabet_engine is not None:
                res = self._alphabet_engine.reset()
                self._sync_alphabet_state_locked()
                return res
            return {}

    def alphabet_get_translation(self) -> dict:
        with self._lock:
            cur_lang = self._state.get("language", "English")
            sentence = self._state.get("alphabet_sentence", "").strip()
            word = self._state.get("alphabet_current_word", "").strip()
            src = (sentence + " " + word).strip() if sentence else word
            trans = translate_sentence(src, cur_lang) if src else ""
            return {"source": src, "language": cur_lang, "translation": trans}

    def speak_current(self) -> bool:
        with self._lock:
            if not self._speech_enabled:
                return False
            sentence = self._state.get("alphabet_sentence", "").strip()
            word = self._state.get("alphabet_current_word", "").strip()
            text_to_speak = (sentence + " " + word).strip() if sentence else word
            target_lang = self._state.get("language", "English")

        if not text_to_speak:
            return False
        if self._tts_engine is None:
            self._tts_engine = TTSEngine()

        if target_lang != "English":
            translated = translate_sentence(text_to_speak, target_lang)
            return self._tts_engine.speak(translated, target_lang)
        return self._tts_engine.speak(text_to_speak, "English")

    def alphabet_speak(self, language: str = "English") -> bool:
        with self._lock:
            if self._alphabet_engine is not None:
                self._sync_alphabet_state_locked()
            if not self._speech_enabled:
                return False
            if self._tts_engine and self._tts_engine.is_speaking:
                return False
            sentence = self._state.get("alphabet_sentence", "").strip()
            word = self._state.get("alphabet_current_word", "").strip()
            text_to_speak = (sentence + " " + word).strip() if sentence else word

        if not text_to_speak:
            return False
        if self._tts_engine is None:
            self._tts_engine = TTSEngine()
        return self._tts_engine.speak(text_to_speak, language)

    def alphabet_speak_translation(self) -> bool:
        with self._lock:
            if self._alphabet_engine is not None:
                self._sync_alphabet_state_locked()
            if not self._speech_enabled:
                return False
            if self._tts_engine and self._tts_engine.is_speaking:
                return False
            sentence = self._state.get("alphabet_sentence", "").strip()
            word = self._state.get("alphabet_current_word", "").strip()
            text_to_speak = (sentence + " " + word).strip() if sentence else word
            target_lang = self._state.get("language", "English")

        if not text_to_speak:
            return False
        if self._tts_engine is None:
            self._tts_engine = TTSEngine()

        if target_lang != "English":
            translated = translate_sentence(text_to_speak, target_lang)
            return self._tts_engine.speak(translated, language=target_lang)
        return self._tts_engine.speak(text_to_speak, language="English")

    # ------------------------------------------------------------------ #
    #  Internal Helpers                                                  #
    # ------------------------------------------------------------------ #

    def _preload_alphabet_model(self) -> None:
        if self._alphabet_engine is None:
            self._alphabet_engine = AlphabetEngine(tts_engine=self._tts_engine)

    def _sync_alphabet_state_locked(self) -> None:
        if self._alphabet_engine is None:
            return
        snapshot = self._alphabet_engine.get_state_snapshot()
        cur_lang = self._state.get("language", "English")
        sent = snapshot.get("sentence", "")
        word = snapshot.get("current_word", "")
        full_text = snapshot.get("full_text", "")
        src = full_text.strip()
        trans = translate_sentence(src, cur_lang) if src else ""

        self._state.update({
            "alphabet_current_word": word,
            "alphabet_sentence": sent,
            "alphabet_full_text": full_text,
            "alphabet_target_language": cur_lang,
            "alphabet_translation": trans,
            "alphabet_state": snapshot.get("state", "IDLE"),
            "alphabet_emission_id": snapshot.get("emission_id", 0),
            "emission_id": snapshot.get("emission_id", 0),
        })

    def _get_alphabet_state_locked(self) -> dict:
        if self._alphabet_engine is None:
            return {"app_mode": self._app_mode}
        base = self._alphabet_engine.get_state_snapshot()
        cur_lang = self._state.get("language", "English")
        sent = base.get("sentence", "")
        word = base.get("current_word", "")
        full_text = base.get("full_text", "")
        src = full_text.strip()
        trans = translate_sentence(src, cur_lang) if src else ""
        base["target_language"] = cur_lang
        base["alphabet_target_language"] = cur_lang
        base["translation"] = trans
        base["alphabet_translation"] = trans
        base["alphabet_current_word"] = word
        base["alphabet_sentence"] = sent
        base["alphabet_full_text"] = full_text
        base["app_mode"] = self._app_mode
        base["confidence"] = base.get("raw_confidence", 0.0)
        base["prediction"] = base.get("raw_prediction")
        base["alphabet_raw_letter"] = base.get("raw_prediction")
        base["alphabet_stable_letter"] = base.get("stable_letter")
        base["alphabet_emitted_letter"] = base.get("emitted_letter")
        base["alphabet_emission_id"] = base.get("emission_id", 0)
        return base

    def _create_standby_frame(self, message: str = "Webcam Paused") -> bytes:
        """Create a clean, styled standby image when camera is paused or disconnected."""
        img = np.zeros((self._camera_height, self._camera_width, 3), dtype=np.uint8)
        img[:] = (20, 24, 30)  # Dark theme slate background

        # Header accent bar
        cv2.rectangle(img, (0, 0), (self._camera_width, 4), (0, 180, 160), -1)

        # Center title
        cv2.putText(
            img, "ISL TRANSLATOR", (self._camera_width // 2 - 120, self._camera_height // 2 - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
        )
        # Message
        cv2.putText(
            img, message, (self._camera_width // 2 - 140, self._camera_height // 2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 180), 1, cv2.LINE_AA
        )
        # Hint
        cv2.putText(
            img, "Click 'Start Camera' or verify permissions", (self._camera_width // 2 - 170, self._camera_height // 2 + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA
        )

        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    # ------------------------------------------------------------------ #
    #  Inference Thread Loop & Frame Processor                           #
    # ------------------------------------------------------------------ #

    def process_frame(
        self,
        frame: np.ndarray,
        is_mirrored: bool = False,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image frame through MediaPipe Holistic tracking,
        multi-modal decision fusion (AlphabetEngine + RGB CNN), WordBuffer debouncing,
        and multilingual Indic translation.

        Thread-safe: Can be called from local hardware loop or cloud browser frame endpoint.
        """
        if frame is None or frame.size == 0:
            return {"ok": False, "error": "Empty frame"}

        now = current_time if current_time is not None else time.time()

        # Mirror image for natural user interaction if not already mirrored
        image = frame if is_mirrored else cv2.flip(frame, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        t_start = time.time()

        with self._infer_lock:
            t_mp_start = time.time()
            if self._hands_detector is None:
                mp_hands = mp.solutions.hands
                self._mp_drawing = mp.solutions.drawing_utils
                self._mp_drawing_styles = mp.solutions.drawing_styles
                self._mp_hands = mp_hands
                self._hands_detector = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.50,
                )

            hand_res = self._hands_detector.process(rgb)
            rgb.flags.writeable = True
            t_mediapipe_ms = round((time.time() - t_mp_start) * 1000, 2)

            multi_lms = hand_res.multi_hand_landmarks
            multi_handedness = hand_res.multi_handedness

            # Extract normalized landmark coordinates for browser client overlay
            extracted_landmarks = []
            if multi_lms:
                for idx, hand_lms in enumerate(multi_lms):
                    h_label = "Unknown"
                    h_score = 1.0
                    if multi_handedness and idx < len(multi_handedness):
                        try:
                            h_class = multi_handedness[idx].classification[0]
                            h_label = h_class.label  # "Left" or "Right"
                            h_score = float(h_class.score)
                        except Exception:
                            pass
                    pts = [
                        [round(float(lm.x), 4), round(float(lm.y), 4), round(float(lm.z), 4)]
                        for lm in hand_lms.landmark
                    ]
                    extracted_landmarks.append({
                        "handedness": h_label,
                        "score": round(h_score, 4),
                        "points": pts,
                    })

            # Check if MJPEG encoding is needed (local hardware mode or active web stream)
            jpeg_bytes = None
            needs_jpeg = (not self._is_cloud) or getattr(self, "_has_video_subscribers", False)
            if needs_jpeg:
                if multi_lms and hasattr(self, "_mp_drawing"):
                    for hand_landmarks in multi_lms:
                        self._mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self._mp_hands.HAND_CONNECTIONS,
                            self._mp_drawing_styles.get_default_hand_landmarks_style(),
                            self._mp_drawing_styles.get_default_hand_connections_style(),
                        )
                _, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = jpeg.tobytes()

            # Execute Multi-Modal Recognition and Fusion under _infer_lock (WITHOUT holding self._lock!)
            t_model_start = time.time()
            if self._alphabet_engine is not None:
                alpha_status = self._alphabet_engine.process_dual_landmarks(
                    multi_lms, multi_handedness, rgb_frame=rgb, current_time=now
                )
            else:
                alpha_status = {}
            t_infer_ms = round((time.time() - t_model_start) * 1000, 2)

            recent_emitted = getattr(self, "_recent_emitted", None)
            emitted_timer = getattr(self, "_emitted_display_timer", 0.0)

            if alpha_status.get("emitted_letter"):
                recent_emitted = alpha_status["emitted_letter"]
                emitted_timer = now + 1.2
            elif now > emitted_timer:
                recent_emitted = None

            self._recent_emitted = recent_emitted
            self._emitted_display_timer = emitted_timer

            cur_sent = alpha_status.get("sentence", "")
            cur_full = alpha_status.get("full_text", "")
            cur_word = alpha_status.get("current_word", "")

            with self._lock:
                cur_lang = self._state.get("language", "English")

            proc_time = round((time.time() - t_start) * 1000, 1)

            # Snapshot capture for feedback learning (outside self._lock)
            prediction_id = None
            pred_candidate = alpha_status.get("fused_prediction") or alpha_status.get("raw_prediction")
            if extracted_landmarks and len(extracted_landmarks) > 0 and pred_candidate and pred_candidate not in ("--", "None", "..."):
                try:
                    from feedback_learning.feedback_api import get_feedback_manager
                    fb_mgr = get_feedback_manager()
                    pts = extracted_landmarks[0].get("points", [])
                    if len(pts) >= 21:
                        from alphabet_engine import normalize_hand_landmarks
                        feats = normalize_hand_landmarks(np.array(pts[:21], dtype=np.float32))
                        active_uid = self.get_session_user_id() or "usr_local"
                        snapshot = fb_mgr.capture_prediction_snapshot(
                            predicted_alphabet=pred_candidate,
                            confidence_score=float(alpha_status.get("fused_confidence") or alpha_status.get("raw_confidence") or 0.0),
                            feature_vector=feats,
                            raw_landmarks=extracted_landmarks,
                            tracking_quality={"score": extracted_landmarks[0].get("score", 1.0), "hand_count": alpha_status.get("hand_count", 1)},
                            metadata={"fusion_mode": alpha_status.get("fusion_mode"), "emission_id": alpha_status.get("emission_id", 0)},
                            user_id=active_uid,
                        )
                        prediction_id = snapshot.prediction_id
                except Exception:
                    pass

            # Compute translation outside self._lock
            src_text = cur_full.strip()
            prev_trans_key = getattr(self, "_last_trans_key", None)
            cur_trans_key = (src_text, cur_lang)

            if cur_trans_key != prev_trans_key:
                alpha_trans = translate_sentence(src_text, cur_lang) if src_text else ""
                self._last_trans_key = cur_trans_key
                self._cached_translation = alpha_trans
            else:
                alpha_trans = getattr(self, "_cached_translation", "")

            # Acquire self._lock ONLY for microsecond state update (<0.05 ms)
            with self._lock:
                self._state.update({
                    "app_mode": "alphabet",
                    "camera_ok": True,
                    "prediction_id": prediction_id or self._state.get("prediction_id"),
                    "pred_label": alpha_status.get("raw_prediction") or "--",
                    "translated_label": alpha_status.get("raw_prediction") or "--",
                    "confidence": round(float(alpha_status.get("raw_confidence", 0.0)), 4),
                    "alphabet_raw_letter": alpha_status.get("raw_prediction") or "--",
                    "alphabet_confidence": round(float(alpha_status.get("raw_confidence", 0.0)), 4),
                    "alphabet_stable_letter": alpha_status.get("stable_letter") or "--",
                    "alphabet_last_stable_letter": alpha_status.get("last_stable_letter") or self._state.get("alphabet_last_stable_letter") or "--",
                    "last_stable_letter": alpha_status.get("last_stable_letter") or self._state.get("last_stable_letter") or "--",
                    "stability_progress": alpha_status.get("stability_progress", 0.0),
                    "consecutive_count": alpha_status.get("consecutive_count", 0),
                    "stability_frames": alpha_status.get("stability_frames", 5),
                    "alphabet_emitted_letter": recent_emitted,
                    "alphabet_emission_id": alpha_status.get("emission_id", 0),
                    "emission_id": alpha_status.get("emission_id", 0),
                    "alphabet_current_word": cur_word,
                    "alphabet_sentence": cur_sent,
                    "alphabet_full_text": cur_full,
                    "alphabet_target_language": cur_lang,
                    "alphabet_translation": alpha_trans,
                    "alphabet_state": alpha_status.get("state", "IDLE"),
                    "alphabet_hand_detected": alpha_status.get("hand_detected", False),
                    "alphabet_hand_count": alpha_status.get("hand_count", 0),
                    "alphabet_hand_status": alpha_status.get("hand_status", "NO HAND"),
                    "hand_detected": alpha_status.get("hand_detected", False),
                    "landmarks": extracted_landmarks,
                    "processing_time_ms": proc_time,
                    "latency_ms": proc_time,
                    "t_mediapipe_ms": t_mediapipe_ms,
                    "t_infer_ms": t_infer_ms,
                    "speech_status": "Speaking..." if (self._tts_engine and self._tts_engine.is_speaking) else "Ready",
                    # Pathway Telemetry
                    "landmark_prediction": alpha_status.get("landmark_prediction"),
                    "landmark_confidence": round(float(alpha_status.get("landmark_confidence", 0.0)), 4),
                    "landmark_model": alpha_status.get("active_pipeline", "Landmark"),
                    "cnn_prediction": alpha_status.get("cnn_prediction"),
                    "cnn_confidence": round(float(alpha_status.get("cnn_confidence", 0.0)), 4),
                    "cnn_status": alpha_status.get("cnn_status", "OFFLINE"),
                    "fused_prediction": alpha_status.get("fused_prediction"),
                    "fused_confidence": round(float(alpha_status.get("fused_confidence", 0.0)), 4),
                    "fusion_mode": alpha_status.get("fusion_mode", "LANDMARK_ONLY"),
                    "fusion_reason": alpha_status.get("fusion_reason", ""),
                })

            # Video stream frame notification on independent _frame_condition
            with self._frame_condition:
                if jpeg_bytes is not None:
                    self._frame_bytes = jpeg_bytes
                self._frame_id += 1
                self._frame_condition.notify_all()

        with self._lock:
            return dict(self._state)

    def _run_cloud(self) -> None:
        """Cloud mode loop: server-side hardware camera loop is completely disabled."""
        self._tts_engine = TTSEngine()
        try:
            self._preload_alphabet_model()
            with self._lock:
                self._state["model_ok"] = True
        except Exception as e:
            with self._lock:
                self._state["speech_status"] = f"Model load error: {e}"

        initial_standby = self._create_standby_frame("Awaiting Browser Webcam Feed")
        with self._frame_condition:
            self._frame_bytes = initial_standby
            self._frame_id += 1
            self._frame_condition.notify_all()

        print("[InferenceEngine] Cloud environment detected. Server-side physical camera loop disabled.")
        print("[InferenceEngine] Ready for browser webcam frames via /api/camera/frame.")
        with self._lock:
            self._state["camera_ok"] = True
            self._state["camera_mode"] = "cloud_browser"
            self._state["alphabet_hand_status"] = "AWAITING BROWSER WEBCAM"

        while self._running:
            time.sleep(0.5)

        if self._hands_detector is not None:
            try:
                self._hands_detector.close()
            except Exception:
                pass
            self._hands_detector = None

    def _run_camera_capture(self) -> None:
        """Dedicated thread for low-latency hardware camera capture, dropping stale frames."""
        fps_buffer = deque(maxlen=30)
        last_frame_time = time.time()
        consecutive_cam_failures = 0
        cap = None

        while self._running:
            with self._lock:
                camera_enabled = self._camera_enabled

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                    self._cap = None
                with self._lock:
                    self._state["camera_ok"] = False
                    self._state["alphabet_hand_status"] = "CAMERA PAUSED"

                standby_bytes = self._create_standby_frame("Webcam Stream Paused")
                with self._frame_condition:
                    self._frame_bytes = standby_bytes
                    self._frame_id += 1
                    self._frame_condition.notify_all()
                time.sleep(0.1)
                continue

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self._camera_index)
                self._cap = cap
                if not cap.isOpened():
                    consecutive_cam_failures += 1
                    with self._lock:
                        self._state["camera_ok"] = False
                        self._state["speech_status"] = "Camera hardware unavailable"
                        self._state["alphabet_hand_status"] = "NO CAMERA ACCESS"

                    standby_bytes = self._create_standby_frame("Camera Hardware Offline")
                    with self._frame_condition:
                        self._frame_bytes = standby_bytes
                        self._frame_id += 1
                        self._frame_condition.notify_all()
                    backoff = min(2.0, 0.5 * consecutive_cam_failures)
                    time.sleep(backoff)
                    continue

                consecutive_cam_failures = 0
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                with self._lock:
                    self._state["camera_ok"] = True

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)
            smoothed_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

            with self._raw_frame_lock:
                self._latest_raw_frame = frame
                self._raw_frame_seq += 1
                self._raw_frame_cond.notify_all()

            with self._lock:
                self._state["camera_fps"] = round(smoothed_fps, 1)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
        with self._lock:
            self._state["camera_ok"] = False

    def _run_inference_worker(self) -> None:
        """Dedicated worker thread consuming latest camera frame and running neural inference."""
        self._tts_engine = TTSEngine()
        try:
            self._preload_alphabet_model()
            with self._lock:
                self._state["model_ok"] = True
        except Exception as e:
            with self._lock:
                self._state["speech_status"] = f"Model load error: {e}"

        initial_msg = "Initializing Camera..."
        initial_standby = self._create_standby_frame(initial_msg)
        with self._frame_condition:
            self._frame_bytes = initial_standby
            self._frame_id += 1
            self._frame_condition.notify_all()

        last_consumed_seq = 0
        infer_fps_buffer = deque(maxlen=30)
        last_infer_time = time.time()

        while self._running:
            with self._raw_frame_lock:
                while self._running and self._raw_frame_seq == last_consumed_seq:
                    self._raw_frame_cond.wait(timeout=0.1)
                if not self._running:
                    break
                frame = self._latest_raw_frame
                last_consumed_seq = self._raw_frame_seq

            if frame is None or not self._camera_enabled:
                time.sleep(0.02)
                continue

            now = time.time()
            self.process_frame(frame, is_mirrored=False, current_time=now)

            dt = now - last_infer_time
            last_infer_time = now
            if dt > 0:
                infer_fps_buffer.append(1.0 / dt)
            smoothed_infer_fps = float(np.mean(infer_fps_buffer)) if infer_fps_buffer else 0.0

            with self._lock:
                self._state["inference_fps"] = round(smoothed_infer_fps, 1)
                self._state["fps"] = round(smoothed_infer_fps, 1)

    def _run(self) -> None:
        """Unified background entrypoint; routes to cloud or worker loop."""
        if self._is_cloud:
            self._run_cloud()
        else:
            self._run_inference_worker()
