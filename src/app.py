#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask application for Indian Sign Language (ISL) Alphabet & Fingerspelling Translator.

Routes:
  GET  /             → Single-page application (index.html)
  GET  /video_feed   → MJPEG camera stream
  GET  /events       → Server-Sent Events (SSE) — real-time prediction updates
  POST /api/language → Switch active language  { "language": "Hindi" }
  POST /api/speak    → Manually trigger TTS for current prediction
  GET  /api/status   → Health check JSON
"""

import base64
import json
import os
import sys
import threading
import time
import uuid
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, render_template_string, request, session, make_response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_engine import InferenceEngine
from translator import LANGUAGE_DISPLAY_NAMES, NATIVE_LANGUAGE_NAMES

# ------------------------------------------------------------------ #
#  App & engine initialisation                                         #
# ------------------------------------------------------------------ #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "isl-translator-2025"

# Register Feedback and Continuous Learning Blueprint (v2)
try:
    from feedback_learning import feedback_bp
    app.register_blueprint(feedback_bp)
except ImportError:
    from src.feedback_learning import feedback_bp
    app.register_blueprint(feedback_bp)

# Register Independent Multilingual Translation & Speech Blueprint (v2)
try:
    from language_services.api import language_bp
    app.register_blueprint(language_bp)
except ImportError:
    try:
        from src.language_services.api import language_bp
        app.register_blueprint(language_bp)
    except Exception as _lang_bp_err:
        pass

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, X-User-ID, X-Session-ID, X-Session-Token"
    response.headers["Access-Control-Expose-Headers"] = "X-Session-Token"
    return response

@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def options_cors(subpath):
    return Response(status=204)

engine = InferenceEngine(
    camera_index=0,
)
_default_local_user_id = f"usr_{uuid.uuid4().hex[:12]}"
engine.set_session_user_id(_default_local_user_id)

# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    if "user_id" not in session:
        session["user_id"] = engine.get_session_user_id() or _default_local_user_id
    if "feedback_session_token" not in session:
        try:
            from feedback_learning import get_feedback_manager
            fb_mgr = get_feedback_manager()
            session["feedback_session_token"] = fb_mgr.privacy_manager.generate_session_token(session["user_id"])
        except Exception:
            pass

    languages = list(LANGUAGE_DISPLAY_NAMES.keys())
    language_display = LANGUAGE_DISPLAY_NAMES
    native_languages = NATIVE_LANGUAGE_NAMES
    models = engine.get_available_models()
    active_model = engine.get_active_model()
    active_mode = engine.get_app_mode()
    return render_template(
        "index.html",
        languages=languages,
        language_display=language_display,
        native_languages=native_languages,
        models=models,
        active_model=active_model,
        active_mode=active_mode,
    )


@app.route("/language-test")
def language_test():
    """Standalone multilingual translation & neural TTS test interface (no webcam required)."""
    try:
        from language_services.test_ui import TEST_PAGE_HTML
    except ImportError:
        from src.language_services.test_ui import TEST_PAGE_HTML
    return render_template_string(TEST_PAGE_HTML)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — browser renders as live video via <img src='/video_feed'>."""
    def generate():
        engine._has_video_subscribers = True
        last_frame_id = -1
        try:
            while engine._running:
                frame_id, frame = engine.wait_for_frame(last_frame_id=last_frame_id, timeout=0.1)
                if frame is not None and frame_id != last_frame_id:
                    last_frame_id = frame_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                elif not engine._running:
                    break
        finally:
            engine._has_video_subscribers = False
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events", methods=["GET", "OPTIONS"])
def events():
    """
    Authenticated Server-Sent Events stream — pushes real-time prediction state.
    Requires cryptographically signed session token via 'X-Session-Token' or 'Authorization' header.
    Rejects unauthorized, missing, or expired tokens with HTTP 401.
    """
    if request.method == "OPTIONS":
        return Response(status=204)

    cancel_disconnect_timer()

    # Extract session token strictly from request headers or flask session cookie (never from URL query)
    req_token = (
        request.headers.get("X-Session-Token")
        or request.headers.get("Authorization")
        or session.get("feedback_session_token")
    )
    if req_token and req_token.strip().lower().startswith("bearer "):
        req_token = req_token.strip()[7:].strip()
    if req_token:
        req_token = req_token.strip()

    if not req_token:
        return jsonify({
            "ok": False,
            "error": "Authentication required: Missing session token. Provide 'X-Session-Token' header.",
        }), 401

    try:
        from feedback_learning import get_feedback_manager
        fb_mgr = get_feedback_manager()
        valid, token_uid, err_msg = fb_mgr.privacy_manager.verify_session_token(req_token)
        if not valid or not token_uid:
            return jsonify({
                "ok": False,
                "error": f"Unauthorized: {err_msg or 'Invalid session token'}",
            }), 401

        old_uid = engine.get_session_user_id()
        engine.set_session_user_id(token_uid)
        if old_uid and old_uid != token_uid:
            try:
                fb_mgr.reassign_snapshots(old_uid, token_uid)
            except Exception:
                pass
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Authentication validation failed: {str(e)}",
        }), 401

    def generate():
        last_state = {}
        idle_count = 0
        try:
            while engine._running:
                state = engine.get_latest_state()
                if state != last_state:
                    last_state = state
                    data = json.dumps(state)
                    yield f"data: {data}\n\n"
                    idle_count = 0
                else:
                    idle_count += 1
                    if idle_count >= 15:  # Every ~1.2s when idle
                        idle_count = 0
                        yield ": ping\n\n"
                time.sleep(0.08)  # ~12 Hz update rate — smooth for text, low CPU
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            pass
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/language", methods=["GET", "POST"])
def language_endpoint():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        language = data.get("language", "English")
        if language in LANGUAGE_DISPLAY_NAMES:
            engine.set_language(language)
            return jsonify({"ok": True, "language": language})
        return jsonify({"ok": False, "error": "Unknown language"}), 400
    state = engine.get_latest_state()
    current_lang = state.get("language", "English")
    return jsonify({
        "ok": True,
        "language": current_lang,
        "display_name": LANGUAGE_DISPLAY_NAMES.get(current_lang, current_lang),
        "supported_languages": list(LANGUAGE_DISPLAY_NAMES.keys())
    })


@app.route("/api/speak", methods=["POST"])
def speak():
    triggered = engine.speak_current()
    return jsonify({"ok": True, "triggered": triggered})


@app.route("/api/speech/toggle", methods=["POST"])
def toggle_speech():
    res = engine.toggle_speech()
    return jsonify(res)


@app.route("/api/camera/toggle", methods=["POST"])
def toggle_camera():
    res = engine.toggle_camera()
    return jsonify(res)


@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    if not engine._running:
        engine.start()
    res = engine.start_camera()
    return jsonify(res)


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    res = engine.stop_camera()
    return jsonify(res)


@app.route("/api/fusion/status", methods=["GET"])
def fusion_status():
    st = engine.get_latest_state()
    return jsonify({
        "ok": True,
        "landmark_prediction": st.get("landmark_prediction"),
        "landmark_confidence": st.get("landmark_confidence"),
        "landmark_model": st.get("landmark_model"),
        "cnn_prediction": st.get("cnn_prediction"),
        "cnn_confidence": st.get("cnn_confidence"),
        "cnn_status": st.get("cnn_status"),
        "fused_prediction": st.get("fused_prediction"),
        "fused_confidence": st.get("fused_confidence"),
        "fusion_mode": st.get("fusion_mode"),
        "fusion_reason": st.get("fusion_reason"),
    })


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return all available models with metadata."""
    return jsonify({
        "models": engine.get_available_models(),
        "active_model": engine.get_active_model(),
    })


@app.route("/api/model", methods=["GET", "POST"])
def active_model():
    """Get active model configuration or switch to a new model."""
    if request.method == "GET":
        return jsonify({
            "active_model": engine.get_active_model()
        })

    data = request.get_json(silent=True) or {}
    model_id = data.get("model")
    if not model_id:
        return jsonify({"success": False, "error": "Missing 'model' field"}), 400

    try:
        res = engine.set_model(model_id)
        return jsonify({
            "success": True,
            "active_model": engine.get_active_model(),
            "details": res,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/mode", methods=["GET", "POST"])
def app_mode():
    """Get or switch active application mode ('word' vs 'alphabet')."""
    if request.method == "GET":
        return jsonify({
            "mode": engine.get_app_mode()
        })

    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if not mode:
        return jsonify({"success": False, "error": "Missing 'mode' field"}), 400

    try:
        res = engine.set_app_mode(mode)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ------------------------------------------------------------------ #
#  Alphabet Mode & WordBuffer APIs                                     #
# ------------------------------------------------------------------ #

@app.route("/api/alphabet/state", methods=["GET"])
def alphabet_state():
    """Return snapshot of current alphabet recognition and word buffer state."""
    return jsonify(engine.get_alphabet_state())


@app.route("/api/alphabet/space", methods=["POST"])
def alphabet_space():
    """Commit current word and append space to sentence."""
    return jsonify(engine.alphabet_add_space())


@app.route("/api/alphabet/backspace", methods=["POST"])
def alphabet_backspace():
    """Delete last character from current word or sentence."""
    return jsonify(engine.alphabet_backspace())


@app.route("/api/alphabet/clear-word", methods=["POST"])
def alphabet_clear_word():
    """Clear current word buffer."""
    return jsonify(engine.alphabet_clear_word())


@app.route("/api/alphabet/clear-sentence", methods=["POST"])
def alphabet_clear_sentence():
    """Clear accumulated sentence buffer."""
    return jsonify(engine.alphabet_clear_sentence())


@app.route("/api/alphabet/clear-all", methods=["POST"])
def alphabet_clear_all():
    """Clear both current word and accumulated sentence."""
    return jsonify(engine.alphabet_clear_all())


@app.route("/api/alphabet/commit", methods=["POST"])
def alphabet_commit():
    """Commit current word to sentence. Optionally accepts {'word': '...'} or {'sentence': '...'}."""
    data = request.get_json(silent=True) or {}
    word = data.get("word")
    sentence = data.get("sentence")
    if engine._alphabet_engine is not None:
        if sentence is not None:
            with engine._lock:
                engine._alphabet_engine.word_buffer.sentence = str(sentence).strip() + " "
                engine._alphabet_engine.word_buffer.current_word = ""
                engine._sync_alphabet_state_locked()
            return jsonify(engine.get_alphabet_state())
        elif word is not None:
            with engine._lock:
                engine._alphabet_engine.word_buffer.current_word = str(word).strip().upper()
    return jsonify(engine.alphabet_commit_word())


@app.route("/api/alphabet/commit-letter", methods=["POST"])
def alphabet_commit_letter():
    """Manually append a single recognized letter to current word buffer."""
    data = request.get_json(silent=True) or {}
    letter = data.get("letter", "").strip()
    if not letter:
        return jsonify({"ok": False, "error": "No letter provided"}), 400
    res = engine.alphabet_add_letter(letter)
    return jsonify(res)


@app.route("/api/alphabet/speak", methods=["POST"])
def alphabet_speak():
    """Speak current sentence/text via TTS (original English or translated)."""
    data = request.get_json(silent=True) or {}
    speak_type = data.get("type", "original")
    if speak_type == "translated":
        triggered = engine.alphabet_speak_translation()
    else:
        triggered = engine.alphabet_speak(language="English")
    return jsonify({"ok": True, "triggered": bool(triggered)})


@app.route("/api/alphabet/translation", methods=["GET"])
def alphabet_translation():
    """Return English source sentence, selected language, and translated sentence."""
    return jsonify(engine.alphabet_get_translation())


@app.route("/api/alphabet/speak-translation", methods=["POST"])
def alphabet_speak_translation():
    """Speak translated sentence in the target language via TTS."""
    triggered = engine.alphabet_speak_translation()
    return jsonify({"ok": True, "triggered": bool(triggered)})


@app.route("/api/tts/voices", methods=["GET"])
def tts_voices():
    """Return voice catalog and availability status across all supported languages."""
    from tts_engine import TTSEngine
    tts = TTSEngine()
    return jsonify({
        "ok": True,
        "catalog": tts.get_voice_catalog(),
    })


@app.route("/api/tts/speak", methods=["POST"])
def tts_speak():
    """
    Synthesize text to speech using the neural speech adapter if enabled,
    or instruct the client to use browser Web Speech API fallback.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    language = data.get("language", data.get("target_lang", "English"))
    voice_id = data.get("voice_id")

    try:
        from speech_adapter import get_speech_adapter
    except ImportError:
        from src.speech_adapter import get_speech_adapter

    adapter = get_speech_adapter()
    res = adapter.synthesize(text=text, target_lang=language, voice_id=voice_id)

    return jsonify({
        "ok": True,
        **res,
    })


@app.route("/api/alphabet/reset", methods=["POST"])
def alphabet_reset():
    """Reset alphabet debounce state machine."""
    return jsonify(engine.alphabet_reset())


# ------------------------------------------------------------------ #
#  WordBuffer Route Aliases                                            #
# ------------------------------------------------------------------ #

@app.route("/api/buffer/space", methods=["POST"])
def buffer_space():
    return jsonify(engine.alphabet_add_space())


@app.route("/api/buffer/backspace", methods=["POST"])
def buffer_backspace():
    return jsonify(engine.alphabet_backspace())


@app.route("/api/buffer/commit", methods=["POST"])
def buffer_commit():
    return jsonify(engine.alphabet_commit_word())


@app.route("/api/buffer/clear_word", methods=["POST"])
def buffer_clear_word():
    return jsonify(engine.alphabet_clear_word())


@app.route("/api/buffer/clear_sentence", methods=["POST"])
def buffer_clear_sentence():
    return jsonify(engine.alphabet_clear_sentence())


@app.route("/api/buffer/clear_all", methods=["POST"])
def buffer_clear_all():
    return jsonify(engine.alphabet_clear_all())


@app.route("/api/buffer/reset", methods=["POST"])
def buffer_reset():
    return jsonify(engine.alphabet_reset())


# ------------------------------------------------------------------ #
#  Application Lifecycle & Shutdown Coordinator                        #
# ------------------------------------------------------------------ #

_disconnect_timer: threading.Timer | None = None
_disconnect_lock = threading.Lock()


def cancel_disconnect_timer() -> None:
    """Cancel pending disconnect timer when client reconnects or pings."""
    global _disconnect_timer
    with _disconnect_lock:
        if _disconnect_timer is not None:
            _disconnect_timer.cancel()
            _disconnect_timer = None


def _perform_graceful_shutdown() -> None:
    """Execute clean, idempotent shutdown of engine, camera, and server."""
    time.sleep(0.3)
    try:
        engine.stop()
    except Exception as e:
        print(f"[Shutdown Warning] Engine stop exception: {e}", flush=True)
    if os.environ.get("ISL_PACKAGED") == "1":
        os._exit(0)
    else:
        print("[Application Lifecycle] Camera released and engine stopped. Server standing for demo.", flush=True)


@app.route("/api/shutdown", methods=["POST"])
def shutdown_app():
    """Explicit application quit requested by UI."""
    if engine.is_cloud or os.environ.get("RENDER"):
        return jsonify({"ok": True, "message": "Cloud session disconnected (shared server preserved)."}), 200
    print("\n[Application Lifecycle] /api/shutdown requested. Halting...", flush=True)
    threading.Thread(target=_perform_graceful_shutdown, daemon=True, name="ShutdownThread").start()
    return jsonify({"ok": True, "message": "Application shutting down and camera released."})


@app.route("/api/client-disconnect", methods=["POST"])
def client_disconnect():
    """Beacon sent on page unload. In packaged app mode, triggers shutdown after grace period."""
    global _disconnect_timer
    with _disconnect_lock:
        if _disconnect_timer is not None:
            _disconnect_timer.cancel()
            _disconnect_timer = None
        # Only auto-shutdown if running as standalone packaged desktop app
        if os.environ.get("ISL_PACKAGED") == "1":
            _disconnect_timer = threading.Timer(4.0, _perform_graceful_shutdown)
            _disconnect_timer.daemon = True
            _disconnect_timer.start()
    return jsonify({"ok": True})


@app.route("/api/heartbeat", methods=["GET", "POST"])
def heartbeat():
    """Client heartbeat to indicate active session and keep server alive."""
    cancel_disconnect_timer()
    return jsonify({"ok": True, "alive": True})


@app.route("/api/healthz", methods=["GET"])
def healthz():
    """Lightweight health check endpoint for cloud orchestrators (Render, Docker, Kubernetes)."""
    cancel_disconnect_timer()
    return jsonify({
        "status": "healthy",
        "camera_mode": "cloud_browser" if engine.is_cloud else "local_hardware",
        "cloud_mode": engine.is_cloud,
        "ready": engine._running,
        "model_ok": engine._state.get("model_ok", False),
    }), 200


@app.route("/api/status")
def status():
    cancel_disconnect_timer()
    st = engine.is_ready()
    try:
        import resource
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = round(raw_rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
    except Exception:
        rss_mb = 0.0
    st["memory_rss_mb"] = rss_mb
    st["active_engines"] = getattr(engine.__class__, "_instance_count", 1)
    return jsonify(st), 200


@app.route("/api/camera/frame", methods=["POST"])
@app.route("/api/process_frame", methods=["POST"])
def camera_frame():
    """
    Receive webcam frame from browser client (for cloud deployments or browser webcam).
    Accepts:
      1. JSON: {"image": "data:image/jpeg;base64,...", "mirrored": false} or {"frame": "base64..."}
      2. Form-data: file/blob in 'frame' or 'image' field
      3. Binary raw image/jpeg body
    """
    try:
        frame = None
        content_type = request.content_type or ""
        is_mirrored = False

        if "application/json" in content_type:
            data = request.get_json(silent=True) or {}
            b64_str = data.get("image") or data.get("frame") or data.get("data") or data.get("blob")
            is_mirrored = bool(data.get("mirrored", False))
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            uploaded = request.files.get("frame") or request.files.get("image") or request.files.get("file")
            if uploaded:
                img_bytes = uploaded.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if request.form.get("mirrored") in ("1", "true", "True"):
                is_mirrored = True
        else:
            raw_bytes = request.get_data()
            if raw_bytes:
                nparr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({
                "ok": False,
                "error": "Invalid or empty image frame. Provide binary JPEG, multipart file, or base64 JSON."
            }), 400

        res = engine.process_frame(frame, is_mirrored=is_mirrored)
        cancel_disconnect_timer()

        # Safely record prediction snapshot for feedback learning
        prediction_id = None
        session_token = None
        try:
            from feedback_learning import get_feedback_manager
            fb_mgr = get_feedback_manager()
            user_token = request.headers.get("X-Session-Token")
            auth_user_id = None
            if user_token:
                valid, token_uid, _ = fb_mgr.privacy_manager.verify_session_token(user_token)
                if valid:
                    auth_user_id = token_uid
                    session_token = user_token
            if not auth_user_id:
                auth_user_id = session.get("user_id") or f"usr_{uuid.uuid4().hex[:12]}"
                session["user_id"] = auth_user_id
                session_token = fb_mgr.privacy_manager.generate_session_token(auth_user_id)
                session["feedback_session_token"] = session_token
            if hasattr(engine, "set_session_user_id"):
                engine.set_session_user_id(auth_user_id)

            pred_letter = res.get("fused_prediction") or res.get("pred_label") or res.get("alphabet_stable_letter")
            if pred_letter and pred_letter not in ("--", "None", ""):
                landmarks_list = res.get("landmarks", [])
                if landmarks_list and len(landmarks_list) > 0:
                    pts = landmarks_list[0].get("points", [])
                    if len(pts) >= 21:
                        from alphabet_engine import normalize_hand_landmarks
                        feats = normalize_hand_landmarks(np.array(pts[:21], dtype=np.float32))
                        snapshot = fb_mgr.capture_prediction_snapshot(
                            predicted_alphabet=pred_letter,
                            confidence_score=float(res.get("fused_confidence") or res.get("confidence") or 0.0),
                            feature_vector=feats,
                            raw_landmarks=landmarks_list,
                            tracking_quality={"score": landmarks_list[0].get("score", 1.0), "hand_count": res.get("alphabet_hand_count", 1)},
                            metadata={"fusion_mode": res.get("fusion_mode"), "frame_id": res.get("frame_id")},
                            user_id=auth_user_id,
                        )
                        prediction_id = snapshot.prediction_id
        except Exception:
            pass

        resp_data = {
            "ok": True,
            "prediction": res.get("pred_label"),
            "confidence": res.get("confidence"),
            "fused_prediction": res.get("fused_prediction"),
            "fused_confidence": res.get("fused_confidence"),
            "fusion_mode": res.get("fusion_mode"),
            "prediction_id": prediction_id,
            "session_token": session_token,
            "word": res.get("alphabet_current_word", ""),
            "sentence": res.get("alphabet_sentence", ""),
            "full_text": res.get("alphabet_full_text", ""),
            "translation": res.get("alphabet_translation", ""),
            "hand_detected": res.get("hand_detected", False),
            "hand_count": res.get("alphabet_hand_count", 0),
            "landmarks": res.get("landmarks", []),
            "last_stable_letter": res.get("last_stable_letter", "--"),
            "stability_progress": res.get("stability_progress", 0.0),
            "frame_id": res.get("frame_id", 0),
            "processing_time_ms": res.get("processing_time_ms", 0.0),
            "state": res,
        }
        resp = make_response(jsonify(resp_data), 200)
        if session_token:
            resp.headers["X-Session-Token"] = session_token
        return resp
    except Exception as e:
        app.logger.error("Error processing camera frame: %s", e)
        return jsonify({"ok": False, "error": "Internal error processing camera frame"}), 500


@app.route("/api/translate", methods=["POST"])
def translate_endpoint():
    """Translate arbitrary text or current composed sentence into target language."""
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    lang = data.get("language") or engine._state.get("language", "English")
    if text is None:
        cur_sent = engine._state.get("alphabet_sentence", "").strip()
        cur_word = engine._state.get("alphabet_current_word", "").strip()
        text = (cur_sent + " " + cur_word).strip() if cur_sent else cur_word

    from translation_adapter import get_translation_adapter
    adapter = get_translation_adapter()
    res = adapter.translate(text=text, target_lang=lang)
    trans = res.get("translation", "")

    with engine._lock:
        engine._state["alphabet_translation"] = trans

    return jsonify({
        "ok": True,
        "source": text,
        "language": lang,
        "translation": trans,
        "engine": res.get("engine", "legacy"),
        "cached": res.get("cached", False),
    }), 200


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def open_browser(port):
    if engine.is_cloud or os.environ.get("ISL_NO_BROWSER") or os.environ.get("RENDER"):
        return
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    import socket
    host = os.environ.get("HOST", "0.0.0.0")
    port_env = os.environ.get("PORT")

    if port_env:
        port = int(port_env)
    else:
        port = 5001
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 5000))
            port = 5000
        except OSError:
            port = 5001
        finally:
            sock.close()

    print("\n🤟  ISL Translator — Professional UI")
    print("     Starting inference engine...")
    engine.start()

    # Open browser after a short delay (local only)
    if not engine.is_cloud and not os.environ.get("ISL_NO_BROWSER"):
        browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
        browser_thread.start()

    print(f"     Server: http://{host}:{port}")
    print(f"     Mode:   {'Cloud (Browser Webcam Ingestion)' if engine.is_cloud else 'Local (Hardware Webcam)'}")
    print("     Press Ctrl+C to quit.\n")

    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\n✓ Shutdown complete.")

