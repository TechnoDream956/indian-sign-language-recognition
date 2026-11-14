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
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

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

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

@app.route("/api/<path:subpath>", methods=["OPTIONS"])
def options_cors(subpath):
    return Response(status=204)

engine = InferenceEngine(
    camera_index=0,
)

# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@app.route("/")
def index():
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


@app.route("/video_feed")
def video_feed():
    """MJPEG stream — browser renders as live video via <img src='/video_feed'>."""
    def generate():
        last_frame_id = -1
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
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events")
def events():
    """Server-Sent Events stream — pushes prediction state to browser in real-time."""
    cancel_disconnect_timer()
    def generate():
        last_state = {}
        idle_count = 0
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
    return jsonify(engine.is_ready()), 200


@app.route("/api/camera/frame", methods=["POST"])
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

        return jsonify({
            "ok": True,
            "prediction": res.get("pred_label"),
            "confidence": res.get("confidence"),
            "fused_prediction": res.get("fused_prediction"),
            "fused_confidence": res.get("fused_confidence"),
            "fusion_mode": res.get("fusion_mode"),
            "word": res.get("alphabet_current_word", ""),
            "sentence": res.get("alphabet_sentence", ""),
            "full_text": res.get("alphabet_full_text", ""),
            "translation": res.get("alphabet_translation", ""),
            "hand_detected": res.get("hand_detected", False),
            "hand_count": res.get("alphabet_hand_count", 0),
            "state": res,
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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

