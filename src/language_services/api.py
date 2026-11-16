#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REST API Blueprint for Version 2 Language Subsystem.
Exposes versioned endpoints:
  POST /api/v2/language/translate
  POST /api/v2/language/speak
  GET  /api/v2/language/voices
  GET  /api/v2/language/health
  GET  /api/v2/language/audio/<audio_id>
  GET  /api/v2/language/languages
  GET  /api/v2/language/test
"""

import io
import logging
import os
import sys

# Support direct script execution as well as package imports
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from language_services.config import default_config
    from language_services.service import get_language_service
else:
    from .config import default_config
    from .service import get_language_service

from flask import Blueprint, Flask, Response, abort, jsonify, render_template_string, request, send_file

logger = logging.getLogger(__name__)

language_bp = Blueprint("language_services_v2", __name__)


@language_bp.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@language_bp.route("/api/v2/language/translate", methods=["POST", "OPTIONS"])
def translate_endpoint():
    """
    POST /api/v2/language/translate
    Body:
    {
        "text": "Hello, how are you?",
        "target_lang": "Hindi" | "hi" | "hin_Deva",
        "source_lang": "en" (optional)
    }
    """
    if request.method == "OPTIONS":
        return Response(status=204)

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    target_lang = data.get("target_lang", "Hindi")
    source_lang = data.get("source_lang", "en")

    if not text or not str(text).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'text' field",
            "source_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translated_text": "",
        }), 400

    service = get_language_service()
    result = service.translate(
        text=str(text),
        target_lang=str(target_lang),
        source_lang=str(source_lang),
    )

    status_code = 200 if result.success else 422
    return jsonify(result.to_dict()), status_code


@language_bp.route("/api/v2/language/speak", methods=["POST", "OPTIONS"])
def speak_endpoint():
    """
    POST /api/v2/language/speak
    Body:
    {
        "text": "नमस्ते, आप कैसे हैं?",
        "target_lang": "Hindi" | "hi",
        "voice_id": "mms-hin-vits" (optional)
    }
    """
    if request.method == "OPTIONS":
        return Response(status=204)

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    target_lang = data.get("target_lang", "Hindi")
    voice_id = data.get("voice_id")

    if not text or not str(text).strip():
        return jsonify({
            "success": False,
            "error": "Missing or empty 'text' field",
            "audio_id": None,
            "data_uri": None,
        }), 400

    service = get_language_service()
    res = service.speak(
        text=str(text),
        target_lang=str(target_lang),
        voice_id=voice_id,
    )

    status_code = 200 if res.get("success") else 422
    return jsonify(res), status_code


@language_bp.route("/api/v2/language/voices", methods=["GET"])
def voices_endpoint():
    """
    GET /api/v2/language/voices?lang=Hindi
    Returns catalog of available neural voices.
    """
    lang = request.args.get("lang")
    service = get_language_service()
    voices = service.get_voices(target_lang=lang)
    return jsonify({
        "success": True,
        "total_voices": len(voices),
        "voices": voices,
    })


@language_bp.route("/api/v2/language/languages", methods=["GET"])
def languages_endpoint():
    """
    GET /api/v2/language/languages
    Returns catalog of supported Indian languages.
    """
    service = get_language_service()
    languages = service.get_supported_languages()
    return jsonify({
        "success": True,
        "total_languages": len(languages),
        "languages": languages,
    })


@language_bp.route("/api/v2/language/health", methods=["GET"])
def health_endpoint():
    """
    GET /api/v2/language/health
    Returns health status, telemetry, device and memory stats.
    """
    service = get_language_service()
    health = service.get_health()
    return jsonify(health)


@language_bp.route("/api/v2/language/audio/<audio_id>", methods=["GET"])
def audio_endpoint(audio_id: str):
    """
    GET /api/v2/language/audio/<audio_id>
    Stream or download WAV file.
    """
    service = get_language_service()
    audio_entry = service.audio_manager.get_audio(audio_id)
    if not audio_entry:
        abort(404, description="Audio sample not found or expired")

    audio_bytes, meta = audio_entry
    as_attachment = request.args.get("download", "0") == "1"
    filename = f"{audio_id}.wav"

    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=as_attachment,
        download_name=filename,
    )


# Standalone runner
def create_standalone_app() -> Flask:
    """Create a completely standalone Flask application for testing."""
    app = Flask(__name__)
    app.register_blueprint(language_bp)

    @app.route("/")
    @app.route("/test")
    @app.route("/language-test")
    def test_page():
        try:
            from .test_ui import TEST_PAGE_HTML
        except ImportError:
            from language_services.test_ui import TEST_PAGE_HTML
        return render_template_string(TEST_PAGE_HTML)

    return app


if __name__ == "__main__":
    app = create_standalone_app()
    port = default_config.port
    print(f"🚀 Starting Independent Language Services on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
