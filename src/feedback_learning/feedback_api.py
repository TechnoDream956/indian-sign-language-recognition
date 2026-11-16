"""
Feedback and Continuous Learning API Blueprint (v2).
====================================================
Independent, versioned REST endpoints providing prediction snapshotting,
confirmation/correction feedback, personal adaptation, and privacy controls.
Protected with cryptographically signed HMAC-SHA256 session authentication.
"""

import uuid
from typing import Optional, Tuple
import numpy as np
from flask import Blueprint, jsonify, request, session, make_response

from .feedback_manager import FeedbackManager

feedback_bp = Blueprint("feedback_api_v2", __name__, url_prefix="/api/v2")

# Global singleton manager instance
_feedback_manager: FeedbackManager = None


def get_feedback_manager() -> FeedbackManager:
    """Lazy initialize and return singleton FeedbackManager instance."""
    global _feedback_manager
    if _feedback_manager is None:
        # Cross-check alternate sys.modules path to ensure unified singleton across import styles
        import sys
        for mod_name in ("feedback_learning.feedback_api", "src.feedback_learning.feedback_api"):
            mod = sys.modules.get(mod_name)
            if mod is not None and getattr(mod, "_feedback_manager", None) is not None:
                _feedback_manager = mod._feedback_manager
                return _feedback_manager
        _feedback_manager = FeedbackManager()
    return _feedback_manager


def _get_app_engine():
    """
    Safely retrieve the running InferenceEngine instance without ever re-importing
    or re-executing src.app, which would duplicate model loading into RAM.
    """
    try:
        from inference_engine import InferenceEngine
        inst = InferenceEngine.get_active_instance()
        if inst is not None:
            return inst
    except Exception:
        pass
    import sys
    for mod_name in ("__main__", "app", "src.app"):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "engine", None) is not None:
            return mod.engine
    return None


def _extract_token_from_request() -> Optional[str]:
    """Extract session token from header, Authorization header, or flask session."""
    # 1. Custom explicit header
    token = request.headers.get("X-Session-Token")
    if token and token.strip():
        return token.strip()

    # 2. Authorization Bearer token
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.strip().lower().startswith("bearer "):
        return auth_header.strip()[7:].strip()

    # 3. Flask session cookie fallback
    token_sess = session.get("feedback_session_token")
    if token_sess and str(token_sess).strip():
        return str(token_sess).strip()

    return None


def _authenticate_session(require_token: bool = True):
    """
    Authenticate request against cryptographically signed session token.
    Detects and blocks spoofed or mismatched user identifiers.
    Returns: (user_id, session_token, error_response_tuple_or_None)
    """
    manager = get_feedback_manager()
    token = _extract_token_from_request()

    if not token:
        if require_token:
            return None, None, (
                jsonify({
                    "ok": False,
                    "error": "Authentication required: Missing signed session token. "
                             "Provide 'X-Session-Token' or 'Authorization: Bearer <token>' header.",
                }),
                401,
            )
        else:
            # Check if engine or flask session already has an established user_id
            active_uid = None
            try:
                engine = _get_app_engine()
                if engine is not None and hasattr(engine, "get_session_user_id"):
                    active_uid = engine.get_session_user_id()
            except Exception:
                pass
            if not active_uid:
                active_uid = session.get("user_id") or f"usr_{uuid.uuid4().hex[:12]}"

            new_token = manager.privacy_manager.generate_session_token(active_uid)
            session["feedback_session_token"] = new_token
            session["user_id"] = active_uid
            return active_uid, new_token, None

    # Validate provided token
    is_valid, token_uid, err_msg = manager.privacy_manager.verify_session_token(token)
    if not is_valid:
        return None, None, (
            jsonify({
                "ok": False,
                "error": f"Unauthorized: {err_msg}",
            }),
            401,
        )

    # Cross-check against client-supplied user_id in headers, query, or body (spoofing detection)
    header_uid = request.headers.get("X-User-ID")
    if header_uid and header_uid.strip() and header_uid.strip() != token_uid:
        return None, None, (
            jsonify({
                "ok": False,
                "error": "Forbidden: User identity does not match authenticated session.",
            }),
            403,
        )

    query_uid = request.args.get("user_id")
    if query_uid and query_uid.strip() and query_uid.strip() != token_uid:
        return None, None, (
            jsonify({
                "ok": False,
                "error": "Forbidden: User identity does not match authenticated session.",
            }),
            403,
        )

    if request.is_json:
        body = request.get_json(silent=True) or {}
        body_uid = body.get("user_id")
        if body_uid and str(body_uid).strip() and str(body_uid).strip() != token_uid:
            return None, None, (
                jsonify({
                    "ok": False,
                    "error": "Forbidden: User identity does not match authenticated session.",
                }),
                403,
            )

    return token_uid, token, None


# ------------------------------------------------------------------ #
# Session Provisioning & Synchronization                             #
# ------------------------------------------------------------------ #

@feedback_bp.route("/session", methods=["GET", "POST"])
def get_or_create_session():
    """
    Establish, verify, or refresh cryptographically signed HMAC-SHA256 session token.
    Ensures frontend has an authenticated session and synchronizes engine session.
    """
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=False)
    if err_tuple:
        return err_tuple

    # Synchronize engine session in local environment
    try:
        engine = _get_app_engine()
        if engine is not None and hasattr(engine, "set_session_user_id"):
            old_uid = engine.get_session_user_id()
            engine.set_session_user_id(user_id)
            if old_uid and old_uid != user_id:
                manager.reassign_snapshots(old_uid, user_id)
    except Exception:
        pass

    resp = make_response(
        jsonify({
            "ok": True,
            "user_id": user_id,
            "session_token": token,
        }),
        200,
    )
    resp.headers["X-Session-Token"] = token
    return resp


# ------------------------------------------------------------------ #
# Feedback Endpoints                                                 #
# ------------------------------------------------------------------ #

@feedback_bp.route("/feedback/prediction", methods=["POST"])
def register_prediction():
    """
    Register or record a prediction snapshot.
    If no token is supplied, issues a new anonymous session token.
    Body:
      predicted_alphabet: str (e.g. "A")
      confidence_score: float
      feature_vector: list of floats (63D)
      feature_schema: str (optional)
      raw_landmarks: list (optional)
      tracking_quality: dict (optional)
      metadata: dict (optional)
    """
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=False)
    if err_tuple:
        return err_tuple

    data = request.get_json(silent=True) or {}
    pred_alpha = data.get("predicted_alphabet", "")
    conf = float(data.get("confidence_score", 0.0))
    raw_feats = data.get("feature_vector", [])

    if not raw_feats:
        return jsonify({"ok": False, "error": "Missing 'feature_vector' array"}), 400

    feats = np.array(raw_feats, dtype=np.float32)
    schema = data.get("feature_schema", "schema_v1_single_hand_63d")

    snapshot = manager.capture_prediction_snapshot(
        predicted_alphabet=pred_alpha,
        confidence_score=conf,
        feature_vector=feats,
        feature_schema=schema,
        raw_landmarks=data.get("raw_landmarks"),
        tracking_quality=data.get("tracking_quality"),
        metadata=data.get("metadata"),
        user_id=user_id,
    )

    resp = make_response(
        jsonify({
            "ok": True,
            "prediction_id": snapshot.prediction_id,
            "predicted_alphabet": snapshot.predicted_alphabet,
            "confidence_score": snapshot.confidence_score,
            "feature_schema": snapshot.feature_schema,
            "timestamp": snapshot.timestamp,
            "user_id": user_id,
            "session_token": token,
        }),
        201,
    )
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/feedback/confirm", methods=["POST"])
def confirm_prediction():
    """
    User confirmed prediction was correct ("Yes").
    Body:
      prediction_id: str
    """
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    data = request.get_json(silent=True) or {}
    pid = data.get("prediction_id")
    if not pid:
        return jsonify({"ok": False, "error": "Missing 'prediction_id'"}), 400

    res = manager.confirm_prediction(prediction_id=pid, user_id=user_id)
    if not res.get("ok", False):
        if str(res.get("error", "")).startswith("Forbidden:"):
            return jsonify(res), 403
        return jsonify(res), 400

    resp = make_response(jsonify(res), 200)
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/feedback/correct", methods=["POST"])
def correct_prediction():
    """
    User corrected an incorrect prediction ("No" -> actual letter).
    Body:
      prediction_id: str
      actual_label: str (single letter A-Z)
    """
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    data = request.get_json(silent=True) or {}
    pid = data.get("prediction_id")
    actual = data.get("actual_label")

    if not pid:
        return jsonify({"ok": False, "error": "Missing 'prediction_id'"}), 400
    if not actual:
        return jsonify({"ok": False, "error": "Missing 'actual_label'"}), 400

    success, res = manager.correct_prediction(
        prediction_id=pid,
        actual_label=str(actual).strip().upper(),
        user_id=user_id,
    )

    if not success:
        if str(res.get("error", "")).startswith("Forbidden:"):
            return jsonify(res), 403
        return jsonify(res), 400

    resp = make_response(jsonify(res), 200)
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/feedback/history", methods=["GET"])
def feedback_history():
    """Retrieve recent feedback history for current user."""
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    limit = min(100, int(request.args.get("limit", 50)))
    history = manager.get_history(user_id=user_id, limit=limit)
    resp = make_response(
        jsonify({
            "ok": True,
            "user_id": user_id,
            "count": len(history),
            "history": history,
        }),
        200,
    )
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """Return learning and review statistics for current user."""
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    stats = manager.get_stats(user_id=user_id)
    resp = make_response(
        jsonify({
            "ok": True,
            "user_id": user_id,
            "stats": stats,
        }),
        200,
    )
    resp.headers["X-Session-Token"] = token
    return resp


# ------------------------------------------------------------------ #
# Learning & Preferences Endpoints                                   #
# ------------------------------------------------------------------ #

@feedback_bp.route("/learning/preferences", methods=["GET", "PUT"])
def learning_preferences():
    """Get or update user learning preferences."""
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    if request.method == "GET":
        pref = manager.get_preferences(user_id=user_id)
        resp = make_response(
            jsonify({
                "ok": True,
                "user_id": user_id,
                "personalization_enabled": bool(pref.get("personalization_enabled", True)),
                "global_contribution_enabled": bool(pref.get("global_contribution_enabled", False)),
                "consent_version": pref.get("consent_version", "1.0"),
            }),
            200,
        )
        resp.headers["X-Session-Token"] = token
        return resp

    data = request.get_json(silent=True) or {}
    pers = data.get("personalization_enabled")
    glob = data.get("global_contribution_enabled")

    updated = manager.update_preferences(
        user_id=user_id,
        personalization_enabled=bool(pers) if pers is not None else None,
        global_contribution_enabled=bool(glob) if glob is not None else None,
    )

    resp = make_response(
        jsonify({
            "ok": True,
            "user_id": user_id,
            "personalization_enabled": bool(updated.get("personalization_enabled", True)),
            "global_contribution_enabled": bool(updated.get("global_contribution_enabled", False)),
            "message": "Preferences updated successfully.",
        }),
        200,
    )
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/learning/personalize", methods=["POST"])
def personalize_prediction():
    """
    Evaluate feature vector against user personal store to adapt prediction.
    Body:
      feature_vector: list of floats (63D)
      production_prediction: str (e.g. "A")
      production_confidence: float
    """
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    data = request.get_json(silent=True) or {}
    raw_feats = data.get("feature_vector", [])
    if not raw_feats:
        return jsonify({"ok": False, "error": "Missing 'feature_vector'"}), 400

    feats = np.array(raw_feats, dtype=np.float32)
    prod_pred = data.get("production_prediction")
    prod_conf = float(data.get("production_confidence", 0.0))

    res = manager.adapt_prediction(
        user_id=user_id,
        features=feats,
        production_prediction=prod_pred,
        production_confidence=prod_conf,
    )
    resp = make_response(jsonify(res), 200)
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/learning/personal-data", methods=["DELETE"])
def reset_personal_data():
    """Delete all personal learning data and withdraw contributions for current user."""
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    res = manager.reset_personal_data(user_id=user_id)
    resp = make_response(jsonify(res), 200)
    resp.headers["X-Session-Token"] = token
    return resp


@feedback_bp.route("/learning/global-consent", methods=["POST", "DELETE"])
def manage_global_consent():
    """Opt-in (POST) or withdraw (DELETE) global dataset contribution consent."""
    manager = get_feedback_manager()
    user_id, token, err_tuple = _authenticate_session(require_token=True)
    if err_tuple:
        return err_tuple

    if request.method == "POST":
        res = manager.global_contribution.grant_consent(user_id=user_id)
        resp = make_response(jsonify(res), 200)
        resp.headers["X-Session-Token"] = token
        return resp

    # DELETE -> Withdraw consent
    res = manager.global_contribution.withdraw_consent(user_id=user_id, withdraw_existing_samples=True)
    resp = make_response(jsonify(res), 200)
    resp.headers["X-Session-Token"] = token
    return resp
