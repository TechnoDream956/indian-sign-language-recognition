"""
Feedback Learning Configuration Module.
========================================
Centralized configuration for Human-in-the-Loop continuous learning,
privacy controls, quality filtering, and model registry.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Workspace and storage paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def is_production_environment() -> bool:
    """
    Detect whether the application is running in a cloud production environment.
    Checks explicit ISL_PRODUCTION, FLASK_ENV, RENDER, or VERCEL flags while
    respecting explicit testing overrides.
    """
    # Force production check when explicitly requested (e.g. security unit tests)
    if os.environ.get("ISL_FORCE_PRODUCTION_CHECK") == "1":
        return True

    # 1. Explicit testing override
    if os.environ.get("ISL_TESTING", "").lower() in ("1", "true", "yes"):
        return False
    if "unittest" in sys.modules:
        return False

    # 2. Production indicators
    if os.environ.get("ISL_PRODUCTION", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("FLASK_ENV") == "production" or os.environ.get("ENVIRONMENT") == "production":
        return True
    if os.environ.get("RENDER") and str(os.environ.get("RENDER")).lower() not in ("0", "false"):
        return True
    return False


# Feedback data directory (configurable for Render Persistent Disk or local storage)
_default_feedback_dir = WORKSPACE_ROOT / "data" / "feedback"
_env_feedback_dir = os.environ.get("FEEDBACK_DATA_DIR", "").strip()
FEEDBACK_DATA_DIR = Path(_env_feedback_dir).resolve() if _env_feedback_dir else _default_feedback_dir.resolve()
DB_PATH = FEEDBACK_DATA_DIR / "feedback.db"
LANDMARKS_STORAGE_DIR = FEEDBACK_DATA_DIR / "landmarks"
GLOBAL_DATASET_DIR = FEEDBACK_DATA_DIR / "global_dataset"
REGISTRY_DIR = FEEDBACK_DATA_DIR / "model_registry"
PERSONAL_STORES_DIR = FEEDBACK_DATA_DIR / "personal_stores"


def ensure_feedback_directories(base_dir: Optional[Path] = None) -> None:
    """
    Ensure that feedback storage base and required subdirectories exist safely
    and verify persistent mount writability.

    Reliability & Security Guarantees:
    - Production and explicitly configured FEEDBACK_DATA_DIR fail-fast with
      clear RuntimeError if directories cannot be created or accessed.
    - Never silently falls back to ephemeral storage if configured storage fails.
    - Tests write access via canary probe to verify volume mount permissions.
    - Local unconfigured development logs clear warnings without crashing.
    """
    target_base = Path(base_dir).resolve() if base_dir else FEEDBACK_DATA_DIR
    is_explicit = bool(base_dir is not None or _env_feedback_dir)
    is_prod = is_production_environment()
    fail_fast = is_prod or is_explicit

    subdirs = [
        target_base,
        target_base / "landmarks",
        target_base / "global_dataset",
        target_base / "model_registry",
        target_base / "personal_stores",
    ]

    for d in subdirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            msg = (
                f"CRITICAL STORAGE ERROR: Cannot create required storage directory '{d.name}'. "
                "Persistent storage initialization failed."
            )
            if fail_fast:
                raise RuntimeError(msg) from e
            else:
                logger.warning("Feedback storage directory warning: %s", msg)
                return

    # Verify write access via probe file
    probe_file = target_base / ".probe_storage_write_test"
    try:
        with open(probe_file, "w") as f:
            f.write("probe")
        probe_file.unlink(missing_ok=True)
    except Exception as e:
        msg = (
            "CRITICAL STORAGE ERROR: Persistent feedback storage mount is not writable. "
            "Check disk mount permissions."
        )
        if fail_fast:
            raise RuntimeError(msg) from e
        else:
            logger.warning("Feedback storage directory warning: %s", msg)


# Ensure directories exist upon configuration import
ensure_feedback_directories()

# Production Model Metadata (Preserved and Non-Negotiable)
PRODUCTION_MODEL_ID = "production_single_hand_mlp_v1"
PRODUCTION_MODEL_PATH = WORKSPACE_ROOT / "models" / "alphabet_model_26class.h5"
PRODUCTION_MODEL_CHECKSUM = "af429c007ba6e62b20385ac6e194898209c3c5cc44f781fc691f34bcb4f295b3"
PRODUCTION_MODEL_VERSION = "1.0.0-alphabet-26class"
PRODUCTION_LABEL_MAP_PATH = WORKSPACE_ROOT / "models" / "alphabet_label_map.json"

# Production Dual-Hand & CNN Model Reference
DUAL_HAND_MODEL_PATH = WORKSPACE_ROOT / "models" / "experimental" / "phase5b2_prompt23_class_k" / "config_d_ablation_144d" / "best_val_model.h5"
DUAL_HAND_MODEL_CHECKSUM = "80b0aae713ad0e57373b252c78ab5d7666ae43387b65c68e822776bc76613d70"
CNN_MODEL_PATH = WORKSPACE_ROOT / "models" / "cnn_sign_model.h5"
CNN_MODEL_CHECKSUM = "be2af2071bc979fb4fb7257b87e7205a063b08a81d914354f1f862dd4648ffb6"

# Feature Schemas
FEATURE_SCHEMA_SINGLE_HAND = "schema_v1_single_hand_63d"
FEATURE_SCHEMA_DUAL_HAND = "schema_v1_dual_hand_126d"
SUPPORTED_SCHEMAS = [FEATURE_SCHEMA_SINGLE_HAND, FEATURE_SCHEMA_DUAL_HAND]

# Valid Target Classes
VALID_ALPHABET_CLASSES: List[str] = [chr(65 + i) for i in range(26)]  # ['A' .. 'Z']

# Snapshot Buffer Configuration
SNAPSHOT_RING_BUFFER_MAX_SIZE = 120
SNAPSHOT_TTL_SECONDS = 60.0  # Retain prediction snapshots for 60 seconds

# Quality Filter Thresholds
MIN_LANDMARK_SCORE = 0.50
MAX_NORMALIZED_COORD_VALUE = 4.0
MIN_SAMPLE_QUALITY_SCORE = 0.60
DUPLICATE_DISTANCE_EPSILON = 0.015

# Personal Learning Configuration
PERSONAL_SIMILARITY_THRESHOLD = 0.88  # Cosine similarity required to consider override
PERSONAL_MIN_SAMPLES_REQUIRED = 2    # Require at least 2 independent corrections for a sign
PERSONAL_CONFIDENCE_WEIGHT = 0.65

# Global Contribution & Privacy
GLOBAL_CONTRIBUTION_DEFAULT_ENABLED = False  # Non-negotiable default
CURRENT_CONSENT_VERSION = "1.0"

# Rate Limiting & Bounds
MAX_FEEDBACK_EVENTS_PER_MINUTE = 60
MAX_STORED_SAMPLES_PER_USER = 1000

# ==============================================================================
# Session Security, Cryptographic Tokens & Production Secret Management
# ==============================================================================

KNOWN_INSECURE_SECRETS: Set[str] = {
    "isl-feedback-secret-2026-auth-token",
    "isl-feedback-secret-default-change-in-production",
    "secret",
    "changeme",
    "default",
    "dev",
    "test",
    "password",
    "123456",
}

DEV_ONLY_FALLBACK_SECRET = "isl-dev-local-only-insecure-secret"


def resolve_feedback_secret_key() -> str:
    """
    Resolve and validate the HMAC-SHA256 secret key for feedback learning tokens.

    Production Guarantees:
    - Production MUST NEVER use a hardcoded fallback secret.
    - If production is detected and FEEDBACK_SECRET_KEY is missing, empty, or insecure,
      it fails closed with a clear RuntimeError at startup.
    - Local development falls back to an explicit dev-only secret without silently
      reusing production keys.
    - The actual secret value is never printed or logged.
    """
    secret = (os.environ.get("FEEDBACK_SECRET_KEY") or os.environ.get("SECRET_KEY") or "").strip()
    is_prod = is_production_environment()

    if is_prod:
        if not secret:
            raise RuntimeError(
                "CRITICAL STARTUP ERROR: FEEDBACK_SECRET_KEY is not configured. "
                "Production deployment requires an explicit, cryptographically secure "
                "FEEDBACK_SECRET_KEY environment variable. Startup aborted."
            )
        if secret in KNOWN_INSECURE_SECRETS or len(secret) < 16:
            raise RuntimeError(
                "CRITICAL STARTUP ERROR: FEEDBACK_SECRET_KEY is known-insecure or too short (<16 chars). "
                "Production deployment requires a strong random secret. Startup aborted."
            )
        return secret

    # Local development or testing environment
    if secret:
        return secret

    return DEV_ONLY_FALLBACK_SECRET


FEEDBACK_SECRET_KEY = resolve_feedback_secret_key()
FEEDBACK_TOKEN_TTL_SECONDS = int(os.environ.get("FEEDBACK_TOKEN_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 days

