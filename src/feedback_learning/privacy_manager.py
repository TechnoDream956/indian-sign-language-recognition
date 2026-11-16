"""
Privacy and Security Manager.
==============================
Enforces strict user isolation, consent verification, data deletion,
audit logging without raw biometric leakage, and data export.
"""

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional, Tuple
from .config import CURRENT_CONSENT_VERSION, FEEDBACK_SECRET_KEY, FEEDBACK_TOKEN_TTL_SECONDS
from .sample_collector import SampleCollector
from .global_contribution import GlobalContributionManager


def generate_session_token(
    user_id: str,
    secret_key: Optional[str] = None,
    timestamp: Optional[int] = None,
) -> str:
    """
    Generate a cryptographically signed HMAC-SHA256 session token.
    Format: <user_id>.<timestamp>.<signature>
    """
    secret = (secret_key or FEEDBACK_SECRET_KEY).encode("utf-8")
    clean_uid = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_")).strip()[:64]
    if not clean_uid:
        clean_uid = "anonymous_local"
    ts = int(timestamp if timestamp is not None else time.time())
    payload = f"{clean_uid}.{ts}".encode("utf-8")
    sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{clean_uid}.{ts}.{sig}"


def verify_session_token(
    token: Optional[str],
    secret_key: Optional[str] = None,
    max_age_seconds: Optional[int] = None,
) -> Tuple[bool, Optional[str], str]:
    """
    Verify signature, expiration, and format of a session token.
    Returns: (is_valid, user_id, error_message)
    """
    if not token or not isinstance(token, str):
        return False, None, "Missing session token"

    token = token.strip()
    parts = token.split(".")
    if len(parts) != 3:
        return False, None, "Malformed session token format"

    token_uid, ts_str, sig = parts

    # Verify timestamp
    try:
        ts = int(ts_str)
    except ValueError:
        return False, None, "Invalid timestamp in session token"

    max_age = max_age_seconds if max_age_seconds is not None else FEEDBACK_TOKEN_TTL_SECONDS
    now = int(time.time())

    # Check expiration
    if (now - ts) > max_age:
        return False, None, f"Session token expired (age: {now - ts}s > ttl: {max_age}s)"

    # Check clock skew (future timestamp beyond 5 minutes)
    if ts > (now + 300):
        return False, None, "Session token timestamp is in the future"

    # Verify HMAC signature using constant-time comparison
    secret = (secret_key or FEEDBACK_SECRET_KEY).encode("utf-8")
    payload = f"{token_uid}.{ts_str}".encode("utf-8")
    expected_sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(sig, expected_sig):
        return False, None, "Invalid session token signature"

    # Sanitize user ID
    clean_uid = "".join(c for c in str(token_uid) if c.isalnum() or c in ("-", "_")).strip()[:64]
    if not clean_uid or clean_uid != token_uid:
        return False, None, "Invalid user identifier characters in session token"

    return True, clean_uid, ""


class PrivacyManager:
    """Oversees user data isolation, consent adherence, and compliance."""

    def __init__(
        self,
        collector: Optional[SampleCollector] = None,
        contribution_manager: Optional[GlobalContributionManager] = None,
        secret_key: Optional[str] = None,
    ):
        self.collector = collector or SampleCollector()
        self.contribution_manager = contribution_manager or GlobalContributionManager(self.collector)
        self.secret_key = secret_key or FEEDBACK_SECRET_KEY

    def generate_session_token(self, user_id: str, timestamp: Optional[int] = None) -> str:
        """Issue signed session token for a user."""
        return generate_session_token(user_id=user_id, secret_key=self.secret_key, timestamp=timestamp)

    def verify_session_token(self, token: Optional[str], max_age_seconds: Optional[int] = None) -> Tuple[bool, Optional[str], str]:
        """Verify signed session token."""
        return verify_session_token(token=token, secret_key=self.secret_key, max_age_seconds=max_age_seconds)


    def verify_user_access(self, requested_user_id: str, session_user_id: str) -> bool:
        """Ensure session user can only access their own data."""
        if not requested_user_id or not session_user_id:
            return False
        return requested_user_id.strip() == session_user_id.strip()

    def sanitize_user_id(self, user_id: Optional[str]) -> str:
        """Validate and sanitize user identity or fallback to safe anonymous identifier."""
        if not user_id:
            return "anonymous_local"
        clean = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_")).strip()
        return clean[:64] if clean else "anonymous_local"

    def can_contribute_globally(self, user_id: str) -> bool:
        """Verify whether user has explicitly opted into global dataset contribution."""
        user = self.collector.get_or_create_user(user_id)
        return bool(user.get("global_contribution_enabled", False))

    def delete_user_account_data(self, user_id: str) -> Dict[str, Any]:
        """
        Complete Right-to-be-Forgotten erasure:
        1. Withdraws all global contributions
        2. Erases personal corrections, prediction events, and feature files
        3. Removes user record
        """
        clean_uid = self.sanitize_user_id(user_id)

        # 1. Withdraw global contributions
        self.contribution_manager.withdraw_consent(clean_uid, withdraw_existing_samples=True)

        # 2. Reset personal store and sample files
        personal_res = self.collector.reset_personal_data(clean_uid)

        # 3. Remove user entry
        with self.collector._lock:
            with self.collector._get_connection() as conn:
                conn.execute("DELETE FROM users WHERE user_id = ?", (clean_uid,))
                conn.commit()

        return {
            "ok": True,
            "user_id": clean_uid,
            "personal_data_erased": personal_res,
            "message": "All user data and contributions completely erased or withdrawn.",
        }

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data associated with a user for data portability."""
        clean_uid = self.sanitize_user_id(user_id)
        user_info = self.collector.get_or_create_user(clean_uid)
        history = self.collector.get_feedback_history(clean_uid, limit=200)
        stats = self.collector.get_feedback_stats(clean_uid)

        return {
            "user_profile": user_info,
            "stats": stats,
            "history_count": len(history),
            "feedback_history": history,
            "data_policy": "ISL Continuous Learning Privacy Guarantee: Raw video is never stored.",
        }
