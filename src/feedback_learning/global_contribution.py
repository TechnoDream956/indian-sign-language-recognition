"""
Global Contribution and Consent Management.
============================================
Enforces strict opt-in consent for contributing landmark correction samples
to the curated global dataset. Handles consent granting, revocation, and withdrawal.
"""

import time
from typing import Any, Dict, List, Optional
from .config import CURRENT_CONSENT_VERSION, GLOBAL_CONTRIBUTION_DEFAULT_ENABLED
from .sample_collector import SampleCollector


class GlobalContributionManager:
    """Manages explicit user consent and contribution lifecycle."""

    def __init__(self, collector: Optional[SampleCollector] = None):
        self.collector = collector or SampleCollector()

    def get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """Check current global contribution consent status for a user."""
        user = self.collector.get_or_create_user(user_id)
        enabled = bool(user["global_contribution_enabled"])
        return {
            "user_id": user["user_id"],
            "global_contribution_enabled": enabled,
            "consent_version": user["consent_version"],
            "current_consent_version": CURRENT_CONSENT_VERSION,
            "terms": self.get_consent_terms(),
        }

    def grant_consent(self, user_id: str, consent_version: str = CURRENT_CONSENT_VERSION) -> Dict[str, Any]:
        """User explicitly grants opt-in consent to contribute future correction samples."""
        updated = self.collector.update_user_preferences(
            user_id=user_id,
            global_contribution_enabled=True,
            consent_version=consent_version,
        )
        return {
            "ok": True,
            "user_id": user_id,
            "global_contribution_enabled": True,
            "consent_version": updated["consent_version"],
            "message": "Opt-in consent recorded. Eligible correction samples will be queued for review.",
        }

    def withdraw_consent(self, user_id: str, withdraw_existing_samples: bool = True) -> Dict[str, Any]:
        """
        User revokes opt-in consent.
        Optionally withdraws pending and accepted global contributions from future training.
        """
        self.collector.update_user_preferences(
            user_id=user_id,
            global_contribution_enabled=False,
        )

        withdrawn_count = 0
        if withdraw_existing_samples:
            now = time.time()
            with self.collector._lock:
                with self.collector._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        UPDATE global_contributions
                        SET contribution_status = 'withdrawn',
                            review_notes = 'Withdrawn by user consent revocation',
                            updated_at = ?
                        WHERE user_id = ? AND contribution_status IN ('pending', 'accepted')
                        """,
                        (now, user_id),
                    )
                    withdrawn_count = cur.rowcount
                    conn.commit()

        return {
            "ok": True,
            "user_id": user_id,
            "global_contribution_enabled": False,
            "withdrawn_contributions_count": withdrawn_count,
            "message": "Global contribution consent successfully revoked.",
        }

    @staticmethod
    def get_consent_terms() -> Dict[str, Any]:
        """Explain exactly what is collected and privacy guarantees."""
        return {
            "summary": "Help improve Indian Sign Language recognition accuracy for everyone.",
            "what_is_collected": [
                "Normalized 3D hand landmark coordinates (63 numbers per hand).",
                "Fingerspelling alphabet label (A-Z).",
                "Recognition confidence score and timestamp.",
            ],
            "what_is_NOT_collected": [
                "Raw webcam video or photos (strictly NOT collected by default).",
                "Facial identity, IP address, or private hardware identifiers.",
            ],
            "how_data_is_used": [
                "Samples are reviewed through quality filters to eliminate noise or duplicates.",
                "Approved samples become part of an open, curated ISL dataset for periodic model training.",
            ],
            "your_rights": [
                "Opt-in is strictly voluntary and disabled by default.",
                "Personal learning works independently even if global contribution is disabled.",
                "You can withdraw consent at any time and withdraw your submitted samples.",
            ],
        }
