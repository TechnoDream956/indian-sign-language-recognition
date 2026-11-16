"""
Personal Adapter Layer.
=======================
Conservative personalization layer that adapts alphabet predictions based on
verified individual user feedback without modifying the production model.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .config import (
    PERSONAL_SIMILARITY_THRESHOLD,
    PERSONAL_MIN_SAMPLES_REQUIRED,
    FEATURE_SCHEMA_SINGLE_HAND,
)
from .personal_store import PersonalStore


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized feature vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


class PersonalAdapter:
    """
    Conservative personalization layer.
    Compares input feature representations with user-specific verified exemplars.
    Requires multiple independent examples (>=2) and high cosine similarity (>=0.88)
    to adjust a prediction.
    """

    def __init__(
        self,
        personal_store: Optional[PersonalStore] = None,
        similarity_threshold: float = PERSONAL_SIMILARITY_THRESHOLD,
        min_samples: int = PERSONAL_MIN_SAMPLES_REQUIRED,
    ):
        self.store = personal_store or PersonalStore()
        self.similarity_threshold = similarity_threshold
        self.min_samples = min_samples

    def adapt_prediction(
        self,
        user_id: str,
        features: np.ndarray,
        production_prediction: Optional[str],
        production_confidence: float,
        personalization_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate whether a user-specific correction should adapt the prediction.
        """
        orig_pred = production_prediction
        orig_conf = float(production_confidence)

        # Baseline response (unchanged)
        res = {
            "original_prediction": orig_pred,
            "adjusted_prediction": orig_pred,
            "confidence": orig_conf,
            "similarity_score": 0.0,
            "personalization_applied": False,
            "matched_class": None,
            "reason": "Personalization not applied",
        }

        # If personalization is explicitly turned off, retain production prediction
        if not personalization_enabled:
            res["reason"] = "Personalization disabled in user preferences"
            return res

        if features is None or not isinstance(features, np.ndarray) or features.size == 0:
            res["reason"] = "No valid feature vector provided"
            return res

        # Retrieve user exemplars
        user_exemplars = self.store.get_user_exemplars(user_id)
        if not user_exemplars:
            res["reason"] = "No personal exemplars stored for user"
            return res

        # Evaluate similarity across stored classes
        best_class = None
        best_similarity = -1.0
        best_sample_count = 0

        target_features = features.flatten().astype(np.float32)

        for label, samples in user_exemplars.items():
            if len(samples) < self.min_samples:
                # Require sufficient independent samples before considering an override
                continue

            # Compute similarities to all exemplars of this class
            sims = [cosine_similarity(target_features, s.flatten()) for s in samples]
            if not sims:
                continue

            # Top-k or average similarity
            sims.sort(reverse=True)
            top_k_sim = float(np.mean(sims[:min(3, len(sims))]))

            if top_k_sim > best_similarity:
                best_similarity = top_k_sim
                best_class = label
                best_sample_count = len(samples)

        res["similarity_score"] = round(max(0.0, best_similarity), 4)
        res["matched_class"] = best_class

        # Check override conditions:
        # 1. Similarity exceeds the conservative threshold (e.g. 0.88)
        # 2. Number of samples meets minimum requirement (e.g. >= 2)
        # 3. If best_class is different from orig_pred, require either low production confidence
        #    or very strong personal match (e.g. similarity > production_confidence or >= 0.92)
        if (
            best_class is not None
            and best_similarity >= self.similarity_threshold
            and best_sample_count >= self.min_samples
        ):
            if best_class == orig_pred:
                # Corroborates production prediction with higher personalized confidence
                boosted_conf = min(0.99, max(orig_conf, best_similarity * 0.95))
                res["adjusted_prediction"] = best_class
                res["confidence"] = round(boosted_conf, 4)
                res["personalization_applied"] = False  # Did not change label, reinforced
                res["reason"] = f"Reinforced production prediction via {best_sample_count} personal exemplars"
                return res

            # Proposed override
            can_override = False
            if orig_conf < 0.70 and best_similarity >= self.similarity_threshold:
                can_override = True
            elif best_similarity >= 0.93 and best_similarity > orig_conf:
                can_override = True

            if can_override:
                adjusted_conf = round(best_similarity * 0.92, 4)
                res["adjusted_prediction"] = best_class
                res["confidence"] = adjusted_conf
                res["personalization_applied"] = True
                res["reason"] = (
                    f"Personal adaptation override: {orig_pred} -> {best_class} "
                    f"({best_sample_count} exemplars, similarity={best_similarity:.2f})"
                )
                return res
            else:
                res["reason"] = (
                    f"Personal candidate '{best_class}' (sim={best_similarity:.2f}) "
                    f"did not overcome production confidence ({orig_conf:.2f})"
                )
                return res

        res["reason"] = f"No personal class met similarity threshold {self.similarity_threshold}"
        return res
