"""
PHASE 3.5: CROSS-ENCODER WRAPPER

Wrapper for cross-encoder reranking (pairwise scoring).

Responsibility:
- Score query-candidate pairs using cross-encoder model
- Return scores for TOP-K candidates only (expensive operation)

Authority: VRIDDHI Architecture Document
Dependencies: sentence-transformers

Author: Claude (Ranking Engine)
Date: 2026-01-12
"""

from typing import List, Dict, Tuple, Optional
from sentence_transformers import CrossEncoder
import numpy as np


# ============================================================================
# CROSS-ENCODER MODEL
# ============================================================================

# Default model (can be overridden)
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderScorer:
    """Cross-encoder model wrapper for pairwise reranking."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        """
        Initialize cross-encoder model.

        Args:
            model_name: Hugging Face model name
        """
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def score_pairs(
        self,
        query_text: str,
        candidate_texts: List[str]
    ) -> np.ndarray:
        """
        Score query-candidate pairs.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts

        Returns:
            Array of scores (length = len(candidate_texts))
        """
        # Create pairs: [(query, candidate1), (query, candidate2), ...]
        pairs = [(query_text, candidate_text) for candidate_text in candidate_texts]

        # Score all pairs
        scores = self.model.predict(pairs)

        return scores


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def score_with_cross_encoder(
    query_text: str,
    candidates: List[Dict],
    candidate_texts: Dict[str, str],
    cross_encoder: CrossEncoderScorer,
    top_k: Optional[int] = None
) -> Dict[str, float]:
    """
    Score candidates using cross-encoder model.

    Args:
        query_text: Query text (for cross-encoder input)
        candidates: List of candidate listings
        candidate_texts: Dict of listing_id → text (for cross-encoder input)
        cross_encoder: CrossEncoder model wrapper
        top_k: Score only top-k candidates (None = score all)

    Returns:
        Dict of listing_id → cross-encoder score

    Notes:
        - Cross-encoder is expensive (quadratic time)
        - Should be applied to TOP-K candidates only
        - Returns scores for all candidates that have text
    """
    scores = {}

    # Determine candidates to score
    candidates_to_score = candidates[:top_k] if top_k else candidates

    # Extract texts for scoring
    candidate_ids = []
    texts = []

    for candidate in candidates_to_score:
        listing_id = candidate.get("id") or candidate.get("listing_id")
        if not listing_id:
            continue

        # Get candidate text
        candidate_text = candidate_texts.get(listing_id)
        if not candidate_text:
            # Skip candidates without text
            continue

        candidate_ids.append(listing_id)
        texts.append(candidate_text)

    if not texts:
        # No candidates to score
        return scores

    # Score all pairs
    cross_encoder_scores = cross_encoder.score_pairs(query_text, texts)

    # Map scores to listing IDs
    for listing_id, score in zip(candidate_ids, cross_encoder_scores):
        scores[listing_id] = float(score)

    return scores


# ============================================================================
# UTILITIES
# ============================================================================

def build_cross_encoder_text(listing: Dict) -> str:
    """
    Build text representation for cross-encoder input.

    Uses same logic as embedding text construction (Phase 3.3).

    Args:
        listing: Normalized listing object

    Returns:
        Text string for cross-encoder input
    """
    # Import from embedding_builder to ensure consistency
    from embedding_builder import build_embedding_text
    return build_embedding_text(listing)
