"""
PHASE 3.5: RANKING ENGINE

Responsibility:
- Rank candidates that have ALREADY passed boolean matching
- NEVER filter, reject, or modify constraint checking
- Use multiple scoring methods + RRF fusion

Pipeline:
1. Dense similarity scoring (cosine)
2. BM25 scoring (product/service only)
3. ColBERT scoring (optional)
4. Cross-encoder scoring (top-k only)
5. RRF fusion
6. Final ordering

Authority: VRIDDHI Architecture Document
Dependencies: numpy, sentence-transformers (optional)

Author: Claude (Ranking Engine)
Date: 2026-01-12
"""

from typing import Dict, List, Optional, Literal, TypedDict
import numpy as np
from rrf import reciprocal_rank_fusion, validate_rrf_weights, create_rankings_from_scores


# ============================================================================
# CONFIGURATION
# ============================================================================

class RankingConfig(TypedDict):
    """Ranking configuration."""
    intent: Literal["product", "service", "mutual"]
    use_bm25: bool
    use_colbert: bool
    use_cross_encoder: bool
    rrf_k: int
    weights: Dict[str, float]
    cross_encoder_top_k: int


# Default weights (LOCKED)
PRODUCT_SERVICE_WEIGHTS = {
    "dense": 0.35,
    "bm25": 0.25,
    "colbert": 0.20,
    "cross_encoder": 0.20
}

MUTUAL_WEIGHTS = {
    "dense": 0.50,
    "colbert": 0.20,
    "cross_encoder": 0.30
}

DEFAULT_RRF_K = 60
DEFAULT_CROSS_ENCODER_TOP_K = 20


def create_ranking_config(
    intent: Literal["product", "service", "mutual"],
    use_bm25: bool = True,
    use_colbert: bool = False,
    use_cross_encoder: bool = False,
    cross_encoder_top_k: int = DEFAULT_CROSS_ENCODER_TOP_K
) -> RankingConfig:
    """
    Create ranking configuration for given intent.

    Args:
        intent: Intent type
        use_bm25: Enable BM25 scoring (product/service only)
        use_colbert: Enable ColBERT scoring
        use_cross_encoder: Enable cross-encoder reranking
        cross_encoder_top_k: Number of candidates for cross-encoder

    Returns:
        RankingConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    # Select base weights
    if intent in ["product", "service"]:
        base_weights = PRODUCT_SERVICE_WEIGHTS.copy()
    elif intent == "mutual":
        base_weights = MUTUAL_WEIGHTS.copy()
        # Ensure BM25 not used for mutual
        use_bm25 = False
    else:
        raise ValueError(f"Unknown intent: {intent}")

    # Adjust weights based on enabled methods
    active_weights = {}
    for method, weight in base_weights.items():
        # Check if method is enabled
        if method == "bm25" and not use_bm25:
            continue
        if method == "colbert" and not use_colbert:
            continue
        if method == "cross_encoder" and not use_cross_encoder:
            continue

        active_weights[method] = weight

    # Renormalize weights to sum to 1.0
    total_weight = sum(active_weights.values())
    if total_weight > 0:
        active_weights = {k: v / total_weight for k, v in active_weights.items()}

    # Validate
    validate_rrf_weights(active_weights, intent)

    return RankingConfig(
        intent=intent,
        use_bm25=use_bm25,
        use_colbert=use_colbert,
        use_cross_encoder=use_cross_encoder,
        rrf_k=DEFAULT_RRF_K,
        weights=active_weights,
        cross_encoder_top_k=cross_encoder_top_k
    )


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def compute_dense_similarity(
    query_embedding: np.ndarray,
    candidate_embeddings: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute cosine similarity scores.

    Args:
        query_embedding: Query embedding vector
        candidate_embeddings: Dict of listing_id → embedding vector

    Returns:
        Dict of listing_id → cosine similarity score
    """
    scores = {}

    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        # Degenerate case: return zero scores
        return {listing_id: 0.0 for listing_id in candidate_embeddings.keys()}

    query_normalized = query_embedding / query_norm

    # Compute cosine similarity for each candidate
    for listing_id, candidate_embedding in candidate_embeddings.items():
        candidate_norm = np.linalg.norm(candidate_embedding)
        if candidate_norm == 0:
            scores[listing_id] = 0.0
            continue

        candidate_normalized = candidate_embedding / candidate_norm

        # Cosine similarity: dot product of normalized vectors
        similarity = float(np.dot(query_normalized, candidate_normalized))
        scores[listing_id] = similarity

    return scores


def apply_bm25_scores(
    bm25_scores: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Apply BM25 scores (pass-through with validation).

    Args:
        bm25_scores: Dict of listing_id → BM25 score (or None)

    Returns:
        Dict of listing_id → BM25 score (empty if None)
    """
    if bm25_scores is None:
        return {}

    # Return as-is (no modification)
    return bm25_scores


def apply_colbert_scores(
    colbert_scores: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Apply ColBERT scores (pass-through with validation).

    Args:
        colbert_scores: Dict of listing_id → ColBERT score (or None)

    Returns:
        Dict of listing_id → ColBERT score (empty if None)
    """
    if colbert_scores is None:
        return {}

    # Return as-is (no modification)
    return colbert_scores


def apply_cross_encoder_scores(
    cross_encoder_scores: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """
    Apply cross-encoder scores (pass-through with validation).

    Args:
        cross_encoder_scores: Dict of listing_id → cross-encoder score (or None)

    Returns:
        Dict of listing_id → cross-encoder score (empty if None)
    """
    if cross_encoder_scores is None:
        return {}

    # Return as-is (no modification)
    return cross_encoder_scores


# ============================================================================
# RANKING PIPELINE
# ============================================================================

def rank_candidates(
    query_embedding: np.ndarray,
    candidate_listings: List[Dict],
    candidate_embeddings: Dict[str, np.ndarray],
    config: RankingConfig,
    bm25_scores: Optional[Dict[str, float]] = None,
    colbert_scores: Optional[Dict[str, float]] = None,
    cross_encoder_scores: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Rank candidates using RRF fusion of multiple scoring methods.

    CRITICAL: This function assumes all candidates have ALREADY passed
    boolean matching (Phase 2.8). It ONLY ranks, NEVER filters.

    Pipeline:
    1. Dense similarity (cosine)
    2. BM25 (product/service only)
    3. ColBERT (optional)
    4. Cross-encoder (optional)
    5. RRF fusion
    6. Final ordering

    Args:
        query_embedding: Query embedding vector
        candidate_listings: List of candidate listing dicts
        candidate_embeddings: Dict of listing_id → embedding vector
        config: RankingConfig object
        bm25_scores: Optional dict of listing_id → BM25 score
        colbert_scores: Optional dict of listing_id → ColBERT score
        cross_encoder_scores: Optional dict of listing_id → cross-encoder score

    Returns:
        List of ranked results:
        [
            {
                "listing_id": str,
                "listing": dict,
                "final_score": float,
                "rank": int,
                "scores": {
                    "dense": float,
                    "bm25": float | None,
                    "colbert": float | None,
                    "cross_encoder": float | None
                }
            },
            ...
        ]

    Guarantees:
        - NEVER filters candidates
        - NEVER modifies input listings
        - Deterministic output (stable sort)
        - Graceful degradation (missing scores = method skipped)
    """
    # Validate intent-specific rules
    if config["intent"] == "mutual" and config["use_bm25"]:
        raise ValueError("Mutual intent MUST NOT use BM25")

    # Step 1: Compute dense similarity
    dense_scores = compute_dense_similarity(query_embedding, candidate_embeddings)

    # Step 2: Apply BM25 (product/service only)
    bm25_scores_dict = {}
    if config["use_bm25"] and config["intent"] in ["product", "service"]:
        bm25_scores_dict = apply_bm25_scores(bm25_scores)

    # Step 3: Apply ColBERT (optional)
    colbert_scores_dict = {}
    if config["use_colbert"]:
        colbert_scores_dict = apply_colbert_scores(colbert_scores)

    # Step 4: Apply cross-encoder (optional)
    cross_encoder_scores_dict = {}
    if config["use_cross_encoder"]:
        cross_encoder_scores_dict = apply_cross_encoder_scores(cross_encoder_scores)

    # Step 5: Prepare score dictionaries for RRF
    all_scores = {
        "dense": dense_scores
    }

    if bm25_scores_dict:
        all_scores["bm25"] = bm25_scores_dict

    if colbert_scores_dict:
        all_scores["colbert"] = colbert_scores_dict

    if cross_encoder_scores_dict:
        all_scores["cross_encoder"] = cross_encoder_scores_dict

    # Step 6: Convert scores to rankings
    rankings = create_rankings_from_scores(all_scores)

    # Step 7: RRF fusion
    rrf_results = reciprocal_rank_fusion(
        rankings=rankings,
        weights=config["weights"],
        k=config["rrf_k"]
    )

    # Step 8: Build final results
    # Create mapping from listing_id to listing
    listing_map = {}
    for candidate in candidate_listings:
        listing_id = candidate.get("id") or candidate.get("listing_id")
        if listing_id:
            listing_map[listing_id] = candidate

    # Build ranked results
    ranked_results = []
    for rank_1indexed, (listing_id, rrf_score) in enumerate(rrf_results, start=1):
        listing = listing_map.get(listing_id)
        if not listing:
            # Skip if listing not found (shouldn't happen)
            continue

        result = {
            "listing_id": listing_id,
            "listing": listing,
            "final_score": rrf_score,
            "rank": rank_1indexed,
            "scores": {
                "dense": dense_scores.get(listing_id),
                "bm25": bm25_scores_dict.get(listing_id),
                "colbert": colbert_scores_dict.get(listing_id),
                "cross_encoder": cross_encoder_scores_dict.get(listing_id)
            }
        }

        ranked_results.append(result)

    return ranked_results


# ============================================================================
# SIMPLIFIED API
# ============================================================================

def rank_candidates_simple(
    query_embedding: np.ndarray,
    candidate_listings: List[Dict],
    candidate_embeddings: Dict[str, np.ndarray],
    intent: Literal["product", "service", "mutual"],
    bm25_scores: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    Simplified ranking API with default configuration.

    Uses:
    - Dense similarity (always)
    - BM25 (product/service only, if scores provided)
    - No ColBERT
    - No cross-encoder

    Args:
        query_embedding: Query embedding vector
        candidate_listings: List of candidate listing dicts
        candidate_embeddings: Dict of listing_id → embedding vector
        intent: Intent type
        bm25_scores: Optional BM25 scores

    Returns:
        List of ranked results (same format as rank_candidates)
    """
    # Create simple config
    config = create_ranking_config(
        intent=intent,
        use_bm25=(bm25_scores is not None and intent in ["product", "service"]),
        use_colbert=False,
        use_cross_encoder=False
    )

    # Rank
    return rank_candidates(
        query_embedding=query_embedding,
        candidate_listings=candidate_listings,
        candidate_embeddings=candidate_embeddings,
        config=config,
        bm25_scores=bm25_scores
    )
