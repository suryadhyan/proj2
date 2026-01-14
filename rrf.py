"""
PHASE 3.5: RECIPROCAL RANK FUSION (RRF)

Pure RRF implementation with configurable weights.

Formula:
    RRF(d) = Σ_m [w_m / (k + rank_m(d))]

Where:
    k = constant (default 60)
    rank_m(d) = rank of document d in method m (1-indexed)
    w_m = weight of method m

Authority: VRIDDHI Architecture Document
Dependencies: None (pure Python)

Author: Claude (Ranking Engine)
Date: 2026-01-12
"""

from typing import Dict, List, Tuple


# ============================================================================
# RRF IMPLEMENTATION
# ============================================================================

def reciprocal_rank_fusion(
    rankings: Dict[str, List[str]],
    weights: Dict[str, float],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Compute Reciprocal Rank Fusion scores.

    Formula:
        RRF(d) = Σ_m [w_m / (k + rank_m(d))]

    Args:
        rankings: Dict of method_name → ranked list of doc_ids
                  Example: {"dense": ["id1", "id2", ...], "bm25": [...]}
        weights: Dict of method_name → weight
                 Example: {"dense": 0.35, "bm25": 0.25}
        k: RRF constant (default 60)

    Returns:
        List of (doc_id, rrf_score) tuples, sorted by descending score

    Guarantees:
        - Deterministic output (stable sort)
        - No inference (only uses provided rankings)
        - No filtering (all docs in any ranking are included)
    """
    rrf_scores: Dict[str, float] = {}

    # Compute RRF score for each document
    for method_name, ranked_docs in rankings.items():
        method_weight = weights.get(method_name, 0.0)

        if method_weight == 0.0:
            # Skip methods with zero weight
            continue

        # Iterate over ranked list (1-indexed ranks)
        for rank_1indexed, doc_id in enumerate(ranked_docs, start=1):
            # RRF contribution: w_m / (k + rank)
            contribution = method_weight / (k + rank_1indexed)

            # Accumulate score
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += contribution

    # Sort by descending RRF score (stable sort for determinism)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results


def validate_rrf_weights(weights: Dict[str, float], intent: str) -> None:
    """
    Validate RRF weights for given intent.

    Rules:
    - Product/Service: Can have "dense", "bm25", "colbert", "cross_encoder"
    - Mutual: MUST NOT have "bm25"
    - All weights must be non-negative
    - Weights should sum to 1.0 (warning if not)

    Args:
        weights: Weight dictionary
        intent: Intent type ("product", "service", or "mutual")

    Raises:
        ValueError: If weights are invalid
    """
    # Check for negative weights
    for method, weight in weights.items():
        if weight < 0:
            raise ValueError(f"Negative weight for {method}: {weight}")

    # Check mutual doesn't have BM25
    if intent == "mutual" and "bm25" in weights:
        raise ValueError("Mutual intent MUST NOT use BM25 weights")

    # Warning if weights don't sum to 1.0 (not error)
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        import warnings
        warnings.warn(f"RRF weights sum to {total_weight}, not 1.0")


# ============================================================================
# UTILITIES
# ============================================================================

def create_rankings_from_scores(
    scores: Dict[str, Dict[str, float]]
) -> Dict[str, List[str]]:
    """
    Convert score dictionaries to ranked lists.

    Args:
        scores: Dict of method_name → {doc_id: score}
                Example: {"dense": {"id1": 0.9, "id2": 0.8}, "bm25": {...}}

    Returns:
        Dict of method_name → ranked list of doc_ids (descending score)

    Example:
        Input: {"dense": {"id1": 0.9, "id2": 0.8}}
        Output: {"dense": ["id1", "id2"]}
    """
    rankings = {}

    for method_name, score_dict in scores.items():
        # Sort by descending score
        ranked_docs = sorted(
            score_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        rankings[method_name] = [doc_id for doc_id, _ in ranked_docs]

    return rankings
