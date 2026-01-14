"""
PHASE 3.5: RANKING ENGINE TESTS

Test suite for ranking engine, RRF, and cross-encoder wrapper.

Author: Claude (Ranking Engine)
Date: 2026-01-12
"""

import numpy as np
from typing import Dict, List
from ranking_engine import (
    rank_candidates,
    rank_candidates_simple,
    create_ranking_config,
    compute_dense_similarity,
    PRODUCT_SERVICE_WEIGHTS,
    MUTUAL_WEIGHTS
)
from rrf import reciprocal_rank_fusion, validate_rrf_weights


# ============================================================================
# TEST: RRF STABILITY
# ============================================================================

def test_rrf_deterministic():
    """Test RRF produces deterministic output."""
    print("=" * 70)
    print("TEST: RRF Deterministic Output")
    print("=" * 70)

    rankings = {
        "dense": ["id1", "id2", "id3"],
        "bm25": ["id2", "id1", "id3"]
    }
    weights = {"dense": 0.5, "bm25": 0.5}
    k = 60

    # Run RRF multiple times
    result1 = reciprocal_rank_fusion(rankings, weights, k)
    result2 = reciprocal_rank_fusion(rankings, weights, k)
    result3 = reciprocal_rank_fusion(rankings, weights, k)

    assert result1 == result2 == result3, "RRF not deterministic"
    print("✓ RRF produces identical output on repeated runs")
    print()


def test_rrf_score_calculation():
    """Test RRF score calculation is correct."""
    print("=" * 70)
    print("TEST: RRF Score Calculation")
    print("=" * 70)

    rankings = {
        "dense": ["id1", "id2", "id3"],
        "bm25": ["id2", "id1", "id3"]
    }
    weights = {"dense": 0.5, "bm25": 0.5}
    k = 60

    result = reciprocal_rank_fusion(rankings, weights, k)

    # Manual calculation for id1:
    # dense: rank 1 → 0.5 / (60 + 1) = 0.5 / 61 ≈ 0.00819672
    # bm25:  rank 2 → 0.5 / (60 + 2) = 0.5 / 62 ≈ 0.00806452
    # Total: ≈ 0.01626124

    # Manual calculation for id2:
    # dense: rank 2 → 0.5 / (60 + 2) = 0.5 / 62 ≈ 0.00806452
    # bm25:  rank 1 → 0.5 / (60 + 1) = 0.5 / 61 ≈ 0.00819672
    # Total: ≈ 0.01626124

    # id1 and id2 should have same score (tied)
    id1_score = None
    id2_score = None
    for doc_id, score in result:
        if doc_id == "id1":
            id1_score = score
        elif doc_id == "id2":
            id2_score = score

    assert id1_score is not None and id2_score is not None
    assert abs(id1_score - id2_score) < 1e-6, f"id1 and id2 should have same RRF score, got {id1_score} vs {id2_score}"

    print(f"✓ id1 score: {id1_score:.8f}")
    print(f"✓ id2 score: {id2_score:.8f}")
    print(f"✓ RRF calculation correct (tied ranks)")
    print()


def test_rrf_graceful_degradation():
    """Test RRF handles missing methods gracefully."""
    print("=" * 70)
    print("TEST: RRF Graceful Degradation")
    print("=" * 70)

    # Only one method
    rankings = {
        "dense": ["id1", "id2", "id3"]
    }
    weights = {"dense": 1.0}
    k = 60

    result = reciprocal_rank_fusion(rankings, weights, k)

    # Should still produce valid ranking
    assert len(result) == 3
    assert result[0][0] == "id1"  # Highest ranked
    assert result[1][0] == "id2"
    assert result[2][0] == "id3"

    print("✓ RRF works with single method")
    print()


# ============================================================================
# TEST: WEIGHT APPLICATION
# ============================================================================

def test_weight_application():
    """Test RRF weights are applied correctly."""
    print("=" * 70)
    print("TEST: Weight Application")
    print("=" * 70)

    rankings = {
        "dense": ["id1", "id2"],
        "bm25": ["id2", "id1"]
    }

    # Case 1: Equal weights → tie
    weights_equal = {"dense": 0.5, "bm25": 0.5}
    result_equal = reciprocal_rank_fusion(rankings, weights_equal, k=60)
    assert result_equal[0][1] == result_equal[1][1], "Equal weights should produce tie"
    print("✓ Equal weights produce tied scores")

    # Case 2: Dense-heavy weights → id1 wins
    weights_dense = {"dense": 0.9, "bm25": 0.1}
    result_dense = reciprocal_rank_fusion(rankings, weights_dense, k=60)
    assert result_dense[0][0] == "id1", "Dense-heavy weights should favor id1"
    print("✓ Dense-heavy weights favor dense top-ranked")

    # Case 3: BM25-heavy weights → id2 wins
    weights_bm25 = {"dense": 0.1, "bm25": 0.9}
    result_bm25 = reciprocal_rank_fusion(rankings, weights_bm25, k=60)
    assert result_bm25[0][0] == "id2", "BM25-heavy weights should favor id2"
    print("✓ BM25-heavy weights favor BM25 top-ranked")

    print()


# ============================================================================
# TEST: INTENT-SPECIFIC LOGIC
# ============================================================================

def test_mutual_no_bm25():
    """Test mutual intent does not use BM25."""
    print("=" * 70)
    print("TEST: Mutual Intent - No BM25")
    print("=" * 70)

    # Create mutual config
    config = create_ranking_config("mutual", use_bm25=True)

    # BM25 should be disabled
    assert not config["use_bm25"], "Mutual intent MUST NOT use BM25"
    assert "bm25" not in config["weights"], "Mutual weights MUST NOT include BM25"

    print("✓ Mutual config correctly disables BM25")
    print()


def test_product_service_weights():
    """Test product/service use correct weights."""
    print("=" * 70)
    print("TEST: Product/Service Weights")
    print("=" * 70)

    # Default config
    config = create_ranking_config("product", use_bm25=True)

    # Should have dense + bm25
    assert "dense" in config["weights"]
    assert "bm25" in config["weights"]

    print(f"✓ Product weights: {config['weights']}")
    print()


def test_mutual_weights():
    """Test mutual uses correct weights."""
    print("=" * 70)
    print("TEST: Mutual Weights")
    print("=" * 70)

    # Default config
    config = create_ranking_config("mutual")

    # Should have dense only (no BM25)
    assert "dense" in config["weights"]
    assert "bm25" not in config["weights"]

    print(f"✓ Mutual weights: {config['weights']}")
    print()


# ============================================================================
# TEST: DENSE SIMILARITY
# ============================================================================

def test_dense_similarity():
    """Test cosine similarity calculation."""
    print("=" * 70)
    print("TEST: Dense Similarity (Cosine)")
    print("=" * 70)

    query_embedding = np.array([1.0, 0.0, 0.0])

    candidate_embeddings = {
        "id1": np.array([1.0, 0.0, 0.0]),  # Identical → similarity = 1.0
        "id2": np.array([0.0, 1.0, 0.0]),  # Orthogonal → similarity = 0.0
        "id3": np.array([0.5, 0.5, 0.0])   # Partial overlap
    }

    scores = compute_dense_similarity(query_embedding, candidate_embeddings)

    # Check id1 (identical)
    assert abs(scores["id1"] - 1.0) < 1e-6, f"Expected similarity 1.0, got {scores['id1']}"
    print(f"✓ Identical vectors: similarity = {scores['id1']:.6f}")

    # Check id2 (orthogonal)
    assert abs(scores["id2"]) < 1e-6, f"Expected similarity 0.0, got {scores['id2']}"
    print(f"✓ Orthogonal vectors: similarity = {scores['id2']:.6f}")

    # Check id3 (partial)
    assert 0 < scores["id3"] < 1, f"Expected 0 < similarity < 1, got {scores['id3']}"
    print(f"✓ Partial overlap: similarity = {scores['id3']:.6f}")

    print()


# ============================================================================
# TEST: FULL RANKING PIPELINE
# ============================================================================

def test_ranking_pipeline_product():
    """Test full ranking pipeline for product intent."""
    print("=" * 70)
    print("TEST: Full Ranking Pipeline (Product)")
    print("=" * 70)

    # Query
    query_embedding = np.array([1.0, 0.0, 0.0])

    # Candidates
    candidate_listings = [
        {"listing_id": "id1", "intent": "product"},
        {"listing_id": "id2", "intent": "product"},
        {"listing_id": "id3", "intent": "product"}
    ]

    candidate_embeddings = {
        "id1": np.array([1.0, 0.0, 0.0]),
        "id2": np.array([0.5, 0.5, 0.0]),
        "id3": np.array([0.0, 1.0, 0.0])
    }

    # BM25 scores (favor id3)
    bm25_scores = {
        "id1": 0.5,
        "id2": 0.7,
        "id3": 0.9
    }

    # Rank
    results = rank_candidates_simple(
        query_embedding=query_embedding,
        candidate_listings=candidate_listings,
        candidate_embeddings=candidate_embeddings,
        intent="product",
        bm25_scores=bm25_scores
    )

    # Verify results
    assert len(results) == 3, "Should have 3 ranked results"
    assert results[0]["rank"] == 1, "First result should have rank 1"
    assert results[1]["rank"] == 2, "Second result should have rank 2"
    assert results[2]["rank"] == 3, "Third result should have rank 3"

    # Verify descending scores
    assert results[0]["final_score"] >= results[1]["final_score"]
    assert results[1]["final_score"] >= results[2]["final_score"]

    print("✓ Ranking pipeline produces valid output")
    print(f"  Rank 1: {results[0]['listing_id']} (score: {results[0]['final_score']:.6f})")
    print(f"  Rank 2: {results[1]['listing_id']} (score: {results[1]['final_score']:.6f})")
    print(f"  Rank 3: {results[2]['listing_id']} (score: {results[2]['final_score']:.6f})")
    print()


def test_ranking_pipeline_mutual():
    """Test full ranking pipeline for mutual intent."""
    print("=" * 70)
    print("TEST: Full Ranking Pipeline (Mutual)")
    print("=" * 70)

    # Query
    query_embedding = np.array([1.0, 0.0, 0.0])

    # Candidates
    candidate_listings = [
        {"listing_id": "id1", "intent": "mutual"},
        {"listing_id": "id2", "intent": "mutual"}
    ]

    candidate_embeddings = {
        "id1": np.array([1.0, 0.0, 0.0]),
        "id2": np.array([0.5, 0.5, 0.0])
    }

    # Rank (no BM25 for mutual)
    results = rank_candidates_simple(
        query_embedding=query_embedding,
        candidate_listings=candidate_listings,
        candidate_embeddings=candidate_embeddings,
        intent="mutual",
        bm25_scores=None  # Should be ignored
    )

    # Verify results
    assert len(results) == 2
    assert results[0]["scores"]["bm25"] is None, "Mutual should not have BM25 scores"

    print("✓ Mutual ranking works without BM25")
    print(f"  Rank 1: {results[0]['listing_id']} (score: {results[0]['final_score']:.6f})")
    print(f"  Rank 2: {results[1]['listing_id']} (score: {results[1]['final_score']:.6f})")
    print()


# ============================================================================
# TEST: FAILURE SCENARIOS
# ============================================================================

def test_empty_candidates():
    """Test ranking with no candidates."""
    print("=" * 70)
    print("TEST: Empty Candidates")
    print("=" * 70)

    query_embedding = np.array([1.0, 0.0, 0.0])
    candidate_listings = []
    candidate_embeddings = {}

    results = rank_candidates_simple(
        query_embedding=query_embedding,
        candidate_listings=candidate_listings,
        candidate_embeddings=candidate_embeddings,
        intent="product"
    )

    assert len(results) == 0, "Empty candidates should produce empty results"
    print("✓ Empty candidates handled gracefully")
    print()


def test_missing_embeddings():
    """Test ranking with missing embeddings."""
    print("=" * 70)
    print("TEST: Missing Embeddings")
    print("=" * 70)

    query_embedding = np.array([1.0, 0.0, 0.0])

    candidate_listings = [
        {"listing_id": "id1"},
        {"listing_id": "id2"}
    ]

    # Only one embedding
    candidate_embeddings = {
        "id1": np.array([1.0, 0.0, 0.0])
    }

    results = rank_candidates_simple(
        query_embedding=query_embedding,
        candidate_listings=candidate_listings,
        candidate_embeddings=candidate_embeddings,
        intent="product"
    )

    # Only id1 should be ranked (id2 missing embedding)
    assert len(results) == 1, "Only candidates with embeddings should be ranked"
    assert results[0]["listing_id"] == "id1"

    print("✓ Missing embeddings handled gracefully")
    print()


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all ranking engine tests."""
    print()
    print("=" * 70)
    print("RANKING ENGINE TEST SUITE")
    print("=" * 70)
    print()

    # RRF tests
    test_rrf_deterministic()
    test_rrf_score_calculation()
    test_rrf_graceful_degradation()

    # Weight tests
    test_weight_application()

    # Intent-specific tests
    test_mutual_no_bm25()
    test_product_service_weights()
    test_mutual_weights()

    # Similarity tests
    test_dense_similarity()

    # Full pipeline tests
    test_ranking_pipeline_product()
    test_ranking_pipeline_mutual()

    # Failure scenario tests
    test_empty_candidates()
    test_missing_embeddings()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ RRF is deterministic and stable")
    print("✓ Weights are applied correctly")
    print("✓ Mutual intent does NOT use BM25")
    print("✓ Product/service intent uses BM25")
    print("✓ Dense similarity calculation is correct")
    print("✓ Full pipeline produces valid rankings")
    print("✓ Graceful degradation for missing data")
    print()


if __name__ == "__main__":
    run_all_tests()
