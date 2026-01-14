"""
Test suite for listing_matcher.py
Demonstrates full listing orchestration (Phase 2.7)
"""

from listing_matcher import listing_matches


# ============================================================================
# HELPER: Simple term implication for testing
# ============================================================================

def test_implies_fn(candidate_value: str, required_value: str) -> bool:
    """Test implication function."""
    if candidate_value == required_value:
        return True

    implications = {
        "excellent": ["good"],
        "new": ["excellent", "good"],
        "downtown": ["urban"],
    }

    return required_value in implications.get(candidate_value, [])


# ============================================================================
# TEST: M-01 Intent Equality
# ============================================================================

def test_intent_equality():
    """Test M-01: Intent Equality Rule"""
    print("=" * 70)
    print("TEST: M-01 - Intent Equality")
    print("=" * 70)

    # Intent match (product)
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Intent match: product-product")

    # Intent mismatch
    A["intent"] = "product"
    B["intent"] = "service"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Intent mismatch: product vs service → FAIL")

    print()


# ============================================================================
# TEST: M-02 SubIntent Inverse (product/service)
# ============================================================================

def test_subintent_inverse():
    """Test M-02: SubIntent Inverse Rule"""
    print("=" * 70)
    print("TEST: M-02 - SubIntent Inverse (product/service)")
    print("=" * 70)

    # Product: buyer-seller (valid)
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Product: buyer-seller (inverse) → MATCH")

    # Product: buyer-buyer (invalid)
    B["subintent"] = "buyer"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Product: buyer-buyer (same) → FAIL")

    # Service: seeker-provider (valid)
    A["intent"] = "service"
    A["subintent"] = "seeker"
    B["intent"] = "service"
    B["subintent"] = "provider"
    result = listing_matches(A, B)
    assert result == True
    print("✓ Service: seeker-provider (inverse) → MATCH")

    # Service: seeker-seeker (invalid)
    B["subintent"] = "seeker"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Service: seeker-seeker (same) → FAIL")

    print()


# ============================================================================
# TEST: M-03 SubIntent Same (mutual)
# ============================================================================

def test_subintent_same():
    """Test M-03: SubIntent Same Rule"""
    print("=" * 70)
    print("TEST: M-03 - SubIntent Same (mutual)")
    print("=" * 70)

    # Mutual: exchange-exchange (valid)
    A = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books"],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books"],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Mutual: exchange-exchange (same) → MATCH")

    print()


# ============================================================================
# TEST: M-05 Domain Intersection (product/service)
# ============================================================================

def test_domain_intersection():
    """Test M-05: Domain Intersection Rule"""
    print("=" * 70)
    print("TEST: M-05 - Domain Intersection (product/service)")
    print("=" * 70)

    # Common domain
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics", "gadgets"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Common domain: electronics ∩ [electronics, gadgets] → MATCH")

    # No common domain
    B["domain"] = ["furniture", "home"]
    result = listing_matches(A, B)
    assert result == False
    print("✓ No common domain: [electronics, gadgets] ∩ [furniture, home] = ∅ → FAIL")

    print()


# ============================================================================
# TEST: M-06 Category Intersection (mutual)
# ============================================================================

def test_category_intersection():
    """Test M-06: Category Intersection Rule"""
    print("=" * 70)
    print("TEST: M-06 - Category Intersection (mutual)")
    print("=" * 70)

    # Common category
    A = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books", "magazines"],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books"],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Common category: books ∩ [books, magazines] → MATCH")

    # No common category
    B["category"] = ["electronics", "gadgets"]
    result = listing_matches(A, B)
    assert result == False
    print("✓ No common category: [books, magazines] ∩ [electronics, gadgets] = ∅ → FAIL")

    print()


# ============================================================================
# TEST: Full Integration (all phases)
# ============================================================================

def test_full_integration():
    """Test full listing matching with all constraint types"""
    print("=" * 70)
    print("TEST: Full Integration - All Phases")
    print("=" * 70)

    # Complete match: buyer wants iPhone, seller offers iPhone
    print("\nScenario 1: Complete Product Match")
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [
            {
                "type": "smartphone",
                "itemexclusions": ["used"],
                "categorical": {"brand": "apple"},
                "min": {}, "max": {"price": 100000}, "range": {}
            }
        ],
        "other": {
            "categorical": {"seller_type": "individual"},
            "min": {"rating": 4.0},
            "max": {}, "range": {},
            "otherexclusions": ["dealer"]
        },
        "self": {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        "location": {
            "categorical": {"area": "urban"},
            "min": {},
            "max": {"distance": 10},
            "range": {},
            "locationexclusions": []
        }
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [
            {
                "type": "smartphone",
                "categorical": {"brand": "apple", "condition": "new"},
                "min": {}, "max": {},
                "range": {"price": [95000, 95000]}
            }
        ],
        "other": {
            "categorical": {"area": "urban"},
            "min": {}, "max": {},
            "range": {"distance": [5, 5]},
            "otherlocationexclusions": []
        },
        "self": {
            "categorical": {"seller_type": "individual"},
            "min": {}, "max": {},
            "range": {"rating": [4.5, 4.5]},
            "selfexclusions": []
        },
        "location": {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        }
    }

    result = listing_matches(A, B, implies_fn=test_implies_fn)
    assert result == True
    print("  ✓ All constraints satisfied → MATCH")
    print("  ✓ Intent gate: product-product, buyer-seller")
    print("  ✓ Domain gate: electronics common")
    print("  ✓ Items: smartphone matches, no exclusions")
    print("  ✓ Other→Self: individual seller, rating 4.5 >= 4.0")
    print("  ✓ Location: urban area, distance 5 <= 10")

    print()


def test_failure_at_each_gate():
    """Test failures at different gates"""
    print("=" * 70)
    print("TEST: Failure at Each Gate")
    print("=" * 70)

    # Base listing
    A_base = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [{"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B_base = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }

    # Failure 1: Intent gate
    A = A_base.copy()
    B = B_base.copy()
    B["intent"] = "service"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at intent gate (M-01)")

    # Failure 2: SubIntent gate
    A = A_base.copy()
    B = B_base.copy()
    B["subintent"] = "buyer"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at subintent gate (M-02)")

    # Failure 3: Domain gate
    A = A_base.copy()
    B = B_base.copy()
    B["domain"] = ["furniture"]
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at domain gate (M-05)")

    # Failure 4: Items gate
    A = A_base.copy()
    B = B_base.copy()
    A["items"][0]["type"] = "laptop"
    B["items"][0]["type"] = "phone"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at items gate (M-07)")

    # Failure 5: Other/Self gate
    A = A_base.copy()
    B = B_base.copy()
    A["other"]["categorical"]["seller_type"] = "individual"
    B["self"]["categorical"]["seller_type"] = "dealer"
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at other/self gate (M-13)")

    # Failure 6: Location gate
    A = A_base.copy()
    B = B_base.copy()
    A["location"]["max"]["distance"] = 10
    B["other"]["range"]["distance"] = [20, 20]
    result = listing_matches(A, B)
    assert result == False
    print("✓ Failure at location gate (M-24)")

    print()


# ============================================================================
# TEST: Real-World Scenarios
# ============================================================================

def test_real_world_scenarios():
    """Test realistic matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Service matching (yoga instructor)
    print("\nScenario 1: Service Matching - Yoga Instructor")
    A = {
        "intent": "service",
        "subintent": "seeker",
        "domain": ["fitness", "wellness"],
        "category": [],
        "items": [
            {
                "type": "yoga_instruction",
                "itemexclusions": ["online"],
                "categorical": {"mode": "home_visit"},
                "min": {}, "max": {}, "range": {}
            }
        ],
        "other": {
            "categorical": {"certified": "yes"},
            "min": {"experience_years": 2},
            "max": {"hourly_rate": 5000},
            "range": {},
            "otherexclusions": []
        },
        "self": {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        "location": {
            "categorical": {"type": "urban"},
            "min": {},
            "max": {"distance": 10},
            "range": {},
            "locationexclusions": []
        }
    }
    B = {
        "intent": "service",
        "subintent": "provider",
        "domain": ["fitness", "wellness"],
        "category": [],
        "items": [
            {
                "type": "yoga_instruction",
                "categorical": {"mode": "home_visit", "style": "hatha"},
                "min": {}, "max": {},
                "range": {}
            }
        ],
        "other": {
            "categorical": {"type": "urban"},
            "min": {}, "max": {},
            "range": {"distance": [5, 5]},
            "otherlocationexclusions": []
        },
        "self": {
            "categorical": {"certified": "yes"},
            "min": {}, "max": {},
            "range": {"experience_years": [5, 5], "hourly_rate": [4000, 4000]},
            "selfexclusions": []
        },
        "location": {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        }
    }

    result = listing_matches(A, B, implies_fn=test_implies_fn)
    assert result == True
    print("  ✓ MATCH - Yoga instructor meets all requirements")

    # Scenario 2: Mutual exchange (book swap)
    print("\nScenario 2: Mutual Exchange - Book Swap")
    A = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books", "literature"],
        "items": [
            {
                "type": "novel",
                "itemexclusions": ["damaged"],
                "categorical": {"genre": "fiction"},
                "min": {}, "max": {}, "range": {}
            }
        ],
        "other": {
            "categorical": {"condition": "good"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        "self": {
            "categorical": {"condition": "good"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        "location": {
            "categorical": {},
            "min": {},
            "max": {"distance": 20},
            "range": {},
            "locationexclusions": []
        }
    }
    B = {
        "intent": "mutual",
        "subintent": "exchange",
        "domain": [],
        "category": ["books"],
        "items": [
            {
                "type": "novel",
                "categorical": {"genre": "fiction", "author": "tolkien"},
                "min": {}, "max": {},
                "range": {}
            }
        ],
        "other": {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [10, 10]},
            "otherlocationexclusions": []
        },
        "self": {
            "categorical": {"condition": "excellent"},
            "min": {}, "max": {},
            "range": {},
            "selfexclusions": []
        },
        "location": {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        }
    }

    result = listing_matches(A, B, implies_fn=test_implies_fn)
    assert result == True
    print("  ✓ MATCH - Book exchange requirements satisfied")

    print()


# ============================================================================
# TEST: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases"""
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Empty items arrays
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [],  # Empty
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Empty required items → VACUOUSLY TRUE")

    # Multiple domains, at least one common
    A["domain"] = ["electronics", "gadgets", "computers"]
    B["domain"] = ["furniture", "electronics"]
    result = listing_matches(A, B)
    assert result == True
    print("✓ Multiple domains with intersection → MATCH")

    # Empty constraints at all levels
    A = {
        "intent": "product",
        "subintent": "buyer",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    B = {
        "intent": "product",
        "subintent": "seller",
        "domain": ["electronics"],
        "category": [],
        "items": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}, "otherlocationexclusions": []},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}, "selfexclusions": []},
        "location": {"categorical": {}, "min": {}, "max": {}, "range": {}, "locationexclusions": []}
    }
    result = listing_matches(A, B)
    assert result == True
    print("✓ Minimal constraints (all empty) → MATCH")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LISTING MATCHER TEST SUITE")
    print("Testing full listing orchestration (Phase 2.7)")
    print("=" * 70)
    print()

    test_intent_equality()
    test_subintent_inverse()
    test_subintent_same()
    test_domain_intersection()
    test_category_intersection()
    test_full_integration()
    test_failure_at_each_gate()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-01 to M-06: Intent and domain/category gates enforced")
    print("✓ M-07 to M-28: Delegated to Phases 2.4, 2.5, 2.6")
    print("✓ Mandatory evaluation order:")
    print("  1. Intent gate (M-01, M-02/M-03)")
    print("  2. Domain/Category gate (M-05/M-06)")
    print("  3. Items matching (Phase 2.4)")
    print("  4. Other→Self matching (Phase 2.5)")
    print("  5. Location matching (Phase 2.6)")
    print("✓ Short-circuit: First failure → False")
    print("✓ Strict matching: Boolean only (no scoring)")
    print("✓ Directional: A→B only (not bidirectional)")
    print("✓ Dynamic attributes preserved")
    print("✓ Pure composition: No logic reimplementation")
    print()
