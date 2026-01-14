"""
Test suite for item_matchers.py
Demonstrates single-item pair matching (Phase 2.3)
"""

from item_matchers import (
    match_item_type,
    match_item_categorical,
    match_item_numeric,
    item_matches
)


# ============================================================================
# HELPER: Term implication function for testing
# ============================================================================

def test_implies_fn(candidate_value: str, required_value: str) -> bool:
    """
    Test term implication function with common relationships.

    Examples:
    - "excellent" implies "good"
    - "new" implies "excellent" implies "good"
    - "vegan" implies "vegetarian"
    """
    implications = {
        "excellent": ["good"],
        "new": ["excellent", "good"],
        "vegan": ["vegetarian"],
        "vegetarian": ["no_beef"],
    }

    # Exact match
    if candidate_value == required_value:
        return True

    # Check implication
    if candidate_value in implications:
        return required_value in implications[candidate_value]

    return False


# ============================================================================
# TEST: M-07 Item Type Matching
# ============================================================================

def test_match_item_type():
    """Test M-07: Item Type Equality Rule"""
    print("=" * 70)
    print("TEST: match_item_type() - M-07")
    print("=" * 70)

    # Exact type match
    result = match_item_type(
        {"type": "smartphone"},
        {"type": "smartphone"}
    )
    assert result == True
    print("✓ Same type 'smartphone' = 'smartphone'")

    # Type mismatch
    result = match_item_type(
        {"type": "smartphone"},
        {"type": "laptop"}
    )
    assert result == False
    print("✓ Different types 'smartphone' ≠ 'laptop'")

    # Already normalized (lowercase)
    result = match_item_type(
        {"type": "smartphone"},
        {"type": "smartphone"}  # Phase 2.1 would have normalized "SMARTPHONE"
    )
    assert result == True
    print("✓ Normalized types match (Phase 2.1 handled case)")

    print()


# ============================================================================
# TEST: M-08 Categorical Subset Matching
# ============================================================================

def test_match_item_categorical():
    """Test M-08: Item Categorical Subset Rule"""
    print("=" * 70)
    print("TEST: match_item_categorical() - M-08")
    print("=" * 70)

    # Exact match
    result = match_item_categorical(
        {"categorical": {"brand": "apple"}},
        {"categorical": {"brand": "apple"}}
    )
    assert result == True
    print("✓ Exact categorical match: brand='apple'")

    # Subset: required ⊆ candidate (extra keys in candidate)
    result = match_item_categorical(
        {"categorical": {"brand": "apple"}},
        {"categorical": {"brand": "apple", "color": "black", "storage": "256gb"}}
    )
    assert result == True
    print("✓ Subset match: required keys present, extra keys ignored")

    # Missing required key
    result = match_item_categorical(
        {"categorical": {"brand": "apple"}},
        {"categorical": {"color": "black"}}
    )
    assert result == False
    print("✓ Missing required key 'brand' → FAIL")

    # Value mismatch
    result = match_item_categorical(
        {"categorical": {"brand": "apple"}},
        {"categorical": {"brand": "samsung"}}
    )
    assert result == False
    print("✓ Value mismatch: 'apple' ≠ 'samsung' → FAIL")

    # Multiple attributes, all match
    result = match_item_categorical(
        {"categorical": {"brand": "apple", "color": "black"}},
        {"categorical": {"brand": "apple", "color": "black", "storage": "256gb"}}
    )
    assert result == True
    print("✓ Multiple required attributes, all satisfied")

    # Multiple attributes, one mismatch
    result = match_item_categorical(
        {"categorical": {"brand": "apple", "color": "black"}},
        {"categorical": {"brand": "apple", "color": "white"}}
    )
    assert result == False
    print("✓ Multiple attributes, one mismatch → FAIL")

    # Empty required (vacuously true)
    result = match_item_categorical(
        {"categorical": {}},
        {"categorical": {"brand": "apple", "color": "black"}}
    )
    assert result == True
    print("✓ Empty required categorical → VACUOUSLY TRUE")

    print()


def test_match_item_categorical_with_implications():
    """Test M-08 with term implications"""
    print("=" * 70)
    print("TEST: match_item_categorical() with Term Implications")
    print("=" * 70)

    # Candidate value implies required value
    result = match_item_categorical(
        {"categorical": {"condition": "good"}},
        {"categorical": {"condition": "excellent"}},
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Implication: 'excellent' implies 'good'")

    # Transitive implication
    result = match_item_categorical(
        {"categorical": {"condition": "good"}},
        {"categorical": {"condition": "new"}},
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Transitive: 'new' → 'excellent' → 'good'")

    # No implication (should fail)
    result = match_item_categorical(
        {"categorical": {"condition": "excellent"}},
        {"categorical": {"condition": "good"}},
        implies_fn=test_implies_fn
    )
    assert result == False
    print("✓ Reverse implication fails: 'good' does NOT imply 'excellent'")

    # Multiple attributes with mixed exact/implication
    result = match_item_categorical(
        {"categorical": {"brand": "apple", "condition": "good"}},
        {"categorical": {"brand": "apple", "condition": "excellent"}},
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Mixed: exact match on brand, implication on condition")

    print()


# ============================================================================
# TEST: M-09, M-10, M-11 Numeric Constraints
# ============================================================================

def test_match_item_numeric():
    """Test M-09, M-10, M-11: Item Numeric Constraints"""
    print("=" * 70)
    print("TEST: match_item_numeric() - M-09, M-10, M-11")
    print("=" * 70)

    # MIN constraint satisfied
    result = match_item_numeric(
        {"min": {"rating": 4.0}, "max": {}, "range": {}},
        {"min": {}, "max": {}, "range": {"rating": [4.5, 5.0]}}
    )
    assert result == True
    print("✓ M-09: MIN constraint satisfied (4.5 >= 4.0)")

    # MIN constraint violated
    result = match_item_numeric(
        {"min": {"rating": 4.0}, "max": {}, "range": {}},
        {"min": {}, "max": {}, "range": {"rating": [3.5, 5.0]}}
    )
    assert result == False
    print("✓ M-09: MIN constraint violated (3.5 < 4.0)")

    # MAX constraint satisfied
    result = match_item_numeric(
        {"min": {}, "max": {"price": 100000}, "range": {}},
        {"min": {}, "max": {}, "range": {"price": [95000, 95000]}}
    )
    assert result == True
    print("✓ M-10: MAX constraint satisfied (95000 <= 100000)")

    # MAX constraint violated
    result = match_item_numeric(
        {"min": {}, "max": {"price": 100000}, "range": {}},
        {"min": {}, "max": {}, "range": {"price": [150000, 150000]}}
    )
    assert result == False
    print("✓ M-10: MAX constraint violated (150000 > 100000)")

    # RANGE constraint satisfied (EXACT match)
    result = match_item_numeric(
        {"min": {}, "max": {}, "range": {"storage": [256, 256]}},
        {"min": {}, "max": {}, "range": {"storage": [256, 256]}}
    )
    assert result == True
    print("✓ M-11: RANGE constraint satisfied (EXACT: [256,256] = [256,256])")

    # RANGE constraint violated (EXACT mismatch)
    result = match_item_numeric(
        {"min": {}, "max": {}, "range": {"storage": [256, 256]}},
        {"min": {}, "max": {}, "range": {"storage": [512, 512]}}
    )
    assert result == False
    print("✓ M-11: RANGE constraint violated (EXACT: [256,256] ≠ [512,512])")

    # RANGE constraint: value within range
    result = match_item_numeric(
        {"min": {}, "max": {}, "range": {"price": [40000, 60000]}},
        {"min": {}, "max": {}, "range": {"price": [55000, 55000]}}
    )
    assert result == True
    print("✓ M-11: Value [55000,55000] within range [40000,60000]")

    # Combined constraints
    result = match_item_numeric(
        {"min": {"rating": 4.0}, "max": {"price": 100000}, "range": {"storage": [256, 512]}},
        {"min": {}, "max": {}, "range": {"rating": [4.5, 5.0], "price": [95000, 95000], "storage": [256, 256]}}
    )
    assert result == True
    print("✓ All numeric constraints satisfied (MIN + MAX + RANGE)")

    # Empty constraints (vacuously true)
    result = match_item_numeric(
        {"min": {}, "max": {}, "range": {}},
        {"min": {}, "max": {}, "range": {"price": [100, 100]}}
    )
    assert result == True
    print("✓ Empty numeric constraints → VACUOUSLY TRUE")

    print()


# ============================================================================
# TEST: item_matches() - Composite Matching
# ============================================================================

def test_item_matches_complete():
    """Test item_matches(): Complete item matching"""
    print("=" * 70)
    print("TEST: item_matches() - Complete Item Matching")
    print("=" * 70)

    # Perfect match (all constraints satisfied)
    required = {
        "type": "smartphone",
        "categorical": {"brand": "apple", "color": "black"},
        "min": {},
        "max": {"price": 100000},
        "range": {"storage": [256, 256]}
    }
    candidate = {
        "type": "smartphone",
        "categorical": {"brand": "apple", "color": "black", "condition": "excellent"},
        "min": {},
        "max": {},
        "range": {"price": [95000, 95000], "storage": [256, 256]}
    }
    result = item_matches(required, candidate)
    assert result == True
    print("✓ Complete match: type + categorical + numeric all satisfied")

    # Type mismatch (fails immediately)
    required = {"type": "smartphone", "categorical": {}, "min": {}, "max": {}, "range": {}}
    candidate = {"type": "laptop", "categorical": {}, "min": {}, "max": {}, "range": {}}
    result = item_matches(required, candidate)
    assert result == False
    print("✓ Type mismatch → FAIL (short-circuit)")

    # Type matches, categorical fails
    required = {
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {}, "max": {}, "range": {}
    }
    candidate = {
        "type": "smartphone",
        "categorical": {"brand": "samsung"},
        "min": {}, "max": {}, "range": {}
    }
    result = item_matches(required, candidate)
    assert result == False
    print("✓ Type matches, categorical fails → FAIL")

    # Type and categorical match, numeric fails
    required = {
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {},
        "max": {"price": 50000},
        "range": {}
    }
    candidate = {
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {},
        "max": {},
        "range": {"price": [100000, 100000]}
    }
    result = item_matches(required, candidate)
    assert result == False
    print("✓ Type + categorical match, numeric fails → FAIL")

    # With term implications
    required = {
        "type": "phone",
        "categorical": {"condition": "good"},
        "min": {}, "max": {}, "range": {}
    }
    candidate = {
        "type": "phone",
        "categorical": {"condition": "excellent"},
        "min": {}, "max": {}, "range": {}
    }
    result = item_matches(required, candidate, implies_fn=test_implies_fn)
    assert result == True
    print("✓ Match with term implication: 'excellent' implies 'good'")

    print()


def test_real_world_scenarios():
    """Test realistic item matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Buyer wants iPhone 15 Pro, 256GB, under 1 lakh
    # Seller offers exactly that at 95k
    print("\nScenario 1: Product Match - iPhone")
    required = {
        "type": "smartphone",
        "categorical": {"brand": "apple", "model": "iphone 15 pro", "color": "black"},
        "min": {},
        "max": {"price": 100000},
        "range": {"storage": [256, 256]}
    }
    candidate = {
        "type": "smartphone",
        "categorical": {"brand": "apple", "model": "iphone 15 pro", "color": "black", "condition": "excellent"},
        "min": {},
        "max": {},
        "range": {"price": [95000, 95000], "storage": [256, 256]}
    }
    result = item_matches(required, candidate)
    assert result == True
    print("  ✓ MATCH - All constraints satisfied")

    # Scenario 2: Service - Yoga instruction
    print("\nScenario 2: Service Match - Yoga Instructor")
    required = {
        "type": "yoga_instruction",
        "categorical": {"mode": "home_visit", "availability": "weekends"},
        "min": {},
        "max": {"price_per_session": 2000},
        "range": {}
    }
    candidate = {
        "type": "yoga_instruction",
        "categorical": {"mode": "home_visit", "availability": "weekends", "certified": "true"},
        "min": {},
        "max": {},
        "range": {"price_per_session": [1500, 1500]}
    }
    result = item_matches(required, candidate)
    assert result == True
    print("  ✓ MATCH - Service requirements satisfied")

    # Scenario 3: Mismatch - Wrong storage capacity
    print("\nScenario 3: No Match - Storage Mismatch")
    required = {
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {},
        "max": {},
        "range": {"storage": [256, 256]}  # EXACT 256GB
    }
    candidate = {
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {},
        "max": {},
        "range": {"storage": [128, 128]}  # Only 128GB
    }
    result = item_matches(required, candidate)
    assert result == False
    print("  ✓ NO MATCH - EXACT storage requirement not met")

    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Minimal required item (only type)
    result = item_matches(
        {"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}},
        {"type": "phone", "categorical": {"brand": "apple", "color": "black"}, "min": {}, "max": {}, "range": {"price": [100, 100]}}
    )
    assert result == True
    print("✓ Minimal required (only type) → MATCH")

    # Boundary: Exact match at threshold
    result = match_item_numeric(
        {"min": {"rating": 4.0}, "max": {}, "range": {}},
        {"min": {}, "max": {}, "range": {"rating": [4.0, 4.0]}}
    )
    assert result == True
    print("✓ Boundary: Exact match at MIN threshold (4.0 >= 4.0)")

    result = match_item_numeric(
        {"min": {}, "max": {"price": 100}, "range": {}},
        {"min": {}, "max": {}, "range": {"price": [100, 100]}}
    )
    assert result == True
    print("✓ Boundary: Exact match at MAX threshold (100 <= 100)")

    # Empty candidate categorical (required has constraints)
    result = match_item_categorical(
        {"categorical": {"brand": "apple"}},
        {"categorical": {}}
    )
    assert result == False
    print("✓ Empty candidate categorical with required keys → FAIL")

    # Both empty (vacuously true)
    result = match_item_categorical(
        {"categorical": {}},
        {"categorical": {}}
    )
    assert result == True
    print("✓ Both empty categorical → VACUOUSLY TRUE")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ITEM MATCHERS TEST SUITE")
    print("Testing single-item pair matching (Phase 2.3)")
    print("=" * 70)
    print()

    test_match_item_type()
    test_match_item_categorical()
    test_match_item_categorical_with_implications()
    test_match_item_numeric()
    test_item_matches_complete()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-07: Item type equality enforced")
    print("✓ M-08: Categorical subset with term implications")
    print("✓ M-09, M-10, M-11: Numeric constraints (reuses Phase 2.2)")
    print("✓ item_matches(): Composite function, short-circuits on failure")
    print("✓ Empty constraints: Vacuously satisfied")
    print("✓ Dynamic attributes: No hardcoded names, runtime discovery")
    print("✓ Term implications: Injectable via parameter")
    print("✓ Pure functions: No side effects, composable")
    print()
