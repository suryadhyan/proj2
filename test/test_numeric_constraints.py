"""
Test suite for numeric_constraints.py
Demonstrates range-based constraint evaluation
"""

from numeric_constraints import (
    Range,
    NEGATIVE_INFINITY,
    POSITIVE_INFINITY,
    range_contains,
    satisfies_min_constraint,
    satisfies_max_constraint,
    evaluate_min_constraints,
    evaluate_max_constraints,
    evaluate_range_constraints
)


def test_range_contains():
    """Test range containment (subset) checking"""
    print("=" * 70)
    print("TEST: range_contains() - Range Subset Checking")
    print("=" * 70)

    # Basic containment
    assert range_contains([50, 60], [40, 80]) == True
    print("✓ [50, 60] ⊆ [40, 80]")

    # EXACT value within range
    assert range_contains([256, 256], [256, 512]) == True
    print("✓ [256, 256] ⊆ [256, 512] (EXACT within range)")

    # EXACT match
    assert range_contains([256, 256], [256, 256]) == True
    print("✓ [256, 256] ⊆ [256, 256] (EXACT match)")

    # Exceeds upper bound
    assert range_contains([100, 200], [50, 150]) == False
    print("✓ [100, 200] ⊄ [50, 150] (exceeds max)")

    # Below lower bound
    assert range_contains([10, 100], [50, 150]) == False
    print("✓ [10, 100] ⊄ [50, 150] (below min)")

    # Partial overlap not enough
    assert range_contains([50, 70], [60, 80]) == False
    print("✓ [50, 70] ⊄ [60, 80] (partial overlap insufficient)")

    # Unbounded ranges
    assert range_contains([50, 100], [NEGATIVE_INFINITY, POSITIVE_INFINITY]) == True
    print("✓ [50, 100] ⊆ [-∞, +∞] (any finite range within unbounded)")

    print()


def test_satisfies_min_constraint():
    """Test MIN constraint satisfaction"""
    print("=" * 70)
    print("TEST: satisfies_min_constraint() - M-09, M-14, M-19")
    print("=" * 70)

    # Basic satisfaction
    assert satisfies_min_constraint(4.0, [4.5, 5.0]) == True
    print("✓ rating.min=4.5 >= required_min=4.0")

    # Boundary case (exact match)
    assert satisfies_min_constraint(4.0, [4.0, 4.0]) == True
    print("✓ rating.min=4.0 >= required_min=4.0 (boundary)")

    # Violation
    assert satisfies_min_constraint(4.0, [3.5, 5.0]) == False
    print("✓ rating.min=3.5 < required_min=4.0 → FAIL")

    # EXACT value satisfies min
    assert satisfies_min_constraint(36, [48, 48]) == True
    print("✓ experience.min=48 >= required_min=36 (EXACT value)")

    # Zero threshold
    assert satisfies_min_constraint(0, [10, 20]) == True
    print("✓ value.min=10 >= required_min=0")

    print()


def test_satisfies_max_constraint():
    """Test MAX constraint satisfaction"""
    print("=" * 70)
    print("TEST: satisfies_max_constraint() - M-10, M-15, M-20")
    print("=" * 70)

    # Basic satisfaction
    assert satisfies_max_constraint(100000, [95000, 95000]) == True
    print("✓ price.max=95000 <= required_max=100000")

    # Boundary case (exact match)
    assert satisfies_max_constraint(100, [100, 100]) == True
    print("✓ price.max=100 <= required_max=100 (boundary)")

    # Violation - range exceeds max
    assert satisfies_max_constraint(100, [50, 150]) == False
    print("✓ price.max=150 > required_max=100 → FAIL")

    # Range within budget
    assert satisfies_max_constraint(100000, [80000, 95000]) == True
    print("✓ price.max=95000 <= required_max=100000 (range)")

    print()


def test_evaluate_min_constraints():
    """Test MIN constraint evaluation on dicts"""
    print("=" * 70)
    print("TEST: evaluate_min_constraints() - Dict-level MIN")
    print("=" * 70)

    # All constraints satisfied
    result = evaluate_min_constraints(
        {"rating": 4.0, "experience": 36},
        {"rating": [4.5, 5.0], "experience": [48, 60]}
    )
    assert result == True
    print("✓ All MIN constraints satisfied")

    # Missing required attribute
    result = evaluate_min_constraints(
        {"rating": 4.0},
        {"experience": [48, 60]}  # Missing "rating"
    )
    assert result == False
    print("✓ Missing required attribute → FAIL")

    # One constraint fails
    result = evaluate_min_constraints(
        {"rating": 4.0, "experience": 36},
        {"rating": [3.5, 5.0], "experience": [48, 60]}  # rating.min too low
    )
    assert result == False
    print("✓ One constraint violation → FAIL")

    # Empty required constraints (vacuously true)
    result = evaluate_min_constraints(
        {},
        {"rating": [3.0, 5.0]}
    )
    assert result == True
    print("✓ Empty constraints → VACUOUSLY TRUE")

    # Extra attributes in candidate (ignored)
    result = evaluate_min_constraints(
        {"rating": 4.0},
        {"rating": [4.5, 5.0], "experience": [48, 60], "extra": [10, 20]}
    )
    assert result == True
    print("✓ Extra candidate attributes ignored")

    print()


def test_evaluate_max_constraints():
    """Test MAX constraint evaluation on dicts"""
    print("=" * 70)
    print("TEST: evaluate_max_constraints() - Dict-level MAX")
    print("=" * 70)

    # Budget constraint satisfied
    result = evaluate_max_constraints(
        {"price": 100000},
        {"price": [95000, 95000]}
    )
    assert result == True
    print("✓ Budget constraint satisfied")

    # Price exceeds budget
    result = evaluate_max_constraints(
        {"price": 100000},
        {"price": [50000, 150000]}  # Range exceeds max
    )
    assert result == False
    print("✓ Price range exceeds budget → FAIL")

    # Multiple constraints
    result = evaluate_max_constraints(
        {"price": 100000, "delivery_days": 7},
        {"price": [95000, 95000], "delivery_days": [3, 5]}
    )
    assert result == True
    print("✓ Multiple MAX constraints satisfied")

    # Empty constraints
    result = evaluate_max_constraints(
        {},
        {"price": [1000000, 1000000]}
    )
    assert result == True
    print("✓ Empty MAX constraints → VACUOUSLY TRUE")

    print()


def test_evaluate_range_constraints():
    """Test RANGE constraint evaluation on dicts"""
    print("=" * 70)
    print("TEST: evaluate_range_constraints() - Dict-level RANGE")
    print("=" * 70)

    # EXACT match
    result = evaluate_range_constraints(
        {"storage": [256, 256]},  # EXACT 256GB
        {"storage": [256, 256]}
    )
    assert result == True
    print("✓ EXACT constraint satisfied [256, 256] = [256, 256]")

    # EXACT mismatch
    result = evaluate_range_constraints(
        {"storage": [256, 256]},  # EXACT 256GB
        {"storage": [512, 512]}   # EXACT 512GB
    )
    assert result == False
    print("✓ EXACT mismatch [256, 256] ≠ [512, 512] → FAIL")

    # EXACT vs range (too wide)
    result = evaluate_range_constraints(
        {"storage": [256, 256]},  # EXACT 256GB
        {"storage": [128, 512]}   # Range
    )
    assert result == False
    print("✓ EXACT vs wide range → FAIL (includes non-exact values)")

    # Value within range
    result = evaluate_range_constraints(
        {"price": [40000, 60000]},
        {"price": [55000, 55000]}  # EXACT 55000
    )
    assert result == True
    print("✓ EXACT value [55000, 55000] within range [40000, 60000]")

    # Range within range
    result = evaluate_range_constraints(
        {"price": [40000, 60000]},
        {"price": [45000, 55000]}  # Narrower range
    )
    assert result == True
    print("✓ Narrow range [45000, 55000] ⊆ [40000, 60000]")

    # Range exceeds bounds
    result = evaluate_range_constraints(
        {"price": [40000, 60000]},
        {"price": [50000, 70000]}  # Exceeds max
    )
    assert result == False
    print("✓ Range exceeds bounds → FAIL")

    # Multiple RANGE constraints
    result = evaluate_range_constraints(
        {"storage": [256, 256], "ram": [8, 16]},
        {"storage": [256, 256], "ram": [16, 16]}  # RAM at upper bound
    )
    assert result == True
    print("✓ Multiple RANGE constraints satisfied")

    # Empty constraints
    result = evaluate_range_constraints(
        {},
        {"storage": [128, 1024]}
    )
    assert result == True
    print("✓ Empty RANGE constraints → VACUOUSLY TRUE")

    print()


def test_real_world_scenarios():
    """Test realistic matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Buyer wants iPhone under 1 lakh budget
    # Seller offers at 95k
    print("\nScenario 1: Product Budget Match")
    print("  Buyer: max price = 100000")
    print("  Seller: price = [95000, 95000]")
    result = evaluate_max_constraints(
        {"price": 100000},
        {"price": [95000, 95000]}
    )
    assert result == True
    print("  ✓ MATCH - within budget")

    # Scenario 2: Service seeker wants min 5 years experience
    # Provider has 7 years
    print("\nScenario 2: Service Experience Match")
    print("  Seeker: min experience = 60 months")
    print("  Provider: experience = [84, 84]")
    result = evaluate_min_constraints(
        {"experience": 60},
        {"experience": [84, 84]}
    )
    assert result == True
    print("  ✓ MATCH - sufficient experience")

    # Scenario 3: Mutual - age range compatibility
    # A wants 25-30, B is 27
    print("\nScenario 3: Mutual Age Range Match")
    print("  A wants: age in [25, 30]")
    print("  B is: age = [27, 27]")
    result = evaluate_range_constraints(
        {"age": [25, 30]},
        {"age": [27, 27]}
    )
    assert result == True
    print("  ✓ MATCH - age within desired range")

    # Scenario 4: Combined constraints (all types)
    print("\nScenario 4: Combined Constraints")
    print("  Required: min rating=4.0, max price=100000, range storage=[256,512]")
    print("  Candidate: rating=[4.5,5.0], price=[95000,95000], storage=[256,256]")

    min_ok = evaluate_min_constraints(
        {"rating": 4.0},
        {"rating": [4.5, 5.0]}
    )
    max_ok = evaluate_max_constraints(
        {"price": 100000},
        {"price": [95000, 95000]}
    )
    range_ok = evaluate_range_constraints(
        {"storage": [256, 512]},
        {"storage": [256, 256]}
    )

    assert min_ok and max_ok and range_ok
    print("  ✓ ALL CONSTRAINTS SATISFIED")

    print()


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Zero values
    assert satisfies_min_constraint(0, [0, 10]) == True
    print("✓ Zero threshold with zero min")

    # Negative values
    assert satisfies_min_constraint(-10, [-5, 5]) == True
    print("✓ Negative threshold satisfied")

    # Very large numbers
    assert satisfies_max_constraint(1000000, [999999, 999999]) == True
    print("✓ Large numbers handled correctly")

    # Unbounded ranges
    result = evaluate_range_constraints(
        {"value": [NEGATIVE_INFINITY, POSITIVE_INFINITY]},
        {"value": [100, 200]}
    )
    assert result == True
    print("✓ Finite range within unbounded range")

    # Single point range (EXACT)
    assert range_contains([42, 42], [42, 42]) == True
    print("✓ Single point EXACT match")

    # Boundary exact match on min
    assert satisfies_min_constraint(100, [100, 100]) == True
    print("✓ Exact match at min boundary")

    # Boundary exact match on max
    assert satisfies_max_constraint(100, [100, 100]) == True
    print("✓ Exact match at max boundary")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NUMERIC CONSTRAINTS TEST SUITE")
    print("Testing range-based constraint evaluation (Phase 2.2)")
    print("=" * 70)
    print()

    test_range_contains()
    test_satisfies_min_constraint()
    test_satisfies_max_constraint()
    test_evaluate_min_constraints()
    test_evaluate_max_constraints()
    test_evaluate_range_constraints()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-09, M-10, M-11: Item-level numeric constraints implemented")
    print("✓ M-14, M-15, M-16: Other→Self numeric constraints implemented")
    print("✓ M-19, M-20, M-21: Self→Other numeric constraints implemented")
    print("✓ Range semantics: No scalar collapse, full range evaluation")
    print("✓ EXACT constraints: Represented as range[x,x] (I-02 compliance)")
    print("✓ Empty constraints: Vacuously satisfied (returns True)")
    print("✓ Missing attributes: Required attributes must be present (returns False)")
    print("✓ Boundary cases: Exact matches at thresholds work correctly")
    print()
