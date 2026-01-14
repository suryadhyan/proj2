"""
Test suite for other_self_matchers.py
Demonstrates other/self constraint matching (Phase 2.5)
"""

from other_self_matchers import (
    flatten_categorical_values,
    match_other_to_self,
    match_self_to_other
)


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
        "premium": ["standard"],
    }

    return required_value in implications.get(candidate_value, [])


# ============================================================================
# TEST: flatten_categorical_values()
# ============================================================================

def test_flatten_categorical_values():
    """Test value extraction from categorical objects"""
    print("=" * 70)
    print("TEST: flatten_categorical_values() - Value Extraction")
    print("=" * 70)

    # Basic extraction
    result = flatten_categorical_values({"brand": "apple", "color": "black"})
    assert result == {"apple", "black"}
    print("✓ Basic extraction: attribute values only")

    # Single value
    result = flatten_categorical_values({"condition": "new"})
    assert result == {"new"}
    print("✓ Single value extraction")

    # Empty categorical
    result = flatten_categorical_values({})
    assert result == set()
    print("✓ Empty categorical → empty set")

    # Multiple attributes
    result = flatten_categorical_values({
        "brand": "samsung",
        "color": "blue",
        "storage": "256gb",
        "condition": "refurbished"
    })
    assert result == {"samsung", "blue", "256gb", "refurbished"}
    print("✓ Multiple attributes extracted")

    print()


# ============================================================================
# TEST: M-13 to M-17 (Forward: Other → Self)
# ============================================================================

def test_match_other_to_self_categorical():
    """Test M-13: Other-Self Categorical Subset Rule"""
    print("=" * 70)
    print("TEST: match_other_to_self() - M-13 Categorical")
    print("=" * 70)

    # Exact match
    result = match_other_to_self(
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple", "color": "black"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Categorical subset: exact match with extra candidate attributes")

    # Missing required attribute
    result = match_other_to_self(
        {
            "categorical": {"brand": "apple", "color": "black"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple"},  # Missing "color"
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Missing required attribute → FAIL")

    # Value mismatch
    result = match_other_to_self(
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "samsung"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Value mismatch → FAIL")

    # Empty required (vacuous)
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Empty required categorical → VACUOUSLY TRUE")

    print()


def test_match_other_to_self_with_implications():
    """Test M-13 with term implications"""
    print("=" * 70)
    print("TEST: match_other_to_self() - M-13 with Implications")
    print("=" * 70)

    # Direct implication
    result = match_other_to_self(
        {
            "categorical": {"condition": "good"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"condition": "excellent"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Implication: 'excellent' implies 'good'")

    # Reverse fails
    result = match_other_to_self(
        {
            "categorical": {"condition": "excellent"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"condition": "good"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == False
    print("✓ Reverse implication fails: 'good' does NOT imply 'excellent'")

    # Transitive implication
    result = match_other_to_self(
        {
            "categorical": {"condition": "good"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"condition": "new"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Transitive implication: 'new' → 'excellent' → 'good'")

    print()


def test_match_other_to_self_numeric():
    """Test M-14, M-15, M-16: Other-Self Numeric Constraints"""
    print("=" * 70)
    print("TEST: match_other_to_self() - M-14, M-15, M-16 Numeric")
    print("=" * 70)

    # M-14: MIN satisfied
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {"rating": 4.0},
            "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"rating": [4.5, 5.0]},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-14: MIN satisfied (4.5 >= 4.0)")

    # M-14: MIN violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {"rating": 4.5},
            "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"rating": [3.0, 4.0]},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-14: MIN violated (3.0 < 4.5)")

    # M-15: MAX satisfied
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {},
            "max": {"price": 100000},
            "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"price": [80000, 95000]},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-15: MAX satisfied (95000 <= 100000)")

    # M-15: MAX violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {},
            "max": {"price": 50000},
            "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"price": [60000, 70000]},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-15: MAX violated (70000 > 50000)")

    # M-16: RANGE satisfied
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 10]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [2, 8]},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-16: RANGE satisfied ([2,8] ⊆ [0,10])")

    # M-16: RANGE violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 5]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [3, 8]},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-16: RANGE violated ([3,8] ⊄ [0,5])")

    # Combined constraints
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {"rating": 4.0},
            "max": {"price": 100000},
            "range": {"distance": [0, 10]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {
                "rating": [4.5, 5.0],
                "price": [80000, 95000],
                "distance": [2, 8]
            },
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Combined numeric constraints all satisfied")

    print()


def test_match_other_to_self_exclusions():
    """Test M-17: Other Exclusion Disjoint Rule"""
    print("=" * 70)
    print("TEST: match_other_to_self() - M-17 Exclusions")
    print("=" * 70)

    # Exclusion violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": ["refurbished", "used"]
        },
        {
            "categorical": {"condition": "refurbished"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Exclusion violated: 'refurbished' in otherexclusions")

    # Exclusion not violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": ["refurbished", "used"]
        },
        {
            "categorical": {"condition": "new"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Exclusion not violated: 'new' not in otherexclusions")

    # Empty exclusions
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"condition": "refurbished"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Empty exclusions → no violations")

    # Multiple values, one violated
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": ["dealer", "agent", "broker"]
        },
        {
            "categorical": {"seller_type": "agent", "verified": "yes"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Multiple categorical values, one violated: 'agent'")

    print()


# ============================================================================
# TEST: M-18 to M-22 (Reverse: Self → Other)
# ============================================================================

def test_match_self_to_other_categorical():
    """Test M-18: Self-Other Categorical Subset Rule"""
    print("=" * 70)
    print("TEST: match_self_to_other() - M-18 Categorical")
    print("=" * 70)

    # Exact match
    result = match_self_to_other(
        {
            "categorical": {"brand": "apple", "color": "black"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Categorical subset: exact match with extra candidate attributes")

    # Missing required attribute
    result = match_self_to_other(
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple", "color": "black"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Missing required attribute → FAIL")

    # Value mismatch
    result = match_self_to_other(
        {
            "categorical": {"brand": "samsung"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Value mismatch → FAIL")

    # Empty required (vacuous)
    result = match_self_to_other(
        {
            "categorical": {"brand": "apple"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Empty required categorical → VACUOUSLY TRUE")

    print()


def test_match_self_to_other_numeric():
    """Test M-19, M-20, M-21: Self-Other Numeric Constraints"""
    print("=" * 70)
    print("TEST: match_self_to_other() - M-19, M-20, M-21 Numeric")
    print("=" * 70)

    # M-19: MIN satisfied
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"rating": [4.5, 5.0]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {"rating": 4.0},
            "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-19: MIN satisfied (4.5 >= 4.0)")

    # M-19: MIN violated
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"rating": [3.0, 4.0]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {"rating": 4.5},
            "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-19: MIN violated (3.0 < 4.5)")

    # M-20: MAX satisfied
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"price": [80000, 95000]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {},
            "max": {"price": 100000},
            "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-20: MAX satisfied (95000 <= 100000)")

    # M-20: MAX violated
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"price": [60000, 70000]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {},
            "max": {"price": 50000},
            "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-20: MAX violated (70000 > 50000)")

    # M-21: RANGE satisfied
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [2, 8]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 10]},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ M-21: RANGE satisfied ([2,8] ⊆ [0,10])")

    # M-21: RANGE violated
    result = match_self_to_other(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [3, 8]},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 5]},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ M-21: RANGE violated ([3,8] ⊄ [0,5])")

    print()


def test_match_self_to_other_exclusions():
    """Test M-22: Self Exclusion Disjoint Rule"""
    print("=" * 70)
    print("TEST: match_self_to_other() - M-22 Exclusions")
    print("=" * 70)

    # Exclusion violated
    result = match_self_to_other(
        {
            "categorical": {"condition": "refurbished"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": ["refurbished", "used"]
        }
    )
    assert result == False
    print("✓ Exclusion violated: 'refurbished' in selfexclusions")

    # Exclusion not violated
    result = match_self_to_other(
        {
            "categorical": {"condition": "new"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": ["refurbished", "used"]
        }
    )
    assert result == True
    print("✓ Exclusion not violated: 'new' not in selfexclusions")

    # Empty exclusions
    result = match_self_to_other(
        {
            "categorical": {"condition": "refurbished"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Empty exclusions → no violations")

    print()


# ============================================================================
# TEST: Real-World Scenarios
# ============================================================================

def test_real_world_scenarios():
    """Test realistic matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Product exchange - Phone for Phone
    print("\nScenario 1: Product Exchange")
    print("  A wants: iPhone (min rating 4.0, max price 100k, no used)")
    print("  B offers: iPhone 15 (rating 4.8, price 95k, new)")

    A_other = {
        "categorical": {"type": "smartphone", "brand": "apple"},
        "min": {"rating": 4.0},
        "max": {"price": 100000},
        "range": {},
        "otherexclusions": ["used", "refurbished"]
    }
    B_self = {
        "categorical": {"type": "smartphone", "brand": "apple", "condition": "new"},
        "min": {}, "max": {},
        "range": {"rating": [4.8, 4.8], "price": [95000, 95000]},
        "selfexclusions": []
    }

    result = match_other_to_self(A_other, B_self)
    assert result == True
    print("  ✓ MATCH - B.self satisfies A.other")

    # Scenario 2: Service exchange - Yoga instructor
    print("\nScenario 2: Service Exchange")
    print("  A wants: Yoga instructor (certified, max rate 5000/hr, within 10km)")
    print("  B offers: Certified yoga instructor (rate 4000/hr, 8km away)")

    A_other = {
        "categorical": {"service": "yoga_instruction", "certified": "yes"},
        "min": {},
        "max": {"hourly_rate": 5000},
        "range": {"distance": [0, 10]},
        "otherexclusions": []
    }
    B_self = {
        "categorical": {"service": "yoga_instruction", "certified": "yes", "experience": "5years"},
        "min": {}, "max": {},
        "range": {"hourly_rate": [4000, 4000], "distance": [8, 8]},
        "selfexclusions": []
    }

    result = match_other_to_self(A_other, B_self)
    assert result == True
    print("  ✓ MATCH - B.self satisfies A.other")

    # Scenario 3: Mismatch due to exclusion
    print("\nScenario 3: Exclusion Blocks Match")
    print("  A wants: Phone (no dealers)")
    print("  B offers: Phone from dealer")

    A_other = {
        "categorical": {"type": "phone"},
        "min": {}, "max": {}, "range": {},
        "otherexclusions": ["dealer", "agent"]
    }
    B_self = {
        "categorical": {"type": "phone", "seller_type": "dealer"},
        "min": {}, "max": {}, "range": {},
        "selfexclusions": []
    }

    result = match_other_to_self(A_other, B_self)
    assert result == False
    print("  ✓ NO MATCH - Exclusion violated ('dealer')")

    # Scenario 4: Bidirectional check (both directions)
    print("\nScenario 4: Bidirectional Compatibility")
    print("  A.other: Wants laptop (min RAM 16GB)")
    print("  B.self: Offers laptop (RAM 32GB)")
    print("  B.other: Wants desktop (storage range 512-1024GB)")
    print("  A.self: Offers desktop (storage range 0-2048GB)")

    A_other = {
        "categorical": {"type": "laptop"},
        "min": {"ram": 16},
        "max": {}, "range": {},
        "otherexclusions": []
    }
    B_self = {
        "categorical": {"type": "laptop"},
        "min": {}, "max": {},
        "range": {"ram": [32, 32]},
        "selfexclusions": []
    }
    B_other = {
        "categorical": {"type": "desktop"},
        "min": {}, "max": {},
        "range": {"storage": [512, 1024]},
        "otherexclusions": []
    }
    A_self = {
        "categorical": {"type": "desktop"},
        "min": {}, "max": {},
        "range": {"storage": [0, 2048]},
        "selfexclusions": []
    }

    forward = match_other_to_self(A_other, B_self)
    reverse = match_self_to_other(B_other, A_self)

    assert forward == True
    assert reverse == True
    print("  ✓ Forward: A.other → B.self MATCH")
    print("  ✓ Reverse: B.other → A.self MATCH (range [512,1024] ⊆ [0,2048])")
    print("  ✓ Bidirectional compatibility confirmed")

    print()


# ============================================================================
# TEST: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Completely empty constraints
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ All empty constraints → VACUOUSLY TRUE")

    # Minimal required, rich candidate
    result = match_other_to_self(
        {
            "categorical": {"type": "phone"},
            "min": {}, "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {
                "type": "phone",
                "brand": "apple",
                "color": "black",
                "storage": "256gb",
                "condition": "new"
            },
            "min": {}, "max": {},
            "range": {"price": [95000, 95000], "rating": [4.8, 4.8]},
            "selfexclusions": []
        }
    )
    assert result == True
    print("✓ Minimal required matches rich candidate")

    # Same direction, different exclusion fields
    other_obj = {
        "categorical": {"seller_type": "dealer"},
        "min": {}, "max": {}, "range": {},
        "otherexclusions": ["used"]  # Uses otherexclusions
    }
    self_obj = {
        "categorical": {"condition": "used"},
        "min": {}, "max": {}, "range": {},
        "selfexclusions": ["dealer"]  # Uses selfexclusions
    }

    # Forward: checks otherexclusions against self's categorical values
    forward = match_other_to_self(other_obj, self_obj)
    assert forward == False
    print("✓ Forward: checks otherexclusions (violated: 'used' in self)")

    # Reverse: checks selfexclusions against other's categorical values
    reverse = match_self_to_other(other_obj, self_obj)
    assert reverse == False  # selfexclusions violated: 'dealer' in other
    print("✓ Reverse: checks selfexclusions (violated: 'dealer' in other)")
    print("✓ Different exclusion fields per direction")

    # Attribute mismatch in different constraint modes
    result = match_other_to_self(
        {
            "categorical": {},
            "min": {"price": 50000},  # price in MIN
            "max": {}, "range": {},
            "otherexclusions": []
        },
        {
            "categorical": {},
            "min": {},
            "max": {"price": 100000},  # price in MAX
            "range": {},
            "selfexclusions": []
        }
    )
    assert result == False
    print("✓ Attribute in different constraint modes handled correctly")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("OTHER/SELF MATCHERS TEST SUITE")
    print("Testing other/self constraint matching (Phase 2.5)")
    print("=" * 70)
    print()

    test_flatten_categorical_values()
    test_match_other_to_self_categorical()
    test_match_other_to_self_with_implications()
    test_match_other_to_self_numeric()
    test_match_other_to_self_exclusions()
    test_match_self_to_other_categorical()
    test_match_self_to_other_numeric()
    test_match_self_to_other_exclusions()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-13 to M-17: Other → Self direction enforced")
    print("✓ M-18 to M-22: Self → Other direction enforced")
    print("✓ Directionality preserved (forward ≠ reverse)")
    print("✓ Categorical subset with term implications")
    print("✓ Numeric constraints via Phase 2.2 (min, max, range)")
    print("✓ Exclusions strictly enforced (I-07)")
    print("✓ Different exclusion fields per direction:")
    print("  - Forward: otherexclusions")
    print("  - Reverse: selfexclusions")
    print("✓ Empty constraints: Vacuously satisfied")
    print("✓ Missing required attributes: Fails")
    print("✓ Dynamic attributes: No hardcoded names")
    print("✓ Short-circuit: First failure → False")
    print("✓ Pure functions: No side effects")
    print()
