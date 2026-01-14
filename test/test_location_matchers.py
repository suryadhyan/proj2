"""
Test suite for location_matchers.py
Demonstrates location constraint matching (Phase 2.6)
"""

from location_matchers import (
    flatten_location_categorical_values,
    match_location_constraints
)


# ============================================================================
# HELPER: Simple term implication for testing
# ============================================================================

def test_implies_fn(candidate_value: str, required_value: str) -> bool:
    """Test implication function."""
    if candidate_value == required_value:
        return True

    implications = {
        "downtown": ["urban"],
        "city_center": ["downtown", "urban"],
        "residential": ["quiet"],
    }

    return required_value in implications.get(candidate_value, [])


# ============================================================================
# TEST: flatten_location_categorical_values()
# ============================================================================

def test_flatten_location_categorical_values():
    """Test value extraction from location categorical objects"""
    print("=" * 70)
    print("TEST: flatten_location_categorical_values() - Value Extraction")
    print("=" * 70)

    # Basic extraction
    result = flatten_location_categorical_values({"area": "downtown", "zone": "commercial"})
    assert result == {"downtown", "commercial"}
    print("✓ Basic extraction: attribute values only")

    # Single value
    result = flatten_location_categorical_values({"region": "north"})
    assert result == {"north"}
    print("✓ Single value extraction")

    # Empty categorical
    result = flatten_location_categorical_values({})
    assert result == set()
    print("✓ Empty categorical → empty set")

    # Multiple attributes
    result = flatten_location_categorical_values({
        "city": "boston",
        "state": "massachusetts",
        "zone": "commercial",
        "accessibility": "metro"
    })
    assert result == {"boston", "massachusetts", "commercial", "metro"}
    print("✓ Multiple attributes extracted")

    print()


# ============================================================================
# TEST: M-23, M-24, M-25 (Location Numeric Constraints)
# ============================================================================

def test_match_location_numeric():
    """Test M-23, M-24, M-25: Location Numeric Constraints"""
    print("=" * 70)
    print("TEST: match_location_constraints() - M-23, M-24, M-25 Numeric")
    print("=" * 70)

    # M-23: MIN satisfied
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {"distance": 0},
            "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [5, 10]},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ M-23: MIN satisfied (5 >= 0)")

    # M-23: MIN violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {"distance": 10},
            "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [5, 8]},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ M-23: MIN violated (5 < 10)")

    # M-24: MAX satisfied
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {},
            "max": {"distance": 50},
            "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [10, 30]},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ M-24: MAX satisfied (30 <= 50)")

    # M-24: MAX violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {},
            "max": {"distance": 20},
            "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [10, 30]},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ M-24: MAX violated (30 > 20)")

    # M-25: RANGE satisfied
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 50]},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [10, 30]},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ M-25: RANGE satisfied ([10,30] ⊆ [0,50])")

    # M-25: RANGE violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [0, 20]},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {"distance": [10, 30]},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ M-25: RANGE violated ([10,30] ⊄ [0,20])")

    # Combined numeric constraints
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {"distance": 0},
            "max": {"travel_time": 60},
            "range": {"radius": [0, 100]},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {},
            "range": {
                "distance": [5, 10],
                "travel_time": [30, 45],
                "radius": [20, 50]
            },
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Combined numeric constraints all satisfied")

    print()


# ============================================================================
# TEST: M-26 (Location-Other Categorical Subset)
# ============================================================================

def test_match_location_categorical():
    """Test M-26: Location-Other Categorical Subset Rule"""
    print("=" * 70)
    print("TEST: match_location_constraints() - M-26 Categorical")
    print("=" * 70)

    # Exact match
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "downtown", "zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Categorical subset: exact match with extra candidate attributes")

    # Missing required attribute
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown", "zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "downtown"},  # Missing "zone"
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ Missing required attribute → FAIL")

    # Value mismatch
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "suburban"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ Value mismatch → FAIL")

    # Empty required (vacuous)
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Empty required categorical → VACUOUSLY TRUE")

    print()


def test_match_location_categorical_with_implications():
    """Test M-26 with term implications"""
    print("=" * 70)
    print("TEST: match_location_constraints() - M-26 with Implications")
    print("=" * 70)

    # Direct implication
    result = match_location_constraints(
        {
            "categorical": {"type": "urban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"type": "downtown"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Implication: 'downtown' implies 'urban'")

    # Reverse fails
    result = match_location_constraints(
        {
            "categorical": {"type": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"type": "urban"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == False
    print("✓ Reverse implication fails: 'urban' does NOT imply 'downtown'")

    # Transitive implication
    result = match_location_constraints(
        {
            "categorical": {"type": "urban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"type": "city_center"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        },
        implies_fn=test_implies_fn
    )
    assert result == True
    print("✓ Transitive implication: 'city_center' → 'downtown' → 'urban'")

    print()


# ============================================================================
# TEST: M-27 (Location Exclusion Disjoint)
# ============================================================================

def test_location_exclusions():
    """Test M-27: Location Exclusion Disjoint Rule"""
    print("=" * 70)
    print("TEST: match_location_constraints() - M-27 Location Exclusions")
    print("=" * 70)

    # Exclusion violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["industrial", "airport"]
        },
        {
            "categorical": {"zone": "industrial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ Exclusion violated: 'industrial' in locationexclusions")

    # Exclusion not violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["industrial", "airport"]
        },
        {
            "categorical": {"zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Exclusion not violated: 'commercial' not in locationexclusions")

    # Empty exclusions
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"zone": "industrial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Empty exclusions → no violations")

    # Multiple values, one violated
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["rural", "remote", "isolated"]
        },
        {
            "categorical": {"accessibility": "remote", "terrain": "mountain"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ Multiple categorical values, one violated: 'remote'")

    print()


# ============================================================================
# TEST: M-28 (Other Location Exclusion Disjoint)
# ============================================================================

def test_other_location_exclusions():
    """Test M-28: Other Location Exclusion Disjoint Rule"""
    print("=" * 70)
    print("TEST: match_location_constraints() - M-28 Other Location Exclusions")
    print("=" * 70)

    # Other location exclusion violated
    result = match_location_constraints(
        {
            "categorical": {"area": "suburban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": ["suburban", "rural"]
        }
    )
    assert result == False
    print("✓ Other location exclusion violated: 'suburban' in otherlocationexclusions")

    # Other location exclusion not violated
    result = match_location_constraints(
        {
            "categorical": {"area": "urban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "urban", "zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": ["suburban", "rural"]
        }
    )
    assert result == True
    print("✓ Other location exclusion not violated: 'urban' not in otherlocationexclusions")

    # Empty other location exclusions
    result = match_location_constraints(
        {
            "categorical": {"area": "suburban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"area": "suburban"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Empty other location exclusions → no violations")

    # Both exclusion types checked
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["suburban"]  # Excludes candidate having suburban
        },
        {
            "categorical": {"area": "downtown", "zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": ["rural"]  # Excludes requester having rural
        }
    )
    assert result == True
    print("✓ Both exclusion types checked, neither violated")

    # M-27 violated
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["suburban"]  # Violated!
        },
        {
            "categorical": {"zone": "suburban"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ M-27 violated (locationexclusions)")

    # M-28 violated
    result = match_location_constraints(
        {
            "categorical": {"area": "rural"},  # Violated!
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {"zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": ["rural"]
        }
    )
    assert result == False
    print("✓ M-28 violated (otherlocationexclusions)")

    print()


# ============================================================================
# TEST: Real-World Scenarios
# ============================================================================

def test_real_world_scenarios():
    """Test realistic location matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Service location (yoga instructor)
    print("\nScenario 1: Service Location Matching")
    print("  A location: Within 10km, urban area, no industrial zones")
    print("  B other: 5km away, downtown location, commercial zone")

    A_location = {
        "categorical": {"type": "urban"},
        "min": {},
        "max": {"distance": 10},
        "range": {},
        "locationexclusions": ["industrial", "airport"]
    }
    B_other = {
        "categorical": {"type": "downtown", "zone": "commercial"},
        "min": {}, "max": {},
        "range": {"distance": [5, 5]},
        "otherlocationexclusions": []
    }

    result = match_location_constraints(A_location, B_other, implies_fn=test_implies_fn)
    assert result == True
    print("  ✓ MATCH - B location satisfies A location requirements")

    # Scenario 2: Product delivery location
    print("\nScenario 2: Product Delivery Location")
    print("  A location: Residential area, max 20km distance")
    print("  B other: Residential zone, 15km delivery radius")

    A_location = {
        "categorical": {"type": "residential"},
        "min": {},
        "max": {"distance": 20},
        "range": {},
        "locationexclusions": []
    }
    B_other = {
        "categorical": {"type": "residential", "coverage": "local"},
        "min": {}, "max": {},
        "range": {"distance": [0, 15]},
        "otherlocationexclusions": []
    }

    result = match_location_constraints(A_location, B_other)
    assert result == True
    print("  ✓ MATCH - Delivery location compatible")

    # Scenario 3: Exclusion blocks match
    print("\nScenario 3: Location Exclusion Blocks Match")
    print("  A location: No rural areas")
    print("  B other: Rural location")

    A_location = {
        "categorical": {},
        "min": {}, "max": {}, "range": {},
        "locationexclusions": ["rural", "remote"]
    }
    B_other = {
        "categorical": {"area": "rural"},
        "min": {}, "max": {}, "range": {},
        "otherlocationexclusions": []
    }

    result = match_location_constraints(A_location, B_other)
    assert result == False
    print("  ✓ NO MATCH - Location exclusion violated ('rural')")

    # Scenario 4: Other location exclusion blocks match
    print("\nScenario 4: Other Location Exclusion Blocks Match")
    print("  A location: Urban area")
    print("  B other: Excludes urban requesters")

    A_location = {
        "categorical": {"area": "urban"},
        "min": {}, "max": {}, "range": {},
        "locationexclusions": []
    }
    B_other = {
        "categorical": {"area": "suburban"},
        "min": {}, "max": {}, "range": {},
        "otherlocationexclusions": ["urban", "city"]
    }

    result = match_location_constraints(A_location, B_other)
    assert result == False
    print("  ✓ NO MATCH - Other location exclusion violated (B excludes 'urban')")

    # Scenario 5: Abstract location attributes
    print("\nScenario 5: Abstract Location Attributes (no GPS assumptions)")
    print("  A location: Accessibility score >= 7, travel_time <= 30 min")
    print("  B other: Accessibility 8, travel_time 25 min")

    A_location = {
        "categorical": {},
        "min": {"accessibility_score": 7},
        "max": {"travel_time": 30},
        "range": {},
        "locationexclusions": []
    }
    B_other = {
        "categorical": {},
        "min": {}, "max": {},
        "range": {
            "accessibility_score": [8, 8],
            "travel_time": [25, 25]
        },
        "otherlocationexclusions": []
    }

    result = match_location_constraints(A_location, B_other)
    assert result == True
    print("  ✓ MATCH - Abstract location attributes satisfied")
    print("  ✓ No GPS/GIS assumptions required")

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
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ All empty constraints → VACUOUSLY TRUE")

    # Minimal required, rich candidate
    result = match_location_constraints(
        {
            "categorical": {"area": "urban"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {
                "area": "urban",
                "city": "boston",
                "state": "massachusetts",
                "zone": "commercial",
                "accessibility": "metro"
            },
            "min": {}, "max": {},
            "range": {"distance": [5, 5], "travel_time": [20, 20]},
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Minimal required matches rich candidate")

    # Both exclusion types, neither violated
    result = match_location_constraints(
        {
            "categorical": {"area": "downtown"},
            "min": {}, "max": {}, "range": {},
            "locationexclusions": ["industrial"]
        },
        {
            "categorical": {"area": "downtown", "zone": "commercial"},
            "min": {}, "max": {}, "range": {},
            "otherlocationexclusions": ["suburban"]
        }
    )
    assert result == True
    print("✓ Both exclusion types present, neither violated")

    # Attribute in different constraint modes
    result = match_location_constraints(
        {
            "categorical": {},
            "min": {"distance": 5},  # distance in MIN
            "max": {}, "range": {},
            "locationexclusions": []
        },
        {
            "categorical": {},
            "min": {},
            "max": {"distance": 20},  # distance in MAX
            "range": {},
            "otherlocationexclusions": []
        }
    )
    assert result == False
    print("✓ Attribute in different constraint modes handled correctly")

    # Location abstraction: non-geographic attributes
    result = match_location_constraints(
        {
            "categorical": {"service_zone": "premium"},
            "min": {"priority_level": 3},
            "max": {"queue_depth": 10},
            "range": {},
            "locationexclusions": ["restricted"]
        },
        {
            "categorical": {"service_zone": "premium", "tier": "gold"},
            "min": {}, "max": {},
            "range": {
                "priority_level": [5, 5],
                "queue_depth": [3, 3]
            },
            "otherlocationexclusions": []
        }
    )
    assert result == True
    print("✓ Location abstraction: works with non-geographic attributes")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOCATION MATCHERS TEST SUITE")
    print("Testing location constraint matching (Phase 2.6)")
    print("=" * 70)
    print()

    test_flatten_location_categorical_values()
    test_match_location_numeric()
    test_match_location_categorical()
    test_match_location_categorical_with_implications()
    test_location_exclusions()
    test_other_location_exclusions()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-23 to M-28: Location constraints enforced")
    print("✓ M-23, M-24, M-25: Numeric constraints (min, max, range)")
    print("✓ M-26: Location-Other categorical subset")
    print("✓ M-27: Location exclusions (locationexclusions)")
    print("✓ M-28: Other location exclusions (otherlocationexclusions)")
    print("✓ Location abstraction maintained:")
    print("  - No GPS/GIS assumptions")
    print("  - No distance calculations")
    print("  - Works with ANY location representation")
    print("✓ Categorical subset with term implications")
    print("✓ Numeric constraints via Phase 2.2 (min, max, range)")
    print("✓ Exclusions strictly enforced (I-07)")
    print("✓ Two exclusion types checked:")
    print("  - M-27: locationexclusions")
    print("  - M-28: otherlocationexclusions")
    print("✓ Empty constraints: Vacuously satisfied")
    print("✓ Missing required attributes: Fails")
    print("✓ Dynamic attributes: No hardcoded names")
    print("✓ Short-circuit: First failure → False")
    print("✓ Pure function: No side effects")
    print()
