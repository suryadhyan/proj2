"""
Test suite for item_array_matchers.py
Demonstrates item array matching with exclusions (Phase 2.4)
"""

from item_array_matchers import (
    flatten_item_values,
    violates_item_exclusions,
    required_item_has_match,
    all_required_items_match
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
    }

    return required_value in implications.get(candidate_value, [])


# ============================================================================
# TEST: flatten_item_values()
# ============================================================================

def test_flatten_item_values():
    """Test string value extraction from items"""
    print("=" * 70)
    print("TEST: flatten_item_values() - String Extraction")
    print("=" * 70)

    # Basic extraction
    result = flatten_item_values({
        "type": "smartphone",
        "categorical": {"brand": "apple", "color": "black"}
    })
    assert result == {"smartphone", "apple", "black"}
    print("✓ Basic extraction: type + categorical values")

    # With condition attribute
    result = flatten_item_values({
        "type": "laptop",
        "categorical": {"brand": "dell", "condition": "refurbished"}
    })
    assert result == {"laptop", "dell", "refurbished"}
    print("✓ Extracts 'refurbished' from categorical")

    # Type only
    result = flatten_item_values({
        "type": "phone",
        "categorical": {}
    })
    assert result == {"phone"}
    print("✓ Type-only item")

    # Does not extract numeric values
    result = flatten_item_values({
        "type": "phone",
        "categorical": {"brand": "apple"},
        "min": {"price": 50000},
        "max": {"price": 100000},
        "range": {"storage": [256, 256]}
    })
    # Should only get type and categorical values, NOT numeric
    assert result == {"phone", "apple"}
    print("✓ Does NOT extract numeric constraint values")

    # Nested values
    result = flatten_item_values({
        "type": "service",
        "categorical": {
            "mode": "home_visit",
            "availability": "weekends"
        }
    })
    assert result == {"service", "home_visit", "weekends"}
    print("✓ Extracts nested categorical values")

    print()


# ============================================================================
# TEST: M-12 Item Exclusion Disjoint Rule
# ============================================================================

def test_violates_item_exclusions():
    """Test M-12: Item Exclusion Disjoint Rule"""
    print("=" * 70)
    print("TEST: violates_item_exclusions() - M-12")
    print("=" * 70)

    # Exclusion violated
    result = violates_item_exclusions(
        {"itemexclusions": ["refurbished", "used"]},
        {"type": "phone", "categorical": {"condition": "refurbished"}}
    )
    assert result == True
    print("✓ Exclusion violated: 'refurbished' in candidate")

    # Exclusions not violated
    result = violates_item_exclusions(
        {"itemexclusions": ["refurbished", "used"]},
        {"type": "phone", "categorical": {"condition": "new"}}
    )
    assert result == False
    print("✓ Exclusions not violated: 'new' not in exclusions")

    # Empty exclusions (always valid)
    result = violates_item_exclusions(
        {"itemexclusions": []},
        {"type": "phone", "categorical": {"condition": "refurbished"}}
    )
    assert result == False
    print("✓ Empty exclusions → no violations")

    # Multiple exclusions, one violated
    result = violates_item_exclusions(
        {"itemexclusions": ["dealer", "agent", "broker"]},
        {"type": "service", "categorical": {"seller_type": "agent"}}
    )
    assert result == True
    print("✓ Multiple exclusions, one violated: 'agent'")

    # Multiple exclusions, none violated
    result = violates_item_exclusions(
        {"itemexclusions": ["dealer", "agent", "broker"]},
        {"type": "service", "categorical": {"seller_type": "individual"}}
    )
    assert result == False
    print("✓ Multiple exclusions, none violated")

    # Exclusion matches type
    result = violates_item_exclusions(
        {"itemexclusions": ["smartphone"]},
        {"type": "smartphone", "categorical": {}}
    )
    assert result == True
    print("✓ Exclusion can match type field")

    print()


# ============================================================================
# TEST: required_item_has_match()
# ============================================================================

def test_required_item_has_match():
    """Test single required item matching"""
    print("=" * 70)
    print("TEST: required_item_has_match() - Single Required Item")
    print("=" * 70)

    # Has valid match
    result = required_item_has_match(
        {
            "type": "phone",
            "itemexclusions": ["used"],
            "categorical": {},
            "min": {}, "max": {}, "range": {}
        },
        [
            {
                "type": "phone",
                "categorical": {"condition": "used"},  # Excluded
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "phone",
                "categorical": {"condition": "new"},  # Valid
                "min": {}, "max": {}, "range": {}
            }
        ]
    )
    assert result == True
    print("✓ Has valid match (second candidate not excluded)")

    # All candidates excluded
    result = required_item_has_match(
        {
            "type": "phone",
            "itemexclusions": ["used", "refurbished"],
            "categorical": {},
            "min": {}, "max": {}, "range": {}
        },
        [
            {
                "type": "phone",
                "categorical": {"condition": "used"},
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "phone",
                "categorical": {"condition": "refurbished"},
                "min": {}, "max": {}, "range": {}
            }
        ]
    )
    assert result == False
    print("✓ All candidates excluded → no match")

    # Empty candidate list
    result = required_item_has_match(
        {"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}},
        []
    )
    assert result == False
    print("✓ Empty candidate list → no match")

    # Type mismatch (no valid candidates)
    result = required_item_has_match(
        {"type": "smartphone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}},
        [{"type": "laptop", "categorical": {}, "min": {}, "max": {}, "range": {}}]
    )
    assert result == False
    print("✓ Type mismatch → no match")

    # First candidate valid (short-circuit)
    result = required_item_has_match(
        {"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}},
        [
            {"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}},  # Valid
            {"type": "laptop", "categorical": {}, "min": {}, "max": {}, "range": {}}  # Not checked
        ]
    )
    assert result == True
    print("✓ First candidate valid (short-circuit)")

    print()


# ============================================================================
# TEST: all_required_items_match()
# ============================================================================

def test_all_required_items_match():
    """Test array-level matching"""
    print("=" * 70)
    print("TEST: all_required_items_match() - Array Matching")
    print("=" * 70)

    # All required items matched
    result = all_required_items_match(
        [
            {"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}},
            {"type": "charger", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}}
        ],
        [
            {"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}},
            {"type": "charger", "categorical": {}, "min": {}, "max": {}, "range": {}},
            {"type": "case", "categorical": {}, "min": {}, "max": {}, "range": {}}  # Extra, ignored
        ]
    )
    assert result == True
    print("✓ All required items matched (extra candidate ignored)")

    # One required item without match
    result = all_required_items_match(
        [
            {"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}},
            {"type": "charger", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}}
        ],
        [
            {"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}
            # Missing charger
        ]
    )
    assert result == False
    print("✓ One required item without match → FAIL")

    # Empty required items (vacuously true)
    result = all_required_items_match(
        [],
        [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}]
    )
    assert result == True
    print("✓ Empty required items → VACUOUSLY TRUE")

    # Empty candidates with non-empty required
    result = all_required_items_match(
        [{"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}}],
        []
    )
    assert result == False
    print("✓ Empty candidates with non-empty required → FAIL")

    # Exclusions prevent all matches
    result = all_required_items_match(
        [
            {
                "type": "phone",
                "itemexclusions": ["used", "refurbished"],
                "categorical": {},
                "min": {}, "max": {}, "range": {}
            }
        ],
        [
            {"type": "phone", "categorical": {"condition": "used"}, "min": {}, "max": {}, "range": {}},
            {"type": "phone", "categorical": {"condition": "refurbished"}, "min": {}, "max": {}, "range": {}}
        ]
    )
    assert result == False
    print("✓ Exclusions prevent all matches → FAIL")

    # Same candidate matches multiple required items
    result = all_required_items_match(
        [
            {"type": "phone", "itemexclusions": [], "categorical": {"brand": "apple"}, "min": {}, "max": {}, "range": {}},
            {"type": "phone", "itemexclusions": [], "categorical": {"color": "black"}, "min": {}, "max": {}, "range": {}}
        ],
        [
            {"type": "phone", "categorical": {"brand": "apple", "color": "black"}, "min": {}, "max": {}, "range": {}}
        ]
    )
    assert result == True
    print("✓ Same candidate matches multiple required items")

    print()


# ============================================================================
# TEST: Real-World Scenarios
# ============================================================================

def test_real_world_scenarios():
    """Test realistic matching scenarios"""
    print("=" * 70)
    print("TEST: Real-World Scenarios")
    print("=" * 70)

    # Scenario 1: Product listing - iPhone with accessories
    print("\nScenario 1: Product Bundle - iPhone + Charger")
    print("  Buyer wants: iPhone (no used/refurbished) + Original charger")
    print("  Seller offers: New iPhone + Original charger + Case")

    result = all_required_items_match(
        [
            {
                "type": "smartphone",
                "itemexclusions": ["used", "refurbished"],
                "categorical": {"brand": "apple", "model": "iphone 15 pro"},
                "min": {}, "max": {"price": 100000}, "range": {}
            },
            {
                "type": "charger",
                "itemexclusions": [],
                "categorical": {"type": "original"},
                "min": {}, "max": {}, "range": {}
            }
        ],
        [
            {
                "type": "smartphone",
                "categorical": {"brand": "apple", "model": "iphone 15 pro", "condition": "new"},
                "min": {}, "max": {}, "range": {"price": [95000, 95000]}
            },
            {
                "type": "charger",
                "categorical": {"type": "original", "brand": "apple"},
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "case",
                "categorical": {"brand": "apple"},
                "min": {}, "max": {}, "range": {}
            }
        ]
    )
    assert result == True
    print("  ✓ MATCH - All required items present, no exclusions violated")

    # Scenario 2: Exclusion violation
    print("\nScenario 2: Exclusion Blocks Match")
    print("  Buyer wants: Phone (no dealers)")
    print("  Seller: Dealer offering phone")

    result = all_required_items_match(
        [
            {
                "type": "phone",
                "itemexclusions": ["dealer", "agent"],
                "categorical": {},
                "min": {}, "max": {}, "range": {}
            }
        ],
        [
            {
                "type": "phone",
                "categorical": {"seller_type": "dealer"},
                "min": {}, "max": {}, "range": {}
            }
        ]
    )
    assert result == False
    print("  ✓ NO MATCH - Exclusion violated ('dealer' in candidate)")

    # Scenario 3: Service listing - Multiple services
    print("\nScenario 3: Service Bundle")
    print("  Seeker wants: Yoga + Meditation classes")
    print("  Provider offers: Yoga, Meditation, Diet consultation")

    result = all_required_items_match(
        [
            {
                "type": "yoga_instruction",
                "itemexclusions": ["online"],
                "categorical": {"mode": "home_visit"},
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "meditation_class",
                "itemexclusions": [],
                "categorical": {},
                "min": {}, "max": {}, "range": {}
            }
        ],
        [
            {
                "type": "yoga_instruction",
                "categorical": {"mode": "home_visit", "certified": "true"},
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "meditation_class",
                "categorical": {"duration": "60min"},
                "min": {}, "max": {}, "range": {}
            },
            {
                "type": "diet_consultation",
                "categorical": {},
                "min": {}, "max": {}, "range": {}
            }
        ]
    )
    assert result == True
    print("  ✓ MATCH - Both required services available")

    print()


# ============================================================================
# TEST: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    print("=" * 70)
    print("TEST: Edge Cases")
    print("=" * 70)

    # Minimal required item (only type)
    result = all_required_items_match(
        [{"type": "phone", "itemexclusions": [], "categorical": {}, "min": {}, "max": {}, "range": {}}],
        [{"type": "phone", "categorical": {"brand": "apple", "color": "black"}, "min": {}, "max": {}, "range": {}}]
    )
    assert result == True
    print("✓ Minimal required item (only type) matches")

    # Exclusion list with no matches
    values = flatten_item_values({
        "type": "phone",
        "categorical": {"brand": "apple"}
    })
    assert "used" not in values
    print("✓ Exclusion 'used' not in candidate values")

    # Empty exclusions never violate
    for i in range(3):
        result = violates_item_exclusions(
            {"itemexclusions": []},
            {"type": "any", "categorical": {"anything": "here"}}
        )
        assert result == False
    print("✓ Empty exclusions never violated")

    # Both required and candidate arrays empty
    result = all_required_items_match([], [])
    assert result == True
    print("✓ Both arrays empty → VACUOUSLY TRUE")

    # Exclusion case sensitivity (already normalized)
    result = violates_item_exclusions(
        {"itemexclusions": ["refurbished"]},  # lowercase (normalized)
        {"type": "phone", "categorical": {"condition": "refurbished"}}  # lowercase (normalized)
    )
    assert result == True
    print("✓ Case-insensitive exclusion check (normalized by Phase 2.1)")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ITEM ARRAY MATCHERS TEST SUITE")
    print("Testing item array matching with exclusions (Phase 2.4)")
    print("=" * 70)
    print()

    test_flatten_item_values()
    test_violates_item_exclusions()
    test_required_item_has_match()
    test_all_required_items_match()
    test_real_world_scenarios()
    test_edge_cases()

    print("=" * 70)
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("✓ M-12: Item Exclusion Disjoint Rule enforced")
    print("✓ M-07 to M-11: Enforced via item_matches() (Phase 2.3)")
    print("✓ Required coverage: ALL required items must have matches")
    print("✓ Exclusion-first evaluation (cheap, strict filter)")
    print("✓ Empty required items: Vacuously satisfied")
    print("✓ Empty candidates with non-empty required: Fails")
    print("✓ Candidate reuse: Same candidate can match multiple requireds")
    print("✓ Short-circuit: First valid candidate sufficient")
    print("✓ Dynamic attributes: No hardcoded names")
    print()
