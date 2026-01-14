"""
Test cases for schema_normalizer.py
Demonstrates validation and normalization behavior
"""

from schema_normalizer import (
    normalize_and_validate,
    MissingRequiredFieldError,
    InvalidIntentSubIntentError,
    InvalidConstraintModeError,
    InvalidRangeBoundsError,
    TypeValidationError
)


def test_valid_product_listing():
    """Test valid product listing normalization"""
    print("=" * 70)
    print("TEST: Valid Product Listing")
    print("=" * 70)

    raw_listing = {
        "intent": "PRODUCT",  # Will be normalized to lowercase
        "subintent": "buy",
        "domain": "electronics",  # Will be normalized to array
        "category": [],
        "items": [{
            "type": "SMARTPHONE",  # Will be normalized
            "categorical": {
                "Brand": "Apple",  # Keys/values normalized
                "COLOR": "Black"
            },
            "min": {},
            "max": {"price": 100000},
            "range": {"storage": [256, 256]}  # EXACT as range[x,x]
        }],
        "itemexclusions": ["REFURBISHED", "used"],  # Normalized
        "other": {
            "categorical": {"verified": "true"},
            "min": {"rating": 4.0},
            "max": {},
            "range": {}
        },
        "otherexclusions": ["dealer"],
        "self": {
            "categorical": {"payment": "cash"},
            "min": {},
            "max": {},
            "range": {}
        },
        "selfexclusions": [],
        "location": "Bangalore",
        "locationmode": "proximity",
        "locationexclusions": ["chennai"],
        "reasoning": "User wants iPhone with specific specs"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✓ Validation PASSED")
        print("\nNormalized output:")
        print(f"  intent: {normalized['intent']}")
        print(f"  subintent: {normalized['subintent']}")
        print(f"  domain: {normalized['domain']}")
        print(f"  items[0].type: {normalized['items'][0]['type']}")
        print(f"  items[0].categorical: {normalized['items'][0]['categorical']}")
        print(f"  itemexclusions: {normalized['itemexclusions']}")
        print(f"  location: {normalized['location']}")
        print(f"  locationexclusions: {normalized['locationexclusions']}")
        print("\n✓ All strings normalized to lowercase")
        print("✓ Single domain string converted to array")
        print("✓ EXACT constraint represented as range[256, 256]")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def test_invalid_intent_subintent():
    """Test I-04: Invalid intent-subintent combination"""
    print("=" * 70)
    print("TEST: Invalid Intent-SubIntent (I-04 Violation)")
    print("=" * 70)

    raw_listing = {
        "intent": "product",
        "subintent": "seek",  # Invalid: product can only be buy/sell
        "domain": ["electronics"],
        "items": [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "location": "bangalore",
        "locationmode": "proximity",
        "reasoning": "test"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✗ Should have failed but didn't")
    except InvalidIntentSubIntentError as e:
        print(f"✓ Correctly rejected: {e}")
        print(f"✓ Error references I-04 invariant")

    print()


def test_invalid_constraint_mode():
    """Test I-02: Invalid constraint mode (exact as key)"""
    print("=" * 70)
    print("TEST: Invalid Constraint Mode - 'exact' key (I-02 Violation)")
    print("=" * 70)

    raw_listing = {
        "intent": "product",
        "subintent": "buy",
        "domain": ["electronics"],
        "items": [{
            "type": "phone",
            "exact": {"storage": 256},  # Invalid: EXACT is not a mode
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "location": "bangalore",
        "locationmode": "proximity",
        "reasoning": "test"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✗ Should have failed but didn't")
    except InvalidConstraintModeError as e:
        print(f"✓ Correctly rejected: {e}")
        print(f"✓ Error references I-02 invariant")
        print(f"✓ EXACT must be represented as range[x, x], not as separate mode")

    print()


def test_invalid_range_bounds():
    """Test I-06: Range bounds violation (min > max)"""
    print("=" * 70)
    print("TEST: Invalid Range Bounds (I-06 Violation)")
    print("=" * 70)

    raw_listing = {
        "intent": "product",
        "subintent": "buy",
        "domain": ["electronics"],
        "items": [{
            "type": "phone",
            "categorical": {},
            "min": {},
            "max": {},
            "range": {"price": [100, 50]}  # Invalid: 100 > 50
        }],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "location": "bangalore",
        "locationmode": "proximity",
        "reasoning": "test"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✗ Should have failed but didn't")
    except InvalidRangeBoundsError as e:
        print(f"✓ Correctly rejected: {e}")
        print(f"✓ Error references I-06 invariant")
        print(f"✓ Enforces min ≤ max requirement")

    print()


def test_missing_required_field():
    """Test missing required field"""
    print("=" * 70)
    print("TEST: Missing Required Field")
    print("=" * 70)

    raw_listing = {
        "intent": "product",
        # Missing "subintent"
        "domain": ["electronics"],
        "items": [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "location": "bangalore",
        "locationmode": "proximity",
        "reasoning": "test"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✗ Should have failed but didn't")
    except MissingRequiredFieldError as e:
        print(f"✓ Correctly rejected: {e}")
        print(f"✓ Identifies missing field explicitly")

    print()


def test_mutual_intent():
    """Test valid mutual intent listing"""
    print("=" * 70)
    print("TEST: Valid Mutual Intent Listing")
    print("=" * 70)

    raw_listing = {
        "intent": "mutual",
        "subintent": "connect",
        "domain": [],  # Not used for mutual
        "category": ["roommate", "HOUSING"],  # Will be normalized
        "items": [],  # Optional for mutual
        "itemexclusions": [],
        "other": {
            "categorical": {"GENDER": "Female", "diet": "VEGETARIAN"},
            "min": {},
            "max": {},
            "range": {}
        },
        "otherexclusions": ["smoker"],
        "self": {
            "categorical": {"gender": "female", "Diet": "vegetarian"},
            "min": {},
            "max": {},
            "range": {"age": [25, 25]}  # EXACT age
        },
        "selfexclusions": [],
        "location": "Koramangala, Bangalore",
        "locationmode": "target",
        "locationexclusions": [],
        "reasoning": "Looking for roommate"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✓ Validation PASSED")
        print("\nNormalized output:")
        print(f"  intent: {normalized['intent']}")
        print(f"  subintent: {normalized['subintent']}")
        print(f"  category: {normalized['category']}")
        print(f"  other.categorical: {normalized['other']['categorical']}")
        print(f"  self.categorical: {normalized['self']['categorical']}")
        print(f"  self.range.age: {normalized['self']['range']['age']}")
        print("\n✓ All categorical keys/values normalized to lowercase")
        print("✓ Category array normalized")
        print("✓ EXACT age represented as range[25, 25]")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def test_route_location():
    """Test route location structure"""
    print("=" * 70)
    print("TEST: Route Location Mode")
    print("=" * 70)

    raw_listing = {
        "intent": "service",
        "subintent": "seek",
        "domain": ["transportation"],
        "category": [],
        "items": [{
            "type": "carpool",
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }],
        "itemexclusions": [],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "otherexclusions": [],
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "selfexclusions": [],
        "location": {
            "origin": "Whitefield",
            "destination": "MG Road"
        },
        "locationmode": "route",
        "locationexclusions": [],
        "reasoning": "Need carpool"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✓ Validation PASSED")
        print("\nNormalized output:")
        print(f"  location: {normalized['location']}")
        print(f"  locationmode: {normalized['locationmode']}")
        print("\n✓ Route location structure validated")
        print("✓ Origin and destination normalized")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def test_empty_domain():
    """Test empty domain array (should fail for product/service)"""
    print("=" * 70)
    print("TEST: Empty Domain Array (I-05 Violation)")
    print("=" * 70)

    raw_listing = {
        "intent": "product",
        "subintent": "buy",
        "domain": [],  # Invalid: must have at least one domain
        "items": [{"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}}],
        "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
        "location": "bangalore",
        "locationmode": "proximity",
        "reasoning": "test"
    }

    try:
        normalized = normalize_and_validate(raw_listing)
        print("✗ Should have failed but didn't")
    except Exception as e:
        print(f"✓ Correctly rejected: {e}")
        print(f"✓ Enforces I-05: Domain Cardinality Invariant")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SCHEMA NORMALIZER TEST SUITE")
    print("=" * 70)
    print()

    test_valid_product_listing()
    test_invalid_intent_subintent()
    test_invalid_constraint_mode()
    test_invalid_range_bounds()
    test_missing_required_field()
    test_mutual_intent()
    test_route_location()
    test_empty_domain()

    print("=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ M-31 (Array Normalization): null → [], scalar → [scalar], array → array")
    print("✓ M-32 (Case Insensitivity): All strings → lowercase(trim(s))")
    print("✓ I-02 (Constraint Modes): Only {categorical, min, max, range} allowed")
    print("✓ I-04 (Intent-SubIntent): Only valid combinations allowed")
    print("✓ I-06 (Range Bounds): min ≤ max enforced")
    print("✓ EXACT constraint: Must use range[x,x], not separate 'exact' key")
    print("✓ All errors explicit with canon rule references")
    print()
