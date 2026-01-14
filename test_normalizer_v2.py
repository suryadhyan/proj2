"""Quick test script for schema_normalizer_v2.py"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2

# Load first example from stage3_extraction1.json
with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    examples = json.load(f)

# Test with first example (macbook pro)
example = examples[0]

print("="*80)
print("TESTING SCHEMA NORMALIZER V2")
print("="*80)
print("\nOriginal NEW schema (first 300 chars):")
print(json.dumps(example, indent=2)[:300] + "...")

try:
    # Transform
    transformed = normalize_and_validate_v2(example)

    print("\n" + "="*80)
    print("✅ TRANSFORMATION SUCCESSFUL")
    print("="*80)

    print("\nTransformed to OLD format:")
    print(json.dumps(transformed, indent=2))

    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)

    # Verify field names changed
    assert "other" in transformed, "Missing 'other' field"
    assert "self" in transformed, "Missing 'self' field"
    assert "location" in transformed, "Missing 'location' field"
    assert "locationmode" in transformed, "Missing 'locationmode' field"
    assert "itemexclusions" in transformed, "Missing 'itemexclusions' field"
    print("✅ Field renaming correct")

    # Verify constraints flattened
    items = transformed.get("items", [])
    if items:
        first_item = items[0]
        assert "min" in first_item, "Missing 'min' in item"
        assert "max" in first_item, "Missing 'max' in item"

        # Check that min/max are flat dicts, not axis-based
        if first_item["min"]:
            # Should be {memory: 16}, NOT {capacity: [{...}]}
            for key, value in first_item["min"].items():
                assert not isinstance(value, list), f"min['{key}'] should be scalar, got list"
        print("✅ Axis constraints flattened correctly")

    # Verify location simplified
    location = transformed.get("location")
    if location:
        # Should be string (bangalore), not {"name": "bangalore"}
        assert isinstance(location, str), f"location should be string, got {type(location)}"
        print(f"✅ Location simplified: '{location}'")

    # Verify domain normalized
    domain = transformed.get("domain", [])
    if domain:
        assert isinstance(domain, list), "domain should be list"
        assert all(isinstance(d, str) for d in domain), "domain items should be strings"
        # Check lowercase
        assert domain[0] == domain[0].lower(), "domain should be lowercase"
        print(f"✅ Domain normalized: {domain}")

    print("\n" + "="*80)
    print("ALL CHECKS PASSED! ✅")
    print("="*80)

except Exception as e:
    print(f"\n❌ TRANSFORMATION FAILED")
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
