"""Test location transformation specifically"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2

# Load examples
with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    examples = json.load(f)

# Test with example 3 (plumber in jayanagar - has explicit location)
example = examples[2]  # Index 2 is the plumber example

print("="*80)
print("TESTING LOCATION TRANSFORMATION")
print("="*80)
print(f"\nQuery: {example['query']}")
print(f"\nOriginal location:")
print(f"  target_location: {json.dumps(example['target_location'])}")
print(f"  location_match_mode: {example['location_match_mode']}")

try:
    transformed = normalize_and_validate_v2(example)

    print(f"\n✅ TRANSFORMED:")
    print(f"  location: {json.dumps(transformed['location'])}")
    print(f"  locationmode: {transformed['locationmode']}")

    # Verify
    assert transformed['location'] == 'jayanagar', f"Expected 'jayanagar', got {transformed['location']}"
    assert transformed['locationmode'] == 'explicit', f"Expected 'explicit', got {transformed['locationmode']}"

    print("\n✅ Location transformation correct!")

    # Also test other_party_preferences transformation (has language: kannada)
    print(f"\nOriginal other_party_preferences:")
    print(json.dumps(example['other_party_preferences'], indent=2))

    print(f"\n✅ TRANSFORMED to 'other':")
    print(json.dumps(transformed['other'], indent=2))

    # Verify categorical preserved
    assert transformed['other']['categorical']['language'] == 'kannada'
    print("\n✅ Other party preferences transformation correct!")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
