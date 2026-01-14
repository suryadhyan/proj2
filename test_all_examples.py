"""Test transformation with all 10 examples from stage3_extraction1.json"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2

# Load all examples
with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    examples = json.load(f)

print("="*80)
print(f"TESTING ALL {len(examples)} EXAMPLES")
print("="*80)

passed = 0
failed = 0

for i, example in enumerate(examples, 1):
    query = example.get("query", "")
    print(f"\n[{i}/10] Testing: \"{query[:60]}...\"")

    try:
        # Transform
        transformed = normalize_and_validate_v2(example)

        # Verify key transformations
        checks = []

        # Check field renaming
        assert "other" in transformed, "Missing 'other' field"
        assert "self" in transformed, "Missing 'self' field"
        assert "location" in transformed, "Missing 'location' field"
        checks.append("Field renaming ✓")

        # Check constraint flattening in items
        items = transformed.get("items", [])
        if items:
            for item in items:
                # Verify min/max are flat scalars, range can be [min, max]
                if item.get("min"):
                    for key, value in item["min"].items():
                        assert not isinstance(value, list), f"min['{key}'] should be scalar"
                        assert not isinstance(value, dict), f"min['{key}'] should be scalar"
                if item.get("max"):
                    for key, value in item["max"].items():
                        assert not isinstance(value, list), f"max['{key}'] should be scalar"
                        assert not isinstance(value, dict), f"max['{key}'] should be scalar"
                if item.get("range"):
                    for key, value in item["range"].items():
                        # Range values should be [min, max] arrays
                        assert isinstance(value, list), f"range['{key}'] should be list [min, max]"
                        assert len(value) == 2, f"range['{key}'] should have exactly 2 values"
            checks.append("Item constraints flattened ✓")

        # Check domain normalized
        domain = transformed.get("domain", [])
        if domain:
            assert all(d == d.lower() for d in domain), "Domain should be lowercase"
            checks.append("Domain normalized ✓")

        # Check location simplified
        location = transformed.get("location")
        if location and not isinstance(location, dict):
            # Should be string (unless route mode)
            checks.append(f"Location simplified ✓")

        print(f"  ✅ PASS - {', '.join(checks)}")
        passed += 1

    except Exception as e:
        print(f"  ❌ FAIL - {type(e).__name__}: {e}")
        failed += 1

print("\n" + "="*80)
print(f"RESULTS: {passed}/{len(examples)} passed, {failed}/{len(examples)} failed")
print("="*80)

if failed == 0:
    print("\n✅ ALL EXAMPLES TRANSFORMED SUCCESSFULLY!")
else:
    print(f"\n⚠️  {failed} examples failed")
