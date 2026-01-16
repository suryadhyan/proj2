"""
Test script to verify schema_normalizer_v2.py handles the new prompt format correctly.

Tests the transformation of identity, lifestyle, and habits fields.
"""

from schema_normalizer_v2 import normalize_and_validate_v2
import json

# Example extracted JSON from the user's query:
# "looking for flatmate who is better in cooking and doesn't smoke and i don't drink as well"
example_extracted_json = {
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["real estate & property"],
    "primary_mutual_category": ["roommates"],

    "items": [],

    "other_party_preferences": {
        "identity": [
            {
                "type": "skill",
                "value": "cooking"
            }
        ],
        "lifestyle": [],
        "habits": {
            "smoking": "no"
        },
        "min": {},
        "max": {},
        "range": {}
    },

    "other_party_exclusions": [],

    "self_attributes": {
        "identity": [],
        "lifestyle": [],
        "habits": {
            "drinking": "no"
        },
        "min": {},
        "max": {},
        "range": {}
    },

    "self_exclusions": [],

    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],

    "item_exclusions": [],

    "reasoning": "User is seeking a shared living arrangement with a flatmate. Preference for flatmate skilled in cooking and non-smoker. User mentions not drinking."
}

print("=" * 80)
print("TESTING SCHEMA NORMALIZATION V2")
print("=" * 80)
print("\nINPUT (NEW SCHEMA):")
print(json.dumps(example_extracted_json, indent=2))

try:
    # Normalize and validate
    normalized_listing = normalize_and_validate_v2(example_extracted_json)

    print("\n" + "=" * 80)
    print("OUTPUT (OLD SCHEMA - for matching engine):")
    print("=" * 80)
    print(json.dumps(normalized_listing, indent=2))

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    # Verify other_party_preferences transformation
    other = normalized_listing.get("other", {})
    other_categorical = other.get("categorical", {})

    print("\nother_party_preferences.categorical:")
    print(f"  - skill: {other_categorical.get('skill', 'NOT FOUND')}")
    print(f"  - smoking: {other_categorical.get('smoking', 'NOT FOUND')}")

    expected_skill = "cooking"
    expected_smoking = "no"

    if other_categorical.get("skill") == expected_skill:
        print(f"  ✅ skill correctly transformed to '{expected_skill}'")
    else:
        print(f"  ❌ skill NOT correctly transformed (expected '{expected_skill}', got '{other_categorical.get('skill')}')")

    if other_categorical.get("smoking") == expected_smoking:
        print(f"  ✅ smoking correctly transformed to '{expected_smoking}'")
    else:
        print(f"  ❌ smoking NOT correctly transformed (expected '{expected_smoking}', got '{other_categorical.get('smoking')}')")

    # Verify self_attributes transformation
    self_obj = normalized_listing.get("self", {})
    self_categorical = self_obj.get("categorical", {})

    print("\nself_attributes.categorical:")
    print(f"  - drinking: {self_categorical.get('drinking', 'NOT FOUND')}")

    expected_drinking = "no"

    if self_categorical.get("drinking") == expected_drinking:
        print(f"  ✅ drinking correctly transformed to '{expected_drinking}'")
    else:
        print(f"  ❌ drinking NOT correctly transformed (expected '{expected_drinking}', got '{self_categorical.get('drinking')}')")

    print("\n" + "=" * 80)
    print("✅ SCHEMA TRANSFORMATION SUCCESSFUL!")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("❌ ERROR:")
    print("=" * 80)
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
