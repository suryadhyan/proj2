"""
Quick script to add missing NEW schema fields to test JSON files
"""

import json
from pathlib import Path

test_dir = Path("test_queries")

# Fields to add with default values
default_fields = {
    "primary_mutual_category": [],
    "item_exclusions": [],
    "other_party_exclusions": [],
    "self_exclusions": [],
    "location_match_mode": "explicit",
    "location_exclusions": [],
    "reasoning": ""
}

# Process all JSON files
for json_file in test_dir.glob("*.json"):
    print(f"Processing {json_file.name}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Add missing fields
    for field, default_value in default_fields.items():
        if field not in data:
            data[field] = default_value

    # Generate reasoning if empty
    if not data.get("reasoning"):
        intent = data.get("intent", "unknown")
        subintent = data.get("subintent", "unknown")
        data["reasoning"] = f"Test query for {intent}/{subintent} - {json_file.stem}"

    # Fix location_match_mode based on target_location
    if "target_location" in data:
        data["location_match_mode"] = data["target_location"].get("mode", "explicit")

    # For mutual intent, set primary_mutual_category from category if exists
    if data.get("intent") == "mutual" and "category" in data:
        data["primary_mutual_category"] = data.get("category", [])
        # Remove old category field
        if "category" in data:
            del data["category"]

    # Write back
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  ✅ Updated {json_file.name}")

print("\n✅ All test files updated!")
