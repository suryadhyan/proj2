"""
Debug test to see what candidates are returned
"""

import json
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).parent / "test_queries"

def load_json(filename):
    with open(TEST_DIR / filename, 'r') as f:
        return json.load(f)

# Test 1: Product search
print("=" * 80)
print("TEST 1: Product Match (iPhone buyer)")
print("=" * 80)

target_1 = load_json("target_1_product_buyer.json")
print(f"\nQuery: {target_1['reasoning']}")
print(f"Intent: {target_1['intent']}/{target_1['subintent']}")
print(f"Domain: {target_1.get('domain', [])}")

response = requests.post(f"{BASE_URL}/search", json={"listing": target_1})
result = response.json()

print(f"\n✓ Status: {result['status']}")
print(f"✓ Count: {result['count']}")
print(f"✓ Candidates: {result['candidates'][:5]}")  # Show first 5

# Now test matching against the correct candidate
print("\n" + "=" * 80)
print("TEST: Direct match against match_1_iphone_seller")
print("=" * 80)

match_1 = load_json("match_1_iphone_seller.json")
print(f"\nCandidate: {match_1['reasoning']}")
print(f"Intent: {match_1['intent']}/{match_1['subintent']}")
print(f"Domain: {match_1.get('domain', [])}")

match_response = requests.post(
    f"{BASE_URL}/match",
    json={
        "listing_a": target_1,
        "listing_b": match_1
    }
)
match_result = match_response.json()

print(f"\n✓ Match result: {match_result['match']}")
print(f"✓ Details: {match_result['details']}")
