"""
Debug which matches are being returned
"""

import json
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_DIR = Path(__file__).parent / "test_queries"

def load_json(filename):
    with open(TEST_DIR / filename, 'r') as f:
        return json.load(f)

# Candidate filenames and user IDs
candidates = {
    "user-seller-1": "match_1_iphone_seller.json",
    "user-roommate-1": "match_2_female_roommate.json",
    "user-seller-2": "trap_1_new_iphone.json",
    "user-seller-3": "trap_2_samsung_used.json",
    "user-seller-4": "trap_3_price_too_high.json",
    "user-seller-5": "trap_4_language_mismatch.json",
    "user-service-1": "trap_5_wrong_intent.json",
    "user-seller-6": "trap_6_storage_too_low.json",
    "user-roommate-2": "trap_7_male_roommate.json",
    "user-roommate-3": "trap_8_smoker_roommate.json",
}

print("=" * 80)
print("TEST 1: Product Match (iPhone buyer)")
print("=" * 80)

target_1 = load_json("target_1_product_buyer.json")
response = requests.post(
    f"{BASE_URL}/search-and-match-direct",
    json={"listing_json": target_1, "user_id": "user-buyer-1"}
)
result = response.json()

print(f"\nMatched {result['match_count']} listings:")
for user_id in result['matches']:
    filename = candidates.get(user_id, "unknown")
    print(f"  ✓ {user_id:20} ({filename})")

print("\n" + "=" * 80)
print("TEST 2: Mutual Match (Female roommate)")
print("=" * 80)

target_2 = load_json("target_2_mutual_roommate.json")
response = requests.post(
    f"{BASE_URL}/search-and-match-direct",
    json={"listing_json": target_2, "user_id": "user-seeker-1"}
)
result = response.json()

print(f"\nMatched {result['match_count']} listings:")
for user_id in result['matches']:
    filename = candidates.get(user_id, "unknown")
    print(f"  ✓ {user_id:20} ({filename})")
