"""
Direct Integration Test (bypassing GPT extraction for testing)

Tests the matching pipeline directly using pre-formatted JSON files.
"""

import json
import requests
from pathlib import Path
import time

# API Configuration
BASE_URL = "http://localhost:8000"
STORE_LISTING_URL = f"{BASE_URL}/store-listing"

# Test data directory
TEST_DIR = Path(__file__).parent / "test_queries"

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def load_json(filename):
    with open(TEST_DIR / filename, 'r') as f:
        return json.load(f)


def store_listing(listing_json, user_id):
    payload = {"listing_json": listing_json, "user_id": user_id, "match_id": None}
    response = requests.post(STORE_LISTING_URL, json=payload)
    response.raise_for_status()
    return response.json()


def search_direct(query_json, user_id):
    """Direct search using pre-formatted query JSON (bypass GPT)"""
    url = f"{BASE_URL}/search-and-match-direct"
    # Use the /search-and-match-direct endpoint that takes listing JSON and does full matching
    response = requests.post(url, json={"listing_json": query_json, "user_id": user_id})
    response.raise_for_status()
    return response.json()


def print_section(title):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_success(message):
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message):
    print(f"{RED}❌ {message}{RESET}")


def print_info(message):
    print(f"{YELLOW}ℹ️  {message}{RESET}")


def test_direct():
    print_section("VRIDDHI Direct Integration Test (Bypass GPT)")

    results = {"total_tests": 0, "passed": 0, "failed": 0}

    # Phase 1: Store candidates
    print_section("PHASE 1: Storing Candidate Listings")

    candidates = [
        ("match_1_iphone_seller.json", "user-seller-1"),
        ("match_2_female_roommate.json", "user-roommate-1"),
        ("trap_1_new_iphone.json", "user-seller-2"),
        ("trap_2_samsung_used.json", "user-seller-3"),
        ("trap_3_price_too_high.json", "user-seller-4"),
        ("trap_4_language_mismatch.json", "user-seller-5"),
        ("trap_5_wrong_intent.json", "user-service-1"),
        ("trap_6_storage_too_low.json", "user-seller-6"),
        ("trap_7_male_roommate.json", "user-roommate-2"),
        ("trap_8_smoker_roommate.json", "user-roommate-3"),
    ]

    stored = {}
    for filename, user_id in candidates:
        try:
            listing_json = load_json(filename)
            result = store_listing(listing_json, user_id)
            stored[user_id] = result
            print_success(f"Stored {filename}: {result['listing_id']}")
        except Exception as e:
            print_error(f"Failed {filename}: {e}")
            return results

    print_info(f"Stored {len(stored)} listings")
    time.sleep(2)

    # Phase 2: Test Product Match
    print_section("PHASE 2: Testing Product Match")

    target_1 = load_json("target_1_product_buyer.json")
    print_info("Target: Used iPhone buyer, 128GB+, max ₹50k, Hindi, Mumbai")

    try:
        result = search_direct(target_1, "user-buyer-1")
        matches = result.get("matches", [])

        print_info(f"Found {len(matches)} matches")

        results["total_tests"] += 1
        if len(matches) == 1:
            print_success("TEST 1 PASSED: Found 1 match")
            results["passed"] += 1

            # Check if it's the right match
            match_user = matches[0] if isinstance(matches[0], str) else matches[0].get("user_id")
            results["total_tests"] += 1
            if match_user == "user-seller-1":
                print_success("TEST 2 PASSED: Matched correct user (match_1)")
                results["passed"] += 1
            else:
                print_error(f"TEST 2 FAILED: Wrong match {match_user}")
                results["failed"] += 1
        else:
            print_error(f"TEST 1 FAILED: Expected 1, got {len(matches)}")
            results["failed"] += 1
            results["total_tests"] += 1
            results["failed"] += 1

    except Exception as e:
        print_error(f"Product test failed: {e}")
        results["total_tests"] += 2
        results["failed"] += 2

    # Phase 3: Test Mutual Match
    print_section("PHASE 3: Testing Mutual Match")

    target_2 = load_json("target_2_mutual_roommate.json")
    print_info("Target: Female roommate, non-smoker, no pets, Bangalore")

    try:
        result = search_direct(target_2, "user-seeker-1")
        matches = result.get("matches", [])

        print_info(f"Found {len(matches)} matches")

        results["total_tests"] += 1
        if len(matches) == 1:
            print_success("TEST 3 PASSED: Found 1 match")
            results["passed"] += 1

            match_user = matches[0] if isinstance(matches[0], str) else matches[0].get("user_id")
            results["total_tests"] += 1
            if match_user == "user-roommate-1":
                print_success("TEST 4 PASSED: Matched correct user (match_2)")
                results["passed"] += 1
            else:
                print_error(f"TEST 4 FAILED: Wrong match {match_user}")
                results["failed"] += 1
        else:
            print_error(f"TEST 3 FAILED: Expected 1, got {len(matches)}")
            results["failed"] += 1
            results["total_tests"] += 1
            results["failed"] += 1

    except Exception as e:
        print_error(f"Mutual test failed: {e}")
        results["total_tests"] += 2
        results["failed"] += 2

    # Summary
    print_section("TEST SUMMARY")
    print(f"Total Tests: {results['total_tests']}")
    print(f"{GREEN}Passed: {results['passed']}{RESET}")
    print(f"{RED}Failed: {results['failed']}{RESET}")

    if results['failed'] == 0:
        print_success("🎉 ALL TESTS PASSED!")
    else:
        print_error(f"⚠️ {results['failed']} tests failed")

    return results


if __name__ == "__main__":
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print_success("Server running\n")

        results = test_direct()
        exit(0 if results['failed'] == 0 else 1)

    except requests.exceptions.ConnectionError:
        print_error("Server not running! Start with: uvicorn main:app --reload")
        exit(1)
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
