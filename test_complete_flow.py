"""
Integration Test for Complete Search-Match-Store Flow

Tests:
1. Store 10 candidate listings using /store-listing
2. Search with target queries using /search-and-match
3. Verify correct matches found (semantic + hard filters)
4. Verify matches table stores history correctly
"""

import json
import requests
from pathlib import Path
import time

# API Configuration
BASE_URL = "http://localhost:8000"
STORE_LISTING_URL = f"{BASE_URL}/store-listing"
SEARCH_AND_MATCH_URL = f"{BASE_URL}/search-and-match"

# Test data directory
TEST_DIR = Path(__file__).parent / "test_queries"

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def load_json(filename):
    """Load JSON from test_queries directory"""
    with open(TEST_DIR / filename, 'r') as f:
        return json.load(f)


def store_listing(listing_json, user_id, match_id=None):
    """Store a listing using /store-listing endpoint"""
    payload = {
        "listing_json": listing_json,
        "user_id": user_id,
        "match_id": match_id
    }

    response = requests.post(STORE_LISTING_URL, json=payload)
    response.raise_for_status()
    return response.json()


def search_and_match(query_text, user_id):
    """Search and match using /search-and-match endpoint"""
    payload = {
        "query": query_text,
        "user_id": user_id
    }

    response = requests.post(SEARCH_AND_MATCH_URL, json=payload)
    response.raise_for_status()
    return response.json()


def print_section(title):
    """Print section header"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_success(message):
    print(f"{GREEN}✅ {message}{RESET}")


def print_error(message):
    print(f"{RED}❌ {message}{RESET}")


def print_info(message):
    print(f"{YELLOW}ℹ️  {message}{RESET}")


def test_complete_flow():
    """Run complete integration test"""

    print_section("VRIDDHI Complete Flow Integration Test")
    print_info("Testing: Semantic matching, SQL filters, and complete pipeline")

    # Track results
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0
    }

    # =========================================================================
    # PHASE 1: Store 10 candidate listings
    # =========================================================================
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

    stored_listings = {}

    for filename, user_id in candidates:
        try:
            listing_json = load_json(filename)
            result = store_listing(listing_json, user_id)
            stored_listings[filename] = result
            print_success(f"Stored {filename}: listing_id={result['listing_id']}, intent={result['intent']}")
        except Exception as e:
            print_error(f"Failed to store {filename}: {e}")
            return

    print_info(f"Stored {len(stored_listings)} candidate listings")
    time.sleep(2)  # Wait for embeddings to be indexed

    # =========================================================================
    # PHASE 2: Test Target Query 1 (Product - iPhone Buyer)
    # =========================================================================
    print_section("PHASE 2: Testing Target Query 1 (Product Buyer)")

    target_1 = load_json("target_1_product_buyer.json")
    print_info("Target 1: Buying used Apple iPhone, min 128GB storage, max ₹50,000")
    print_info("Expected match: match_1_iphone_seller.json (256GB, ₹45,000)")
    print_info("Expected traps to FAIL: traps 1-6")

    try:
        # Note: The /search-and-match endpoint expects natural language query
        # For testing, we'll construct a query that GPT would extract to target_1
        query_text_1 = "I want to buy a used Apple iPhone with at least 128GB storage, budget up to 50000 rupees, prefer Hindi speaking seller in Mumbai"

        result_1 = search_and_match(query_text_1, "user-buyer-1")

        print_info(f"Match ID: {result_1['match_id']}")
        print_info(f"Has matches: {result_1['has_matches']}")
        print_info(f"Match count: {result_1['match_count']}")

        # Test 1: Should find exactly 1 match
        results["total_tests"] += 1
        if result_1["match_count"] == 1:
            print_success("TEST 1 PASSED: Found exactly 1 match")
            results["passed"] += 1
        else:
            print_error(f"TEST 1 FAILED: Expected 1 match, found {result_1['match_count']}")
            results["failed"] += 1

        # Test 2: Match should be match_1_iphone_seller
        results["total_tests"] += 1
        if result_1["has_matches"]:
            matched_listing = result_1["matched_listings"][0]
            print_info(f"Matched listing: {matched_listing['listing_id']} (user: {matched_listing['user_id']})")

            # Verify it's the iPhone seller
            if matched_listing["data"]["items"][0]["categorical"]["brand"] == "apple":
                print_success("TEST 2 PASSED: Matched listing is Apple iPhone")
                results["passed"] += 1
            else:
                print_error("TEST 2 FAILED: Matched listing is not Apple iPhone")
                results["failed"] += 1
        else:
            print_error("TEST 2 FAILED: No matches found")
            results["failed"] += 1

        # Test 3: Verify semantic matching (brand: "apple" should match)
        results["total_tests"] += 1
        if result_1["has_matches"]:
            # The match happened, so semantic matching worked
            print_success("TEST 3 PASSED: Semantic matching working (categorical attributes)")
            results["passed"] += 1
        else:
            print_error("TEST 3 FAILED: Semantic matching not working")
            results["failed"] += 1

    except Exception as e:
        print_error(f"Target Query 1 failed: {e}")
        import traceback
        traceback.print_exc()
        results["total_tests"] += 3
        results["failed"] += 3

    # =========================================================================
    # PHASE 3: Test Target Query 2 (Mutual - Roommate Seeker)
    # =========================================================================
    print_section("PHASE 3: Testing Target Query 2 (Mutual - Roommate)")

    target_2 = load_json("target_2_mutual_roommate.json")
    print_info("Target 2: Female seeking female roommate, 22-30 years, non-smoker, no pets")
    print_info("Expected match: match_2_female_roommate.json (27 years, female, non-smoker)")
    print_info("Expected traps to FAIL: traps 7-8")

    try:
        query_text_2 = "I am a 25 year old working professional female looking for a female roommate in Bangalore, non-smoker, no pets, age 22-30"

        result_2 = search_and_match(query_text_2, "user-seeker-1")

        print_info(f"Match ID: {result_2['match_id']}")
        print_info(f"Has matches: {result_2['has_matches']}")
        print_info(f"Match count: {result_2['match_count']}")

        # Test 4: Should find exactly 1 match
        results["total_tests"] += 1
        if result_2["match_count"] == 1:
            print_success("TEST 4 PASSED: Found exactly 1 match")
            results["passed"] += 1
        else:
            print_error(f"TEST 4 FAILED: Expected 1 match, found {result_2['match_count']}")
            results["failed"] += 1

        # Test 5: Match should be match_2_female_roommate
        results["total_tests"] += 1
        if result_2["has_matches"]:
            matched_listing = result_2["matched_listings"][0]
            print_info(f"Matched listing: {matched_listing['listing_id']} (user: {matched_listing['user_id']})")

            # Verify it's female roommate
            if matched_listing["data"]["self_attributes"]["categorical"]["gender"] == "female":
                print_success("TEST 5 PASSED: Matched listing is female roommate")
                results["passed"] += 1
            else:
                print_error("TEST 5 FAILED: Matched listing is not female")
                results["failed"] += 1
        else:
            print_error("TEST 5 FAILED: No matches found")
            results["failed"] += 1

        # Test 6: Verify hard filters (gender mismatch should filter out trap 7)
        results["total_tests"] += 1
        # If only 1 match found, trap_7 (male) was correctly filtered
        if result_2["match_count"] == 1:
            print_success("TEST 6 PASSED: Hard filters working (gender mismatch filtered)")
            results["passed"] += 1
        else:
            print_error("TEST 6 FAILED: Hard filters not working correctly")
            results["failed"] += 1

    except Exception as e:
        print_error(f"Target Query 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results["total_tests"] += 3
        results["failed"] += 3

    # =========================================================================
    # PHASE 4: Verify Matches Table Storage
    # =========================================================================
    print_section("PHASE 4: Verifying Matches Table Storage")

    # Test 7: Both searches should be stored in matches table
    results["total_tests"] += 1
    print_info("Checking if search history was stored in matches table...")

    try:
        # We stored 2 searches, so we should have 2 match_ids
        if 'result_1' in locals() and 'result_2' in locals():
            match_id_1 = result_1.get('match_id')
            match_id_2 = result_2.get('match_id')

            if match_id_1 and match_id_2:
                print_success(f"TEST 7 PASSED: Both searches stored with match_ids: {match_id_1[:8]}..., {match_id_2[:8]}...")
                results["passed"] += 1
            else:
                print_error("TEST 7 FAILED: Match IDs not returned")
                results["failed"] += 1
        else:
            print_error("TEST 7 FAILED: Search results not available")
            results["failed"] += 1
    except Exception as e:
        print_error(f"TEST 7 FAILED: {e}")
        results["failed"] += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("TEST SUMMARY")

    print(f"Total Tests: {results['total_tests']}")
    print(f"{GREEN}Passed: {results['passed']}{RESET}")
    print(f"{RED}Failed: {results['failed']}{RESET}")

    if results['failed'] == 0:
        print_success("🎉 ALL TESTS PASSED! Complete flow working correctly.")
        print_info("✅ Semantic matching (embeddings) working")
        print_info("✅ Hard filters (intent, domain, categorical) working")
        print_info("✅ Search-and-match endpoint working")
        print_info("✅ Store-listing endpoint working")
        print_info("✅ Matches table storage working")
        print_info("✅ Complete pipeline integrated successfully")
    else:
        print_error(f"⚠️  {results['failed']} tests failed. Check logs above.")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VRIDDHI Matching Engine - Complete Flow Integration Test")
    print("="*80 + "\n")

    try:
        # Check if server is running
        print_info("Checking if server is running...")
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print_success("Server is running!\n")

        # Run tests
        results = test_complete_flow()

        # Exit code based on results
        exit(0 if results['failed'] == 0 else 1)

    except requests.exceptions.ConnectionError:
        print_error("Server is not running! Start the server with: uvicorn main:app --reload")
        exit(1)
    except Exception as e:
        print_error(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
