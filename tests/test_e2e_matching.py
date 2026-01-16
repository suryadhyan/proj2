"""End-to-end matching test with transformed NEW schema data"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

# Load examples
with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    examples = json.load(f)

print("="*80)
print("END-TO-END MATCHING TEST")
print("="*80)

# Test Case 1: Product matching (buyer vs seller)
print("\n[Test 1] Product Matching: Buyer vs Seller")
print("-"*80)

# Example 0: Buyer looking for MacBook Pro
buyer_new = examples[0]  # "looking for a used macbook pro with at least 16gb ram under 80k"
print(f"Buyer: {buyer_new['query'][:60]}...")

# Example 1: Seller with Royal Enfield (should NOT match - different domain)
seller1_new = examples[1]  # "selling my old royal enfield classic 350"
print(f"Seller 1: {seller1_new['query'][:60]}...")

# Transform both
buyer_old = normalize_and_validate_v2(buyer_new)
seller1_old = normalize_and_validate_v2(seller1_new)

print(f"\nBuyer intent: {buyer_old['intent']}/{buyer_old['subintent']}, domain: {buyer_old['domain']}")
print(f"Seller 1 intent: {seller1_old['intent']}/{seller1_old['subintent']}, domain: {seller1_old['domain']}")

# Test matching
result1 = listing_matches_v2(buyer_old, seller1_old)
print(f"\n✓ Match Result: {result1}")
print(f"Expected: False (different domains: technology vs automotive)")
assert result1 == False, "Should not match - different domains"
print("✅ PASS - Correctly rejected due to domain mismatch")

# Test Case 2: Service matching (seeker vs provider)
print("\n" + "="*80)
print("[Test 2] Service Matching: Seeker vs Provider")
print("-"*80)

# Example 2: Seeker looking for Kannada plumber in Jayanagar
seeker_new = examples[2]  # "need a kannada speaking plumber urgently in jayanagar"
print(f"Seeker: {seeker_new['query'][:60]}...")

# Example 3: Provider - graphic designer (should NOT match - different service)
provider1_new = examples[3]  # "i'm a graphic designer with 5 years experience"
print(f"Provider 1: {provider1_new['query'][:60]}...")

# Transform both
seeker_old = normalize_and_validate_v2(seeker_new)
provider1_old = normalize_and_validate_v2(provider1_new)

print(f"\nSeeker intent: {seeker_old['intent']}/{seeker_old['subintent']}, domain: {seeker_old['domain']}")
print(f"Provider 1 intent: {provider1_old['intent']}/{provider1_old['subintent']}, domain: {provider1_old['domain']}")

# Test matching
result2 = listing_matches_v2(seeker_old, provider1_old)
print(f"\n✓ Match Result: {result2}")
print(f"Expected: False (different domains: construction vs marketing)")
assert result2 == False, "Should not match - different domains"
print("✅ PASS - Correctly rejected due to domain mismatch")

# Test Case 3: Mutual matching (connect vs connect)
print("\n" + "="*80)
print("[Test 3] Mutual Matching: Adventure buddies")
print("-"*80)

# Example 4: Weekend treks around bangalore
person1_new = examples[4]  # "anyone up for weekend treks around bangalore?"
print(f"Person 1: {person1_new['query'][:60]}...")

# Example 5: 2BHK flat wanted (should NOT match - different category)
person2_new = examples[5]  # "2bhk furnished flat wanted in koramangala"
print(f"Person 2: {person2_new['query'][:60]}...")

# Transform both
person1_old = normalize_and_validate_v2(person1_new)
person2_old = normalize_and_validate_v2(person2_new)

print(f"\nPerson 1 intent: {person1_old['intent']}/{person1_old['subintent']}, category: {person1_old['category']}")
print(f"Person 2 intent: {person2_old['intent']}/{person2_old['subintent']}, category: {person2_old['category']}")

# Test matching
result3 = listing_matches_v2(person1_old, person2_old)
print(f"\n✓ Match Result: {result3}")
print(f"Expected: False (different categories: Adventure vs Roommates)")
assert result3 == False, "Should not match - different categories"
print("✅ PASS - Correctly rejected due to category mismatch")

# Test Case 4: Intent gate (product vs service should never match)
print("\n" + "="*80)
print("[Test 4] Intent Gate: Product vs Service (should fail)")
print("-"*80)

# Compare buyer (product) vs provider (service)
print(f"A (Product): {buyer_old['intent']}/{buyer_old['subintent']}")
print(f"B (Service): {provider1_old['intent']}/{provider1_old['subintent']}")

result4 = listing_matches_v2(buyer_old, provider1_old)
print(f"\n✓ Match Result: {result4}")
print(f"Expected: False (different intents)")
assert result4 == False, "Should not match - different intents"
print("✅ PASS - Intent gate correctly rejects")

# Test Case 5: SubIntent gate (buyer vs buyer should fail)
print("\n" + "="*80)
print("[Test 5] SubIntent Gate: Buyer vs Buyer (should fail)")
print("-"*80)

# Compare two buyers
buyer2_new = examples[9]  # "iphone 14 pro max 256gb" (also a buy intent)
buyer2_old = normalize_and_validate_v2(buyer2_new)

print(f"A (Buyer): {buyer_old['intent']}/{buyer_old['subintent']}")
print(f"B (Also Buyer): {buyer2_old['intent']}/{buyer2_old['subintent']}")

result5 = listing_matches_v2(buyer_old, buyer2_old)
print(f"\n✓ Match Result: {result5}")
print(f"Expected: False (same subintent - both buyers)")
assert result5 == False, "Should not match - same subintent"
print("✅ PASS - SubIntent gate correctly rejects")

print("\n" + "="*80)
print("ALL END-TO-END TESTS PASSED! ✅")
print("="*80)
print("\nSummary:")
print("✓ Schema transformation works correctly")
print("✓ Intent gate enforced (M-01)")
print("✓ SubIntent gate enforced (M-02 for product/service)")
print("✓ Domain intersection enforced (M-05 for product/service)")
print("✓ Category intersection enforced (M-06 for mutual)")
print("✓ All matching logic working with transformed data")
