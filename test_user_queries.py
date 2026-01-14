"""
Test user queries against database
Database: stage3_extraction1.json (10 listings)
Queries: 3 MacBook Pro sellers with different prices
"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

print("="*80)
print("USER QUERY MATCHING TEST")
print("="*80)

# ============================================================================
# LOAD DATABASE
# ============================================================================

print("\n[STEP 1] Loading Database (stage3_extraction1.json)...")
print("-"*80)

with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    db_listings = json.load(f)

print(f"✓ Loaded {len(db_listings)} listings from database")
print("\nDatabase Contents:")
for i, listing in enumerate(db_listings, 1):
    intent = listing.get('intent', 'unknown')
    subintent = listing.get('subintent', 'unknown')
    query_text = listing.get('query', '')[:60]
    print(f"  [{i}] {intent}/{subintent}: {query_text}...")

# ============================================================================
# QUERY 1: "looking to sell used macbook pro which has 16gb ram price is 60k"
# ============================================================================

print("\n" + "="*80)
print("QUERY 1: Looking to SELL used MacBook Pro, 16GB RAM, 60k")
print("="*80)

query1_new = {
    "query": "looking to sell used macbook pro which has 16gb ram price is 60k",
    "intent": "product",
    "subintent": "sell",
    "domain": ["Technology & Electronics"],
    "primary_mutual_category": [],
    "items": [{
        "type": "laptop",
        "categorical": {
            "condition": "used",
            "brand": "apple",
            "model": "macbook pro"
        },
        "min": {},
        "max": {},
        "range": {
            "capacity": [{"type": "memory", "min": 16, "max": 16, "unit": "gb"}],
            "cost": [{"type": "price", "min": 60000, "max": 60000, "unit": "inr"}]
        }
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],
    "reasoning": "Seller offering used MacBook Pro with 16GB RAM at 60k INR"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query1_new, indent=2))

print("\n[Transforming to OLD format...]")
query1_old = normalize_and_validate_v2(query1_new)
print("✓ Transformed successfully")

print(f"\nSeller Details (OLD format):")
print(f"  Intent: {query1_old['intent']}/{query1_old['subintent']}")
print(f"  Domain: {query1_old['domain']}")
print(f"  Item: {query1_old['items'][0]['type']}")
print(f"  Categorical: {query1_old['items'][0]['categorical']}")
print(f"  Range: {query1_old['items'][0]['range']}")

print("\n[Matching against database...]")
print("Looking for BUYERS (subintent=buy) in same domain...")

matches_q1 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    # For seller → buyer matching, check if buyer's requirements match seller's offering
    # So we test: does seller (query1) satisfy buyer's requirements?
    # Match direction: buyer → seller
    result = listing_matches_v2(db_listing_old, query1_old)

    if result:
        matches_q1.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")
        print(f"      Intent: {db_listing_new['intent']}/{db_listing_new['subintent']}")

if not matches_q1:
    print("  ✗ No matches found")

    # Show why potential matches failed
    print("\n  Checking potential buyers:")
    for i, db_listing_new in enumerate(db_listings, 1):
        if db_listing_new['intent'] == 'product' and db_listing_new['subintent'] == 'buy':
            db_listing_old = normalize_and_validate_v2(db_listing_new)
            print(f"\n  Buyer #{i}: {db_listing_new['query'][:60]}...")
            print(f"    Domain: {db_listing_old['domain']}")
            print(f"    Items: {db_listing_old['items']}")

            # Check domain match
            if query1_old['domain'] and db_listing_old['domain']:
                domain_match = bool(set(query1_old['domain']) & set(db_listing_old['domain']))
                print(f"    Domain match: {domain_match}")

print(f"\n✓ Query 1 Results: {len(matches_q1)} matches")

# ============================================================================
# QUERY 2: "selling 3 months used macbook pro with 16gb ram 80,000"
# ============================================================================

print("\n" + "="*80)
print("QUERY 2: SELLING 3 months used MacBook Pro, 16GB RAM, 80,000")
print("="*80)

query2_new = {
    "query": "selling 3 months used macbook pro with 16gb ram 80,000",
    "intent": "product",
    "subintent": "sell",
    "domain": ["Technology & Electronics"],
    "primary_mutual_category": [],
    "items": [{
        "type": "laptop",
        "categorical": {
            "condition": "used",
            "brand": "apple",
            "model": "macbook pro"
        },
        "min": {},
        "max": {},
        "range": {
            "capacity": [{"type": "memory", "min": 16, "max": 16, "unit": "gb"}],
            "cost": [{"type": "price", "min": 80000, "max": 80000, "unit": "inr"}],
            "time": [{"type": "age", "min": 3, "max": 3, "unit": "months"}]
        }
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],
    "reasoning": "Seller offering 3 months used MacBook Pro with 16GB RAM at 80k INR"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query2_new, indent=2))

print("\n[Transforming to OLD format...]")
query2_old = normalize_and_validate_v2(query2_new)
print("✓ Transformed successfully")

print(f"\nSeller Details (OLD format):")
print(f"  Intent: {query2_old['intent']}/{query2_old['subintent']}")
print(f"  Range: {query2_old['items'][0]['range']}")

print("\n[Matching against database...]")

matches_q2 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)
    result = listing_matches_v2(db_listing_old, query2_old)

    if result:
        matches_q2.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")

if not matches_q2:
    print("  ✗ No matches found")

print(f"\n✓ Query 2 Results: {len(matches_q2)} matches")

# ============================================================================
# QUERY 3: "selling 3 months used macbook pro with 16gb ram 120000"
# ============================================================================

print("\n" + "="*80)
print("QUERY 3: SELLING 3 months used MacBook Pro, 16GB RAM, 120,000")
print("="*80)

query3_new = {
    "query": "selling 3 months used macbook pro with 16gb ram 120000",
    "intent": "product",
    "subintent": "sell",
    "domain": ["Technology & Electronics"],
    "primary_mutual_category": [],
    "items": [{
        "type": "laptop",
        "categorical": {
            "condition": "used",
            "brand": "apple",
            "model": "macbook pro"
        },
        "min": {},
        "max": {},
        "range": {
            "capacity": [{"type": "memory", "min": 16, "max": 16, "unit": "gb"}],
            "cost": [{"type": "price", "min": 120000, "max": 120000, "unit": "inr"}],
            "time": [{"type": "age", "min": 3, "max": 3, "unit": "months"}]
        }
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],
    "reasoning": "Seller offering 3 months used MacBook Pro with 16GB RAM at 120k INR"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query3_new, indent=2))

print("\n[Transforming to OLD format...]")
query3_old = normalize_and_validate_v2(query3_new)
print("✓ Transformed successfully")

print(f"\nSeller Details (OLD format):")
print(f"  Intent: {query3_old['intent']}/{query3_old['subintent']}")
print(f"  Range: {query3_old['items'][0]['range']}")

print("\n[Matching against database...]")

matches_q3 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)
    result = listing_matches_v2(db_listing_old, query3_old)

    if result:
        matches_q3.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")

if not matches_q3:
    print("  ✗ No matches found")

print(f"\n✓ Query 3 Results: {len(matches_q3)} matches")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nQuery 1 (Selling @ 60k): {len(matches_q1)} matches")
if matches_q1:
    for idx, listing in matches_q1:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print(f"\nQuery 2 (Selling @ 80k): {len(matches_q2)} matches")
if matches_q2:
    for idx, listing in matches_q2:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print(f"\nQuery 3 (Selling @ 120k): {len(matches_q3)} matches")
if matches_q3:
    for idx, listing in matches_q3:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print("\n" + "="*80)
print("✓ ALL QUERIES PROCESSED")
print("="*80)
