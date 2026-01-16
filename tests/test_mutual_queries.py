"""
Test mutual intent queries against database
Database: stage3_extraction1.json (10 listings)
Queries: 4 weekend trek queries with different locations
"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

print("="*80)
print("MUTUAL INTENT QUERY MATCHING TEST")
print("="*80)

# ============================================================================
# LOAD DATABASE
# ============================================================================

print("\n[STEP 1] Loading Database (stage3_extraction1.json)...")
print("-"*80)

with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    db_listings = json.load(f)

print(f"✓ Loaded {len(db_listings)} listings from database")
print("\nDatabase - Mutual Intent Listings:")
for i, listing in enumerate(db_listings, 1):
    if listing.get('intent') == 'mutual':
        query_text = listing.get('query', '')[:70]
        category = listing.get('primary_mutual_category', [])
        location = listing.get('target_location', {})
        print(f"  [{i}] {query_text}...")
        print(f"      Category: {category}, Location: {location}")

# ============================================================================
# QUERY 1: "anyone up for weekend treks around bangalore"
# ============================================================================

print("\n" + "="*80)
print("QUERY 1: Anyone up for weekend treks around bangalore")
print("="*80)

query1_new = {
    "query": "anyone up for weekend treks around bangalore",
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["Sports & Outdoors"],
    "primary_mutual_category": ["Adventure"],
    "items": [{
        "type": "trekking",
        "categorical": {
            "schedule": "weekend"
        },
        "min": {},
        "max": {},
        "range": {}
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {"name": "bangalore"},
    "location_match_mode": "explicit",
    "location_exclusions": [],
    "reasoning": "Looking for weekend trek partners around Bangalore"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query1_new, indent=2))

print("\n[Transforming to OLD format...]")
query1_old = normalize_and_validate_v2(query1_new)
print("✓ Transformed successfully")

print(f"\nQuery Details (OLD format):")
print(f"  Intent: {query1_old['intent']}/{query1_old['subintent']}")
print(f"  Category: {query1_old['category']}")
print(f"  Item: {query1_old['items'][0]['type']}")
print(f"  Categorical: {query1_old['items'][0]['categorical']}")
print(f"  Location: {query1_old['location']}")
print(f"  Location Mode: {query1_old['locationmode']}")

print("\n[Matching against database...]")
print("Looking for MUTUAL/CONNECT with same category and location...")

matches_q1 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    # For mutual matching, both directions should match (bidirectional)
    # Check A → B and B → A
    result_ab = listing_matches_v2(query1_old, db_listing_old)
    result_ba = listing_matches_v2(db_listing_old, query1_old)

    # Both directions should match for mutual
    if result_ab and result_ba:
        matches_q1.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")
        print(f"      Category: {db_listing_new.get('primary_mutual_category', [])}")
        print(f"      Location: {db_listing_new.get('target_location', {})}")

if not matches_q1:
    print("  ✗ No mutual matches found")

print(f"\n✓ Query 1 Results: {len(matches_q1)} matches")

# ============================================================================
# QUERY 2: "i was going for weekend treks around bangalore is their any one to join"
# ============================================================================

print("\n" + "="*80)
print("QUERY 2: I was going for weekend treks around bangalore, anyone to join?")
print("="*80)

query2_new = {
    "query": "i was going for weekend treks around bangalore is their any one to join",
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["Sports & Outdoors"],
    "primary_mutual_category": ["Adventure"],
    "items": [{
        "type": "trekking",
        "categorical": {
            "schedule": "weekend"
        },
        "min": {},
        "max": {},
        "range": {}
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {"name": "bangalore"},
    "location_match_mode": "explicit",
    "location_exclusions": [],
    "reasoning": "Looking for weekend trek partners around Bangalore"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query2_new, indent=2))

print("\n[Transforming to OLD format...]")
query2_old = normalize_and_validate_v2(query2_new)
print("✓ Transformed successfully")

print(f"\nQuery Details: Category={query2_old['category']}, Location={query2_old['location']}")

print("\n[Matching against database...]")

matches_q2 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    result_ab = listing_matches_v2(query2_old, db_listing_old)
    result_ba = listing_matches_v2(db_listing_old, query2_old)

    if result_ab and result_ba:
        matches_q2.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")

if not matches_q2:
    print("  ✗ No mutual matches found")

print(f"\n✓ Query 2 Results: {len(matches_q2)} matches")

# ============================================================================
# QUERY 3: "anyone up for weekend treks around chennai?"
# ============================================================================

print("\n" + "="*80)
print("QUERY 3: Anyone up for weekend treks around Chennai?")
print("="*80)

query3_new = {
    "query": "anyone up for weekend treks around chennai?",
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["Sports & Outdoors"],
    "primary_mutual_category": ["Adventure"],
    "items": [{
        "type": "trekking",
        "categorical": {
            "schedule": "weekend"
        },
        "min": {},
        "max": {},
        "range": {}
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {"name": "chennai"},
    "location_match_mode": "explicit",
    "location_exclusions": [],
    "reasoning": "Looking for weekend trek partners around Chennai"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query3_new, indent=2))

print("\n[Transforming to OLD format...]")
query3_old = normalize_and_validate_v2(query3_new)
print("✓ Transformed successfully")

print(f"\nQuery Details: Category={query3_old['category']}, Location={query3_old['location']}")

print("\n[Matching against database...]")
print("Note: Looking for Chennai-based trek partners...")

matches_q3 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    result_ab = listing_matches_v2(query3_old, db_listing_old)
    result_ba = listing_matches_v2(db_listing_old, query3_old)

    if result_ab and result_ba:
        matches_q3.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")
    elif result_ab or result_ba:
        print(f"  ⚠ PARTIAL MATCH #{i}: {db_listing_new['query'][:60]}...")
        print(f"      A→B: {result_ab}, B→A: {result_ba}")
        print(f"      DB Location: {db_listing_old.get('location', 'N/A')}")

if not matches_q3:
    print("  ✗ No mutual matches found")

print(f"\n✓ Query 3 Results: {len(matches_q3)} matches")

# ============================================================================
# QUERY 4: "anyone up for weekend treks" (NO LOCATION)
# ============================================================================

print("\n" + "="*80)
print("QUERY 4: Anyone up for weekend treks (no location specified)")
print("="*80)

query4_new = {
    "query": "anyone up for weekend treks",
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["Sports & Outdoors"],
    "primary_mutual_category": ["Adventure"],
    "items": [{
        "type": "trekking",
        "categorical": {
            "schedule": "weekend"
        },
        "min": {},
        "max": {},
        "range": {}
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],
    "reasoning": "Looking for weekend trek partners, location flexible"
}

print("\n[NEW Schema JSON]:")
print(json.dumps(query4_new, indent=2))

print("\n[Transforming to OLD format...]")
query4_old = normalize_and_validate_v2(query4_new)
print("✓ Transformed successfully")

print(f"\nQuery Details: Category={query4_old['category']}, Location={query4_old['location']} (empty=flexible)")

print("\n[Matching against database...]")
print("Note: No location constraint, should match any location...")

matches_q4 = []
for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    result_ab = listing_matches_v2(query4_old, db_listing_old)
    result_ba = listing_matches_v2(db_listing_old, query4_old)

    if result_ab and result_ba:
        matches_q4.append((i, db_listing_new))
        print(f"  ✓ MATCH #{i}: {db_listing_new['query'][:60]}...")

if not matches_q4:
    print("  ✗ No mutual matches found")

print(f"\n✓ Query 4 Results: {len(matches_q4)} matches")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nQuery 1 (Weekend treks around Bangalore): {len(matches_q1)} matches")
if matches_q1:
    for idx, listing in matches_q1:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print(f"\nQuery 2 (Going for weekend treks around Bangalore, anyone join?): {len(matches_q2)} matches")
if matches_q2:
    for idx, listing in matches_q2:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print(f"\nQuery 3 (Weekend treks around Chennai): {len(matches_q3)} matches")
if matches_q3:
    for idx, listing in matches_q3:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")
else:
    print("  - No matches (expected: Chennai != Bangalore)")

print(f"\nQuery 4 (Weekend treks, no location): {len(matches_q4)} matches")
if matches_q4:
    for idx, listing in matches_q4:
        print(f"  - Match #{idx}: {listing['query'][:60]}...")

print("\n" + "="*80)
print("✓ ALL MUTUAL QUERIES PROCESSED")
print("="*80)
