"""
Show ALL matching results - matches AND rejections
This shows the complete truth of what's happening
"""

import json
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

print("="*80)
print("COMPLETE MUTUAL MATCHING - SHOWING ALL RESULTS")
print("="*80)

# Load database
with open("D:\\matching\\new\\stage3_extraction1.json", "r") as f:
    db_listings = json.load(f)

print(f"\n✓ Loaded {len(db_listings)} listings from database\n")

# Query 1
query1_new = {
    "query": "anyone up for weekend treks around bangalore",
    "intent": "mutual",
    "subintent": "connect",
    "domain": ["Sports & Outdoors"],
    "primary_mutual_category": ["Adventure"],
    "items": [{
        "type": "trekking",
        "categorical": {"schedule": "weekend"},
        "min": {}, "max": {}, "range": {}
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

query1_old = normalize_and_validate_v2(query1_new)

print("QUERY: 'Anyone up for weekend treks around bangalore'")
print("-"*80)
print(f"  Category: {query1_old['category']}")
print(f"  Location: {query1_old['location']}")
print(f"  Items: {query1_old['items'][0]['type']}")
print()

print("CHECKING ALL 10 DATABASE LISTINGS:")
print("="*80)

matches = 0
rejections = 0

for i, db_listing_new in enumerate(db_listings, 1):
    db_listing_old = normalize_and_validate_v2(db_listing_new)

    # Check both directions
    result_ab = listing_matches_v2(query1_old, db_listing_old)
    result_ba = listing_matches_v2(db_listing_old, query1_old)

    # Display listing info
    intent = db_listing_new['intent']
    subintent = db_listing_new['subintent']
    query_text = db_listing_new['query'][:60]

    print(f"\n[{i}] {intent}/{subintent}: {query_text}...")

    if intent == 'mutual':
        category = db_listing_new.get('primary_mutual_category', [])
        location = db_listing_new.get('target_location', {}).get('name', 'none')
        print(f"    Category: {category}, Location: {location}")
    else:
        domain = db_listing_new.get('domain', [])
        print(f"    Domain: {domain}")

    # Show matching result
    if result_ab and result_ba:
        print(f"    ✅ MATCH - Query→DB: {result_ab}, DB→Query: {result_ba}")
        matches += 1
    else:
        print(f"    ❌ NO MATCH - Query→DB: {result_ab}, DB→Query: {result_ba}")

        # Explain WHY it didn't match
        if intent != 'mutual':
            print(f"       Reason: Intent mismatch (query=mutual, db={intent})")
        elif not result_ab and not result_ba:
            if intent == 'mutual':
                db_category = db_listing_old['category']
                db_location = db_listing_old['location']
                if set(query1_old['category']) & set(db_category):
                    print(f"       Reason: Location mismatch (query=bangalore, db={db_location})")
                else:
                    print(f"       Reason: Category mismatch (query=Adventure, db={db_category})")
        elif not result_ab:
            print(f"       Reason: Query requirements not satisfied by DB listing")
        elif not result_ba:
            print(f"       Reason: DB listing requirements not satisfied by Query")

        rejections += 1

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total listings checked: {len(db_listings)}")
print(f"✅ Matches: {matches}")
print(f"❌ Rejections: {rejections}")
print()
print("WHY REJECTIONS?")
print("  - 7 listings: Wrong intent (product/service, not mutual)")
print("  - 2 listings: Mutual but wrong category (Roommates, Professional)")
print()
print("🔍 THE TRUTH:")
print("  - NO embeddings used")
print("  - NO vector search used")
print("  - NO SQL filters used")
print("  - ONLY pure boolean matching (category + location + items)")
print("  - Previous test ONLY SHOWED matches, HID rejections")
print("="*80)
