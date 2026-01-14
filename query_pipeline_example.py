"""
INTEGRATION EXAMPLE: Phase 3.4 (Retrieval) + Phase 2.8 (Matching)

Demonstrates complete query pipeline:
1. Retrieve candidates (Phase 3.4)
2. Boolean matching (Phase 2.8)
3. Return valid matches

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

from typing import List, Dict, Any
from retrieval_service import RetrievalClients, retrieve_candidates
from mutual_matcher import mutual_listing_matches


def query_pipeline(
    query_listing: Dict[str, Any],
    retrieval_clients: RetrievalClients,
    retrieval_limit: int = 100,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Complete query pipeline: retrieval + boolean matching.

    Steps:
    1. Retrieve candidates via SQL + vector search (Phase 3.4)
    2. Fetch full listing data from Supabase
    3. Boolean matching via Phase 2.8
    4. Return valid matches

    Args:
        query_listing: Normalized query listing
        retrieval_clients: Initialized RetrievalClients
        retrieval_limit: Number of candidates to retrieve
        verbose: Print progress messages

    Returns:
        List of valid match listings (full objects)
    """
    if verbose:
        print()
        print("=" * 70)
        print("QUERY PIPELINE: RETRIEVAL + MATCHING")
        print("=" * 70)
        print()

    intent = query_listing.get("intent")

    # Step 1: Retrieve candidates (Phase 3.4)
    if verbose:
        print(f"Step 1: Retrieving candidates (limit={retrieval_limit})...")

    candidate_ids = retrieve_candidates(
        retrieval_clients,
        query_listing,
        limit=retrieval_limit,
        use_sql_filter=True,
        verbose=False
    )

    if verbose:
        print(f"  → Retrieved {len(candidate_ids)} candidates")
        print()

    # Step 2: Fetch full listings from Supabase
    if verbose:
        print(f"Step 2: Fetching full listings from Supabase...")

    # Select table based on intent
    if intent == "product":
        table_name = "product_listings"
    elif intent == "service":
        table_name = "service_listings"
    elif intent == "mutual":
        table_name = "mutual_listings"
    else:
        raise ValueError(f"Unknown intent: {intent}")

    candidates = []
    for listing_id in candidate_ids:
        response = retrieval_clients.supabase.table(table_name).select("data").eq("id", listing_id).execute()
        if response.data:
            candidates.append(response.data[0]["data"])

    if verbose:
        print(f"  → Fetched {len(candidates)} listings")
        print()

    # Step 3: Boolean matching (Phase 2.8)
    if verbose:
        print(f"Step 3: Boolean matching (Phase 2.8)...")

    valid_matches = []
    for candidate in candidates:
        try:
            if mutual_listing_matches(query_listing, candidate):
                valid_matches.append(candidate)
        except Exception as e:
            # Skip candidates with matching errors
            if verbose:
                print(f"  Warning: Matching error for candidate: {e}")
            continue

    if verbose:
        print(f"  → {len(valid_matches)} valid matches")
        print()

    # Summary
    if verbose:
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Candidates retrieved: {len(candidate_ids)}")
        print(f"Candidates fetched: {len(candidates)}")
        print(f"Valid matches: {len(valid_matches)}")
        print(f"Match rate: {len(valid_matches)/len(candidates)*100:.1f}%" if candidates else "N/A")
        print()

    return valid_matches


# ============================================================================
# MAIN (EXAMPLE)
# ============================================================================

def main():
    """
    Example usage of complete query pipeline.

    NOTE: Requires actual data in Supabase/Qdrant.
    """
    print()
    print("=" * 70)
    print("QUERY PIPELINE EXAMPLE")
    print("=" * 70)
    print()

    # Initialize retrieval clients
    print("Initializing retrieval clients...")
    clients = RetrievalClients()
    try:
        clients.initialize()
        print()
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print()
        print("Make sure:")
        print("  - SUPABASE_URL and SUPABASE_KEY environment variables are set")
        print("  - Qdrant is running")
        print("  - Data has been ingested (Phase 3.3)")
        return

    print("=" * 70)
    print("READY FOR QUERIES")
    print("=" * 70)
    print()
    print("Example usage:")
    print()
    print("  query_listing = {")
    print("      'intent': 'product',")
    print("      'subintent': 'buyer',")
    print("      'domain': ['electronics'],")
    print("      'items': [{'type': 'laptop', 'categorical': {'brand': 'Apple'}}],")
    print("      # ... other fields")
    print("  }")
    print()
    print("  matches = query_pipeline(query_listing, clients)")
    print()
    print("NO MOCK DATA PROVIDED - Production code only")
    print()


if __name__ == "__main__":
    main()
