"""
PHASE 3.4: RETRIEVAL SERVICE (CANDIDATE SELECTION)

Responsibilities:
- SQL filtering via Supabase
- Qdrant vector search with payload filters
- Return candidate listing_ids ONLY

NO ranking.
NO boolean matching.
NO scoring.
NO inference.

Authority: VRIDDHI Architecture Document
Dependencies: supabase-py, qdrant-client, sentence-transformers

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from sentence_transformers import SentenceTransformer

from embedding_builder import build_embedding_text


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supabase (from environment)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Qdrant
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")  # Cloud endpoint
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")    # Cloud API key
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")  # Local fallback
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))  # Local fallback

# Embedding model (same as ingestion)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval parameters
DEFAULT_LIMIT = 100  # Top-k candidates to return


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

class RetrievalClients:
    """Container for retrieval clients."""

    def __init__(self):
        self.supabase: Optional[Client] = None
        self.qdrant: Optional[QdrantClient] = None
        self.embedding_model: Optional[SentenceTransformer] = None

    def initialize(self):
        """Initialize all clients."""
        # Supabase
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables required"
            )

        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"✓ Connected to Supabase: {SUPABASE_URL}")

        # Qdrant (Cloud or Local)
        if QDRANT_ENDPOINT and QDRANT_API_KEY:
            # Use Qdrant Cloud
            self.qdrant = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
            print(f"✓ Connected to Qdrant Cloud: {QDRANT_ENDPOINT}")
        else:
            # Use local Qdrant
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            print(f"✓ Connected to Qdrant (local): {QDRANT_HOST}:{QDRANT_PORT}")

        # Embedding model (for query encoding)
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Loaded embedding model: {EMBEDDING_MODEL}")


# ============================================================================
# SQL FILTERING (SUPABASE)
# ============================================================================

def sql_filter_product_service(
    client: Client,
    query_listing: Dict[str, Any],
    limit: Optional[int] = None
) -> List[str]:
    """
    SQL filter for product/service listings.

    Filters:
    - intent = query intent
    - domain intersection (query.domain ∩ candidate.domain ≠ ∅)

    Args:
        client: Supabase client
        query_listing: Normalized query listing
        limit: Optional limit on results

    Returns:
        List of listing_ids that pass SQL filters
    """
    intent = query_listing.get("intent")
    if not intent or intent not in ["product", "service"]:
        raise ValueError(f"Invalid intent for product/service filter: {intent}")

    # Select table
    table_name = "product_listings" if intent == "product" else "service_listings"

    # Query domains
    query_domains = query_listing.get("domain", [])
    if not query_domains:
        # No domain filter, return all (up to limit)
        query = client.table(table_name).select("id")
        if limit:
            query = query.limit(limit)
        response = query.execute()
        return [row["id"] for row in response.data]

    # Filter by domain intersection
    # PostgreSQL: Check if arrays overlap using && operator
    # data->'domain' ?| array['electronics', 'computers']
    # This checks if any element in query_domains exists in data->'domain'

    # Note: Supabase Python client doesn't expose ?| directly
    # We need to fetch all and filter in Python (or use RPC)
    # For production, create a Postgres function or use PostgREST filter

    # TEMPORARY: Fetch all and filter in Python
    # TODO: Optimize with Postgres function for domain intersection
    query = client.table(table_name).select("id, data")
    if limit:
        # Fetch more than limit to account for filtering
        query = query.limit(limit * 10)

    response = query.execute()

    # Filter by domain intersection in Python
    filtered_ids = []
    query_domain_set = set(query_domains)

    for row in response.data:
        candidate_domains = row["data"].get("domain", [])
        candidate_domain_set = set(candidate_domains)

        # Check intersection
        if query_domain_set & candidate_domain_set:
            filtered_ids.append(row["id"])

            if limit and len(filtered_ids) >= limit:
                break

    return filtered_ids


def sql_filter_mutual(
    client: Client,
    query_listing: Dict[str, Any],
    limit: Optional[int] = None
) -> List[str]:
    """
    SQL filter for mutual listings.

    Filters:
    - category intersection (query.category ∩ candidate.category ≠ ∅)

    Args:
        client: Supabase client
        query_listing: Normalized query listing
        limit: Optional limit on results

    Returns:
        List of listing_ids that pass SQL filters
    """
    # Query categories
    query_categories = query_listing.get("category", [])
    if not query_categories:
        # No category filter, return all (up to limit)
        query = client.table("mutual_listings").select("id")
        if limit:
            query = query.limit(limit)
        response = query.execute()
        return [row["id"] for row in response.data]

    # Filter by category intersection (same logic as domain)
    # TEMPORARY: Fetch all and filter in Python
    # TODO: Optimize with Postgres function

    query = client.table("mutual_listings").select("id, data")
    if limit:
        query = query.limit(limit * 10)

    response = query.execute()

    # Filter by category intersection in Python
    filtered_ids = []
    query_category_set = set(query_categories)

    for row in response.data:
        candidate_categories = row["data"].get("category", [])
        candidate_category_set = set(candidate_categories)

        # Check intersection
        if query_category_set & candidate_category_set:
            filtered_ids.append(row["id"])

            if limit and len(filtered_ids) >= limit:
                break

    return filtered_ids


# ============================================================================
# QDRANT VECTOR SEARCH
# ============================================================================

def qdrant_search_product_service(
    client: QdrantClient,
    model: SentenceTransformer,
    query_listing: Dict[str, Any],
    sql_filtered_ids: Optional[List[str]] = None,
    limit: int = DEFAULT_LIMIT
) -> List[str]:
    """
    Vector search for product/service listings.

    Steps:
    1. Build query embedding text
    2. Generate query embedding
    3. Search Qdrant with:
       - Dense vector similarity
       - Payload filter (intent, domain)
       - Optional: Filter by sql_filtered_ids

    Args:
        client: Qdrant client
        model: Embedding model
        query_listing: Normalized query listing
        sql_filtered_ids: Optional list of IDs from SQL filtering
        limit: Number of candidates to return

    Returns:
        List of listing_ids (ordered by vector similarity)
    """
    intent = query_listing.get("intent")
    if not intent or intent not in ["product", "service"]:
        raise ValueError(f"Invalid intent: {intent}")

    # Select collection
    collection_name = "product_vectors" if intent == "product" else "service_vectors"

    # Build query embedding
    query_text = build_embedding_text(query_listing)
    query_vector = model.encode(query_text, convert_to_tensor=False).tolist()

    # Build payload filter
    filter_conditions = []

    # Filter by intent
    filter_conditions.append(
        FieldCondition(key="intent", match=MatchValue(value=intent))
    )

    # Filter by domain (intersection check)
    query_domains = query_listing.get("domain", [])
    if query_domains:
        # MatchAny: Returns points where field contains ANY of the specified values
        filter_conditions.append(
            FieldCondition(key="domain", match=MatchAny(any=query_domains))
        )

    # Optionally filter by SQL-filtered IDs
    if sql_filtered_ids:
        # Note: Qdrant doesn't have native "ID IN list" filter
        # For small lists, we can search and post-filter
        # For large lists, this is inefficient - would need alternative approach
        pass  # Post-filter after search

    # Build filter object
    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    # Search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit
    )

    # Extract listing_ids
    candidate_ids = []
    for scored_point in search_result:
        listing_id = scored_point.payload.get("listing_id")
        if listing_id:
            # Post-filter by SQL-filtered IDs if provided
            if sql_filtered_ids is None or listing_id in sql_filtered_ids:
                candidate_ids.append(listing_id)

    return candidate_ids


def qdrant_search_mutual(
    client: QdrantClient,
    model: SentenceTransformer,
    query_listing: Dict[str, Any],
    sql_filtered_ids: Optional[List[str]] = None,
    limit: int = DEFAULT_LIMIT
) -> List[str]:
    """
    Semantic vector search for mutual listings.

    Steps:
    1. Build query embedding text (natural language)
    2. Generate query embedding
    3. Search Qdrant with:
       - Dense vector similarity (semantic)
       - Payload filter (intent, category)

    Args:
        client: Qdrant client
        model: Embedding model
        query_listing: Normalized query listing
        sql_filtered_ids: Optional list of IDs from SQL filtering
        limit: Number of candidates to return

    Returns:
        List of listing_ids (ordered by semantic similarity)
    """
    intent = query_listing.get("intent")
    if intent != "mutual":
        raise ValueError(f"Invalid intent for mutual search: {intent}")

    collection_name = "mutual_vectors"

    # Build query embedding (natural language)
    query_text = build_embedding_text(query_listing)
    query_vector = model.encode(query_text, convert_to_tensor=False).tolist()

    # Build payload filter
    filter_conditions = []

    # Filter by intent
    filter_conditions.append(
        FieldCondition(key="intent", match=MatchValue(value="mutual"))
    )

    # Filter by category (intersection check)
    query_categories = query_listing.get("category", [])
    if query_categories:
        filter_conditions.append(
            FieldCondition(key="category", match=MatchAny(any=query_categories))
        )

    # Build filter object
    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    # Search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit
    )

    # Extract listing_ids
    candidate_ids = []
    for scored_point in search_result:
        listing_id = scored_point.payload.get("listing_id")
        if listing_id:
            # Post-filter by SQL-filtered IDs if provided
            if sql_filtered_ids is None or listing_id in sql_filtered_ids:
                candidate_ids.append(listing_id)

    return candidate_ids


# ============================================================================
# ORCHESTRATION
# ============================================================================

def retrieve_candidates(
    clients: RetrievalClients,
    query_listing: Dict[str, Any],
    limit: int = DEFAULT_LIMIT,
    use_sql_filter: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Retrieve candidate listing_ids for a query listing.

    Pipeline:
    1. SQL filter (Supabase) - optional
    2. Qdrant vector search with payload filters
    3. Return candidate listing_ids

    NO ranking.
    NO boolean matching.
    NO scoring returned.

    Args:
        clients: Initialized RetrievalClients
        query_listing: Normalized query listing
        limit: Number of candidates to return
        use_sql_filter: Whether to apply SQL filtering first
        verbose: Print progress messages

    Returns:
        List of candidate listing_ids (up to limit)

    Raises:
        ValueError: If intent unknown or retrieval fails
    """
    intent = query_listing.get("intent")
    if not intent:
        raise ValueError("Query listing missing 'intent' field")

    if verbose:
        print(f"Retrieving candidates for intent: {intent}")

    # Step 1: SQL filtering (optional)
    sql_filtered_ids = None
    if use_sql_filter:
        if verbose:
            print(f"  [1/2] SQL filtering...")

        if intent == "product" or intent == "service":
            sql_filtered_ids = sql_filter_product_service(
                clients.supabase,
                query_listing,
                limit=limit * 10  # Fetch more for vector filtering
            )
        elif intent == "mutual":
            sql_filtered_ids = sql_filter_mutual(
                clients.supabase,
                query_listing,
                limit=limit * 10
            )
        else:
            raise ValueError(f"Unknown intent: {intent}")

        if verbose:
            print(f"        ✓ SQL filtered to {len(sql_filtered_ids)} candidates")

    # Step 2: Qdrant vector search
    if verbose:
        print(f"  [2/2] Qdrant vector search...")

    if intent == "product" or intent == "service":
        candidate_ids = qdrant_search_product_service(
            clients.qdrant,
            clients.embedding_model,
            query_listing,
            sql_filtered_ids=sql_filtered_ids,
            limit=limit
        )
    elif intent == "mutual":
        candidate_ids = qdrant_search_mutual(
            clients.qdrant,
            clients.embedding_model,
            query_listing,
            sql_filtered_ids=sql_filtered_ids,
            limit=limit
        )
    else:
        raise ValueError(f"Unknown intent: {intent}")

    if verbose:
        print(f"        ✓ Retrieved {len(candidate_ids)} candidates")
        print(f"✓ Retrieval complete")
        print()

    return candidate_ids


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

def main():
    """
    Main function for testing retrieval service.

    NOTE: This is for testing only.
    """
    print()
    print("=" * 70)
    print("PHASE 3.4: RETRIEVAL SERVICE TEST")
    print("=" * 70)
    print()

    # Initialize clients
    print("Initializing clients...")
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
        return

    print("=" * 70)
    print("READY FOR RETRIEVAL")
    print("=" * 70)
    print()
    print("Usage:")
    print("  from retrieval_service import RetrievalClients, retrieve_candidates")
    print("  clients = RetrievalClients()")
    print("  clients.initialize()")
    print("  candidate_ids = retrieve_candidates(clients, query_listing)")
    print()
    print("NO MOCK DATA PROVIDED - Production code only")
    print()


if __name__ == "__main__":
    main()
