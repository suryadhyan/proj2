"""
PHASE 3.3: INGESTION PIPELINE

Responsibilities:
- Insert normalized listings into Supabase
- Generate 1024D embeddings
- Insert embeddings into Qdrant with payload

Authority: VRIDDHI Architecture Document
Dependencies:
- supabase-py
- qdrant-client
- sentence-transformers

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from supabase import create_client, Client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from embedding_builder import build_embedding_text


# ============================================================================
# CONFIGURATION
# ============================================================================

# Supabase (from environment variables)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Qdrant (Cloud or Local)
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")  # Cloud endpoint
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")    # Cloud API key
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")  # Local fallback
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))  # Local fallback

# Embedding model
# Default to a smaller model for safety in cloud environments (avoid OOM)
# Can be overridden by env var to use "BAAI/bge-large-en-v1.5" if resources permit
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 

# Dimension depends on model:
# - BAAI/bge-large-en-v1.5: 1024
# - all-MiniLM-L6-v2: 384
EMBEDDING_DIM = 1024 if "large" in EMBEDDING_MODEL else 384


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

class IngestionClients:
    """Container for Supabase and Qdrant clients."""

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

        # Embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Loaded embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIM}D)")


# ============================================================================
# SUPABASE INSERTION
# ============================================================================

def insert_to_supabase(
    client: Client,
    listing: Dict[str, Any],
    listing_id: Optional[str] = None
) -> str:
    """
    Insert listing into appropriate Supabase table.

    Table selection based on intent:
    - product → product_listings
    - service → service_listings
    - mutual → mutual_listings

    Args:
        client: Supabase client
        listing: Normalized listing object
        listing_id: Optional UUID (generated if not provided)

    Returns:
        listing_id (UUID as string)

    Raises:
        ValueError: If intent unknown or insertion fails
    """
    intent = listing.get("intent")
    if not intent:
        raise ValueError("Listing missing 'intent' field")

    # Generate ID if not provided
    if not listing_id:
        listing_id = str(uuid.uuid4())

    # Select table
    if intent == "product":
        table_name = "product_listings"
    elif intent == "service":
        table_name = "service_listings"
    elif intent == "mutual":
        table_name = "mutual_listings"
    else:
        raise ValueError(f"Unknown intent: {intent}")

    # Prepare data for insertion
    data = {
        "id": listing_id,
        "data": listing,  # Store entire listing as JSONB
        "created_at": datetime.utcnow().isoformat()
    }

    # Insert
    try:
        response = client.table(table_name).insert(data).execute()
        return listing_id
    except Exception as e:
        raise ValueError(f"Supabase insertion failed for {table_name}: {e}")


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embedding(
    model: SentenceTransformer,
    text: str
) -> list:
    """
    Generate 1024D embedding vector from text.

    Args:
        model: Sentence transformer model
        text: Input text

    Returns:
        Embedding vector (list of floats, length 1024)

    Raises:
        ValueError: If embedding dimension incorrect
    """
    embedding = model.encode(text, convert_to_tensor=False)

    # Verify dimension
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}"
        )

    return embedding.tolist()


# ============================================================================
# QDRANT INSERTION
# ============================================================================

def insert_to_qdrant(
    client: QdrantClient,
    listing_id: str,
    listing: Dict[str, Any],
    embedding: list
) -> None:
    """
    Insert embedding + payload into appropriate Qdrant collection.

    Collection selection based on intent:
    - product → product_vectors
    - service → service_vectors
    - mutual → mutual_vectors

    Payload structure:
    - listing_id (UUID)
    - intent (string)
    - domain (array, for product/service) OR category (array, for mutual)
    - created_at (unix timestamp)

    Args:
        client: Qdrant client
        listing_id: UUID string
        listing: Normalized listing object
        embedding: 1024D vector

    Raises:
        ValueError: If intent unknown or insertion fails
    """
    intent = listing.get("intent")
    if not intent:
        raise ValueError("Listing missing 'intent' field")

    # Select collection
    if intent == "product":
        collection_name = "product_vectors"
    elif intent == "service":
        collection_name = "service_vectors"
    elif intent == "mutual":
        collection_name = "mutual_vectors"
    else:
        raise ValueError(f"Unknown intent: {intent}")

    # Build payload
    payload = {
        "listing_id": listing_id,
        "intent": intent,
        "created_at": int(time.time())
    }

    # Add domain or category
    if intent == "product" or intent == "service":
        payload["domain"] = listing.get("domain", [])
    elif intent == "mutual":
        payload["category"] = listing.get("category", [])

    # Create point
    point = PointStruct(
        id=listing_id,  # Use listing_id as Qdrant point ID
        vector=embedding,
        payload=payload
    )

    # Upsert (insert or update)
    try:
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )
    except Exception as e:
        raise ValueError(f"Qdrant insertion failed for {collection_name}: {e}")


# ============================================================================
# ORCHESTRATION
# ============================================================================

def ingest_listing(
    clients: IngestionClients,
    listing: Dict[str, Any],
    listing_id: Optional[str] = None,
    verbose: bool = True
) -> Tuple[str, list]:
    """
    Complete ingestion pipeline: Supabase + Qdrant.

    Steps:
    1. Validate listing structure
    2. Insert to Supabase (get listing_id)
    3. Build embedding text
    4. Generate embedding vector
    5. Insert to Qdrant with payload

    Args:
        clients: Initialized IngestionClients
        listing: Normalized listing object
        listing_id: Optional UUID (generated if not provided)
        verbose: Print progress messages

    Returns:
        Tuple of (listing_id, embedding_vector)

    Raises:
        ValueError: If any step fails
    """
    if verbose:
        print(f"Ingesting listing (intent: {listing.get('intent')})")

    # Step 1: Insert to Supabase
    if verbose:
        print("  [1/4] Inserting to Supabase...")
    listing_id = insert_to_supabase(clients.supabase, listing, listing_id)
    if verbose:
        print(f"        ✓ Inserted with ID: {listing_id}")

    # Step 2: Build embedding text
    if verbose:
        print("  [2/4] Building embedding text...")
    embedding_text = build_embedding_text(listing)
    if verbose:
        print(f"        ✓ Built text ({len(embedding_text)} chars)")

    # Step 3: Generate embedding
    if verbose:
        print("  [3/4] Generating embedding...")
    embedding = generate_embedding(clients.embedding_model, embedding_text)
    if verbose:
        print(f"        ✓ Generated {len(embedding)}D vector")

    # Step 4: Insert to Qdrant
    if verbose:
        print("  [4/4] Inserting to Qdrant...")
    insert_to_qdrant(clients.qdrant, listing_id, listing, embedding)
    if verbose:
        print(f"        ✓ Inserted to Qdrant")

    if verbose:
        print(f"✓ Ingestion complete: {listing_id}")
        print()

    return listing_id, embedding


def ingest_batch(
    clients: IngestionClients,
    listings: list,
    verbose: bool = True
) -> list:
    """
    Ingest multiple listings in batch.

    Args:
        clients: Initialized IngestionClients
        listings: List of normalized listing objects
        verbose: Print progress messages

    Returns:
        List of listing_ids

    Raises:
        ValueError: If any ingestion fails
    """
    if verbose:
        print(f"=" * 70)
        print(f"BATCH INGESTION: {len(listings)} listings")
        print(f"=" * 70)
        print()

    listing_ids = []
    failed = []

    for i, listing in enumerate(listings, 1):
        try:
            if verbose:
                print(f"[{i}/{len(listings)}]")

            listing_id, _ = ingest_listing(clients, listing, verbose=verbose)
            listing_ids.append(listing_id)

        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed.append((i, listing, e))
            print()

    # Summary
    if verbose:
        print(f"=" * 70)
        print(f"BATCH COMPLETE")
        print(f"=" * 70)
        print(f"Success: {len(listing_ids)}/{len(listings)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for idx, _, error in failed:
                print(f"  - Listing {idx}: {error}")
        print()

    return listing_ids


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

def main():
    """
    Main function for testing ingestion pipeline.

    NOTE: This is for testing only. Production use should call ingest_listing() directly.
    """
    print()
    print("=" * 70)
    print("PHASE 3.3: INGESTION PIPELINE TEST")
    print("=" * 70)
    print()

    # Initialize clients
    print("Initializing clients...")
    clients = IngestionClients()
    try:
        clients.initialize()
        print()
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print()
        print("Make sure:")
        print("  - SUPABASE_URL and SUPABASE_KEY environment variables are set")
        print("  - Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        return

    # Example: NO MOCK DATA - user must provide real listings
    print("=" * 70)
    print("READY FOR INGESTION")
    print("=" * 70)
    print()
    print("Usage:")
    print("  from ingestion_pipeline import IngestionClients, ingest_listing")
    print("  clients = IngestionClients()")
    print("  clients.initialize()")
    print("  listing_id, embedding = ingest_listing(clients, listing)")
    print()
    print("NO MOCK DATA PROVIDED - This is production code only")
    print()


if __name__ == "__main__":
    main()
