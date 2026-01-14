"""
PHASE 3.2: QDRANT RESET & COLLECTION CREATION

Responsibilities:
- Delete existing Qdrant collections (if present)
- Create new collections for product, service, and mutual vectors
- Define vector configuration (1024D, cosine similarity)
- Define payload schema for SQL-like filtering

Authority: VRIDDHI Architecture Document
Dependencies: qdrant-client

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    OptimizersConfigDiff,
    HnswConfigDiff
)
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

VECTOR_SIZE = 1024
DISTANCE_METRIC = Distance.COSINE  # Standard for text embeddings

COLLECTIONS = [
    "product_vectors",
    "service_vectors",
    "mutual_vectors"
]


# ============================================================================
# COLLECTION CREATION
# ============================================================================

def reset_collections(client: QdrantClient, dry_run: bool = False) -> None:
    """
    Delete existing collections if they exist.

    Args:
        client: Qdrant client instance
        dry_run: If True, only report what would be deleted
    """
    print("=" * 70)
    print("STEP 1: COLLECTION RESET")
    print("=" * 70)
    print()

    existing_collections = client.get_collections().collections
    existing_names = {col.name for col in existing_collections}

    for collection_name in COLLECTIONS:
        if collection_name in existing_names:
            if dry_run:
                print(f"[DRY RUN] Would delete collection: {collection_name}")
            else:
                print(f"Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
                print(f"✓ Deleted: {collection_name}")
        else:
            print(f"Collection does not exist (skip): {collection_name}")

    print()


def create_collections(client: QdrantClient, dry_run: bool = False) -> None:
    """
    Create Qdrant collections with proper vector configuration.

    Collection specifications:
    - Vector size: 1024D (dense embeddings)
    - Distance metric: Cosine similarity (standard for text)
    - Payload indexed fields: listing_id, intent, domain, category, created_at

    Args:
        client: Qdrant client instance
        dry_run: If True, only report what would be created
    """
    print("=" * 70)
    print("STEP 2: COLLECTION CREATION")
    print("=" * 70)
    print()

    vector_config = VectorParams(
        size=VECTOR_SIZE,
        distance=DISTANCE_METRIC
    )

    # Optimizers configuration for production use
    optimizers_config = OptimizersConfigDiff(
        indexing_threshold=20000,  # Start indexing after 20k vectors
        memmap_threshold=50000     # Use memory mapping after 50k vectors
    )

    # HNSW index configuration
    hnsw_config = HnswConfigDiff(
        m=16,                 # Number of edges per node
        ef_construct=100,     # Construction time accuracy
        full_scan_threshold=10000  # Use full scan for small collections
    )

    for collection_name in COLLECTIONS:
        if dry_run:
            print(f"[DRY RUN] Would create collection: {collection_name}")
            print(f"  - Vector size: {VECTOR_SIZE}D")
            print(f"  - Distance metric: {DISTANCE_METRIC}")
        else:
            print(f"Creating collection: {collection_name}")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config,
                optimizers_config=optimizers_config,
                hnsw_config=hnsw_config
            )

            print(f"✓ Created: {collection_name}")
            print(f"  - Vector size: {VECTOR_SIZE}D")
            print(f"  - Distance metric: {DISTANCE_METRIC}")

        print()


def create_payload_indexes(client: QdrantClient, dry_run: bool = False) -> None:
    """
    Create payload indexes for efficient filtering.

    Indexed fields:
    - listing_id: UUID (exact match filtering)
    - intent: string (exact match filtering)
    - domain: array of strings (product/service filtering)
    - category: array of strings (mutual filtering)
    - created_at: timestamp (range filtering)

    Args:
        client: Qdrant client instance
        dry_run: If True, only report what would be created
    """
    print("=" * 70)
    print("STEP 3: PAYLOAD INDEX CREATION")
    print("=" * 70)
    print()

    # Product and Service collections: index by domain
    product_service_indexes = [
        ("listing_id", "keyword"),      # UUID exact match
        ("intent", "keyword"),           # "product" or "service"
        ("domain", "keyword"),           # Array of domain strings
        ("created_at", "integer")        # Unix timestamp for range queries
    ]

    # Mutual collection: index by category (not domain)
    mutual_indexes = [
        ("listing_id", "keyword"),
        ("intent", "keyword"),           # "mutual"
        ("category", "keyword"),         # Array of category strings
        ("created_at", "integer")
    ]

    # Create indexes for product and service collections
    for collection_name in ["product_vectors", "service_vectors"]:
        print(f"Creating payload indexes for: {collection_name}")

        for field_name, field_type in product_service_indexes:
            if dry_run:
                print(f"  [DRY RUN] Would create index: {field_name} ({field_type})")
            else:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                print(f"  ✓ Indexed: {field_name} ({field_type})")

        print()

    # Create indexes for mutual collection
    print(f"Creating payload indexes for: mutual_vectors")

    for field_name, field_type in mutual_indexes:
        if dry_run:
            print(f"  [DRY RUN] Would create index: {field_name} ({field_type})")
        else:
            client.create_payload_index(
                collection_name="mutual_vectors",
                field_name=field_name,
                field_schema=field_type
            )
            print(f"  ✓ Indexed: {field_name} ({field_type})")

    print()


def verify_setup(client: QdrantClient) -> bool:
    """
    Verify that all collections were created correctly.

    Returns:
        True if all collections exist and are configured correctly
    """
    print("=" * 70)
    print("STEP 4: VERIFICATION")
    print("=" * 70)
    print()

    all_valid = True

    for collection_name in COLLECTIONS:
        try:
            info = client.get_collection(collection_name)

            print(f"Collection: {collection_name}")
            print(f"  - Vectors count: {info.vectors_count}")
            print(f"  - Points count: {info.points_count}")
            print(f"  - Status: {info.status}")
            print(f"  - Vector size: {info.config.params.vectors.size}D")
            print(f"  - Distance: {info.config.params.vectors.distance}")

            # Verify configuration
            if info.config.params.vectors.size != VECTOR_SIZE:
                print(f"  ✗ ERROR: Vector size mismatch (expected {VECTOR_SIZE})")
                all_valid = False
            elif info.config.params.vectors.distance != DISTANCE_METRIC:
                print(f"  ✗ ERROR: Distance metric mismatch (expected {DISTANCE_METRIC})")
                all_valid = False
            else:
                print(f"  ✓ Configuration valid")

            print()

        except Exception as e:
            print(f"✗ ERROR: Collection {collection_name} not found or invalid")
            print(f"  Error: {e}")
            print()
            all_valid = False

    if all_valid:
        print("=" * 70)
        print("✅ ALL COLLECTIONS VERIFIED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("=" * 70)
        print("❌ VERIFICATION FAILED")
        print("=" * 70)

    return all_valid


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(dry_run: bool = False):
    """
    Execute Qdrant setup: reset and create collections.

    Args:
        dry_run: If True, only report what would be done without making changes
    """
    print()
    print("=" * 70)
    print("PHASE 3.2: QDRANT RESET & COLLECTION CREATION")
    print("=" * 70)
    print()

    if dry_run:
        print("⚠ DRY RUN MODE: No changes will be made")
        print()

    # Connect to Qdrant
    try:
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print("✓ Connected to Qdrant")
        print()
    except Exception as e:
        print(f"✗ ERROR: Failed to connect to Qdrant")
        print(f"  Error: {e}")
        print()
        print("Make sure Qdrant is running:")
        print("  docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # Execute setup steps
    reset_collections(client, dry_run=dry_run)
    create_collections(client, dry_run=dry_run)

    if not dry_run:
        create_payload_indexes(client, dry_run=dry_run)

        # Verify setup
        if not verify_setup(client):
            print("Setup completed with errors. Please review the output above.")
            sys.exit(1)

    print()
    print("=" * 70)
    print("PHASE 3.2 SETUP COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - PHASE 3.3: Implement ingestion pipeline")
    print("  - PHASE 3.4: Implement embedding generation")
    print("  - PHASE 3.5: Implement retrieval + ranking")
    print()


if __name__ == "__main__":
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    main(dry_run=dry_run)
