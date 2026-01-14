"""
PHASE 3.2: QDRANT SETUP VERIFICATION

Quick verification script to check Qdrant collections.
Run this after qdrant_setup.py to confirm setup.

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance


def verify_collections():
    """Verify Qdrant collections are set up correctly."""

    print()
    print("=" * 70)
    print("QDRANT SETUP VERIFICATION")
    print("=" * 70)
    print()

    # Connect to Qdrant
    try:
        client = QdrantClient(host="localhost", port=6333)
        print("✓ Connected to Qdrant (localhost:6333)")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    print()

    # Expected collections
    expected = ["product_vectors", "service_vectors", "mutual_vectors"]

    # Get all collections
    collections = client.get_collections().collections
    collection_names = {col.name for col in collections}

    print("Collections found:")
    for name in collection_names:
        print(f"  - {name}")
    print()

    # Check each expected collection
    all_ok = True
    for collection_name in expected:
        if collection_name not in collection_names:
            print(f"✗ Missing collection: {collection_name}")
            all_ok = False
            continue

        # Get collection info
        try:
            info = client.get_collection(collection_name)

            print(f"Collection: {collection_name}")
            print(f"  Status: {info.status}")
            print(f"  Vector size: {info.config.params.vectors.size}D")
            print(f"  Distance: {info.config.params.vectors.distance}")
            print(f"  Points: {info.points_count}")
            print(f"  Vectors: {info.vectors_count}")

            # Verify configuration
            if info.config.params.vectors.size != 1024:
                print(f"  ✗ ERROR: Expected 1024D vectors")
                all_ok = False
            elif info.config.params.vectors.distance != Distance.COSINE:
                print(f"  ✗ ERROR: Expected COSINE distance")
                all_ok = False
            else:
                print(f"  ✓ Configuration OK")

            print()

        except Exception as e:
            print(f"✗ Error checking {collection_name}: {e}")
            all_ok = False

    print("=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ VERIFICATION FAILED")
    print("=" * 70)
    print()

    return all_ok


if __name__ == "__main__":
    verify_collections()
