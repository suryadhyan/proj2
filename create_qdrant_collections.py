"""
Create Qdrant collections in Qdrant Cloud
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

print(f"🔗 Connecting to Qdrant Cloud: {QDRANT_ENDPOINT}")

client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

print("✅ Connected to Qdrant Cloud!\n")

collections_to_create = [
    "product_vectors",
    "service_vectors",
    "mutual_vectors"
]

print(f"📦 Creating {len(collections_to_create)} collections...")
print(f"   Vector dimension: {EMBEDDING_DIM}D (all-MiniLM-L6-v2)")
print(f"   Distance metric: Cosine\n")

for collection_name in collections_to_create:
    try:
        # Check if collection exists
        try:
            client.get_collection(collection_name)
            print(f"  ℹ️  {collection_name} - Already exists")
            continue
        except Exception:
            pass  # Collection doesn't exist, create it

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        print(f"  ✅ {collection_name} - Created successfully")

    except Exception as e:
        print(f"  ❌ {collection_name} - Error: {e}")

print("\n" + "="*80)
print("✅ QDRANT COLLECTIONS READY!")
print("="*80)
print("\nYou can now run: python3 test_complete_flow.py")
