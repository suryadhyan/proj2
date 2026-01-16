"""
Quick test to debug Qdrant search issue
"""

from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Connect to Qdrant Cloud
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

print(f"Connecting to Qdrant: {QDRANT_ENDPOINT}")
client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

# Load model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test search on product_vectors
print("\nTesting search on product_vectors...")
try:
    query_text = "used iphone 128gb"
    query_vector = model.encode(query_text, convert_to_tensor=False).tolist()

    filter_conditions = [
        FieldCondition(key="intent", match=MatchValue(value="product"))
    ]
    query_filter = Filter(must=filter_conditions)

    search_result = client.search(
        collection_name="product_vectors",
        query_vector=query_vector,
        query_filter=query_filter,
        limit=10
    )

    print(f"✓ Search successful! Found {len(search_result)} results")
    for i, result in enumerate(search_result[:3]):
        print(f"  {i+1}. listing_id={result.payload.get('listing_id')}, score={result.score}")

except Exception as e:
    print(f"✗ Search failed: {e}")
    import traceback
    traceback.print_exc()
