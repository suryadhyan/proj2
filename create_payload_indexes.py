"""
Create payload indexes for Qdrant collections to enable filtering
"""

from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# Connect to Qdrant Cloud
QDRANT_ENDPOINT = os.environ.get("QDRANT_ENDPOINT")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

print(f"Connecting to Qdrant: {QDRANT_ENDPOINT}")
client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)

# Create indexes for product_vectors collection
print("\n1. Creating indexes for product_vectors...")
try:
    client.create_payload_index(
        collection_name="product_vectors",
        field_name="intent",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'intent' field")
except Exception as e:
    print(f"   ⚠️ Intent index: {e}")

try:
    client.create_payload_index(
        collection_name="product_vectors",
        field_name="domain",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'domain' field")
except Exception as e:
    print(f"   ⚠️ Domain index: {e}")

# Create indexes for service_vectors collection
print("\n2. Creating indexes for service_vectors...")
try:
    client.create_payload_index(
        collection_name="service_vectors",
        field_name="intent",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'intent' field")
except Exception as e:
    print(f"   ⚠️ Intent index: {e}")

try:
    client.create_payload_index(
        collection_name="service_vectors",
        field_name="domain",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'domain' field")
except Exception as e:
    print(f"   ⚠️ Domain index: {e}")

# Create indexes for mutual_vectors collection
print("\n3. Creating indexes for mutual_vectors...")
try:
    client.create_payload_index(
        collection_name="mutual_vectors",
        field_name="intent",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'intent' field")
except Exception as e:
    print(f"   ⚠️ Intent index: {e}")

try:
    client.create_payload_index(
        collection_name="mutual_vectors",
        field_name="category",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print("   ✓ Created index for 'category' field")
except Exception as e:
    print(f"   ⚠️ Category index: {e}")

print("\n✅ All payload indexes created!")
