import os
from sentence_transformers import SentenceTransformer

# Use a smaller model for cloud deployment to avoid OOM and timeouts
# This script is run during the build phase
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

print(f"Downloading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)
print("✓ Model downloaded successfully.")
