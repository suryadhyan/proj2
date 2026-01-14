# PHASE 3.3: INGESTION PIPELINE

## Quick Start

### 1. Install Dependencies

```bash
pip install supabase sentence-transformers qdrant-client torch
```

Note: `torch` is required by sentence-transformers and will auto-install.

### 2. Set Environment Variables

```bash
# Supabase credentials (REQUIRED)
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-supabase-anon-key"

# Qdrant connection (optional, defaults to localhost:6333)
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

### 3. Ensure Services Running

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Supabase should already be set up (Phase 3.1)
```

### 4. Initialize Clients

```python
from ingestion_pipeline import IngestionClients

clients = IngestionClients()
clients.initialize()
```

**First run**: Embedding model will download (~1.3GB). This is one-time only.

### 5. Ingest a Listing

```python
from ingestion_pipeline import ingest_listing

# Your normalized listing object
listing = {
    "intent": "product",
    "subintent": "buyer",
    "domain": ["electronics"],
    "items": [{
        "type": "laptop",
        "categorical": {"brand": "Apple"},
        "min": {"ram": 16},
        "max": {},
        "range": {}
    }],
    "other": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "location": {"categorical": {}, "min": {}, "max": {}, "range": {}}
}

listing_id, embedding = ingest_listing(clients, listing)
print(f"✓ Ingested: {listing_id}")
```

---

## What This Does

### Step-by-Step Pipeline

1. **Insert to Supabase**
   - Stores listing in appropriate table (product_listings/service_listings/mutual_listings)
   - Entire listing saved as JSONB in `data` column
   - Returns listing_id (UUID)

2. **Build Embedding Text**
   - Product/Service: Structured keyword text
   - Mutual: Natural language semantic text
   - NO hard-coded attributes

3. **Generate Embedding**
   - Uses BAAI/bge-large-en-v1.5 model
   - Produces 1024D vector
   - Verifies dimension correctness

4. **Insert to Qdrant**
   - Stores embedding in appropriate collection (product_vectors/service_vectors/mutual_vectors)
   - Includes payload: listing_id, intent, domain/category, timestamp
   - Upsert behavior (updates if exists)

---

## Embedding Text Examples

### Product/Service (Structured)

Input:
```python
{
    "intent": "product",
    "subintent": "buyer",
    "domain": ["electronics", "computers"],
    "items": [{"type": "laptop", "categorical": {"brand": "Apple"}, "min": {"ram": 16}}]
}
```

Embedding text:
```
product buyer electronics computers laptop brand Apple ram
```

### Mutual (Natural Language)

Input:
```python
{
    "intent": "mutual",
    "category": ["books", "electronics"],
    "items": [{"type": "book", "categorical": {"genre": "fiction"}}],
    "other": {"categorical": {"type": "electronics"}}
}
```

Embedding text:
```
mutual exchange in categories: books and electronics offering book genre: fiction wanting type: electronics
```

---

## Preview Embedding Text (Debugging)

```python
from embedding_builder import preview_embedding_text

preview_embedding_text(listing)
```

Output:
```
Intent: product
Embedding text (45 chars):
  product buyer electronics laptop brand Apple ram
```

---

## Batch Ingestion

```python
from ingestion_pipeline import ingest_batch

listings = [listing1, listing2, listing3, ...]

listing_ids = ingest_batch(clients, listings, verbose=True)

print(f"Successfully ingested {len(listing_ids)} listings")
```

---

## Architecture

### Data Flow

```
Normalized Listing
        ↓
┌───────────────────┐
│  Supabase Insert  │ → product_listings/service_listings/mutual_listings
└───────────────────┘
        ↓
┌───────────────────┐
│ Build Embed Text  │ → "product buyer electronics laptop..."
└───────────────────┘
        ↓
┌───────────────────┐
│ Generate Embedding│ → [0.123, 0.456, ..., 0.789] (1024D)
└───────────────────┘
        ↓
┌───────────────────┐
│  Qdrant Insert    │ → product_vectors/service_vectors/mutual_vectors
└───────────────────┘
```

### Storage Mapping

```
Supabase Table         Qdrant Collection
─────────────────────  ─────────────────────
product_listings   →   product_vectors
service_listings   →   service_vectors
mutual_listings    →   mutual_vectors

Linking: listing.id = point.payload.listing_id
```

---

## Troubleshooting

### Error: "SUPABASE_URL and SUPABASE_KEY environment variables required"

**Solution**:
```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-key-here"
```

### Error: Connection refused (Qdrant)

**Solution**: Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Error: "Collection not found"

**Solution**: Run Phase 3.2 setup:
```bash
python qdrant_setup.py
```

### Error: Model download slow/fails

**Cause**: First-time model download (~1.3GB)

**Solution**:
- Wait for download to complete (one-time only)
- Check internet connection
- Model cached in `~/.cache/huggingface/`

### Error: "Embedding dimension mismatch"

**Cause**: Wrong model loaded

**Solution**: Check model name in `ingestion_pipeline.py`:
```python
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Must be 1024D
```

---

## What Was Built

✓ Supabase insertion (intent-based table routing)
✓ Embedding text construction (product/service vs mutual)
✓ 1024D embedding generation (BAAI/bge-large-en-v1.5)
✓ Qdrant insertion (with payload metadata)
✓ Batch ingestion support
✓ Error handling with clear messages

---

## What Was NOT Built (Out of Scope)

❌ Retrieval/search logic (Phase 3.5)
❌ Ranking logic (Phase 3.5)
❌ Boolean matching (Phase 2.8, already done)
❌ BM25/sparse vectors (future phase)
❌ ColBERT multi-vector (future phase)
❌ Cross-encoder reranking (future phase)

---

## Files

- `embedding_builder.py` - Embedding text construction
- `ingestion_pipeline.py` - Main ingestion pipeline
- `PHASE_3_3_SUMMARY.md` - Detailed documentation
- `PHASE_3_3_README.md` - This file

---

## Next Phase

Phase 3.3 is complete. Collections now contain listings.

**Do NOT proceed yet.** Wait for explicit approval.

Next phases (NOT YET IMPLEMENTED):
- Phase 3.4: Sparse vectors + BM25
- Phase 3.5: Retrieval + ranking
