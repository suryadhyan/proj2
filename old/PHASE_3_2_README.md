# PHASE 3.2: QDRANT SETUP

## Quick Start

### 1. Ensure Qdrant is Running

```bash
# Option A: Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Option B: Using Docker Compose (if configured)
docker-compose up qdrant
```

Verify Qdrant is accessible:
```bash
curl http://localhost:6333/collections
```

Expected output: `{"result":{"collections":[]}}`

### 2. Install Dependencies

```bash
pip install qdrant-client
```

### 3. Preview Setup (Dry Run)

```bash
python qdrant_setup.py --dry-run
```

This shows what will be done WITHOUT making changes.

### 4. Execute Setup

```bash
python qdrant_setup.py
```

Expected output:
```
======================================================================
PHASE 3.2: QDRANT RESET & COLLECTION CREATION
======================================================================

Connecting to Qdrant at localhost:6333...
✓ Connected to Qdrant

======================================================================
STEP 1: COLLECTION RESET
======================================================================

Deleting existing collection: product_vectors
✓ Deleted: product_vectors
...

======================================================================
STEP 2: COLLECTION CREATION
======================================================================

Creating collection: product_vectors
✓ Created: product_vectors
  - Vector size: 1024D
  - Distance metric: COSINE
...

======================================================================
STEP 3: PAYLOAD INDEX CREATION
======================================================================

Creating payload indexes for: product_vectors
  ✓ Indexed: listing_id (keyword)
  ✓ Indexed: intent (keyword)
  ✓ Indexed: domain (keyword)
  ✓ Indexed: created_at (integer)
...

======================================================================
STEP 4: VERIFICATION
======================================================================

Collection: product_vectors
  - Vectors count: 0
  - Points count: 0
  - Status: green
  - Vector size: 1024D
  - Distance: COSINE
  ✓ Configuration valid
...

======================================================================
✅ ALL COLLECTIONS VERIFIED SUCCESSFULLY
======================================================================

======================================================================
PHASE 3.2 SETUP COMPLETE
======================================================================
```

### 5. Verify Setup

```bash
python verify_qdrant.py
```

Expected output:
```
======================================================================
QDRANT SETUP VERIFICATION
======================================================================

✓ Connected to Qdrant (localhost:6333)

Collections found:
  - product_vectors
  - service_vectors
  - mutual_vectors

Collection: product_vectors
  Status: green
  Vector size: 1024D
  Distance: COSINE
  Points: 0
  Vectors: 0
  ✓ Configuration OK
...

======================================================================
✅ ALL CHECKS PASSED
======================================================================
```

---

## What Was Created

### Collections

1. **product_vectors**
   - Purpose: Store embeddings for product listings
   - Vector size: 1024D
   - Distance: Cosine similarity
   - Indexed fields: listing_id, intent, domain, created_at

2. **service_vectors**
   - Purpose: Store embeddings for service listings
   - Vector size: 1024D
   - Distance: Cosine similarity
   - Indexed fields: listing_id, intent, domain, created_at

3. **mutual_vectors**
   - Purpose: Store embeddings for mutual exchange listings
   - Vector size: 1024D
   - Distance: Cosine similarity
   - Indexed fields: listing_id, intent, category, created_at

### Payload Schema

Each vector point will have this payload structure:

**Product/Service**:
```json
{
  "listing_id": "uuid-string",
  "intent": "product" or "service",
  "domain": ["electronics", "gadgets"],
  "created_at": 1704067200
}
```

**Mutual**:
```json
{
  "listing_id": "uuid-string",
  "intent": "mutual",
  "category": ["books", "electronics"],
  "created_at": 1704067200
}
```

---

## Troubleshooting

### Error: "Failed to connect to Qdrant"

**Cause**: Qdrant is not running

**Solution**:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Error: "Collection already exists"

**Cause**: Collections were created but not deleted

**Solution**: The script automatically handles this via `reset_collections()`. Just re-run:
```bash
python qdrant_setup.py
```

### Error: "Vector size mismatch"

**Cause**: Collections were created with different configuration

**Solution**: Delete collections manually and re-run setup:
```bash
# Via Qdrant API
curl -X DELETE http://localhost:6333/collections/product_vectors
curl -X DELETE http://localhost:6333/collections/service_vectors
curl -X DELETE http://localhost:6333/collections/mutual_vectors

# Then re-run setup
python qdrant_setup.py
```

---

## Next Steps

Phase 3.2 is complete. Collections are ready for ingestion.

**Do NOT proceed to next phases yet**. Wait for explicit approval.

Next phases (NOT YET IMPLEMENTED):
- Phase 3.3: Ingestion Pipeline
- Phase 3.4: Embedding Generation
- Phase 3.5: Retrieval + Ranking

---

## Files

- `qdrant_setup.py` - Main setup script
- `verify_qdrant.py` - Verification script
- `PHASE_3_2_SUMMARY.md` - Detailed documentation
- `PHASE_3_2_README.md` - This file
