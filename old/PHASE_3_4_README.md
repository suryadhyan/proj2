# PHASE 3.4: RETRIEVAL SERVICE

## Quick Start

### 1. Install Dependencies

```bash
pip install supabase qdrant-client sentence-transformers
```

### 2. Set Environment Variables

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-supabase-key"
export QDRANT_HOST="localhost"  # Optional, defaults to localhost
export QDRANT_PORT="6333"       # Optional, defaults to 6333
```

### 3. Ensure Services Running

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Supabase (already set up)
```

### 4. Initialize Clients

```python
from retrieval_service import RetrievalClients

clients = RetrievalClients()
clients.initialize()
```

### 5. Retrieve Candidates

```python
from retrieval_service import retrieve_candidates

# Normalized query listing
query_listing = {
    "intent": "product",
    "subintent": "buyer",
    "domain": ["electronics"],
    "items": [{"type": "laptop", "categorical": {"brand": "Apple"}}],
    # ... other fields
}

# Retrieve candidates
candidate_ids = retrieve_candidates(
    clients,
    query_listing,
    limit=100,
    use_sql_filter=True
)

print(f"Found {len(candidate_ids)} candidates")
```

---

## What This Does

### Retrieval Pipeline

```
Query Listing
     ↓
[Step 1] SQL Filter (Supabase)
     - Product/Service: domain intersection
     - Mutual: category intersection
     ↓
SQL-filtered listing_ids
     ↓
[Step 2] Qdrant Vector Search
     - Generate query embedding (1024D)
     - Search with payload filters (intent, domain/category)
     - Post-filter by SQL results
     ↓
Candidate listing_ids (top-k by similarity)
```

**Output**: List of listing_ids only (NO scores, NO ranks)

---

## Product/Service Retrieval

### SQL Filters

- Intent match: `query.intent = candidate.intent`
- Domain intersection: `query.domain ∩ candidate.domain ≠ ∅`

### Qdrant Filters

- Payload: `intent`, `domain` (MatchAny)
- Vector similarity: Cosine similarity (1024D embeddings)

### Example

```python
query_listing = {
    "intent": "product",
    "subintent": "buyer",
    "domain": ["electronics", "computers"],
    "items": [{"type": "laptop", "categorical": {"brand": "Apple"}}]
}

candidate_ids = retrieve_candidates(clients, query_listing, limit=50)
# Returns up to 50 candidate listing_ids
```

---

## Mutual Retrieval

### SQL Filters

- Category intersection: `query.category ∩ candidate.category ≠ ∅`

### Qdrant Filters

- Payload: `intent="mutual"`, `category` (MatchAny)
- Vector similarity: Semantic similarity (natural language embeddings)

### Example

```python
query_listing = {
    "intent": "mutual",
    "category": ["books", "electronics"],
    "items": [{"type": "book", "categorical": {"genre": "fiction"}}],
    "other": {"categorical": {"type": "electronics"}}
}

candidate_ids = retrieve_candidates(clients, query_listing, limit=50)
# Returns up to 50 candidate listing_ids (semantically similar)
```

---

## Parameters

### `retrieve_candidates()`

- **clients**: RetrievalClients (initialized)
- **query_listing**: Normalized query listing object
- **limit**: Number of candidates (default 100)
- **use_sql_filter**: Apply SQL filtering first (default True)
- **verbose**: Print progress messages (default True)

**Returns**: `List[str]` - Candidate listing_ids

---

## Integration with Boolean Matching

### Complete Query Pipeline

```python
from retrieval_service import RetrievalClients, retrieve_candidates
from mutual_matcher import mutual_listing_matches

# Step 1: Retrieve candidates
clients = RetrievalClients()
clients.initialize()

candidate_ids = retrieve_candidates(clients, query_listing, limit=100)

# Step 2: Fetch full listings
table_name = "product_listings"  # Based on intent
candidates = []
for listing_id in candidate_ids:
    response = clients.supabase.table(table_name).select("data").eq("id", listing_id).execute()
    if response.data:
        candidates.append(response.data[0]["data"])

# Step 3: Boolean matching (Phase 2.8)
valid_matches = []
for candidate in candidates:
    if mutual_listing_matches(query_listing, candidate):
        valid_matches.append(candidate)

print(f"Candidates: {len(candidate_ids)}")
print(f"Valid matches: {len(valid_matches)}")
```

**Flow**:
- Phase 3.4: Candidate selection (SQL + vector)
- Phase 2.8: Boolean matching (compatibility check)
- Future: Ranking (similarity-based ordering)

---

## Troubleshooting

### Error: "Environment variables required"

**Solution**:
```bash
export SUPABASE_URL="..."
export SUPABASE_KEY="..."
```

### Error: Connection refused (Qdrant)

**Solution**: Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Error: No candidates found

**Cause**: Query too restrictive or no data ingested

**Solutions**:
- Check if listings are ingested (Phase 3.3)
- Relax domain/category filters
- Disable SQL filter: `use_sql_filter=False`

### Error: Model download slow

**Cause**: First-time model download (~1.3GB)

**Solution**: Wait for download (one-time only)

---

## What This Does NOT Do

❌ Boolean matching (use Phase 2.8: `mutual_listing_matches()`)
❌ Ranking/scoring (future phase)
❌ RRF fusion (future phase)
❌ BM25/sparse vectors (future phase)
❌ Cross-encoder reranking (future phase)

**This phase**: Candidate selection ONLY

---

## Architecture Notes

### SQL Filtering (Current Implementation)

**Note**: SQL filtering currently fetches data and filters in Python. For production, should optimize with PostgreSQL RPC function.

**Future optimization**:
```sql
CREATE FUNCTION filter_by_domain_intersection(query_domains TEXT[])
RETURNS TABLE(id UUID) AS $$
    SELECT id FROM product_listings
    WHERE data->'domain' ?| query_domains;
$$ LANGUAGE sql;
```

### Embedding Model

**Model**: BAAI/bge-large-en-v1.5 (1024D)

**Critical**: Must match ingestion model (Phase 3.3)

---

## Files

- `retrieval_service.py` - Main retrieval service
- `PHASE_3_4_SUMMARY.md` - Detailed documentation
- `PHASE_3_4_README.md` - This file

---

## Next Steps

Phase 3.4 is complete. Candidate retrieval is functional.

**Integration**: Combine with Phase 2.8 (boolean matching) for complete query pipeline.

**Future phases** (NOT YET IMPLEMENTED):
- Phase 3.5: Ranking + RRF
- Phase 3.6: BM25 + Sparse vectors
- Phase 3.7: ColBERT multi-vector
- Phase 3.8: Cross-encoder reranking
