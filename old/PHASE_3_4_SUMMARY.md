# PHASE 3.4 COMPLETION SUMMARY

**Date**: 2026-01-12
**Phase**: 3.4 — Retrieval (Candidate Selection)
**Authority**: VRIDDHI Architecture Document
**Agent**: Claude (Implementation Engine)

---

## IMPLEMENTATION STATUS

✅ **PHASE 3.4 COMPLETE**

**Deliverable**: `retrieval_service.py` (415 lines)
**Purpose**: Candidate selection via SQL + vector search

---

## WHAT WAS BUILT

### 1. SQL Filtering (Supabase)

**Purpose**: Pre-filter candidates using structured constraints before vector search.

#### Product/Service SQL Filter

**Function**: `sql_filter_product_service(client, query_listing, limit)`

**Filters Applied**:
1. **Intent match**: Query intent = Candidate intent
2. **Domain intersection**: `query.domain ∩ candidate.domain ≠ ∅`

**Logic**:
```python
query_domains = ["electronics", "computers"]
candidate_domains = ["electronics", "gadgets"]

# Check intersection
if {"electronics", "computers"} & {"electronics", "gadgets"}:
    # "electronics" in both → PASS
```

**Returns**: List of listing_ids that pass SQL filters

**Note**: Current implementation fetches data and filters in Python. For production, should use PostgreSQL array intersection operator (`&&`) via RPC function.

#### Mutual SQL Filter

**Function**: `sql_filter_mutual(client, query_listing, limit)`

**Filters Applied**:
1. **Category intersection**: `query.category ∩ candidate.category ≠ ∅`

**Logic**: Same as domain intersection (uses category field instead)

**Returns**: List of listing_ids that pass SQL filters

### 2. Qdrant Vector Search

**Purpose**: Semantic similarity search with payload filtering.

#### Product/Service Vector Search

**Function**: `qdrant_search_product_service(client, model, query_listing, sql_filtered_ids, limit)`

**Steps**:
1. Build query embedding text (structured format from Phase 3.3)
2. Generate 1024D query vector
3. Search Qdrant collection (product_vectors or service_vectors)
4. Apply payload filters:
   - `intent` = query intent
   - `domain` ∈ query domains (using MatchAny)
5. Post-filter by SQL-filtered IDs (if provided)
6. Return top-k listing_ids

**Payload Filter Example**:
```python
Filter(
    must=[
        FieldCondition(key="intent", match=MatchValue(value="product")),
        FieldCondition(key="domain", match=MatchAny(any=["electronics", "computers"]))
    ]
)
```

**Returns**: List of listing_ids ordered by cosine similarity

#### Mutual Vector Search

**Function**: `qdrant_search_mutual(client, model, query_listing, sql_filtered_ids, limit)`

**Steps**:
1. Build query embedding text (natural language format from Phase 3.3)
2. Generate 1024D query vector
3. Search Qdrant collection (mutual_vectors)
4. Apply payload filters:
   - `intent` = "mutual"
   - `category` ∈ query categories (using MatchAny)
5. Post-filter by SQL-filtered IDs (if provided)
6. Return top-k listing_ids

**Semantic Search**: Uses natural language embedding, NOT keyword matching

**Returns**: List of listing_ids ordered by semantic similarity

### 3. Orchestration

**Function**: `retrieve_candidates(clients, query_listing, limit, use_sql_filter, verbose)`

**Pipeline**:
```
Query Listing
     ↓
[1] SQL Filter (Supabase)
     ↓
SQL-filtered listing_ids
     ↓
[2] Qdrant Vector Search (with payload filters)
     ↓
Candidate listing_ids (top-k by similarity)
```

**Parameters**:
- `query_listing`: Normalized query listing object
- `limit`: Number of candidates to return (default 100)
- `use_sql_filter`: Whether to apply SQL filtering first (default True)
- `verbose`: Print progress messages (default True)

**Returns**: List of candidate listing_ids (up to limit)

**NO scoring returned**: Only listing_ids (no similarity scores, no ranks)

---

## WHAT WAS VERIFIED

### 1. NO Ranking Logic

✓ Returns listing_ids only
✓ NO similarity scores exposed
✓ NO candidate ranking (order is by Qdrant internal similarity, but not exposed)

### 2. NO Boolean Matching

✓ NO constraint evaluation
✓ NO Phase 2.8 matching logic
✓ Candidate selection only

### 3. NO Scoring

✓ NO compatibility scores
✓ NO match quality metrics
✓ Pure candidate retrieval

### 4. NO Inference

✓ Only uses query listing attributes
✓ NO attribute expansion
✓ NO default values

### 5. SQL-First Architecture

✓ SQL filtering before vector search (if enabled)
✓ Qdrant payload filtering
✓ Post-filtering by SQL results

### 6. Intent-Based Routing

✓ Product → product_listings → product_vectors
✓ Service → service_listings → service_vectors
✓ Mutual → mutual_listings → mutual_vectors

✓ Product/Service filter by domain
✓ Mutual filters by category

### 7. Vector Search

✓ Uses same embedding model as ingestion (BAAI/bge-large-en-v1.5)
✓ Generates query vector from query listing
✓ Searches with cosine similarity

---

## WHAT WAS NOT DONE

### ❌ Boolean Matching

**NOT implemented**: No Phase 2.8 matching logic
**Reason**: Out of scope (separate phase, already complete)

### ❌ Ranking/Scoring

**NOT implemented**: No similarity scores, no reranking
**Reason**: Out of scope (future phase)

### ❌ RRF (Reciprocal Rank Fusion)

**NOT implemented**: No multi-retriever fusion
**Reason**: Out of scope (future phase)

### ❌ BM25 / Sparse Vectors

**NOT implemented**: Only dense vector search
**Reason**: Out of scope (future phase)

### ❌ ColBERT Multi-Vector

**NOT implemented**: Only single dense vector
**Reason**: Out of scope (future phase)

### ❌ Cross-Encoder Reranking

**NOT implemented**: No pairwise reranking
**Reason**: Out of scope (future phase)

### ❌ Filtering Optimization

**NOT implemented**: SQL filters done in Python (not optimized Postgres queries)
**Reason**: Simplicity for Phase 3.4; can optimize with RPC functions later

### ❌ Batch Retrieval

**NOT implemented**: Single query only
**Reason**: Out of scope

### ❌ Caching

**NOT implemented**: No result caching
**Reason**: Out of scope

---

## ARCHITECTURAL DECISIONS

### 1. Why SQL Filter First?

**Decision**: Apply SQL filters before Qdrant search (optional)

**Rationale**:
- Structured filters (domain/category intersection) are cheap in SQL
- Reduces Qdrant search space
- Enables hard constraints before semantic similarity

**Alternative**: Qdrant-only filtering
- **Rejected**: SQL better for structured data queries

**Note**: SQL filtering is optional (can disable with `use_sql_filter=False`)

### 2. Why Post-Filter SQL Results in Qdrant?

**Decision**: Pass SQL-filtered IDs to Qdrant search, then post-filter

**Rationale**:
- Qdrant doesn't natively support "ID IN (list)" filter for large lists
- Post-filtering is simple and correct
- For small result sets, performance is acceptable

**Optimization for future**: If SQL-filtered list is large, could batch Qdrant searches or use alternative approach

### 3. Why MatchAny for Domain/Category?

**Decision**: Use Qdrant's `MatchAny` for array intersection checks

**Rationale**:
- `MatchAny(any=[a, b, c])` returns points where field contains ANY of [a, b, c]
- Implements intersection check: query ∩ candidate ≠ ∅
- Native Qdrant filter (efficient)

**Example**:
```python
FieldCondition(key="domain", match=MatchAny(any=["electronics", "computers"]))
# Returns candidates with domain containing "electronics" OR "computers"
```

### 4. Why No Similarity Scores Returned?

**Decision**: Return only listing_ids, not scores

**Rationale**:
- This phase is candidate selection ONLY
- Scoring/ranking is separate phase (3.5+)
- Simplifies interface
- Boolean matching (Phase 2.8) is the true compatibility check

**Alternative**: Return (listing_id, score) tuples
- **Rejected**: Violates separation of concerns (retrieval ≠ ranking)

### 5. Why Same Embedding Model as Ingestion?

**Decision**: Use BAAI/bge-large-en-v1.5 for query encoding

**Rationale**:
- Must match ingestion model for correct similarity
- Same embedding space required for cosine similarity
- Consistency across pipeline

**Critical**: If ingestion model changes, retrieval model MUST change too.

### 6. Why No Default Filters?

**Decision**: Only apply filters explicitly in query listing

**Rationale**:
- NO inference (strict rule)
- NO default values (strict rule)
- Only user-specified constraints

**Example**:
```python
query_listing = {"intent": "product", "domain": []}
# Empty domain → NO domain filter applied (returns all domains)
```

---

## USAGE

### Prerequisites

```bash
# Install dependencies (if not already installed)
pip install supabase qdrant-client sentence-transformers

# Set environment variables
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-supabase-key"

# Ensure services running
docker run -p 6333:6333 qdrant/qdrant
```

### Initialize Clients

```python
from retrieval_service import RetrievalClients

clients = RetrievalClients()
clients.initialize()
```

Output:
```
✓ Connected to Supabase: https://your-project.supabase.co
✓ Connected to Qdrant: localhost:6333
Loading embedding model: BAAI/bge-large-en-v1.5...
✓ Loaded embedding model: BAAI/bge-large-en-v1.5
```

### Retrieve Candidates

```python
from retrieval_service import retrieve_candidates

# Query listing (normalized)
query_listing = {
    "intent": "product",
    "subintent": "buyer",
    "domain": ["electronics", "computers"],
    "items": [{"type": "laptop", "categorical": {"brand": "Apple"}}],
    # ... other fields
}

# Retrieve up to 100 candidates
candidate_ids = retrieve_candidates(
    clients,
    query_listing,
    limit=100,
    use_sql_filter=True,
    verbose=True
)

print(f"Found {len(candidate_ids)} candidates")
print(f"Candidate IDs: {candidate_ids[:5]}...")  # Show first 5
```

Output:
```
Retrieving candidates for intent: product
  [1/2] SQL filtering...
        ✓ SQL filtered to 452 candidates
  [2/2] Qdrant vector search...
        ✓ Retrieved 100 candidates
✓ Retrieval complete

Found 100 candidates
Candidate IDs: ['uuid-1', 'uuid-2', 'uuid-3', 'uuid-4', 'uuid-5']...
```

### Without SQL Filtering

```python
# Skip SQL filter, use only Qdrant
candidate_ids = retrieve_candidates(
    clients,
    query_listing,
    limit=100,
    use_sql_filter=False
)
```

---

## INTEGRATION WITH MATCHING ENGINE

### Complete Pipeline (Phase 3.4 → Phase 2.8)

```python
from retrieval_service import RetrievalClients, retrieve_candidates
from mutual_matcher import mutual_listing_matches

# Step 1: Initialize retrieval clients
clients = RetrievalClients()
clients.initialize()

# Step 2: Retrieve candidates
candidate_ids = retrieve_candidates(clients, query_listing, limit=100)

# Step 3: Fetch full listings from Supabase
table_name = "product_listings"  # Based on intent
candidates = []
for listing_id in candidate_ids:
    response = clients.supabase.table(table_name).select("data").eq("id", listing_id).execute()
    if response.data:
        candidates.append(response.data[0]["data"])

# Step 4: Boolean matching (Phase 2.8)
valid_matches = []
for candidate in candidates:
    if mutual_listing_matches(query_listing, candidate):
        valid_matches.append(candidate)

print(f"Candidates: {len(candidate_ids)}")
print(f"Valid matches: {len(valid_matches)}")
```

**Flow**:
```
Query Listing
     ↓
Phase 3.4: Retrieve candidates (SQL + vector)
     ↓
Candidate listing_ids
     ↓
Fetch full listings (Supabase)
     ↓
Phase 2.8: Boolean matching
     ↓
Valid matches (compatibility-checked)
     ↓
Phase 3.5+: Rank by similarity (future)
     ↓
Final results
```

---

## SQL OPTIMIZATION (Future Work)

### Current Implementation

**Problem**: SQL filtering done in Python (fetch all → filter in memory)

```python
# Current: Inefficient
response = client.table(table_name).select("id, data").execute()
# Filter in Python
filtered = [row for row in response.data if has_intersection(...)]
```

### Recommended Optimization

**Solution**: Create PostgreSQL RPC function for array intersection

```sql
CREATE OR REPLACE FUNCTION filter_by_domain_intersection(
    query_domains TEXT[],
    result_limit INT
)
RETURNS TABLE(id UUID) AS $$
BEGIN
    RETURN QUERY
    SELECT p.id
    FROM product_listings p
    WHERE p.data->'domain' ?| query_domains
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;
```

**Usage**:
```python
response = client.rpc("filter_by_domain_intersection", {
    "query_domains": ["electronics", "computers"],
    "result_limit": 1000
}).execute()
```

**Benefits**:
- Runs filter in database (faster)
- Reduces data transfer (only IDs returned)
- Uses PostgreSQL indexes

**Status**: NOT implemented in Phase 3.4 (marked as TODO)

---

## GUARANTEES

After Phase 3.4, the following are guaranteed:

✓ Candidate retrieval via SQL + vector search
✓ Product/Service filter by intent + domain intersection
✓ Mutual filters by category intersection
✓ Vector search uses same embedding as ingestion (1024D)
✓ Qdrant payload filtering (intent, domain/category)
✓ Returns listing_ids only (NO scores, NO ranks)
✓ NO boolean matching (separate phase)
✓ NO inference (only explicit query attributes)
✓ Intent-based routing (product/service/mutual)
✓ Optional SQL filtering (can disable)

---

## FAILURE MODES & ERROR HANDLING

### Supabase Connection Failure

**Error**: Environment variables not set

**Resolution**: Set `SUPABASE_URL` and `SUPABASE_KEY`

### Qdrant Connection Failure

**Error**: Connection refused

**Resolution**: Start Qdrant (`docker run -p 6333:6333 qdrant/qdrant`)

### Embedding Model Failure

**Error**: Model not found or download fails

**Resolution**: Check internet connection, model will auto-download on first run

### No Candidates Found

**Error**: Empty result list

**Cause**: Query too restrictive or no matching candidates in database

**Resolution**: Relax filters or verify data ingestion

### Intent Mismatch

**Error**: `ValueError: Invalid intent`

**Cause**: Query listing has invalid or missing intent field

**Resolution**: Ensure query listing is normalized with valid intent

---

## NEXT STEPS (NOT IMPLEMENTED)

### Phase 3.5: Ranking + RRF

**Task**: Rank candidates by multiple signals

**Requirements**:
- Multiple retrieval methods (dense, sparse, BM25)
- Reciprocal Rank Fusion (RRF)
- Score aggregation

### Phase 3.6: BM25 + Sparse Vectors

**Task**: Add keyword-based retrieval

**Requirements**:
- BM25 tokenization
- Sparse vector generation
- Hybrid search (dense + sparse)

### Phase 3.7: ColBERT Multi-Vector

**Task**: Late-interaction retrieval

**Requirements**:
- ColBERT token embeddings
- MaxSim scoring
- Multi-vector storage

### Phase 3.8: Cross-Encoder Reranking

**Task**: Final pairwise reranking

**Requirements**:
- Cross-encoder model
- Pairwise scoring
- Top-k reranking

---

## FILES CREATED

### retrieval_service.py

**Size**: 415 lines

**Classes**:
- `RetrievalClients`: Container for Supabase, Qdrant, embedding model

**Functions**:
- `sql_filter_product_service()`: SQL filtering for product/service
- `sql_filter_mutual()`: SQL filtering for mutual
- `qdrant_search_product_service()`: Vector search for product/service
- `qdrant_search_mutual()`: Vector search for mutual
- `retrieve_candidates()`: Complete retrieval pipeline

**Key feature**: Returns listing_ids only (NO scores)

---

## COMPLETION CHECKLIST

- [x] SQL filtering (product/service: domain intersection)
- [x] SQL filtering (mutual: category intersection)
- [x] Qdrant vector search (product/service)
- [x] Qdrant vector search (mutual)
- [x] Qdrant payload filtering (intent, domain/category)
- [x] Query embedding generation (same model as ingestion)
- [x] Post-filtering by SQL results
- [x] Intent-based routing
- [x] Returns listing_ids only (NO scores)
- [x] NO boolean matching (out of scope)
- [x] NO ranking logic (out of scope)
- [x] NO inference (strict rule enforced)
- [x] Documentation complete

---

## PHASE 3.4 COMPLETE

**Candidate retrieval ready for integration with boolean matching (Phase 2.8).**

**Next step: Integrate with matching engine for full query pipeline.**

---

**End of Phase 3.4 Summary**
