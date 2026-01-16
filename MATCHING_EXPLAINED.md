# 🔍 Matching System - Complete Explanation

**Date:** 2026-01-15
**Question:** Are we using semantic matching with embeddings or just hard filters?

**Answer:** **BOTH!** The system uses a **hybrid approach** combining vector search, semantic embeddings, and hard constraints.

---

## 📊 The Complete Matching Pipeline

```
Query Listing
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 1: CANDIDATE RETRIEVAL (retrieval_service.py) │
│ Uses: Qdrant Vector Search + SQL Filters   │
├─────────────────────────────────────────────┤
│ Step 1: SQL Filter (Supabase)              │
│   - Intent equality                         │
│   - Domain/Category intersection            │
│                                             │
│ Step 2: Vector Search (Qdrant)             │
│   - Generate query embedding ✅ SEMANTIC    │
│   - Search by cosine similarity ✅ SEMANTIC │
│   - Payload filters (intent, domain)        │
│   - Returns Top-K candidates (100)          │
└─────────────────────────────────────────────┘
    ↓ Top 100 candidates
┌─────────────────────────────────────────────┐
│ PHASE 2: BOOLEAN MATCHING (listing_matcher_v2.py) │
│ Uses: Hard constraints + Semantic implies  │
├─────────────────────────────────────────────┤
│ For Each Candidate:                        │
│                                             │
│ 1. Intent Gate (M-01 to M-04)              │
│    ❌ HARD: Exact string match              │
│                                             │
│ 2. Domain/Category Gate (M-05, M-06)       │
│    ❌ HARD: Set intersection                │
│                                             │
│ 3. Items Matching (M-07 to M-12)           │
│    • Type: ❌ HARD (exact match)            │
│    • Categorical: ✅ SEMANTIC (implies_fn)  │
│    • Numeric (min/max/range): ❌ HARD       │
│                                             │
│ 4. Other→Self Matching (M-13 to M-17)      │
│    • Categorical: ✅ SEMANTIC (implies_fn)  │
│    • Numeric (min/max/range): ❌ HARD       │
│                                             │
│ 5. Location Matching (M-23 to M-28)        │
│    ❌ HARD: Name equality or mode logic     │
│                                             │
│ Result: TRUE/FALSE for each candidate      │
└─────────────────────────────────────────────┘
    ↓
Final Matches (all candidates that returned TRUE)
```

---

## ✅ Where Embeddings ARE Used

### 1. **Candidate Retrieval (Vector Search)**

**File:** `retrieval_service.py`

**Lines 287-292:**
```python
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,  # ✅ Embedding vector
    query_filter=query_filter,
    limit=limit
)
```

**How it works:**
1. Query listing converted to text using `build_embedding_text()` (embedding_builder.py)
2. Text encoded to 384D or 1024D vector using sentence-transformers
3. Qdrant performs **cosine similarity search** against stored vectors
4. Returns Top-K most similar candidates (default: 100)

**Storage:**
- **Qdrant collections:** `product_vectors`, `service_vectors`, `mutual_vectors`
- **Embedding dimension:** 384D (all-MiniLM-L6-v2) or 1024D (configurable)
- **Stored in:** Each listing has vector stored with payload (listing_id, intent, domain)

---

### 2. **Categorical Matching (Semantic Implies)**

**File:** `main.py` (lines 105-116)

```python
def semantic_implies(candidate_val: str, required_val: str) -> bool:
    """
    Check if candidate_val semantically implies required_val using embeddings.
    """
    if not ingestion_clients.embedding_model:
        return candidate_val.lower() == required_val.lower()

    v1 = ingestion_clients.embedding_model.encode(candidate_val)  # ✅ Embedding
    v2 = ingestion_clients.embedding_model.encode(required_val)   # ✅ Embedding

    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(sim) > 0.82  # ✅ Cosine similarity threshold
```

**Where it's used:**

#### A. Item Categorical Matching
**File:** `item_matchers.py` (line 231)
```python
for attr, required_value in required_item["categorical"].items():
    if attr not in candidate_item["categorical"]:
        return False
    candidate_value = candidate_item["categorical"][attr]
    if not implies_fn(candidate_value, required_value):  # ✅ SEMANTIC
        return False
```

**Example:**
```python
Required: {"brand": "apple"}
Candidate: {"brand": "Apple Inc"}
→ implies_fn("Apple Inc", "apple") = True ✅ (high similarity)
```

#### B. Other/Self Categorical Matching
**File:** `other_self_matchers.py` (lines 166, 258)
```python
# M-13: Other-Self Categorical Subset Rule
for attr, required_value in other["categorical"].items():
    if attr not in self_obj["categorical"]:
        return False
    candidate_value = self_obj["categorical"][attr]
    if not implies_fn(candidate_value, required_value):  # ✅ SEMANTIC
        return False
```

**Example:**
```python
Required: other_party_preferences = {"language": "hindi"}
Candidate: self_attributes = {"language": "Hindi language"}
→ implies_fn("Hindi language", "hindi") = True ✅ (high similarity)
```

---

## ❌ Where Embeddings Are NOT Used (Hard Filters)

### 1. **Intent Matching**
```python
# listing_matcher_v2.py line 85
if A["intent"] != B["intent"]:
    return False  # ❌ HARD: Exact string equality
```

### 2. **SubIntent Matching**
```python
# listing_matcher_v2.py lines 96-104
if A["subintent"] == B["subintent"]:  # ❌ HARD
    return False  # For product/service: must be inverse
```

### 3. **Domain Intersection**
```python
# listing_matcher_v2.py line 118
if not _has_intersection(A["domain"], B["domain"]):  # ❌ HARD: Set intersection
    return False
```

### 4. **Item Type Matching**
```python
# item_matchers.py line 466
if required_item["type"] != candidate_item["type"]:  # ❌ HARD: Exact match
    return False
```

### 5. **Numeric Constraints**
```python
# item_matchers.py lines 478-490
# Min constraint
for attr, required_min in required_item["min"].items():
    if candidate_value < required_min:  # ❌ HARD: Numeric comparison
        return False

# Max constraint
for attr, required_max in required_item["max"].items():
    if candidate_value > required_max:  # ❌ HARD: Numeric comparison
        return False
```

### 6. **Location Matching**
```python
# location_matcher_v2.py
if required_mode == "explicit":
    return required_location == candidate_location  # ❌ HARD: String equality
```

---

## 📍 Where Embeddings Are Stored

### **Qdrant (Vector Database)**

**Collections:**
```
product_vectors    - Product listings embeddings
service_vectors    - Service listings embeddings
mutual_vectors     - Mutual listings embeddings
```

**Each Point Contains:**
```python
{
    "id": "uuid-string",
    "vector": [0.123, -0.456, ...],  # 384D or 1024D
    "payload": {
        "listing_id": "uuid-string",
        "intent": "product",
        "domain": ["technology & electronics"],
        "created_at": 1737849600
    }
}
```

**Storage Location:**
- **Local:** `localhost:6333` (Docker container)
- **Render:** Private service (qdrant) with persistent disk
- **Disk Mount:** `/qdrant/storage` (1GB)

### **Supabase (Relational Database)**

**Tables:**
```
product_listings   - Full JSON data
service_listings   - Full JSON data
mutual_listings    - Full JSON data
```

**Schema:**
```sql
CREATE TABLE product_listings (
    id UUID PRIMARY KEY,
    data JSONB,  -- Full listing object
    created_at TIMESTAMP
);
```

**Note:** Supabase stores full listing data, NOT embeddings. Embeddings only in Qdrant.

---

## 🎯 Test Results Analysis

### **Your Tests (test_single_query.py, test_canonicalization.py)**

**What they tested:**
- ✅ GPT API extraction (natural language → structured data)
- ✅ Schema format (categorical vs identity/habits)
- ✅ Polysemy resolution (programming vs spoken language)
- ✅ Canonicalization (phone/mobile → smartphone)

**What they did NOT test:**
- ❌ Vector search (no Qdrant queries)
- ❌ Semantic matching (no `implies_fn` calls)
- ❌ Boolean matching (no calls to `listing_matches_v2`)
- ❌ End-to-end pipeline (extraction → storage → retrieval → matching)

**Why?**
Because the tests only called:
1. OpenAI API (extraction)
2. Schema normalization
3. **NOT the `/match` endpoint** (which uses semantic matching)

---

## 🔬 How to Test Semantic Matching

### **Test 1: Semantic Implies Function**

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "listing_a": {
      "intent": "product",
      "subintent": "buy",
      "domain": ["technology & electronics"],
      "items": [{
        "type": "smartphone",
        "categorical": {"brand": "apple"},
        "min": {}, "max": {}, "range": {}
      }],
      ...
    },
    "listing_b": {
      "intent": "product",
      "subintent": "sell",
      "domain": ["technology & electronics"],
      "items": [{
        "type": "smartphone",
        "categorical": {"brand": "Apple Inc"},  ← Similar but not exact
        "min": {}, "max": {}, "range": {}
      }],
      ...
    }
  }'
```

**Expected:**
```json
{
  "match": true  ← semantic_implies("Apple Inc", "apple") returns true
}
```

### **Test 2: Vector Search**

```bash
# 1. Ingest a listing
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "listing": {
      "intent": "product",
      "subintent": "sell",
      "domain": ["technology & electronics"],
      "items": [{"type": "laptop", ...}],
      ...
    }
  }'

# 2. Search for similar listings
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "listing": {
      "intent": "product",
      "subintent": "buy",
      "domain": ["technology & electronics"],
      "items": [{"type": "notebook computer", ...}],  ← Semantically similar
      ...
    }
  }'
```

**Expected:**
- Qdrant vector search finds "laptop" listing even though query said "notebook computer"
- Returns listing_id of the ingested laptop

---

## 📊 Summary Table

| Component | Uses Embeddings? | Type | Where |
|-----------|-----------------|------|-------|
| **Candidate Retrieval** | ✅ YES | Semantic (Vector Search) | `retrieval_service.py` L287 |
| **Categorical Matching** | ✅ YES | Semantic (Cosine Similarity) | `main.py` L105, `item_matchers.py` L231 |
| **Intent Gate** | ❌ NO | Hard (Exact Match) | `listing_matcher_v2.py` L85 |
| **Domain Gate** | ❌ NO | Hard (Set Intersection) | `listing_matcher_v2.py` L118 |
| **Item Type** | ❌ NO | Hard (Exact Match) | `item_matchers.py` L466 |
| **Numeric Constraints** | ❌ NO | Hard (< > = comparisons) | `item_matchers.py` L478-490 |
| **Location** | ❌ NO | Hard (String Equality) | `location_matcher_v2.py` |

---

## 🎯 Final Answer

### **Your Question:** "Are we matching semantically using embeddings or just hard filters?"

**Answer:** **BOTH!**

1. **Semantic (Embeddings):**
   - ✅ Vector search in Qdrant for candidate retrieval
   - ✅ Categorical attribute matching via `semantic_implies()`
   - ✅ Embeddings stored in Qdrant (384D or 1024D vectors)
   - ✅ Cosine similarity threshold: 0.82

2. **Hard Filters:**
   - ❌ Intent/subintent matching (exact)
   - ❌ Domain/category intersection (set ops)
   - ❌ Item type matching (exact)
   - ❌ Numeric constraints (>, <, =)
   - ❌ Location matching (string equality)

### **Your Tests:**
- Tested: ✅ Extraction, schema format, canonicalization
- Did NOT test: ❌ Vector search, semantic matching, boolean matching
- Reason: Only tested GPT API, not the `/match` or `/search` endpoints

### **To Test Full System:**
1. Use `/ingest` to store listings (generates embeddings)
2. Use `/search` to retrieve candidates (vector search)
3. Use `/match` to check if two listings match (semantic + hard)

**The system IS using semantic matching with embeddings! Your tests just didn't exercise those parts yet.** 🚀

---

## 📁 Key Files Reference

| File | Purpose | Embeddings Used? |
|------|---------|-----------------|
| `retrieval_service.py` | Candidate retrieval | ✅ YES (Qdrant search) |
| `ingestion_pipeline.py` | Store listings + embeddings | ✅ YES (Generate & store) |
| `main.py` | API endpoints + `semantic_implies()` | ✅ YES (Categorical matching) |
| `listing_matcher_v2.py` | Boolean matching orchestration | ⚠️ Passes `implies_fn` |
| `item_matchers.py` | Item matching logic | ✅ YES (Uses `implies_fn`) |
| `other_self_matchers.py` | Other/Self matching | ✅ YES (Uses `implies_fn`) |
| `location_matcher_v2.py` | Location matching | ❌ NO (Hard filters) |
| `embedding_builder.py` | Convert listing → text for embedding | N/A (Helper) |

---

**Conclusion:** The matching system is a sophisticated hybrid that combines the best of both worlds - semantic understanding via embeddings for flexible matching, and hard constraints for precise requirements. 🎯
