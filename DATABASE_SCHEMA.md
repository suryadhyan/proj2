# 🗄️ Database Schema Documentation

**Date:** 2026-01-15
**Purpose:** Complete schema for matching system with search history

---

## 📊 **Supabase Tables**

### **1. Matches Table** (NEW - Search History)

```sql
CREATE TABLE matches (
    match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_user_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    query_json JSONB NOT NULL,
    has_matches BOOLEAN NOT NULL,
    match_count INTEGER NOT NULL DEFAULT 0,
    matched_user_ids UUID[] DEFAULT '{}',
    matched_listing_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_matches_user ON matches(query_user_id);
CREATE INDEX idx_matches_created ON matches(created_at DESC);
CREATE INDEX idx_matches_has_matches ON matches(has_matches);
```

**Purpose:** Store every search query and its results (whether matches found or not)

**Fields:**
- `match_id` - Unique identifier for this search
- `query_user_id` - User who performed the search
- `query_text` - Original natural language query
- `query_json` - GPT extracted structured JSON
- `has_matches` - TRUE if matches found, FALSE if empty
- `match_count` - Number of matches found
- `matched_user_ids` - Array of user IDs who matched
- `matched_listing_ids` - Array of listing IDs that matched
- `created_at` - Timestamp

---

### **2. Product Listings Table** (UPDATED)

```sql
CREATE TABLE product_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_product_user ON product_listings(user_id);
CREATE INDEX idx_product_created ON product_listings(created_at DESC);
CREATE INDEX idx_product_match ON product_listings(match_id);
```

**Purpose:** Store product intent listings

**Fields:**
- `id` - Listing UUID (used in Qdrant too)
- `user_id` - User who owns this listing
- `match_id` - Reference to matches table (if listing came from search)
- `data` - Full listing JSON (JSONB)
- `created_at` - Timestamp

---

### **3. Service Listings Table** (UPDATED)

```sql
CREATE TABLE service_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_service_user ON service_listings(user_id);
CREATE INDEX idx_service_created ON service_listings(created_at DESC);
CREATE INDEX idx_service_match ON service_listings(match_id);
```

**Purpose:** Store service intent listings

---

### **4. Mutual Listings Table** (UPDATED)

```sql
CREATE TABLE mutual_listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    match_id UUID REFERENCES matches(match_id),
    data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_mutual_user ON mutual_listings(user_id);
CREATE INDEX idx_mutual_created ON mutual_listings(created_at DESC);
CREATE INDEX idx_mutual_match ON mutual_listings(match_id);
```

**Purpose:** Store mutual intent listings

---

## 🔍 **Qdrant Collections** (NO CHANGES)

### **Product Vectors**
```python
Collection: "product_vectors"
Vector Dimension: 384 or 1024
Distance: Cosine

Payload:
{
    "listing_id": "uuid-string",
    "intent": "product",
    "domain": ["technology & electronics"],
    "created_at": timestamp
}
```

### **Service Vectors**
```python
Collection: "service_vectors"
Vector Dimension: 384 or 1024
Distance: Cosine

Payload:
{
    "listing_id": "uuid-string",
    "intent": "service",
    "domain": ["business services"],
    "created_at": timestamp
}
```

### **Mutual Vectors**
```python
Collection: "mutual_vectors"
Vector Dimension: 384 or 1024
Distance: Cosine

Payload:
{
    "listing_id": "uuid-string",
    "intent": "mutual",
    "category": ["roommates"],
    "created_at": timestamp
}
```

---

## 📋 **Data Flow**

### **Endpoint 1: `/search-and-match`**
```
Writes to: matches table ONLY
Reads from: product_listings, service_listings, mutual_listings, Qdrant
```

### **Endpoint 2: `/store-listing`**
```
Writes to:
  - product_listings OR service_listings OR mutual_listings
  - Qdrant (product_vectors OR service_vectors OR mutual_vectors)
Reads from: None
```

---

## 🔗 **Relationships**

```
matches (1) ←→ (0..1) product_listings
  ↓
  A match entry MAY lead to a stored listing
  (if user clicks "Store my query")

matches.query_user_id → User who searched
matched_user_ids[] → Users who have matching listings
matched_listing_ids[] → Listings that matched
```

---

## 📊 **Example Data**

### **Matches Table Entry (No matches found)**
```json
{
  "match_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_user_id": "user-123",
  "query_text": "I want to buy used iPhone in Mumbai",
  "query_json": {
    "intent": "product",
    "subintent": "buy",
    "items": [{"type": "smartphone", "categorical": {"brand": "apple"}}],
    ...
  },
  "has_matches": false,
  "match_count": 0,
  "matched_user_ids": [],
  "matched_listing_ids": [],
  "created_at": "2026-01-15T10:30:00Z"
}
```

### **Matches Table Entry (3 matches found)**
```json
{
  "match_id": "660e8400-e29b-41d4-a716-446655440111",
  "query_user_id": "user-456",
  "query_text": "Selling iPhone 13 in Mumbai",
  "query_json": {
    "intent": "product",
    "subintent": "sell",
    ...
  },
  "has_matches": true,
  "match_count": 3,
  "matched_user_ids": ["user-111", "user-222", "user-333"],
  "matched_listing_ids": ["listing-aaa", "listing-bbb", "listing-ccc"],
  "created_at": "2026-01-15T11:00:00Z"
}
```

### **Product Listings Entry (Stored from search)**
```json
{
  "id": "listing-xyz-789",
  "user_id": "user-123",
  "match_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "intent": "product",
    "subintent": "buy",
    "domain": ["technology & electronics"],
    "items": [{"type": "smartphone", ...}],
    ...
  },
  "created_at": "2026-01-15T10:35:00Z"
}
```

---

## 🔧 **Migration Steps**

### **1. Create matches table:**
```sql
CREATE TABLE matches (
    match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_user_id UUID NOT NULL,
    query_text TEXT NOT NULL,
    query_json JSONB NOT NULL,
    has_matches BOOLEAN NOT NULL,
    match_count INTEGER NOT NULL DEFAULT 0,
    matched_user_ids UUID[] DEFAULT '{}',
    matched_listing_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_matches_user ON matches(query_user_id);
CREATE INDEX idx_matches_created ON matches(created_at DESC);
CREATE INDEX idx_matches_has_matches ON matches(has_matches);
```

### **2. Add match_id to existing tables:**
```sql
ALTER TABLE product_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);
ALTER TABLE service_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);
ALTER TABLE mutual_listings ADD COLUMN match_id UUID REFERENCES matches(match_id);

CREATE INDEX idx_product_match ON product_listings(match_id);
CREATE INDEX idx_service_match ON service_listings(match_id);
CREATE INDEX idx_mutual_match ON mutual_listings(match_id);
```

### **3. Add user_id if missing:**
```sql
-- Check if user_id exists, if not add it
ALTER TABLE product_listings ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE service_listings ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE mutual_listings ADD COLUMN IF NOT EXISTS user_id UUID;

-- If user_id doesn't have NOT NULL constraint, add it
-- (May need to set default values for existing rows first)
ALTER TABLE product_listings ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE service_listings ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE mutual_listings ALTER COLUMN user_id SET NOT NULL;
```

---

## ✅ **Verification Queries**

### **Check if tables exist:**
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('matches', 'product_listings', 'service_listings', 'mutual_listings');
```

### **Check matches table structure:**
```sql
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'matches'
ORDER BY ordinal_position;
```

### **Get recent searches:**
```sql
SELECT match_id, query_user_id, query_text, has_matches, match_count, created_at
FROM matches
ORDER BY created_at DESC
LIMIT 10;
```

### **Get user's search history:**
```sql
SELECT match_id, query_text, has_matches, match_count, created_at
FROM matches
WHERE query_user_id = 'user-123'
ORDER BY created_at DESC;
```

### **Get stored listings linked to searches:**
```sql
SELECT l.id, l.user_id, l.match_id, l.created_at, m.query_text
FROM product_listings l
LEFT JOIN matches m ON l.match_id = m.match_id
WHERE l.match_id IS NOT NULL
ORDER BY l.created_at DESC;
```

---

## 📊 **Summary**

**New Table:**
- ✅ `matches` - Stores all search queries and results

**Updated Tables:**
- ✅ `product_listings` - Added `user_id`, `match_id`
- ✅ `service_listings` - Added `user_id`, `match_id`
- ✅ `mutual_listings` - Added `user_id`, `match_id`

**No Changes:**
- ✅ Qdrant collections remain the same

**Ready for:**
- ✅ `/search-and-match` endpoint implementation
- ✅ `/store-listing` endpoint implementation
- ✅ Complete flow testing
