# 🚀 Quick Start Guide - Testing the Integration

**Status:** Ready to test
**Time Required:** 5-10 minutes

---

## Step 1: Database Setup (Supabase SQL Editor)

Copy and paste this into Supabase SQL Editor:

```sql
-- Create matches table
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

-- Indexes
CREATE INDEX idx_matches_user ON matches(query_user_id);
CREATE INDEX idx_matches_created ON matches(created_at DESC);
CREATE INDEX idx_matches_has_matches ON matches(has_matches);

-- Add columns to existing tables
ALTER TABLE product_listings ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE product_listings ADD COLUMN IF NOT EXISTS match_id UUID REFERENCES matches(match_id);

ALTER TABLE service_listings ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE service_listings ADD COLUMN IF NOT EXISTS match_id UUID REFERENCES matches(match_id);

ALTER TABLE mutual_listings ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE mutual_listings ADD COLUMN IF NOT EXISTS match_id UUID REFERENCES matches(match_id);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_product_user ON product_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_product_match ON product_listings(match_id);
CREATE INDEX IF NOT EXISTS idx_service_user ON service_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_service_match ON service_listings(match_id);
CREATE INDEX IF NOT EXISTS idx_mutual_user ON mutual_listings(user_id);
CREATE INDEX IF NOT EXISTS idx_mutual_match ON mutual_listings(match_id);
```

Click "Run" and verify no errors.

---

## Step 2: Start Local Server

```bash
cd /d/matching-github/proj2

# Set environment variables (replace with your actual values)
export OPENAI_API_KEY=sk-proj-...your-key...
export SUPABASE_URL=https://...your-project....supabase.co
export SUPABASE_KEY=eyJ...your-key...

# Start server
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Wait for this output:**
```
✅ OpenAI client ready
✅ Extraction prompt loaded
✅ Ingestion clients initialized
✅ Retrieval clients initialized
✅ ALL clients initialized successfully
```

Keep this terminal open.

---

## Step 3: Run Test Script

Open a new terminal:

```bash
cd /d/matching-github/proj2

# Run test
python3 test_complete_flow.py
```

**Expected Output:**
```
================================================================================
COMPLETE INTEGRATION TEST - MATCHING SYSTEM
================================================================================

📦 PHASE 1: Storing Candidate Listings
...

🎯 PHASE 2: Testing Target Query 1 (Product)
...
✅ PASS: Found expected match (Match 1)
✅ PASS: All product traps correctly filtered

🎯 PHASE 3: Testing Target Query 2 (Mutual)
...
✅ PASS: Found expected match (Match 2)
✅ PASS: All mutual traps correctly filtered

================================================================================
SUMMARY
================================================================================

📊 Test Results: 4/4 tests passed

🎉 ALL TESTS PASSED! The matching system is working correctly.
```

---

## Step 4: Manual Test (Optional)

Test the endpoints manually with curl:

### Test /search-and-match:
```bash
curl -X POST http://localhost:8000/search-and-match \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to buy a used iPhone in Mumbai",
    "user_id": "test-user-123"
  }'
```

### Test /store-listing:
```bash
curl -X POST http://localhost:8000/store-listing \
  -H "Content-Type: application/json" \
  -d '{
    "listing_json": {
      "intent": "product",
      "subintent": "sell",
      "domain": ["technology & electronics"],
      "items": [{
        "type": "smartphone",
        "categorical": {"brand": "apple", "condition": "used"},
        "min": {},
        "max": {},
        "range": {}
      }],
      "self_attributes": {
        "categorical": {"language": "hindi"},
        "numeric": {},
        "string": {}
      },
      "other_party_preferences": {
        "categorical": {},
        "numeric": {},
        "string": {}
      },
      "location_self": {
        "mode": "explicit",
        "city": "mumbai",
        "state": "",
        "country": "",
        "other": ""
      },
      "location_other": {
        "mode": "any",
        "city": "",
        "state": "",
        "country": "",
        "other": ""
      },
      "urgency": "medium",
      "additional_context": ""
    },
    "user_id": "manual-test-user",
    "match_id": null
  }'
```

---

## Troubleshooting

### Server won't start:
- Check environment variables are set: `echo $OPENAI_API_KEY`
- Make sure Qdrant is running: `docker ps | grep qdrant`
- If Qdrant not running: `docker run -p 6333:6333 -d qdrant/qdrant`

### Test fails:
- Check server logs for errors
- Verify database schema was created correctly
- Make sure all environment variables are set

### Database errors:
- Check Supabase connection: Visit Supabase dashboard
- Verify tables exist: Run `SELECT * FROM matches LIMIT 1;`
- Check columns: `SELECT column_name FROM information_schema.columns WHERE table_name = 'matches';`

---

## What's Being Tested?

### Test Coverage:
- ✅ GPT extraction (natural language → JSON)
- ✅ Schema normalization (NEW → OLD)
- ✅ Qdrant vector search
- ✅ Supabase SQL filtering
- ✅ Boolean matching with semantic_implies
- ✅ Search history storage (matches table)
- ✅ Listing storage (listings tables + Qdrant)
- ✅ Semantic trap filtering (brand, condition, location, habits, etc.)

### Expected Results:
- Target 1 finds Match 1 (used iPhone seller in Mumbai who speaks Hindi)
- Target 1 filters Traps 1-6 (wrong condition, brand, location, language, etc.)
- Target 2 finds Match 2 (female vegetarian non-smoker roommate in Bangalore)
- Target 2 filters Traps 7-10 (wrong gender, habits, age, location)

---

## Files Created:

```
test_queries/
  ├── target_1_product_buyer.json       # Product buyer (iPhone)
  ├── target_2_mutual_roommate.json     # Roommate seeker (female)
  ├── match_1_iphone_seller.json        # Should match Target 1
  ├── match_2_female_roommate.json      # Should match Target 2
  ├── trap_1_new_iphone.json            # Semantic trap (condition)
  ├── trap_2_samsung_used.json          # Semantic trap (brand)
  ├── trap_3_price_too_high.json        # Hard filter trap (price)
  ├── trap_4_language_mismatch.json     # Semantic trap (language)
  ├── trap_5_wrong_intent.json          # Hard filter trap (intent)
  ├── trap_6_storage_too_low.json       # Hard filter trap (storage)
  ├── trap_7_male_roommate.json         # Semantic trap (gender)
  └── trap_8_smoker_roommate.json       # Semantic trap (smoking)

migrations/
  └── 001_create_matches_table.sql      # Database migration

test_complete_flow.py           # Integration test script
run_migration.py                # Migration helper script
DATABASE_SCHEMA.md              # Complete schema documentation
MATCHING_EXPLAINED.md           # Detailed matching explanation
INTEGRATION_COMPLETE.md         # Detailed integration guide
QUICK_START.md                  # This file
```

---

## Next Steps After Successful Test:

1. ✅ Local tests pass (4/4)
2. 🔄 Deploy to Render
3. 🔄 Run same tests against Render URL
4. 🔄 Monitor production usage
5. 🔄 Collect training data for Mistral fine-tuning

---

**Ready to test! 🚀**
