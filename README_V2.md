# Schema V2 Quick Start Guide

## 🚀 Getting Started in 30 Seconds

```python
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

# 1. Load your NEW schema listing
listing_new = {
    "intent": "product",
    "subintent": "buy",
    "domain": ["Technology & Electronics"],
    "primary_mutual_category": [],
    "items": [{
        "type": "laptop",
        "categorical": {"brand": "apple"},
        "min": {
            "capacity": [{"type": "memory", "value": 16, "unit": "gb"}]
        },
        "max": {
            "cost": [{"type": "price", "value": 80000, "unit": "inr"}]
        },
        "range": {}
    }],
    "item_exclusions": [],
    "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "other_party_exclusions": [],
    "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
    "self_exclusions": [],
    "target_location": {},
    "location_match_mode": "near_me",
    "location_exclusions": [],
    "reasoning": "..."
}

# 2. Transform NEW → OLD format
listing_old = normalize_and_validate_v2(listing_new)

# 3. Use with existing matching engine
matches = listing_matches_v2(listing_a_old, listing_b_old)

# 4. Generate embeddings (works with transformed data)
from embedding_builder import build_embedding_text
embedding_text = build_embedding_text(listing_old)

# 5. Ingest to Supabase + Qdrant (works with transformed data)
from ingestion_pipeline import IngestionClients, ingest_listing
clients = IngestionClients()
clients.initialize()
listing_id, embedding = ingest_listing(clients, listing_old)
```

## 📊 What's Different?

### NEW Schema → OLD Schema Transformation

| Aspect | NEW Schema | OLD Schema (after transform) |
|--------|-----------|------------------------------|
| **Field Names** | `other_party_preferences`, `self_attributes`, `target_location` | `other`, `self`, `location` |
| **Constraints** | Axis-based: `{capacity: [{type: "memory", value: 16}]}` | Flattened: `{memory: 16}` |
| **Location** | Object: `{"name": "bangalore"}` | String: `"bangalore"` |

## ✅ Compatibility Matrix

| Component | Status | Note |
|-----------|--------|------|
| `schema_normalizer_v2.py` | ✅ NEW | Transforms NEW → OLD |
| `location_matcher_v2.py` | ✅ NEW | Simplified location matching |
| `listing_matcher_v2.py` | ✅ NEW | Uses v2 location matcher |
| `embedding_builder.py` | ✅ UNCHANGED | Works with OLD format |
| `retrieval_service.py` | ✅ UNCHANGED | Works with OLD format |
| `ingestion_pipeline.py` | ✅ UNCHANGED | Expects OLD format |
| All other matchers | ✅ UNCHANGED | Work with transformed data |

## 📁 Key Files

### V2 Files (New)
- `schema_normalizer_v2.py` - Validates & transforms NEW schema
- `location_matcher_v2.py` - Simplified location matching (name-based)
- `listing_matcher_v2.py` - Orchestration with v2 location logic

### Test Files
- `integration_example_v2.py` - **START HERE** - Complete example
- `test_all_examples.py` - Test with all 10 examples
- `test_e2e_matching.py` - End-to-end matching tests
- `test_embedding_v2.py` - Embedding generation tests

### Documentation
- `MIGRATION_SUMMARY.md` - Complete technical documentation
- `GLOBAL_REFERENCE_CONTEXT.md` - NEW schema specification (in `new/` folder)

## 🎯 Common Tasks

### Task 1: Transform a NEW schema listing

```python
from schema_normalizer_v2 import normalize_and_validate_v2

listing_old = normalize_and_validate_v2(listing_new)
```

### Task 2: Match two listings

```python
from listing_matcher_v2 import listing_matches_v2

# Both listings must be in OLD format (after transformation)
result = listing_matches_v2(listing_a_old, listing_b_old)
# Returns: True if B satisfies A's requirements, False otherwise
```

### Task 3: Generate embeddings

```python
from embedding_builder import build_embedding_text

# listing must be in OLD format
embedding_text = build_embedding_text(listing_old)
# Use with sentence-transformers: model.encode(embedding_text)
```

### Task 4: Ingest into database

```python
from ingestion_pipeline import IngestionClients, ingest_listing

clients = IngestionClients()
clients.initialize()  # Connects to Supabase + Qdrant

# listing must be in OLD format
listing_id, embedding = ingest_listing(clients, listing_old, verbose=True)
```

## 🧪 Testing Your Integration

### Run the complete integration example:

```bash
python3 integration_example_v2.py
```

Expected output:
```
✅ V2 Pipeline Complete:
  1. ✓ NEW schema → schema_normalizer_v2 → OLD format
  2. ✓ OLD format → listing_matcher_v2 → boolean matching
  3. ✓ OLD format → embedding_builder → embedding text
  4. ✓ OLD format → ready for ingestion_pipeline
```

### Run all 10 examples test:

```bash
python3 test_all_examples.py
```

Expected output:
```
RESULTS: 10/10 passed, 0/10 failed
✅ ALL EXAMPLES TRANSFORMED SUCCESSFULLY!
```

## ⚠️ Important Notes

### 1. Always Transform First
```python
# ❌ WRONG: Using NEW schema directly
result = listing_matches_v2(listing_new, ...)  # Will fail!

# ✅ CORRECT: Transform first
listing_old = normalize_and_validate_v2(listing_new)
result = listing_matches_v2(listing_old, ...)
```

### 2. Location Matching is Simplified
- OLD: Complex constraints (distance, zones, accessibility)
- NEW (V2): Simple name-based matching
- Supported modes: near_me, explicit, target_only, route, global

### 3. Units are Stripped
- NEW schema has units: `{"type": "price", "value": 50000, "unit": "inr"}`
- OLD format strips units: `{"price": 50000}`
- Units are not used in matching currently (can add normalization later)

### 4. Field Names Changed
When accessing transformed data, use OLD field names:
```python
# Use "other" not "other_party_preferences"
other_prefs = listing_old["other"]

# Use "location" not "target_location"
location = listing_old["location"]
```

## 🐛 Troubleshooting

### Error: "Missing required fields"
**Cause**: NEW schema listing missing fields
**Fix**: Ensure all 14 NEW schema fields are present (see `GLOBAL_REFERENCE_CONTEXT.md`)

### Error: "Invalid axis"
**Cause**: Using non-standard axis name
**Fix**: Only use 10 valid axes: identity, capacity, performance, quality, quantity, time, space, cost, mode, skill

### Error: "Constraint should be flat"
**Cause**: Trying to use NEW schema format directly in matching
**Fix**: Transform with `normalize_and_validate_v2()` first

## 📚 Learning Path

1. **Start Here**: Run `integration_example_v2.py` to see the full pipeline
2. **Understand Transformation**: Read `schema_normalizer_v2.py` header comments
3. **Test Examples**: Review `stage3_extraction1.json` for real-world examples
4. **Deep Dive**: Read `MIGRATION_SUMMARY.md` for complete technical details

## 🎯 Production Checklist

Before deploying to production:

- [ ] Test with your actual NEW schema data
- [ ] Verify all 14 fields present in your listings
- [ ] Run `test_all_examples.py` successfully
- [ ] Test end-to-end: transformation → matching → embedding → ingestion
- [ ] Validate Qdrant payload structure
- [ ] Verify Supabase insertion works
- [ ] Test with representative sample (100+ listings recommended)

## 📞 Need Help?

- **Examples**: See `integration_example_v2.py`
- **Tests**: Check `test_*.py` files for edge cases
- **Documentation**: Read `MIGRATION_SUMMARY.md`
- **Schema Reference**: `GLOBAL_REFERENCE_CONTEXT.md` (in `new/` folder)

---

**Status**: ✅ Production Ready
**Version**: V2.0
**Last Updated**: 2026-01-13
