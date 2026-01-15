# 🔄 Schema Pipeline - Complete Flow Explanation

## 📊 **The Full Architecture**

```
User Query
    ↓
┌──────────────────────────────────────────────────────┐
│ PHASE 1: LLM EXTRACTION (GPT-4o API)                │
│ Output: NEW Schema (14 fields, axis-based)          │
└──────────────────────────────────────────────────────┘
    ↓
    NEW Schema Format:
    {
      "intent": "product",
      "subintent": "buy",
      "domain": ["technology & electronics"],
      "primary_mutual_category": [],
      "items": [{
        "type": "laptop",
        "categorical": {"brand": "apple", "condition": "used"},
        "min": {
          "capacity": [{"type": "memory", "value": 16, "unit": "gb"}]
        },
        "max": {
          "cost": [{"type": "price", "value": 80000, "unit": "inr"}]
        },
        "range": {}
      }],
      "item_exclusions": [],
      "other_party_preferences": {
        "identity": [{"type": "language", "value": "kannada"}],
        "habits": {},
        "min": {},
        "max": {},
        "range": {}
      },
      "other_party_exclusions": {},
      "self_attributes": {
        "identity": [],
        "habits": {},
        "min": {},
        "max": {},
        "range": {}
      },
      "self_exclusions": {},
      "target_location": {"name": "bangalore"},
      "location_match_mode": "explicit",
      "location_exclusions": [],
      "reasoning": "..."
    }
    ↓
┌──────────────────────────────────────────────────────┐
│ PHASE 2.1: SCHEMA NORMALIZER V2                     │
│ File: schema_normalizer_v2.py                       │
│ Function: normalize_and_validate_v2()                │
│                                                      │
│ Transforms:                                          │
│ - NEW schema (14 fields) → OLD schema (12 fields)   │
│ - Axis-based constraints → Flat constraints         │
│ - Field renames (12 mappings)                       │
└──────────────────────────────────────────────────────┘
    ↓
    OLD Schema Format:
    {
      "intent": "product",
      "subintent": "buy",
      "domain": ["technology & electronics"],
      "category": [],  // ← renamed from primary_mutual_category
      "items": [{
        "type": "laptop",
        "categorical": {"brand": "apple", "condition": "used"},
        "min": {"memory": 16},  // ← flattened from axis-based
        "max": {"price": 80000},  // ← flattened from axis-based
        "range": {}
      }],
      "itemexclusions": [],  // ← renamed from item_exclusions
      "other": {  // ← renamed from other_party_preferences
        "categorical": {"language": "kannada"},  // ← flattened from identity axis
        "min": {},
        "max": {},
        "range": {},
        "otherexclusions": []  // ← nested inside
      },
      "self": {  // ← renamed from self_attributes
        "categorical": {},
        "min": {},
        "max": {},
        "range": {},
        "selfexclusions": []  // ← nested inside
      },
      "location": "bangalore",  // ← simplified from target_location
      "locationmode": "explicit",  // ← renamed from location_match_mode
      "locationexclusions": [],
      "reasoning": "..."
    }
    ↓
┌──────────────────────────────────────────────────────┐
│ PHASE 2.4-2.7: MATCHING ENGINE                      │
│ Files:                                               │
│ - listing_matcher_v2.py (orchestration)             │
│ - item_array_matchers.py (item matching)            │
│ - other_self_matchers.py (other/self matching)      │
│ - location_matcher_v2.py (location matching)        │
│                                                      │
│ Expects: OLD schema format                          │
└──────────────────────────────────────────────────────┘
    ↓
    Match Result: True/False
```

---

## 🔑 **Key Transformations in schema_normalizer_v2.py**

### **1. Field Name Mappings (12 renames)**

| NEW Field Name | OLD Field Name | Notes |
|----------------|----------------|-------|
| `intent` | `intent` | No change |
| `subintent` | `subintent` | No change |
| `domain` | `domain` | No change |
| `items` | `items` | No change (but contents transformed) |
| `reasoning` | `reasoning` | No change |
| `primary_mutual_category` | `category` | ✅ Renamed |
| `item_exclusions` | `itemexclusions` | ✅ Renamed |
| `other_party_preferences` | `other` | ✅ Renamed |
| `other_party_exclusions` | `other.otherexclusions` | ✅ Nested inside |
| `self_attributes` | `self` | ✅ Renamed |
| `self_exclusions` | `self.selfexclusions` | ✅ Nested inside |
| `target_location` | `location` | ✅ Renamed + simplified |
| `location_match_mode` | `locationmode` | ✅ Renamed |
| `location_exclusions` | `locationexclusions` | No change |

---

### **2. Axis-Based → Flat Constraint Transformation**

#### **For Items:**

**NEW (axis-based):**
```json
"min": {
  "capacity": [
    {"type": "memory", "value": 16, "unit": "gb"},
    {"type": "storage", "value": 256, "unit": "gb"}
  ]
}
```

**OLD (flat):**
```json
"min": {
  "memory": 16,
  "storage": 256
}
```

#### **For People (other/self):**

**NEW (axis-based with identity/habits):**
```json
"other_party_preferences": {
  "identity": [
    {"type": "language", "value": "kannada"},
    {"type": "profession", "value": "plumber"}
  ],
  "habits": {
    "smoking": "no",
    "drinking": "no"
  },
  "min": {
    "time": [{"type": "experience", "value": 60, "unit": "months"}]
  }
}
```

**OLD (flat categorical):**
```json
"other": {
  "categorical": {
    "language": "kannada",
    "profession": "plumber",
    "smoking": "no",
    "drinking": "no"
  },
  "min": {
    "experience": 60
  },
  "max": {},
  "range": {}
}
```

---

### **3. Location Simplification**

**NEW:**
```json
"target_location": {"name": "bangalore"},
"location_match_mode": "explicit"
```

**OLD:**
```json
"location": "bangalore",  // Simple string
"locationmode": "explicit"
```

For route mode:
```json
// NEW
"target_location": {"origin": "delhi", "destination": "mumbai"}

// OLD
"location": {"origin": "delhi", "destination": "mumbai"}  // Dict for route
```

---

## ✅ **Your Questions Answered**

### **Q1: Is the API doing its work correctly?**

**YES! ✅** The API outputs **NEW schema format** correctly:
- 14 fields present ✅
- Axis-based constraints ✅
- Proper structure ✅

The test examples were using **OLD format**, which caused false failures.

---

### **Q2: Does the matching engine schema match API output?**

**NO - But that's by design! ✅**

- **API outputs:** NEW schema (14 fields, axis-based)
- **Matching engine expects:** OLD schema (12 fields, flat)
- **Solution:** `schema_normalizer_v2.py` transforms between them

**This is the correct architecture!**

---

### **Q3: If schema mismatch, no matches?**

**Correct! But transformation layer prevents this:**

```
API (NEW) → Normalizer → OLD → Matching ✅
```

Without normalizer:
```
API (NEW) → Matching (expects OLD) → ❌ FAIL
```

---

## 🔧 **Pre-processing: Rule-Based vs API Call?**

### **Current Architecture (Rule-Based Transformation):**

```python
# schema_normalizer_v2.py
def normalize_and_validate_v2(listing: Dict) -> Dict:
    # 1. Validate NEW schema
    validate_new_schema(listing)

    # 2. Transform NEW → OLD (deterministic rules)
    old_listing = transform_new_to_old(listing)

    # 3. Return OLD format
    return old_listing
```

**This is rule-based transformation ✅**
- Fast (milliseconds)
- Deterministic (100% consistent)
- No API cost
- Already implemented!

---

### **Alternative: Another API Call (NOT NEEDED)**

```
Query → API 1 (extract) → NEW → API 2 (validate/transform) → OLD → Matching
```

**Why this is NOT needed:**
- ❌ 2x cost
- ❌ 2x latency
- ❌ More failure points
- ✅ Rule-based transformation already works!

---

## 🎯 **What You Actually Need**

### **1. API Output Post-Processing (For Edge Cases)**

After API but before normalizer:

```python
def post_process_api_output(api_output: Dict) -> Dict:
    """
    Fix edge cases before normalization:
    - Currency: "local" → "inr"
    - Domain case: "technology & electronics" → "Technology & Electronics"
    - Missing constraints: Add exact values as ranges
    """
    # Use /data/synonyms.json, /pos/data/currencies.json
    return cleaned_output
```

### **2. The Complete Pipeline**

```
Query
    ↓
GPT-4o API (extraction)
    ↓
API Output (NEW schema, might have edge cases)
    ↓
┌─────────────────────────────┐
│ POST-PROCESSOR (NEW)        │
│ - Canonicalize domains      │
│ - Fix currency units        │
│ - Add missing constraints   │
│ - Validate schema           │
└─────────────────────────────┘
    ↓
Cleaned NEW schema
    ↓
┌─────────────────────────────┐
│ SCHEMA NORMALIZER V2        │
│ (EXISTING - Works!)         │
│ - Transform NEW → OLD       │
│ - Flatten axes              │
│ - Rename fields             │
└─────────────────────────────┘
    ↓
OLD schema
    ↓
MATCHING ENGINE V2
```

---

## 📝 **Recommendations**

### **What You Have (Already Working):**

1. ✅ API extraction (NEW schema)
2. ✅ schema_normalizer_v2.py (NEW → OLD transformation)
3. ✅ Matching engine V2 (expects OLD schema)

**This pipeline is CORRECT and WORKING!**

---

### **What You Need to Add:**

1. **Post-processor (before normalizer):**
   ```python
   # new file: api_post_processor.py
   def post_process(api_output):
       # Fix currency: local → inr
       # Fix domain case
       # Add missing exact constraints
       # Canonicalize using /data/synonyms.json
       return cleaned_output
   ```

2. **Update test examples:**
   - Change from OLD format to NEW format
   - Or keep OLD and test after normalization

---

### **Do NOT Need:**

- ❌ Another API call for transformation
- ❌ Change matching engine (it's correct)
- ❌ Change schema_normalizer_v2.py (it's correct)

---

## 🎯 **Final Answer to Your Question**

**"Pre-process: Rule-based or API call?"**

**Answer: Rule-based ✅ (Already implemented!)**

- `schema_normalizer_v2.py` IS your rule-based preprocessor
- It transforms NEW → OLD deterministically
- No API call needed for this transformation
- Just add light post-processing for edge cases (currency, domain case)

**Your architecture is already correct!**

---

## 📊 **Summary**

| Component | Format | Status |
|-----------|--------|--------|
| **API Output** | NEW schema (14 fields) | ✅ Working correctly |
| **schema_normalizer_v2.py** | NEW → OLD transform | ✅ Already implemented |
| **Matching Engine** | OLD schema (12 fields) | ✅ Working correctly |
| **Post-processor (NEW)** | Edge case fixes | 🔧 Need to add |

**Bottom Line:** Your verification script was wrong, not the API. The architecture is correct. Just add edge case post-processing and you're done!
