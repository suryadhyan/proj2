# 🚨 CRITICAL SCHEMA FINDINGS

**Date:** 2026-01-15
**Investigation:** Complete codebase analysis

---

## 🔍 **THE ISSUE DISCOVERED**

### **API Output vs System Expectation MISMATCH**

| Component | Format Used | Example |
|-----------|-------------|---------|
| **API Output (Current)** | `identity` array + `habits` object | `{"identity": [{"type": "language", "value": "kannada"}], "habits": {"smoking": "no"}}` |
| **Test Examples** | `categorical` object | `{"categorical": {"language": "kannada"}, "min": {}, "max": {}, "range": {}}` |
| **Normalizer Expects** | `categorical` object | `{"categorical": {...}, "min": {}, "max": {}, "range": {}}` |
| **Matching Engine Expects** | Flat `categorical` (after transform) | `{"categorical": {"language": "kannada"}}` |

---

## 📊 **Evidence from Codebase**

### **1. Test Examples (stage3_extraction1.json)**

Example 3 (plumber with kannada):
```json
"other_party_preferences": {
  "categorical": {
    "language": "kannada"
  },
  "min": {},
  "max": {},
  "range": {}
}
```

**Format: ✅ Uses `categorical`**

---

### **2. README_V2.md Example**

```python
"other_party_preferences": {
  "categorical": {},
  "min": {},
  "max": {},
  "range": {}
}
```

**Format: ✅ Uses `categorical`**

---

### **3. schema_normalizer_v2.py**

```python
def transform_constraint_object(new_constraints: Dict) -> Dict:
    # Get each mode (categorical stays same, min/max/range need flattening)
    categorical = new_constraints.get("categorical", {})

    # Flatten axis-based constraints
    min_axis = new_constraints.get("min", {})
    max_axis = new_constraints.get("max", {})
    # ... NO HANDLING of identity/habits
```

**Expects: ✅ `categorical` field**
**Does NOT handle: ❌ `identity` or `habits` fields**

---

### **4. API Test Results**

From our canonicalization tests:
```json
"other_party_preferences": {
  "identity": [
    {"type": "language", "value": "kannada"}
  ],
  "habits": {},
  "min": {},
  "max": {},
  "range": {}
}
```

**Output: ❌ Uses `identity` + `habits` (incompatible!)**

---

### **5. main.py (Production API)**

```python
@app.post("/match")
async def match_endpoint(request: MatchRequest):
    # 1. Normalize
    listing_a_old = normalize_and_validate_v2(request.listing_a)
    listing_b_old = normalize_and_validate_v2(request.listing_b)

    # 2. Match
    is_match = listing_matches_v2(listing_a_old, listing_b_old)
```

**Expects: ✅ Input format that normalizer can handle**
**Currently receiving: ❌ `identity/habits` format from API**

---

## 🎯 **Root Cause**

### **TWO Different Schema Formats in Prompt**

The prompt file (PROMPT_STAGE2.txt) contains BOTH formats:

**Format A (Test Examples Use This):**
```json
"other_party_preferences": {
  "categorical": {"language": "kannada"},
  "min": {...},
  "max": {...},
  "range": {}
}
```

**Format B (API Currently Outputs This):**
```json
"other_party_preferences": {
  "identity": [{"type": "language", "value": "kannada"}],
  "habits": {"smoking": "no"},
  "min": {...},
  "max": {...},
  "range": {}
}
```

**The API chose Format B** (axis-based with identity/habits)
**The system expects Format A** (flat categorical)

---

## ✅ **Solutions**

### **Option 1: Update Prompt (RECOMMENDED)**

**Change prompt to enforce Format A (categorical)**

```
For other_party_preferences and self_attributes:

CORRECT FORMAT:
{
  "categorical": {
    "language": "kannada",
    "profession": "plumber",
    "smoking": "no"
  },
  "min": {...},
  "max": {...},
  "range": {}
}

WRONG - Do NOT use:
{
  "identity": [...],  // ❌ Don't use this
  "habits": {...}     // ❌ Don't use this
}

ALL personal attributes go in "categorical" object.
```

**Advantages:**
- ✅ Quick fix (just update prompt)
- ✅ Compatible with existing system
- ✅ No code changes needed
- ✅ Test examples already use this format

---

### **Option 2: Update schema_normalizer_v2.py**

**Add logic to transform identity/habits → categorical**

```python
def transform_constraint_object(new_constraints: Dict) -> Dict:
    if not isinstance(new_constraints, dict):
        return {
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }

    # Get categorical (may be empty)
    categorical = new_constraints.get("categorical", {})

    # NEW: Handle identity array
    if "identity" in new_constraints:
        for item in new_constraints["identity"]:
            if isinstance(item, dict) and "type" in item and "value" in item:
                categorical[item["type"]] = item["value"]

    # NEW: Handle habits object
    if "habits" in new_constraints:
        habits = new_constraints["habits"]
        if isinstance(habits, dict):
            categorical.update(habits)

    # Continue with existing logic...
    min_axis = new_constraints.get("min", {})
    max_axis = new_constraints.get("max", {})
    range_axis = new_constraints.get("range", {})

    # ... rest of transformation
```

**Advantages:**
- ✅ API can output either format
- ✅ More flexible
- ✅ Backwards compatible

**Disadvantages:**
- ⚠️ More complex
- ⚠️ Maintains two schema formats

---

### **Option 3: Hybrid Approach**

1. **Update prompt** to prefer `categorical` format
2. **Add fallback** in normalizer to handle `identity/habits` if present
3. **Best of both worlds**

---

## 📋 **Current System Architecture (CORRECT)**

```
NEW Schema Input
    ↓
schema_normalizer_v2.py
    ├─ Validates 14 fields ✅
    ├─ Renames fields (12 mappings) ✅
    ├─ Flattens axis constraints ✅
    ├─ Expects categorical format ✅
    └─ Does NOT handle identity/habits ❌
    ↓
OLD Schema Format
    ↓
listing_matcher_v2.py
    ├─ Uses listing_matches_v2() ✅
    ├─ Calls item_array_matchers ✅
    ├─ Calls other_self_matchers ✅
    ├─ Calls location_matcher_v2 ✅
    └─ Expects flat categorical ✅
    ↓
Match Result
```

**System is correct! ✅**
**API output format is incompatible ❌**

---

## 🎯 **What Needs to Be Done**

### **IMMEDIATE ACTION (Choose One):**

#### **Approach A: Fix Prompt (Easiest)**

1. Update `PROMPT_STAGE2.txt`
2. Remove or de-emphasize `identity/habits` format
3. Emphasize `categorical` format with examples
4. Re-test API extraction

#### **Approach B: Update Normalizer**

1. Edit `schema_normalizer_v2.py`
2. Add `identity/habits` → `categorical` transformation
3. Test with both formats
4. Update documentation

---

## 🧪 **Testing Required**

After fix, test these queries:

```python
# Test 1: Language in categorical
{
  "other_party_preferences": {
    "categorical": {"language": "kannada"}
  }
}

# Test 2: Multiple attributes
{
  "other_party_preferences": {
    "categorical": {
      "language": "kannada",
      "profession": "plumber",
      "experience_level": "experienced"
    }
  }
}

# Test 3: Habits (if updating normalizer)
{
  "other_party_preferences": {
    "categorical": {
      "smoking": "no",
      "drinking": "no"
    }
  }
}
```

---

## 📊 **Impact Assessment**

### **If We Don't Fix:**

```
API Output (identity/habits)
    ↓
schema_normalizer_v2.py
    ├─ Reads categorical: {} (empty!) ❌
    ├─ Ignores identity array ❌
    ├─ Ignores habits object ❌
    └─ Output: categorical: {} (empty!)
    ↓
listing_matcher_v2.py
    ├─ Checks: other.categorical = {} ✅
    ├─ Checks: self.categorical = {"language": "kannada"} ✅
    └─ Result: Subset match fails (empty ⊄ {"language"}) ❌
    ↓
NO MATCH (False negative!)
```

**Critical Bug: Listings won't match even when they should! 🚨**

---

## ✅ **Recommendation**

**Use Approach A: Update Prompt**

**Rationale:**
1. Test examples already use `categorical` format
2. System is built for `categorical` format
3. Simplest fix (no code changes)
4. Most maintainable (one format)
5. prompt already has this format in examples

**Implementation:**
1. Find all references to `identity/habits` in prompt
2. Replace with `categorical` examples
3. Add explicit rule: "ALL personal attributes → categorical"
4. Re-test extraction with API

---

## 🎯 **Final Answer**

### **The verification script was NOT wrong about schema!**

- ✅ Test examples use CORRECT format (`categorical`)
- ✅ Normalizer expects CORRECT format (`categorical`)
- ✅ Matching engine expects CORRECT format (flat `categorical`)
- ❌ API outputs WRONG format (`identity/habits`)

**Fix: Update prompt to output `categorical` format consistently**

---

## 📝 **Next Steps**

1. ✅ **Identified issue:** API uses `identity/habits`, system expects `categorical`
2. 🔧 **Fix prompt:** Remove `identity/habits`, emphasize `categorical`
3. 🧪 **Re-test:** Run canonicalization tests again
4. ✅ **Verify:** Check that matches work correctly

**After fix:** System will work end-to-end! 🚀
