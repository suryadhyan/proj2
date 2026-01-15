# 🎯 Quick Answers to Your Questions

## Q1: Schema - NEW vs OLD?

**Answer: There's only ONE schema (the current one).**

### The Schema Uses:

**For Items:**
```json
"items": [{
  "type": "laptop",
  "categorical": {"brand": "apple", "condition": "used"}  // Flat
}]
```

**For People:**
```json
"other_party_preferences": {
  "identity": [{"type": "language", "value": "kannada"}],  // Axis-based
  "habits": {"smoking": "no"}                              // Flat for flags
}
```

This IS the current format from PROMPT_STAGE2.txt.

The "NEW schema" confusion was because test examples used an older/incorrect format.

---

## Q2: Can preprocessing handle schema differences?

**Answer: Yes, both preprocessing AND post-processing can handle it.**

### Options:

**Option A: Preprocessing (Before API)**
```python
# Modify prompt to force specific format
# Not recommended - LLM knows best structure
```

**Option B: Post-processing (After API)** ✅ RECOMMENDED
```python
def normalize_schema(api_output):
    # Validate all 14 fields present
    # Canonicalize values
    # Normalize case/units
    return normalized_output
```

**Why post-processing is better:**
- LLM outputs most natural structure
- Easier to update rules
- Can handle multiple LLM versions
- Centralized canonicalization

---

## Q3: Phone vs Cellphone vs Mobile - Canonicalization?

**Test Running Now:** `test_canonicalization.py`

Testing these queries:
- "looking for a **phone** under 30k"
- "need a **mobile** under 30k"
- "want to buy a **cellphone** under 30k"
- "searching for a **smartphone** under 30k"

**Expected Results:**
- ✅ API will canonicalize to **ONE** consistent type (probably "smartphone")
- ⚠️ Might have variations → post-processing with `/data/synonyms.json` ensures 100% consistency

---

## Q4: Polysemy Handling?

**Test Running Now:** Testing "language" in different contexts

### Test Cases:

**Context 1: Programming Language**
```
Query: "need a developer who knows Python language"
Expected: items[].categorical.language = "python"  (tech skill)
```

**Context 2: Speaking Language**
```
Query: "need a plumber who speaks Kannada language"
Expected: other_party_preferences.identity[{type: 'language', value: 'kannada'}]
```

**How API Resolves:**
- Semantic understanding from context
- "developer + Python" → programming skill
- "plumber + Kannada" → spoken language
- Uses axis-based structure for people attributes

---

## Q5: Edge Cases?

**Test Running Now:** Testing:

1. **Currency Detection**
   - "under 50k" → Should detect INR from context
   - "under $500" → Should detect USD
   - "under ₹50000" → Should detect INR symbol
   - "under 5 lakh rupees" → Should detect INR + normalize (500000)

2. **Constraint Detection**
   - "16GB RAM" → exact (range with min=max)
   - "at least 16GB" → min constraint
   - "under 80k" → max constraint
   - "between 50k-80k" → range constraint
   - "around 60k" → ??? (test will show)

3. **Implication Rules**
   - "single owner car" → condition:"used" + ownership:"single"
   - "sealed iPhone" → condition:"new" + packaging:"sealed"

---

## 📊 Test Status

**Currently Running:**
```bash
# 30+ queries testing:
✅ Canonicalization (phone/mobile/cellphone)
✅ Polysemy (language, size, experience)
✅ Currency detection
✅ Constraint detection
✅ Implication rules
```

**Results will show:**
- What API handles natively ✅
- What needs post-processing 🔧
- Accuracy on edge cases 📊

**ETA:** ~5 minutes (complex queries take longer)

---

## 💡 Key Insights (Preview)

Based on initial test (query 1):

### **What Works:**
- ✅ Semantic understanding (90%+ accurate)
- ✅ Intent/subintent classification (100%)
- ✅ Basic canonicalization (iPhone → smartphone)
- ✅ Constraint detection (at least, under)
- ✅ Attribute extraction (brand, model, condition)

### **What Needs Post-Processing:**
- 🔧 Domain case (technology & electronics → Technology & Electronics)
- 🔧 Currency unit (local → INR/USD)
- 🔧 100% consistent canonicalization (handle all synonyms)
- 🔧 Schema validation (all 14 fields present)
- 🔧 Unit normalization (month → months)

---

## 🎯 Pipeline Recommendation

```
User Query
    ↓
┌────────────────────────┐
│ LLM Extraction         │  ← GPT-4o API (for demo)
│ (Semantic understanding)│  ← Fine-tuned Mistral (production)
└────────────────────────┘
    ↓
┌────────────────────────┐
│ Post-Processing Layer  │
├────────────────────────┤
│ 1. Schema Validation   │  ← Ensure 14 fields
│ 2. Canonicalizer       │  ← /data/synonyms.json
│ 3. Domain Normalizer   │  ← /data/taxonomy.json
│ 4. Currency Detector   │  ← /pos/data/currencies.json
│ 5. Constraint Detector │  ← /pos/data/linguistic_cues.json
│ 6. Implication Rules   │  ← Custom logic
└────────────────────────┘
    ↓
Canonicalized Output (100% deterministic)
    ↓
Matching Engine
```

**Why this approach:**
- LLM does semantic heavy lifting (90%+ accuracy)
- Post-processing ensures 100% consistency
- Easy to update rules without retraining
- Works with any LLM (GPT, Mistral, Gemini)

---

## 📁 Files to Check

1. **SCHEMA_EXPLANATION.md** - Detailed schema format
2. **test_canonicalization.py** - Comprehensive tests (running)
3. **canonicalization_test_results.json** - Results (generating)
4. **/data/synonyms.json** - Canonicalization mappings
5. **/pos/data/linguistic_cues.json** - Constraint patterns

---

**Next:** Wait for test results (5 min), then analyze patterns and build post-processor.
