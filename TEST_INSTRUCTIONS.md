# 🧪 API Extraction Testing Guide

## 📋 Overview

Test the GPT API extraction against expected outputs from `stage3_extraction1.json`.

**What we're testing:**
- Query → Structured JSON (14 fields)
- Comparing actual output vs expected output
- Measuring accuracy and identifying issues

---

## 🚀 Quick Start

### 1. Setup API Key

Create `.env` file (copy from `.env.template`):
```bash
cp .env.template .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

### 2. Install Dependencies

```bash
pip install openai python-dotenv
```

---

## 🧪 Testing Approaches

### **Option 1: Test Single Query** (Recommended for debugging)

Test one query at a time:

```bash
python test_single_query.py
```

Or test custom query:
```bash
python test_single_query.py "need a yoga instructor in koramangala"
```

**Output:**
- Extracted JSON
- Token usage statistics
- Immediate feedback

---

### **Option 2: Full Test Suite - Single Stage**

One API call for full extraction:

```bash
python test_extraction_api.py single
```

**How it works:**
- Loads all 10 test queries
- One API call per query (full extraction)
- Compares actual vs expected
- Generates `test_results_single.json`

**Pros:**
- ✅ Simpler (one API call)
- ✅ Faster execution
- ✅ Lower latency

**Cons:**
- ⚠️ Long prompt (~100K chars)
- ⚠️ Might miss nuances

---

### **Option 3: Full Test Suite - Two Stage**

Two API calls: Classification → Extraction:

```bash
python test_extraction_api.py two-stage
```

**How it works:**
1. **Stage 2 API call**: Classification only (intent, subintent, domain)
2. **Stage 3 API call**: Full extraction using Stage 2 context

**Pros:**
- ✅ Cleaner separation
- ✅ Easier debugging (know which stage fails)
- ✅ Matches production pipeline

**Cons:**
- ⚠️ Two API calls (higher cost)
- ⚠️ Higher latency

---

## 📊 Output Files

After running tests, you'll get:

### `test_results_single.json`
```json
[
  {
    "query": "looking for a used macbook...",
    "success": true,
    "actual": { /* extracted output */ },
    "differences": []
  },
  ...
]
```

### Console Output
```
🧪 EXTRACTION API TEST SUITE (SINGLE approach)
================================================================================

📂 Loading files...
✅ Loaded prompt (300000 chars)
✅ Loaded 10 test examples

Test 1/10: looking for a used macbook pro with at least 16gb ram...
================================================================================
🔹 Single API call for full extraction...
✅ PASS - Output matches expected

...

📊 TEST SUMMARY
================================================================================
✅ Passed: 8/10
❌ Failed: 2/10
📈 Success Rate: 80.0%

💾 Results saved to: test_results_single.json
```

---

## 🔍 Understanding Results

### ✅ PASS
- All 13 fields match (excluding reasoning)
- Reasoning field skipped (non-deterministic)

### ❌ FAIL
- Shows which fields differ
- Expected vs Actual comparison
- Common issues:
  - Item type canonicalization (laptop vs notebook)
  - Attribute placement (min vs range)
  - Domain selection differences

---

## 🐛 Debugging Failed Tests

If test fails, check:

1. **Field Differences**
   ```
   ❌ FAIL - Differences found:
      • items: value_mismatch
        Expected: [{"type": "laptop", ...}]
        Actual: [{"type": "notebook", ...}]
   ```

2. **Common Issues:**
   - **Item type:** "laptop" vs "notebook" → Need canonicalization
   - **Constraint type:** Using `min` instead of `range` for exact values
   - **Domain:** "Technology & Electronics" vs "IT Services"
   - **Missing fields:** Check if API returned all 14 fields

3. **Solutions:**
   - Add examples to prompt
   - Use post-canonicalization layer (synonyms.json)
   - Adjust prompt instructions for specific issue

---

## 💰 Cost Estimation

**GPT-4o Pricing (Nov 2024):**
- Prompt: $2.50 / 1M tokens
- Completion: $10.00 / 1M tokens

**Estimated cost per query:**
- Prompt: ~100K tokens ≈ $0.25
- Completion: ~500 tokens ≈ $0.005
- **Total: ~$0.255 per query**

**Full test suite (10 queries):**
- Single stage: ~$2.55
- Two stage: ~$5.10 (double calls)

---

## 📈 Next Steps

After testing:

1. **Analyze results** → Identify patterns in failures
2. **Iterate on prompt** → Fix common issues
3. **Add post-processing** → Canonicalization layer
4. **Fine-tune Mistral** → Use API-generated data for training
5. **Deploy on Azure** → Production-ready model

---

## 🎯 Success Metrics

**Target:**
- ✅ 90%+ field-level accuracy
- ✅ 100% schema compliance (all 14 fields present)
- ✅ Deterministic canonicalization (post-processing)

**Current challenges:**
- Item type variations (laptop/notebook/computer)
- Constraint type selection (min/max/range)
- Domain ambiguity (multi-domain queries)

**Solutions available:**
- `/data/synonyms.json` → Canonical forms
- `/pos/data/linguistic_cues.json` → Constraint detection
- `/pos/data/attributes_schema.json` → Attribute classification

---

## 📞 Troubleshooting

### API Key Issues
```
❌ ERROR: OPENAI_API_KEY not found
```
→ Check `.env` file exists and has correct key

### Import Errors
```
ModuleNotFoundError: No module named 'openai'
```
→ Run: `pip install openai python-dotenv`

### JSON Decode Errors
```
JSONDecodeError: Expecting value: line 1 column 1
```
→ API returned non-JSON. Check model output format setting.

### Rate Limits
```
RateLimitError: Rate limit reached
```
→ Add delay between requests or upgrade API plan

---

## 📝 Test Examples

The 10 test queries cover:

1. ✅ Product buy (used laptop with constraints)
2. ✅ Product sell (motorcycle with ownership)
3. ✅ Service seek (plumber with language preference)
4. ✅ Service provide (graphic designer with experience)
5. ✅ Mutual adventure (trekking buddy)
6. ✅ Mutual roommate (apartment with preferences)
7. ✅ Product free (giving away sofa)
8. ✅ Service education (math tutor)
9. ✅ Mutual professional (cofounder search)
10. ✅ Product simple (iPhone query)

---

## 🔄 Iterative Testing Workflow

```
1. Run test_single_query.py with one example
   ↓
2. Inspect output, identify issues
   ↓
3. Adjust prompt or add examples
   ↓
4. Run full test suite
   ↓
5. Analyze success rate
   ↓
6. If < 90%: Iterate on prompt
   If ≥ 90%: Add post-processing layer
   ↓
7. Generate training data for fine-tuning
```

---

Happy testing! 🚀
