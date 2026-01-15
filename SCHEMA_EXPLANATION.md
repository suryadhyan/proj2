# 📋 Schema Format Explanation

## ❓ What is "NEW Schema" vs "OLD Schema"?

**TLDR: There's only ONE schema, but test examples had inconsistencies.**

---

## 🔍 The Actual Schema Format

Looking at `PROMPT_STAGE2.txt`, the current schema uses **TWO different structures** for different purposes:

### **1. For ITEMS (Products/Services)**

Uses **flat `categorical` structure**:

```json
{
  "items": [
    {
      "type": "laptop",
      "categorical": {              // ← Flat key-value pairs
        "brand": "apple",
        "model": "macbook pro",
        "condition": "used"
      },
      "min": {
        "capacity": [
          {"type": "memory", "value": 16, "unit": "gb"}
        ]
      },
      "max": {
        "cost": [
          {"type": "price", "value": 80000, "unit": "inr"}
        ]
      },
      "range": {}
    }
  ]
}
```

**Why flat?** Items have finite, well-known attributes (brand, model, condition, color, etc.)

---

### **2. For PEOPLE (Preferences/Attributes)**

Uses **axis-based structure**:

```json
{
  "other_party_preferences": {
    "identity": [                   // ← Array of identity attributes
      {"type": "language", "value": "kannada"},
      {"type": "profession", "value": "plumber"},
      {"type": "gender", "value": "female"}
    ],
    "habits": {                     // ← Flat for binary flags
      "smoking": "no",
      "drinking": "no",
      "pets": "yes"
    },
    "min": {
      "time": [
        {"type": "experience", "value": 60, "unit": "months"}
      ]
    },
    "max": {},
    "range": {}
  },

  "self_attributes": {
    "identity": [
      {"type": "profession", "value": "graphic designer"}
    ],
    "min": {
      "time": [
        {"type": "experience", "value": 60, "unit": "months"}
      ]
    },
    "max": {},
    "range": {}
  }
}
```

**Why axis-based?** People have diverse, unbounded attributes that need flexible categorization:
- **identity**: gender, age, profession, language, nationality, etc.
- **habits**: smoking, drinking, diet, pets, lifestyle, etc.
- **capacity**: team size, number of people, etc.
- **time**: experience, availability, age, etc.
- **space**: location preferences, travel distance, etc.

---

## 🔑 The 10 Axes (For Numeric Constraints)

When extracting numeric attributes, map to one of 10 axes:

1. **identity** - gender, profession, certification
2. **capacity** - RAM, storage, rooms, seats, team size
3. **performance** - speed, odometer, mileage, refresh rate
4. **quality** - rating, grade, condition level
5. **quantity** - count, number, amount
6. **time** - age, experience, duration, availability
7. **space** - area, distance, dimensions
8. **cost** - price, budget, salary, fees
9. **mode** - delivery mode, service mode, work type
10. **skill** - certifications, proficiency level

---

## 📊 Why This Confusion?

The test examples in `stage3_extraction1.json` had **inconsistencies**:

**Test Example (WRONG):**
```json
{
  "other_party_preferences": {
    "categorical": {"language": "kannada"}  // ← Wrong! Should use identity axis
  }
}
```

**Correct Format:**
```json
{
  "other_party_preferences": {
    "identity": [
      {"type": "language", "value": "kannada"}
    ]
  }
}
```

**Why?** The test examples were created before the axis-based structure was finalized.

---

## 🔄 Preprocessing vs Post-processing

### **Preprocessing** (Before API Call)
- User query → Clean text
- Remove emojis, special chars
- Handle typos? (Usually not needed)

### **LLM Extraction** (API Call)
```
Raw Query → [GPT-4o API] → Structured JSON
```

### **Post-processing** (After API Call)
1. **Schema Validation**
   - Ensure all 14 fields present
   - Add missing empty objects `{}`

2. **Canonicalization** (Using `/data/synonyms.json`)
   ```python
   "phone" → "smartphone"
   "mobile" → "smartphone"
   "cellphone" → "smartphone"
   "notebook" → "laptop"
   ```

3. **Domain Normalization** (Using `/data/taxonomy.json`)
   ```python
   "technology & electronics" → "Technology & Electronics"
   ```

4. **Currency Detection** (Using `/pos/data/currencies.json`)
   ```python
   "unit": "local" → infer from context → "unit": "INR"
   "₹5 lakh" → {"value": 500000, "currency": "INR"}
   ```

5. **Constraint Detection** (Using `/pos/data/linguistic_cues.json`)
   ```python
   "under 50k" → max: 50000
   "at least 16gb" → min: 16
   "around 60k" → range: {min: 55000, max: 65000} (±10%)
   ```

6. **Implication Rules**
   ```python
   "single owner" → condition: "used" + ownership: "single"
   "first owner" → condition: "used" + ownership: "first"
   "sealed box" → condition: "new" + packaging: "sealed"
   ```

---

## 🎯 Full Pipeline

```
User Query: "need kannada speaking plumber with 5 years experience"
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING (Optional)                                 │
│    - Text cleaning                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. LLM EXTRACTION (GPT-4o API or Fine-tuned Mistral)       │
│    Raw JSON output with axis-based structure                │
└─────────────────────────────────────────────────────────────┘
    ↓
    Output:
    {
      "intent": "service",
      "subintent": "seek",
      "domain": ["construction & trades"],
      "items": [{"type": "plumbing"}],
      "other_party_preferences": {
        "identity": [
          {"type": "language", "value": "kannada"}
        ],
        "min": {
          "time": [
            {"type": "experience", "value": 60, "unit": "month"}
          ]
        }
      }
    }
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. POST-PROCESSING                                          │
│    a. Schema Validation (14 fields check)                   │
│    b. Canonicalization (plumbing → plumbing ✓)             │
│    c. Domain normalization (construction & trades →         │
│       Construction & Trades)                                │
│    d. Unit normalization (month → months)                   │
└─────────────────────────────────────────────────────────────┘
    ↓
    Final Output:
    {
      "intent": "service",
      "subintent": "seek",
      "domain": ["Construction & Trades"],  // ← Canonicalized
      "items": [{"type": "plumbing"}],
      "other_party_preferences": {
        "identity": [
          {"type": "language", "value": "kannada"}
        ],
        "min": {
          "time": [
            {"type": "experience", "value": 60, "unit": "months"}  // ← Normalized
          ]
        }
      },
      // ... all 14 fields present
    }
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MATCHING ENGINE                                          │
│    Use canonicalized output for SQL/vector search           │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ So What Do We Need?

### **The API Already Does:**
- ✅ Semantic understanding (intent, subintent)
- ✅ Attribute extraction (brands, models, constraints)
- ✅ Polysemy resolution (language context-based)
- ✅ Basic canonicalization (iPhone → smartphone, most of the time)
- ✅ Constraint detection (at least, under, between)

### **Post-Processing Adds:**
- 🔧 100% consistent canonicalization (phone/mobile/cellphone → smartphone)
- 🔧 Domain case normalization
- 🔧 Currency detection (local → INR/USD)
- 🔧 Schema validation (all 14 fields)
- 🔧 Unit normalization (month → months)
- 🔧 Implication rules enforcement (single owner → used + ownership)

---

## 🎯 Summary

**There's only ONE schema format:**
- Items use `categorical: {key: value}`
- People use axis-based `identity/habits/etc`

**Test failure was because:**
- Test examples used old/incorrect format
- API is using the correct current format

**Solution:**
- Update test examples to match current schema
- Add post-processing for canonicalization
- Schema itself is correct!

---

**Generated:** 2026-01-15
