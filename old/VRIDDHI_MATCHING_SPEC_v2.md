# VRIDDHI Matching System - Complete Technical Specification v2.0

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Schema Specification](#2-schema-specification)
3. [Constraint Modes](#3-constraint-modes)
4. [Intent Matching Rules](#4-intent-matching-rules)
5. [Field Matching Logic](#5-field-matching-logic)
6. [Complete Matching Examples](#6-complete-matching-examples)
7. [SQL Implementation](#7-sql-implementation)
8. [Vector Strategy](#8-vector-strategy)
9. [Matching Algorithm](#9-matching-algorithm)
10. [Edge Cases & Handling](#10-edge-cases--handling)

---

## 1. System Overview

### 1.1 Purpose
VRIDDHI is a semantic reasoning platform that connects users across three primary intents:
- **Product**: Buy ↔ Sell transactions
- **Service**: Seek ↔ Provide engagements
- **Mutual**: Connect ↔ Connect relationships

### 1.2 Technology Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Structured Data | PostgreSQL (Supabase) | Hard filters, SQL matching (~90%) |
| Vector Store | Qdrant | Semantic search, multi-vector support |
| Sparse Embeddings | BM-25 (1024D) | Keyword matching |
| Dense Embeddings | Sentence Transformers | Semantic similarity |
| Late Interaction | ColBERT | Token-level matching |
| Query Processor | Fine-tuned Mistral 7B | Classification, extraction |

### 1.3 Matching Philosophy
```
STRICT MATCHING ONLY
- No fallbacks
- No "prefer" modes
- No superset suggestions
- User gets exactly what they ask for, or no match
```

---

## 2. Schema Specification

### 2.1 Complete Schema (14 Fields)

```json
{
  "intent": "product | service | mutual",
  "subintent": "buy | sell | seek | provide | connect",
  "domain": ["array of domain strings"],
  "category": ["array of mutual categories - only for mutual intent"],

  "items": [{
    "type": "item type string",
    "categorical": {
      "key": "value"
    },
    "min": {
      "key": numeric_value
    },
    "max": {
      "key": numeric_value
    },
    "range": {
      "key": [min_value, max_value]
    }
  }],
  "itemexclusions": ["excluded item types or attributes"],

  "other": {
    "categorical": {"key": "value"},
    "min": {"key": numeric_value},
    "max": {"key": numeric_value},
    "range": {"key": [min, max]}
  },
  "otherexclusions": ["excluded other party attributes"],

  "self": {
    "categorical": {"key": "value"},
    "min": {"key": numeric_value},
    "max": {"key": numeric_value},
    "range": {"key": [min, max]}
  },
  "selfexclusions": [],

  "location": "location string" | {"origin": "X", "destination": "Y"},
  "locationmode": "proximity | target | route | flexible",
  "locationexclusions": ["excluded locations"],

  "reasoning": "Chain of Thought trace"
}
```

### 2.2 Field Definitions

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `intent` | enum | Primary intent category | Yes |
| `subintent` | enum | Action within intent | Yes |
| `domain` | array | Domain categories (max 21 product, 18 service) | Yes |
| `category` | array | Mutual categories (25 options) | Only for mutual |
| `items` | array | Item specifications with constraints | Yes for product/service |
| `itemexclusions` | array | Rejected items/attributes | No |
| `other` | object | What user wants in other party | Yes |
| `otherexclusions` | array | Rejected other party attributes | No |
| `self` | object | User's self-description | Yes |
| `selfexclusions` | array | User's own exclusions | No |
| `location` | string/object | Target location or route | Yes |
| `locationmode` | enum | How to match location | Yes |
| `locationexclusions` | array | Locations to avoid | No |
| `reasoning` | string | CoT explanation | Yes |

### 2.3 Constraint Object Structure

Every constraint object (in items, other, self) has 4 sub-objects:

```json
{
  "categorical": {},  // Exact string matches
  "min": {},          // Lower bound constraints
  "max": {},          // Upper bound constraints
  "range": {}         // Bounded range (includes EXACT as [x, x])
}
```

---

## 3. Constraint Modes

### 3.1 The Three Modes (ONLY THREE)

| Mode | Purpose | Storage Format | Matching Logic |
|------|---------|----------------|----------------|
| **min** | Lower bound | `"key": value` | B.value >= A.min |
| **max** | Upper bound | `"key": value` | B.value <= A.max |
| **range** | Bounded range | `"key": [min, max]` | A.range[0] <= B.value <= A.range[1] |

### 3.2 Critical Invariant: EXACT = range[x, x]

```
EXACT is NOT a separate mode.
EXACT is represented as: range: {"key": [x, x]}

Example: "exactly 256GB storage"
Storage: range: {"storage": [256, 256]}
NOT: exact: {"storage": 256}  // WRONG - no "exact" mode exists
```

### 3.3 Signal Word Detection

| Signal Words | Extracted Mode | Example Input | Storage |
|--------------|----------------|---------------|---------|
| (no qualifier) | range[x,x] | "256GB storage" | `range: {"storage": [256, 256]}` |
| "exactly", "only" | range[x,x] | "exactly 8GB RAM" | `range: {"ram": [8, 8]}` |
| "under", "below", "max", "budget", "upto", "less than" | max | "under 50k" | `max: {"price": 50000}` |
| "minimum", "at least", "above", "more than", "over" | min | "minimum 3 years exp" | `min: {"experience": 36}` |
| "between X and Y", "X to Y", "X-Y" | range | "between 50k-80k" | `range: {"price": [50000, 80000]}` |

### 3.4 Standard Unit Conversions

| Dimension | Standard Unit | Conversion Examples |
|-----------|---------------|---------------------|
| Time | Months | 5 years = 60 months |
| Storage | GB | 1 TB = 1024 GB |
| Distance | KM | 5 miles ≈ 8 KM |
| Area | SQM | 1000 sqft ≈ 93 SQM |
| Currency | Base unit | 50k = 50000, 1.5L = 150000 |
| Skill Level | 1-5 scale | beginner=1, expert=5 |

---

## 4. Intent Matching Rules

### 4.1 Intent-SubIntent Pairing

| Intent | Valid SubIntents | Matching Rule |
|--------|------------------|---------------|
| **Product** | buy, sell | buy ↔ sell (INVERSE) |
| **Service** | seek, provide | seek ↔ provide (INVERSE) |
| **Mutual** | connect | connect = connect (SAME) |

### 4.2 Matching Direction

```
PRODUCT & SERVICE:
┌─────────────────────────────────────────────────────────────┐
│  User A (Buyer/Seeker)          User B (Seller/Provider)    │
│  subintent: buy/seek            subintent: sell/provide     │
│                                                             │
│  A.other ─────────────────────► B.self                      │
│  (what A wants in B)            (what B is)                 │
│                                                             │
│  A.self ◄───────────────────── B.other                      │
│  (what A is)                    (what B wants in A)         │
└─────────────────────────────────────────────────────────────┘

MUTUAL:
┌─────────────────────────────────────────────────────────────┐
│  User A (Connect)               User B (Connect)            │
│  subintent: connect             subintent: connect          │
│                                                             │
│  A.other ─────────────────────► B.self    ✓ Must pass       │
│  A.self ◄───────────────────── B.other    ✓ Must pass       │
│                                                             │
│  BIDIRECTIONAL: Both directions must match                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Field Matching Logic

### 5.1 Complete Field Matching Matrix

#### For PRODUCT & SERVICE Intent:

| A's Field | Operator | B's Field | Logic |
|-----------|----------|-----------|-------|
| `intent` | = | `intent` | Must be identical |
| `subintent` | ≠ | `subintent` | Must be INVERSE (buy↔sell, seek↔provide) |
| `domain` | ∩≠∅ | `domain` | Must have at least one common domain |
| `items[].type` | = | `items[].type` | Must match item type |
| `items[].categorical` | ⊆ | `items[].categorical` | A's requirements ⊆ B's offerings |
| `items[].min` | ≤ | `items[].*` | B's value ≥ A's minimum |
| `items[].max` | ≥ | `items[].*` | B's value ≤ A's maximum |
| `items[].range` | ∋ | `items[].*` | B's value within A's range |
| `itemexclusions` | ∩=∅ | `items[].*` | No overlap (disjoint) |
| `other.categorical` | ⊆ | `self.categorical` | A's requirements ⊆ B's attributes |
| `other.min` | ≤ | `self.*` | B's value ≥ A's minimum |
| `other.max` | ≥ | `self.*` | B's value ≤ A's maximum |
| `other.range` | ∋ | `self.*` | B's value within A's range |
| `otherexclusions` | ∩=∅ | `self.*` | No overlap (disjoint) |
| `location` | ~ | `location` | Based on locationmode |
| `locationexclusions` | ∩=∅ | `location` | B's location not in A's exclusions |

#### Reverse Check (B → A):

| B's Field | Operator | A's Field | Logic |
|-----------|----------|-----------|-------|
| `other.categorical` | ⊆ | `self.categorical` | B's requirements ⊆ A's attributes |
| `other.min` | ≤ | `self.*` | A's value ≥ B's minimum |
| `other.max` | ≥ | `self.*` | A's value ≤ B's maximum |
| `other.range` | ∋ | `self.*` | A's value within B's range |
| `otherexclusions` | ∩=∅ | `self.*` | No overlap (disjoint) |

### 5.2 Categorical Matching Rules

```
EXACT MATCH:
A.categorical.brand = "apple"
B.categorical.brand = "apple"
Result: MATCH ✓

A.categorical.brand = "apple"
B.categorical.brand = "samsung"
Result: NO MATCH ✗

SUBSET LOGIC (A's requirements ⊆ B's offerings):
A.categorical = {brand: "apple"}
B.categorical = {brand: "apple", color: "black", storage: "256gb"}
Result: MATCH ✓ (A's single requirement satisfied by B)

A.categorical = {brand: "apple", color: "white"}
B.categorical = {brand: "apple", color: "black"}
Result: NO MATCH ✗ (color mismatch)
```

### 5.3 Numeric Matching Rules

```python
# MIN constraint: B must meet or exceed A's minimum
A.min.experience = 36  # months
B.self.experience = 48  # months
# Check: 48 >= 36 → MATCH ✓

# MAX constraint: B must not exceed A's maximum
A.max.price = 50000
B.items.price = 45000
# Check: 45000 <= 50000 → MATCH ✓

# RANGE constraint: B must be within A's range (inclusive)
A.range.price = [40000, 60000]
B.items.price = 55000
# Check: 40000 <= 55000 <= 60000 → MATCH ✓

# EXACT (as range[x,x]): B must match exactly
A.range.storage = [256, 256]  # EXACT 256GB
B.items.storage = 256
# Check: 256 <= 256 <= 256 → MATCH ✓

B.items.storage = 512
# Check: 256 <= 512 <= 256 → NO MATCH ✗
```

### 5.4 Exclusion Matching Rules

```
EXCLUSIONS ARE STRICT - DISJOINT SETS REQUIRED

A.itemexclusions = ["refurbished", "used"]
B.items.condition = "refurbished"
Result: NO MATCH ✗ (B's condition in A's exclusions)

A.otherexclusions = ["agent", "dealer"]
B.self.type = "individual"
Result: MATCH ✓ (B's type not in A's exclusions)

A.locationexclusions = ["chennai", "delhi"]
B.location = "bangalore"
Result: MATCH ✓ (B's location not in A's exclusions)
```

---

## 6. Complete Matching Examples

### 6.1 PRODUCT Intent Example (Buy ↔ Sell)

#### User A (Buyer) Input:
```
"Looking for iPhone 15 Pro, black color, 256GB storage,
under 1 lakh budget. Seller should be verified with
minimum 4.0 rating. No dealers or agents.
Located in Bangalore, avoid Chennai sellers."
```

#### User A Extracted Schema:
```json
{
  "intent": "product",
  "subintent": "buy",
  "domain": ["electronics"],
  "category": [],

  "items": [{
    "type": "smartphone",
    "categorical": {
      "brand": "apple",
      "model": "iphone 15 pro",
      "color": "black"
    },
    "min": {},
    "max": {
      "price": 100000
    },
    "range": {
      "storage": [256, 256]
    }
  }],
  "itemexclusions": ["refurbished", "used"],

  "other": {
    "categorical": {
      "verified": true
    },
    "min": {
      "rating": 4.0
    },
    "max": {},
    "range": {}
  },
  "otherexclusions": ["dealer", "agent"],

  "self": {
    "categorical": {
      "payment": "cash"
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "selfexclusions": [],

  "location": "bangalore",
  "locationmode": "proximity",
  "locationexclusions": ["chennai"],

  "reasoning": "User wants to buy iPhone 15 Pro with specific specs..."
}
```

#### User B (Seller) Input:
```
"Selling my iPhone 15 Pro, titanium black, 256GB,
excellent condition, asking 95000. I'm a verified individual
seller with 4.5 rating. Based in Bangalore.
Prefer cash buyers, no EMI requests."
```

#### User B Extracted Schema:
```json
{
  "intent": "product",
  "subintent": "sell",
  "domain": ["electronics"],
  "category": [],

  "items": [{
    "type": "smartphone",
    "categorical": {
      "brand": "apple",
      "model": "iphone 15 pro",
      "color": "black",
      "condition": "excellent"
    },
    "min": {},
    "max": {},
    "range": {
      "price": [95000, 95000],
      "storage": [256, 256]
    }
  }],
  "itemexclusions": [],

  "other": {
    "categorical": {
      "payment": "cash"
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "otherexclusions": ["emi"],

  "self": {
    "categorical": {
      "verified": true,
      "type": "individual"
    },
    "min": {},
    "max": {},
    "range": {
      "rating": [4.5, 4.5]
    }
  },
  "selfexclusions": [],

  "location": "bangalore",
  "locationmode": "proximity",
  "locationexclusions": [],

  "reasoning": "User is selling their iPhone 15 Pro..."
}
```

#### Step-by-Step Matching:

```
STEP 1: Intent Check
A.intent = "product" = B.intent = "product"
Result: ✓ PASS

STEP 2: SubIntent Check (INVERSE required)
A.subintent = "buy" ≠ B.subintent = "sell"
Result: ✓ PASS (correctly inverse)

STEP 3: Domain Check
A.domain ∩ B.domain = ["electronics"] ∩ ["electronics"] = ["electronics"]
Result: ✓ PASS (non-empty intersection)

STEP 4: Items Matching (A.items → B.items)

  4a. Type Match:
  A.items[0].type = "smartphone" = B.items[0].type = "smartphone"
  Result: ✓ PASS

  4b. Categorical Match (A ⊆ B):
  A.categorical = {brand: "apple", model: "iphone 15 pro", color: "black"}
  B.categorical = {brand: "apple", model: "iphone 15 pro", color: "black", condition: "excellent"}
  Check: A ⊆ B? Yes, all A's keys exist in B with matching values
  Result: ✓ PASS

  4c. Max Constraint:
  A.max.price = 100000
  B's price = 95000 (from B.range.price[0])
  Check: 95000 <= 100000
  Result: ✓ PASS

  4d. Range Constraint (EXACT):
  A.range.storage = [256, 256]
  B.range.storage = [256, 256], so B's storage = 256
  Check: 256 <= 256 <= 256
  Result: ✓ PASS

STEP 5: Item Exclusions Check
A.itemexclusions = ["refurbished", "used"]
B.items[0].categorical.condition = "excellent"
Check: "excellent" ∉ ["refurbished", "used"]
Result: ✓ PASS

STEP 6: Other → Self Matching (A.other → B.self)

  6a. Categorical:
  A.other.categorical = {verified: true}
  B.self.categorical = {verified: true, type: "individual"}
  Check: A ⊆ B? Yes
  Result: ✓ PASS

  6b. Min Constraint:
  A.other.min.rating = 4.0
  B.self.range.rating = [4.5, 4.5], so B's rating = 4.5
  Check: 4.5 >= 4.0
  Result: ✓ PASS

STEP 7: Other Exclusions Check
A.otherexclusions = ["dealer", "agent"]
B.self.categorical.type = "individual"
Check: "individual" ∉ ["dealer", "agent"]
Result: ✓ PASS

STEP 8: Reverse Check (B.other → A.self)

  8a. Categorical:
  B.other.categorical = {payment: "cash"}
  A.self.categorical = {payment: "cash"}
  Check: B ⊆ A? Yes
  Result: ✓ PASS

STEP 9: Reverse Exclusions Check
B.otherexclusions = ["emi"]
A.self.categorical.payment = "cash"
Check: "cash" ∉ ["emi"]
Result: ✓ PASS

STEP 10: Location Check
A.location = "bangalore"
B.location = "bangalore"
A.locationmode = "proximity"
Check: Same location
Result: ✓ PASS

STEP 11: Location Exclusions Check
A.locationexclusions = ["chennai"]
B.location = "bangalore"
Check: "bangalore" ∉ ["chennai"]
Result: ✓ PASS

═══════════════════════════════════════════
FINAL RESULT: ✓ MATCH
All 11 steps passed. A and B are compatible.
═══════════════════════════════════════════
```

---

### 6.2 SERVICE Intent Example (Seek ↔ Provide)

#### User A (Seeker) Input:
```
"Need a Kannada-speaking yoga instructor for home sessions,
minimum 5 years experience, female preferred.
Budget max 2000 per session. Available weekends.
Jayanagar area, Bangalore."
```

#### User A Extracted Schema:
```json
{
  "intent": "service",
  "subintent": "seek",
  "domain": ["fitness", "wellness"],
  "category": [],

  "items": [{
    "type": "yoga_instruction",
    "categorical": {
      "mode": "home_visit",
      "availability": "weekends"
    },
    "min": {},
    "max": {
      "price_per_session": 2000
    },
    "range": {}
  }],
  "itemexclusions": ["online", "group_class"],

  "other": {
    "categorical": {
      "language": "kannada",
      "gender": "female"
    },
    "min": {
      "experience": 60
    },
    "max": {},
    "range": {}
  },
  "otherexclusions": [],

  "self": {
    "categorical": {
      "availability": "weekends"
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "selfexclusions": [],

  "location": "jayanagar, bangalore",
  "locationmode": "proximity",
  "locationexclusions": [],

  "reasoning": "User seeking home yoga instruction with specific requirements..."
}
```

#### User B (Provider) Input:
```
"Certified yoga instructor offering home sessions.
7 years experience. Fluent in Kannada and English.
Female. Charge 1500 per session. Available all days.
Serve Jayanagar, Koramangala, BTM areas."
```

#### User B Extracted Schema:
```json
{
  "intent": "service",
  "subintent": "provide",
  "domain": ["fitness", "wellness"],
  "category": [],

  "items": [{
    "type": "yoga_instruction",
    "categorical": {
      "mode": "home_visit",
      "certified": true
    },
    "min": {},
    "max": {},
    "range": {
      "price_per_session": [1500, 1500]
    }
  }],
  "itemexclusions": [],

  "other": {
    "categorical": {},
    "min": {},
    "max": {},
    "range": {}
  },
  "otherexclusions": [],

  "self": {
    "categorical": {
      "language": "kannada",
      "gender": "female",
      "availability": "all_days"
    },
    "min": {},
    "max": {},
    "range": {
      "experience": [84, 84]
    }
  },
  "selfexclusions": [],

  "location": "jayanagar, koramangala, btm",
  "locationmode": "target",
  "locationexclusions": [],

  "reasoning": "User is a yoga instructor offering home sessions..."
}
```

#### Matching Result:
```
STEP 1: intent = intent ✓
STEP 2: seek ≠ provide ✓ (inverse)
STEP 3: domain intersection ✓
STEP 4: items.type match ✓
STEP 5: items.mode = "home_visit" ✓
STEP 6: A.max.price_per_session(2000) >= B.price(1500) ✓
STEP 7: A.other.language = B.self.language ("kannada") ✓
STEP 8: A.other.gender = B.self.gender ("female") ✓
STEP 9: A.other.min.experience(60) <= B.self.experience(84) ✓
STEP 10: B.self.availability("all_days") covers A.self.availability("weekends") ✓
STEP 11: location overlap (jayanagar) ✓

FINAL RESULT: ✓ MATCH
```

---

### 6.3 MUTUAL Intent Example (Connect ↔ Connect)

#### User A Input:
```
"Looking for a roommate in Koramangala. I'm a 25-year-old
female software engineer, vegetarian, non-smoker.
Looking for female roommate, preferably vegetarian,
no smoking/drinking. Budget 15-20k for rent share."
```

#### User A Extracted Schema:
```json
{
  "intent": "mutual",
  "subintent": "connect",
  "domain": [],
  "category": ["roommate", "housing"],

  "items": [{
    "type": "shared_accommodation",
    "categorical": {},
    "min": {},
    "max": {},
    "range": {
      "rent_share": [15000, 20000]
    }
  }],
  "itemexclusions": [],

  "other": {
    "categorical": {
      "gender": "female",
      "diet": "vegetarian",
      "smoking": false,
      "drinking": false
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "otherexclusions": ["smoker", "heavy_drinker"],

  "self": {
    "categorical": {
      "gender": "female",
      "age": 25,
      "profession": "software_engineer",
      "diet": "vegetarian",
      "smoking": false,
      "drinking": false
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "selfexclusions": [],

  "location": "koramangala, bangalore",
  "locationmode": "target",
  "locationexclusions": [],

  "reasoning": "User seeking female vegetarian roommate..."
}
```

#### User B Input:
```
"Female professional, 27, looking for roommate in
Koramangala or HSR. Vegetarian, no bad habits.
Prefer female roommate who is also vegetarian and
doesn't smoke. Budget around 18k."
```

#### User B Extracted Schema:
```json
{
  "intent": "mutual",
  "subintent": "connect",
  "domain": [],
  "category": ["roommate", "housing"],

  "items": [{
    "type": "shared_accommodation",
    "categorical": {},
    "min": {},
    "max": {},
    "range": {
      "rent_share": [18000, 18000]
    }
  }],
  "itemexclusions": [],

  "other": {
    "categorical": {
      "gender": "female",
      "diet": "vegetarian",
      "smoking": false
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "otherexclusions": ["smoker"],

  "self": {
    "categorical": {
      "gender": "female",
      "age": 27,
      "profession": "professional",
      "diet": "vegetarian",
      "smoking": false,
      "drinking": false
    },
    "min": {},
    "max": {},
    "range": {}
  },
  "selfexclusions": [],

  "location": "koramangala, hsr, bangalore",
  "locationmode": "target",
  "locationexclusions": [],

  "reasoning": "User seeking female vegetarian roommate..."
}
```

#### Bidirectional Matching:

```
═══════════════════════════════════════════
DIRECTION 1: A.other → B.self
═══════════════════════════════════════════

A.other.categorical.gender = "female"
B.self.categorical.gender = "female"
Result: ✓ MATCH

A.other.categorical.diet = "vegetarian"
B.self.categorical.diet = "vegetarian"
Result: ✓ MATCH

A.other.categorical.smoking = false
B.self.categorical.smoking = false
Result: ✓ MATCH

A.other.categorical.drinking = false
B.self.categorical.drinking = false
Result: ✓ MATCH

A.otherexclusions = ["smoker", "heavy_drinker"]
B.self attributes don't contain these
Result: ✓ PASS

Direction 1: ✓ ALL PASSED

═══════════════════════════════════════════
DIRECTION 2: B.other → A.self
═══════════════════════════════════════════

B.other.categorical.gender = "female"
A.self.categorical.gender = "female"
Result: ✓ MATCH

B.other.categorical.diet = "vegetarian"
A.self.categorical.diet = "vegetarian"
Result: ✓ MATCH

B.other.categorical.smoking = false
A.self.categorical.smoking = false
Result: ✓ MATCH

B.otherexclusions = ["smoker"]
A.self.smoking = false (not a smoker)
Result: ✓ PASS

Direction 2: ✓ ALL PASSED

═══════════════════════════════════════════
ADDITIONAL CHECKS
═══════════════════════════════════════════

Items (rent budget compatibility):
A.range.rent_share = [15000, 20000]
B.range.rent_share = [18000, 18000]
Check: 18000 within [15000, 20000]?
Result: ✓ COMPATIBLE

Location:
A.location includes "koramangala"
B.location includes "koramangala"
Result: ✓ OVERLAP

═══════════════════════════════════════════
FINAL RESULT: ✓ BIDIRECTIONAL MATCH
Both A→B and B→A passed all checks.
═══════════════════════════════════════════
```

---

## 7. SQL Implementation

### 7.1 Database Schema

```sql
-- Main listings table
CREATE TABLE listings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),

    -- Core fields
    intent VARCHAR(10) NOT NULL CHECK (intent IN ('product', 'service', 'mutual')),
    subintent VARCHAR(10) NOT NULL CHECK (subintent IN ('buy', 'sell', 'seek', 'provide', 'connect')),
    domain TEXT[] NOT NULL,
    category TEXT[],

    -- JSONB fields for flexible structure
    items JSONB NOT NULL DEFAULT '[]',
    itemexclusions TEXT[] DEFAULT '{}',

    other_preferences JSONB NOT NULL DEFAULT '{}',
    otherexclusions TEXT[] DEFAULT '{}',

    self_attributes JSONB NOT NULL DEFAULT '{}',
    selfexclusions TEXT[] DEFAULT '{}',

    -- Location
    location TEXT,
    location_coords GEOGRAPHY(POINT, 4326),
    locationmode VARCHAR(20) DEFAULT 'proximity',
    locationexclusions TEXT[] DEFAULT '{}',

    -- Metadata
    reasoning TEXT,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Indexes
    CONSTRAINT valid_subintent CHECK (
        (intent = 'product' AND subintent IN ('buy', 'sell')) OR
        (intent = 'service' AND subintent IN ('seek', 'provide')) OR
        (intent = 'mutual' AND subintent = 'connect')
    )
);

-- GIN indexes for JSONB queries
CREATE INDEX idx_listings_items ON listings USING GIN (items);
CREATE INDEX idx_listings_other ON listings USING GIN (other_preferences);
CREATE INDEX idx_listings_self ON listings USING GIN (self_attributes);

-- B-tree indexes for common queries
CREATE INDEX idx_listings_intent ON listings (intent, subintent);
CREATE INDEX idx_listings_domain ON listings USING GIN (domain);
CREATE INDEX idx_listings_status ON listings (status) WHERE status = 'active';

-- Spatial index for location queries
CREATE INDEX idx_listings_location ON listings USING GIST (location_coords);
```

### 7.2 Term Implications Table (Smart Matching)

```sql
-- For synonym/implication matching
CREATE TABLE term_implications (
    id SERIAL PRIMARY KEY,
    term VARCHAR(100) NOT NULL,
    implies VARCHAR(100) NOT NULL,
    domain VARCHAR(50),
    confidence FLOAT DEFAULT 1.0,
    bidirectional BOOLEAN DEFAULT FALSE,

    UNIQUE(term, implies, domain)
);

-- Example data
INSERT INTO term_implications (term, implies, domain, bidirectional) VALUES
('vegan', 'vegetarian', 'lifestyle', FALSE),
('vegetarian', 'no_beef', 'lifestyle', FALSE),
('non_smoker', 'no_smoking', 'lifestyle', TRUE),
('fitness', 'gym', 'interests', TRUE),
('teetotaler', 'no_drinking', 'lifestyle', FALSE),
('software_engineer', 'professional', 'profession', FALSE),
('excellent', 'good', 'condition', FALSE),
('new', 'excellent', 'condition', FALSE);

-- Function to check term implications
CREATE OR REPLACE FUNCTION terms_match(required TEXT, offered TEXT, check_domain TEXT DEFAULT NULL)
RETURNS BOOLEAN AS $$
BEGIN
    -- Direct match
    IF required = offered THEN
        RETURN TRUE;
    END IF;

    -- Check if offered implies required
    RETURN EXISTS (
        SELECT 1 FROM term_implications
        WHERE term = offered
        AND implies = required
        AND (check_domain IS NULL OR domain = check_domain OR domain IS NULL)
    );
END;
$$ LANGUAGE plpgsql;
```

### 7.3 Core Matching Functions

```sql
-- Check if numeric value satisfies constraints
CREATE OR REPLACE FUNCTION check_numeric_constraints(
    constraint_min JSONB,
    constraint_max JSONB,
    constraint_range JSONB,
    offered_value NUMERIC,
    key TEXT
) RETURNS BOOLEAN AS $$
DECLARE
    min_val NUMERIC;
    max_val NUMERIC;
    range_arr JSONB;
BEGIN
    -- Check MIN constraint
    IF constraint_min ? key THEN
        min_val := (constraint_min->>key)::NUMERIC;
        IF offered_value < min_val THEN
            RETURN FALSE;
        END IF;
    END IF;

    -- Check MAX constraint
    IF constraint_max ? key THEN
        max_val := (constraint_max->>key)::NUMERIC;
        IF offered_value > max_val THEN
            RETURN FALSE;
        END IF;
    END IF;

    -- Check RANGE constraint
    IF constraint_range ? key THEN
        range_arr := constraint_range->key;
        min_val := (range_arr->>0)::NUMERIC;
        max_val := (range_arr->>1)::NUMERIC;
        IF offered_value < min_val OR offered_value > max_val THEN
            RETURN FALSE;
        END IF;
    END IF;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Check categorical subset
CREATE OR REPLACE FUNCTION categorical_subset(required JSONB, offered JSONB)
RETURNS BOOLEAN AS $$
DECLARE
    key TEXT;
    req_val TEXT;
    off_val TEXT;
BEGIN
    FOR key, req_val IN SELECT * FROM jsonb_each_text(required)
    LOOP
        IF NOT offered ? key THEN
            RETURN FALSE;
        END IF;

        off_val := offered->>key;
        IF NOT terms_match(req_val, off_val) THEN
            RETURN FALSE;
        END IF;
    END LOOP;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Check exclusions (disjoint sets)
CREATE OR REPLACE FUNCTION check_exclusions(exclusions TEXT[], attributes JSONB)
RETURNS BOOLEAN AS $$
DECLARE
    excl TEXT;
    attr_val TEXT;
BEGIN
    FOREACH excl IN ARRAY exclusions
    LOOP
        FOR attr_val IN SELECT value FROM jsonb_each_text(attributes)
        LOOP
            IF terms_match(excl, attr_val) OR attr_val = excl THEN
                RETURN FALSE;
            END IF;
        END LOOP;
    END LOOP;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;
```

### 7.4 Main Matching Query

```sql
-- Find matches for a listing
CREATE OR REPLACE FUNCTION find_matches(
    listing_id UUID,
    max_results INT DEFAULT 50
) RETURNS TABLE (
    matched_id UUID,
    matched_user_id UUID,
    match_score FLOAT,
    match_details JSONB
) AS $$
DECLARE
    source RECORD;
    target_subintent TEXT;
BEGIN
    -- Get source listing
    SELECT * INTO source FROM listings WHERE id = listing_id;

    -- Determine target subintent
    CASE source.subintent
        WHEN 'buy' THEN target_subintent := 'sell';
        WHEN 'sell' THEN target_subintent := 'buy';
        WHEN 'seek' THEN target_subintent := 'provide';
        WHEN 'provide' THEN target_subintent := 'seek';
        WHEN 'connect' THEN target_subintent := 'connect';
    END CASE;

    RETURN QUERY
    SELECT
        l.id,
        l.user_id,
        1.0::FLOAT as match_score,
        jsonb_build_object(
            'intent_match', TRUE,
            'subintent_match', TRUE,
            'domain_overlap', source.domain && l.domain
        ) as match_details
    FROM listings l
    WHERE
        -- Basic filters
        l.status = 'active'
        AND l.id != listing_id
        AND l.user_id != source.user_id

        -- Intent matching
        AND l.intent = source.intent
        AND l.subintent = target_subintent

        -- Domain overlap
        AND l.domain && source.domain

        -- Location exclusions (source excludes target's location)
        AND NOT (source.locationexclusions && ARRAY[l.location])
        AND NOT (l.locationexclusions && ARRAY[source.location])

        -- Item exclusions check
        AND check_exclusions(source.itemexclusions, l.items->0)

        -- Other preferences → Self attributes (source.other ⊆ target.self)
        AND categorical_subset(
            source.other_preferences->'categorical',
            l.self_attributes->'categorical'
        )

        -- Other exclusions check
        AND check_exclusions(source.otherexclusions, l.self_attributes->'categorical')

        -- Reverse check (target.other ⊆ source.self)
        AND categorical_subset(
            l.other_preferences->'categorical',
            source.self_attributes->'categorical'
        )

        -- Reverse exclusions check
        AND check_exclusions(l.otherexclusions, source.self_attributes->'categorical')

    ORDER BY l.created_at DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;
```

---

## 8. Vector Strategy

### 8.1 Embedding Strategy by Intent

| Intent | Dense | Sparse (BM25) | ColBERT | Fields to Embed |
|--------|-------|---------------|---------|-----------------|
| **Product** | ✓ | ✓ | ✓ | preferences, soft_attrs, exclusions, reasoning |
| **Service** | ✓ | ✓ | ✓ | preferences, soft_attrs, exclusions, reasoning |
| **Mutual** | ✓ | ✗ | ✓ | original_input, preferences, keywords, ALL attrs, exclusions, reasoning, categories |

### 8.2 Why No Sparse for Mutual?

Mutual intent focuses on:
- Personality compatibility (semantic, not keyword)
- Lifestyle alignment (abstract concepts)
- Relationship goals (nuanced meanings)

BM25 keyword matching is less effective for these abstract concepts.

### 8.3 Qdrant Collection Configuration

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    CreateCollection, NamedVector
)

# Product/Service collection with multi-vectors
client.create_collection(
    collection_name="vriddhi_product_service",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
        "sparse": VectorParams(size=1024, distance=Distance.DOT),  # BM25
    },
    # ColBERT stored separately due to variable token count
)

# Mutual collection (no sparse)
client.create_collection(
    collection_name="vriddhi_mutual",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
    }
)

# ColBERT collection (multi-vector per document)
client.create_collection(
    collection_name="vriddhi_colbert",
    vectors_config={
        "token": VectorParams(size=128, distance=Distance.COSINE),
    },
    # Store token embeddings with metadata linking to main listing
)
```

### 8.4 Embedding Generation

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

# Dense embeddings
dense_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# ColBERT
colbert_tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
colbert_model = AutoModel.from_pretrained('colbert-ir/colbertv2.0')

def generate_embeddings(listing: dict) -> dict:
    """Generate all embeddings for a listing."""

    # Construct text for embedding
    text_parts = []

    # Add preferences
    if listing.get('other'):
        text_parts.append(f"Looking for: {json.dumps(listing['other'])}")

    # Add self attributes
    if listing.get('self'):
        text_parts.append(f"I am: {json.dumps(listing['self'])}")

    # Add exclusions
    if listing.get('otherexclusions'):
        text_parts.append(f"Avoid: {', '.join(listing['otherexclusions'])}")

    # Add reasoning
    if listing.get('reasoning'):
        text_parts.append(listing['reasoning'])

    full_text = " ".join(text_parts)

    # Dense embedding
    dense_embedding = dense_model.encode(full_text)

    # ColBERT token embeddings
    tokens = colbert_tokenizer(full_text, return_tensors='pt',
                                max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = colbert_model(**tokens)
        token_embeddings = outputs.last_hidden_state[0].numpy()

    return {
        'dense': dense_embedding.tolist(),
        'colbert_tokens': token_embeddings.tolist(),
        'text': full_text
    }
```

---

## 9. Matching Algorithm

### 9.1 Complete Matching Pipeline

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class MatchResult(Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    PARTIAL = "partial"

@dataclass
class MatchDetail:
    field: str
    passed: bool
    reason: str

@dataclass
class MatchOutput:
    result: MatchResult
    score: float
    details: List[MatchDetail]

def match_listings(a: dict, b: dict) -> MatchOutput:
    """
    Complete matching logic for two listings.
    Returns detailed match result.
    """
    details = []

    # Step 1: Intent check
    if a['intent'] != b['intent']:
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=[MatchDetail('intent', False, f"Intent mismatch: {a['intent']} vs {b['intent']}")]
        )
    details.append(MatchDetail('intent', True, 'Intent match'))

    # Step 2: SubIntent check
    valid_pairs = {
        ('buy', 'sell'), ('sell', 'buy'),
        ('seek', 'provide'), ('provide', 'seek'),
        ('connect', 'connect')
    }
    if (a['subintent'], b['subintent']) not in valid_pairs:
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=[MatchDetail('subintent', False, f"SubIntent incompatible: {a['subintent']} vs {b['subintent']}")]
        )
    details.append(MatchDetail('subintent', True, 'SubIntent compatible'))

    # Step 3: Domain overlap
    domain_overlap = set(a.get('domain', [])) & set(b.get('domain', []))
    if not domain_overlap and a['intent'] != 'mutual':
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=[MatchDetail('domain', False, 'No domain overlap')]
        )
    details.append(MatchDetail('domain', True, f'Domain overlap: {domain_overlap}'))

    # Step 4: Items matching (for product/service)
    if a['intent'] in ['product', 'service']:
        items_match = match_items(a.get('items', []), b.get('items', []))
        if not items_match.passed:
            return MatchOutput(
                result=MatchResult.NO_MATCH,
                score=0.0,
                details=details + [items_match]
            )
        details.append(items_match)

    # Step 5: Item exclusions
    if not check_exclusions_disjoint(a.get('itemexclusions', []), b.get('items', [])):
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [MatchDetail('itemexclusions', False, 'Item exclusion violated')]
        )
    details.append(MatchDetail('itemexclusions', True, 'No item exclusions violated'))

    # Step 6: A.other → B.self
    other_self_match = match_preferences(a.get('other', {}), b.get('self', {}))
    if not other_self_match.passed:
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [other_self_match]
        )
    details.append(other_self_match)

    # Step 7: A.otherexclusions check
    if not check_exclusions_disjoint(a.get('otherexclusions', []), b.get('self', {})):
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [MatchDetail('otherexclusions', False, 'Other exclusion violated')]
        )
    details.append(MatchDetail('otherexclusions', True, 'No other exclusions violated'))

    # Step 8: Reverse check B.other → A.self
    reverse_match = match_preferences(b.get('other', {}), a.get('self', {}))
    if not reverse_match.passed:
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [MatchDetail('reverse', False, f'Reverse check failed: {reverse_match.reason}')]
        )
    details.append(MatchDetail('reverse', True, 'Reverse check passed'))

    # Step 9: B.otherexclusions check
    if not check_exclusions_disjoint(b.get('otherexclusions', []), a.get('self', {})):
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [MatchDetail('reverse_exclusions', False, 'Reverse exclusion violated')]
        )
    details.append(MatchDetail('reverse_exclusions', True, 'No reverse exclusions violated'))

    # Step 10: Location check
    location_match = match_location(a, b)
    if not location_match.passed:
        return MatchOutput(
            result=MatchResult.NO_MATCH,
            score=0.0,
            details=details + [location_match]
        )
    details.append(location_match)

    # All checks passed
    return MatchOutput(
        result=MatchResult.MATCH,
        score=1.0,
        details=details
    )


def match_items(a_items: List[dict], b_items: List[dict]) -> MatchDetail:
    """Match item specifications."""
    for a_item in a_items:
        matched = False
        for b_item in b_items:
            if match_single_item(a_item, b_item):
                matched = True
                break
        if not matched:
            return MatchDetail('items', False, f"No match for item: {a_item.get('type')}")
    return MatchDetail('items', True, 'All items matched')


def match_single_item(a_item: dict, b_item: dict) -> bool:
    """Match a single item's constraints."""
    # Type must match
    if a_item.get('type') != b_item.get('type'):
        return False

    # Categorical: A's requirements ⊆ B's offerings
    a_cat = a_item.get('categorical', {})
    b_cat = b_item.get('categorical', {})
    for key, val in a_cat.items():
        if key not in b_cat or not terms_match(val, b_cat[key]):
            return False

    # Min constraints: B's value >= A's min
    a_min = a_item.get('min', {})
    for key, min_val in a_min.items():
        b_val = get_numeric_value(b_item, key)
        if b_val is None or b_val < min_val:
            return False

    # Max constraints: B's value <= A's max
    a_max = a_item.get('max', {})
    for key, max_val in a_max.items():
        b_val = get_numeric_value(b_item, key)
        if b_val is None or b_val > max_val:
            return False

    # Range constraints: B's value within [min, max]
    a_range = a_item.get('range', {})
    for key, (range_min, range_max) in a_range.items():
        b_val = get_numeric_value(b_item, key)
        if b_val is None or b_val < range_min or b_val > range_max:
            return False

    return True


def match_preferences(required: dict, offered: dict) -> MatchDetail:
    """Match other preferences against self attributes."""
    # Categorical matching
    req_cat = required.get('categorical', {})
    off_cat = offered.get('categorical', {})
    for key, val in req_cat.items():
        if key not in off_cat:
            return MatchDetail('preferences', False, f"Missing attribute: {key}")
        if not terms_match(val, off_cat[key]):
            return MatchDetail('preferences', False, f"Mismatch on {key}: wanted {val}, got {off_cat[key]}")

    # Min constraints
    req_min = required.get('min', {})
    for key, min_val in req_min.items():
        off_val = get_numeric_from_offered(offered, key)
        if off_val is None or off_val < min_val:
            return MatchDetail('preferences', False, f"Min constraint failed: {key} ({off_val} < {min_val})")

    # Max constraints
    req_max = required.get('max', {})
    for key, max_val in req_max.items():
        off_val = get_numeric_from_offered(offered, key)
        if off_val is None or off_val > max_val:
            return MatchDetail('preferences', False, f"Max constraint failed: {key} ({off_val} > {max_val})")

    # Range constraints
    req_range = required.get('range', {})
    for key, (range_min, range_max) in req_range.items():
        off_val = get_numeric_from_offered(offered, key)
        if off_val is None or off_val < range_min or off_val > range_max:
            return MatchDetail('preferences', False, f"Range constraint failed: {key}")

    return MatchDetail('preferences', True, 'All preferences matched')


def check_exclusions_disjoint(exclusions: List[str], attributes: dict) -> bool:
    """Check that exclusions don't overlap with attributes."""
    if not exclusions:
        return True

    # Flatten all attribute values
    all_values = set()

    def extract_values(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                extract_values(v)
        elif isinstance(obj, list):
            for item in obj:
                extract_values(item)
        elif isinstance(obj, str):
            all_values.add(obj.lower())

    extract_values(attributes)

    # Check for any overlap
    for excl in exclusions:
        excl_lower = excl.lower()
        if excl_lower in all_values:
            return False
        # Also check term implications
        for val in all_values:
            if terms_match(excl_lower, val):
                return False

    return True


def match_location(a: dict, b: dict) -> MatchDetail:
    """Match location based on mode."""
    a_loc = a.get('location', '')
    b_loc = b.get('location', '')
    a_mode = a.get('locationmode', 'proximity')
    a_excl = a.get('locationexclusions', [])
    b_excl = b.get('locationexclusions', [])

    # Check exclusions both ways
    if b_loc and any(excl.lower() in b_loc.lower() for excl in a_excl):
        return MatchDetail('location', False, f"B's location in A's exclusions")
    if a_loc and any(excl.lower() in a_loc.lower() for excl in b_excl):
        return MatchDetail('location', False, f"A's location in B's exclusions")

    # Mode-based matching
    if a_mode == 'flexible':
        return MatchDetail('location', True, 'Flexible location mode')

    if a_mode == 'proximity':
        # Check if locations overlap or are nearby
        if location_overlaps(a_loc, b_loc):
            return MatchDetail('location', True, 'Locations overlap')
        return MatchDetail('location', False, 'Locations too far apart')

    if a_mode == 'target':
        # A specifies where they want, B must be there
        if b_loc and a_loc.lower() in b_loc.lower():
            return MatchDetail('location', True, 'Target location matched')
        return MatchDetail('location', False, 'Target location not matched')

    return MatchDetail('location', True, 'Location check passed')


# Helper functions
def terms_match(required: str, offered: str) -> bool:
    """Check if offered term satisfies required term (with implications)."""
    if required.lower() == offered.lower():
        return True
    # Add term implications lookup here
    implications = {
        'vegan': ['vegetarian'],
        'vegetarian': ['no_beef', 'no_pork'],
        'non_smoker': ['no_smoking'],
        'teetotaler': ['no_drinking'],
        'new': ['excellent', 'good'],
        'excellent': ['good'],
    }
    return required.lower() in implications.get(offered.lower(), [])


def get_numeric_value(item: dict, key: str) -> Optional[float]:
    """Extract numeric value from item for a given key."""
    # Check in range first (for exact values stored as [x,x])
    if key in item.get('range', {}):
        range_val = item['range'][key]
        if isinstance(range_val, list) and len(range_val) == 2:
            if range_val[0] == range_val[1]:
                return range_val[0]
            return (range_val[0] + range_val[1]) / 2  # midpoint for range

    # Check in min/max
    if key in item.get('min', {}):
        return item['min'][key]
    if key in item.get('max', {}):
        return item['max'][key]

    # Check in categorical (if numeric stored there)
    if key in item.get('categorical', {}):
        try:
            return float(item['categorical'][key])
        except:
            pass

    return None


def get_numeric_from_offered(offered: dict, key: str) -> Optional[float]:
    """Get numeric value from offered attributes."""
    # Check range first
    if key in offered.get('range', {}):
        range_val = offered['range'][key]
        if isinstance(range_val, list) and len(range_val) == 2:
            return range_val[0] if range_val[0] == range_val[1] else range_val[0]

    # Check min/max
    if key in offered.get('min', {}):
        return offered['min'][key]
    if key in offered.get('max', {}):
        return offered['max'][key]

    # Check categorical
    if key in offered.get('categorical', {}):
        try:
            return float(offered['categorical'][key])
        except:
            pass

    return None


def location_overlaps(loc_a: str, loc_b: str) -> bool:
    """Check if two locations overlap (simplified)."""
    if not loc_a or not loc_b:
        return True  # If either is empty, assume flexible

    a_parts = set(loc_a.lower().replace(',', ' ').split())
    b_parts = set(loc_b.lower().replace(',', ' ').split())

    return bool(a_parts & b_parts)
```

---

## 10. Edge Cases & Handling

### 10.1 Missing Fields

| Scenario | Handling |
|----------|----------|
| A.other is empty | No preference constraints, any B.self matches |
| B.self is empty | Only matches if A.other is also empty |
| A.items is empty | Error for product/service, OK for mutual |
| Location missing | Treated as flexible if locationmode not set |

### 10.2 Type Mismatches

```python
# Handle type conversion gracefully
def safe_numeric_compare(a_val, b_val, operator: str) -> bool:
    try:
        a_num = float(a_val) if not isinstance(a_val, (int, float)) else a_val
        b_num = float(b_val) if not isinstance(b_val, (int, float)) else b_val

        if operator == '>=':
            return b_num >= a_num
        elif operator == '<=':
            return b_num <= a_num
        elif operator == '==':
            return abs(a_num - b_num) < 0.0001
        elif operator == 'in_range':
            return a_num[0] <= b_num <= a_num[1]
    except (ValueError, TypeError):
        return False
    return False
```

### 10.3 Array vs Single Value

```python
# Both formats are valid:
# Single: "domain": "electronics"
# Array: "domain": ["electronics", "gadgets"]

def normalize_to_array(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
```

### 10.4 Case Sensitivity

```
ALL string comparisons are case-INSENSITIVE

"Apple" = "apple" = "APPLE" ✓
"Vegetarian" = "vegetarian" ✓
"Bangalore" = "bangalore" = "BANGALORE" ✓
```

### 10.5 Partial Attribute Matching

```python
# For attributes that can have partial matches (like skills)
# Use contains logic

a.other.skills = ["python", "sql"]
b.self.skills = ["python", "sql", "javascript", "docker"]

# Check: A's skills ⊆ B's skills
# Result: ✓ MATCH (A's requirements are subset of B's offerings)
```

### 10.6 Semantic Matching Fallback

When SQL matching returns 0 results, fall back to semantic search:

```python
def find_matches_with_fallback(listing_id: str) -> List[dict]:
    # Step 1: Try SQL matching
    sql_matches = sql_find_matches(listing_id)

    if sql_matches:
        return sql_matches

    # Step 2: Fallback to semantic search
    # Relax some constraints and use vector similarity
    semantic_matches = vector_find_similar(
        listing_id,
        intent_filter=True,      # Keep intent filter
        subintent_filter=True,   # Keep subintent filter
        domain_filter=False,     # Relax domain
        location_filter=False,   # Relax location
        top_k=50
    )

    # Step 3: Re-apply exclusion filters (never relax)
    final_matches = [
        m for m in semantic_matches
        if passes_exclusion_checks(listing_id, m)
    ]

    return final_matches
```

---

## Appendix A: Domain Lists

### Product Domains (21)
1. electronics
2. fashion
3. home_appliances
4. furniture
5. vehicles
6. real_estate
7. books
8. sports
9. toys
10. health
11. beauty
12. jewelry
13. art
14. antiques
15. collectibles
16. music_instruments
17. pet_supplies
18. garden
19. tools
20. office
21. food

### Service Domains (18)
1. education
2. healthcare
3. legal
4. finance
5. technology
6. creative
7. construction
8. automotive
9. cleaning
10. beauty_wellness
11. events
12. travel
13. food_catering
14. fitness
15. childcare
16. pet_care
17. home_services
18. professional

### Mutual Categories (25)
1. roommate
2. travel_buddy
3. study_partner
4. workout_partner
5. business_partner
6. mentor
7. language_exchange
8. hobby_group
9. sports_team
10. music_band
11. startup_cofounder
12. carpool
13. pet_playdate
14. book_club
15. cooking_club
16. hiking_group
17. photography_club
18. volunteer_group
19. gaming_team
20. movie_club
21. parenting_group
22. support_group
23. networking
24. dating
25. friendship

---

## Appendix B: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRIDDHI MATCHING QUICK REFERENCE             │
├─────────────────────────────────────────────────────────────────┤
│ MODES (Only 3):                                                 │
│   min:   {"key": value}      → B.value >= A.min                 │
│   max:   {"key": value}      → B.value <= A.max                 │
│   range: {"key": [lo, hi]}   → lo <= B.value <= hi              │
│   EXACT = range[x, x]                                           │
├─────────────────────────────────────────────────────────────────┤
│ INTENT MATCHING:                                                │
│   Product: buy ↔ sell (INVERSE)                                 │
│   Service: seek ↔ provide (INVERSE)                             │
│   Mutual:  connect = connect (SAME, bidirectional)              │
├─────────────────────────────────────────────────────────────────┤
│ FIELD MATCHING:                                                 │
│   A.other → B.self (what A wants = what B is)                   │
│   B.other → A.self (reverse check)                              │
│   A.items → B.items (A's requirements ⊆ B's offerings)          │
├─────────────────────────────────────────────────────────────────┤
│ EXCLUSIONS:                                                     │
│   Always STRICT - disjoint sets required                        │
│   A.exclusions ∩ B.attributes = ∅                               │
├─────────────────────────────────────────────────────────────────┤
│ SIGNAL WORDS:                                                   │
│   (none)/"exactly" → range[x,x]                                 │
│   "under"/"max"/"below" → max constraint                        │
│   "minimum"/"at least" → min constraint                         │
│   "between X and Y" → range[X, Y]                               │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 2.0*
*Last Updated: January 2026*
*Author: VRIDDHI Engineering Team*
