GLOBAL_REFERENCE_CONTEXT.md

# VRIDDHI — GLOBAL REFERENCE CONTEXT# (READ-ONLY · NEVER EDITED · INJECTED INTO EVERY STAGE)


### domains must follow these don't invent newer domains try to fit in this 

10. FIXED DOMAIN/CATEGORY LISTS

### 21 Product Domains

1. Technology & Electronics
2. Healthcare & Wellness
3. Fashion & Apparel
4. Home & Furniture
5. Food & Beverage
6. Automotive & Vehicles
7. Sports & Outdoors
8. Office & Stationery
9. Books, Media & Entertainment
10. Pets & Animals
11. Real Estate & Property
12. Manufacturing & Production
13. Agriculture & Farming
14. Environmental & Sustainability
15. Textile & Clothing Manufacturing
16. Jewelry & Accessories Manufacturing
17. Beauty & Cosmetics
18. Handicrafts & Artisan Products
19. Energy & Utilities
20. Security & Safety
21. Mining & Quarrying

### 18 Service Domains

1. Education & Training
2. Finance, Insurance & Legal
3. Transportation & Logistics
4. Hospitality, Travel & Accommodation
5. Business Services & Consulting
6. Marketing, Advertising & Design
7. Construction & Trades
8. Entertainment & Events
9. Personal Services
10. Government & Regulatory
11. Utilities & Infrastructure
12. Telecommunication & Internet
13. Nonprofit & Charity Services
14. Repair & Maintenance Services
15. Customs & Culture Services
16. Alternative & Holistic Health
17. Research & Development
18. Government & Public Administration

### 25 Mutual Categories

1. Housing
2. Roommates
3. Fitness
4. Sports
5. Partners
6. Travel
7. Adventure
8. Learning
9. Study
10. Professional
11. Career
12. Social
13. Friendship
14. Dating
15. Relationships
16. Parenting
17. Family
18. Hobbies
19. Interests
20. Pets
21. Animals
22. Support
23. Caregiving
24. Community
25. Volunteering

---

### Read this very important 
## <field_name>

1. Definition
2. When this field MUST be populated
3. When this field MUST be empty
4. Allowed structure & data type
5. Standardization & normalization rules
6. What the model MUST do
7. What the model MUST NEVER do
8. Positive examples (TRUE POSITIVES)
9. Negative examples
   - Hard negatives
   - False positives
   - False negatives
10. Edge cases & ambiguity handling
11. Validation checks
No deviation. No creativity.

## INTENT
### SECTION A: INTENT CLASSIFICATION (Q1-Q5)

1. Definition

intent defines the fundamental nature of the user’s request.

It answers WHAT kind of interaction the user wants, independent of domain, attributes, or constraints.

The system recognizes exactly three intents:

product | service | mutual

2. When this field MUST be populated

ALWAYS

intent is a mandatory field

No query is allowed to proceed without a resolved intent

If intent cannot be determined unambiguously → sample must be REJECTED in Stage 4

3. When this field MUST be empty

NEVER

intent must never be empty, null, or missing

4. Allowed structure & data type
"intent": "product" | "service" | "mutual"


Type: string (enum)

Single value only

Case-insensitive at input, stored lowercase

No arrays

No alternative labels

No extensions

5. Standardization & normalization rules

Normalize all variants to exact enum values

Synonyms, wording, or phrasing do not affect output value

User expression	Normalized intent
buy, sell, phone, car	product
tutor, repair, design, teaching	service
partner, buddy, roommate	mutual

⚠️ Never invent new intent types

6. What the model MUST do

Decide intent semantically, not via keyword match

Use context and meaning, not word presence alone

Assign exactly one intent

Resolve ambiguity using intent priority rules (see Section 10)

7. What the model MUST NEVER do

🚫 Infer intent from domain alone
🚫 Create hybrid intents
🚫 Output multiple intents
🚫 Leave intent undefined
🚫 Emit reasoning steps or decision trees in output
🚫 Change enum values

8. INTERNAL CoT QUESTIONS (DATA GENERATION ONLY)

⚠️ IMPORTANT

These questions are INTERNAL scaffolding for data generation.

❌ They MUST NOT appear in:

Training samples

Model outputs

reasoning field

✅ They MAY be used by the generator to ensure consistency.

Q1: Is there a PRODUCT (tangible item, ownership transfer)?
    Signals: buy, sell, purchase, phone, car, laptop, furniture
    → YES → intent = product
    → NO → continue

Q2: Is there a SERVICE (work, expertise, task performed, no ownership)?
    Signals: need X person, -er professions, services, repair, consultation
    → YES → intent = service
    → NO → continue

Q3: Is there a MUTUAL activity (shared participation, symmetric roles)?
    Signals: partner, buddy, flatmate, companion, together, with me
    → YES → intent = mutual

Q4: If multiple signals exist:
    - Ownership transfer dominates → product
    - Work performed dominates → service
    - Symmetric relationship dominates → mutual

Q5: If still unclear → mark sample INVALID

9. Positive examples (TRUE POSITIVES)
Product
Query	intent
“looking to buy an iphone”	product
“selling my bike”	product
“anyone selling used furniture”	product
Service
Query	intent
“need a plumber”	service
“looking for math tutor”	service
“i offer graphic design services”	service
Mutual
Query	intent
“looking for a gym buddy”	mutual
“need a roommate”	mutual
“anyone want to travel together”	mutual
10. Negative examples
❌ Hard Negatives (must be rejected or corrected)
Query	Wrong intent	Why
“selling my time to startups”	product	Time is not a product
“buying mentorship sessions”	product	No ownership transfer
“hiring cofounder”	service	Cofounder is mutual
❌ False Positives
Query	Incorrect	Correct
“looking for a tennis partner” → service	❌	mutual
“buying consulting hours” → product	❌	service
❌ False Negatives
Query	Missed intent
“anyone up for morning walks?”	mutual
“need someone to fix my sink”	service
11. Edge cases & ambiguity handling
Case 1: Product + Service mentioned together

“buy a laptop and need help setting it up”

intent = product

service aspects handled later (ignored at intent stage)

Case 2: Mutual + Service ambiguity

“looking for a cofounder to build a startup”

No payment

Symmetric roles
→ intent = mutual

Case 3: Vague phrasing

“need help with my website”

No ownership transfer

Task performed
→ intent = service

Case 4: Multiple intents implied

“selling my camera and teaching photography”

⚠️ Ambiguous
→ INVALID sample
(or split into two queries at Stage 1)

12. Validation checks

A sample is INVALID if:

intent ∉ {product, service, mutual}

intent is missing or empty

Multiple intents implied without dominance

Intent inferred but not stated semantically

Output leaks CoT / decision steps

FINAL LOCK (DO NOT CHANGE)

Intent answers WHAT type of interaction exists.
It is decided once, first, and never revised downstream.

### SUB_INTENT
FIELD SPECIFICATION — subintent
1. Definition

subintent specifies the direction of action within a resolved intent.

It answers WHAT SIDE of the interaction the user is on (demand vs supply), after intent is fixed.

subintent is dependent on intent and cannot exist independently.

2. When this field MUST be populated

ALWAYS, once intent is resolved

Mandatory for all three intents

If intent exists and subintent is missing → INVALID sample

3. When this field MUST be empty

NEVER

subintent must never be null, empty, or omitted

4. Allowed structure & data type
"subintent": "buy" | "sell" | "seek" | "provide" | "connect"


Type: string (enum)

Single value only

Lowercase only

No arrays

No aliases

No extensions

5. Standardization & normalization rules
Subintent mapping by intent (LOCKED)
intent	Allowed subintent(s)
product	buy, sell
service	seek, provide
mutual	connect

🚫 Any other combination is INVALID.

Direction normalization
User language	Normalized subintent
want, need, looking for	buy / seek
selling, offering, available	sell / provide
partner, buddy, together	connect
6. What the model MUST do

Assign exactly one subintent

Ensure subintent is compatible with intent

Decide direction semantically, not by keywords alone

Resolve tense, phrasing, and implied direction correctly

7. What the model MUST NEVER do

🚫 Assign multiple subintents
🚫 Use subintent values outside allowed enum
🚫 Infer supply when demand is stated
🚫 Change intent–subintent pairing
🚫 Emit direction reasoning or decision trees in output

8. INTERNAL CoT QUESTIONS (DATA GENERATION ONLY)

⚠️ INTERNAL USE ONLY — NEVER OUTPUT

Q1: Is the user DEMANDING something?
    Signals: DON'T GO ON THE KEYWORDS UNDERSTAND THE EMOTION
    → demand

Q2: Is the user OFFERING something?
    Signals: DON'T GO ON THE KEYWORDS UNDERSTAND THE EMOTION
    → supply

Q3: Combine with intent:
    product + demand → buy
    product + supply → sell
    service + demand → seek
    service + supply → provide
    mutual → connect (always)

Q4: If both demand and supply are present:
    → SEE IF IT WAS EXCHANGE IT WILL GO TO MUTUAL

9. Positive examples (TRUE POSITIVES)
Product
Query	subintent
“looking to buy a used iphone”	buy
“selling my old laptop”	sell
Service
Query	subintent
“need a math tutor”	seek
“i offer freelance design”	provide
Mutual
Query	subintent
“looking for a gym buddy”	connect
“need a cofounder”	connect
10. Negative examples
❌ Hard Negatives
Case	Why invalid
product + seek	seek is not allowed for product
service + buy	buy is ownership transfer only
mutual + provide	mutual has no direction
❌ False Positives
Query	Wrong	Correct
“selling consultation hours”	sell	provide
“buying tutoring sessions”	buy	seek
❌ False Negatives
Query	Missed subintent
“any tutors available?”	seek
“designer here for freelance work”	provide
11. Edge cases & ambiguity handling
Case 1: Self-description implying supply

“i’m a backend developer, open to projects”

→ service + provide

Case 2: Question form implies demand

“any good plumbers around?”

→ service + seek

Case 3: Mutual phrased as demand

“need a roommate”

→ mutual + connect
(Direction is symmetric, not demand/supply)

Case 4: Both demand and supply present

“selling my camera and looking to buy another”

⚠️ INVALID SINGLE SAMPLE
Must be split upstream.

12. Validation checks

A sample is INVALID if:

subintent does not match allowed set for intent

subintent missing or null

Multiple directions implied

Direction inferred without semantic support

CoT / decision logic appears in output

FINAL LOCK (DO NOT CHANGE)

subintent defines direction, not desire.
It is strictly constrained by intent.
One intent → one direction → one subintent.

### DOMAIN 
FIELD SPECIFICATION — domain

"""
### 21 Product Domains

1. Technology & Electronics
2. Healthcare & Wellness
3. Fashion & Apparel
4. Home & Furniture
5. Food & Beverage
6. Automotive & Vehicles
7. Sports & Outdoors
8. Office & Stationery
9. Books, Media & Entertainment
10. Pets & Animals
11. Real Estate & Property
12. Manufacturing & Production
13. Agriculture & Farming
14. Environmental & Sustainability
15. Textile & Clothing Manufacturing
16. Jewelry & Accessories Manufacturing
17. Beauty & Cosmetics
18. Handicrafts & Artisan Products
19. Energy & Utilities
20. Security & Safety
21. Mining & Quarrying

### 18 Service Domains

1. Education & Training
2. Finance, Insurance & Legal
3. Transportation & Logistics
4. Hospitality, Travel & Accommodation
5. Business Services & Consulting
6. Marketing, Advertising & Design
7. Construction & Trades
8. Entertainment & Events
9. Personal Services
10. Government & Regulatory
11. Utilities & Infrastructure
12. Telecommunication & Internet
13. Nonprofit & Charity Services
14. Repair & Maintenance Services
15. Customs & Culture Services
16. Alternative & Holistic Health
17. Research & Development
18. Government & Public Administration
"""
1. Definition

domain defines the high-level problem space or market vertical to which the user query belongs.

It answers “WHAT general area is this request about?”, independent of intent direction, attributes, or constraints.

domain does NOT describe how, how much, or with what qualities — only WHAT space.

2. When this field MUST be populated

ALWAYS

Mandatory for product, service, and mutual

At least one domain must be assigned

If no suitable domain exists → choose the closest predefined domain
If still impossible → INVALID sample

3. When this field MUST be empty

NEVER

domain cannot be empty, null, or missing

4. Allowed structure & data type
"domain": ["<domain_string>"]


Type: array of strings

Minimum length: 1

Maximum length: N (multi-domain allowed)

All values:

lowercase

predefined

no free-text

no creativity

🚫 INVALID:

"domain": "electronics"
"domain": ["tech", "gadgets"]
"domain": []

5. Standardization & normalization rules
5.1 Domain source (LOCKED)

Domains come from predefined lists only:

Product Domains

Service Domains

Mutual Categories (mapped separately, see primary_mutual_category)

⚠️ The model must NEVER invent new domains

5.2 Multi-domain rules

Use multiple domains only if the query genuinely spans multiple spaces

Do NOT over-assign

Prefer specific over generic

Example:

“repair laptop screen”

"domain": ["technology & electronics"]


Not:

["technology", "services", "hardware"]

6. What the model MUST do

Select the closest matching predefined domain

Use semantic meaning, not keyword frequency

Prefer market-understood categories

Assign minimum necessary domains

7. What the model MUST NEVER do

🚫 Invent new domain names
🚫 Use sub-domain terms as domains
🚫 Encode attributes in domain
🚫 Omit domain
🚫 Change schema shape
🚫 Over-generalize when specificity exists

8. INTERNAL CoT QUESTIONS (DATA GENERATION ONLY)

⚠️ INTERNAL USE ONLY — NEVER OUTPUT

Q1: Is the request about a tangible item?
    → Use Product Domain list

Q2: Is the request about work, skill, or service?
    → Use Service Domain list

Q3: Is the request about a human relationship or shared activity?
    → Use Mutual Categories (domain stays empty or generic)

Q4: What is the CLOSEST predefined domain?
    (Do not invent new labels)

Q5: Does the request truly span two domains?
    → If YES, include both
    → If NO, choose the dominant one

Q6: If no reasonable domain fits → INVALID
Q12: What TYPE of entity is being sought?
    THING (tangible item) → Product domains (21 options)
    WORK/EXPERTISE (service) → Service domains (18 options)
    PERSON (peer for shared activity) → Mutual categories (25 options)

Q13: Which EXISTING domain/category is CLOSEST match?
    Use semantic similarity to map to fixed list
    NEVER create new domains - always map to existing 64
    For unseen entities, find closest semantic match

Q14: VALIDATE domain/category assignment:
    For PRODUCT/SERVICE: domain = valid, primary_mutual_category = null
    For MUTUAL: domain = null, primary_mutual_category = valid

9. Positive examples (TRUE POSITIVES)
Product
Query	domain
“buy iphone 14”	["technology & electronics"]
“selling used bike”	["automotive & vehicles"]
Service
Query	domain
“need math tutor”	["education & training"]
“plumber needed”	["construction & trades"]
Mutual
Query	domain
“looking for a roommate”	["real estate & property"]
“gym buddy needed”	["sports & outdoors"]
10. Negative examples
❌ Hard Negatives
Case	Why invalid
new domain invented	schema violation
empty domain	mandatory missing
attribute used as domain	misuse
❌ False Positives
Query	Wrong domain	Correct
“buy gaming laptop”	["gaming"]	["technology & electronics"]
“need yoga instructor”	["yoga"]	["alternative & holistic health"]
❌ False Negatives
Query	Missed domain
“need car repair”	["automotive & vehicles"]
“selling office desk”	["home & furniture"]
11. Edge cases & ambiguity handling
Case 1: Generic phrasing

“need help with something online”

→ INVALID (no clear domain)

Case 2: Platform vs domain

“selling on amazon”

Domain is NOT “e-commerce”
Choose based on item/service, not platform.

Case 3: Multiple domain overlap

“fitness app development”

"domain": ["technology & electronics", "fitness & wellness"]


Only if BOTH are essential.

12. Validation checks

A sample is INVALID if:

domain is empty

domain not in predefined list

domain invented or free-text

domain contradicts intent

excessive domains without semantic justification

FINAL LOCK (DO NOT CHANGE)

domain answers WHAT space the problem belongs to.
It never encodes attributes, direction, or constraints.
It is chosen from a fixed universe and nothing else.

### PRIMARY_MUTUAL_CATEGORY
FIELD SPECIFICATION — primary_mutual_category
1. Definition

primary_mutual_category identifies the core human relationship or shared activity type in a mutual intent.

It answers:

“WHAT kind of human-to-human connection is being sought?”

This field exists ONLY to specialize mutual intent beyond domain-level abstraction.

2. When this field MUST be populated

ONLY IF

intent = "mutual"


Mandatory for all mutual queries

If intent = mutual and this field is empty → INVALID sample

3. When this field MUST be empty

ALWAYS EMPTY if:

intent = product OR service


If populated for product/service → INVALID

4. Allowed structure & data type
"primary_mutual_category": ["<category_string>"]


Type: array of strings

Minimum length: 1

Maximum length: 1 (single primary category only)

Lowercase only

Predefined list only

No free-text

No creativity

🚫 INVALID:

"primary_mutual_category": []
"primary_mutual_category": ["friendship", "travel"]
"primary_mutual_category": "roommate"

5. Standardization & normalization rules
5.1 Source of truth (LOCKED)

Categories must come from the predefined Mutual Category List (25 items you locked earlier):

housing
roommates
fitness
sports
partners
travel
adventure
learning
study
professional
career
social
friendship
dating
relationships
parenting
family
hobbies
interests
pets
animals
support
caregiving
community
volunteering


⚠️ The model must NEVER invent a new category.

5.2 One-category rule (CRITICAL)

Choose ONE category only

Pick the dominant shared purpose

Secondary interests belong in attributes, NOT here

6. What the model MUST do

Populate this field only for mutual intent

Select exactly one category

Use semantic meaning, not keyword matching

Prefer human-understood relationship types

Choose the strongest signal

7. What the model MUST NEVER do

🚫 Populate for product or service
🚫 Leave empty for mutual
🚫 Output multiple categories
🚫 Encode attributes or preferences
🚫 Invent categories
🚫 Use domain names as categories

8. INTERNAL CoT QUESTIONS (DATA GENERATION ONLY)

⚠️ INTERNAL USE ONLY — NEVER OUTPUT

Q1: Is the user seeking a HUMAN connection?
    → If NO → this field must be empty

Q2: Is participation SYMMETRIC?
    → If NO → not mutual

Q3: What is the PRIMARY shared purpose?
    (Ignore attributes, preferences, constraints)

Q4: Which ONE mutual category best describes it?
    → Choose ONLY ONE

Q5: If none fit → INVALID

9. Positive examples (TRUE POSITIVES)
Query	primary_mutual_category
“need a roommate in indiranagar”	["roommates"]
“looking for gym buddy”	["fitness"]
“want a travel partner to goa”	["travel"]
“seeking cofounder for startup”	["professional"]
“looking for hiking group”	["adventure"]
10. Negative examples
❌ Hard Negatives
Case	Why invalid
empty for mutual	mandatory missing
more than one category	schema violation
category invented	not predefined
❌ False Positives
Query	Wrong category	Correct
“looking for gym buddy”	["sports"]	["fitness"]
“need cofounder”	["career"]	["professional"]
❌ False Negatives
Query	Missed category
“someone to share flat with”	["roommates"]
“anyone into book clubs?”	["hobbies"]
11. Edge cases & ambiguity handling
Case 1: Multiple activities mentioned

“looking for friend to travel and trek”

Choose dominant motivation: IF BOTH DOMAINS ARE LOOKING THEN IT CAN BE BOTH 

["travel"]


Secondary interests handled as attributes later.

Case 2: Relationship implied indirectly

“need someone to split rent with”

→ ["roommates"]

Case 3: Professional mutual vs service

“looking for business partner”

No payment

Equal stake
→ mutual + ["professional"]

12. Validation checks

A sample is INVALID if:

intent ≠ mutual AND category populated

intent = mutual AND category missing

category not in predefined list

more than one category

category used to encode attributes

FINAL LOCK (DO NOT CHANGE)

primary_mutual_category defines the HUMAN RELATIONSHIP TYPE.
One mutual query → one dominant category → nothing more.

### SECTION E: ITEMS & ITEM_EXCLUSIONS
ITEMS, ATTRIBUTES & EXCLUSIONS

(CRITICAL CORE — DO NOT WEAKEN)

Covers Fields

items

item_exclusions

(attributes live inside items using axis → min/max/range)

This section is the foundation of matching.
Mistakes here break SQL filtering and determinism.

FIELD: items
1. Definition

items represents WHAT thing is being interacted with,
regardless of intent, ownership, state, or action.

It is intent-agnostic.

product → thing

service → service-type
 
"""irrespective of the intent if product or service is present in the query then add "

2. When this field MUST be populated

Populate items IF AND ONLY IF the query mentions:

a tangible object

a service / work type

a subject being exchanged, discussed, or interacted with  and it must be product or service cannonicalize to market standard in language level not ontology level using polysemy

Examples:

“iphone”, “laptop”, “bike”

“plumber”, “math tutor”, “yoga instructor”

“language exchange”, “skill swap”, “lost phone”

3. When this field MUST be empty

Leave items = [] ONLY when:

the query is purely social/emotional with no subject

the query is meta (“anyone here?”, “just browsing”)

4. Allowed structure & data type
"items": [
  {
    "type": "<canonical market noun>",
    "categorical": {
      "<key>": "<value>"
    },
    "min": {
      "<axis>": [{ "type": "", "value": <num>, "unit": "" }]
    },
    "max": {
      "<axis>": [{ "type": "", "value": <num>, "unit": "" }]
    },
    "range": {
      "<axis>": [{ "type": "", "min": <num>, "max": <num>, "unit": "" }]
    }
  }
]

FIELD PURPOSES:
- type: What it is (canonical market noun)
- categorical: Non-numeric attributes (condition, fuel, color, brand, etc.)
- min: Minimum constraint (numeric attributes with axis)
- max: Maximum constraint (numeric attributes with axis)
- range: Exact value (min=max) OR range (min≠max)

Rules:

type → REQUIRED

categorical → OPTIONAL (only if non-numeric attributes stated)

min/max/range → OPTIONAL (only if numeric constraints stated)

Multiple items allowed

Each item is independent

5. Standardization & Normalization Rules
5.1 Type Standardization (LANGUAGE-LEVEL)

✔️ Allowed (LLM responsibility):

Query phrase	items.type
iphone	smartphone
mobile phone	smartphone
pipe leakage	plumbing
need a plumber	plumbing
math tutor	tutoring
yoga instructor	yoga
lost phone	smartphone

❌ Forbidden:

Creating new ontology nodes

Encoding state/action into type

Examples ❌:

lost_item

skill_exchange

language_exchange

5.2 Polysemy Handling (MANDATORY)

CoT Decision Gate (USED FOR GENERATION, NOT OUTPUT)

Q1: Is the phrase referring to an OBJECT, SERVICE, or SUBJECT?
Q2: Is the surface word ambiguous?
Q3: Does the surrounding language clarify the base noun?
→ Map to the BASE MARKET NOUN


✔️ “leakage in pipes” → plumbing
✔️ “cracked phone” → smartphone
✔️ “math teacher” → tutoring

6. Attributes (INSIDE items)

Attributes are constraints on items
They MUST attach to one of the fixed axes.

Fixed Attribute Axes (LOCKED)
identity
capacity
performance
quality
quantity
time
space
cost
mode
skill


No new axes. Ever.

7. Attribute Extraction — CoT (CONTROL QUESTIONS)

These questions guide extraction but NEVER appear in output.

Q15: Does the query specify a PROPERTY of the item?
     (storage, price, experience, rating, size, speed)

Q16: Is the value quantitative, categorical, boolean, or temporal?

Q17: Is a constraint implied?
     - under / below → max
     - over / at least → min
     - between → range
     - exact mention → min=max

Q18: Is the unit explicitly stated?
     - If yes → extract
     - If no → use 'local' ONLY for currency
**Covers Fields**: other_party_preferences, self_attributes



8. Min / Max / Range Rules (STRICT)

Exact DOES NOT EXIST

Exact = min == max

Valid
"min": {
  "capacity": [
    { "type": "storage", "value": 1024, "unit": "gb" }
  ]
}

INVALID ❌
"capacity": [{ "type": "storage", "value": 1024 }]

9. Numeric Normalization Rules
Allowed (LOSSLESS ONLY)
Input	Stored
512mb	0.5 gb
1024mb	1 gb
1tb	1024 gb
3 years	36 months
Forbidden ❌

HDD ↔ SSD

USD → INR

degree equivalence

assumed units

10. Country-Specific vs Universal
Universal (convertible)

time

distance

storage

weight

area

Country-specific (preserve)

currency

education labels

legal terms

If currency not stated → unit = local

11. What the model MUST do

✔️ Extract only stated facts
✔️ Normalize numeric values losslessly
✔️ Standardize type to market noun
✔️ Attach attributes ONLY via axes
✔️ Use min/max/range consistently
✔️ Keep schema shape identical always

12. What the model MUST NEVER do

🚫 Invent attributes
🚫 Encode action or state in type
🚫 Create new axes
🚫 Guess missing values
🚫 Perform ontology reasoning
🚫 Convert country-specific units

13. Positive Examples (TRUE POSITIVES)
Example 1

“looking to buy a laptop with 1tb ssd and 24gb ram”

"items": [
  {
    "type": "laptop",
    "min": {
      "capacity": [
        { "type": "storage", "value": 1024, "unit": "gb" },
        { "type": "memory", "value": 24, "unit": "gb" }
      ]
    }
  }
]

Example 2

“need someonewho fix  pipe leakage”

"items": [
  { "type": "plumbing" }
]

14. Negative Examples
Hard Negatives ❌

lost_item

skill_exchange

language_exchange

pipe_leak_service

False Positive ❌

Extracting attributes not mentioned

False Negative ❌

Failing to extract smartphone in “lost phone”

15. Edge Cases & Ambiguity Handling

✔️ If unsure → choose base noun
✔️ If multiple items → extract all
✔️ If attributes vague → skip attributes
✔️ Intent NEVER changes item extraction

16. Validation Checks (STAGE 4)

Reject if:

axis appears outside min/max/range

type encodes action/state

non-market word used

schema deviates

numeric unit missing

FINAL LOCK (DO NOT CHANGE)

Items capture WHAT, not HOW, not WHY, not STATE.

Attributes capture constraints only when explicit.

All intelligence beyond language normalization happens AFTER the model.


### OTHER_PREFERENCES

Definition

other_party_preferences represents constraints and expectations about the OTHER PERSON,
not about the product/service itself and not about the user.

It answers:

“What must the other person be / have / do?”

This field is person-centric only.

2️⃣ When this field MUST be populated

Populate only if the query explicitly states requirements about the other person, such as:

Skills or experience of the person

Personal traits (gender, age, profession)

Language, location, background

Lifestyle or habits (non-smoker, vegetarian, etc.)

Examples (MUST populate)

“Need a software developer who speaks Kannada”

“Looking for a roommate female, age 25–30”

“Need a tutor with 3 years of experience”

“Driver from Karnataka”

“Non-smoker preferred”

3️⃣ When this field MUST be empty

MUST be {} if no person-specific constraints exist.

Examples (MUST be empty)

“Looking for an iPhone under 30k”

“Laptop repair service needed”

“Selling my Royal Enfield”

“Need plumbing service urgently” (unless plumber traits are specified)

🚫 Do NOT move product/service attributes here.

COT:
```
Q20: What preferences about the OTHER PERSON are mentioned?
    For buy: About SELLER (language, verified, no agents)
    For sell: About BUYER (serious, payment method)
    For seek: About PROVIDER (experience, rating, certified)
    For provide: About CUSTOMER (budget, timeline)
    For connect: About PARTNER (age, gender, diet, smoking)
    → Extract to other_party_preferences

Q21: What SELF attributes are mentioned (user describing themselves)?
    "I am female", "I'm a software engineer", "non-smoker here"
    → Extract to self_attributes
    Note: Mainly used for MUTUAL and SERVICE provide

Q22: Apply YES/NO FLAGS for person traits:
    Positive: "verified seller" → verified: "yes"
    Negative: "no agents" → agent: "no"
    Semantic expansion:
    - "no bad habits" → smoking: "no", drinking: "no"
    - "clean lifestyle" → smoking: "no", drinking: "no"
    - "vegetarian" → diet: "vegetarian"
```

4️⃣ Allowed structure & data types (LOCKED)
"other_party_preferences": {
  "identity": [ { "type": "", "value": "" } ],
  "lifestyle": [ { "type": "", "value": "" } ],
  "habits": {
    "<flag_name>": "yes | no"
  },
  "min": {
    "<axis>": [
      { "type": "", "value": <number>, "unit": "" }
    ]
  },
  "max": {
    "<axis>": [
      { "type": "", "value": <number>, "unit": "" }
    ]
  },
  "range": {
    "<axis>": [
      { "type": "", "min": <number>, "max": <number>, "unit": "" }
    ]
  }
}


Empty arrays/objects are VALID.

5️⃣ Standardization & Normalization Rules
Identity

Single-word, market terms

Lowercase

No assumptions

Examples:

female, male
software engineer
student
teacher

Numeric attributes

Must use min / max / range

Exact = min = max

Normalize units (lossless only)

Examples:

3 years → 36 months
25 years old → range.age = [25,25]

Location (person-based)

Extract ONLY if referring to the other person

Example:

“Developer from Karnataka”

"identity": [
  { "type": "location", "value": "karnataka" }
]

Habits (FLAGS ONLY)

Binary ONLY.

Examples:

smoking: "no"
drinking: "no"
pets: "yes"


🚫 No exclusions array
🚫 No partial values

6️⃣ What the model MUST do

✅ Extract only person-related facts
✅ Use identity / lifestyle / habits correctly
✅ Normalize numeric values
✅ Use market-standard words
✅ Preserve country-specific semantics
✅ Leave empty if not stated

7️⃣ What the model MUST NEVER do

🚫 Move product/service attributes here
🚫 Infer skills, age, gender
🚫 Convert currency, language, education
🚫 Use exclusions instead of flags
🚫 Invent new keys
🚫 Use free-text blobs

8️⃣ Positive Examples (TRUE POSITIVES)
Example 1

Query

“Need a software developer who speaks Kannada and has 3 years of React experience.”

"other_party_preferences": {
  "identity": [
    { "type": "language", "value": "kannada" }
  ],
  "min": {
    "time": [
      { "type": "experience", "value": 36, "unit": "month" }
    ]
  }
}


(React skill → goes to items, NOT here)

Example 2

Query

“Looking for a roommate, female, 25–30, non-smoker.”

"other_party_preferences": {
  "identity": [
    { "type": "gender", "value": "female" }
  ],
  "range": {
    "time": [
      { "type": "age", "min": 25, "max": 30, "unit": "year" }
    ]
  },
  "habits": {
    "smoking": "no"
  }
}

9️⃣ Negative Examples
❌ Hard Negative
"other_party_preferences": {
  "min": {
    "cost": [ { "type": "price", "value": 30000 } ]
  }
}


❌ Price belongs to item/service, not person.

❌ False Positive

Extracting preferences when none stated.

Query:

“Need laptop repair”

❌ Any person preferences extracted → WRONG

❌ False Negative

Ignoring explicit person constraint.

Query:

“Need a driver who speaks Tamil”

❌ Leaving preferences empty → WRONG

🔟 Edge Cases & Ambiguity Handling
“Preferably female”

→ Extract identity.gender = female

“Experienced developer”

→ ONLY extract if numeric given
❌ “experienced” alone → ignore

“Good person / reliable”

→ Ignore unless measurable (rating, years)

11️⃣ Validation Checks (MANDATORY)

Reject if:

Any product attribute appears here

Any inferred preference exists

Any habit not binary

Any numeric without min/max/range

Any invented key appears

🔒 FINAL INVARIANT (SAVE THIS)
other_party_preferences =
ONLY person-related requirements
NO product/service traits
NO inference
NO creativity
FLAGS instead of exclusions

### self_attributes
FIELD SPECIFICATION: self_attributes
1️⃣ Definition

self_attributes represents facts the USER explicitly states about THEMSELVES.

It answers:

“What am I?”

This field is self-descriptive only, never aspirational and never inferred.

2️⃣ When this field MUST be populated

Populate only when the user explicitly states information about themselves, such as:

Skills they have

Their profession / role

Age, gender (only if stated)

Languages they speak

Habits or lifestyle choices

Location they belong to (if stated as identity, not target)

Examples (MUST populate)

“I’m a software developer with 5 years experience”

“I’m a non-smoker”

“25-year-old female”

“I run every morning”

“I’m from Karnataka”

3️⃣ When this field MUST be empty

MUST be {} when:

User does not describe themselves

Query is purely about buying/selling/seeking

The description is aspirational or implied

Examples (MUST be empty)

“Need a plumber urgently”

“Looking for a roommate”

“Selling my laptop”

“Need a developer”

4️⃣ Allowed structure & data types (LOCKED)
"self_attributes": {
  "identity": [
    { "type": "", "value": "" }
  ],
  "lifestyle": [
    { "type": "", "value": "" }
  ],
  "habits": {
    "<flag_name>": "yes | no"
  },
  "min": {
    "<axis>": [
      { "type": "", "value": <number>, "unit": "" }
    ]
  },
  "max": {
    "<axis>": [
      { "type": "", "value": <number>, "unit": "" }
    ]
  },
  "range": {
    "<axis>": [
      { "type": "", "min": <number>, "max": <number>, "unit": "" }
    ]
  }
}


Empty arrays / objects are VALID and EXPECTED.

```
Q20: What preferences about the OTHER PERSON are mentioned?
    For buy: About SELLER (language, verified, no agents)
    For sell: About BUYER (serious, payment method)
    For seek: About PROVIDER (experience, rating, certified)
    For provide: About CUSTOMER (budget, timeline)
    For connect: About PARTNER (age, gender, diet, smoking)
    → Extract to other_party_preferences

Q21: What SELF attributes are mentioned (user describing themselves)?
    "I am female", "I'm a software engineer", "non-smoker here"
    → Extract to self_attributes
    Note: Mainly used for MUTUAL and SERVICE provide

Q22: Apply YES/NO FLAGS for person traits:
    Positive: "verified seller" → verified: "yes"
    Negative: "no agents" → agent: "no"
    Semantic expansion:
    - "no bad habits" → smoking: "no", drinking: "no"
    - "clean lifestyle" → smoking: "no", drinking: "no"
    - "vegetarian" → diet: "vegetarian"
```

5️⃣ Standardization & Normalization Rules
Identity

Single word

Lowercase

Market-recognized term

Examples:

software developer
student
designer
female

Numeric attributes

Must follow min / max / range

Exact = min = max

Normalize losslessly

Examples:

5 years → 60 months
age 25 → range.age [25,25]

Habits (FLAGS ONLY — critical)

Binary, explicit, deterministic.

Examples:

smoking: "no"
drinking: "no"
pets: "yes"


🚫 Do NOT use exclusions
🚫 Do NOT infer

6️⃣ What the model MUST do

✅ Extract only what user explicitly says about themselves
✅ Use flags for habits
✅ Normalize numeric values
✅ Use standard market words
✅ Leave empty if unstated

7️⃣ What the model MUST NEVER do

🚫 Infer self attributes from intent
🚫 Assume profession or skill
🚫 Copy other_party_preferences here
🚫 Convert education/country terms
🚫 Use exclusions instead of flags
🚫 Invent descriptors

8️⃣ Positive Examples (TRUE POSITIVES)
Example 1

Query

“I’m a software engineer with 4 years experience, non-smoker.”

"self_attributes": {
  "identity": [
    { "type": "profession", "value": "software engineer" }
  ],
  "min": {
    "time": [
      { "type": "experience", "value": 48, "unit": "month" }
    ]
  },
  "habits": {
    "smoking": "no"
  }
}

Example 2

Query

“25-year-old female, based in Bangalore.”

"self_attributes": {
  "identity": [
    { "type": "gender", "value": "female" },
    { "type": "location", "value": "bangalore" }
  ],
  "range": {
    "time": [
      { "type": "age", "min": 25, "max": 25, "unit": "year" }
    ]
  }
}

9️⃣ Negative Examples
❌ Hard Negative
"self_attributes": {
  "min": {
    "cost": [ { "type": "price", "value": 20000 } ]
  }
}


❌ Cost belongs to item/service, never the person.

❌ False Positive

Extracting attributes when user didn’t self-identify.

Query:

“Need a roommate”

❌ Any self_attributes → WRONG

❌ False Negative

Failing to extract explicit self description.

Query:

“I don’t smoke and I’m vegetarian”

❌ Empty self_attributes → WRONG

🔟 Edge Cases & Ambiguity Handling
“I prefer not to smoke”

→ ❌ This is preference, not self
→ Goes to other_party_preferences.habits

“I usually work weekends”

→ Only extract if clearly about SELF availability
Otherwise ignore.

“Experienced professional”

→ ❌ Ignore unless numeric provided

11️⃣ Validation Checks (MANDATORY)

Reject if:

Any product or service attribute appears

Any inferred self info exists

Habits are not binary

Numeric values lack min/max/range

Non-deterministic language appears

🔒 FINAL LOCK FOR self_attributes
self_attributes =
FACTS ABOUT USER ONLY
NO ASSUMPTIONS
NO ASPIRATION
NO PRODUCT LOGIC
FLAGS > EXCLUSIONS

### LOCATION

FIELD SPECIFICATION: LOCATION HANDLING

Covers Fields

target_location

location_match_mode

location_exclusions

These must always be reasoned together. No partial extraction.

1️⃣ Definition
target_location

Represents where matching should happen, not where the user currently is.

location_match_mode

Defines HOW location should be interpreted, never inferred.

location_exclusions

Defines places explicitly excluded by the user.

Location is query-driven only.
App/device/user profile location is injected later by the system — never by the model.

2️⃣ When these fields MUST be populated
location_match_mode

✅ ALWAYS populated
It is never optional.

target_location

Populate ONLY if:

A location is explicitly mentioned

OR movement / destination is mentioned

OR route is mentioned

location_exclusions

Populate ONLY if:

User explicitly excludes a place

3️⃣ When these fields MUST be empty
target_location MUST be {} when:

No explicit location is mentioned

Location is implied but not stated

location_exclusions MUST be [] when:

No exclusions are stated

4️⃣ Allowed Values & Structure (LOCKED)
location_match_mode (ENUM – FROZEN)
near_me | explicit | target_only | route | global

target_location shapes

Single location

"target_location": {
  "name": "bangalore"
}


Route

"target_location": {
  "origin": "delhi",
  "destination": "mumbai"
}


Global / Remote

"target_location": {}

location_exclusions
"location_exclusions": ["chennai", "noida"]


Lowercase

Plain names

No geo-coding

No hierarchy

5️⃣ Standardization & Normalization Rules

Preserve user-stated strings

Lowercase only

Do NOT infer country/state

Do NOT expand abbreviations unless obvious (blr → bangalore ❌)

Do NOT geo-resolve

6️⃣ LOCATION COT (FOR DATA GENERATION & FINE-TUNING ONLY)

This logic is allowed internally, but must NEVER appear in output.

Q6: Is a location explicitly mentioned?
    YES → extract
    NO → target_location = {}

Q7: Is there movement or relocation?
    "moving to", "relocating to" → target_only
    "travel from X to Y" → route

Q8: Are two locations mentioned?
    YES → origin + destination
    NO → single location

Q9: Is it remote / online / anywhere?
    YES → global

Q10: Are exclusions mentioned?
    YES → location_exclusions[]
    NO → []

Q11: Assign mode:
    Default → near_me

7️⃣ What the model MUST do

✅ Always assign location_match_mode
✅ Extract location ONLY if explicitly stated
✅ Distinguish static vs movement vs route
✅ Preserve exclusions exactly
✅ Leave target_location empty when unstated

8️⃣ What the model MUST NEVER do

🚫 Infer current location
🚫 Inject app/device location
🚫 Guess geography
🚫 Convert city → state → country
🚫 Add exclusions implicitly
🚫 Create new modes

9️⃣ Positive Examples (TRUE POSITIVES)
Example 1 — No location

Query

“Need a yoga instructor”

"target_location": {},
"location_match_mode": "near_me",
"location_exclusions": []

Example 2 — Explicit local

Query

“Looking for a plumber in Andheri West”

"target_location": { "name": "andheri west" },
"location_match_mode": "explicit",
"location_exclusions": []

Example 3 — Relocation

Query

“Moving to Pune, need a roommate”

"target_location": { "name": "pune" },
"location_match_mode": "target_only",
"location_exclusions": []

Example 4 — Travel route

Query

“Looking for a travel buddy from Bangalore to Goa”

"target_location": {
  "origin": "bangalore",
  "destination": "goa"
},
"location_match_mode": "route",
"location_exclusions": []

Example 5 — Remote / global

Query

“Looking for a remote frontend developer”

"target_location": {},
"location_match_mode": "global",
"location_exclusions": []

Example 6 — Exclusion

Query

“Need a flatmate in Bangalore, not Whitefield”

"target_location": { "name": "bangalore" },
"location_match_mode": "explicit",
"location_exclusions": ["whitefield"]

🔟 Negative Examples
❌ Hard Negative

Inferring location.

Query:

“Need a developer”

❌ Extracting any location → INVALID

❌ False Positive
"location_match_mode": "explicit"


when no location is mentioned → INVALID

❌ False Negative

Failing to extract exclusion.

Query:

“Anywhere except Delhi”

❌ location_exclusions empty → INVALID

11️⃣ Edge Cases & Ambiguity Handling
“near metro”

→ ❌ Too vague → ignore location

“around my office”

→ ❌ Not explicit → ignore

“within India”

→ Still geographic constraint but non-local
→ Treat as:

location_match_mode: "global"

“hybrid / remote preferred”

→ global

12️⃣ Validation Rules (MANDATORY)

Reject sample if:

location_match_mode missing

Mode contradicts query (e.g., route but only one city)

Exclusion overlaps with target_location

Location inferred, not stated

Mode outside ENUM

🔒 FINAL LOCK (SAVE THIS)
Location is query-driven.
Mode is mandatory.
No inference.
No geography resolution.
No creativity.

###
FIELD SPECIFICATION: reasoning
1️⃣ Definition

reasoning is a post-hoc factual justification of the final extracted output.

It explains WHAT was extracted and WHY,
NOT HOW the model thought.

Reasoning ≠ Chain-of-Thought
Reasoning ≠ Hidden deliberation
Reasoning ≠ Model thinking process

It is descriptive, not procedural.

2️⃣ When this field MUST be populated

✅ ALWAYS populated
No exceptions.

This is mandatory for:

SFT (Supervised Fine-Tuning)

PEFT (LoRA)

Validation

Drift detection

3️⃣ When this field MUST be empty

🚫 NEVER empty

If reasoning is missing → INVALID SAMPLE

4️⃣ Allowed Structure & Data Type
"reasoning": "<single paragraph string>"


Rules:

Exactly one paragraph

Plain English

Neutral, factual tone

Past tense preferred

2–5 sentences (recommended)

5️⃣ Standardization & Normalization Rules

Reference only extracted facts

Mention only fields that were populated

Do NOT restate schema keys

Do NOT add new facts

Do NOT mention rules, questions, or logic trees

6️⃣ What the model MUST do

✅ Describe:

Intent + subintent decision

Why certain items were extracted

Why constraints were placed as min/max/range

Why location mode was selected (if relevant)

Why exclusions were applied (if any)

✅ Keep language deterministic

“was classified as…”

“was extracted as…”

“was treated as…”

7️⃣ What the model MUST NEVER do (CRITICAL)

🚫 NEVER include step-by-step thinking

🚫 NEVER include words like:

“I thought”

“I analyzed”

“Then I decided”

“First / second / finally”

“Because it might mean…”

“I inferred”

“Possibly”

🚫 NEVER explore alternatives

🚫 NEVER justify ambiguity

🚫 NEVER explain rules or COT questions

🚫 NEVER leak internal logic

If ANY of these appear → REJECT SAMPLE

8️⃣ Positive Examples (TRUE POSITIVES)
✅ Example 1 — Product

Query

“Looking for a second hand iPhone under 30k, good condition”

Reasoning

“The query was classified as a product purchase because the user is seeking to buy a tangible item. An iPhone was extracted as the item type, with condition marked as used and a maximum price constraint extracted from the stated budget. The condition requirement was captured as a categorical attribute. No location was explicitly mentioned, so local matching applies.”

✅ Example 2 — Service

Query

“Need a yoga instructor who can come home in the mornings”

Reasoning

“The query was classified as a service request because the user is seeking professional assistance. Yoga instruction was extracted as the service item, with home visit mode and morning availability captured from explicit wording. No constraints were applied to pricing or experience. Since no location was specified, proximity-based matching applies.”

✅ Example 3 — Mutual

Query

“Looking for a trekking partner for weekends”

Reasoning

“The query was classified as a mutual intent because it involves shared participation between individuals. Trekking was identified as the subject of connection, with weekend availability extracted as a preference. No exclusions or location details were specified, so local matching applies.”

9️⃣ Negative Examples
❌ Chain-of-Thought (HARD NEGATIVE)

“First I checked whether this was a product or service. Then I noticed the word ‘looking’, so I assumed the user wants to hire someone. After that I…”

❌ INVALID
(Leaking reasoning process)

❌ False Positive (Invented Explanation)

“The user probably wants an experienced person, so experience was assumed.”

❌ INVALID
(Inference not stated in query)

❌ False Negative (Too Vague)

“The user wants something.”

❌ INVALID
(Does not justify extraction)

🔟 Edge Cases & Ambiguity Handling
Ambiguous query

“Need help with my laptop”

Allowed reasoning

“The query was classified as a service request because the user is seeking assistance. Laptop repair was extracted as the service item based on the context of help. No specific constraints or preferences were mentioned.”

✅ Do NOT mention ambiguity resolution
✅ Do NOT mention alternative interpretations

11️⃣ Validation Checks (MANDATORY)

Reject sample if:

Reasoning includes procedural words (first, then, etc.)

Reasoning mentions “inference”, “guess”, “assume”

Reasoning introduces facts not in output

Reasoning explains rules or questions

Reasoning length > 1 paragraph

Reasoning contradicts extracted fields

🔒 FINAL LOCK (SAVE THIS)
Reasoning explains the final output.
It never explains how the model decided.
It never thinks aloud.0 data 
### SEMANTIC UNDERSTANDING & CANONICALIZATION (CRITICAL)

----------------------------------------------------------------
CORE PRINCIPLE: SEMANTIC UNDERSTANDING, NOT KEYWORD MATCHING
----------------------------------------------------------------

The model MUST understand MEANING, not match keywords.

Different words with SAME meaning → SAME canonical output
Same word with DIFFERENT meaning → DIFFERENT output based on context

This enables deterministic SQL matching.

----------------------------------------------------------------
HIERARCHICAL ATTRIBUTE EXTRACTION (IMPLICATION RULES)
----------------------------------------------------------------

When attribute X IMPLIES attribute Y, extract BOTH.

Rule: Child attributes ALWAYS imply parent attributes.

ITEM CONDITION HIERARCHY:
```
used (PARENT)
├── single owner → condition: "used" + ownership: "single"
├── second owner → condition: "used" + ownership: "second"
├── multiple owners → condition: "used" + ownership: "multiple"
├── first owner → condition: "used" + ownership: "first"
└── (unspecified) → condition: "used" only

new (PARENT)
├── sealed/unopened → condition: "new" + packaging: "sealed"
├── open box → condition: "new" + packaging: "open-box"
└── (unspecified) → condition: "new" only

refurbished (PARENT)
├── certified refurbished → condition: "refurbished" + certification: "certified"
├── seller refurbished → condition: "refurbished" + certification: "seller"
└── (unspecified) → condition: "refurbished" only
```

EXAMPLES:
| User Says | Understands | Extracts |
|-----------|-------------|----------|
| "single owner car" | used + ownership info | condition: "used", ownership: "single" |
| "used car" | pre-owned | condition: "used" |
| "second hand bike" | pre-owned | condition: "used" |
| "old laptop" | pre-owned | condition: "used" |
| "purana phone" | pre-owned (Hindi) | condition: "used" |
| "pre-owned watch" | pre-owned | condition: "used" |
| "2nd hand furniture" | pre-owned | condition: "used" |
| "first owner bike" | used + first owner | condition: "used", ownership: "first" |

MATCHING LOGIC:
- Buyer searches "used car" → SQL: WHERE condition = 'used' → Returns ALL used cars (including single owner)
- Buyer searches "single owner car" → SQL: WHERE condition = 'used' AND ownership = 'single' → Returns ONLY single owner

----------------------------------------------------------------
VEHICLE-SPECIFIC HIERARCHIES
----------------------------------------------------------------

FUEL TYPE:
```
fuel (PARENT)
├── petrol → fuel: "petrol"
├── diesel → fuel: "diesel"
├── electric → fuel: "electric"
├── hybrid → fuel: "hybrid"
├── cng → fuel: "cng"
└── lpg → fuel: "lpg"
```

TRANSMISSION:
```
transmission (PARENT)
├── manual → transmission: "manual"
├── automatic → transmission: "automatic"
├── amt → transmission: "amt"
├── cvt → transmission: "cvt"
└── dct → transmission: "dct"
```

BODY TYPE:
```
vehicle_type (PARENT - for cars)
├── sedan → vehicle_type: "sedan"
├── suv → vehicle_type: "suv"
├── hatchback → vehicle_type: "hatchback"
├── muv → vehicle_type: "muv"
└── coupe → vehicle_type: "coupe"
```

----------------------------------------------------------------
REAL ESTATE HIERARCHIES
----------------------------------------------------------------

FURNISHING:
```
furnishing (PARENT)
├── furnished → furnishing: "furnished"
├── semi-furnished → furnishing: "semi-furnished"
└── unfurnished → furnishing: "unfurnished"
```

PROPERTY TYPE:
```
property_type (PARENT)
├── apartment/flat → property_type: "apartment"
├── independent house → property_type: "house"
├── villa → property_type: "villa"
├── plot → property_type: "plot"
└── pg/hostel → property_type: "pg"
```

BHK EXTRACTION:
| User Says | Extracts |
|-----------|----------|
| "2BHK flat" | bedrooms: 2, property_type: "apartment" |
| "3BHK apartment" | bedrooms: 3, property_type: "apartment" |
| "1RK" | bedrooms: 1, property_type: "apartment", layout: "rk" |

----------------------------------------------------------------
PERSON ATTRIBUTE HIERARCHIES
----------------------------------------------------------------

DIET:
```
diet (PARENT)
├── vegetarian/veg → diet: "vegetarian"
├── non-vegetarian/non-veg → diet: "non-vegetarian"
├── vegan → diet: "vegan"
├── eggetarian → diet: "eggetarian"
└── jain → diet: "jain"
```

LIFESTYLE FLAGS:
```
habits (BINARY FLAGS ONLY)
├── smoking: "yes" | "no"
├── drinking: "yes" | "no"
├── pets: "yes" | "no"
└── early_riser: "yes" | "no"
```

SEMANTIC EXPANSIONS:
| User Says | Expands To |
|-----------|------------|
| "no bad habits" | smoking: "no", drinking: "no" |
| "clean lifestyle" | smoking: "no", drinking: "no" |
| "teetotaler" | drinking: "no" |
| "non-smoker" | smoking: "no" |
| "pet-friendly" | pets: "yes" |

----------------------------------------------------------------
UNIVERSAL CANONICALIZATION RULES (PATTERN-BASED)
----------------------------------------------------------------

The model CANNOT memorize all possible values - infinite exist in the world.
Instead, it learns PATTERNS for extracting, standardizing, and normalizing.

CORE PRINCIPLE:
The model learns HOW to identify and process attributes, NOT specific lists.
This makes it FUTURE-PROOF for any unseen product/service/attribute.

----------------------------------------------------------------
CATEGORICAL vs NUMERIC CLASSIFICATION
----------------------------------------------------------------

For EVERY attribute in a query, ask:
"Can this attribute have INFINITE measurable values?"

YES → NUMERIC
- Goes to min/max/range with axis
- Examples: price, storage, experience, age, area, odometer

NO → CATEGORICAL
- Goes to categorical: { key: value }
- Examples: condition, fuel, color, brand, gender, diet, transmission

----------------------------------------------------------------
CATEGORICAL ATTRIBUTES
----------------------------------------------------------------

Definition: Finite discrete choices that cannot be measured with numbers

Structure:
"categorical": {
  "<key>": "<value>"
}

KEY RULES:
- Must be market-standard term
- Lowercase always
- No underscores (use compound nouns if needed)
- Examples: condition, fuel, transmission, furnishing, color, brand

VALUE RULES:
- Canonical form, lowercase
- Standardize synonyms to single term
- Market-recognized choices

EXAMPLES:
| Query Says | Key | Value |
|------------|-----|-------|
| "second hand phone" | condition | used |
| "diesel car" | fuel | diesel |
| "automatic transmission" | transmission | automatic |
| "SSD laptop" | drive | ssd |
| "semi-furnished flat" | furnishing | semi-furnished |
| "red color" | color | red |
| "Apple iPhone" | brand | apple |

----------------------------------------------------------------
NUMERIC ATTRIBUTES
----------------------------------------------------------------

Definition: Measurable quantities with units that map to axes

Structure: min/max/range with axis array
"min": { "<axis>": [{ "type": "", "value": <num>, "unit": "" }] }
"max": { "<axis>": [{ "type": "", "value": <num>, "unit": "" }] }
"range": { "<axis>": [{ "type": "", "min": <num>, "max": <num>, "unit": "" }] }

CONSTRAINT DETECTION (SEMANTIC, NOT KEYWORD):
----------------------------------------------------------------

DEFAULT RULE: A value WITHOUT any modifier is EXACT.
The model must SEMANTICALLY understand the constraint, not match keywords.

| Query | Semantic Meaning | Structure |
|-------|------------------|-----------|
| "128GB" | exactly 128 (no modifier = exact) | range: { min=128, max=128 } |
| "128GB storage" | exactly 128 (no modifier = exact) | range: { min=128, max=128 } |
| "exactly 128GB" | exactly 128 (explicit exact) | range: { min=128, max=128 } |
| "under 50k" | less than 50k | max: { value=50000 } |
| "below 30000" | less than 30000 | max: { value=30000 } |
| "budget 2 lakhs" | at most 2 lakhs | max: { value=200000 } |
| "within 5km" | at most 5km | max: { value=5 } |
| "at least 3 years" | 3 years or more | min: { value=36 } (months) |
| "minimum 5 star" | 5 star or more | min: { value=5 } |
| "above 1500 sqft" | more than 1500 | min: { value=1500 } |
| "15000 km done" | exactly 15000 (stated fact) | range: { min=15000, max=15000 } |
| "asking 85k" | exactly 85k (stated price) | range: { min=85000, max=85000 } |
| "between 20-30 lakhs" | 20 to 30 range | range: { min=2000000, max=3000000 } |
| "10 to 15 years" | 10 to 15 range | range: { min=120, max=180 } (months) |

SEMANTIC UNDERSTANDING (NOT keyword matching):
- "128GB" = user wants EXACTLY 128GB (no flexibility stated)
- "under 50k" = user wants LESS THAN 50k (upper bound)
- "at least 8GB" = user wants 8GB OR MORE (lower bound)
- "5 years experience" = user has/wants EXACTLY 5 years

**CRITICAL: EXACT DOES NOT EXIST AS A FIELD. Exact = range with min=max**
**CRITICAL: No modifier = EXACT (default behavior)**

AXIS MAPPING (10 FIXED AXES - NEVER CHANGES):
| Attribute Type | Axis |
|----------------|------|
| price, budget, salary, cost | cost |
| RAM, storage, rooms, seats | capacity |
| speed, odometer, mileage, refresh rate | performance |
| rating, grade, condition level | quality |
| count, number, quantity | quantity |
| age, experience, duration, usage | time |
| area, distance, dimensions | space |
| gender, profession, certification | identity |
| delivery mode, service mode | mode |
| certifications, proficiency level | skill |

----------------------------------------------------------------
CANONICALIZATION PROCESS (HOW MODEL WORKS)
----------------------------------------------------------------

Step 1: EXTRACT - Identify attribute from query
Step 2: CLASSIFY - Categorical or numeric?
Step 3: STANDARDIZE - Key/value or axis/type to market terms
Step 4: NORMALIZE - Units (if numeric)
Step 5: PLACE - Correct structure in schema

SEMANTIC EQUIVALENCE EXAMPLES (Model learns PATTERNS, not lists):

Condition synonyms → "used":
- "second hand" → used
- "pre-owned" → used
- "purana" (Hindi) → used
- "old" (in item context) → used
- "2nd hand" → used

Fuel synonyms:
- "petrol" / "gasoline" → petrol (India context)
- "diesel" / "gasoil" → diesel

Storage type:
- "SSD" / "Solid State" → ssd
- "HDD" / "Hard Disk" → hdd

The model GENERALIZES from patterns, NOT memorizes lists.

----------------------------------------------------------------
IMPLICATION RULES (Child implies Parent)
----------------------------------------------------------------

When a CHILD attribute is stated, the PARENT must also be extracted.

OWNERSHIP → CONDITION:
| User Says | Extracts |
|-----------|----------|
| "single owner" | condition: "used", ownership: "single" |
| "first owner" | condition: "used", ownership: "first" |
| "second owner" | condition: "used", ownership: "second" |
| "used car" | condition: "used" (no ownership) |

PACKAGING → CONDITION:
| User Says | Extracts |
|-----------|----------|
| "sealed box" | condition: "new", packaging: "sealed" |
| "open box" | condition: "new", packaging: "open-box" |

----------------------------------------------------------------
MATCHING LOGIC (Specific vs Vague)
----------------------------------------------------------------

Extraction determines matching. More specific = narrower SQL match.

RULE: Extract ONLY what is stated.

| User Says | Extracts | SQL Matches |
|-----------|----------|-------------|
| "used car" | condition: "used" | ALL used cars |
| "single owner car" | condition: "used", ownership: "single" | ONLY single owner |
| "car" | (no condition) | ALL cars (new + used) |
| "laptop with SSD" | drive: "ssd" | ONLY SSD laptops |
| "laptop" | (no drive) | ALL laptops |

The extraction is DETERMINISTIC.
The matching is NATURAL consequence of SQL WHERE clauses.

----------------------------------------------------------------
GENERALIZATION (FUTURE-PROOF)
----------------------------------------------------------------

FIXED ELEMENTS (Never changes):
- 14 fields
- 10 axes
- Schema structure
- Constraint model (min/max/range)

FLEXIBLE ELEMENTS (Model generalizes):
- Item types (any market noun)
- Attribute keys (any market-standard categorical)
- Attribute types within axes (any measurable attribute)
- Values (standardized from query)

EXAMPLE - UNSEEN ITEM:
Query: "Selling my quantum computer with 1000 qubits"
```json
{
  "type": "quantum computer",
  "min": {
    "capacity": [
      { "type": "qubits", "value": 1000, "unit": "count" }
    ]
  }
}
```
- "quantum computer" = new type (valid market noun)
- "qubits" = new attribute type (follows pattern)
- Schema remains identical

EXAMPLE - UNSEEN ATTRIBUTE:
Query: "Looking for electric scooter with 100km range"
```json
{
  "type": "scooter",
  "categorical": { "fuel": "electric" },
  "min": {
    "performance": [
      { "type": "range", "value": 100, "unit": "km" }
    ]
  }
}
```

----------------------------------------------------------------
ITEMS STRUCTURE (COMPLETE)
----------------------------------------------------------------

```json
"items": [
  {
    "type": "<canonical market noun>",
    "categorical": {
      "<key>": "<value>"
    },
    "min": {
      "<axis>": [{ "type": "", "value": <number>, "unit": "" }]
    },
    "max": {
      "<axis>": [{ "type": "", "value": <number>, "unit": "" }]
    },
    "range": {
      "<axis>": [{ "type": "", "min": <number>, "max": <number>, "unit": "" }]
    }
  }
]
```

KEY RULES:
- type: Canonical market noun (what the item IS)
- categorical: Non-numeric attributes
  - key: market-standard term (lowercase, no underscores)
  - value: standardized choice (lowercase)
- min/max/range: Numeric attributes mapped to 10 axes

CATEGORICAL vs NUMERIC:
- categorical: Finite discrete choices (condition, fuel, color, brand)
- min/max/range: Measurable quantities (price, storage, experience, age)

EXACT VALUES (NUMERIC):
- Use range with min = max
- Example: "exactly 16GB RAM"
```json
"range": {
  "capacity": [{ "type": "memory", "min": 16, "max": 16, "unit": "gb" }]
}
```

----------------------------------------------------------------
VALIDATION FOR SEMANTIC EXTRACTION
----------------------------------------------------------------

A sample is VALID only if:
1. Implied parent attributes are extracted (single owner → condition: used + ownership: single)
2. Categorical keys/values are market-standard terms (lowercase, no underscores)
3. Synonyms are canonicalized to standard terms
4. Schema structure is preserved (14 fields, 10 axes)
5. Semantic understanding demonstrated (not keyword matching)
6. All stated attributes are extracted (nothing skipped)
7. No unstated attributes are added (no inference)

A sample is INVALID if:
1. Parent attribute missing when child is present
2. Non-standard key/value used (invented terms)
3. Keyword matched instead of semantically understood
4. Schema structure violated (new fields/axes)
5. Stated attribute missing from output
6. Unstated attribute added (inference detected)

### EXTRACTION FLOW SUMMARY

```
STEP 1: CLASSIFY (Q1-Q5)
→ Determine intent + sub_intent

STEP 2: LOCATE (Q6-Q11)
→ Extract location, mode, exclusions

STEP 3: CATEGORIZE (Q12-Q14)
→ Assign domain OR mutual_category

STEP 4: EXTRACT ITEMS (Q15-Q19)
→ Items, attributes, item_exclusions

STEP 5: EXTRACT PEOPLE (Q20-Q22)
→ other_party_preferences, self_attributes

STEP 6: NORMALIZE (Q23-Q24)
→ Convert units, resolve polysemy

STEP 7: VALIDATE (Q25-Q26)
→ Check all 12 fields, valid JSON
```


### 
NORMALIZATION & UNIT CONVERSION (Q23-Q24)

**Covers**: All numeric fields across schema

**IMPORTANT**: Units are CONTEXT-DEPENDENT, not fixed. The model learns to REASON about appropriate units based on domain and attribute type.

```
Q23: Determine category and apply appropriate normalization:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STEP 1: Is this UNIVERSAL or COUNTRY-SPECIFIC?                          │
    └────────────────────────────────┬────────────────────────────────────────┘
                                     │
               ┌─────────────────────┴─────────────────────┐
               ▼                                           ▼
          UNIVERSAL                                 COUNTRY-SPECIFIC
    (Physics doesn't change)                    (Value changes by country)
               │                                           │
               ▼                                           ▼
    ┌──────────────────────┐                    Preserve Type Always!
    │ STEP 2: Needs        │                    • Currency: {"max": 5000, "currency": "USD"}
    │ Generalization?      │                    • Clothing: {"size": "8", "system": "US"}
    └──────────┬───────────┘                    • Shoe: {"size": "10", "system": "US"}
               │                                • Grade: {"grade": "3.5", "system": "GPA"}
        ┌──────┴──────┐
        ▼             ▼
       YES            NO
    (Multiple     (Already Global
     units exist)  Standard)
        │             │
        ▼             ▼
    NORMALIZE      KEEP AS-IS
    to standard    (Industry std)
        │             │
        ▼             ▼
    100000m → 100km   pixels: "1080p"
    1024 miles → 1638km   carats: 2
    5 years → 60 months   BHP: 150
    1 TB → 1024 GB        Mbps: 100

# GLOBAL REFERENCE CONTEXT — VRIDDHI
(Read-only · Injected into EVERY stage · Never edited during generation)

This document defines the permanent, system-level truths.
It contains NO TASKS and NO PROMPTS.

Any output that violates this file is INVALID.

----------------------------------------------------------------
CORE PHILOSOPHY (LOCKED)
----------------------------------------------------------------

The model extracts facts.
The system standardizes and canonicalizes.
The model NEVER assumes, invents, infers, or converts implicitly.

Strict separation:

User Query
→ Extraction (LLM, deterministic)
→ Normalization / Standardization (rule-based system)
→ Canonicalization (post-processing)
→ Matching

If this boundary is violated, the system becomes non-deterministic.

----------------------------------------------------------------
MODEL RESPONSIBILITIES (DETERMINISTIC EXTRACTION)
----------------------------------------------------------------

The model's ONLY job is to:

1. EXTRACT
   - Identify ALL attributes mentioned in query
   - Don't add what's not stated
   - Don't skip what is stated

2. STANDARDIZE
   - Key: canonical attribute category (lowercase, market term)
   - Value: canonical attribute value (lowercase, market term)
   - Examples: "second hand" → "used", "SSD" → "ssd"

3. NORMALIZE
   - Convert units to standard: 3 years → 36 months
   - Convert storage: 1TB → 1024GB
   - Preserve currency type (don't convert USD to INR)

4. OUTPUT IN FIXED SCHEMA
   - 14 fields (fixed)
   - 10 axes (fixed)
   - categorical{} for non-numeric
   - min/max/range{} for numeric
   - NO new fields, NO creativity

MODEL MUST NEVER:
✗ Infer attributes not stated in query
✗ Guess missing values
✗ Create new fields or axes
✗ Add subjective interpretation
✗ Be creative with key/value naming
✗ Skip stated attributes

----------------------------------------------------------------

   UNIVERSAL + NEEDS GENERALIZATION (normalize to standard unit):

    Different expressions, SAME actual value - normalize for comparison:
    │ Distance: 100000 meters, 1024 miles, 50 km → all to KM
    │ Weight: 500 grams, 2 pounds, 5 kg → all to KG
    │ Storage: 1 TB, 512 MB, 256 GB → all to GB
    │ Time: 5 years, 60 months, 1825 days → all to MONTHS
    │ Area: 1000 sqft, 2 acres, 500 sqm → all to SQM
    │
    │ → Output: {"min": 100} (just the normalized number)

    ═══════════════════════════════════════════════════════════════════════════

    UNIVERSAL + GLOBAL STANDARD (keep as-is, no conversion needed):

    Industry uses these units globally - no alternative units exist:
    │ Display: pixels (1080p, 4K), inches (screen size)
    │ Jewelry: carats (nobody says "0.4 gram diamond")
    │ Vehicles: BHP/HP, CC/L (same worldwide)
    │ Internet: Mbps/Gbps (same worldwide)
    │ Camera: megapixels (same worldwide)
    │
    │ → Output: {"resolution": "4K"} or {"power": 150} (keep original unit)

    ═══════════════════════════════════════════════════════════════════════════

    COUNTRY-SPECIFIC (preserve type - values differ by country):

    Same number means DIFFERENT things in different countries:
    │ Currency: 5000 USD ≠ 5000 INR ≠ 5000 EUR
    │ Clothing Size: US 8 ≠ UK 8 ≠ EU 38
    │ Shoe Size: US 10 ≠ UK 9 ≠ EU 44
    │ Education: 3.5 GPA (US) ≠ 85% (India) ≠ First Class (UK)
    │
    │ → Output: {"max": 5000, "currency": "USD"}
    │ → Output: {"size": "8", "system": "US"}

    "5 years experience" → 60 months
    "60 months experience" → 60 months
    "1825 days experience" → 60 months
    ALL become {"min": 60} → They can MATCH!

    STANDARD UNITS BY ATTRIBUTE TYPE:

    EXPERIENCE/DURATION → MONTHS (standard)
    - 1 year = 12 months
    - 1 week = 0.25 months
    - 1 day = 0.033 months

    DEADLINE/URGENCY → HOURS (standard)
    - 1 day = 24 hours
    - 1 week = 168 hours
    - 1 month = 720 hours

    AGE → YEARS (standard)
    - Keep as years (natural measurement)

    STORAGE → GB (standard)
    - 1 TB = 1024 GB
    - 1 MB = 0.001 GB
    - 1 PB = 1048576 GB

    CURRENCY → BASE UNIT + CURRENCY LABEL (preserve both!)
    - Normalize amount: k = ×1000, M = ×1000000, lakh = ×100000, crore = ×10000000
    - PRESERVE currency type: USD, EUR, INR, AED, GBP, JPY, etc.
    - Format: {"max": 50000, "currency": "USD"}
    - If no currency mentioned: {"max": 50000} (infer from location or omit)

    DISTANCE → KILOMETERS (standard)
    - 1 mile = 1.6 km
    - 1 meter = 0.001 km

    AREA → SQUARE METERS/SQM (standard)
    - 1 sqft = 0.093 sqm
    - 1 acre = 4047 sqm
    - 1 hectare = 10000 sqm

    WEIGHT → KILOGRAMS (standard)
    - 1 gram = 0.001 kg
    - 1 pound = 0.45 kg
    - 1 ton = 1000 kg

    PROFICIENCY → 1-5 SCALE (standard)
    - fresher/novice → 1
    - beginner/basic → 2
    - intermediate/experienced → 3
    - advanced/expert/proficient → 4
    - master/guru/specialist → 5

    DOMAIN-SPECIFIC (keep as-is):
    - Jewelry: carats (industry standard)
    - Vehicles: BHP/HP, CC/L
    - Display: inches, pixels
    - Internet: Mbps/Gbps

Q24: Handle POLYSEMY (same word, different meanings based on context):

    WHAT IS POLYSEMY?
    Same word can mean different things in different contexts.
    The model uses DOMAIN + INTENT + WHO context to determine:
    1. What the word MEANS in this context
    2. WHERE it goes in the schema

    HOW TO RESOLVE + WHERE IT GOES:

    "language":
    ├── Tech domain + about code → programming language
    │   WHERE: items[].attributes: {code: ["python", "rust"]}
    ├── About OTHER person (seller/provider/partner) → speaking language
    │   WHERE: other_party_preferences: {language: "spanish"}
    └── About SELF ("I speak...") → speaking language
        WHERE: self_attributes: {language: "english"}

    Example: "developer who speaks Spanish and knows Python"
    → other_party_preferences: {language: "spanish"}
    → items[].attributes: {code: ["python"]}
    (BOTH can exist in same query!)

    "size":
    ├── Tech domain (storage context)
    │   WHERE: items[].attributes: {storage: 256}
    ├── Fashion domain (clothing context)
    │   WHERE: items[].attributes: {size: "XL"}
    └── Real Estate domain (area context)
        WHERE: items[].attributes: {area: {"min": 93}}

    "experience":
    ├── Time-based ("5 years experience", "60 months exp")
    │   WHERE: other_party_preferences OR self_attributes (based on WHO)
    │   VALUE: {experience: {"min": 60}} (normalized to months)
    └── Skill-based ("experienced person", "expert level")
        WHERE: other_party_preferences OR self_attributes (based on WHO)
        VALUE: {proficiency: {"min": 3}} OR {proficiency: {"min": 4}}
    ⚠️ NEVER convert between them - they are DIFFERENT dimensions!

    "condition":
    ├── Product domain → item condition
    │   WHERE: items[].attributes: {condition: "excellent"}
    └── Health/Medical context → health condition
        WHERE: context-specific field

    KEY PRINCIPLE:
    These are EXAMPLES to teach the model the PATTERN:
    1. Identify the ambiguous word
    2. Use CONTEXT (domain, intent, who) to determine meaning
    3. Route to correct FIELD based on what/who it describes
```
Q26: VALIDATE JSON structure:
    All 12 fields present?
    Valid JSON syntax?
    Reasoning field explains classification?

----------------------------------------------------------------
1. LOCKED 14-FIELD OUTPUT SCHEMA (FROZEN)
----------------------------------------------------------------

Classification (ALWAYS present):
1. intent
2. subintent
3. domain
4. primary_mutual_category

Extraction:
5. items
6. item_exclusions
7. other_party_preferences
8. other_party_exclusions
9. self_attributes
10. self_exclusions
11. target_location
12. location_match_mode
13. location_exclusions
14. reasoning

No extra fields allowed.
No renaming.
No reordering.
Empty arrays / objects are VALID.

----------------------------------------------------------------
2. INTENTS (ENUM · FIXED)
----------------------------------------------------------------

intent ∈ { product | service | mutual }

----------------------------------------------------------------
3. SUBINTENTS (ENUM · FIXED)
----------------------------------------------------------------

product → buy | sell  
service → seek | provide  
mutual  → connect

----------------------------------------------------------------
4. LOCATION MATCH MODES (ENUM · FIXED)
----------------------------------------------------------------

near_me | explicit | target_only | route | global

----------------------------------------------------------------
5. LOCATION RULES (QUERY-DRIVEN ONLY)
----------------------------------------------------------------

• No location mentioned → near_me  
• Explicit place name → explicit  
• Relocating / moving → target_only  
• Travel from X to Y → route  
• Remote / anywhere / online → global  

The system MAY inject app/GPS location later.
The model NEVER assumes location.

----------------------------------------------------------------
6. ATTRIBUTE AXES (CLOSED SET · NEVER CHANGES)
----------------------------------------------------------------

Every extracted fact MUST map to exactly one axis.

identity  
capacity  
performance  
quality  
quantity  
time  
space  
cost  
mode  
skill  

No new axes ever.

----------------------------------------------------------------
7. ATTRIBUTE TYPE NAMING RULES (NO CREATIVITY)
----------------------------------------------------------------

Type MUST be:
• Single word
• Lowercase
• Market-standard noun
• No underscores
• No invention

Examples:
price, experience, storage, memory, battery,
speed, refresh, rating, age, duration

----------------------------------------------------------------
8. CONSTRAINT MODEL (ABSOLUTE)
----------------------------------------------------------------

Exact DOES NOT exist.

Exact = min = max.

Allowed blocks ONLY:
min | max | range

----------------------------------------------------------------
9. CONSTRAINT SHAPE (FROZEN)
----------------------------------------------------------------

min / max:
{
  "<axis>": [
    {
      "type": "<market term>",
      "value": <number|string>,
      "unit": "<standard | local | omitted>"
    }
  ]
}

range:
{
  "<axis>": [
    {
      "type": "<market term>",
      "min": <number>,
      "max": <number>,
      "unit": "<standard | omitted>"
    }
  ]
}

❌ Axis MUST NEVER appear outside min/max/range.

----------------------------------------------------------------
10. NORMALIZATION & STANDARDIZATION RULES
----------------------------------------------------------------

Normalization happens AFTER extraction.
Extraction NEVER assumes convertibility.

A value MAY be normalized ONLY IF:
1) Axis is known
2) Value is explicit in query
3) Conversion is lossless & deterministic
4) Unit meaning is globally stable

----------------------------------------------------------------
11. STANDARDIZABLE AXES (MINIMUM GUARANTEE)
----------------------------------------------------------------

Convertible (standardize to base unit):

• time     → month
• distance → meter
• area     → sqm
• weight   → kg
• storage  → byte (gb allowed)

Universal (NEVER converted):
• pixels
• hz
• count
• boolean

----------------------------------------------------------------
12. COUNTRY-SPECIFIC / NON-CONVERTIBLE
----------------------------------------------------------------

• currency  → preserve if explicit, else unit=local
• education → preserve label exactly
• legal     → preserve term exactly

❌ NEVER infer currency
❌ NEVER convert currency
❌ NEVER normalize education or legal labels

----------------------------------------------------------------
13. NEW / UNKNOWN UNITS (FUTURE-PROOF RULE)
----------------------------------------------------------------

If axis is known but unit is unknown:
• Preserve unit verbatim
• Do NOT convert
• Do NOT reject

Example:
“speed 5 warp-units”
→ axis: performance
→ type: speed
→ value: 5
→ unit: warp-units

Schema NEVER changes.

----------------------------------------------------------------
14. NO-ASSUMPTION RULE (ABSOLUTE)
----------------------------------------------------------------

If NOT stated in the query:
• Do not guess
• Do not infer
• Do not convert
• Do not enrich

Empty is VALID.
Silence is VALID.

----------------------------------------------------------------
15. REASONING (NOT CHAIN-OF-THOUGHT)
----------------------------------------------------------------

Purpose:
• SFT alignment
• Deterministic justification

Rules:
• Single paragraph
• Descriptive, post-hoc
• Explains WHAT was extracted and WHY
• No step-by-step thinking
• No alternatives
• No hidden deliberation

Chain-of-Thought is FORBIDDEN.

----------------------------------------------------------------
FINAL INVARIANT (LOCK THIS)
----------------------------------------------------------------

Everything in the world reduces to:
• a fixed semantic axis
• a market-standard type
• an explicit value
• an optional unit
• a constraint (min | max | range)

Exact does not exist.
Inference does not exist.
Only facts exist. 