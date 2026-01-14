"""
VRIDDHI MATCHING SYSTEM - ITEM ARRAY MATCHING & ITEM EXCLUSIONS
Phase 2.4

Authority: MATCHING_CANON.md v1.1 (LOCKED)
Input Contract:
  - Normalized by schema_normalizer.py (Phase 2.1)
  - Item matching from item_matchers.py (Phase 2.3)
  - Numeric logic from numeric_constraints.py (Phase 2.2)

Purpose: Match arrays of items with exclusion handling
Scope: Item-array-level matching ONLY - NO listing orchestration

Implements Canon Rules:
- M-07 to M-11: Via item_matchers.item_matches() (Phase 2.3)
- M-12: Item Exclusion Disjoint Rule (NEW in this phase)

Matching Semantics:
- Required coverage: ∀ r ∈ required_items: ∃ c ∈ candidate_items: valid(r,c)
- Item exclusions: required.itemexclusions ∩ Flatten(candidate) = ∅ (STRICT)
- Vacuous: empty required_items → PASS
- Failure: any required item without valid candidate → FAIL
"""

from typing import List, Dict, Set, Any, Callable
from item_matchers import item_matches, ImplicationFn


# ============================================================================
# HELPER: FLATTEN ITEM VALUES
# ============================================================================

def flatten_item_values(item: Dict[str, Any]) -> Set[str]:
    """
    Recursively extract all string values from an item.

    Used by M-12 (Item Exclusion Disjoint Rule) to check if candidate item
    contains any excluded values.

    Extraction Rules:
    - Extracts from categorical, min/max/range keys (not values - those are numeric)
    - Extracts from type
    - Recursively processes nested dicts and lists
    - Converts all to lowercase (normalized)
    - Skips non-string values (numbers, booleans, etc.)

    Args:
        item: Item dict (schema-normalized)

    Returns:
        Set of all string values found in item (lowercase)

    Examples:
        flatten_item_values({
            "type": "smartphone",
            "categorical": {"brand": "apple", "color": "black", "condition": "refurbished"}
        }) → {"smartphone", "apple", "black", "refurbished"}

        flatten_item_values({
            "type": "laptop",
            "categorical": {"brand": "dell", "condition": "new"}
        }) → {"laptop", "dell", "new"}

    Note:
        - Numeric values in min/max/range are NOT extracted (they're constraints, not attributes)
        - Keys are NOT extracted (only values)
        - Already normalized by Phase 2.1 (lowercase, trimmed)
    """
    values = set()

    def extract_strings(obj: Any) -> None:
        """Recursively extract string values."""
        if isinstance(obj, dict):
            for value in obj.values():
                extract_strings(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_strings(item)
        elif isinstance(obj, str):
            # Already normalized by Phase 2.1 (lowercase, trimmed)
            values.add(obj)
        # Skip other types (int, float, bool, None)

    # Extract type (always present, guaranteed by Phase 2.1)
    if "type" in item:
        values.add(item["type"])

    # Extract from categorical values
    if "categorical" in item:
        extract_strings(item["categorical"])

    # Note: We do NOT extract from min/max/range because those contain
    # numeric values (constraints), not categorical string values

    return values


# ============================================================================
# M-12: ITEM EXCLUSION DISJOINT RULE
# ============================================================================

def violates_item_exclusions(
    required_item: Dict[str, Any],
    candidate_item: Dict[str, Any]
) -> bool:
    """
    Check if candidate item violates required item's exclusions.

    Enforces Canon Rule M-12: Item Exclusion Disjoint Rule

    Formal Expression:
        ∀ items ia ∈ A.items, ib ∈ B.items where ia.type = ib.type:
          Match(ia, ib) ⇒ A.itemexclusions ∩ Flatten(ib) = ∅

    Preconditions:
        - Both items are schema-normalized
        - itemexclusions is array of strings (lowercase, trimmed)
        - Candidate item has extractable string values

    Pass Condition (exclusions NOT violated):
        ∀ excl ∈ required.itemexclusions:
          excl ∉ Flatten(candidate_item)

    Fail Condition (exclusions violated):
        ∃ excl ∈ required.itemexclusions:
          excl ∈ Flatten(candidate_item)

    Args:
        required_item: Item from listing A (with exclusions)
        candidate_item: Item from listing B (being checked)

    Returns:
        True if exclusions are VIOLATED (candidate invalid)
        False if exclusions are NOT violated (candidate valid)

    Examples:
        # Exclusion violated
        violates_item_exclusions(
            {"itemexclusions": ["refurbished", "used"]},
            {"categorical": {"condition": "refurbished"}}
        ) → True (VIOLATED - contains excluded value)

        # Exclusions not violated
        violates_item_exclusions(
            {"itemexclusions": ["refurbished", "used"]},
            {"categorical": {"condition": "new"}}
        ) → False (NOT VIOLATED - no excluded values)

        # Empty exclusions (always valid)
        violates_item_exclusions(
            {"itemexclusions": []},
            {"categorical": {"condition": "refurbished"}}
        ) → False (no exclusions to violate)

        # Multiple exclusions, one violated
        violates_item_exclusions(
            {"itemexclusions": ["dealer", "agent", "broker"]},
            {"categorical": {"seller_type": "dealer"}}
        ) → True (VIOLATED)

    Notes:
        - Exclusions are STRICT (I-07: Exclusion Strictness Invariant)
        - Any overlap → DISJOINT requirement violated
        - Empty exclusions → no restrictions
        - Case-insensitive (Phase 2.1 normalized)
    """
    # Get exclusions from required item (Phase 2.1 ensures this is a list)
    required_exclusions = required_item.get("itemexclusions", [])

    # Empty exclusions → no restrictions, no violations
    if not required_exclusions:
        return False

    # Extract all string values from candidate
    candidate_values = flatten_item_values(candidate_item)

    # M-12: Check for any overlap (STRICT disjoint requirement)
    # If ANY exclusion appears in candidate → VIOLATED
    for exclusion in required_exclusions:
        if exclusion in candidate_values:
            return True  # VIOLATION FOUND

    # No violations found
    return False


# ============================================================================
# REQUIRED ITEM MATCHING (SINGLE)
# ============================================================================

def required_item_has_match(
    required_item: Dict[str, Any],
    candidate_items: List[Dict[str, Any]],
    implies_fn: ImplicationFn = None
) -> bool:
    """
    Check if a single required item has at least one valid matching candidate.

    Semantics:
        ∃ c ∈ candidate_items:
          ¬violates_item_exclusions(required_item, c) ∧
          item_matches(required_item, c, implies_fn)

    Evaluation Order (per candidate):
        1. Exclusion check (cheap, strict) - M-12
        2. item_matches() (expensive) - M-07 to M-11

    Short-circuit: Returns True on first valid candidate found.

    Args:
        required_item: Single required item from A.items
        candidate_items: All candidate items from B.items
        implies_fn: Optional term implication function (passed to item_matches)

    Returns:
        True if at least one valid candidate found, False otherwise

    Examples:
        # Has valid match
        required_item_has_match(
            {"type": "phone", "itemexclusions": ["used"], ...},
            [
                {"type": "phone", "categorical": {"condition": "used"}, ...},  # Excluded
                {"type": "phone", "categorical": {"condition": "new"}, ...}    # Valid
            ]
        ) → True (second candidate is valid)

        # All candidates excluded
        required_item_has_match(
            {"type": "phone", "itemexclusions": ["used", "refurbished"], ...},
            [
                {"type": "phone", "categorical": {"condition": "used"}, ...},
                {"type": "phone", "categorical": {"condition": "refurbished"}, ...}
            ]
        ) → False (all candidates excluded)

        # Empty candidate list
        required_item_has_match(
            {"type": "phone", ...},
            []
        ) → False (no candidates to match)

        # Type mismatch (no valid candidates)
        required_item_has_match(
            {"type": "smartphone", ...},
            [{"type": "laptop", ...}]
        ) → False (type mismatch)
    """
    # Empty candidate list → no match possible
    if not candidate_items:
        return False

    # Try to find at least one valid candidate
    for candidate_item in candidate_items:
        # M-12: Check exclusions FIRST (cheap, strict)
        if violates_item_exclusions(required_item, candidate_item):
            # This candidate is excluded, try next one
            continue

        # M-07 to M-11: Check if item matches (via Phase 2.3)
        if item_matches(required_item, candidate_item, implies_fn):
            # Found a valid match, we're done
            return True

    # No valid candidate found
    return False


# ============================================================================
# ALL REQUIRED ITEMS MATCHING (ARRAY)
# ============================================================================

def all_required_items_match(
    required_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    implies_fn: ImplicationFn = None
) -> bool:
    """
    Check if ALL required items have at least one valid matching candidate.

    Enforces Required Coverage Rule:
        ∀ r ∈ required_items:
          ∃ c ∈ candidate_items:
            ¬violates_item_exclusions(r, c) ∧
            item_matches(r, c, implies_fn)

    Short-circuit: Returns False on first required item without valid match.

    Vacuous Behavior:
        - required_items = [] → VACUOUSLY TRUE (no requirements)
        - candidate_items = [] AND required_items ≠ [] → FALSE (unmet requirements)

    Args:
        required_items: List of required items from A.items
        candidate_items: List of candidate items from B.items
        implies_fn: Optional term implication function

    Returns:
        True if ALL required items have valid matches, False otherwise

    Examples:
        # All required items matched
        all_required_items_match(
            [
                {"type": "phone", "itemexclusions": [], ...},
                {"type": "charger", "itemexclusions": [], ...}
            ],
            [
                {"type": "phone", ...},
                {"type": "charger", ...},
                {"type": "case", ...}  # Extra, ignored
            ]
        ) → True (both required items have matches)

        # One required item without match
        all_required_items_match(
            [
                {"type": "phone", ...},
                {"type": "charger", ...}
            ],
            [
                {"type": "phone", ...}
                # Missing charger
            ]
        ) → False (charger has no match)

        # Empty required items (vacuously true)
        all_required_items_match([], anything) → True

        # Empty candidates with non-empty required
        all_required_items_match(
            [{"type": "phone", ...}],
            []
        ) → False (required item has no candidates)

        # Exclusions prevent all matches
        all_required_items_match(
            [{"type": "phone", "itemexclusions": ["used", "refurbished"], ...}],
            [
                {"type": "phone", "categorical": {"condition": "used"}, ...},
                {"type": "phone", "categorical": {"condition": "refurbished"}, ...}
            ]
        ) → False (all candidates excluded)

    Notes:
        - Evaluation stops at first unmatched required item (fail-fast)
        - Extra candidate items are ignored (not required)
        - Same candidate can match multiple required items
        - Order of required_items doesn't matter (all must match)
    """
    # Validate inputs
    if not isinstance(required_items, list):
        raise TypeError(f"required_items must be list, got {type(required_items).__name__}")

    if not isinstance(candidate_items, list):
        raise TypeError(f"candidate_items must be list, got {type(candidate_items).__name__}")

    # Empty required items → VACUOUSLY TRUE (no requirements to satisfy)
    if not required_items:
        return True

    # Non-empty required items with empty candidates → FAIL
    if not candidate_items:
        return False

    # Check each required item has at least one valid match
    for required_item in required_items:
        if not required_item_has_match(required_item, candidate_items, implies_fn):
            # This required item has no valid match → FAIL immediately
            return False

    # All required items have valid matches
    return True


# ============================================================================
# PHASE 2.4 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.4 COMPLETION REPORT
============================

1. WHICH CANON RULES (M-07…M-12) ARE ENFORCED
----------------------------------------------

M-07: Item Type Equality Rule
- Location: item_matches() from Phase 2.3 (reused)
- Application: Called for each (required, candidate) pair
- Enforcement: Indirect via item_matches()

M-08: Item Categorical Subset Rule
- Location: item_matches() from Phase 2.3 (reused)
- Application: Called for each (required, candidate) pair
- Enforcement: Indirect via item_matches()

M-09: Item Min Constraint Rule
- Location: item_matches() from Phase 2.3 (reused)
- Application: Called for each (required, candidate) pair
- Enforcement: Indirect via item_matches() → numeric_constraints

M-10: Item Max Constraint Rule
- Location: item_matches() from Phase 2.3 (reused)
- Application: Called for each (required, candidate) pair
- Enforcement: Indirect via item_matches() → numeric_constraints

M-11: Item Range Constraint Rule
- Location: item_matches() from Phase 2.3 (reused)
- Application: Called for each (required, candidate) pair
- Enforcement: Indirect via item_matches() → numeric_constraints

M-12: Item Exclusion Disjoint Rule (NEW in Phase 2.4)
- Location: violates_item_exclusions()
- Semantics: required.itemexclusions ∩ Flatten(candidate) = ∅
- Implementation:
  * Extract all string values from candidate item
  * Check if ANY exclusion appears in extracted values
  * Any overlap → candidate INVALID for this required item
- Applied BEFORE item_matches() (cheap, strict filter)
- Enforcement: Direct implementation in this phase

Required Coverage Rule (Composite):
- Location: all_required_items_match()
- Semantics: ∀ r ∈ required_items: ∃ c ∈ candidate_items: valid(r,c)
- Implementation:
  * Iterate over required items
  * For each, check if at least one candidate satisfies M-07 to M-12
  * Short-circuit on first unmatched required item
- Combines all 6 rules (M-07 to M-12)


2. HOW ITEM EXCLUSIONS ARE APPLIED
-----------------------------------

Evaluation Order (per required/candidate pair):
1. FIRST: Check M-12 exclusions via violates_item_exclusions()
2. SECOND: If exclusions pass, check M-07 to M-11 via item_matches()

Rationale:
- Exclusion check is cheap (string set intersection)
- item_matches() is expensive (type, categorical, numeric)
- Fail-fast optimization: skip expensive checks if excluded

Exclusion Semantics (M-12):
- STRICT disjoint requirement (I-07: Exclusion Strictness Invariant)
- ANY overlap → candidate INVALID
- No fuzzy matching
- No "prefer not" logic
- Binary: excluded or not excluded

Extraction Process:
- flatten_item_values() extracts ALL string values from candidate
- Includes: type, categorical values
- Excludes: numeric constraint values (min/max/range)
- Case-insensitive (Phase 2.1 normalized)

Example Flow:
```
required = {"type": "phone", "itemexclusions": ["used", "refurbished"], ...}
candidate1 = {"type": "phone", "categorical": {"condition": "used"}, ...}
candidate2 = {"type": "phone", "categorical": {"condition": "new"}, ...}

Check candidate1:
  1. violates_item_exclusions(required, candidate1)?
     → flatten: {"phone", "used"}
     → "used" in ["used", "refurbished"]? YES
     → VIOLATED, skip item_matches(), try next candidate

Check candidate2:
  1. violates_item_exclusions(required, candidate2)?
     → flatten: {"phone", "new"}
     → "new" in ["used", "refurbished"]? NO
     → NOT VIOLATED, proceed to step 2
  2. item_matches(required, candidate2)?
     → Check M-07 to M-11
     → Result: True or False
```

Empty Exclusions:
- required.itemexclusions = [] → no restrictions
- violates_item_exclusions() → always returns False
- All candidates pass exclusion check

Exclusion Coverage:
- Same exclusion list applies to ALL candidates for this required item
- Different required items can have different exclusions
- Candidate rejected by one required item can match another

Exclusion vs Other Constraints:
- Exclusions are pre-filter (eliminate candidates)
- Other constraints are matching criteria (evaluate candidates)
- Exclusions cannot be satisfied (only avoided)
- Other constraints can be satisfied


3. HOW DYNAMIC ATTRIBUTES BEHAVE
---------------------------------

Item Type (Dynamic):
- No hardcoded types
- Any string value accepted
- Runtime discovery
- Example: "smartphone", "laptop", "yoga_instruction", "carpool"

Categorical Attributes (Dynamic):
- Keys: Dynamic strings discovered at runtime
- Values: Dynamic strings (extracted by flatten_item_values)
- No predefined schema
- Example: {"brand": "apple", "color": "black", "seller_type": "individual"}

Numeric Attributes (Dynamic):
- Keys: Dynamic strings in min/max/range
- Values: Numeric (not extracted for exclusions)
- No hardcoded attribute names
- Example: {"price": 100000, "rating": 4.0, "storage": 256}

Exclusion Values (Dynamic):
- Dynamic strings in itemexclusions array
- No predefined vocabulary
- Matched against extracted candidate values
- Example: ["used", "refurbished", "dealer", "agent"]

Extraction Behavior:
- flatten_item_values() discovers values at runtime
- No assumptions about attribute structure
- Works for ANY domain (product, service, mutual)
- Recursive extraction from nested structures

Required Item Independence:
- Each required item evaluated independently
- No coordination between required items
- Same candidate can match multiple required items
- Different required items can have different exclusions

Candidate Item Reuse:
- Same candidate can satisfy multiple required items
- No "consumption" of candidates
- All candidates available for all required items
- Example: One "charger" candidate can match multiple "charger" requirements

Array Order Irrelevance:
- required_items order doesn't matter (all must match)
- candidate_items order doesn't matter (find any match)
- Implementation uses iteration, but semantics are set-based

Missing Attributes:
- Required item without itemexclusions field → treated as []
- Candidate item with only type → valid (minimal attributes)
- No required attributes beyond what's specified
- Dynamic discovery enables flexible schemas


4. WHAT CASES CORRECTLY FAIL
-----------------------------

No Valid Candidate for Required Item:
✓ Empty candidate_items with non-empty required_items
  Example: all_required_items_match([{...}], []) → False

✓ Type mismatch (no matching type in candidates)
  Example:
    required: [{"type": "smartphone", ...}]
    candidates: [{"type": "laptop", ...}]
    → False

✓ All candidates excluded for a required item
  Example:
    required: [{"type": "phone", "itemexclusions": ["used", "refurbished"], ...}]
    candidates: [
      {"type": "phone", "categorical": {"condition": "used"}, ...},
      {"type": "phone", "categorical": {"condition": "refurbished"}, ...}
    ]
    → False (all candidates violated exclusions)

✓ Categorical mismatch (M-08 failure)
  Example:
    required: [{"type": "phone", "categorical": {"brand": "apple"}, ...}]
    candidates: [{"type": "phone", "categorical": {"brand": "samsung"}, ...}]
    → False

✓ Numeric constraint violation (M-09, M-10, or M-11 failure)
  Example:
    required: [{"type": "phone", "max": {"price": 50000}, ...}]
    candidates: [{"type": "phone", "range": {"price": [100000, 100000]}, ...}]
    → False (price exceeds max)

Multiple Required Items, One Unmatched:
✓ Partial coverage insufficient
  Example:
    required: [{"type": "phone", ...}, {"type": "charger", ...}]
    candidates: [{"type": "phone", ...}]  # Missing charger
    → False (charger unmatched)

Exclusion Violations:
✓ Single exclusion violated
  Example:
    required.itemexclusions: ["dealer"]
    candidate.categorical.seller_type: "dealer"
    → violates_item_exclusions() → True → candidate rejected

✓ Multiple exclusions, any one violated
  Example:
    required.itemexclusions: ["used", "refurbished", "damaged"]
    candidate.categorical.condition: "refurbished"
    → violates_item_exclusions() → True

✓ Exclusion in nested values
  Example:
    required.itemexclusions: ["agent"]
    candidate.categorical: {"type": "agent", "verified": "true"}
    → flatten: {"phone", "agent", "true"}
    → "agent" in exclusions → violated

Programmer Errors (exceptions raised):
✓ required_items not a list → TypeError
✓ candidate_items not a list → TypeError


5. WHAT ASSUMPTIONS WERE EXPLICITLY REJECTED
---------------------------------------------

❌ REJECTED: Listing-Level Orchestration
- Did NOT implement full A vs B matching
- Did NOT decide overall Match/No-Match
- Did NOT handle other/self constraints
- Reason: Per directive scope limitation
- Impact: This module handles items arrays only

❌ REJECTED: Bidirectional Logic
- Did NOT implement A→B and B→A checks
- Did NOT handle forward/reverse matching
- Did NOT orchestrate mutual intent
- Reason: Per directive scope limitation
- Impact: Caller must orchestrate bidirectional

❌ REJECTED: Best Candidate Selection
- Did NOT rank candidates
- Did NOT score matches
- Did NOT select "best" match
- Reason: Boolean matching only (strict)
- Impact: First valid candidate sufficient

❌ REJECTED: Domain/Category Matching
- Did NOT implement domain overlap checks
- Did NOT implement category matching
- Reason: Out of scope for Phase 2.4 (items only)
- Impact: Domain/category handled in later phase

❌ REJECTED: Other/Self Matching
- Did NOT implement other→self constraints
- Did NOT implement self→other constraints
- Reason: Out of scope for Phase 2.4 (items only)
- Impact: Other/self handled in later phase

❌ REJECTED: Location Logic
- Did NOT implement location matching
- Did NOT handle locationexclusions
- Reason: Out of scope for Phase 2.4 (items only)
- Impact: Location handled in later phase

❌ REJECTED: Exclusion Inference
- Did NOT infer additional exclusions
- Did NOT expand exclusion semantics
- Did NOT apply fuzzy matching to exclusions
- Reason: Strict disjoint only (I-07)
- Impact: Only explicitly stated exclusions enforced

❌ REJECTED: Term Implication in Exclusions
- Did NOT apply term implications to exclusion checking
- Did NOT use implies_fn for exclusions
- Reason: Exclusions are string literals (exact match only)
- Impact: "used" and "refurbished" are distinct exclusion values

❌ REJECTED: Hardcoded Attribute Names
- Did NOT special-case "condition", "seller_type", etc.
- Did NOT assume attribute meanings
- Reason: Per directive, dynamic discovery only
- Impact: Works for any attribute names

❌ REJECTED: Defaults/Fallbacks
- Did NOT assume default exclusions
- Did NOT fill in missing itemexclusions
- Did NOT infer exclusions from attributes
- Reason: Per directive
- Impact: Empty exclusions = no restrictions

❌ REJECTED: Exclusion Priorities
- Did NOT rank exclusions by severity
- Did NOT allow "soft" exclusions
- Did NOT implement "prefer not" logic
- Reason: All exclusions are strict (I-07)
- Impact: Binary: excluded or not

❌ REJECTED: Candidate Consumption
- Did NOT mark candidates as "used"
- Did NOT prevent candidate reuse
- Did NOT assign candidates exclusively
- Reason: Set-based semantics, not assignment
- Impact: Same candidate can match multiple requireds

❌ REJECTED: Match Scoring
- Did NOT compute match quality
- Did NOT return partial match info
- Did NOT provide match details
- Reason: Boolean only (True/False)
- Impact: No gradations, strict matching

❌ REJECTED: Order Sensitivity
- Did NOT assume required_items order matters
- Did NOT assume candidate_items order matters
- Did NOT implement sequential matching
- Reason: Set-based semantics
- Impact: Order-independent evaluation

❌ REJECTED: Empty Candidate Special Handling
- Did NOT treat empty candidate list as wildcard
- Did NOT auto-pass for empty candidates
- Reason: Empty candidates with non-empty required → FAIL
- Impact: Explicit presence required

❌ REJECTED: Mock Data
- Did NOT embed test data
- Did NOT include example items
- Reason: Per directive
- Impact: Pure logic module

❌ REJECTED: Numeric Value Extraction for Exclusions
- Did NOT extract numeric constraint values for exclusion checking
- Did NOT check if prices/ratings match exclusions
- Reason: Exclusions apply to categorical string values only
- Impact: Numeric constraints separate from exclusions


CRITICAL DESIGN DECISIONS
--------------------------

1. Exclusion-First Evaluation:
   - Check exclusions before item_matches()
   - Cheap filter eliminates invalid candidates early
   - Performance optimization (fail-fast)

2. String Value Extraction:
   - flatten_item_values() extracts type + categorical values
   - Does NOT extract numeric constraint values
   - Recursively handles nested structures
   - Case-insensitive (Phase 2.1 normalized)

3. Strict Disjoint Enforcement:
   - ANY overlap → candidate invalid
   - No partial matches
   - No fuzzy logic
   - Aligns with I-07 (Exclusion Strictness Invariant)

4. Vacuous Truth for Empty Required:
   - required_items = [] → True
   - Mathematically correct
   - Enables optional items
   - Consistent with earlier phases

5. Required Coverage Rule:
   - ALL required items must have matches
   - Not sufficient to have "most" matches
   - Strict all-or-nothing semantics
   - Aligns with I-01 (Strict Matching Invariant)

6. Short-Circuit Evaluation:
   - First valid candidate satisfies required item
   - First unmatched required item fails overall
   - Optimization without changing semantics

7. Candidate Reuse:
   - Same candidate can match multiple requireds
   - No consumption or assignment
   - Set-based semantics
   - Realistic for e-commerce (one seller, multiple buyers)

8. Pure Functions:
   - No side effects
   - No state mutation
   - Deterministic
   - Composable


INTEGRATION POINTS
------------------

Upstream Dependencies:
- schema_normalizer.py (Phase 2.1):
  * Normalized item structure
  * itemexclusions field (list of strings)
  * Type field guaranteed

- item_matchers.py (Phase 2.3):
  * item_matches() function
  * ImplicationFn type
  * M-07 to M-11 enforcement

- numeric_constraints.py (Phase 2.2):
  * Indirectly via item_matchers
  * Range semantics

Downstream Usage:
- Phase 2.5+ will:
  * Call all_required_items_match() for items check
  * Handle other/self constraints separately
  * Implement otherexclusions/selfexclusions
  * Orchestrate full listing matching
  * Implement bidirectional checks

Injection Points:
- implies_fn parameter in all public functions
- Passed through to item_matches()
- Enables term implication for categorical matching


SAFE FOR NEXT PHASE
--------------------

Phase 2.5 (Other/Self Matching) and beyond can safely assume:

✓ all_required_items_match() is pure function
✓ Returns True only if ALL required items have valid matches
✓ M-12 (exclusions) enforced correctly
✓ M-07 to M-11 enforced via item_matches()
✓ Empty required_items handled (vacuous truth)
✓ Empty candidate_items with non-empty required fails
✓ Exclusions are strict (disjoint)
✓ Short-circuit evaluation (performance)
✓ Candidate reuse allowed (same candidate, multiple requireds)
✓ Dynamic attributes supported
✓ Pure functions, no side effects

Next phase needs to:
- Implement other/self constraint matching
- Implement otherexclusions/selfexclusions
- Orchestrate domain/category matching
- Orchestrate location matching
- Implement bidirectional checks (for mutual)
- Compose all constraints into full listing matching
"""
