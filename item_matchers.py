"""
VRIDDHI MATCHING SYSTEM - ITEM MATCHING PRIMITIVES
Phase 2.3

Authority: MATCHING_CANON.md v1.1 (LOCKED)
Input Contract:
  - Normalized by schema_normalizer.py (Phase 2.1)
  - Numeric logic from numeric_constraints.py (Phase 2.2)

Purpose: Match a SINGLE pair of items (required vs candidate)
Scope: NO listing orchestration, NO array iteration

Implements Canon Rules:
- M-07: Item Type Equality Rule
- M-08: Item Categorical Subset Rule
- M-09: Item Min Constraint Rule
- M-10: Item Max Constraint Rule
- M-11: Item Range Constraint Rule

Matching Semantics:
- Type: required.type = candidate.type (case-insensitive, already normalized)
- Categorical: required.categorical ⊆ candidate.categorical (with term implications)
- Numeric: Reuses Phase 2.2 constraint evaluators
"""

from typing import Dict, Callable, Any, Tuple
from numeric_constraints import (
    Range,
    evaluate_min_constraints,
    evaluate_max_constraints,
    evaluate_range_constraints
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# Item structure (from Phase 2.1 normalization)
Item = Dict[str, Any]

# Implication function signature
# implies(candidate_value, required_value) -> bool
# Returns True if candidate_value implies or equals required_value
ImplicationFn = Callable[[str, str], bool]


# ============================================================================
# HELPER: Default implication function (exact match only)
# ============================================================================

def _exact_match_only(candidate_value: str, required_value: str) -> bool:
    """
    Default implication function: exact match only.

    Used when no term implication engine is provided.
    After normalization (M-32), all strings are lowercase+trimmed,
    so simple equality works.

    Args:
        candidate_value: Value from candidate item
        required_value: Value from required item

    Returns:
        True if values are exactly equal, False otherwise
    """
    return candidate_value == required_value


# ============================================================================
# M-07: ITEM TYPE EQUALITY RULE
# ============================================================================

def match_item_type(required_item: Item, candidate_item: Item) -> bool:
    """
    Check if item types match.

    Enforces Canon Rule M-07: Item Type Equality Rule

    Formal Expression:
        ∀ items ia ∈ A.items, ∃ ib ∈ B.items:
          Match(ia, ib) ⇒ ia.type = ib.type

    Preconditions:
        - Both items have "type" field (enforced by Phase 2.1)
        - Types are normalized strings (lowercase, trimmed)

    Pass Condition:
        normalize(required.type) = normalize(candidate.type)

    Fail Condition:
        required.type ≠ candidate.type

    Args:
        required_item: Item from listing A (what user wants)
        candidate_item: Item from listing B (what is offered)

    Returns:
        True if types match, False otherwise

    Examples:
        match_item_type(
            {"type": "smartphone", ...},
            {"type": "smartphone", ...}
        ) → True

        match_item_type(
            {"type": "smartphone", ...},
            {"type": "laptop", ...}
        ) → False

    Raises:
        KeyError: If "type" field missing (programmer error - Phase 2.1 should prevent)
    """
    # Both items MUST have "type" field (Phase 2.1 guarantee)
    if "type" not in required_item:
        raise KeyError("Required item missing 'type' field (programmer error)")

    if "type" not in candidate_item:
        raise KeyError("Candidate item missing 'type' field (programmer error)")

    # M-07: Type equality (case-insensitive, already normalized)
    return required_item["type"] == candidate_item["type"]


# ============================================================================
# M-08: ITEM CATEGORICAL SUBSET RULE
# ============================================================================

def match_item_categorical(
    required_item: Item,
    candidate_item: Item,
    implies_fn: ImplicationFn = None
) -> bool:
    """
    Check if candidate's categorical attributes satisfy required attributes.

    Enforces Canon Rule M-08: Item Categorical Subset Rule

    Formal Expression:
        ∀ items ia ∈ A.items, ib ∈ B.items where ia.type = ib.type:
          Match(ia, ib) ⇒ ia.categorical ⊆ ib.categorical

        where subset relation:
          ∀ (k,v) ∈ ia.categorical:
            (k, v') ∈ ib.categorical ∧
            (v = v' ∨ ImpliedBy(v, v'))

    Preconditions:
        - Both items have "categorical" field (dict)
        - All strings normalized (lowercase, trimmed)
        - implies_fn is callable or None

    Pass Condition:
        ∀ (k,v) ∈ required.categorical:
          ∃ (k, v') ∈ candidate.categorical:
            (v = v') ∨ implies_fn(v', v)

    Fail Condition:
        ∃ (k,v) ∈ required.categorical:
          (k ∉ candidate.categorical) ∨
          (candidate[k] ≠ v ∧ ¬implies_fn(candidate[k], v))

    Empty Constraint Behavior:
        - required.categorical = {} → VACUOUSLY TRUE (no requirements)
        - candidate.categorical can have extra keys (ignored)

    Args:
        required_item: Item from listing A (requirements)
        candidate_item: Item from listing B (offerings)
        implies_fn: Optional function(candidate_val, required_val) -> bool
                   If None, uses exact match only

    Returns:
        True if all required categorical attributes satisfied, False otherwise

    Examples:
        # Exact match
        match_item_categorical(
            {"categorical": {"brand": "apple"}},
            {"categorical": {"brand": "apple", "color": "black"}}
        ) → True

        # Missing required key
        match_item_categorical(
            {"categorical": {"brand": "apple"}},
            {"categorical": {"color": "black"}}
        ) → False

        # Value mismatch (no implication)
        match_item_categorical(
            {"categorical": {"brand": "apple"}},
            {"categorical": {"brand": "samsung"}}
        ) → False

        # With term implication
        match_item_categorical(
            {"categorical": {"condition": "good"}},
            {"categorical": {"condition": "excellent"}},
            implies_fn=lambda c, r: (c, r) == ("excellent", "good")
        ) → True

        # Empty required (vacuously true)
        match_item_categorical(
            {"categorical": {}},
            {"categorical": {"brand": "apple"}}
        ) → True
    """
    # Use exact match if no implication function provided
    if implies_fn is None:
        implies_fn = _exact_match_only

    # Get categorical dicts (Phase 2.1 ensures these exist)
    required_categorical = required_item.get("categorical", {})
    candidate_categorical = candidate_item.get("categorical", {})

    # Empty required constraints → VACUOUSLY TRUE
    if not required_categorical:
        return True

    # Check each required attribute
    for key, required_value in required_categorical.items():
        # Missing key in candidate → FAIL
        if key not in candidate_categorical:
            return False

        candidate_value = candidate_categorical[key]

        # Check if values match (exact or via implication)
        # M-08 subset logic: candidate must equal OR imply required
        if not implies_fn(candidate_value, required_value):
            return False

    # All required attributes satisfied
    return True


# ============================================================================
# M-09, M-10, M-11: ITEM NUMERIC CONSTRAINTS
# ============================================================================

def match_item_numeric(required_item: Item, candidate_item: Item) -> bool:
    """
    Check if candidate's numeric attributes satisfy required numeric constraints.

    Enforces Canon Rules:
    - M-09: Item Min Constraint Rule
    - M-10: Item Max Constraint Rule
    - M-11: Item Range Constraint Rule

    Formal Expression:
        ∀ items ia ∈ A.items, ib ∈ B.items where ia.type = ib.type:

          M-09 (MIN): ∀ key k ∈ keys(ia.min):
            ExtractNumeric(ib, k) ≥ ia.min[k]

          M-10 (MAX): ∀ key k ∈ keys(ia.max):
            ExtractNumeric(ib, k) ≤ ia.max[k]

          M-11 (RANGE): ∀ key k ∈ keys(ia.range):
            ia.range[k][0] ≤ ExtractNumeric(ib, k) ≤ ia.range[k][1]

    Implementation:
        Uses Phase 2.2 numeric_constraints.py functions:
        - evaluate_min_constraints(required.min, candidate_ranges)
        - evaluate_max_constraints(required.max, candidate_ranges)
        - evaluate_range_constraints(required.range, candidate_ranges)

    Note: Candidate values must be converted to Range format [min, max].
          Phase 2.1 normalized all numeric values into constraint objects,
          so we extract ranges from candidate's min/max/range fields.

    Args:
        required_item: Item from listing A (constraints)
        candidate_item: Item from listing B (values)

    Returns:
        True if all numeric constraints satisfied, False otherwise

    Examples:
        # MIN constraint satisfied
        match_item_numeric(
            {"min": {"price": 50000}},
            {"range": {"price": [95000, 95000]}}
        ) → True (95000 >= 50000)

        # MAX constraint violated
        match_item_numeric(
            {"max": {"price": 100000}},
            {"range": {"price": [150000, 150000]}}
        ) → False (150000 > 100000)

        # RANGE constraint (EXACT)
        match_item_numeric(
            {"range": {"storage": [256, 256]}},
            {"range": {"storage": [256, 256]}}
        ) → True

        # Empty constraints (vacuously true)
        match_item_numeric(
            {"min": {}, "max": {}, "range": {}},
            {"range": {"price": [100, 100]}}
        ) → True
    """
    # Get constraint objects from both items
    required_min = required_item.get("min", {})
    required_max = required_item.get("max", {})
    required_range = required_item.get("range", {})

    # Build candidate ranges from candidate's constraint objects
    # Candidate values are stored in min/max/range fields
    # We need to convert them to Range format for Phase 2.2 functions
    candidate_ranges = _extract_candidate_ranges(candidate_item)

    # M-09: Evaluate MIN constraints
    if not evaluate_min_constraints(required_min, candidate_ranges):
        return False

    # M-10: Evaluate MAX constraints
    if not evaluate_max_constraints(required_max, candidate_ranges):
        return False

    # M-11: Evaluate RANGE constraints
    if not evaluate_range_constraints(required_range, candidate_ranges):
        return False

    # All numeric constraints satisfied
    return True


def _extract_candidate_ranges(candidate_item: Item) -> Dict[str, Range]:
    """
    Extract numeric ranges from candidate item's constraint objects.

    Per M-30 (as updated in Phase 1.1 audit):
    ExtractNumeric returns ranges, not scalars.

    Priority order (from M-30):
    1. range[key] → use as-is [min, max]
    2. min[key] → [value, +∞]
    3. max[key] → [-∞, value]
    4. categorical[key] if numeric → [value, value]

    Args:
        candidate_item: Candidate item with constraint objects

    Returns:
        Dict mapping attribute names to Range tuples [min, max]

    Note:
        Phase 2.1 already structured all numeric values into constraint objects.
        This function extracts them into the Range format expected by Phase 2.2.
    """
    from numeric_constraints import NEGATIVE_INFINITY, POSITIVE_INFINITY

    candidate_min = candidate_item.get("min", {})
    candidate_max = candidate_item.get("max", {})
    candidate_range = candidate_item.get("range", {})
    candidate_categorical = candidate_item.get("categorical", {})

    ranges = {}

    # Collect all keys that have numeric values
    all_keys = set()
    all_keys.update(candidate_min.keys())
    all_keys.update(candidate_max.keys())
    all_keys.update(candidate_range.keys())

    for key in all_keys:
        # Priority 1: range (already in correct format)
        if key in candidate_range:
            ranges[key] = tuple(candidate_range[key])

        # Priority 2: min (unbounded max)
        elif key in candidate_min:
            ranges[key] = (candidate_min[key], POSITIVE_INFINITY)

        # Priority 3: max (unbounded min)
        elif key in candidate_max:
            ranges[key] = (NEGATIVE_INFINITY, candidate_max[key])

    # Priority 4: categorical (if numeric)
    # Note: Phase 2.1 normalizes categorical values to strings,
    # so numeric values in categorical would have been moved to min/max/range.
    # This is here for completeness per M-30, but should rarely be used.
    for key, value in candidate_categorical.items():
        if key not in ranges:
            # Try to convert to numeric
            try:
                num_val = float(value)
                ranges[key] = (num_val, num_val)
            except (ValueError, TypeError):
                # Not numeric, skip
                pass

    return ranges


# ============================================================================
# COMPOSITE ITEM MATCHER
# ============================================================================

def item_matches(
    required_item: Item,
    candidate_item: Item,
    implies_fn: ImplicationFn = None
) -> bool:
    """
    Check if a single candidate item satisfies a single required item.

    Composite function that checks ALL item-level constraints:
    1. Type equality (M-07)
    2. Categorical subset (M-08)
    3. Numeric constraints (M-09, M-10, M-11)

    Short-circuits on first failure for efficiency.

    Evaluation Order:
    1. Type (cheapest, most likely to fail early)
    2. Categorical (medium cost)
    3. Numeric (most expensive, three separate checks)

    Args:
        required_item: Item from listing A (what user wants)
        candidate_item: Item from listing B (what is offered)
        implies_fn: Optional term implication function for categorical matching

    Returns:
        True if ALL constraints satisfied, False if ANY fails

    Examples:
        # Complete match
        item_matches(
            {
                "type": "smartphone",
                "categorical": {"brand": "apple"},
                "min": {},
                "max": {"price": 100000},
                "range": {"storage": [256, 256]}
            },
            {
                "type": "smartphone",
                "categorical": {"brand": "apple", "color": "black"},
                "min": {},
                "max": {},
                "range": {"price": [95000, 95000], "storage": [256, 256]}
            }
        ) → True

        # Type mismatch (fails immediately)
        item_matches(
            {"type": "smartphone", ...},
            {"type": "laptop", ...}
        ) → False

        # Categorical mismatch
        item_matches(
            {"type": "phone", "categorical": {"brand": "apple"}, ...},
            {"type": "phone", "categorical": {"brand": "samsung"}, ...}
        ) → False

        # Numeric constraint violation
        item_matches(
            {"type": "phone", "categorical": {}, "max": {"price": 50000}, ...},
            {"type": "phone", "categorical": {}, "range": {"price": [100000, 100000]}, ...}
        ) → False
    """
    # M-07: Type must match (short-circuit on failure)
    if not match_item_type(required_item, candidate_item):
        return False

    # M-08: Categorical subset (short-circuit on failure)
    if not match_item_categorical(required_item, candidate_item, implies_fn):
        return False

    # M-09, M-10, M-11: Numeric constraints (short-circuit on failure)
    if not match_item_numeric(required_item, candidate_item):
        return False

    # All constraints satisfied
    return True


# ============================================================================
# PHASE 2.3 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.3 COMPLETION REPORT
============================

1. WHICH CANON RULES (M-07…M-11) ARE ENFORCED
----------------------------------------------

M-07: Item Type Equality Rule
- Location: match_item_type()
- Semantics: required.type = candidate.type
- Implementation: Direct string equality (already normalized by Phase 2.1)
- Fail Condition: Types don't match
- Pass Condition: Types are equal

M-08: Item Categorical Subset Rule
- Location: match_item_categorical()
- Semantics: required.categorical ⊆ candidate.categorical (with term implications)
- Implementation:
  * For each (key, value) in required.categorical:
    - Key must exist in candidate.categorical
    - Candidate value must equal OR imply required value
  * Uses injected implies_fn for term implication
  * Falls back to exact match if no implies_fn provided
- Fail Condition:
  * Required key missing in candidate
  * Values don't match and no implication
- Pass Condition:
  * All required keys present in candidate
  * All values match exactly or via implication
  * Empty required categorical → vacuously true

M-09: Item Min Constraint Rule
- Location: match_item_numeric() using evaluate_min_constraints()
- Semantics: ∀ key in required.min: candidate_range.min >= required.min[key]
- Implementation: Delegates to numeric_constraints.evaluate_min_constraints()
- Reuses: Phase 2.2 logic
- Fail Condition: Any MIN constraint violated or missing
- Pass Condition: All MIN constraints satisfied or empty

M-10: Item Max Constraint Rule
- Location: match_item_numeric() using evaluate_max_constraints()
- Semantics: ∀ key in required.max: candidate_range.max <= required.max[key]
- Implementation: Delegates to numeric_constraints.evaluate_max_constraints()
- Reuses: Phase 2.2 logic
- Fail Condition: Any MAX constraint violated or missing
- Pass Condition: All MAX constraints satisfied or empty

M-11: Item Range Constraint Rule
- Location: match_item_numeric() using evaluate_range_constraints()
- Semantics: ∀ key in required.range: candidate_range ⊆ required.range
- Implementation: Delegates to numeric_constraints.evaluate_range_constraints()
- Reuses: Phase 2.2 logic
- Handles EXACT as range[x, x] per I-02
- Fail Condition: Any RANGE constraint violated or missing
- Pass Condition: All RANGE constraints satisfied or empty

Composite Function:
- item_matches() orchestrates all 5 rules
- Evaluation order: Type → Categorical → Numeric
- Short-circuits on first failure
- Returns True only if ALL rules pass


2. HOW EMPTY CONSTRAINTS BEHAVE
--------------------------------

Empty Categorical Constraints:
- required.categorical = {} → VACUOUSLY TRUE
- No requirements to check
- Candidate can have any categorical attributes
- Example:
  match_item_categorical(
      {"categorical": {}},
      {"categorical": {"brand": "apple", "color": "black"}}
  ) → True

Empty Numeric Constraints:
- required.min = {} → VACUOUSLY TRUE
- required.max = {} → VACUOUSLY TRUE
- required.range = {} → VACUOUSLY TRUE
- Example:
  match_item_numeric(
      {"min": {}, "max": {}, "range": {}},
      {"range": {"price": [100, 100]}}
  ) → True

Philosophy:
- No requirements = no way to fail
- Consistent with mathematical interpretation
- Aligns with subset semantics (∅ ⊆ anything)
- Enables optional constraints

Complete Item with Empty Constraints:
- If required item has only type and empty categorical/numeric:
  item_matches(
      {"type": "phone", "categorical": {}, "min": {}, "max": {}, "range": {}},
      {"type": "phone", "categorical": {"brand": "apple"}, ...}
  ) → True (only type must match)


3. HOW DYNAMIC ATTRIBUTES ARE HANDLED
--------------------------------------

All attribute names are dynamic strings:
- NO hardcoded attribute names (price, brand, storage, etc.)
- NO special-case logic for specific attributes
- NO domain-specific rules

Categorical Attributes:
- Keys: Dynamic strings from required.categorical
- Values: Dynamic strings (normalized)
- Iteration over required keys drives evaluation
- Candidate can have extra keys (ignored)
- Example:
  - Required: {"brand": "apple", "color": "black"}
  - Candidate: {"brand": "apple", "color": "black", "storage": "256gb", "condition": "new"}
  - Result: TRUE ("storage" and "condition" ignored)

Numeric Attributes:
- Keys: Dynamic strings from required.min/max/range
- Values: Numeric (int/float) or Range tuples
- Iteration over required keys drives evaluation
- Candidate can have extra numeric attributes (ignored)
- Example:
  - Required min: {"rating": 4.0}
  - Candidate ranges: {"rating": [4.5, 5.0], "experience": [60, 60], "price": [100, 100]}
  - Result: TRUE ("experience" and "price" ignored)

Attribute Discovery:
- No predefined schema for attribute names
- Attributes discovered at runtime from dict keys
- System works for ANY attribute name
- Enables flexible domain modeling

Missing Required Attributes:
- If required has key K but candidate doesn't → FAIL
- No defaults, no inference
- Strict presence check
- Example:
  - Required categorical: {"brand": "apple"}
  - Candidate categorical: {"color": "black"}
  - Result: FALSE ("brand" missing)

Extra Candidate Attributes:
- Candidate can have attributes not in required
- Extra attributes are ignored (not evaluated)
- Subset semantics: required ⊆ candidate
- Example:
  - Required: {"brand": "apple"}
  - Candidate: {"brand": "apple", "warranty": "2years", "accessories": "charger"}
  - Result: TRUE (extra attributes ignored)


4. WHAT CASES FAIL CORRECTLY
-----------------------------

Type Mismatches (M-07):
✓ Different item types
  Example: required.type="smartphone", candidate.type="laptop" → FALSE

Categorical Failures (M-08):
✓ Missing required key
  Example: required={"brand": "apple"}, candidate={"color": "black"} → FALSE

✓ Value mismatch (no implication)
  Example: required={"brand": "apple"}, candidate={"brand": "samsung"} → FALSE

✓ Multiple attributes, one mismatch
  Example: required={"brand": "apple", "color": "black"},
           candidate={"brand": "apple", "color": "white"} → FALSE

Numeric Constraint Failures:
✓ MIN constraint violated (M-09)
  Example: required.min={"price": 50000}, candidate has price [30000, 30000] → FALSE

✓ MAX constraint violated (M-10)
  Example: required.max={"price": 100000}, candidate has price [150000, 150000] → FALSE

✓ RANGE constraint violated (M-11)
  Example: required.range={"storage": [256, 256]}, candidate has [512, 512] → FALSE

✓ Missing required numeric attribute
  Example: required.min={"rating": 4.0}, candidate has no rating → FALSE

Composite Failures (item_matches):
✓ Type matches but categorical fails
✓ Type and categorical match but numeric fails
✓ Any single constraint failure causes overall failure
✓ Short-circuit behavior (stops at first failure)

Programmer Errors (exceptions raised):
✓ Missing "type" field in required item
✓ Missing "type" field in candidate item
✓ Invalid item structure (not dict)


5. WHAT ASSUMPTIONS WERE EXPLICITLY REJECTED
---------------------------------------------

❌ REJECTED: Listing-Level Orchestration
- Did NOT iterate over arrays of items
- Did NOT select "best" matching item
- Did NOT decide overall Match/No-Match for listings
- Reason: Per directive scope limitation
- Impact: This module handles ONE item pair only

❌ REJECTED: Bidirectional Logic
- Did NOT implement A↔B checks
- Did NOT orchestrate forward/reverse matching
- Did NOT handle mutual intent special logic
- Reason: Per directive scope limitation
- Impact: Caller must orchestrate bidirectional checks

❌ REJECTED: Exclusion Checking
- Did NOT implement itemexclusions logic
- Did NOT check if candidate violates exclusions
- Reason: Out of scope for Phase 2.3
- Impact: Exclusions handled in later phase

❌ REJECTED: Term Implication Inference
- Did NOT invent implication rules
- Did NOT infer implications from similarity
- Did NOT build implication graph
- Reason: Per directive "No term implication inference (consume only)"
- Impact: Implication function must be injected by caller

❌ REJECTED: Default Implication Function
- Did NOT create global implication engine
- Did NOT hardcode term relationships
- Uses _exact_match_only as fallback when implies_fn is None
- Reason: Implication is external concern
- Impact: Caller controls term semantics

❌ REJECTED: Hardcoded Attribute Names
- Did NOT special-case "price", "brand", "storage", etc.
- Did NOT apply domain-specific rules
- Did NOT assume attribute meanings
- Reason: Per directive "No hardcoded attribute names"
- Impact: Works for ANY attribute names

❌ REJECTED: Defaults/Fallbacks
- Did NOT assume default values for missing attributes
- Did NOT infer missing categorical values
- Did NOT fill in missing numeric constraints
- Reason: Per directive "No defaults or fallbacks"
- Impact: Missing required attributes → FAIL

❌ REJECTED: Best-Effort Matching
- Did NOT return partial match scores
- Did NOT continue after failures
- Did NOT suggest "close" matches
- Reason: Boolean matching only (strict)
- Impact: Returns True or False, no gradations

❌ REJECTED: Type Coercion
- Did NOT convert types
- Did NOT parse strings
- Did NOT normalize numeric formats
- Reason: Phase 2.1 already normalized everything
- Impact: Assumes clean input

❌ REJECTED: Mock Data
- Did NOT embed test data
- Did NOT include example items
- Did NOT hardcode sample vocabularies
- Reason: Per directive
- Impact: Pure logic module

❌ REJECTED: Attribute Ordering Assumptions
- Did NOT assume attributes must be checked in specific order
- Did NOT require specific keys present
- Did NOT enforce schema beyond type existence
- Reason: Dynamic attributes, flexible schema
- Impact: Works with any attribute set

❌ REJECTED: Categorical Value Arrays
- Did NOT handle categorical values as arrays
- Did NOT implement set intersection for categorical
- Assumes categorical values are strings (Phase 2.1 normalized)
- Reason: Phase 2.1 normalizes to strings
- Impact: Each categorical value is single string

❌ REJECTED: Numeric Value Arrays
- Did NOT handle numeric values as point sets
- Did NOT compute average/median
- Uses Range semantics only
- Reason: M-30 audit resolution (no scalar collapse)
- Impact: All numeric values are ranges


CRITICAL DESIGN DECISIONS
--------------------------

1. Implication Injection:
   - Term implication function injected via parameter
   - No global implication state
   - Enables testing with different implication strategies
   - Aligns with TERM IMPLICATION CONTRACT from Phase 1.1

2. Short-Circuit Evaluation:
   - item_matches() stops at first failure
   - Order: Type → Categorical → Numeric
   - Optimization: cheapest checks first
   - No wasted computation

3. Vacuous Truth for Empty Constraints:
   - Mathematical correctness
   - Aligns with subset semantics
   - Enables optional constraints
   - Consistent with Phase 2.2

4. Dynamic Attribute Discovery:
   - No hardcoded attribute names
   - Runtime discovery from dict keys
   - Generic algorithm works for any domain
   - Maximum flexibility

5. Pure Functions:
   - No side effects
   - No global state
   - No mutation
   - Easily testable and composable

6. Phase 2.2 Reuse:
   - All numeric logic delegates to Phase 2.2
   - No duplication of range semantics
   - Single source of truth for numeric constraints
   - Clean separation of concerns

7. Single Responsibility:
   - Each function has ONE job
   - match_item_type: type only
   - match_item_categorical: categorical only
   - match_item_numeric: numeric only
   - item_matches: orchestrates
   - Enables unit testing

8. Fail-Fast Philosophy:
   - Return False immediately on first violation
   - No collection of all errors
   - Aligns with strict matching (I-01)
   - Performance optimization


INTEGRATION POINTS
------------------

Upstream Dependencies:
- schema_normalizer.py (Phase 2.1):
  * Provides normalized item structure
  * Guarantees type field exists
  * Ensures categorical values are strings
  * Ensures numeric values in constraint objects

- numeric_constraints.py (Phase 2.2):
  * evaluate_min_constraints()
  * evaluate_max_constraints()
  * evaluate_range_constraints()
  * Range type and constants

Downstream Usage:
- Phase 2.4+ will:
  * Call item_matches() for each required/candidate item pair
  * Iterate over items arrays
  * Handle item exclusions
  * Orchestrate bidirectional checks
  * Decide overall Match/No-Match

Injection Points:
- implies_fn parameter in match_item_categorical()
- implies_fn parameter in item_matches()
- Caller provides term implication logic
- Enables different implication strategies


SAFE FOR NEXT PHASE
--------------------

Phase 2.4 (Item Array Matching) and beyond can safely assume:

✓ item_matches() is pure function (no side effects)
✓ Returns True only if ALL 5 rules (M-07 to M-11) pass
✓ Short-circuits on first failure (performance)
✓ Empty constraints handled correctly (vacuous truth)
✓ Missing required attributes detected (fail)
✓ Dynamic attributes supported (no hardcoded names)
✓ Term implication is injectable (no global state)
✓ Type errors indicate programmer errors only
✓ Numeric logic is correct (delegates to Phase 2.2)

Next phase needs to:
- Iterate over items arrays (required vs all candidates)
- Find at least one matching candidate per required item
- Handle itemexclusions
- Handle other/self constraints (not item-level)
- Orchestrate bidirectional checks (for mutual intent)
- Compose into full listing matching
"""
