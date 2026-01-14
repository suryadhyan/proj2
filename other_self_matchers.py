"""
PHASE 2.5: OTHER/SELF CONSTRAINT MATCHING

Implements directional constraint matching between other and self objects.

Canon Rules Implemented:
- M-13 to M-17: Other → Self direction (forward)
- M-18 to M-22: Self → Other direction (reverse)

Authority: MATCHING_CANON.md v1.1 (LOCKED)

Dependencies:
- Phase 2.1: schema_normalizer.py (input contract)
- Phase 2.2: numeric_constraints.py (numeric evaluation)

Author: Claude (Execution Engine)
Date: 2026-01-11
"""

from typing import Callable, Dict, Set, Optional, Any, Tuple
from numeric_constraints import (
    satisfies_min_constraint,
    satisfies_max_constraint,
    range_contains,
    Range,
    NEGATIVE_INFINITY,
    POSITIVE_INFINITY
)

# ============================================================================
# TYPE ALIASES
# ============================================================================

ImplicationFn = Callable[[str, str], bool]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _exact_match_only(candidate_value: str, required_value: str) -> bool:
    """
    Default implication function: exact match only.

    Used when no custom implication function is provided.

    Args:
        candidate_value: Value from candidate object
        required_value: Value from required object

    Returns:
        True if values are exactly equal, False otherwise
    """
    return candidate_value == required_value


def flatten_categorical_values(categorical: Dict[str, str]) -> Set[str]:
    """
    Extract all values from categorical constraint object.

    Used for exclusion checking (M-17, M-22).
    Does NOT include attribute names, only values.

    Args:
        categorical: Normalized categorical constraint object

    Returns:
        Set of all categorical attribute values

    Example:
        >>> flatten_categorical_values({"brand": "apple", "color": "black"})
        {"apple", "black"}
    """
    return set(categorical.values())


def _extract_numeric_range(obj: Dict[str, Any], attr: str) -> Optional[Range]:
    """
    Extract numeric range for a single attribute from constraint object.

    Per M-30 (as updated in Phase 1.1 audit):
    ExtractNumeric returns ranges, not scalars.

    Priority order (from M-30):
    1. range[attr] → use as-is [min, max]
    2. min[attr] → [value, +∞]
    3. max[attr] → [-∞, value]

    Args:
        obj: Constraint object (other or self) with min/max/range fields
        attr: Attribute name to extract

    Returns:
        Range tuple (min, max) or None if attribute not found

    Example:
        >>> _extract_numeric_range({"range": {"price": [100, 200]}}, "price")
        (100, 200)
        >>> _extract_numeric_range({"min": {"rating": 4.0}}, "rating")
        (4.0, inf)
        >>> _extract_numeric_range({"max": {"price": 1000}}, "price")
        (-inf, 1000)
    """
    # Priority 1: range (already in correct format)
    if attr in obj.get("range", {}):
        range_val = obj["range"][attr]
        return tuple(range_val)  # Convert list to tuple

    # Priority 2: min (unbounded max)
    if attr in obj.get("min", {}):
        return (obj["min"][attr], POSITIVE_INFINITY)

    # Priority 3: max (unbounded min)
    if attr in obj.get("max", {}):
        return (NEGATIVE_INFINITY, obj["max"][attr])

    # Attribute not found in any constraint mode
    return None


# ============================================================================
# FORWARD DIRECTION: OTHER → SELF (M-13 to M-17)
# ============================================================================

def match_other_to_self(other: Dict[str, Any],
                        self_obj: Dict[str, Any],
                        implies_fn: Optional[ImplicationFn] = None) -> bool:
    """
    M-13 to M-17: Check if self_obj satisfies other constraints.

    Direction: A.other → B.self
    - other = A.other (what A requires from B)
    - self_obj = B.self (what B offers to A)

    This checks if B's offering (self) satisfies A's requirements (other).

    Evaluation order (cheap to expensive):
    1. Categorical subset (M-13)
    2. Numeric constraints (M-14, M-15, M-16)
    3. Exclusions (M-17)

    Short-circuits on first failure.

    Args:
        other: Required constraints (A.other)
        self_obj: Candidate offering (B.self)
        implies_fn: Optional term implication function for categorical matching

    Returns:
        True if self_obj satisfies ALL other constraints, False otherwise

    Raises:
        KeyError: If required fields are missing (programmer error)
    """
    if implies_fn is None:
        implies_fn = _exact_match_only

    # M-13: Other-Self Categorical Subset Rule
    # Semantics: other.categorical ⊆ self_obj.categorical (with implications)
    # Every attribute in other.categorical must exist in self_obj.categorical
    # with a compatible value (exact match or implied)
    for attr, required_value in other["categorical"].items():
        if attr not in self_obj["categorical"]:
            return False  # Required attribute missing in candidate
        candidate_value = self_obj["categorical"][attr]
        if not implies_fn(candidate_value, required_value):
            return False  # Value mismatch (no implication)

    # M-14: Other-Self Min Constraint Rule
    # Semantics: For all k in other.min: self_obj[k] >= other.min[k]
    # Candidate's value must meet or exceed the minimum requirement
    for attr, required_min in other["min"].items():
        candidate_range = _extract_numeric_range(self_obj, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_min_constraint(required_min, candidate_range):
            return False  # Candidate value too low

    # M-15: Other-Self Max Constraint Rule
    # Semantics: For all k in other.max: self_obj[k] <= other.max[k]
    # Candidate's value must not exceed the maximum requirement
    for attr, required_max in other["max"].items():
        candidate_range = _extract_numeric_range(self_obj, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_max_constraint(required_max, candidate_range):
            return False  # Candidate value too high

    # M-16: Other-Self Range Constraint Rule
    # Semantics: For all k in other.range: self_obj.range[k] ⊆ other.range[k]
    # Candidate's range must be fully contained within required range
    for attr, required_range in other["range"].items():
        candidate_range = _extract_numeric_range(self_obj, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not range_contains(candidate_range, tuple(required_range)):
            return False  # Candidate range not subset of required range

    # M-17: Other Exclusion Disjoint Rule
    # Semantics: other.otherexclusions ∩ Flatten(self_obj.categorical) = ∅
    # None of the excluded values can appear in candidate's categorical values
    # This is STRICT: any overlap → rejection (I-07 invariant)
    if other["otherexclusions"]:  # Empty list → no check needed
        self_values = flatten_categorical_values(self_obj["categorical"])
        exclusions = set(other["otherexclusions"])
        if exclusions & self_values:  # Non-empty intersection
            return False  # Exclusion violated

    # All constraints satisfied
    return True


# ============================================================================
# REVERSE DIRECTION: SELF → OTHER (M-18 to M-22)
# ============================================================================

def match_self_to_other(other: Dict[str, Any],
                        self_obj: Dict[str, Any],
                        implies_fn: Optional[ImplicationFn] = None) -> bool:
    """
    M-18 to M-22: Check if other satisfies self_obj constraints.

    Direction: A.self → B.other
    - self_obj = A.self (what A requires from B)
    - other = B.other (what B requires from others, offered to A)

    This checks if B's offering (other) satisfies A's requirements (self).

    Evaluation order (cheap to expensive):
    1. Categorical subset (M-18)
    2. Numeric constraints (M-19, M-20, M-21)
    3. Exclusions (M-22)

    Short-circuits on first failure.

    Args:
        other: Candidate offering (B.other)
        self_obj: Required constraints (A.self)
        implies_fn: Optional term implication function for categorical matching

    Returns:
        True if other satisfies ALL self_obj constraints, False otherwise

    Raises:
        KeyError: If required fields are missing (programmer error)
    """
    if implies_fn is None:
        implies_fn = _exact_match_only

    # M-18: Self-Other Categorical Subset Rule
    # Semantics: self_obj.categorical ⊆ other.categorical (with implications)
    # Every attribute in self_obj.categorical must exist in other.categorical
    # with a compatible value (exact match or implied)
    for attr, required_value in self_obj["categorical"].items():
        if attr not in other["categorical"]:
            return False  # Required attribute missing in candidate
        candidate_value = other["categorical"][attr]
        if not implies_fn(candidate_value, required_value):
            return False  # Value mismatch (no implication)

    # M-19: Self-Other Min Constraint Rule
    # Semantics: For all k in self_obj.min: other[k] >= self_obj.min[k]
    # Candidate's value must meet or exceed the minimum requirement
    for attr, required_min in self_obj["min"].items():
        candidate_range = _extract_numeric_range(other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_min_constraint(required_min, candidate_range):
            return False  # Candidate value too low

    # M-20: Self-Other Max Constraint Rule
    # Semantics: For all k in self_obj.max: other[k] <= self_obj.max[k]
    # Candidate's value must not exceed the maximum requirement
    for attr, required_max in self_obj["max"].items():
        candidate_range = _extract_numeric_range(other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_max_constraint(required_max, candidate_range):
            return False  # Candidate value too high

    # M-21: Self-Other Range Constraint Rule
    # Semantics: For all k in self_obj.range: other.range[k] ⊆ self_obj.range[k]
    # Candidate's range must be fully contained within required range
    for attr, required_range in self_obj["range"].items():
        candidate_range = _extract_numeric_range(other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not range_contains(candidate_range, tuple(required_range)):
            return False  # Candidate range not subset of required range

    # M-22: Self Exclusion Disjoint Rule
    # Semantics: self_obj.selfexclusions ∩ Flatten(other.categorical) = ∅
    # None of the excluded values can appear in candidate's categorical values
    # This is STRICT: any overlap → rejection (I-07 invariant)
    if self_obj["selfexclusions"]:  # Empty list → no check needed
        other_values = flatten_categorical_values(other["categorical"])
        exclusions = set(self_obj["selfexclusions"])
        if exclusions & other_values:  # Non-empty intersection
            return False  # Exclusion violated

    # All constraints satisfied
    return True


# ============================================================================
# PHASE 2.5 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.5 COMPLETION REPORT
===========================

1. CANON RULES ENFORCED (M-13 to M-22)
--------------------------------------

FORWARD DIRECTION (Other → Self):
- M-13: Other-Self Categorical Subset Rule
  * Function: match_other_to_self() → categorical iteration
  * Semantics: other.categorical ⊆ self_obj.categorical
  * Implementation: Subset check with term implications
  * Line: ~95-103

- M-14: Other-Self Min Constraint Rule
  * Function: match_other_to_self() → min iteration
  * Semantics: For all k: self_obj[k] >= other.min[k]
  * Implementation: Delegates to Phase 2.2 satisfies_min_constraint()
  * Line: ~105-112

- M-15: Other-Self Max Constraint Rule
  * Function: match_other_to_self() → max iteration
  * Semantics: For all k: self_obj[k] <= other.max[k]
  * Implementation: Delegates to Phase 2.2 satisfies_max_constraint()
  * Line: ~114-121

- M-16: Other-Self Range Constraint Rule
  * Function: match_other_to_self() → range iteration
  * Semantics: For all k: self_obj.range[k] ⊆ other.range[k]
  * Implementation: Delegates to Phase 2.2 range_contains()
  * Line: ~123-130

- M-17: Other Exclusion Disjoint Rule
  * Function: match_other_to_self() → exclusion check
  * Semantics: other.otherexclusions ∩ Flatten(self_obj.categorical) = ∅
  * Implementation: Set intersection check (strict disjoint, I-07)
  * Line: ~132-139

REVERSE DIRECTION (Self → Other):
- M-18: Self-Other Categorical Subset Rule
  * Function: match_self_to_other() → categorical iteration
  * Semantics: self_obj.categorical ⊆ other.categorical
  * Implementation: Subset check with term implications
  * Line: ~171-179

- M-19: Self-Other Min Constraint Rule
  * Function: match_self_to_other() → min iteration
  * Semantics: For all k: other[k] >= self_obj.min[k]
  * Implementation: Delegates to Phase 2.2 satisfies_min_constraint()
  * Line: ~181-188

- M-20: Self-Other Max Constraint Rule
  * Function: match_self_to_other() → max iteration
  * Semantics: For all k: other[k] <= self_obj.max[k]
  * Implementation: Delegates to Phase 2.2 satisfies_max_constraint()
  * Line: ~190-197

- M-21: Self-Other Range Constraint Rule
  * Function: match_self_to_other() → range iteration
  * Semantics: For all k: other.range[k] ⊆ self_obj.range[k]
  * Implementation: Delegates to Phase 2.2 range_contains()
  * Line: ~199-206

- M-22: Self Exclusion Disjoint Rule
  * Function: match_self_to_other() → exclusion check
  * Semantics: self_obj.selfexclusions ∩ Flatten(other.categorical) = ∅
  * Implementation: Set intersection check (strict disjoint, I-07)
  * Line: ~208-215


2. DIRECTIONALITY PRESERVATION
-------------------------------

Forward vs Reverse are COMPLETELY INDEPENDENT:

✓ NO shared implementation between match_other_to_self() and match_self_to_other()
✓ Each function explicitly documents its direction (A.other → B.self vs A.self → B.other)
✓ Parameter order is IDENTICAL (other, self_obj, implies_fn) for both functions
✓ Logic differs in which object is "required" vs "candidate":
  - Forward: other is required, self_obj is candidate
  - Reverse: self_obj is required, other is candidate
✓ Exclusion fields differ:
  - Forward: uses other["otherexclusions"]
  - Reverse: uses self_obj["selfexclusions"]
✓ NO bidirectional orchestration (M-29 explicitly excluded)
✓ NO symmetry assumptions (each direction evaluated independently)

Why this matters:
- Forward checks if B.self satisfies A.other (what A wants from B)
- Reverse checks if B.other satisfies A.self (what A offers to B)
- These are DIFFERENT questions with DIFFERENT semantics
- Caller must orchestrate both checks if mutual compatibility is needed


3. DYNAMIC ATTRIBUTE HANDLING
------------------------------

✓ NO hardcoded attribute names (no "price", "brand", "location", etc.)
✓ Runtime iteration over constraint dictionaries (categorical, min, max, range)
✓ Works for ANY attribute names (domain-agnostic)
✓ Attribute discovery via .items() iteration
✓ Missing required attributes detected (return False)
✓ Extra candidate attributes ignored (subset semantics)

Implementation pattern:
```python
for attr, required_value in required_constraints.items():
    # attr discovered at runtime, not hardcoded
    if attr not in candidate_constraints:
        return False  # Required attribute missing
    # ... check value compatibility
```

This enables:
- Product matching (price, brand, storage, etc.)
- Service matching (duration, mode, availability, etc.)
- Location matching (distance, area, etc.)
- ANY domain without code changes


4. EXCLUSION ENFORCEMENT
-------------------------

Exclusions are STRICT (I-07: Exclusion Strictness Invariant):

✓ Set-based disjoint check: exclusions ∩ candidate_values = ∅
✓ ANY overlap → immediate rejection
✓ NO term implications applied to exclusions (exact match only)
✓ NO partial exclusion tolerance
✓ Empty exclusions → no violations (vacuously true)

Implementation:
```python
if exclusions:  # Skip check if empty
    candidate_values = flatten_categorical_values(candidate["categorical"])
    exclusion_set = set(exclusions)
    if exclusion_set & candidate_values:  # Non-empty intersection
        return False  # STRICT rejection
```

Exclusion fields:
- M-17: other["otherexclusions"] → checked against self_obj.categorical
- M-22: self_obj["selfexclusions"] → checked against other.categorical

Value extraction:
- flatten_categorical_values() extracts ONLY categorical attribute values
- Does NOT include attribute names
- Does NOT include numeric constraint values
- Example: {"brand": "apple", "color": "black"} → {"apple", "black"}


5. ASSUMPTIONS EXPLICITLY REJECTED
-----------------------------------

❌ NO Bidirectional Orchestration (M-29):
   - Did NOT implement mutual intent matching
   - Did NOT orchestrate forward AND reverse checks
   - Did NOT compose both directions into single result
   - Reason: Out of scope for Phase 2.5
   - Impact: Caller orchestrates bidirectional logic

❌ NO Listing-Level Orchestration:
   - Did NOT combine items + other/self + location
   - Did NOT implement full listing matching
   - Did NOT decide overall Match/No-Match
   - Reason: Out of scope for Phase 2.5
   - Impact: Phase 2.6+ will orchestrate full matching

❌ NO Item Matching Logic:
   - Did NOT re-implement item array matching
   - Did NOT call Phase 2.4 functions
   - Reason: Item constraints handled separately
   - Impact: Caller combines items + other/self

❌ NO Location Logic (M-23 to M-28):
   - Did NOT implement location constraints
   - Did NOT check location exclusions
   - Reason: Out of scope for Phase 2.5
   - Impact: Phase 2.6+ will handle location

❌ NO Domain/Category Matching (M-01 to M-06):
   - Did NOT implement domain compatibility
   - Did NOT check listing categories
   - Did NOT validate intent/subintent
   - Reason: Out of scope (preprocessing in Phase 2.1)
   - Impact: Assumed pre-validated by schema normalizer

❌ NO Term Implication Inference:
   - Did NOT invent implication rules
   - Did NOT build implication graph
   - Did NOT infer from similarity
   - Reason: Consume-only contract
   - Impact: Implication injected by caller

❌ NO Symmetry Assumptions:
   - Did NOT assume forward = reverse
   - Did NOT share code between directions
   - Did NOT couple the two checks
   - Reason: Different semantic meanings
   - Impact: Each direction independent

❌ NO Partial Matching:
   - Did NOT return scores or percentages
   - Did NOT count satisfied constraints
   - Did NOT suggest alternatives
   - Reason: Boolean only (strict)
   - Impact: All-or-nothing result

❌ NO Best-Effort Matching:
   - Did NOT tolerate partial constraint satisfaction
   - Did NOT apply "prefer" modes
   - Did NOT use fuzzy logic
   - Reason: I-01, I-03 invariants (strict matching)
   - Impact: Any violation → False

❌ NO Exclusion Implications:
   - Did NOT apply term implications to exclusions
   - Did NOT infer excluded terms
   - Reason: Exclusions are literal (I-07)
   - Impact: Exact string match for exclusions

❌ NO Global State:
   - Did NOT create global implication database
   - Did NOT hardcode term relationships
   - Uses parameter injection only
   - Reason: Clean dependency injection
   - Impact: Testable, flexible

❌ NO Default Values:
   - Did NOT assume missing constraints
   - Did NOT fill gaps
   - Did NOT infer intent
   - Reason: Schema normalizer guarantees structure
   - Impact: Missing required → programmer error (KeyError)


ARCHITECTURAL GUARANTEES:
-------------------------

After Phase 2.5, downstream code can safely assume:

✓ match_other_to_self() and match_self_to_other() are pure functions
✓ Both functions are stateless (no side effects)
✓ Both functions are deterministic (same input → same output)
✓ Short-circuit evaluation (first failure → return False)
✓ Evaluation order: Categorical → Numeric → Exclusion (cheap to expensive)
✓ Empty constraints handled correctly (vacuous truth)
✓ Missing required attributes detected (return False)
✓ Phase 2.2 numeric logic reused (no duplication)
✓ Term implication injectable (no global state)
✓ Dynamic attributes supported (runtime discovery)
✓ Exclusions strictly enforced (I-07)
✓ Directionality preserved (forward ≠ reverse)
✓ NO bidirectional coupling
✓ NO listing-level orchestration

Ready for Phase 2.6+: Location Constraint Matching & Full Listing Orchestration
"""
