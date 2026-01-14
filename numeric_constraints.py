"""
VRIDDHI MATCHING SYSTEM - NUMERIC CONSTRAINT MATCHERS
Phase 2.2

Authority: MATCHING_CANON.md v1.1 (LOCKED)
Input Contract: Normalized by schema_normalizer.py (Phase 2.1)

Purpose: Evaluate numeric constraints using range-based semantics
Scope: ONLY numeric constraint evaluation - NO listing orchestration

Implements Canon Rules:
- M-09, M-10, M-11: Item-level numeric constraints
- M-14, M-15, M-16: Other→Self numeric constraints
- M-19, M-20, M-21: Self→Other numeric constraints

Numeric Semantics (from M-30 Range Comparison Operations):
- MIN constraint: B_range.min >= A_min
- MAX constraint: B_range.max <= A_max
- RANGE constraint: B_range ⊆ A_range (i.e., A.min ≤ B.min AND B.max ≤ A.max)
- Empty constraint set → VACUOUSLY SATISFIED
"""

from typing import Dict, Tuple
import math


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# Range type: [min, max] with inclusive bounds
# Unbounded ranges use -∞ or +∞
Range = Tuple[float, float]

# Infinity constants for unbounded ranges
NEGATIVE_INFINITY = float('-inf')
POSITIVE_INFINITY = float('+inf')


# ============================================================================
# RANGE VALIDATION (Programmer error detection only)
# ============================================================================

def _validate_range(r: Range, context: str = "") -> None:
    """
    Validate range structure (programmer error detection).

    NOT for data validation - assumes input from schema_normalizer.py.
    Only catches programming errors (wrong types, invalid structure).

    Args:
        r: Range tuple to validate
        context: Context string for error messages

    Raises:
        TypeError: If not a tuple/list of length 2
        ValueError: If min > max (violates I-06)
    """
    if not isinstance(r, (tuple, list)) or len(r) != 2:
        raise TypeError(
            f"Range must be tuple/list of length 2, got {type(r).__name__} "
            f"of length {len(r) if isinstance(r, (tuple, list)) else 'N/A'}"
            f"{' in ' + context if context else ''}"
        )

    min_val, max_val = r

    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise TypeError(
            f"Range values must be numeric, got [{type(min_val).__name__}, "
            f"{type(max_val).__name__}]{' in ' + context if context else ''}"
        )

    # I-06: Range Bounds Invariant (programmer error if violated - should be caught in Phase 2.1)
    if min_val > max_val:
        raise ValueError(
            f"Invalid range bounds: [{min_val}, {max_val}]. "
            f"Violates I-06: min must be ≤ max{' in ' + context if context else ''}"
        )


# ============================================================================
# HELPER FUNCTIONS (Single-value operations)
# ============================================================================

def range_contains(inner: Range, outer: Range) -> bool:
    """
    Check if inner range is contained within outer range.

    Semantics: inner ⊆ outer
    Implementation: outer.min ≤ inner.min AND inner.max ≤ outer.max

    Used by RANGE constraint evaluation (M-11, M-16, M-21).

    Args:
        inner: The range that should be contained
        outer: The range that should contain inner

    Returns:
        True if inner ⊆ outer, False otherwise

    Examples:
        range_contains([50, 60], [40, 80]) → True
        range_contains([256, 256], [256, 512]) → True (EXACT within range)
        range_contains([100, 200], [50, 150]) → False (inner.max exceeds outer.max)
    """
    _validate_range(inner, "inner")
    _validate_range(outer, "outer")

    inner_min, inner_max = inner
    outer_min, outer_max = outer

    # inner ⊆ outer ⟺ outer_min ≤ inner_min AND inner_max ≤ outer_max
    return outer_min <= inner_min and inner_max <= outer_max


def satisfies_min_constraint(required_min: float, candidate_range: Range) -> bool:
    """
    Check if candidate range satisfies minimum constraint.

    Semantics (from M-30 Range Comparison Operations):
    - MIN constraint: B_range.min >= A_min
    - Candidate must meet or exceed threshold across ENTIRE range

    Used by MIN constraint rules (M-09, M-14, M-19).

    Args:
        required_min: Minimum threshold value
        candidate_range: Candidate's value range [min, max]

    Returns:
        True if candidate_range.min >= required_min, False otherwise

    Examples:
        satisfies_min_constraint(4.0, [4.5, 5.0]) → True
        satisfies_min_constraint(4.0, [4.0, 4.0]) → True (EXACT match at boundary)
        satisfies_min_constraint(4.0, [3.5, 5.0]) → False (range includes values below min)
    """
    if not isinstance(required_min, (int, float)):
        raise TypeError(f"required_min must be numeric, got {type(required_min).__name__}")

    _validate_range(candidate_range, "candidate_range")

    candidate_min, _ = candidate_range

    # Pass ⇔ candidate_min >= required_min
    # Rationale: Entire candidate range must be >= threshold
    return candidate_min >= required_min


def satisfies_max_constraint(required_max: float, candidate_range: Range) -> bool:
    """
    Check if candidate range satisfies maximum constraint.

    Semantics (from M-30 Range Comparison Operations):
    - MAX constraint: B_range.max <= A_max
    - Candidate must not exceed threshold across ENTIRE range

    Used by MAX constraint rules (M-10, M-15, M-20).

    Args:
        required_max: Maximum threshold value
        candidate_range: Candidate's value range [min, max]

    Returns:
        True if candidate_range.max <= required_max, False otherwise

    Examples:
        satisfies_max_constraint(100, [50, 95]) → True
        satisfies_max_constraint(100, [100, 100]) → True (EXACT match at boundary)
        satisfies_max_constraint(100, [50, 150]) → False (range includes values above max)
    """
    if not isinstance(required_max, (int, float)):
        raise TypeError(f"required_max must be numeric, got {type(required_max).__name__}")

    _validate_range(candidate_range, "candidate_range")

    _, candidate_max = candidate_range

    # Pass ⇔ candidate_max <= required_max
    # Rationale: Entire candidate range must be <= threshold
    return candidate_max <= required_max


# ============================================================================
# CONSTRAINT EVALUATION FUNCTIONS (Dict-level operations)
# ============================================================================

def evaluate_min_constraints(
    required_min: Dict[str, float],
    candidate_ranges: Dict[str, Range]
) -> bool:
    """
    Evaluate MIN constraints for all required attributes.

    Enforces Canon Rules:
    - M-09: Item Min Constraint Rule
    - M-14: Other-Self Min Constraint Rule
    - M-19: Self-Other Min Constraint Rule

    Semantics:
    - For each key in required_min: candidate_ranges[key].min >= required_min[key]
    - Empty required_min → VACUOUSLY TRUE (no requirements)
    - Missing key in candidate_ranges → FAIL (required attribute absent)
    - Extra keys in candidate_ranges → IGNORED (not required)

    Args:
        required_min: Dict of minimum thresholds {attribute: min_value}
        candidate_ranges: Dict of candidate ranges {attribute: [min, max]}

    Returns:
        True if all MIN constraints satisfied, False otherwise

    Examples:
        required = {"rating": 4.0}
        candidate = {"rating": [4.5, 5.0]}
        → True (4.5 >= 4.0)

        required = {"rating": 4.0, "experience": 36}
        candidate = {"rating": [4.5, 5.0], "experience": [48, 48]}
        → True (both constraints satisfied)

        required = {"rating": 4.0}
        candidate = {"experience": [48, 48]}
        → False (required attribute "rating" missing)

        required = {}
        candidate = anything
        → True (vacuously satisfied - no requirements)
    """
    if not isinstance(required_min, dict):
        raise TypeError(f"required_min must be dict, got {type(required_min).__name__}")

    if not isinstance(candidate_ranges, dict):
        raise TypeError(f"candidate_ranges must be dict, got {type(candidate_ranges).__name__}")

    # Empty constraint set → VACUOUSLY SATISFIED
    if not required_min:
        return True

    # Check each required attribute
    for key, min_threshold in required_min.items():
        # Missing attribute in candidate → FAIL
        if key not in candidate_ranges:
            return False

        candidate_range = candidate_ranges[key]

        # Validate and check constraint
        if not satisfies_min_constraint(min_threshold, candidate_range):
            return False

    # All constraints satisfied
    return True


def evaluate_max_constraints(
    required_max: Dict[str, float],
    candidate_ranges: Dict[str, Range]
) -> bool:
    """
    Evaluate MAX constraints for all required attributes.

    Enforces Canon Rules:
    - M-10: Item Max Constraint Rule
    - M-15: Other-Self Max Constraint Rule
    - M-20: Self-Other Max Constraint Rule

    Semantics:
    - For each key in required_max: candidate_ranges[key].max <= required_max[key]
    - Empty required_max → VACUOUSLY TRUE (no requirements)
    - Missing key in candidate_ranges → FAIL (required attribute absent)
    - Extra keys in candidate_ranges → IGNORED (not required)

    Args:
        required_max: Dict of maximum thresholds {attribute: max_value}
        candidate_ranges: Dict of candidate ranges {attribute: [min, max]}

    Returns:
        True if all MAX constraints satisfied, False otherwise

    Examples:
        required = {"price": 100000}
        candidate = {"price": [95000, 95000]}
        → True (95000 <= 100000)

        required = {"price": 100000}
        candidate = {"price": [50000, 150000]}
        → False (range max 150000 > 100000)

        required = {"price": 100000}
        candidate = {"rating": [4.5, 5.0]}
        → False (required attribute "price" missing)

        required = {}
        candidate = anything
        → True (vacuously satisfied - no requirements)
    """
    if not isinstance(required_max, dict):
        raise TypeError(f"required_max must be dict, got {type(required_max).__name__}")

    if not isinstance(candidate_ranges, dict):
        raise TypeError(f"candidate_ranges must be dict, got {type(candidate_ranges).__name__}")

    # Empty constraint set → VACUOUSLY SATISFIED
    if not required_max:
        return True

    # Check each required attribute
    for key, max_threshold in required_max.items():
        # Missing attribute in candidate → FAIL
        if key not in candidate_ranges:
            return False

        candidate_range = candidate_ranges[key]

        # Validate and check constraint
        if not satisfies_max_constraint(max_threshold, candidate_range):
            return False

    # All constraints satisfied
    return True


def evaluate_range_constraints(
    required_ranges: Dict[str, Range],
    candidate_ranges: Dict[str, Range]
) -> bool:
    """
    Evaluate RANGE constraints for all required attributes.

    Enforces Canon Rules:
    - M-11: Item Range Constraint Rule
    - M-16: Other-Self Range Constraint Rule
    - M-21: Self-Other Range Constraint Rule

    Semantics:
    - For each key in required_ranges: candidate_ranges[key] ⊆ required_ranges[key]
    - Subset means: required.min ≤ candidate.min AND candidate.max ≤ required.max
    - Empty required_ranges → VACUOUSLY TRUE (no requirements)
    - Missing key in candidate_ranges → FAIL (required attribute absent)
    - Extra keys in candidate_ranges → IGNORED (not required)

    Special Case (EXACT constraints):
    - EXACT represented as range[x, x] (per I-02: EXACT not a separate mode)
    - Candidate must also be [x, x] to satisfy

    Args:
        required_ranges: Dict of required ranges {attribute: [min, max]}
        candidate_ranges: Dict of candidate ranges {attribute: [min, max]}

    Returns:
        True if all RANGE constraints satisfied, False otherwise

    Examples:
        required = {"storage": [256, 256]}  # EXACT 256GB
        candidate = {"storage": [256, 256]}
        → True (exact match)

        required = {"price": [40000, 60000]}
        candidate = {"price": [55000, 55000]}
        → True (55000 within [40000, 60000])

        required = {"storage": [256, 256]}  # EXACT 256GB
        candidate = {"storage": [128, 512]}
        → False (range too wide, includes non-256 values)

        required = {"price": [40000, 60000]}
        candidate = {"price": [50000, 70000]}
        → False (candidate.max exceeds required.max)

        required = {}
        candidate = anything
        → True (vacuously satisfied - no requirements)
    """
    if not isinstance(required_ranges, dict):
        raise TypeError(f"required_ranges must be dict, got {type(required_ranges).__name__}")

    if not isinstance(candidate_ranges, dict):
        raise TypeError(f"candidate_ranges must be dict, got {type(candidate_ranges).__name__}")

    # Empty constraint set → VACUOUSLY SATISFIED
    if not required_ranges:
        return True

    # Check each required attribute
    for key, required_range in required_ranges.items():
        # Missing attribute in candidate → FAIL
        if key not in candidate_ranges:
            return False

        candidate_range = candidate_ranges[key]

        # Validate ranges and check containment
        # candidate ⊆ required (i.e., candidate is contained within required bounds)
        if not range_contains(candidate_range, required_range):
            return False

    # All constraints satisfied
    return True


# ============================================================================
# PHASE 2.2 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.2 COMPLETION REPORT
============================

1. WHICH CANON RULES (M-XX) ARE ENFORCED AND WHERE
---------------------------------------------------

MIN Constraint Rules:
- M-09: Item Min Constraint Rule
  * Location: evaluate_min_constraints() + satisfies_min_constraint()
  * Semantics: B_range.min >= A_min
  * Application: A.items[].min constraints vs B.items[] ranges

- M-14: Other-Self Min Constraint Rule
  * Location: evaluate_min_constraints() + satisfies_min_constraint()
  * Semantics: B.self value >= A.other.min
  * Application: A.other.min constraints vs B.self ranges

- M-19: Self-Other Min Constraint Rule
  * Location: evaluate_min_constraints() + satisfies_min_constraint()
  * Semantics: A.self value >= B.other.min
  * Application: B.other.min constraints vs A.self ranges

MAX Constraint Rules:
- M-10: Item Max Constraint Rule
  * Location: evaluate_max_constraints() + satisfies_max_constraint()
  * Semantics: B_range.max <= A_max
  * Application: A.items[].max constraints vs B.items[] ranges

- M-15: Other-Self Max Constraint Rule
  * Location: evaluate_max_constraints() + satisfies_max_constraint()
  * Semantics: B.self value <= A.other.max
  * Application: A.other.max constraints vs B.self ranges

- M-20: Self-Other Max Constraint Rule
  * Location: evaluate_max_constraints() + satisfies_max_constraint()
  * Semantics: A.self value <= B.other.max
  * Application: B.other.max constraints vs A.self ranges

RANGE Constraint Rules:
- M-11: Item Range Constraint Rule
  * Location: evaluate_range_constraints() + range_contains()
  * Semantics: B_range ⊆ A_range (A.min ≤ B.min AND B.max ≤ A.max)
  * Application: A.items[].range constraints vs B.items[] ranges
  * Special: Handles EXACT as range[x,x] per I-02

- M-16: Other-Self Range Constraint Rule
  * Location: evaluate_range_constraints() + range_contains()
  * Semantics: B.self range ⊆ A.other.range
  * Application: A.other.range constraints vs B.self ranges

- M-21: Self-Other Range Constraint Rule
  * Location: evaluate_range_constraints() + range_contains()
  * Semantics: A.self range ⊆ B.other.range
  * Application: B.other.range constraints vs A.self ranges

Canon Invariant Enforcement:
- I-06: Range Bounds Invariant (min ≤ max)
  * Location: _validate_range() helper
  * Note: Should never fail if Phase 2.1 worked correctly
  * Purpose: Catch programmer errors only

Range Semantics from M-30:
- All numeric values represented as ranges [min, max]
- No midpoint collapse
- No scalar extraction
- Constraints evaluated on full range, not point values
- Location: All three constraint evaluation functions


2. HOW EMPTY CONSTRAINT OBJECTS ARE HANDLED
--------------------------------------------

Empty Constraint Dict → VACUOUSLY SATISFIED (returns True):

Examples:
  evaluate_min_constraints({}, anything) → True
  evaluate_max_constraints({}, anything) → True
  evaluate_range_constraints({}, anything) → True

Rationale:
- No requirements = no way to fail
- Consistent with mathematical interpretation: ∀x ∈ ∅: P(x) is vacuously true
- Phase 2.1 ensures constraint objects exist (never null)
- Empty dicts are valid and mean "no constraint"

Missing Attributes:
- If required dict has key K but candidate dict does not → FAIL (returns False)
- If required dict is empty → candidate dict contents irrelevant (returns True)
- If required dict has key K and candidate has K → evaluate constraint

Examples:
  required_min = {"rating": 4.0}
  candidate = {}
  → False (required attribute missing)

  required_min = {}
  candidate = {"rating": [5.0, 5.0]}
  → True (no requirements, extra attributes ignored)

  required_min = {"rating": 4.0}
  candidate = {"rating": [4.5, 5.0], "experience": [60, 60]}
  → True ("experience" ignored, only "rating" checked)

This aligns with:
- M-13 to M-21: Only check constraints that are specified
- Subset semantics: A's requirements ⊆ B's capabilities
- Empty set is subset of any set


3. WHAT CASES CORRECTLY FAIL
-----------------------------

Constraint Violations:
✓ MIN: candidate.min < required_min
  Example: satisfies_min_constraint(4.0, [3.5, 5.0]) → False

✓ MAX: candidate.max > required_max
  Example: satisfies_max_constraint(100, [50, 150]) → False

✓ RANGE: candidate not contained in required range
  Example: range_contains([100, 200], [50, 150]) → False (exceeds max)
  Example: range_contains([10, 100], [50, 150]) → False (below min)

Missing Required Attributes:
✓ Required key not in candidate dict
  Example: evaluate_min_constraints({"rating": 4.0}, {}) → False

EXACT Constraint Failures:
✓ EXACT value mismatch (range[x,x] not matching)
  Example: evaluate_range_constraints(
    {"storage": [256, 256]},  # EXACT 256
    {"storage": [512, 512]}   # EXACT 512
  ) → False

✓ EXACT vs range mismatch
  Example: evaluate_range_constraints(
    {"storage": [256, 256]},    # EXACT 256
    {"storage": [128, 512]}     # Range including 256
  ) → False (range too wide)

Programmer Errors (exceptions raised):
✓ Invalid range structure: not tuple/list of length 2
✓ Non-numeric range values
✓ Range bounds violation: min > max (I-06)
✓ Wrong argument types (non-dict where dict expected)


4. WHAT CASES CORRECTLY PASS
-----------------------------

Satisfied Constraints:
✓ MIN: candidate.min >= required_min
  Example: satisfies_min_constraint(4.0, [4.5, 5.0]) → True
  Example: satisfies_min_constraint(4.0, [4.0, 4.0]) → True (boundary)

✓ MAX: candidate.max <= required_max
  Example: satisfies_max_constraint(100, [50, 95]) → True
  Example: satisfies_max_constraint(100, [100, 100]) → True (boundary)

✓ RANGE: candidate ⊆ required
  Example: range_contains([50, 60], [40, 80]) → True
  Example: range_contains([256, 256], [256, 256]) → True (EXACT match)

Empty Constraint Sets (Vacuous Truth):
✓ evaluate_min_constraints({}, anything) → True
✓ evaluate_max_constraints({}, anything) → True
✓ evaluate_range_constraints({}, anything) → True

All Required Attributes Present and Satisfied:
✓ evaluate_min_constraints(
    {"rating": 4.0, "experience": 36},
    {"rating": [4.5, 5.0], "experience": [48, 60]}
  ) → True

Extra Attributes Ignored:
✓ evaluate_min_constraints(
    {"rating": 4.0},
    {"rating": [4.5, 5.0], "experience": [48, 60], "other": [10, 20]}
  ) → True (only "rating" checked)

EXACT Constraints:
✓ EXACT match
  Example: evaluate_range_constraints(
    {"storage": [256, 256]},
    {"storage": [256, 256]}
  ) → True

✓ Single value within range
  Example: evaluate_range_constraints(
    {"price": [40000, 60000]},
    {"price": [55000, 55000]}  # EXACT 55000
  ) → True

Unbounded Ranges:
✓ Unbounded min: [-∞, 100]
✓ Unbounded max: [50, +∞]
✓ Fully unbounded: [-∞, +∞]
  (All valid range structures)


5. WHAT ASSUMPTIONS WERE EXPLICITLY REJECTED
---------------------------------------------

❌ REJECTED: Scalar Collapse / Midpoint
- Did NOT collapse range[a,b] to (a+b)/2
- Did NOT extract single "representative" value
- Reason: Violates Phase 1.1 audit resolution (Blocker 1)
- Impact: All comparisons use full range semantics

❌ REJECTED: Missing Attributes as Fail-Safe
- Did NOT treat missing candidate attributes as auto-pass
- Did NOT allow partial matches
- Reason: Strict matching philosophy (I-01, I-03)
- Impact: Required attributes MUST be present

❌ REJECTED: Default/Inferred Constraints
- Did NOT assume default min = 0
- Did NOT assume default max = infinity
- Did NOT infer constraints from attribute names
- Reason: Per directive "No defaults or inferred constraints"
- Impact: Only explicitly specified constraints evaluated

❌ REJECTED: Hardcoded Attribute Names
- Did NOT special-case "price", "rating", "experience", etc.
- Did NOT apply domain-specific logic
- Reason: Per directive "No hardcoded attribute names"
- Impact: All attributes treated uniformly

❌ REJECTED: Fuzzy/Approximate Matching
- Did NOT allow "close enough" matches
- Did NOT use tolerance thresholds
- Did NOT round values
- Reason: Strict matching (I-01), canon requires exact comparison
- Impact: Boundary cases like [4.0, 4.0] vs requirement 4.0 have exact semantics

❌ REJECTED: Best-Effort Evaluation
- Did NOT continue after first failure
- Did NOT return partial results
- Did NOT score match quality
- Reason: Boolean constraint satisfaction only
- Impact: Returns True or False, no gradations

❌ REJECTED: Automatic Type Coercion
- Did NOT convert ints to floats automatically (Python does this)
- Did NOT parse strings to numbers
- Did NOT accept range as single number
- Reason: Assumes Phase 2.1 normalization complete
- Impact: Type errors raise exceptions (programmer errors)

❌ REJECTED: Bidirectional Orchestration
- Did NOT compare A vs B listings
- Did NOT implement forward/reverse checks
- Did NOT orchestrate other→self vs self→other
- Reason: Per directive "No A/B listing orchestration logic"
- Impact: Functions are primitive operations, not full matching

❌ REJECTED: Categorical/Exclusion Logic
- Did NOT implement categorical matching
- Did NOT implement exclusion checking
- Did NOT implement term implications
- Reason: Per directive scope limitation
- Impact: This module is purely numeric

❌ REJECTED: Empty as "Any Value"
- Did NOT treat empty candidate range as wildcard
- Did NOT treat missing candidate attribute as "matches anything"
- Reason: Missing attribute → FAIL (strict matching)
- Impact: Explicit presence required

❌ REJECTED: Range Expansion
- Did NOT expand point values [x, x] to ranges
- Did NOT widen ranges for "tolerance"
- Reason: Ranges are canonical form from Phase 2.1
- Impact: EXACT constraints work via range[x,x]


CRITICAL DESIGN DECISIONS
--------------------------

1. Range-First Semantics:
   - All values are ranges, no exceptions
   - Single values represented as [x, x]
   - Enforces M-30 audit resolution

2. Vacuous Truth for Empty Constraints:
   - Mathematical correctness
   - Aligns with subset semantics
   - Enables optional constraints

3. Fail on Missing Required Attributes:
   - Strict matching (I-01)
   - No partial matches (I-03)
   - Explicit presence required

4. Primitive Operations Only:
   - No listing orchestration
   - No bidirectional logic
   - Pure functions for constraint checking
   - Enables composition in Phase 2.3+

5. Programmer Error Detection Only:
   - Type validation raises exceptions
   - Range bound validation raises exceptions
   - Data validation happened in Phase 2.1
   - Clear separation of concerns


SAFE FOR NEXT PHASE
--------------------

Phase 2.3 (Item Matching) and beyond can safely assume:

✓ All numeric constraints have Boolean evaluation functions
✓ Range semantics are consistent (no scalar collapse)
✓ Empty constraints handled correctly (vacuously true)
✓ Missing attributes detected correctly (fail)
✓ EXACT constraints work via range[x,x]
✓ All functions are pure (no side effects)
✓ All functions are stateless (no hidden state)
✓ Type errors are programmer errors (not data errors)
✓ Evaluation is deterministic (no randomness)

Next phase needs to:
- Orchestrate forward/reverse constraint checks
- Implement item matching (type + constraints)
- Implement categorical matching
- Implement exclusion checking
- Compose these primitives into full matching logic
"""
