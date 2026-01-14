"""
PHASE 2.6: LOCATION CONSTRAINT MATCHING

Implements location-specific constraint matching.

Canon Rules Implemented:
- M-23 to M-28: Location constraints

Authority: MATCHING_CANON.md v1.1 (LOCKED)

Dependencies:
- Phase 2.1: schema_normalizer.py (input contract)
- Phase 2.2: numeric_constraints.py (numeric evaluation)

LOCATION ABSTRACTION:
This module treats location as an ABSTRACT constraint object.
NO assumptions about:
- GPS coordinates
- Distance calculations
- Geographic information systems
- Specific location formats

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


def flatten_location_categorical_values(categorical: Dict[str, str]) -> Set[str]:
    """
    Extract all values from location categorical constraint object.

    Used for exclusion checking (M-27, M-28).
    Does NOT include attribute names, only values.

    Args:
        categorical: Normalized location categorical constraint object

    Returns:
        Set of all location categorical attribute values

    Example:
        >>> flatten_location_categorical_values({"area": "downtown", "zone": "commercial"})
        {"downtown", "commercial"}
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
        obj: Constraint object with min/max/range fields
        attr: Attribute name to extract

    Returns:
        Range tuple (min, max) or None if attribute not found

    Example:
        >>> _extract_numeric_range({"range": {"distance": [0, 10]}}, "distance")
        (0, 10)
        >>> _extract_numeric_range({"min": {"radius": 5}}, "radius")
        (5, inf)
        >>> _extract_numeric_range({"max": {"distance": 50}}, "distance")
        (-inf, 50)
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
# LOCATION CONSTRAINT MATCHING (M-23 to M-28)
# ============================================================================

def match_location_constraints(required_location: Dict[str, Any],
                               candidate_other: Dict[str, Any],
                               implies_fn: Optional[ImplicationFn] = None) -> bool:
    """
    M-23 to M-28: Check if candidate_other satisfies required_location constraints.

    Direction: A.location → B.other
    - required_location = A.location (what A requires about location from B)
    - candidate_other = B.other (what B offers, includes location info)

    This checks if B's offering (other) satisfies A's location requirements.

    Evaluation order (cheap to expensive):
    1. Categorical subset (M-26)
    2. Numeric constraints (M-23, M-24, M-25)
    3. Location exclusions (M-27)
    4. Other location exclusions (M-28)

    Short-circuits on first failure.

    Args:
        required_location: Required location constraints (A.location)
        candidate_other: Candidate offering (B.other)
        implies_fn: Optional term implication function for categorical matching

    Returns:
        True if candidate_other satisfies ALL required_location constraints, False otherwise

    Raises:
        KeyError: If required fields are missing (programmer error)
    """
    if implies_fn is None:
        implies_fn = _exact_match_only

    # M-26: Location-Other Categorical Subset Rule
    # Semantics: required_location.categorical ⊆ candidate_other.categorical
    # Every attribute in required_location.categorical must exist in candidate_other.categorical
    # with a compatible value (exact match or implied)
    for attr, required_value in required_location["categorical"].items():
        if attr not in candidate_other["categorical"]:
            return False  # Required attribute missing in candidate
        candidate_value = candidate_other["categorical"][attr]
        if not implies_fn(candidate_value, required_value):
            return False  # Value mismatch (no implication)

    # M-23: Location Min Constraint Rule
    # Semantics: For all k in required_location.min: candidate_other[k] >= required_location.min[k]
    # Candidate's value must meet or exceed the minimum requirement
    for attr, required_min in required_location["min"].items():
        candidate_range = _extract_numeric_range(candidate_other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_min_constraint(required_min, candidate_range):
            return False  # Candidate value too low

    # M-24: Location Max Constraint Rule
    # Semantics: For all k in required_location.max: candidate_other[k] <= required_location.max[k]
    # Candidate's value must not exceed the maximum requirement
    for attr, required_max in required_location["max"].items():
        candidate_range = _extract_numeric_range(candidate_other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not satisfies_max_constraint(required_max, candidate_range):
            return False  # Candidate value too high

    # M-25: Location Range Constraint Rule
    # Semantics: For all k in required_location.range: candidate_other.range[k] ⊆ required_location.range[k]
    # Candidate's range must be fully contained within required range
    for attr, required_range in required_location["range"].items():
        candidate_range = _extract_numeric_range(candidate_other, attr)
        if candidate_range is None:
            return False  # Required attribute missing in candidate
        if not range_contains(candidate_range, tuple(required_range)):
            return False  # Candidate range not subset of required range

    # M-27: Location Exclusion Disjoint Rule
    # Semantics: required_location.locationexclusions ∩ Flatten(candidate_other.categorical) = ∅
    # None of the location-excluded values can appear in candidate's categorical values
    # This is STRICT: any overlap → rejection (I-07 invariant)
    if required_location["locationexclusions"]:  # Empty list → no check needed
        candidate_values = flatten_location_categorical_values(candidate_other["categorical"])
        exclusions = set(required_location["locationexclusions"])
        if exclusions & candidate_values:  # Non-empty intersection
            return False  # Location exclusion violated

    # M-28: Other Location Exclusion Disjoint Rule
    # Semantics: candidate_other.otherlocationexclusions ∩ Flatten(required_location.categorical) = ∅
    # None of the candidate's other-location-excluded values can appear in required location's categorical
    # This is the REVERSE check: what candidate excludes must not be in what's required
    if candidate_other["otherlocationexclusions"]:  # Empty list → no check needed
        location_values = flatten_location_categorical_values(required_location["categorical"])
        exclusions = set(candidate_other["otherlocationexclusions"])
        if exclusions & location_values:  # Non-empty intersection
            return False  # Other location exclusion violated

    # All constraints satisfied
    return True


# ============================================================================
# PHASE 2.6 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.6 COMPLETION REPORT
===========================

1. CANON RULES ENFORCED (M-23 to M-28)
--------------------------------------

LOCATION CONSTRAINT MATCHING:

- M-23: Location Min Constraint Rule
  * Function: match_location_constraints() → min iteration
  * Semantics: For all k: candidate_other[k] >= required_location.min[k]
  * Implementation: Delegates to Phase 2.2 satisfies_min_constraint()
  * Line: ~190-197

- M-24: Location Max Constraint Rule
  * Function: match_location_constraints() → max iteration
  * Semantics: For all k: candidate_other[k] <= required_location.max[k]
  * Implementation: Delegates to Phase 2.2 satisfies_max_constraint()
  * Line: ~199-206

- M-25: Location Range Constraint Rule
  * Function: match_location_constraints() → range iteration
  * Semantics: For all k: candidate_other.range[k] ⊆ required_location.range[k]
  * Implementation: Delegates to Phase 2.2 range_contains()
  * Line: ~208-215

- M-26: Location-Other Categorical Subset Rule
  * Function: match_location_constraints() → categorical iteration
  * Semantics: required_location.categorical ⊆ candidate_other.categorical
  * Implementation: Subset check with term implications (injected)
  * Line: ~173-181

- M-27: Location Exclusion Disjoint Rule
  * Function: match_location_constraints() → location exclusion check
  * Semantics: required_location.locationexclusions ∩ Flatten(candidate_other.categorical) = ∅
  * Implementation: Set intersection check (strict disjoint, I-07)
  * Line: ~217-224

- M-28: Other Location Exclusion Disjoint Rule
  * Function: match_location_constraints() → other location exclusion check
  * Semantics: candidate_other.otherlocationexclusions ∩ Flatten(required_location.categorical) = ∅
  * Implementation: Set intersection check (strict disjoint, I-07)
  * Line: ~226-233


2. LOCATION ABSTRACTION RESPECTED
----------------------------------

✓ NO GPS coordinate assumptions
✓ NO distance calculations implemented
✓ NO geographic information system (GIS) dependencies
✓ NO hardcoded location formats (lat/long, addresses, etc.)
✓ Location treated as ABSTRACT constraint object
✓ Numeric attributes can be ANY location-related metric:
  - distance (miles, km, etc.)
  - radius
  - elevation
  - travel_time
  - zone_level
  - ANY domain-specific numeric attribute
✓ Categorical attributes can be ANY location-related term:
  - area, zone, district, region
  - city, state, country
  - accessibility, terrain_type
  - ANY domain-specific categorical attribute

Location abstraction design:
- Reuses Phase 2.2 numeric constraint evaluators (domain-agnostic)
- Uses same categorical subset logic as other/self (Phase 2.5 pattern)
- No special location logic beyond constraint matching
- Works for ANY location representation chosen by application layer


3. DYNAMIC ATTRIBUTE HANDLING
------------------------------

✓ NO hardcoded attribute names (no "distance", "area", "zone", etc.)
✓ Runtime iteration over constraint dictionaries (categorical, min, max, range)
✓ Works for ANY location attribute names (domain-agnostic)
✓ Attribute discovery via .items() iteration
✓ Missing required attributes detected (return False)
✓ Extra candidate attributes ignored (subset semantics)

Implementation pattern:
```python
for attr, required_value in required_location_constraints.items():
    # attr discovered at runtime, not hardcoded
    if attr not in candidate_constraints:
        return False  # Required attribute missing
    # ... check value compatibility
```

This enables:
- Geographic distance matching (distance, radius, etc.)
- Administrative region matching (city, state, zone, etc.)
- Custom location metrics (travel_time, accessibility_score, etc.)
- ANY domain without code changes


4. EXCLUSION ENFORCEMENT
-------------------------

Exclusions are STRICT (I-07: Exclusion Strictness Invariant):

✓ Two exclusion types for location:
  1. M-27: locationexclusions (what requester excludes about candidate's location)
  2. M-28: otherlocationexclusions (what candidate excludes about requester's location)
✓ Set-based disjoint check: exclusions ∩ candidate_values = ∅
✓ ANY overlap → immediate rejection
✓ NO term implications applied to exclusions (exact match only)
✓ NO partial exclusion tolerance
✓ Empty exclusions → no violations (vacuously true)

Implementation:
```python
# M-27: locationexclusions
if required_location["locationexclusions"]:
    candidate_values = flatten_location_categorical_values(candidate_other["categorical"])
    exclusions = set(required_location["locationexclusions"])
    if exclusions & candidate_values:  # Non-empty intersection
        return False  # STRICT rejection

# M-28: otherlocationexclusions (REVERSE check)
if candidate_other["otherlocationexclusions"]:
    location_values = flatten_location_categorical_values(required_location["categorical"])
    exclusions = set(candidate_other["otherlocationexclusions"])
    if exclusions & location_values:  # Non-empty intersection
        return False  # STRICT rejection
```

Exclusion fields:
- M-27: required_location["locationexclusions"] → checked against candidate_other.categorical
- M-28: candidate_other["otherlocationexclusions"] → checked against required_location.categorical

Value extraction:
- flatten_location_categorical_values() extracts ONLY categorical attribute values
- Does NOT include attribute names
- Does NOT include numeric constraint values


5. ASSUMPTIONS EXPLICITLY REJECTED
-----------------------------------

❌ NO Geographic Information System (GIS):
   - Did NOT implement coordinate systems
   - Did NOT implement distance calculations
   - Did NOT use geographic libraries
   - Reason: Location abstraction requirement
   - Impact: Works with ANY location representation

❌ NO Hardcoded Location Formats:
   - Did NOT assume GPS coordinates (lat/long)
   - Did NOT assume addresses (street, city, zip)
   - Did NOT assume administrative regions
   - Reason: Location abstraction requirement
   - Impact: Domain-agnostic location matching

❌ NO Distance Calculations:
   - Did NOT implement haversine formula
   - Did NOT implement Euclidean distance
   - Did NOT implement great circle distance
   - Reason: Location abstraction requirement
   - Impact: Application layer provides distance metrics

❌ NO Listing-Level Orchestration:
   - Did NOT combine items + other/self + location
   - Did NOT implement full listing matching
   - Did NOT decide overall Match/No-Match
   - Reason: Out of scope for Phase 2.6
   - Impact: Phase 2.7+ will orchestrate

❌ NO Bidirectional Orchestration (M-29):
   - Did NOT implement mutual intent matching
   - Did NOT orchestrate forward AND reverse checks
   - Reason: Out of scope for Phase 2.6
   - Impact: Caller orchestrates bidirectional logic

❌ NO Item Matching Logic:
   - Did NOT re-implement item array matching
   - Did NOT call Phase 2.4 functions
   - Reason: Item constraints handled separately
   - Impact: Caller combines items + other/self + location

❌ NO Other/Self Matching Logic:
   - Did NOT re-implement other/self constraint matching
   - Did NOT call Phase 2.5 functions
   - Reason: Other/self constraints handled separately
   - Impact: Caller combines all constraint types

❌ NO Domain/Category Matching (M-01 to M-06):
   - Did NOT implement domain compatibility
   - Did NOT check listing categories
   - Reason: Preprocessing in Phase 2.1
   - Impact: Assumed pre-validated

❌ NO Term Implication Inference:
   - Did NOT invent implication rules
   - Did NOT build implication graph
   - Reason: Consume-only contract
   - Impact: Implication injected by caller

❌ NO Partial Matching:
   - Did NOT return scores or percentages
   - Did NOT count satisfied constraints
   - Reason: Boolean only (strict)
   - Impact: All-or-nothing result

❌ NO Best-Effort Matching:
   - Did NOT tolerate partial constraint satisfaction
   - Did NOT apply "prefer" modes
   - Reason: I-01, I-03 invariants
   - Impact: Any violation → False

❌ NO Exclusion Implications:
   - Did NOT apply term implications to exclusions
   - Did NOT infer excluded terms
   - Reason: Exclusions are literal (I-07)
   - Impact: Exact string match for exclusions

❌ NO Global State:
   - Did NOT create global implication database
   - Uses parameter injection only
   - Reason: Clean dependency injection
   - Impact: Testable, flexible

❌ NO Asymmetric Location Logic:
   - Location matching is ONE direction only (location → other)
   - Did NOT implement other → location (no such rule in canon)
   - Reason: Canon only defines M-23 to M-28
   - Impact: Unidirectional location matching


ARCHITECTURAL GUARANTEES:
-------------------------

After Phase 2.6, downstream code can safely assume:

✓ match_location_constraints() is a pure function
✓ Stateless (no side effects)
✓ Deterministic (same input → same output)
✓ Short-circuit evaluation (first failure → return False)
✓ Evaluation order: Categorical → Numeric → Exclusions (cheap to expensive)
✓ Empty constraints handled correctly (vacuous truth)
✓ Missing required attributes detected (return False)
✓ Phase 2.2 numeric logic reused (no duplication)
✓ Term implication injectable (no global state)
✓ Dynamic attributes supported (runtime discovery)
✓ Exclusions strictly enforced (I-07)
✓ Location abstraction maintained (no GIS assumptions)
✓ Works with ANY location representation

Ready for Phase 2.7+: Full Listing Orchestration & Bidirectional Matching
"""
