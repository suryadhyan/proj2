"""
PHASE 2.7: FULL LISTING ORCHESTRATION (V2)

V2 Changes:
- Works with transformed data from schema_normalizer_v2
- Uses location_matcher_v2 for simplified location matching
- All other logic unchanged (data already transformed to OLD format)

Implements top-level listing matching by composing all constraint types.

Canon Rules Implemented:
- M-01 to M-06: Intent, domain, category gates
- M-07 to M-28: Via composition of Phases 2.4, 2.5, 2.6

Authority: MATCHING_CANON.md v1.1 (LOCKED)

Dependencies:
- Phase 2.1: schema_normalizer_v2.py (transforms NEW → OLD format)
- Phase 2.4: item_array_matchers.py (items matching)
- Phase 2.5: other_self_matchers.py (other/self matching)
- Phase 2.6: location_matcher_v2.py (simplified location matching)

Author: Claude (Execution Engine V2)
Date: 2026-01-13
"""

from typing import Callable, Dict, List, Optional, Any
from item_array_matchers import all_required_items_match
from other_self_matchers import match_other_to_self
from location_matcher_v2 import match_location_v2

# ============================================================================
# TYPE ALIASES
# ============================================================================

ImplicationFn = Callable[[str, str], bool]


# ============================================================================
# PRIMARY FUNCTION
# ============================================================================

def listing_matches_v2(A: Dict[str, Any],
                       B: Dict[str, Any],
                       implies_fn: Optional[ImplicationFn] = None) -> bool:
    """
    Determine if listing B satisfies listing A's requirements (V2).

    V2 Note: Expects data transformed by schema_normalizer_v2
    (NEW schema → OLD format with field renames + axis flattening)

    Direction: A → B (unidirectional)
    - A = requester (what they want)
    - B = candidate (what they offer)

    Returns True only if B satisfies ALL of A's constraints.
    Short-circuits on first failure (I-01: strict matching).

    Evaluation order (MANDATORY, from canon):
    1. Intent gate (M-01, M-02/M-03, M-04)
    2. Domain/Category gate (M-05 for product/service, M-06 for mutual)
    3. Items matching (M-07 to M-12 via Phase 2.4)
    4. Other → Self constraints (M-13 to M-17 via Phase 2.5)
    5. Location constraints (M-23 to M-28 via Phase 2.6 V2)

    Args:
        A: Required listing (requester's requirements) - transformed to OLD format
        B: Candidate listing (candidate's offering) - transformed to OLD format
        implies_fn: Optional term implication function for categorical matching

    Returns:
        True if B satisfies ALL A's constraints, False otherwise

    Raises:
        TypeError: If required fields are missing (programmer error)
    """

    # ========================================================================
    # STEP 1: INTENT GATE (M-01, M-02, M-03, M-04)
    # ========================================================================

    # M-01: Intent Equality Rule
    # Semantics: A.intent = B.intent
    # Both listings must have the same intent type
    if A["intent"] != B["intent"]:
        return False  # Intent mismatch

    # M-02, M-03: SubIntent Rules (depends on intent type)
    # M-04: Intent–SubIntent Validity (implicit - schema normalized)
    intent = A["intent"]

    if intent == "product" or intent == "service":
        # M-02: SubIntent Inverse Rule (product/service)
        # Semantics: A.subintent ≠ B.subintent
        # For product/service: buyer/seller, seeker/provider must be inverse
        if A["subintent"] == B["subintent"]:
            return False  # Same subintent (both buyers or both sellers)

    elif intent == "mutual":
        # M-03: SubIntent Same Rule (mutual)
        # Semantics: A.subintent = B.subintent
        # For mutual: both must be "connect"
        if A["subintent"] != B["subintent"]:
            return False  # Different subintent

    else:
        # Unknown intent type (should never happen if schema normalized)
        raise TypeError(f"Unknown intent type: {intent}")

    # ========================================================================
    # STEP 2: DOMAIN / CATEGORY GATE (M-05, M-06)
    # ========================================================================

    if intent == "product" or intent == "service":
        # M-05: Domain Intersection Rule (product/service)
        # Semantics: A.domain ∩ B.domain ≠ ∅
        # At least one common domain required
        if not _has_intersection(A["domain"], B["domain"]):
            return False  # No common domain

    elif intent == "mutual":
        # M-06: Category Intersection Rule (mutual)
        # Semantics: A.category ∩ B.category ≠ ∅
        # At least one common category required
        if not _has_intersection(A["category"], B["category"]):
            return False  # No common category

    # ========================================================================
    # STEP 3: ITEMS MATCHING (M-07 to M-12 via Phase 2.4)
    # ========================================================================

    # M-07 to M-12 Precondition: A.intent ∈ {product, service}
    # Items matching does NOT apply to mutual intent
    # For mutual: items represent what each party offers (not matched directly)
    # For mutual: matching is via other→self constraints (M-13 to M-22)
    if intent == "product" or intent == "service":
        # Check if B.items satisfies A.items requirements
        # Uses: all_required_items_match() from Phase 2.4
        # Enforces: M-07 to M-12 (type, categorical, numeric, exclusions)
        if not all_required_items_match(A["items"], B["items"], implies_fn):
            return False  # Items requirements not satisfied

    # ========================================================================
    # STEP 4: OTHER → SELF CONSTRAINTS (M-13 to M-17 via Phase 2.5)
    # ========================================================================

    # Check if B.self satisfies A.other requirements
    # Uses: match_other_to_self() from Phase 2.5
    # Enforces: M-13 to M-17 (categorical, numeric, otherexclusions)
    if not match_other_to_self(A["other"], B["self"], implies_fn):
        return False  # Other/self requirements not satisfied

    # ========================================================================
    # STEP 5: LOCATION CONSTRAINTS (M-23 to M-28 via Phase 2.6 V2)
    # ========================================================================

    # V2 Change: Use simplified location matching
    # Check if B.location matches A.location requirements
    if not _match_location_v2(A, B):
        return False  # Location requirements not satisfied

    # ========================================================================
    # ALL CONSTRAINTS SATISFIED
    # ========================================================================

    return True


# ============================================================================
# INTERNAL HELPER FUNCTIONS
# ============================================================================

def _has_intersection(list_a: List[str], list_b: List[str]) -> bool:
    """
    Check if two lists have at least one common element.

    Used for M-05 (domain intersection) and M-06 (category intersection).

    Args:
        list_a: First list (from A)
        list_b: Second list (from B)

    Returns:
        True if lists have non-empty intersection, False otherwise

    Examples:
        >>> _has_intersection(["electronics", "gadgets"], ["electronics"])
        True
        >>> _has_intersection(["electronics"], ["furniture"])
        False
        >>> _has_intersection([], ["electronics"])
        False
    """
    # Convert to sets for efficient intersection
    set_a = set(list_a)
    set_b = set(list_b)

    # Check if intersection is non-empty
    return bool(set_a & set_b)


def _match_location_v2(A: Dict, B: Dict) -> bool:
    """
    Helper function to call location_matcher_v2 with simplified interface.

    Extracts location fields from A and B and calls match_location_v2.

    Args:
        A: Required listing (with location, locationmode, locationexclusions)
        B: Candidate listing (with location, locationmode, locationexclusions)

    Returns:
        True if locations match, False otherwise
    """
    # Extract fields from A
    required_location = A.get("location", "")
    required_mode = A.get("locationmode", "near_me")
    required_exclusions = A.get("locationexclusions", [])

    # Extract fields from B
    candidate_location = B.get("location", "")
    candidate_mode = B.get("locationmode", "near_me")
    candidate_exclusions = B.get("locationexclusions", [])

    # Call v2 location matcher
    return match_location_v2(
        required_location,
        required_mode,
        required_exclusions,
        candidate_location,
        candidate_mode,
        candidate_exclusions
    )


# ============================================================================
# PHASE 2.7 V2 NOTES
# ============================================================================

"""
PHASE 2.7 V2 CHANGES
====================

What Changed:
--------------
1. Uses location_matcher_v2 for simplified location matching
   - Name-based equality instead of complex constraint objects
   - Supports 5 modes: near_me, explicit, target_only, route, global
   - No distance/zone/accessibility constraints

2. Expects transformed data from schema_normalizer_v2
   - NEW schema → OLD format transformation done upstream
   - Field names already renamed (other_party_preferences → other, etc.)
   - Axis constraints already flattened

3. All other logic unchanged
   - Intent gate (M-01 to M-04) same
   - Domain/Category gate (M-05, M-06) same
   - Items matching (M-07 to M-12) same via Phase 2.4
   - Other/Self matching (M-13 to M-17) same via Phase 2.5

What Stayed the Same:
---------------------
✓ Pure function (stateless, deterministic)
✓ Short-circuit evaluation (first failure → False)
✓ Fixed evaluation order (mandatory)
✓ Boolean result only (no scoring)
✓ Directional (A→B only)
✓ Dynamic attributes preserved
✓ Canon rules M-01 to M-28 enforced

Migration Path:
--------------
1. Schema input: Use schema_normalizer_v2 to transform NEW → OLD
2. Matching: Use listing_matches_v2 instead of listing_matches
3. Location: Automatically uses simplified v2 logic
4. Everything else: Works identically

Example Usage:
-------------
```python
from schema_normalizer_v2 import normalize_and_validate_v2
from listing_matcher_v2 import listing_matches_v2

# Transform NEW schema to OLD format
listing_a_old = normalize_and_validate_v2(listing_a_new)
listing_b_old = normalize_and_validate_v2(listing_b_new)

# Match using v2
matches = listing_matches_v2(listing_a_old, listing_b_old)
```
"""
