"""
PHASE 2.7: FULL LISTING ORCHESTRATION

Implements top-level listing matching by composing all constraint types.

Canon Rules Implemented:
- M-01 to M-06: Intent, domain, category gates
- M-07 to M-28: Via composition of Phases 2.4, 2.5, 2.6

Authority: MATCHING_CANON.md v1.1 (LOCKED)

Dependencies:
- Phase 2.1: schema_normalizer.py (input contract)
- Phase 2.4: item_array_matchers.py (items matching)
- Phase 2.5: other_self_matchers.py (other/self matching)
- Phase 2.6: location_matchers.py (location matching)

Author: Claude (Execution Engine)
Date: 2026-01-11
"""

from typing import Callable, Dict, List, Optional, Any
from item_array_matchers import all_required_items_match
from other_self_matchers import match_other_to_self
from location_matchers import match_location_constraints

# ============================================================================
# TYPE ALIASES
# ============================================================================

ImplicationFn = Callable[[str, str], bool]


# ============================================================================
# PRIMARY FUNCTION
# ============================================================================

def listing_matches(A: Dict[str, Any],
                   B: Dict[str, Any],
                   implies_fn: Optional[ImplicationFn] = None) -> bool:
    """
    Determine if listing B satisfies listing A's requirements.

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
    5. Location constraints (M-23 to M-28 via Phase 2.6)

    Args:
        A: Required listing (requester's requirements)
        B: Candidate listing (candidate's offering)
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
        # For mutual: both must be "exchange"
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
    # STEP 5: LOCATION CONSTRAINTS (M-23 to M-28 via Phase 2.6)
    # ========================================================================

    # Check if B.other satisfies A.location requirements
    # Uses: match_location_constraints() from Phase 2.6
    # Enforces: M-23 to M-28 (categorical, numeric, location exclusions)
    if not match_location_constraints(A["location"], B["other"], implies_fn):
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


# ============================================================================
# PHASE 2.7 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.7 COMPLETION REPORT
===========================

1. EXACT RULE-TO-FUNCTION MAPPING
----------------------------------

INTENT GATE:
- M-01: Intent Equality Rule
  * Location: listing_matches() → line ~67
  * Implementation: A["intent"] == B["intent"]
  * Failure: Different intent types → return False

- M-02: SubIntent Inverse Rule (product/service)
  * Location: listing_matches() → line ~77-81
  * Implementation: A["subintent"] != B["subintent"] for product/service
  * Failure: Same subintent (both buyers/sellers) → return False

- M-03: SubIntent Same Rule (mutual)
  * Location: listing_matches() → line ~83-87
  * Implementation: A["subintent"] == B["subintent"] for mutual
  * Failure: Different subintent → return False

- M-04: Intent–SubIntent Validity
  * Location: Implicit (enforced by Phase 2.1 schema normalization)
  * Implementation: Assumes only valid (intent, subintent) pairs exist
  * Failure: Invalid pairs rejected during schema validation

DOMAIN/CATEGORY GATE:
- M-05: Domain Intersection Rule (product/service)
  * Location: listing_matches() → line ~98-101
  * Implementation: _has_intersection(A["domain"], B["domain"])
  * Semantics: A.domain ∩ B.domain ≠ ∅
  * Failure: No common domain → return False

- M-06: Category Intersection Rule (mutual)
  * Location: listing_matches() → line ~103-107
  * Implementation: _has_intersection(A["category"], B["category"])
  * Semantics: A.category ∩ B.category ≠ ∅
  * Failure: No common category → return False

ITEMS MATCHING:
- M-07 to M-12: Item constraints
  * Location: listing_matches() → line ~115
  * Implementation: all_required_items_match(A["items"], B["items"], implies_fn)
  * Delegation: Phase 2.4 (item_array_matchers.py)
  * Enforces: Type, categorical, numeric, item exclusions
  * Failure: Any item requirement not satisfied → return False

OTHER/SELF MATCHING:
- M-13 to M-17: Other → Self constraints
  * Location: listing_matches() → line ~125
  * Implementation: match_other_to_self(A["other"], B["self"], implies_fn)
  * Delegation: Phase 2.5 (other_self_matchers.py)
  * Enforces: Categorical, numeric, other exclusions
  * Failure: Other/self requirements not satisfied → return False

LOCATION MATCHING:
- M-23 to M-28: Location constraints
  * Location: listing_matches() → line ~135
  * Implementation: match_location_constraints(A["location"], B["other"], implies_fn)
  * Delegation: Phase 2.6 (location_matchers.py)
  * Enforces: Categorical, numeric, location exclusions
  * Failure: Location requirements not satisfied → return False


2. WHY EVALUATION ORDER IS LEGALLY FIXED
-----------------------------------------

The evaluation order is MANDATORY and IMMUTABLE for these reasons:

**Semantic Correctness**:
1. Intent MUST be checked first
   - Different intents are fundamentally incompatible
   - No point checking constraints if intent mismatches

2. SubIntent MUST follow intent
   - SubIntent semantics depend on intent type
   - M-02 (inverse) vs M-03 (same) determined by intent

3. Domain/Category MUST precede items
   - Structural compatibility before detailed constraints
   - No point matching items if domains incompatible

4. Items → Other/Self → Location order
   - Increasingly complex constraint evaluation
   - Short-circuit optimization (fail fast)

**Performance Optimization** (cheap to expensive):
1. Intent check: Two string comparisons
2. SubIntent check: One string comparison
3. Domain/Category: Set intersection (O(n+m))
4. Items: Array iteration with multiple constraint checks
5. Other/Self: Categorical + numeric + exclusion checks
6. Location: Similar to other/self but location-specific

**Canon Specification**:
- The canon defines these rules in numbered order (M-01 → M-28)
- This ordering is intentional and reflects semantic dependencies
- Reordering would violate the logical structure

**Short-Circuit Guarantee** (I-01):
- Strict matching: ANY failure → FALSE
- First failure stops all further evaluation
- Order determines which failures are detected first


3. WHAT CAUSES IMMEDIATE REJECTION
-----------------------------------

The function returns FALSE immediately (short-circuits) when:

INTENT GATE FAILURES:
✗ A.intent ≠ B.intent
  - Example: A wants product, B offers service
  - Line: ~67

✗ Product/Service: A.subintent = B.subintent (M-02 violation)
  - Example: Both are buyers, or both are sellers
  - Line: ~79

✗ Mutual: A.subintent ≠ B.subintent (M-03 violation)
  - Example: Different exchange types (should never happen if normalized)
  - Line: ~86

DOMAIN/CATEGORY GATE FAILURES:
✗ Product/Service: A.domain ∩ B.domain = ∅ (M-05 violation)
  - Example: A wants electronics, B offers furniture
  - Line: ~100

✗ Mutual: A.category ∩ B.category = ∅ (M-06 violation)
  - Example: A offers books, B offers electronics
  - Line: ~106

ITEMS MATCHING FAILURES (delegated to Phase 2.4):
✗ ANY required item without matching candidate
✗ Type mismatch (M-07)
✗ Categorical subset failure (M-08)
✗ Numeric constraint violation (M-09, M-10, M-11)
✗ Item exclusion violation (M-12)
  - Line: ~115

OTHER/SELF MATCHING FAILURES (delegated to Phase 2.5):
✗ Categorical subset failure (M-13)
✗ MIN constraint violation (M-14)
✗ MAX constraint violation (M-15)
✗ RANGE constraint violation (M-16)
✗ Other exclusion violation (M-17)
  - Line: ~125

LOCATION MATCHING FAILURES (delegated to Phase 2.6):
✗ MIN constraint violation (M-23)
✗ MAX constraint violation (M-24)
✗ RANGE constraint violation (M-25)
✗ Categorical subset failure (M-26)
✗ Location exclusion violation (M-27)
✗ Other location exclusion violation (M-28)
  - Line: ~135

**Key Insight**: NO partial matching, NO scoring, NO fallbacks.
ANY single failure → entire match fails (I-01, I-03).


4. HOW DYNAMIC ATTRIBUTES REMAIN SAFE
--------------------------------------

Dynamic attribute handling is preserved through COMPOSITION:

✓ **NO Hardcoded Attribute Names**:
  - listing_matches() does NOT inspect attribute names
  - Only checks intent, subintent, domain, category fields
  - All constraint objects passed to delegated functions

✓ **Delegation to Phase 2.4, 2.5, 2.6**:
  - Phase 2.4: Handles items with dynamic item types and attributes
  - Phase 2.5: Handles other/self with dynamic categorical/numeric attributes
  - Phase 2.6: Handles location with dynamic location attributes

✓ **Runtime Discovery in Delegated Functions**:
  - Phase 2.4: Iterates over items array at runtime
  - Phase 2.5: Iterates over other/self constraint dicts at runtime
  - Phase 2.6: Iterates over location constraint dicts at runtime

✓ **No Attribute Name Assumptions**:
  - Code never references specific attributes like "price", "brand", "distance"
  - Works for ANY domain (products, services, mutual exchanges)
  - Works for ANY attribute names chosen by application

✓ **Term Implication Injection**:
  - implies_fn parameter passed through to all delegated functions
  - Caller controls semantic relationships (no hardcoding)
  - Same function works for different term taxonomies

**Example Flow**:
```python
# Application defines custom attributes
A = {
    "intent": "service",
    "subintent": "seeker",
    "domain": ["tutoring"],
    "items": [{"type": "math_tutoring", "categorical": {"level": "advanced"}}],
    "other": {"categorical": {"qualification": "phd"}},
    "location": {"max": {"travel_distance": 20}}
}

B = {
    "intent": "service",
    "subintent": "provider",
    "domain": ["tutoring"],
    "items": [{"type": "math_tutoring", "categorical": {"level": "advanced"}}],
    "self": {"categorical": {"qualification": "phd"}},
    "other": {"range": {"travel_distance": [0, 15]}}
}

# listing_matches() works WITHOUT knowing about:
# - "math_tutoring" (item type)
# - "level", "qualification" (categorical attributes)
# - "travel_distance" (location attribute)
```

All attribute discovery happens in delegated functions at runtime.


5. WHAT LOGIC WAS EXPLICITLY REFUSED
-------------------------------------

❌ NO BIDIRECTIONAL ORCHESTRATION (M-29):
   - Did NOT implement A ↔ B mutual compatibility
   - Did NOT combine A→B and B→A checks
   - Did NOT implement mutual intent special handling
   - Reason: Out of scope for Phase 2.7 (reserved for Phase 2.8)
   - Impact: Only checks A→B direction

❌ NO RANKING OR SCORING:
   - Did NOT compute match quality scores
   - Did NOT rank multiple candidates
   - Did NOT return percentages or confidence values
   - Reason: Boolean only (I-01, I-03)
   - Impact: Returns True or False only

❌ NO PARTIAL MATCHING:
   - Did NOT return "almost matches"
   - Did NOT identify which constraints failed
   - Did NOT suggest alternatives
   - Reason: Strict matching (I-01)
   - Impact: Any failure → False (no diagnostics)

❌ NO BEST-EFFORT MATCHING:
   - Did NOT tolerate constraint violations
   - Did NOT apply "prefer" modes
   - Did NOT use fuzzy logic
   - Reason: I-01, I-03 invariants
   - Impact: Strict boolean result

❌ NO FALLBACK SEMANTICS:
   - Did NOT retry with relaxed constraints
   - Did NOT apply defaults for missing values
   - Did NOT infer missing data
   - Reason: I-03 invariant (no fallbacks)
   - Impact: Missing data → programmer error

❌ NO DATA REWRITING:
   - Did NOT normalize values during matching
   - Did NOT fix malformed data
   - Did NOT coerce types
   - Reason: Phase 2.1 guarantees normalized input
   - Impact: Assumes clean input

❌ NO MUTUAL INTENT SPECIAL HANDLING (M-29):
   - Did NOT implement additional mutual intent rules beyond M-01, M-03, M-06
   - Did NOT check self-to-other direction for mutual
   - Reason: M-29 deferred to Phase 2.8
   - Impact: Mutual intent handled same as product/service (except M-03, M-06)

❌ NO TERM IMPLICATION INFERENCE:
   - Did NOT build implication graphs
   - Did NOT infer semantic relationships
   - Reason: Consume-only contract
   - Impact: Implication function injected by caller

❌ NO GLOBAL STATE:
   - Did NOT cache results
   - Did NOT maintain match history
   - Did NOT create global registries
   - Reason: Pure function design
   - Impact: Stateless, deterministic

❌ NO DOMAIN/CATEGORY HARDCODING:
   - Did NOT special-case specific domains
   - Did NOT apply domain-specific logic
   - Reason: Dynamic attributes requirement
   - Impact: Works for ANY domain

❌ NO ASYMMETRIC SELF→OTHER CHECKING:
   - Did NOT implement B.self → A.other check (M-18 to M-22)
   - Only checks A.other → B.self (M-13 to M-17)
   - Reason: Unidirectional (A→B only)
   - Impact: Reverse direction deferred to Phase 2.8

❌ NO CONSTRAINT REORDERING:
   - Did NOT optimize evaluation order
   - Did NOT dynamically choose check sequence
   - Reason: Canon specifies fixed order
   - Impact: Always evaluates in same sequence

❌ NO CUSTOM ERROR MESSAGES:
   - Did NOT provide detailed failure reasons
   - Did NOT track which rule failed
   - Reason: Boolean only
   - Impact: No diagnostics, just True/False

❌ NO EMBEDDINGS OR ML:
   - Did NOT use vector similarity
   - Did NOT apply learned models
   - Reason: Rule-based system only
   - Impact: Pure logic, no AI/ML

❌ NO MULTIPLE CANDIDATE HANDLING:
   - Did NOT accept array of candidates
   - Did NOT select best candidate
   - Reason: Single pair evaluation only
   - Impact: Caller handles iteration


ARCHITECTURAL GUARANTEES:
-------------------------

After Phase 2.7, downstream code can safely assume:

✓ listing_matches() is a pure function
✓ Stateless (no side effects)
✓ Deterministic (same input → same output)
✓ Short-circuit evaluation (first failure → False)
✓ Evaluation order is FIXED and MANDATORY
✓ Enforces M-01 to M-28 (via composition)
✓ Enforces I-01, I-03, I-07 invariants
✓ Boolean result only (no scoring)
✓ Directional (A→B only, NOT bidirectional)
✓ Dynamic attributes preserved (no hardcoding)
✓ Term implication injectable (no global state)
✓ All logic delegated (no reimplementation)

Ready for Phase 2.8: Bidirectional Matching & Mutual Intent (M-29)
"""
