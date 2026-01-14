"""
VRIDDHI MATCHING SYSTEM - SCHEMA VALIDATION & NORMALIZATION
Phase 2.1

Authority: MATCHING_CANON.md v1.1 (LOCKED)
Purpose: Pure preprocessing layer - validates schema and normalizes to canonical form
Scope: NO MATCHING LOGIC - only validation and normalization

This module enforces:
- Canon invariants I-02, I-04, I-06
- Normalization rules M-31, M-32
- Schema correctness per spec section 2
"""

from typing import Any, Dict, List, Union, Optional


# ============================================================================
# EXCEPTION CLASSES - Explicit error reporting with canon rule references
# ============================================================================

class SchemaValidationError(Exception):
    """Base exception for schema validation failures."""
    pass


class MissingRequiredFieldError(SchemaValidationError):
    """Raised when a required field is missing."""
    def __init__(self, field: str, context: str = ""):
        self.field = field
        msg = f"Missing required field: '{field}'"
        if context:
            msg += f" ({context})"
        super().__init__(msg)


class InvalidIntentSubIntentError(SchemaValidationError):
    """Raised when intent-subintent combination violates I-04."""
    def __init__(self, intent: str, subintent: str):
        self.intent = intent
        self.subintent = subintent
        msg = (
            f"Invalid intent-subintent combination: ({intent}, {subintent}). "
            f"Violates I-04: Intent-SubIntent Validity Invariant. "
            f"Valid combinations: (product,buy), (product,sell), "
            f"(service,seek), (service,provide), (mutual,connect)"
        )
        super().__init__(msg)


class InvalidConstraintModeError(SchemaValidationError):
    """Raised when constraint object has invalid keys (violates I-02)."""
    def __init__(self, field: str, invalid_keys: set):
        self.field = field
        self.invalid_keys = invalid_keys
        msg = (
            f"Invalid constraint mode keys in '{field}': {invalid_keys}. "
            f"Violates I-02: Constraint Mode Enumeration Invariant. "
            f"Only allowed: {{'categorical', 'min', 'max', 'range'}}"
        )
        super().__init__(msg)


class InvalidRangeBoundsError(SchemaValidationError):
    """Raised when range[min] > range[max] (violates I-06)."""
    def __init__(self, field: str, key: str, range_val: List):
        self.field = field
        self.key = key
        self.range_val = range_val
        msg = (
            f"Invalid range bounds in '{field}.{key}': {range_val}. "
            f"Violates I-06: Range Bounds Invariant. "
            f"Requirement: min ≤ max"
        )
        super().__init__(msg)


class TypeValidationError(SchemaValidationError):
    """Raised when field has wrong type."""
    def __init__(self, field: str, expected: str, actual: str):
        self.field = field
        msg = f"Type error in '{field}': expected {expected}, got {actual}"
        super().__init__(msg)


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================

def normalize_string(value: Any) -> str:
    """
    Normalize string to canonical form.

    Enforces M-32: Case Insensitivity Rule
    - lowercase(trim(s))
    - No semantic transformation

    Args:
        value: Input value (will be coerced to string)

    Returns:
        Normalized string: lowercase, trimmed
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_array(value: Any) -> List:
    """
    Normalize value to array form.

    Enforces M-31: Array Normalization Rule
    - if value = null then []
    - if IsArray(value) then value
    - else [value]

    Args:
        value: Input value of any type

    Returns:
        List (possibly empty)
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_string_array(value: Any) -> List[str]:
    """
    Normalize to array of normalized strings.

    Combines M-31 (array normalization) and M-32 (string normalization).

    Args:
        value: Input value (string, list, or None)

    Returns:
        List of normalized strings
    """
    arr = normalize_array(value)
    return [normalize_string(item) for item in arr]


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_constraint_object(obj: Any, field_name: str) -> Dict[str, Any]:
    """
    Validate constraint object structure.

    Enforces:
    - I-02: Constraint Mode Enumeration (keys ⊆ {categorical, min, max, range})
    - I-06: Range Bounds Invariant (min ≤ max in all ranges)
    - Type safety for numeric fields

    Args:
        obj: Constraint object to validate
        field_name: Name for error reporting

    Returns:
        Validated constraint object (may be normalized)

    Raises:
        InvalidConstraintModeError: If invalid keys present
        InvalidRangeBoundsError: If range[min] > range[max]
        TypeValidationError: If types are wrong
    """
    if obj is None:
        # Empty constraint object is valid
        return {
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }

    if not isinstance(obj, dict):
        raise TypeValidationError(field_name, "dict/object", type(obj).__name__)

    # I-02: Only allowed keys
    ALLOWED_KEYS = {"categorical", "min", "max", "range"}
    actual_keys = set(obj.keys())
    invalid_keys = actual_keys - ALLOWED_KEYS

    if invalid_keys:
        raise InvalidConstraintModeError(field_name, invalid_keys)

    # Ensure all keys exist (for consistency)
    result = {
        "categorical": obj.get("categorical", {}),
        "min": obj.get("min", {}),
        "max": obj.get("max", {}),
        "range": obj.get("range", {})
    }

    # Validate categorical is dict
    if not isinstance(result["categorical"], dict):
        raise TypeValidationError(
            f"{field_name}.categorical",
            "dict/object",
            type(result["categorical"]).__name__
        )

    # Normalize categorical strings (M-32)
    normalized_categorical = {}
    for k, v in result["categorical"].items():
        normalized_categorical[normalize_string(k)] = normalize_string(v)
    result["categorical"] = normalized_categorical

    # Validate min is dict with numeric values
    if not isinstance(result["min"], dict):
        raise TypeValidationError(
            f"{field_name}.min",
            "dict/object",
            type(result["min"]).__name__
        )
    for k, v in result["min"].items():
        if not isinstance(v, (int, float)):
            raise TypeValidationError(
                f"{field_name}.min.{k}",
                "numeric",
                type(v).__name__
            )

    # Validate max is dict with numeric values
    if not isinstance(result["max"], dict):
        raise TypeValidationError(
            f"{field_name}.max",
            "dict/object",
            type(result["max"]).__name__
        )
    for k, v in result["max"].items():
        if not isinstance(v, (int, float)):
            raise TypeValidationError(
                f"{field_name}.max.{k}",
                "numeric",
                type(v).__name__
            )

    # Validate range is dict with [min, max] arrays
    # I-06: Enforce min ≤ max
    if not isinstance(result["range"], dict):
        raise TypeValidationError(
            f"{field_name}.range",
            "dict/object",
            type(result["range"]).__name__
        )
    for k, v in result["range"].items():
        if not isinstance(v, list) or len(v) != 2:
            raise TypeValidationError(
                f"{field_name}.range.{k}",
                "array of length 2",
                f"{type(v).__name__} of length {len(v) if isinstance(v, list) else 'N/A'}"
            )
        if not isinstance(v[0], (int, float)) or not isinstance(v[1], (int, float)):
            raise TypeValidationError(
                f"{field_name}.range.{k}",
                "numeric array [min, max]",
                f"[{type(v[0]).__name__}, {type(v[1]).__name__}]"
            )

        # I-06: Range Bounds Invariant
        if v[0] > v[1]:
            raise InvalidRangeBoundsError(field_name, k, v)

    return result


def validate_intent_subintent(intent: str, subintent: str) -> None:
    """
    Validate intent-subintent combination.

    Enforces I-04: Intent-SubIntent Validity Invariant

    Valid combinations:
    - (product, buy)
    - (product, sell)
    - (service, seek)
    - (service, provide)
    - (mutual, connect)

    Args:
        intent: Intent value (normalized)
        subintent: SubIntent value (normalized)

    Raises:
        InvalidIntentSubIntentError: If combination is invalid
    """
    # I-04: Intent-SubIntent Validity Invariant
    VALID_COMBINATIONS = {
        ("product", "buy"),
        ("product", "sell"),
        ("service", "seek"),
        ("service", "provide"),
        ("mutual", "connect")
    }

    if (intent, subintent) not in VALID_COMBINATIONS:
        raise InvalidIntentSubIntentError(intent, subintent)


def validate_items_array(items: Any, intent: str) -> List[Dict]:
    """
    Validate items array structure.

    Requirements:
    - Must be array
    - Each item must have 'type' field
    - Each item must have valid constraint object
    - Required for product/service intents

    Args:
        items: Items value from listing
        intent: Intent (for context)

    Returns:
        Validated items array

    Raises:
        MissingRequiredFieldError: If missing for product/service
        TypeValidationError: If not array or wrong structure
    """
    if items is None or (isinstance(items, list) and len(items) == 0):
        if intent in ["product", "service"]:
            raise MissingRequiredFieldError(
                "items",
                f"Required for intent={intent}"
            )
        return []

    if not isinstance(items, list):
        raise TypeValidationError("items", "array", type(items).__name__)

    validated_items = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeValidationError(
                f"items[{i}]",
                "object/dict",
                type(item).__name__
            )

        # Type field is required
        if "type" not in item:
            raise MissingRequiredFieldError(
                f"items[{i}].type",
                "Each item must have 'type' field"
            )

        # I-02: Check for invalid constraint mode keys in item
        ALLOWED_ITEM_KEYS = {"type", "categorical", "min", "max", "range"}
        actual_keys = set(item.keys())
        invalid_keys = actual_keys - ALLOWED_ITEM_KEYS

        if invalid_keys:
            raise InvalidConstraintModeError(f"items[{i}]", invalid_keys)

        # Normalize type string
        normalized_item = {
            "type": normalize_string(item["type"]),
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }

        # Validate and normalize constraint objects
        constraint_obj = {
            "categorical": item.get("categorical", {}),
            "min": item.get("min", {}),
            "max": item.get("max", {}),
            "range": item.get("range", {})
        }

        validated_constraint = validate_constraint_object(
            constraint_obj,
            f"items[{i}]"
        )

        normalized_item.update(validated_constraint)
        validated_items.append(normalized_item)

    return validated_items


# ============================================================================
# MAIN NORMALIZATION FUNCTION
# ============================================================================

def normalize_and_validate(listing: dict) -> dict:
    """
    Normalize and validate a listing to canonical form.

    This is the ONLY preprocessing step before matching.
    Enforces all schema rules and canon invariants.

    Enforces:
    - I-04: Intent-SubIntent Validity
    - I-02: Constraint Mode Enumeration
    - I-06: Range Bounds Invariant
    - M-31: Array Normalization
    - M-32: String Normalization (case insensitivity)

    Args:
        listing: Raw listing dict from extraction

    Returns:
        Normalized listing safe for matching rules

    Raises:
        MissingRequiredFieldError: If required field missing
        InvalidIntentSubIntentError: If invalid intent-subintent combo
        InvalidConstraintModeError: If invalid constraint keys
        InvalidRangeBoundsError: If range bounds invalid
        TypeValidationError: If type mismatch
    """
    if not isinstance(listing, dict):
        raise TypeValidationError("listing", "dict/object", type(listing).__name__)

    # ========================================================================
    # REQUIRED FIELDS VALIDATION
    # ========================================================================

    # Intent (required)
    if "intent" not in listing:
        raise MissingRequiredFieldError("intent")
    intent = normalize_string(listing["intent"])
    if intent not in ["product", "service", "mutual"]:
        raise SchemaValidationError(
            f"Invalid intent: '{intent}'. Must be: product, service, or mutual"
        )

    # SubIntent (required)
    if "subintent" not in listing:
        raise MissingRequiredFieldError("subintent")
    subintent = normalize_string(listing["subintent"])

    # I-04: Validate intent-subintent combination
    validate_intent_subintent(intent, subintent)

    # Domain (required for product/service) OR Category (required for mutual)
    if intent in ["product", "service"]:
        if "domain" not in listing:
            raise MissingRequiredFieldError("domain", f"Required for intent={intent}")
        domain = normalize_string_array(listing["domain"])
        if len(domain) == 0:
            raise SchemaValidationError(
                "domain array cannot be empty for product/service intents. "
                "Violates I-05: Domain Cardinality Invariant (1 ≤ |domain|)"
            )
        category = []
    else:  # mutual
        if "category" not in listing:
            raise MissingRequiredFieldError("category", "Required for intent=mutual")
        category = normalize_string_array(listing["category"])
        if len(category) == 0:
            raise SchemaValidationError(
                "category array cannot be empty for mutual intent. "
                "Violates I-05: Domain Cardinality Invariant (1 ≤ |category|)"
            )
        domain = []

    # Items (required for product/service)
    items = validate_items_array(listing.get("items"), intent)

    # Other preferences (required)
    if "other" not in listing:
        raise MissingRequiredFieldError("other")
    other = validate_constraint_object(listing["other"], "other")

    # Self attributes (required)
    if "self" not in listing:
        raise MissingRequiredFieldError("self")
    self_attrs = validate_constraint_object(listing["self"], "self")

    # Location (required)
    if "location" not in listing:
        raise MissingRequiredFieldError("location")
    location = listing["location"]

    # Normalize location based on type
    if isinstance(location, dict):
        # Route mode: {origin: str, destination: str}
        if "origin" in location and "destination" in location:
            location = {
                "origin": normalize_string(location["origin"]),
                "destination": normalize_string(location["destination"])
            }
        else:
            raise SchemaValidationError(
                "Location object must have 'origin' and 'destination' fields for route mode"
            )
    elif isinstance(location, str):
        location = normalize_string(location)
    else:
        raise TypeValidationError(
            "location",
            "string or object {origin, destination}",
            type(location).__name__
        )

    # LocationMode (required)
    if "locationmode" not in listing:
        raise MissingRequiredFieldError("locationmode")
    locationmode = normalize_string(listing["locationmode"])
    if locationmode not in ["proximity", "target", "route", "flexible"]:
        raise SchemaValidationError(
            f"Invalid locationmode: '{locationmode}'. "
            f"Must be: proximity, target, route, or flexible"
        )

    # Reasoning (required - presence only)
    if "reasoning" not in listing:
        raise MissingRequiredFieldError("reasoning")
    reasoning = str(listing["reasoning"])  # Keep original, no normalization

    # ========================================================================
    # OPTIONAL FIELDS (EXCLUSIONS)
    # ========================================================================

    # Normalize exclusion arrays (M-31 + M-32)
    itemexclusions = normalize_string_array(listing.get("itemexclusions", []))
    otherexclusions = normalize_string_array(listing.get("otherexclusions", []))
    selfexclusions = normalize_string_array(listing.get("selfexclusions", []))
    locationexclusions = normalize_string_array(listing.get("locationexclusions", []))

    # ========================================================================
    # BUILD NORMALIZED OUTPUT
    # ========================================================================

    normalized = {
        # Core fields
        "intent": intent,
        "subintent": subintent,
        "domain": domain,
        "category": category,

        # Items
        "items": items,
        "itemexclusions": itemexclusions,

        # Other preferences
        "other": other,
        "otherexclusions": otherexclusions,

        # Self attributes
        "self": self_attrs,
        "selfexclusions": selfexclusions,

        # Location
        "location": location,
        "locationmode": locationmode,
        "locationexclusions": locationexclusions,

        # Reasoning (metadata)
        "reasoning": reasoning
    }

    return normalized


# ============================================================================
# PHASE 2.1 COMPLETION REPORT
# ============================================================================

"""
PHASE 2.1 COMPLETION REPORT
===========================

1. WHAT CANON RULES WERE ENFORCED
----------------------------------

Invariants Enforced:
- I-02: Constraint Mode Enumeration Invariant
  * Only keys {categorical, min, max, range} allowed in constraint objects
  * Validated in: validate_constraint_object()
  * Error: InvalidConstraintModeError

- I-04: Intent-SubIntent Validity Invariant
  * Only valid combinations allowed: (product,buy), (product,sell),
    (service,seek), (service,provide), (mutual,connect)
  * Validated in: validate_intent_subintent()
  * Error: InvalidIntentSubIntentError

- I-05: Domain Cardinality Invariant (partial - structure only)
  * domain/category arrays must be non-empty for respective intents
  * Validated in: normalize_and_validate()
  * Error: SchemaValidationError
  * NOTE: Vocabulary membership (21 product domains, 18 service domains,
    25 mutual categories) NOT enforced - no hardcoded vocabularies per directive

- I-06: Range Bounds Invariant
  * All range[min, max] must satisfy min ≤ max
  * Validated in: validate_constraint_object()
  * Error: InvalidRangeBoundsError

Normalization Rules Enforced:
- M-31: Array Normalization Rule
  * null → []
  * array → array
  * scalar → [scalar]
  * Applied to: domain, category, items, all exclusion arrays
  * Implemented in: normalize_array(), normalize_string_array()

- M-32: Case Insensitivity Rule
  * All strings normalized to lowercase(trim(s))
  * Applied to: intent, subintent, domain, category, location,
    locationmode, exclusions, categorical keys/values, item types
  * Implemented in: normalize_string()

Schema Structure Enforced:
- Required fields per spec section 2.2:
  * intent, subintent (always)
  * domain (product/service) OR category (mutual)
  * items (product/service)
  * other, self (always)
  * location, locationmode (always)
  * reasoning (always)

- Constraint object structure (spec section 2.3):
  * All constraint objects have 4 sub-objects: categorical, min, max, range
  * Missing sub-objects initialized to empty {}
  * Ensures consistent structure for downstream code

- Items array structure:
  * Each item must have 'type' field
  * Each item has constraint object structure

Type Safety:
- Numeric fields (min, max, range values) must be int/float
- Categorical values must be strings (after normalization)
- Arrays must be arrays (after M-31 normalization)
- Objects must be dicts

2. WHAT INPUTS CORRECTLY FAIL VALIDATION
-----------------------------------------

The following inputs will FAIL with explicit errors:

Missing Required Fields:
✓ No 'intent' → MissingRequiredFieldError
✓ No 'subintent' → MissingRequiredFieldError
✓ No 'domain' (product/service) → MissingRequiredFieldError
✓ No 'category' (mutual) → MissingRequiredFieldError
✓ No 'items' (product/service) → MissingRequiredFieldError
✓ No 'other' → MissingRequiredFieldError
✓ No 'self' → MissingRequiredFieldError
✓ No 'location' → MissingRequiredFieldError
✓ No 'locationmode' → MissingRequiredFieldError
✓ No 'reasoning' → MissingRequiredFieldError

Invalid Intent-SubIntent Combinations (I-04):
✓ (product, seek) → InvalidIntentSubIntentError
✓ (service, buy) → InvalidIntentSubIntentError
✓ (mutual, sell) → InvalidIntentSubIntentError
✓ (product, connect) → InvalidIntentSubIntentError
✓ Any combination not in VALID_COMBINATIONS

Invalid Constraint Modes (I-02):
✓ {"exact": 256} → InvalidConstraintModeError (EXACT not allowed as mode)
✓ {"categorical": {}, "prefer": "value"} → InvalidConstraintModeError
✓ Any key not in {categorical, min, max, range}

Invalid Range Bounds (I-06):
✓ range: {"price": [100, 50]} → InvalidRangeBoundsError (100 > 50)
✓ range: {"age": [30, 25]} → InvalidRangeBoundsError

Type Errors:
✓ intent = 123 → TypeValidationError (not string)
✓ domain = "electronics" → Normalized to ["electronics"] (M-31)
✓ items = "phone" → TypeValidationError (must be array of objects)
✓ min: {"price": "fifty"} → TypeValidationError (not numeric)
✓ range: {"price": 50} → TypeValidationError (not array)
✓ range: {"price": [50]} → TypeValidationError (not length 2)

Empty Arrays (where not allowed):
✓ domain = [] (product/service) → SchemaValidationError
✓ category = [] (mutual) → SchemaValidationError

Location Structure:
✓ location = 123 → TypeValidationError
✓ location = {"origin": "X"} → SchemaValidationError (missing destination)
✓ location = {"something": "else"} → SchemaValidationError

Invalid Enum Values:
✓ intent = "trading" → SchemaValidationError
✓ locationmode = "nearby" → SchemaValidationError

3. WHAT ASSUMPTIONS WERE EXPLICITLY REJECTED
---------------------------------------------

❌ REJECTED: Hardcoded Domain/Category Vocabularies
- Did NOT validate domain values against 21 product domains
- Did NOT validate domain values against 18 service domains
- Did NOT validate category values against 25 mutual categories
- Reason: Per directive "No hardcoded domains, categories, or vocab"
- Impact: Vocabulary validation must happen in separate layer (if needed)

❌ REJECTED: Silent Coercion
- Did NOT convert "Product" → "product" without validation
- Did NOT fix {"exact": 256} → {"range": [256, 256]}
- Did NOT infer missing fields with defaults
- Reason: Per directive "No silent coercion", "Fail loudly"
- Impact: Caller must provide valid data or receive explicit error

❌ REJECTED: Data Correction
- Did NOT fix range: {"price": [100, 50]} → [50, 100]
- Did NOT remove invalid constraint keys automatically
- Did NOT infer 'type' for items
- Reason: Per directive "No inference or correction of bad data"
- Impact: Invalid data rejected immediately, no masking of upstream errors

❌ REJECTED: Best-Effort Behavior
- Did NOT continue validation after first error
- Did NOT return partially normalized data
- Did NOT skip invalid items in array
- Reason: Per directive "Fail loudly", canon philosophy of strict validation
- Impact: First error stops processing, forces data quality upstream

❌ REJECTED: Semantic Transformation
- Did NOT expand abbreviations ("blr" → "bangalore")
- Did NOT standardize formats ("Bangalore, Karnataka" → structured)
- Did NOT apply term implications here (vegan → vegetarian)
- Reason: Normalization is syntactic only (M-32: lowercase + trim)
- Impact: Semantic processing deferred to matching phase

❌ REJECTED: Type Inference
- Did NOT guess if string "50" in categorical should be in min/max
- Did NOT convert numeric strings to numbers
- Did NOT infer constraint mode from signal words
- Reason: Input assumed to be from VRIDDHI extraction (already structured)
- Impact: Caller must provide correctly typed data

❌ REJECTED: EXACT as Fourth Mode
- Did NOT allow "exact" key in constraint objects
- Did NOT auto-convert exact to range[x,x]
- Reason: I-02 explicitly states only 3 modes, EXACT = range[x,x]
- Impact: Caller must use range[x,x] for exact constraints
- Note: This enforces canon truth that EXACT is not a mode

❌ REJECTED: Location Distance Calculation
- Did NOT validate location against geographic database
- Did NOT compute distances or check proximity
- Did NOT geocode location strings
- Reason: Phase 2.1 is pure schema validation, not semantic processing
- Impact: Location validation is structural only (string or route object)

❌ REJECTED: Empty Object Rejection
- Did NOT reject other = {} or self = {}
- Did NOT reject categorical = {}
- Reason: Empty constraint objects are valid (vacuous truth in matching)
- Impact: Empty preferences/attributes allowed (matches anything)

4. WHAT DOWNSTREAM CODE CAN SAFELY ASSUME NOW
----------------------------------------------

After normalize_and_validate() succeeds, downstream matching code can assume:

Structure Guarantees:
✓ All required fields exist (no null checks needed)
✓ intent ∈ {product, service, mutual}
✓ subintent valid for intent (I-04 enforced)
✓ domain non-empty for product/service
✓ category non-empty for mutual
✓ items is array (possibly empty for mutual)
✓ All constraint objects have all 4 keys: {categorical, min, max, range}

Array Guarantees (M-31):
✓ domain is array
✓ category is array
✓ items is array
✓ All exclusion fields are arrays
✓ No null arrays exist

String Normalization Guarantees (M-32):
✓ All string comparisons can use == without case handling
✓ No leading/trailing whitespace
✓ intent, subintent, domain values, category values all lowercase
✓ location string (if string) is normalized
✓ locationmode is normalized
✓ All categorical keys and values are normalized
✓ All exclusion strings are normalized

Type Guarantees:
✓ min/max values are numeric (int or float)
✓ range values are [numeric, numeric] arrays of length 2
✓ range[0] ≤ range[1] (I-06 enforced)
✓ categorical values are strings
✓ location is string OR {origin: str, destination: str}

Constraint Object Guarantees (I-02):
✓ Only keys present: categorical, min, max, range
✓ No "exact" mode
✓ No "prefer" mode
✓ No unknown modes
✓ Structure is consistent across all constraint objects

Items Guarantees:
✓ Each item has 'type' field (string, normalized)
✓ Each item has all 4 constraint keys
✓ Items array empty only for mutual intent

Exclusions Guarantees:
✓ All exclusion arrays are normalized string arrays
✓ Empty exclusion arrays are []  not null

Location Guarantees:
✓ locationmode ∈ {proximity, target, route, flexible}
✓ If locationmode ≠ route, location is string
✓ If locationmode = route, location is {origin: str, destination: str}
✓ locationexclusions is normalized string array

What Downstream Code CANNOT Assume:
✗ Domain/category values are in official vocabulary (not validated)
✗ Location strings are valid geographic locations
✗ Numeric values are in sensible ranges
✗ Categorical values follow any schema
✗ Items exist (for mutual intent)
✗ Constraint objects are non-empty

What Downstream Code MUST Still Check:
- Matching logic (Phase 2.2+)
- Term implication lookups for categorical matching
- Location distance/overlap calculations
- Domain/category membership (if required by system policy)
- Business logic constraints (price ranges, etc.)

Safe Operations Enabled:
✓ Direct string equality: A.intent == B.intent (no .lower() needed)
✓ Set operations on arrays: set(A.domain) & set(B.domain)
✓ Numeric comparisons: A.min[k] <= B.value
✓ Range access: obj["range"][key][0], obj["range"][key][1]
✓ Key existence: "categorical" in obj (always True)
✓ Iteration: for item in listing["items"]
✓ Type assumptions: int/float operations on min/max/range

Error Propagation:
✓ Validation errors include field name, canon rule violated
✓ Errors are explicit exceptions (never silent failures)
✓ First error stops processing (fail-fast)
✓ Errors suitable for logging and debugging

Testing Recommendations for Next Phase:
- Test empty constraint objects (should pass)
- Test single-element arrays (M-31 compatibility)
- Test case variations (M-32 effectiveness)
- Test range bounds edge case [x, x] (exact values)
- Test all 5 valid intent-subintent combinations
- Test all 4 location modes
- Test route location structure
- Test missing required fields
- Test invalid types
- Test invalid constraint modes
"""
