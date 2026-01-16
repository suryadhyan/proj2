"""
PHASE 2.0: SCHEMA NORMALIZATION (V2 - NEW SCHEMA)

Responsibility:
- Validate NEW schema structure (14 fields, axis-based constraints)
- Transform NEW schema → OLD schema format for existing matching engine
- Normalize strings, arrays, and constraint objects

NEW Schema (Input):
- 14 fields (vs 12 in old schema)
- Axis-based constraints: {axis: [{type, value, unit}]}
- Field names: other_party_preferences, self_attributes, target_location, etc.

OLD Schema (Output after transformation):
- 12 fields
- Flat constraints: {type: value}
- Field names: other, self, location, etc.

Authority: GLOBAL_REFERENCE_CONTEXT.md
Dependencies: None (pure validation + transformation)

Author: Claude (Schema Normalizer V2)
Date: 2026-01-13
"""

from typing import Dict, List, Any, Union, Optional
import copy


# ============================================================================
# CONSTANTS
# ============================================================================

# 14 fields in NEW schema
NEW_SCHEMA_FIELDS = {
    "intent", "subintent", "domain", "primary_mutual_category",
    "items", "item_exclusions",
    "other_party_preferences", "other_party_exclusions",
    "self_attributes", "self_exclusions",
    "target_location", "location_match_mode", "location_exclusions",
    "reasoning"
}

# 10 FIXED AXES (never changes)
VALID_AXES = {
    "identity", "capacity", "performance", "quality", "quantity",
    "time", "space", "cost", "mode", "skill"
}

# Valid intent values
VALID_INTENTS = {"product", "service", "mutual"}

# Valid subintent mappings
VALID_SUBINTENT_PAIRS = {
    ("product", "buy"), ("product", "sell"),
    ("service", "seek"), ("service", "provide"),
    ("mutual", "connect")
}

# Valid location modes
VALID_LOCATION_MODES = {"near_me", "explicit", "target_only", "route", "global"}

# Field name mapping: NEW → OLD
FIELD_NAME_MAPPING = {
    # Keep same
    "intent": "intent",
    "subintent": "subintent",
    "domain": "domain",
    "items": "items",
    "reasoning": "reasoning",

    # Rename
    "primary_mutual_category": "category",
    "item_exclusions": "itemexclusions",
    "other_party_preferences": "other",
    "other_party_exclusions": "otherexclusions",
    "self_attributes": "self",
    "self_exclusions": "selfexclusions",
    "target_location": "location",
    "location_match_mode": "locationmode",
    "location_exclusions": "locationexclusions"
}


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class SchemaValidationError(ValueError):
    """Raised when schema validation fails."""
    pass


class InvalidAxisError(SchemaValidationError):
    """Raised when invalid axis is used."""
    pass


class InvalidConstraintStructureError(SchemaValidationError):
    """Raised when constraint structure is invalid."""
    pass


class MissingFieldError(SchemaValidationError):
    """Raised when required field is missing."""
    pass


# ============================================================================
# STRING & ARRAY NORMALIZATION (Reused from old normalizer)
# ============================================================================

def normalize_string(value: Any) -> str:
    """
    Normalize a string: lowercase + trim.

    M-32: String Case Insensitivity Rule
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value).lower().strip()
    return value.lower().strip()


def normalize_array(value: Any) -> List:
    """
    Normalize to array form.

    M-31: Array Normalization Rule
    - null → []
    - scalar → [scalar]
    - array → array
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_string_array(value: Any) -> List[str]:
    """
    Normalize to string array: apply M-31 + M-32.
    """
    arr = normalize_array(value)
    return [normalize_string(item) for item in arr]


# ============================================================================
# AXIS CONSTRAINT FLATTENING (CRITICAL TRANSFORMATION)
# ============================================================================

def flatten_axis_constraints(axis_dict: Dict[str, List[Dict]]) -> Dict[str, Union[float, int]]:
    """
    Flatten axis-based constraints to flat dict.

    NEW format:
    {
        "capacity": [
            {"type": "storage", "value": 256, "unit": "gb"},
            {"type": "memory", "value": 16, "unit": "gb"}
        ]
    }

    OLD format:
    {
        "storage": 256,
        "memory": 16
    }

    Args:
        axis_dict: Axis-based constraint dictionary

    Returns:
        Flattened dictionary {type: value}
    """
    if not axis_dict:
        return {}

    flat = {}
    for axis, constraint_list in axis_dict.items():
        if not isinstance(constraint_list, list):
            raise InvalidConstraintStructureError(
                f"Axis '{axis}' must have list of constraints, got {type(constraint_list)}"
            )

        for constraint_obj in constraint_list:
            if not isinstance(constraint_obj, dict):
                raise InvalidConstraintStructureError(
                    f"Constraint in axis '{axis}' must be dict, got {type(constraint_obj)}"
                )

            # Extract type and value (ignore unit for now)
            constraint_type = constraint_obj.get("type")
            constraint_value = constraint_obj.get("value")

            if not constraint_type:
                raise InvalidConstraintStructureError(
                    f"Constraint in axis '{axis}' missing 'type' field"
                )
            if constraint_value is None:
                raise InvalidConstraintStructureError(
                    f"Constraint '{constraint_type}' in axis '{axis}' missing 'value' field"
                )

            flat[constraint_type] = constraint_value

    return flat


def flatten_axis_ranges(axis_dict: Dict[str, List[Dict]]) -> Dict[str, List[Union[float, int]]]:
    """
    Flatten axis-based ranges to flat dict of [min, max] tuples.

    NEW format:
    {
        "capacity": [
            {"type": "storage", "min": 256, "max": 512, "unit": "gb"}
        ]
    }

    OLD format:
    {
        "storage": [256, 512]
    }

    Args:
        axis_dict: Axis-based range dictionary

    Returns:
        Flattened dictionary {type: [min, max]}
    """
    if not axis_dict:
        return {}

    flat = {}
    for axis, constraint_list in axis_dict.items():
        if not isinstance(constraint_list, list):
            raise InvalidConstraintStructureError(
                f"Axis '{axis}' must have list of constraints, got {type(constraint_list)}"
            )

        for constraint_obj in constraint_list:
            if not isinstance(constraint_obj, dict):
                raise InvalidConstraintStructureError(
                    f"Constraint in axis '{axis}' must be dict, got {type(constraint_obj)}"
                )

            # Extract type, min, max (ignore unit)
            constraint_type = constraint_obj.get("type")
            min_val = constraint_obj.get("min")
            max_val = constraint_obj.get("max")

            if not constraint_type:
                raise InvalidConstraintStructureError(
                    f"Constraint in axis '{axis}' missing 'type' field"
                )
            if min_val is None or max_val is None:
                raise InvalidConstraintStructureError(
                    f"Range constraint '{constraint_type}' in axis '{axis}' must have both 'min' and 'max'"
                )

            # Validate range bounds (I-06 from old schema)
            if min_val > max_val:
                raise InvalidConstraintStructureError(
                    f"Range constraint '{constraint_type}' has invalid bounds: min={min_val} > max={max_val}"
                )

            flat[constraint_type] = [min_val, max_val]

    return flat


def flatten_identity_lifestyle_to_categorical(identity_list: List[Dict], lifestyle_list: List[Dict]) -> Dict[str, str]:
    """
    Flatten identity and lifestyle arrays to categorical key-value pairs.

    NEW format (from prompt):
    {
        "identity": [
            {"type": "skill", "value": "cooking"},
            {"type": "gender", "value": "female"}
        ],
        "lifestyle": [
            {"type": "diet", "value": "vegetarian"}
        ]
    }

    OLD format (for matching):
    {
        "skill": "cooking",
        "gender": "female",
        "diet": "vegetarian"
    }

    Args:
        identity_list: List of identity objects with type and value
        lifestyle_list: List of lifestyle objects with type and value

    Returns:
        Flat dict mapping type to value
    """
    categorical = {}

    # Flatten identity array
    if identity_list and isinstance(identity_list, list):
        for item in identity_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                item_value = item.get("value", "")
                if item_type and item_value:
                    categorical[normalize_string(item_type)] = normalize_string(item_value)

    # Flatten lifestyle array
    if lifestyle_list and isinstance(lifestyle_list, list):
        for item in lifestyle_list:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                item_value = item.get("value", "")
                if item_type and item_value:
                    categorical[normalize_string(item_type)] = normalize_string(item_value)

    return categorical


def flatten_habits_to_categorical(habits: Dict[str, str]) -> Dict[str, str]:
    """
    Flatten habits dict to categorical key-value pairs.

    NEW format (from prompt):
    {
        "habits": {
            "smoking": "no",
            "drinking": "no"
        }
    }

    OLD format (for matching):
    {
        "smoking": "no",
        "drinking": "no"
    }

    Habits are already in the right format, just need to normalize and validate.

    Args:
        habits: Dict of habit flags (yes/no values)

    Returns:
        Normalized habits dict
    """
    categorical = {}

    if habits and isinstance(habits, dict):
        for key, value in habits.items():
            # Normalize key and value
            norm_key = normalize_string(key)
            norm_value = normalize_string(value)

            # Validate that value is yes/no (optional - be permissive)
            if norm_value in ["yes", "no"]:
                categorical[norm_key] = norm_value
            else:
                # If not yes/no, still include it (be permissive for other habit values)
                categorical[norm_key] = norm_value

    return categorical


def transform_constraint_object(new_constraints: Dict) -> Dict:
    """
    Transform NEW constraint object to OLD format.

    NEW (from GLOBAL_REFERENCE_CONTEXT.md prompt):
    {
        "identity": [{"type": "skill", "value": "cooking"}],
        "lifestyle": [{"type": "diet", "value": "vegetarian"}],
        "habits": {"smoking": "no", "drinking": "no"},
        "min": {"time": [{"type": "experience", "value": 36, "unit": "month"}]},
        "max": {"cost": [{"type": "price", "value": 50000, "unit": "inr"}]},
        "range": {}
    }

    OLD (for matching engine):
    {
        "categorical": {"skill": "cooking", "diet": "vegetarian", "smoking": "no", "drinking": "no"},
        "min": {"experience": 36},
        "max": {"price": 50000},
        "range": {}
    }
    """
    if not isinstance(new_constraints, dict):
        return {
            "categorical": {},
            "min": {},
            "max": {},
            "range": {}
        }

    # Build categorical from multiple sources
    categorical = {}

    # 1. Extract from identity array
    identity_list = new_constraints.get("identity", [])
    lifestyle_list = new_constraints.get("lifestyle", [])
    identity_lifestyle_categorical = flatten_identity_lifestyle_to_categorical(identity_list, lifestyle_list)
    categorical.update(identity_lifestyle_categorical)

    # 2. Extract from habits dict
    habits_dict = new_constraints.get("habits", {})
    habits_categorical = flatten_habits_to_categorical(habits_dict)
    categorical.update(habits_categorical)

    # 3. Add any existing categorical field (for backward compatibility)
    existing_categorical = new_constraints.get("categorical", {})
    if isinstance(existing_categorical, dict):
        for key, value in existing_categorical.items():
            categorical[normalize_string(key)] = normalize_string(value)

    # Flatten axis-based constraints (min/max/range)
    min_axis = new_constraints.get("min", {})
    max_axis = new_constraints.get("max", {})
    range_axis = new_constraints.get("range", {})

    # Handle NEW structure (axis-based) vs OLD structure (already flat)
    # If min/max are already flat dicts, keep them. If axis-based, flatten.
    if min_axis and any(isinstance(v, list) for v in min_axis.values()):
        min_flat = flatten_axis_constraints(min_axis)
    else:
        min_flat = min_axis if isinstance(min_axis, dict) else {}

    if max_axis and any(isinstance(v, list) for v in max_axis.values()):
        max_flat = flatten_axis_constraints(max_axis)
    else:
        max_flat = max_axis if isinstance(max_axis, dict) else {}

    if range_axis and any(isinstance(v, list) for v in range_axis.values()):
        range_flat = flatten_axis_ranges(range_axis)
    else:
        range_flat = range_axis if isinstance(range_axis, dict) else {}

    return {
        "categorical": categorical,
        "min": min_flat,
        "max": max_flat,
        "range": range_flat
    }


# ============================================================================
# SCHEMA VALIDATION (NEW SCHEMA)
# ============================================================================

def validate_new_schema_fields(listing: Dict) -> None:
    """
    Validate that all 14 NEW schema fields are present.

    Raises:
        MissingFieldError: If required field is missing
    """
    missing = NEW_SCHEMA_FIELDS - set(listing.keys())
    if missing:
        raise MissingFieldError(f"Missing required fields: {missing}")


def validate_intent_subintent(intent: str, subintent: str) -> None:
    """
    Validate intent-subintent pair (I-04 invariant).

    Raises:
        SchemaValidationError: If pair is invalid
    """
    intent = normalize_string(intent)
    subintent = normalize_string(subintent)

    if intent not in VALID_INTENTS:
        raise SchemaValidationError(f"Invalid intent: {intent}")

    if (intent, subintent) not in VALID_SUBINTENT_PAIRS:
        raise SchemaValidationError(
            f"Invalid intent-subintent pair: ({intent}, {subintent})"
        )


def validate_axes(constraint_obj: Dict) -> None:
    """
    Validate that only valid axes are used in constraint objects.

    Args:
        constraint_obj: Constraint object with min/max/range

    Raises:
        InvalidAxisError: If invalid axis is used
    """
    for mode in ["min", "max", "range"]:
        if mode in constraint_obj:
            mode_dict = constraint_obj[mode]
            if isinstance(mode_dict, dict):
                invalid_axes = set(mode_dict.keys()) - VALID_AXES
                if invalid_axes:
                    raise InvalidAxisError(
                        f"Invalid axes in '{mode}': {invalid_axes}. "
                        f"Valid axes: {VALID_AXES}"
                    )


def validate_location_mode(mode: str) -> None:
    """
    Validate location_match_mode value.

    Raises:
        SchemaValidationError: If mode is invalid
    """
    mode = normalize_string(mode)
    if mode not in VALID_LOCATION_MODES:
        raise SchemaValidationError(
            f"Invalid location_match_mode: {mode}. "
            f"Valid modes: {VALID_LOCATION_MODES}"
        )


def validate_new_schema(listing: Dict) -> None:
    """
    Validate NEW schema structure.

    Checks:
    - All 14 fields present
    - Intent-subintent pair valid
    - Only valid axes used
    - Location mode valid

    Raises:
        SchemaValidationError: If validation fails
    """
    # Check all fields present
    validate_new_schema_fields(listing)

    # Validate intent-subintent
    validate_intent_subintent(
        listing.get("intent", ""),
        listing.get("subintent", "")
    )

    # Validate axes in items
    items = listing.get("items", [])
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                validate_axes(item)

    # Validate axes in other_party_preferences
    other_prefs = listing.get("other_party_preferences", {})
    if isinstance(other_prefs, dict):
        validate_axes(other_prefs)

    # Validate axes in self_attributes
    self_attrs = listing.get("self_attributes", {})
    if isinstance(self_attrs, dict):
        validate_axes(self_attrs)

    # Validate location mode
    location_mode = listing.get("location_match_mode", "")
    if location_mode:
        validate_location_mode(location_mode)


# ============================================================================
# TRANSFORMATION: NEW → OLD SCHEMA
# ============================================================================

def transform_location(target_location: Dict, location_mode: str) -> Union[str, Dict]:
    """
    Transform NEW location format to OLD format.

    NEW:
        target_location: {"name": "bangalore"}
        location_mode: "explicit"

    OLD:
        location: "bangalore"  (simple string)
        OR
        location: {"origin": "delhi", "destination": "mumbai"}  (for route mode)

    Args:
        target_location: NEW format location object
        location_mode: Location match mode

    Returns:
        OLD format location (string or dict)
    """
    if not target_location or not isinstance(target_location, dict):
        return ""

    # Handle route mode (has origin + destination)
    if "origin" in target_location and "destination" in target_location:
        return {
            "origin": normalize_string(target_location.get("origin", "")),
            "destination": normalize_string(target_location.get("destination", ""))
        }

    # Handle simple name-based location
    name = target_location.get("name", "")
    return normalize_string(name) if name else ""


def transform_items(new_items: List[Dict]) -> List[Dict]:
    """
    Transform items array from NEW to OLD format.

    Flattens axis-based constraints within each item.

    Args:
        new_items: List of items in NEW format

    Returns:
        List of items in OLD format
    """
    if not isinstance(new_items, list):
        return []

    old_items = []
    for item in new_items:
        if not isinstance(item, dict):
            continue

        # Start with item type and categorical (unchanged)
        old_item = {
            "type": normalize_string(item.get("type", "")),
            "categorical": item.get("categorical", {})
        }

        # Transform constraints (flatten axes)
        constraints = transform_constraint_object({
            "min": item.get("min", {}),
            "max": item.get("max", {}),
            "range": item.get("range", {})
        })

        old_item["min"] = constraints["min"]
        old_item["max"] = constraints["max"]
        old_item["range"] = constraints["range"]

        old_items.append(old_item)

    return old_items


def transform_new_to_old(listing: Dict) -> Dict:
    """
    Transform NEW schema to OLD schema format.

    Performs:
    1. Field renaming (12 renames)
    2. Axis constraint flattening
    3. Location simplification
    4. String/array normalization

    Args:
        listing: Listing in NEW schema format

    Returns:
        Listing in OLD schema format
    """
    # Create output with OLD field names
    old_listing = {}

    # Map simple fields (intent, subintent, reasoning)
    old_listing["intent"] = normalize_string(listing.get("intent", ""))
    old_listing["subintent"] = normalize_string(listing.get("subintent", ""))
    old_listing["reasoning"] = listing.get("reasoning", "")  # Keep as-is

    # Map domain/category (note: primary_mutual_category → category)
    old_listing["domain"] = normalize_string_array(listing.get("domain", []))
    old_listing["category"] = normalize_string_array(listing.get("primary_mutual_category", []))

    # Transform items (flatten axis constraints)
    old_listing["items"] = transform_items(listing.get("items", []))

    # Map top-level exclusions
    old_listing["itemexclusions"] = normalize_string_array(listing.get("item_exclusions", []))
    old_listing["locationexclusions"] = normalize_string_array(listing.get("location_exclusions", []))

    # Transform other_party_preferences → other (flatten axes)
    other_prefs = listing.get("other_party_preferences", {})
    old_listing["other"] = transform_constraint_object(other_prefs)
    # Add otherexclusions INSIDE the other object
    old_listing["other"]["otherexclusions"] = normalize_string_array(listing.get("other_party_exclusions", []))

    # Transform self_attributes → self (flatten axes)
    self_attrs = listing.get("self_attributes", {})
    old_listing["self"] = transform_constraint_object(self_attrs)
    # Add selfexclusions INSIDE the self object
    old_listing["self"]["selfexclusions"] = normalize_string_array(listing.get("self_exclusions", []))

    # Transform location (target_location → location)
    target_location = listing.get("target_location", {})
    location_mode = listing.get("location_match_mode", "near_me")
    old_listing["location"] = transform_location(target_location, location_mode)
    old_listing["locationmode"] = normalize_string(location_mode)

    return old_listing


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def normalize_and_validate_v2(listing: Dict) -> Dict:
    """
    Main entry point: Validate NEW schema + transform to OLD format.

    Pipeline:
    1. Validate NEW schema structure (14 fields, axes, etc.)
    2. Transform to OLD schema format (12 fields, flat constraints)
    3. Return OLD format for matching engine

    Args:
        listing: Listing in NEW schema format

    Returns:
        Listing in OLD schema format (ready for matching)

    Raises:
        SchemaValidationError: If validation fails
    """
    # Make a deep copy to avoid mutating input
    listing = copy.deepcopy(listing)

    # Step 1: Validate NEW schema
    validate_new_schema(listing)

    # Step 2: Transform NEW → OLD
    old_listing = transform_new_to_old(listing)

    # Step 3: Return OLD format
    return old_listing
