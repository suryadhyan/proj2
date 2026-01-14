"""
PHASE 2.X: LOCATION MATCHING (V2 - SIMPLIFIED)

Responsibility:
- Match locations using simple name-based equality
- Handle location exclusions
- Support all 5 location modes (near_me, explicit, target_only, route, global)

NEW Schema Location Structure:
- target_location: {"name": "bangalore"} OR {"origin": "X", "destination": "Y"}
- location_match_mode: "near_me" | "explicit" | "target_only" | "route" | "global"
- location_exclusions: ["whitefield", "airport"]

Simplified Logic:
- No distance/zone/accessibility constraints
- Simple name equality checks
- Exclusion list matching

Authority: GLOBAL_REFERENCE_CONTEXT.md (NEW schema)
Dependencies: None

Author: Claude (Location Matcher V2)
Date: 2026-01-13
"""

from typing import Dict, List, Union, Optional


# ============================================================================
# LOCATION MATCHING (SIMPLIFIED)
# ============================================================================

def match_location_simple(
    required_location: Union[str, Dict],
    candidate_location: Union[str, Dict],
    required_exclusions: List[str],
    candidate_exclusions: List[str]
) -> bool:
    """
    Simple name-based location matching.

    Matching Rules:
    1. If required_location empty → always match (no location requirement)
    2. Name equality: required name == candidate name
    3. Exclusions: candidate not in required exclusions, required not in candidate exclusions

    Args:
        required_location: Required location (string or dict with name/origin/destination)
        candidate_location: Candidate location (string or dict)
        required_exclusions: Locations that requester excludes
        candidate_exclusions: Locations that candidate excludes

    Returns:
        True if location matches, False otherwise

    Examples:
        >>> match_location_simple("bangalore", "bangalore", [], [])
        True

        >>> match_location_simple("bangalore", "delhi", [], [])
        False

        >>> match_location_simple("bangalore", "bangalore", ["whitefield"], [])
        True  # bangalore not in exclusions

        >>> match_location_simple("whitefield", "whitefield", ["whitefield"], [])
        False  # whitefield is in required exclusions
    """
    # Normalize inputs
    required_name = _extract_location_name(required_location)
    candidate_name = _extract_location_name(candidate_location)

    # Rule 1: If no location requirement, always match
    if not required_name:
        return True

    # Rule 2: Name equality
    if required_name != candidate_name:
        return False

    # Rule 3: Check exclusions
    # Candidate location must not be in requester's exclusions
    if candidate_name in required_exclusions:
        return False

    # Requester's location must not be in candidate's exclusions
    if required_name in candidate_exclusions:
        return False

    return True


def match_location_route(
    required_route: Dict,
    candidate_route: Dict,
    required_exclusions: List[str],
    candidate_exclusions: List[str]
) -> bool:
    """
    Match route-based locations (origin → destination).

    For route mode, both origin and destination must match.

    Args:
        required_route: Required route {"origin": "X", "destination": "Y"}
        candidate_route: Candidate route {"origin": "A", "destination": "B"}
        required_exclusions: Location exclusions from requester
        candidate_exclusions: Location exclusions from candidate

    Returns:
        True if routes match, False otherwise

    Examples:
        >>> match_location_route(
        ...     {"origin": "bangalore", "destination": "goa"},
        ...     {"origin": "bangalore", "destination": "goa"},
        ...     [], []
        ... )
        True

        >>> match_location_route(
        ...     {"origin": "bangalore", "destination": "goa"},
        ...     {"origin": "delhi", "destination": "goa"},
        ...     [], []
        ... )
        False
    """
    # Extract origins and destinations
    req_origin = required_route.get("origin", "").lower().strip()
    req_dest = required_route.get("destination", "").lower().strip()
    cand_origin = candidate_route.get("origin", "").lower().strip()
    cand_dest = candidate_route.get("destination", "").lower().strip()

    # Both origin and destination must match
    if req_origin != cand_origin:
        return False
    if req_dest != cand_dest:
        return False

    # Check exclusions for both origin and destination
    for location in [cand_origin, cand_dest]:
        if location in required_exclusions:
            return False

    for location in [req_origin, req_dest]:
        if location in candidate_exclusions:
            return False

    return True


def match_location_v2(
    required_location: Union[str, Dict],
    required_mode: str,
    required_exclusions: List[str],
    candidate_location: Union[str, Dict],
    candidate_mode: str,
    candidate_exclusions: List[str]
) -> bool:
    """
    V2 location matching with mode support.

    Modes:
    - near_me: Match if no explicit location, or if locations match
    - explicit: Strict name matching required
    - target_only: Match any candidate (requester moving to target)
    - route: Both origin and destination must match
    - global: Always match (remote/anywhere)

    Args:
        required_location: Requester's location
        required_mode: Requester's location mode
        required_exclusions: Requester's location exclusions
        candidate_location: Candidate's location
        candidate_mode: Candidate's location mode
        candidate_exclusions: Candidate's location exclusions

    Returns:
        True if locations compatible, False otherwise
    """
    # Normalize modes
    required_mode = required_mode.lower().strip() if required_mode else "near_me"
    candidate_mode = candidate_mode.lower().strip() if candidate_mode else "near_me"

    # Mode: global (remote/anywhere)
    # Always matches regardless of location
    if required_mode == "global" or candidate_mode == "global":
        return True

    # Mode: route
    # Both must have route structure
    if required_mode == "route" or candidate_mode == "route":
        # Check if both have route structure
        req_is_route = isinstance(required_location, dict) and "origin" in required_location
        cand_is_route = isinstance(candidate_location, dict) and "origin" in candidate_location

        if req_is_route and cand_is_route:
            return match_location_route(
                required_location,
                candidate_location,
                required_exclusions,
                candidate_exclusions
            )
        else:
            # Route mode but missing route structure → no match
            return False

    # Mode: target_only
    # Requester is moving to target, will match any candidate at target
    if required_mode == "target_only":
        # Just check requester's target isn't in candidate's exclusions
        required_name = _extract_location_name(required_location)
        if required_name and required_name in candidate_exclusions:
            return False
        return True

    # Mode: explicit or near_me
    # Use simple name-based matching
    return match_location_simple(
        required_location,
        candidate_location,
        required_exclusions,
        candidate_exclusions
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_location_name(location: Union[str, Dict, None]) -> str:
    """
    Extract location name from location object or string.

    Args:
        location: Location (string or dict with "name" field)

    Returns:
        Normalized location name (lowercase, trimmed)

    Examples:
        >>> _extract_location_name("Bangalore")
        'bangalore'

        >>> _extract_location_name({"name": "Bangalore"})
        'bangalore'

        >>> _extract_location_name(None)
        ''

        >>> _extract_location_name({})
        ''
    """
    if not location:
        return ""

    if isinstance(location, str):
        return location.lower().strip()

    if isinstance(location, dict):
        name = location.get("name", "")
        return name.lower().strip() if name else ""

    return ""


def normalize_location_exclusions(exclusions: List[str]) -> List[str]:
    """
    Normalize location exclusions to lowercase, trimmed strings.

    Args:
        exclusions: List of location exclusions

    Returns:
        Normalized list

    Examples:
        >>> normalize_location_exclusions(["Whitefield", "  Airport  "])
        ['whitefield', 'airport']
    """
    if not exclusions:
        return []

    return [e.lower().strip() for e in exclusions if e]


# ============================================================================
# COMPATIBILITY WRAPPER (for existing code)
# ============================================================================

def match_location_constraints(
    required_location_obj: Dict,
    candidate_other_obj: Dict,
    location_exclusions: List[str] = None,
    other_location_exclusions: List[str] = None
) -> bool:
    """
    Compatibility wrapper for OLD schema location matching interface.

    Maps OLD interface to NEW v2 matching logic.

    OLD schema format (after transformation from NEW):
    {
        "location": "bangalore",  # string or dict
        "locationmode": "explicit",
        "locationexclusions": ["whitefield"]
    }

    Args:
        required_location_obj: Dict with "location", "locationmode", "locationexclusions"
        candidate_other_obj: Dict with "location", "locationmode", "locationexclusions"
        location_exclusions: Optional override for required exclusions
        other_location_exclusions: Optional override for candidate exclusions

    Returns:
        True if locations match, False otherwise
    """
    # Extract from dicts
    required_location = required_location_obj.get("location", "")
    required_mode = required_location_obj.get("locationmode", "near_me")
    required_exclusions = location_exclusions or required_location_obj.get("locationexclusions", [])

    candidate_location = candidate_other_obj.get("location", "")
    candidate_mode = candidate_other_obj.get("locationmode", "near_me")
    candidate_exclusions = other_location_exclusions or candidate_other_obj.get("locationexclusions", [])

    # Normalize exclusions
    required_exclusions = normalize_location_exclusions(required_exclusions)
    candidate_exclusions = normalize_location_exclusions(candidate_exclusions)

    # Call v2 matching
    return match_location_v2(
        required_location,
        required_mode,
        required_exclusions,
        candidate_location,
        candidate_mode,
        candidate_exclusions
    )
