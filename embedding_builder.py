"""
PHASE 3.3: EMBEDDING TEXT CONSTRUCTION

Responsibilities:
- Construct embedding text from normalized listing objects
- Product/Service: Structured attribute concatenation
- Mutual: Semantic query meaning

Authority: VRIDDHI Architecture Document
Dependencies: None (pure Python)

Author: Claude (Implementation Engine)
Date: 2026-01-12
"""

from typing import Dict, Any, List


# ============================================================================
# EMBEDDING TEXT CONSTRUCTION
# ============================================================================

def build_embedding_text_product_service(listing: Dict[str, Any]) -> str:
    """
    Build embedding text for product or service listings.

    Strategy: Concatenate structured attributes in order:
    - intent
    - subintent
    - domain (array)
    - items (type + categorical attributes)

    NO hard-coded attributes.
    NO inferred values.
    ONLY what exists in listing.

    Args:
        listing: Normalized listing object

    Returns:
        Space-separated text string for embedding
    """
    parts = []

    # 1. Intent
    if "intent" in listing and listing["intent"]:
        parts.append(listing["intent"])

    # 2. Subintent
    if "subintent" in listing and listing["subintent"]:
        parts.append(listing["subintent"])

    # 3. Domain (array)
    if "domain" in listing and listing["domain"]:
        for domain_val in listing["domain"]:
            if domain_val:
                parts.append(str(domain_val))

    # 4. Items (type + categorical attributes)
    if "items" in listing and listing["items"]:
        for item in listing["items"]:
            # Item type
            if "type" in item and item["type"]:
                parts.append(str(item["type"]))

            # Categorical attributes (dynamic)
            if "categorical" in item and item["categorical"]:
                for attr_key, attr_value in item["categorical"].items():
                    if attr_value:
                        # Add both key and value for context
                        parts.append(str(attr_key))
                        parts.append(str(attr_value))

            # Min constraints (attribute names only, for searchability)
            if "min" in item and item["min"]:
                for attr_key in item["min"].keys():
                    parts.append(str(attr_key))

            # Max constraints (attribute names only)
            if "max" in item and item["max"]:
                for attr_key in item["max"].keys():
                    parts.append(str(attr_key))

            # Range constraints (attribute names only)
            if "range" in item and item["range"]:
                for attr_key in item["range"].keys():
                    parts.append(str(attr_key))

    # Join with spaces
    return " ".join(parts)


def build_embedding_text_mutual(listing: Dict[str, Any]) -> str:
    """
    Build embedding text for mutual exchange listings.

    Strategy: Semantic meaning, NOT keyword bias.
    - Construct natural language description of exchange
    - Include what user offers (items)
    - Include what user wants (other)
    - Include self attributes (context)
    - NO slicing by item type

    Args:
        listing: Normalized listing object

    Returns:
        Natural language text for semantic embedding
    """
    parts = []

    # 1. Intent context
    parts.append("mutual exchange")

    # 2. Category (what kind of exchange)
    if "category" in listing and listing["category"]:
        category_text = " and ".join(listing["category"])
        parts.append(f"in categories: {category_text}")

    # 3. What user offers (items)
    if "items" in listing and listing["items"]:
        offer_parts = []
        for item in listing["items"]:
            item_desc = []

            # Type
            if "type" in item and item["type"]:
                item_desc.append(str(item["type"]))

            # Categorical attributes
            if "categorical" in item and item["categorical"]:
                for attr_key, attr_value in item["categorical"].items():
                    if attr_value:
                        item_desc.append(f"{attr_key}: {attr_value}")

            if item_desc:
                offer_parts.append(" ".join(item_desc))

        if offer_parts:
            parts.append("offering " + ", ".join(offer_parts))

    # 4. What user wants (other)
    if "other" in listing:
        other = listing["other"]
        want_parts = []

        # Categorical requirements
        if "categorical" in other and other["categorical"]:
            for attr_key, attr_value in other["categorical"].items():
                if attr_value:
                    want_parts.append(f"{attr_key}: {attr_value}")

        # Min requirements
        if "min" in other and other["min"]:
            for attr_key, attr_value in other["min"].items():
                want_parts.append(f"{attr_key} at least {attr_value}")

        # Max requirements
        if "max" in other and other["max"]:
            for attr_key, attr_value in other["max"].items():
                want_parts.append(f"{attr_key} at most {attr_value}")

        # Range requirements
        if "range" in other and other["range"]:
            for attr_key, attr_value in other["range"].items():
                if isinstance(attr_value, list) and len(attr_value) == 2:
                    want_parts.append(f"{attr_key} between {attr_value[0]} and {attr_value[1]}")

        if want_parts:
            parts.append("wanting " + ", ".join(want_parts))

    # 5. Self attributes (additional context)
    if "self" in listing:
        self_obj = listing["self"]
        self_parts = []

        # Categorical self-description
        if "categorical" in self_obj and self_obj["categorical"]:
            for attr_key, attr_value in self_obj["categorical"].items():
                if attr_value:
                    self_parts.append(f"{attr_key}: {attr_value}")

        if self_parts:
            parts.append("with attributes " + ", ".join(self_parts))

    # Join into natural language
    return " ".join(parts)


def build_embedding_text(listing: Dict[str, Any]) -> str:
    """
    Build embedding text for any listing type.

    Routes to appropriate builder based on intent.

    Args:
        listing: Normalized listing object

    Returns:
        Text string for embedding generation

    Raises:
        ValueError: If intent field missing or unknown
    """
    if "intent" not in listing:
        raise ValueError("Listing missing 'intent' field")

    intent = listing["intent"]

    if intent == "product" or intent == "service":
        return build_embedding_text_product_service(listing)
    elif intent == "mutual":
        return build_embedding_text_mutual(listing)
    else:
        raise ValueError(f"Unknown intent: {intent}")


# ============================================================================
# UTILITIES
# ============================================================================

def preview_embedding_text(listing: Dict[str, Any]) -> None:
    """
    Print embedding text for a listing (debugging/verification).

    Args:
        listing: Normalized listing object
    """
    text = build_embedding_text(listing)
    print(f"Intent: {listing.get('intent')}")
    print(f"Embedding text ({len(text)} chars):")
    print(f"  {text}")
    print()
