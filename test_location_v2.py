"""Test location_matcher_v2.py"""

from location_matcher_v2 import match_location_simple, match_location_v2, match_location_route

print("="*80)
print("TESTING LOCATION MATCHER V2")
print("="*80)

# Test 1: Simple name matching
print("\n[Test 1] Simple name matching")
result = match_location_simple("bangalore", "bangalore", [], [])
assert result == True, "Same location should match"
print("✅ bangalore == bangalore → True")

result = match_location_simple("bangalore", "delhi", [], [])
assert result == False, "Different locations should not match"
print("✅ bangalore != delhi → False")

# Test 2: Empty location (no requirement)
print("\n[Test 2] Empty location (no requirement)")
result = match_location_simple("", "bangalore", [], [])
assert result == True, "Empty required location should match anything"
print("✅ empty == bangalore → True (no requirement)")

# Test 3: Exclusions
print("\n[Test 3] Location exclusions")
result = match_location_simple("whitefield", "whitefield", ["whitefield"], [])
assert result == False, "Excluded location should not match"
print("✅ whitefield with exclusion ['whitefield'] → False")

result = match_location_simple("bangalore", "bangalore", ["whitefield"], [])
assert result == True, "Non-excluded location should match"
print("✅ bangalore with exclusion ['whitefield'] → True")

# Test 4: Route matching
print("\n[Test 4] Route matching")
result = match_location_route(
    {"origin": "bangalore", "destination": "goa"},
    {"origin": "bangalore", "destination": "goa"},
    [], []
)
assert result == True, "Same routes should match"
print("✅ bangalore→goa == bangalore→goa → True")

result = match_location_route(
    {"origin": "bangalore", "destination": "goa"},
    {"origin": "delhi", "destination": "goa"},
    [], []
)
assert result == False, "Different routes should not match"
print("✅ bangalore→goa != delhi→goa → False")

# Test 5: Mode-based matching
print("\n[Test 5] Mode-based matching")

# Global mode
result = match_location_v2("bangalore", "global", [], "delhi", "explicit", [])
assert result == True, "Global mode should always match"
print("✅ global mode → True (always matches)")

# Explicit mode with match
result = match_location_v2("bangalore", "explicit", [], "bangalore", "explicit", [])
assert result == True, "Explicit mode with same location should match"
print("✅ explicit: bangalore == bangalore → True")

# Explicit mode without match
result = match_location_v2("bangalore", "explicit", [], "delhi", "explicit", [])
assert result == False, "Explicit mode with different location should not match"
print("✅ explicit: bangalore != delhi → False")

print("\n" + "="*80)
print("ALL LOCATION TESTS PASSED! ✅")
print("="*80)
