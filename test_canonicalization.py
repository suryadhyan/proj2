"""
Test canonicalization and polysemy handling
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt():
    """Load extraction prompt"""
    with open("D:/matching-github/proj2/prompt/PROMPT_STAGE2.txt", 'r', encoding='utf-8') as f:
        content = f.read()

    stage2_marker = "1. PURPOSE (SINGLE RESPONSIBILITY — LOCKED)"
    if stage2_marker in content:
        idx = content.index(stage2_marker)
        extraction_prompt = content[:idx]
    else:
        extraction_prompt = content[:int(len(content) * 0.85)]

    output_spec = """

## OUTPUT FORMAT (14 FIELDS)

{
  "intent": "product|service|mutual",
  "subintent": "buy|sell|seek|provide|connect",
  "domain": ["Domain"],
  "primary_mutual_category": [],
  "items": [{"type": "", "categorical": {}, "min": {}, "max": {}, "range": {}}],
  "item_exclusions": [],
  "other_party_preferences": {"identity": [], "habits": {}, "min": {}, "max": {}, "range": {}},
  "other_party_exclusions": {},
  "self_attributes": {"identity": [], "habits": {}, "min": {}, "max": {}, "range": {}},
  "self_exclusions": {},
  "target_location": {},
  "location_match_mode": "near_me",
  "location_exclusions": [],
  "reasoning": ""
}
"""
    return extraction_prompt + output_spec

def test_query(query: str, test_name: str):
    """Test a single query"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {test_name}")
    print(f"{'='*80}")
    print(f"Query: {query}")

    prompt = load_prompt()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        print(f"\n✅ SUCCESS")
        print(f"\nIntent: {result.get('intent')} / {result.get('subintent')}")
        print(f"Domain: {result.get('domain')}")

        if result.get('items'):
            print(f"\nItems:")
            for item in result['items']:
                print(f"  • Type: {item.get('type')}")
                if item.get('categorical'):
                    print(f"    Categorical: {json.dumps(item.get('categorical'), indent=6)}")
                if item.get('min'):
                    print(f"    Min: {json.dumps(item.get('min'), indent=6)}")
                if item.get('max'):
                    print(f"    Max: {json.dumps(item.get('max'), indent=6)}")

        if result.get('other_party_preferences'):
            opp = result['other_party_preferences']
            if any(opp.get(k) for k in ['identity', 'habits', 'min', 'max', 'range']):
                print(f"\nOther Party Preferences:")
                print(f"  {json.dumps(opp, indent=2)}")

        if result.get('self_attributes'):
            sa = result['self_attributes']
            if any(sa.get(k) for k in ['identity', 'habits', 'min', 'max', 'range']):
                print(f"\nSelf Attributes:")
                print(f"  {json.dumps(sa, indent=2)}")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("🧪 CANONICALIZATION & POLYSEMY TEST SUITE")
    print("="*80)

    tests = [
        # Canonicalization Tests
        {
            "name": "Test 1: Phone Synonyms",
            "queries": [
                "looking for a phone under 30k",
                "need a mobile under 30k",
                "want to buy a cellphone under 30k",
                "searching for a smartphone under 30k"
            ]
        },
        {
            "name": "Test 2: Laptop Synonyms",
            "queries": [
                "selling my laptop",
                "selling my notebook",
                "selling my portable computer"
            ]
        },
        {
            "name": "Test 3: Condition Synonyms",
            "queries": [
                "used iphone for sale",
                "second hand iphone for sale",
                "pre-owned iphone for sale",
                "2nd hand iphone for sale"
            ]
        },

        # Polysemy Tests
        {
            "name": "Test 4: 'Language' Polysemy - Programming",
            "queries": [
                "need a developer who knows Python language",
                "looking for software engineer with Rust language experience"
            ]
        },
        {
            "name": "Test 5: 'Language' Polysemy - Speaking",
            "queries": [
                "need a plumber who speaks Kannada language",
                "looking for tutor who speaks Hindi language"
            ]
        },
        {
            "name": "Test 6: 'Size' Polysemy",
            "queries": [
                "need a 2BHK flat with 1000 sqft size",  # size = area
                "looking for XL size t-shirt",            # size = clothing size
                "need 256GB size phone"                   # size = storage
            ]
        },
        {
            "name": "Test 7: 'Experience' Polysemy",
            "queries": [
                "need tutor with 5 years experience",     # time-based
                "looking for experienced yoga instructor" # skill level
            ]
        },

        # Edge Cases
        {
            "name": "Test 8: Currency Detection",
            "queries": [
                "laptop under 50k",           # implicit INR
                "laptop under $500",          # explicit USD
                "laptop under ₹50000",        # explicit INR symbol
                "laptop under 5 lakh rupees"  # Indian numbering
            ]
        },
        {
            "name": "Test 9: Constraint Detection",
            "queries": [
                "laptop with 16GB RAM",              # exact
                "laptop with at least 16GB RAM",     # min
                "laptop under 80k",                  # max
                "laptop between 50k and 80k",        # range
                "laptop around 60k"                  # approximate (what does it do?)
            ]
        },
        {
            "name": "Test 10: Implication Rules",
            "queries": [
                "single owner car for sale",         # should extract: condition=used + ownership=single
                "first owner bike",                  # should extract: condition=used + ownership=first
                "sealed iPhone for sale"             # should extract: condition=new + packaging=sealed
            ]
        }
    ]

    results = {}

    for test_group in tests:
        print(f"\n\n{'#'*80}")
        print(f"# {test_group['name']}")
        print(f"{'#'*80}")

        group_results = []

        for query in test_group['queries']:
            result = test_query(query, test_group['name'])
            group_results.append({
                "query": query,
                "result": result
            })

        results[test_group['name']] = group_results

        # Summary for this test group
        print(f"\n{'='*80}")
        print(f"📊 {test_group['name']} - SUMMARY")
        print(f"{'='*80}")

        if "Synonyms" in test_group['name'] or "Phone" in test_group['name'] or "Laptop" in test_group['name'] or "Condition" in test_group['name']:
            # Check canonicalization consistency
            item_types = []
            for gr in group_results:
                if gr['result'] and gr['result'].get('items'):
                    item_type = gr['result']['items'][0].get('type')
                    item_types.append(item_type)
                    print(f"  '{gr['query'][:40]}...' → type: '{item_type}'")

            if len(set(item_types)) == 1:
                print(f"  ✅ CONSISTENT: All canonicalized to '{item_types[0]}'")
            else:
                print(f"  ❌ INCONSISTENT: Got {set(item_types)}")

        elif "Polysemy" in test_group['name']:
            # Check if context correctly resolved
            for gr in group_results:
                if gr['result']:
                    print(f"  '{gr['query'][:50]}...'")
                    if gr['result'].get('items'):
                        items = gr['result']['items'][0]
                        if items.get('categorical'):
                            print(f"    → Items categorical: {items['categorical']}")
                    if gr['result'].get('other_party_preferences', {}).get('identity'):
                        print(f"    → Other party identity: {gr['result']['other_party_preferences']['identity']}")

    # Save results
    with open("canonicalization_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*80}")
    print("💾 Results saved to: canonicalization_test_results.json")
    print("="*80)
