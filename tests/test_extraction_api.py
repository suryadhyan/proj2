"""
Test extraction using OpenAI API
Compares actual output vs expected output from stage3_extraction1.json
"""
import os
import json
from openai import OpenAI
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MODEL = "gpt-4o-2024-11-20"  # Latest GPT-4o model
TEMPERATURE = 0.0  # Deterministic output


def load_prompt(filepath: str) -> str:
    """Load prompt from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_examples(filepath: str) -> List[Dict]:
    """Load test examples with expected outputs"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_stage2_prompt(full_prompt: str) -> str:
    """
    Extract Stage 2 classification prompt from full prompt
    Stage 2: Intent, subintent, domain, primary_mutual_category only
    """
    # Find the Stage 2 specific section (lines 3281 onwards in the file)
    stage2_marker = "1. PURPOSE (SINGLE RESPONSIBILITY — LOCKED)"

    if stage2_marker in full_prompt:
        idx = full_prompt.index(stage2_marker)
        stage2_section = full_prompt[idx:]
    else:
        # Fallback: use last 15% of prompt which contains Stage 2 spec
        stage2_section = full_prompt[-len(full_prompt)//7:]

    # Prepend essential context
    essential_context = """
# VRIDDHI - Query Extraction System

You are a structured data extraction system. Your task is to classify user queries.

## OUTPUT FORMAT
Return ONLY valid JSON with exactly these 4 fields:
{
  "intent": "product" | "service" | "mutual",
  "subintent": "buy" | "sell" | "seek" | "provide" | "connect",
  "domain": ["Domain Name"],
  "primary_mutual_category": ["Category"] or []
}

## RULES
- intent: ALWAYS one of: product, service, mutual
- subintent: Depends on intent (see mappings below)
- domain: Array from predefined domain list
- primary_mutual_category: Array with ONE item IF mutual, else []

"""

    return essential_context + stage2_section


def extract_stage3_prompt(full_prompt: str) -> str:
    """
    Extract Stage 3 full extraction prompt
    Uses first 85% of prompt which contains detailed extraction rules
    """
    # Get everything before Stage 2 section
    stage2_marker = "1. PURPOSE (SINGLE RESPONSIBILITY — LOCKED)"

    if stage2_marker in full_prompt:
        idx = full_prompt.index(stage2_marker)
        stage3_section = full_prompt[:idx]
    else:
        # Use first 85% of prompt
        stage3_section = full_prompt[:int(len(full_prompt) * 0.85)]

    # Add output format specification
    output_spec = """

## FINAL OUTPUT SCHEMA (14 FIELDS - MANDATORY)

Return ONLY valid JSON with ALL 14 fields:

{
  "intent": "product" | "service" | "mutual",
  "subintent": "buy" | "sell" | "seek" | "provide" | "connect",
  "domain": ["Array of domains"],
  "primary_mutual_category": ["Category"] or [],
  "items": [
    {
      "type": "canonical_item_type",
      "categorical": {"key": "value"},
      "min": {"axis": [{"type": "", "value": 0, "unit": ""}]},
      "max": {"axis": [{"type": "", "value": 0, "unit": ""}]},
      "range": {"axis": [{"type": "", "min": 0, "max": 0, "unit": ""}]}
    }
  ],
  "item_exclusions": [],
  "other_party_preferences": {
    "categorical": {},
    "min": {},
    "max": {},
    "range": {}
  },
  "other_party_exclusions": {},
  "self_attributes": {
    "categorical": {},
    "min": {},
    "max": {},
    "range": {}
  },
  "self_exclusions": {},
  "target_location": {"name": "location_name"} or {},
  "location_match_mode": "near_me" | "explicit" | "target_only" | "route" | "global",
  "location_exclusions": [],
  "reasoning": "Single paragraph explanation"
}

## CRITICAL RULES
1. ALL 14 fields must be present
2. Empty arrays/objects are valid
3. No additional fields
4. Exact does NOT exist - use range with min=max
5. Normalize units: years→months, TB→GB, etc.
6. "single owner" → condition:"used" + ownership:"single" (implication rule)

"""

    return stage3_section + output_spec


def call_stage2_api(query: str, prompt: str) -> Dict:
    """
    Call OpenAI API for Stage 2 classification
    Returns: {intent, subintent, domain, primary_mutual_category}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Query: {query}"
                }
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Stage 2 API Error: {e}")
        return None


def call_stage3_api(query: str, stage2_result: Dict, prompt: str) -> Dict:
    """
    Call OpenAI API for Stage 3 full extraction
    Uses Stage 2 results as context
    Returns: Full 14-field extraction
    """
    try:
        # Construct context message
        context = f"""Query: {query}

Classification (from Stage 2):
- Intent: {stage2_result['intent']}
- Subintent: {stage2_result['subintent']}
- Domain: {stage2_result['domain']}
- Primary Mutual Category: {stage2_result['primary_mutual_category']}

Now extract ALL details including items, attributes, location, etc.
"""

        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Stage 3 API Error: {e}")
        return None


def call_single_stage_api(query: str, prompt: str) -> Dict:
    """
    Single API call for full extraction (alternative approach)
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Query: {query}"
                }
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"❌ Single Stage API Error: {e}")
        return None


def compare_outputs(expected: Dict, actual: Dict) -> Dict[str, Any]:
    """
    Compare expected vs actual outputs
    Returns: {match: bool, differences: [...]}
    """
    differences = []

    # Check all 14 fields
    fields = [
        "intent", "subintent", "domain", "primary_mutual_category",
        "items", "item_exclusions",
        "other_party_preferences", "other_party_exclusions",
        "self_attributes", "self_exclusions",
        "target_location", "location_match_mode", "location_exclusions",
        "reasoning"
    ]

    for field in fields:
        if field not in actual:
            differences.append({
                "field": field,
                "issue": "missing_field",
                "expected": expected.get(field),
                "actual": None
            })
        elif expected.get(field) != actual.get(field):
            # Skip reasoning field for comparison (non-deterministic)
            if field != "reasoning":
                differences.append({
                    "field": field,
                    "issue": "value_mismatch",
                    "expected": expected.get(field),
                    "actual": actual.get(field)
                })

    return {
        "match": len([d for d in differences if d["field"] != "reasoning"]) == 0,
        "differences": differences
    }


def run_test_suite(approach: str = "single"):
    """
    Run full test suite
    approach: "single" or "two-stage"
    """
    print("=" * 80)
    print(f"🧪 EXTRACTION API TEST SUITE ({approach.upper()} approach)")
    print("=" * 80)

    # Load prompt and test examples
    print("\n📂 Loading files...")
    full_prompt = load_prompt("D:/matching-github/proj2/prompt/PROMPT_STAGE2.txt")
    test_examples = load_test_examples("D:/matching-github/proj2/new/stage3_extraction1.json")

    print(f"✅ Loaded prompt ({len(full_prompt)} chars)")
    print(f"✅ Loaded {len(test_examples)} test examples")

    # Prepare prompts
    if approach == "two-stage":
        stage2_prompt = extract_stage2_prompt(full_prompt)
        stage3_prompt = extract_stage3_prompt(full_prompt)
        print(f"✅ Stage 2 prompt: {len(stage2_prompt)} chars")
        print(f"✅ Stage 3 prompt: {len(stage3_prompt)} chars")
    else:
        single_prompt = extract_stage3_prompt(full_prompt)
        print(f"✅ Single stage prompt: {len(single_prompt)} chars")

    # Run tests
    results = []

    for i, example in enumerate(test_examples):
        query = example["query"]
        expected = example

        print(f"\n{'='*80}")
        print(f"Test {i+1}/{len(test_examples)}: {query[:60]}...")
        print('='*80)

        if approach == "two-stage":
            # Stage 2: Classification
            print("🔹 Stage 2: Classification...")
            stage2_result = call_stage2_api(query, stage2_prompt)

            if not stage2_result:
                results.append({
                    "query": query,
                    "success": False,
                    "error": "Stage 2 API failed"
                })
                continue

            print(f"   Intent: {stage2_result.get('intent')}")
            print(f"   Subintent: {stage2_result.get('subintent')}")
            print(f"   Domain: {stage2_result.get('domain')}")

            # Stage 3: Full extraction
            print("🔹 Stage 3: Full extraction...")
            actual = call_stage3_api(query, stage2_result, stage3_prompt)

        else:
            # Single stage
            print("🔹 Single API call for full extraction...")
            actual = call_single_stage_api(query, single_prompt)

        if not actual:
            results.append({
                "query": query,
                "success": False,
                "error": "API call failed"
            })
            continue

        # Compare results
        comparison = compare_outputs(expected, actual)

        if comparison["match"]:
            print("✅ PASS - Output matches expected")
        else:
            print("❌ FAIL - Differences found:")
            for diff in comparison["differences"]:
                if diff["field"] != "reasoning":  # Skip reasoning in summary
                    print(f"   • {diff['field']}: {diff['issue']}")
                    print(f"     Expected: {diff['expected']}")
                    print(f"     Actual: {diff['actual']}")

        results.append({
            "query": query,
            "success": comparison["match"],
            "actual": actual,
            "differences": comparison["differences"]
        })

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r["success"])
    failed = len(results) - passed

    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"📈 Success Rate: {passed/len(results)*100:.1f}%")

    # Save results
    output_file = f"test_results_{approach}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or environment variable")
        sys.exit(1)

    # Get approach from command line
    approach = sys.argv[1] if len(sys.argv) > 1 else "single"

    if approach not in ["single", "two-stage"]:
        print("❌ Invalid approach. Use 'single' or 'two-stage'")
        sys.exit(1)

    # Run tests
    run_test_suite(approach=approach)
