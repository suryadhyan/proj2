"""
Quick test script to test a single query with OpenAI API
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_prompt():
    """Load full prompt"""
    with open("D:/matching-github/proj2/prompt/PROMPT_STAGE2.txt", 'r', encoding='utf-8') as f:
        content = f.read()

    # Use Stage 3 extraction section (first 85%)
    stage2_marker = "1. PURPOSE (SINGLE RESPONSIBILITY — LOCKED)"
    if stage2_marker in content:
        idx = content.index(stage2_marker)
        extraction_prompt = content[:idx]
    else:
        extraction_prompt = content[:int(len(content) * 0.85)]

    # Add output format
    output_spec = """

## OUTPUT FORMAT

Return ONLY valid JSON with ALL 14 fields. Empty arrays/objects are allowed.

{
  "intent": "product|service|mutual",
  "subintent": "buy|sell|seek|provide|connect",
  "domain": ["Domain"],
  "primary_mutual_category": [],
  "items": [{"type": "", "categorical": {}, "min": {}, "max": {}, "range": {}}],
  "item_exclusions": [],
  "other_party_preferences": {"categorical": {}, "min": {}, "max": {}, "range": {}},
  "other_party_exclusions": {},
  "self_attributes": {"categorical": {}, "min": {}, "max": {}, "range": {}},
  "self_exclusions": {},
  "target_location": {},
  "location_match_mode": "near_me",
  "location_exclusions": [],
  "reasoning": ""
}
"""

    return extraction_prompt + output_spec


def test_query(query: str):
    """Test a single query"""
    print(f"🔍 Testing query: {query}")
    print("="*80)

    prompt = load_prompt()
    print(f"📄 Prompt length: {len(prompt)} characters")

    print("\n🤖 Calling OpenAI API (gpt-4o)...")

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

        print("\n✅ SUCCESS!")
        print("\n📋 OUTPUT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Token usage
        usage = response.usage
        print(f"\n📊 Token Usage:")
        print(f"   Prompt: {usage.prompt_tokens}")
        print(f"   Completion: {usage.completion_tokens}")
        print(f"   Total: {usage.total_tokens}")

        return result

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None


if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found")
        print("Please add it to .env file")
        sys.exit(1)

    # Default test query
    test_queries = [
        "looking for a used macbook pro with at least 16gb ram under 80k",
        "need a kannada speaking plumber urgently in jayanagar",
        "anyone up for weekend treks around bangalore?"
    ]

    if len(sys.argv) > 1:
        # Use provided query
        query = " ".join(sys.argv[1:])
    else:
        # Use first test query
        query = test_queries[0]

    test_query(query)
