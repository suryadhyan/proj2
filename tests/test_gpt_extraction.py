"""
Test GPT extraction with PROMPT_STAGE2 copy.txt
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import OpenAI

# Load prompt
prompt_path = "prompt/GLOBAL_REFERENCE_CONTEXT.md"
with open(prompt_path, "r", encoding="utf-8") as f:
    extraction_prompt = f.read()

print(f"✓ Loaded prompt: {len(extraction_prompt)} characters")

# Initialize OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Test query
query = "looking for flatmate who is better in cooking and doesn't smoke and i don't drink as well"

print(f"\nQuery: {query}\n")
print("Calling GPT-4o...")

# Call OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": extraction_prompt},
        {"role": "user", "content": query}
    ],
    temperature=0.0,
    response_format={"type": "json_object"}
)

# Parse response
output_text = response.choices[0].message.content
extracted_data = json.loads(output_text)

print("\n" + "="*80)
print("EXTRACTED JSON:")
print("="*80)
print(json.dumps(extracted_data, indent=2))

# Check for required fields
required_fields = [
    "intent", "subintent", "domain", "items",
    "other_party_preferences", "self_attributes", "target_location",
    "primary_mutual_category", "item_exclusions",
    "other_party_exclusions", "self_exclusions",
    "location_match_mode", "location_exclusions", "reasoning"
]

print("\n" + "="*80)
print("FIELD CHECK:")
print("="*80)

missing_fields = []
for field in required_fields:
    if field in extracted_data:
        print(f"✓ {field}")
    else:
        print(f"✗ {field} - MISSING")
        missing_fields.append(field)

if missing_fields:
    print(f"\n❌ Missing {len(missing_fields)} fields: {missing_fields}")
else:
    print(f"\n✅ All {len(required_fields)} fields present!")

# Check items specifically
if "items" in extracted_data:
    items = extracted_data["items"]
    print(f"\n📦 Items: {len(items)} item(s)")
    if items:
        print(json.dumps(items, indent=2))
    else:
        print("⚠️ Items array is empty!")
