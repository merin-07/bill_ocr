import os
import json
from google import genai

# -----------------------------
# FINAL CATEGORY LIST
# -----------------------------
CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Utilities",
    "Entertainment",
    "Healthcare",
    "Savings",
    "Other"
]

MODEL = "models/gemini-2.5-flash"


# -----------------------------
# RECEIPT AUTO-CATEGORIZATION
# -----------------------------
def categorize_receipt(ocr_text: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a receipt understanding engine.

Correct OCR spelling mistakes.
Categorize each item into ONLY one of these categories:
{", ".join(CATEGORIES)}

Rules:
- Output ONLY valid JSON
- No explanations
- JSON must start with {{ and end with }}

Format:
{{
  "merchant": "",
  "items": [
    {{
      "name": "",
      "category": "",
      "price": 0.0
    }}
  ],
  "total": 0.0
}}

Receipt text:
{ocr_text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    raw = response.text.strip()

    # Defensive JSON parsing
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])

    # Validate categories
    for item in data.get("items", []):
        if item.get("category") not in CATEGORIES:
            item["category"] = "Other"

    return data


# -----------------------------
# MANUAL ITEM AUTO-CATEGORIZE
# -----------------------------
def categorize_manual_item(name: str, amount: float) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Categorize this expense into ONLY one of the following categories:

{", ".join(CATEGORIES)}

Item: {name}

Return ONLY the category name.
No explanations.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    category = response.text.strip()

    if category not in CATEGORIES:
        category = "Other"

    return {
        "merchant": "Manual Entry",
        "items": [
            {
                "name": name,
                "category": category,
                "price": amount
            }
        ],
        "total": amount
    }
