"""
Lesson 06 — Structured Output: Making Claude Return Data, Not Text

Pattern: define a "output" tool that Claude MUST call to return its answer.
You get a validated dict instead of a string to parse.

Use case: extract structured info from unstructured text.
Input:  "John is 34, lives in Berlin, works as an engineer."
Output: {"name": "John", "age": 34, "city": "Berlin", "job": "engineer"}
"""

import json
import anthropic

client = anthropic.Anthropic()

# ── The output schema as a tool ───────────────────────────────────────────────
# We're not calling an external function — this tool IS the output format.
# When Claude "calls" it, we just read the inputs as the structured result.

extract_person_tool = {
    "name": "extract_person",
    "description": "Output the structured person data extracted from the text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "name":    {"type": "string",  "description": "Full name"},
            "age":     {"type": ["integer", "null"], "description": "Age in years"},
            "city":    {"type": "string",  "description": "City of residence"},
            "job":     {"type": "string",  "description": "Job title or profession"},
        },
        "required": ["name", "city", "job"]
    }
}

# ── Force Claude to use the tool ─────────────────────────────────────────────
# tool_choice={"type": "tool", "name": "extract_person"}
# This makes the tool call mandatory — Claude cannot respond with text.

def extract_person(text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=[extract_person_tool],
        tool_choice={"type": "tool", "name": "extract_person"},  # force it
        messages=[{
            "role": "user",
            "content": f"Extract the person info from this text:\n\n{text}"
        }]
    )

    # With tool_choice forced, the first content block is always tool_use
    tool_block = response.content[0]
    return tool_block.input   # already a dict — no JSON parsing needed


# ── Test cases ────────────────────────────────────────────────────────────────

texts = [
    "John is 34, lives in Berlin, works as an engineer.",
    "My name is Priya. I'm 28, a designer based in Mumbai.",
    "Call me Lukas. 41 years old. Software architect in Warsaw.",
    "Call me Damir. Software architect in Warsaw.",
]

for text in texts:
    result = extract_person(text)
    print(f"Input:  {text}")
    print(f"Output: {json.dumps(result, indent=2)}")
    print()
