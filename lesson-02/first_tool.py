"""
Lesson 02 — Your First Tool Call

Pattern:
  1. Define a tool (JSON Schema)
  2. Send a message to Claude WITH the tool available
  3. Claude decides to call it → you execute it → send result back
  4. Claude gives a final answer
"""

import anthropic
import json

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ── Step 1: Define the tool ──────────────────────────────────────────────────
# This is a "fake" tool — it returns hardcoded data.
# In a real workflow, this would call an API, database, etc.

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'London'"
                }
            },
            "required": ["city"]
        }
    }
]

def get_weather(city: str) -> str:
    """Fake implementation — replace with a real API call."""
    fake_data = {
        "london": "12°C, overcast",
        "tokyo":  "22°C, sunny",
        "new york": "8°C, windy",
    }
    return fake_data.get(city.lower(), f"No data for {city}")


# ── Step 2: Send the user message with tools available ───────────────────────

messages = [
    {"role": "user", "content": "What's the weather like in Tokyo?"}
]

response = client.messages.create(
    model="claude-haiku-4-5-20251001",   # fast, cheap — good for learning
    max_tokens=1024,
    tools=tools,
    messages=messages
)

print(f"Stop reason: {response.stop_reason}")  # expect: 'tool_use'
print(f"Content blocks: {[b.type for b in response.content]}")


# ── Step 3: Check if Claude wants to call a tool ─────────────────────────────

if response.stop_reason == "tool_use":
    # Find the tool_use block
    tool_use_block = next(b for b in response.content if b.type == "tool_use")

    tool_name  = tool_use_block.name
    tool_input = tool_use_block.input
    tool_id    = tool_use_block.id

    print(f"\nClaude wants to call: {tool_name}({tool_input})")

    # Execute the tool (your code, not Claude's)
    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    else:
        result = "Unknown tool"

    print(f"Tool result: {result}")

    # ── Step 4: Send the result back to Claude ───────────────────────────────
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result
            }
        ]
    })

    final_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    print(f"\nFinal answer: {final_response.content[0].text}")
