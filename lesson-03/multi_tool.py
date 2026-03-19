"""
Lesson 03 — Multiple Tools: Claude Picks the Right One

We give Claude 3 tools:
  - get_weather(city)
  - get_time(city)
  - calculate(expression)

Claude decides which one to call based on the user's question.
We handle ALL possible tool calls in a loop.
"""

import anthropic

client = anthropic.Anthropic()

# ── Toolbox: 3 tools ─────────────────────────────────────────────────────────

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current local time in a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression. E.g. '2 + 2' or '100 * 1.08'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
]

# ── Tool implementations ─────────────────────────────────────────────────────

def get_weather(city: str) -> str:
    data = {"london": "12°C overcast", "tokyo": "22°C sunny", "new york": "8°C windy"}
    return data.get(city.lower(), f"No data for {city}")

def get_time(city: str) -> str:
    data = {"london": "14:30 GMT", "tokyo": "23:30 JST", "new york": "09:30 EST"}
    return data.get(city.lower(), f"No time data for {city}")

def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})  # safe: no builtins
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# ── Tool dispatcher ──────────────────────────────────────────────────────────
# One place to route tool names → functions.
# This pattern scales cleanly as you add more tools.

def run_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":  return get_weather(**inputs)
    if name == "get_time":     return get_time(**inputs)
    if name == "calculate":    return calculate(**inputs)
    return f"Unknown tool: {name}"

# ── The agent loop ───────────────────────────────────────────────────────────
# KEY CHANGE from Lesson 02: we loop until stop_reason == "end_turn"
# Claude may call multiple tools before giving a final answer.

MAX_TURNS = 10  # safety cap — prevents runaway tool loops

def run_agent(user_message: str):
    print(f"\nUser: {user_message}")
    print("-" * 40)

    messages = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            # Claude is done — extract and print the text answer
            answer = next(b.text for b in response.content if b.type == "text")
            print(f"Claude: {answer}")
            break

        if response.stop_reason == "tool_use":
            # Append Claude's response (may include multiple tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Handle every tool call in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → calling {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"  ← result: {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})
            # Loop continues — Claude processes results and may call more tools
    else:
        print(f"ERROR: exceeded {MAX_TURNS} turns without end_turn — aborting")


# ── Test with different questions ────────────────────────────────────────────

run_agent("What's the weather in London?")
run_agent("What's 15% tip on a $47 dinner?")
run_agent("What time is it in Tokyo, and what's the weather there?")
