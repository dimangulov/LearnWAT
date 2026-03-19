"""
Lesson 04 — System Prompts: Giving Your Agent a Role

We take the same tools from Lesson 03 and run the same agent loop,
but swap the system prompt. The agent's personality and behavior
change completely — the tools and loop code don't change at all.
"""

import anthropic

client = anthropic.Anthropic()

# ── Same tools as Lesson 03 ───────────────────────────────────────────────────

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression. E.g. '2 + 2' or '100 * 1.08'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }
    }
]

def get_weather(city):
    data = {"london": "12°C overcast", "tokyo": "22°C sunny", "new york": "8°C windy"}
    return data.get(city.lower(), f"No data for {city}")

def calculate(expression):
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

def run_tool(name, inputs):
    if name == "get_weather": return get_weather(**inputs)
    if name == "calculate":   return calculate(**inputs)
    return f"Unknown tool: {name}"

# ── The one thing that changes: the system prompt ─────────────────────────────

SYSTEMS = {
    "assistant": """
        You are a helpful, friendly general assistant.
        Answer clearly and concisely.
    """,

    "pirate": """
        You are a pirate assistant. Speak in pirate dialect at all times.
        Use tools when needed, but always respond in character.
        Say "Arrr" often. Refer to the user as "landlubber".
    """,

    "strict_weather_only": """
        You are a weather service bot. You ONLY answer weather questions.
        If the user asks about anything else, politely refuse and explain
        you only handle weather queries.
    """
}

# Hard enforcement: only these tool names are available per persona.
# Claude cannot call a tool that isn't in the list — it simply doesn't see it.
ALLOWED_TOOLS = {
    "assistant":           ["get_weather", "calculate"],
    "pirate":              ["get_weather", "calculate"],
    "strict_weather_only": ["get_weather"],          # calculate removed entirely
}

def get_tools_for(persona: str) -> list:
    allowed = ALLOWED_TOOLS[persona]
    return [t for t in tools if t["name"] in allowed]

# ── Agent loop (identical to Lesson 03) ──────────────────────────────────────

MAX_TURNS = 10

def run_agent(user_message: str, persona: str = "assistant"):
    system = SYSTEMS[persona]
    print(f"\n[{persona.upper()}] User: {user_message}")
    print("-" * 40)

    messages = [{"role": "user", "content": user_message}]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=get_tools_for(persona),   # only the allowed subset
            messages=messages
        )

        if response.stop_reason == "end_turn":
            answer = next(b.text for b in response.content if b.type == "text")
            print(f"Claude: {answer}")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → {block.name}({block.input})")
                    result = run_tool(block.name, block.input)
                    print(f"  ← {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
    else:
        print(f"ERROR: exceeded {MAX_TURNS} turns")


# ── Same question, three different agents ─────────────────────────────────────

run_agent("What's the weather in Tokyo?",   persona="assistant")
run_agent("What's the weather in Tokyo?",   persona="pirate")
run_agent("What is 42 * 7?",               persona="strict_weather_only")
