"""
Lesson 05 — Conversation Memory: Keeping Context Across Turns

The only change from Lesson 04:
- messages = [] lives OUTSIDE the per-turn loop
- Each user input is appended; history grows with every turn
- Claude always sees the full conversation

Try:
  > What's the weather in London?
  > What about Tokyo?          ← "what about" requires memory to resolve
  > How many cities did I ask about?
"""

import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]

def get_weather(city):
    data = {"london": "12°C overcast", "tokyo": "22°C sunny", "new york": "8°C windy"}
    return data.get(city.lower(), f"No data for {city}")

def run_tool(name, inputs):
    if name == "get_weather": return get_weather(**inputs)
    return f"Unknown tool: {name}"

SYSTEM = "You are a helpful weather assistant. Remember everything the user has told you."

MAX_TURNS = 10

# ── The key difference: messages persists across user inputs ─────────────────

def agent_step(messages: list, user_input: str) -> str:
    """Add one user message, run the agent loop, return Claude's final reply."""
    messages.append({"role": "user", "content": user_input})

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            answer = next(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            return answer

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

    return "ERROR: max turns exceeded"


# ── REPL: one shared messages list for the whole session ─────────────────────

def main():
    messages = []   # <── lives here, shared across all turns
    print("Weather assistant (type 'quit' to exit, 'history' to see messages)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "history":
            for m in messages:
                role = m["role"]
                content = m["content"]
                # content can be str or list of blocks
                if isinstance(content, str):
                    print(f"  [{role}] {content[:80]}")
                else:
                    for block in content:
                        if hasattr(block, "text"):
                            print(f"  [{role}] {block.text[:80]}")
                        elif hasattr(block, "type"):
                            print(f"  [{role}] <{block.type}>")
            continue

        reply = agent_step(messages, user_input)
        print(f"Claude: {reply}\n")


if __name__ == "__main__":
    main()
