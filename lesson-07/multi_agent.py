"""
Lesson 07 — Multi-Agent: One Claude Orchestrating Others

Task: research a city and produce a travel brief.

Orchestrator (claude-sonnet): plans, delegates, assembles final answer
  ├── worker: weather_agent   → current weather
  ├── worker: facts_agent     → 3 interesting facts
  └── worker: tips_agent      → 2 practical travel tips

Each worker is a plain function that calls Claude with its own
system prompt and returns a string. The orchestrator sees workers
as tools — it calls them the same way it calls any other tool.
"""

import anthropic

client = anthropic.Anthropic()

# ── Workers ───────────────────────────────────────────────────────────────────
# Each worker is a small focused agent with its own system prompt.
# They don't know about each other or the orchestrator.

def weather_agent(city: str) -> str:
    """Returns a one-line weather summary for the city."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap model for simple tasks
        max_tokens=128,
        system="You are a weather reporter. Give a single sentence weather summary.",
        messages=[{"role": "user", "content": f"Weather in {city} right now."}]
    )
    return response.content[0].text

def facts_agent(city: str) -> str:
    """Returns 3 interesting facts about the city."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are a travel encyclopaedia. Return exactly 3 interesting facts as a numbered list.",
        messages=[{"role": "user", "content": f"Tell me about {city}."}]
    )
    return response.content[0].text

def tips_agent(city: str) -> str:
    """Returns 2 practical travel tips for the city."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are a seasoned travel guide. Give exactly 2 practical tips for a first-time visitor.",
        messages=[{"role": "user", "content": f"Tips for visiting {city}."}]
    )
    return response.content[0].text


# ── Worker tools (exposed to the orchestrator) ────────────────────────────────
# The orchestrator sees workers as tools — same pattern as Lessons 03–06.

worker_tools = [
    {
        "name": "get_weather",
        "description": "Get a current weather summary for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "get_facts",
        "description": "Get 3 interesting facts about a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "get_tips",
        "description": "Get 2 practical travel tips for visiting a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]

def run_worker(name: str, inputs: dict) -> str:
    city = inputs["city"]
    print(f"    [worker:{name}] researching {city}...")
    if name == "get_weather": return weather_agent(city)
    if name == "get_facts":   return facts_agent(city)
    if name == "get_tips":    return tips_agent(city)
    return f"Unknown worker: {name}"


# ── Orchestrator ──────────────────────────────────────────────────────────────
# Uses a more capable model — it coordinates, not executes.
# Its job: decide which workers to call and assemble their outputs.

ORCHESTRATOR_SYSTEM = """
You are a travel brief writer. When given a city:
1. Call get_weather, get_facts, and get_tips for that city.
2. Combine the results into a short, well-formatted travel brief.
Do not guess — always call the workers first.
"""

MAX_TURNS = 10

def run_orchestrator(city: str):
    print(f"\nOrchestrator: building travel brief for {city}")
    print("=" * 50)

    messages = [{"role": "user", "content": f"Create a travel brief for {city}."}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # swap to sonnet for better briefs
            max_tokens=1024,
            system=ORCHESTRATOR_SYSTEM,
            tools=worker_tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            brief = next(b.text for b in response.content if b.type == "text")
            print(f"\n{brief}")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_worker(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
    else:
        print("ERROR: max turns exceeded")


run_orchestrator("Tokyo")
