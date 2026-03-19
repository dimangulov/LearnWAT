"""
Lesson 08 — Parallel Workers with Async

Same task as Lesson 07 (travel brief), but all three workers
fire simultaneously using asyncio.

Key changes from Lesson 07:
- Use AsyncAnthropic instead of Anthropic
- Workers are async def
- asyncio.gather() runs them all at once
- Orchestrator tool loop is async too
"""

import asyncio
import anthropic

client = anthropic.AsyncAnthropic()   # async client

# ── Async workers ─────────────────────────────────────────────────────────────

async def weather_agent(city: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system="You are a weather reporter. Give a single sentence weather summary.",
        messages=[{"role": "user", "content": f"Weather in {city} right now."}]
    )
    return response.content[0].text

async def facts_agent(city: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are a travel encyclopaedia. Return exactly 3 interesting facts as a numbered list.",
        messages=[{"role": "user", "content": f"Tell me about {city}."}]
    )
    return response.content[0].text

async def tips_agent(city: str) -> str:
    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system="You are a seasoned travel guide. Give exactly 2 practical tips for a first-time visitor.",
        messages=[{"role": "user", "content": f"Tips for visiting {city}."}]
    )
    return response.content[0].text

# ── Parallel dispatcher ───────────────────────────────────────────────────────
# When the orchestrator calls multiple tools in one turn,
# we fire all of them concurrently with asyncio.gather().

async def run_workers_parallel(tool_calls: list) -> list:
    """Run all tool calls at the same time, return results in order."""

    async def run_one(block):
        city = block.input["city"]
        print(f"    [worker:{block.name}] start — {city}")
        if block.name == "get_weather": result = await weather_agent(city)
        elif block.name == "get_facts": result = await facts_agent(city)
        elif block.name == "get_tips":  result = await tips_agent(city)
        else:                           result = f"Unknown worker: {block.name}"
        print(f"    [worker:{block.name}] done")
        return {"type": "tool_result", "tool_use_id": block.id, "content": result}

    # gather() fires all coroutines simultaneously
    return await asyncio.gather(*[run_one(b) for b in tool_calls])


# ── Async orchestrator loop ───────────────────────────────────────────────────

worker_tools = [
    {
        "name": "get_weather",
        "description": "Get a current weather summary for a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    },
    {
        "name": "get_facts",
        "description": "Get 3 interesting facts about a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    },
    {
        "name": "get_tips",
        "description": "Get 2 practical travel tips for visiting a city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
]

ORCHESTRATOR_SYSTEM = """
You are a travel brief writer. When given a city:
1. Call get_weather, get_facts, and get_tips for that city.
2. Combine the results into a short, well-formatted travel brief.
Do not guess — always call the workers first.
"""

MAX_TURNS = 10

async def run_orchestrator(city: str):
    import time
    print(f"\nBuilding travel brief for {city}")
    print("=" * 50)
    start = time.perf_counter()

    messages = [{"role": "user", "content": f"Create a travel brief for {city}."}]

    for _ in range(MAX_TURNS):
        response = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=ORCHESTRATOR_SYSTEM,
            tools=worker_tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            brief = next(b.text for b in response.content if b.type == "text")
            elapsed = time.perf_counter() - start
            print(f"\n{brief}")
            print(f"\n(completed in {elapsed:.1f}s)")
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # All workers fire at the same time
            tool_results = await run_workers_parallel(tool_blocks)

            messages.append({"role": "user", "content": tool_results})
    else:
        print("ERROR: max turns exceeded")


asyncio.run(run_orchestrator("Tokyo"))
