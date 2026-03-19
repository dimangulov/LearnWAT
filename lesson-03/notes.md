# Lesson 03 — Multiple Tools: Claude Picks the Right One

## What changed from Lesson 02

| Lesson 02 | Lesson 03 |
|-----------|-----------|
| 1 tool    | Many tools |
| Handle once | Loop until `end_turn` |
| One tool call | Multiple tool calls possible |
| Manual dispatch | Tool dispatcher function |

## The Agent Loop (the real pattern)

```python
while True:
    response = call_claude(messages)

    if response.stop_reason == "end_turn":
        return response.text   # done

    if response.stop_reason == "tool_use":
        for each tool_use block:
            result = run_tool(name, inputs)
            append tool_result to messages
        # loop again — Claude processes results
```

This loop IS the agent. Everything else is details.

## How Claude picks the right tool

It reads the `description` field of each tool.
This is the most important part of tool design:
- Be specific and honest about what the tool does
- Mention example inputs if helpful
- Bad description → Claude calls the wrong tool

## Tool dispatcher pattern

```python
def run_tool(name, inputs):
    if name == "get_weather": return get_weather(**inputs)
    if name == "get_time":    return get_time(**inputs)
    ...
```

Centralizes routing. As your toolbox grows, this stays clean.

## Multiple tool calls in one turn

When asked "weather AND time in Tokyo?", Claude may return
TWO tool_use blocks in a single response. You must handle all of them
before sending back tool_results.

## What's Next
Lesson 04: Agents — giving Claude a subagent to delegate to.
