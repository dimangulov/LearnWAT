# Lesson 06 — Structured Output: Making Claude Return Data, Not Text

## The pattern

Define a tool whose schema describes your desired output shape.
Force Claude to call it with `tool_choice={"type": "tool", "name": "..."}`.
Read `response.content[0].input` — it's already a dict.

```python
tool_choice={"type": "tool", "name": "my_output_tool"}
```

## tool_choice options

| Value | Meaning |
|---|---|
| `{"type": "auto"}` | Claude decides whether to use a tool (default) |
| `{"type": "any"}` | Claude must use some tool, its choice which |
| `{"type": "tool", "name": "X"}` | Claude must call tool X specifically |

## Why this beats asking Claude to "return JSON"

| Approach | Risk |
|---|---|
| "Return JSON please" in system prompt | Claude might add prose, wrap in markdown, miss fields |
| `tool_choice` forced | Guaranteed structure, validated against schema, no parsing |

## What the schema buys you

- `"type": "integer"` — Claude returns a number, not "34 years old"
- `"required": [...]` — missing fields cause an API error, not silent None
- `"description"` on each field — guides Claude on what to extract

## Composing with the agent loop

Structured output + tools + memory = the full pattern:
1. User speaks naturally
2. Agent uses real tools to gather data
3. Agent outputs structured result for downstream code to consume

## What's Next
Lesson 07: Multi-agent — one orchestrator Claude delegates subtasks to worker Claudes.
