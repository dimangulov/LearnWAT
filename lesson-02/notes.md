# Lesson 02 — Your First Tool Call

## The 4-Step Pattern (memorize this)

```
1. Define tool  →  JSON Schema describing name, inputs, purpose
2. Send message →  user message + tools list to Claude
3. Execute tool →  Claude returns tool_use block; YOU run the function
4. Send result  →  append tool_result; Claude gives final answer
```

## Key API Details

### Tool definition shape
```python
{
    "name": "tool_name",
    "description": "What it does — Claude reads this to decide when to use it",
    "input_schema": { ... }  # standard JSON Schema
}
```

### Stop reasons
- `"end_turn"`   → Claude answered directly, no tool needed
- `"tool_use"`   → Claude wants to call a tool — you must handle it

### Message structure after tool call
```
messages = [
    user:      "original question"
    assistant: [text_block?, tool_use_block]   ← Claude's response
    user:      [tool_result block]             ← your tool's output
]
```
Then call the API again to get Claude's final answer.

## Critical insight
Claude does NOT execute tools. It only *requests* them.
YOU are responsible for running the function and returning the result.
Claude is the decision-maker; your code is the executor.

## What's Next
Lesson 03: Multiple tools — Claude picks the right one.
