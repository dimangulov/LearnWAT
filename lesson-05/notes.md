# Lesson 05 — Conversation Memory: Keeping Context Across Turns

## The one change

| Lessons 02–04 | Lesson 05 |
|---|---|
| `messages = []` inside `run_agent()` | `messages = []` outside, passed in |
| Fresh context every call | Grows with every turn |
| Stateless | Stateful |

## What the messages list looks like after 2 turns

```
[
  {role: "user",      content: "What's the weather in London?"},
  {role: "assistant", content: [tool_use(get_weather, london)]},
  {role: "user",      content: [tool_result("12°C overcast")]},
  {role: "assistant", content: [text("London is 12°C and overcast.")]},

  {role: "user",      content: "What about Tokyo?"},
  {role: "assistant", content: [tool_use(get_weather, tokyo)]},
  {role: "user",      content: [tool_result("22°C sunny")]},
  {role: "assistant", content: [text("Tokyo is 22°C and sunny.")]},
]
```

Claude resolves "What about Tokyo?" because the London question is still in the list.

## The problem this creates: context window limits

The messages list grows forever. Claude has a finite context window.
Long conversations will eventually hit the token limit and fail.

Solutions (preview for later lessons):
1. **Sliding window** — keep only the last N messages
2. **Summarization** — compress old turns into a summary message
3. **External memory** — store facts in a database, retrieve when relevant

## Rule of thumb

Short-lived sessions (single task): stateless is fine.
Ongoing conversations or long tasks: you need stateful messages.
