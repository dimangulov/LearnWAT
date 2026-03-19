# Lesson 10 — Putting It All Together

## What's in this agent

| Concept | Lesson | Where in code |
|---|---|---|
| System prompt + role | L04 | `SYSTEM` constant |
| Security rules in system prompt | L09 | `SYSTEM` — SECURITY RULES section |
| Hard tool enforcement | L04 | `ALL_TOOLS` list (only what's needed) |
| Conversation memory | L05 | `messages = []` in `main()`, passed into `agent_step` |
| Sanitized tool results | L09 | `sanitize()` called in every tool |
| Async parallel tool calls | L08 | `asyncio.gather()` in `agent_step` |
| Exception-safe gather | L08 | `return_exceptions=True` + fallback |

## What structured output (L06) looks like here

`save_note` uses the tool-as-output pattern passively:
Claude calls it with structured fields, we just store them.
For full forced structured output, add `tool_choice` when you need
Claude to always produce a note regardless of what it decides.

## The complete agent anatomy

```
REPL loop (user input)
    │
    ▼
agent_step(messages, input)        ← stateful: messages persists
    │
    ├── call Claude with system + tools + messages
    │
    ├── stop_reason == end_turn  →  return text answer
    │
    └── stop_reason == tool_use
            │
            ├── asyncio.gather all tool calls in parallel
            │       └── each tool sanitizes its output
            │
            └── append tool_results → loop again

```

## What to build next

Now you have all the primitives:
- Add real tools: file read/write, web search, code execution
- Swap fake data for real APIs
- Add a proper vector DB for long-term memory (beyond L05 sliding window)
- Deploy as a web service with FastAPI + WebSockets for streaming
