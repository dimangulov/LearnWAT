# Lesson 11 — LangChain: Same Agent, Less Boilerplate

## Direct mapping: Lesson 10 → Lesson 11

| Lesson 10 (manual) | Lesson 11 (LangChain) | What it does |
|---|---|---|
| `tools = [{name, description, input_schema}]` | `@tool` decorator | Defines a tool |
| `run_tool()` dispatcher | Built into `AgentExecutor` | Routes tool calls |
| `while True` + `stop_reason` loop | `AgentExecutor` | The agent loop |
| `MAX_TURNS = 10` | `max_iterations=10` | Safety cap |
| `messages = []` passed around | `ChatMessageHistory` | Conversation memory |
| `system=SYSTEM` in every call | `ChatPromptTemplate` | System prompt |
| `print(f"→ {tool}")` | `verbose=True` | Tool call logging |
| `asyncio.gather()` | `RunnableParallel` (separate) | Parallel execution |
| `sanitize()` | **Still manual** | Not provided by LangChain |

## Lines of code comparison

| | Lesson 10 | Lesson 11 |
|---|---|---|
| Agent loop | ~25 lines | 0 (AgentExecutor) |
| Tool definitions | ~30 lines | ~5 lines per @tool |
| Memory wiring | ~5 lines | ~10 lines (more setup, less runtime) |
| Total | ~160 lines | ~120 lines |

Savings are modest here. On a larger agent with 10+ tools, LangChain saves more.

## What LangChain does NOT replace

- Your tool implementations (the actual logic)
- Security: sanitizer, injection defence — always manual
- Your system prompt content
- Your business logic

## When to use LangChain

| Use it | Skip it |
|---|---|
| Prototyping quickly | Full production control needed |
| Switching LLM providers | Non-standard agent loop |
| Using built-in integrations | Team unfamiliar with the abstractions |
| LangGraph for complex multi-agent flows | Simple single-agent scripts |

## Key insight

LangChain is a productivity tool, not a magic box.
Now that you know what's inside the box, you can use it without being surprised.
