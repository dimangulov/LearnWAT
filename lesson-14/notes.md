# Lesson 14 — Debugging, Tracing, Monitoring, Testing

## The four layers

```
Layer 1: stream()     — dev time, local, step-by-step
Layer 2: callbacks    — dev time, local, per LLM call
Layer 3: LangSmith    — production, cloud, full history
Layer 4: testing      — CI/CD, deterministic, no LLM calls
```

Use all four. They catch different things.

## Layer 1 — stream()

```python
for step in graph.stream(inputs, config):
    node_name   = list(step.keys())[0]
    node_output = step[node_name]
```

Use when: agent takes wrong path, wrong tool called, unexpected output.
Tells you: which node ran, in what order, what state changed.

## Layer 2 — Callbacks

```python
class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, ...): ...   # before Claude
    def on_llm_end(self, ...):   ...   # after Claude, has token counts
    def on_tool_start(self, ...): ...  # before tool
    def on_tool_end(self, ...):  ...   # after tool

model = ChatAnthropic(..., callbacks=[MyHandler()])
```

Use when: tracking token usage, latency, cost per call.
Tells you: exactly what was sent to Claude and what came back.

## Layer 3 — LangSmith

Set env vars — no code changes:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_PROJECT=my-project
```

Dashboard at smith.langchain.com shows:
- Full input/output for every node
- Token usage and cost per run
- Latency breakdown
- Run history and comparison
- Alerts on errors or regressions

Free tier: 5,000 traces/month.

## Layer 4 — Testing strategies

| Strategy | LLM calls | What it tests |
|---|---|---|
| Mock the model | No | Node logic, routing, fallbacks |
| Graph structure | No | Nodes and edges exist |
| Integration test | Yes (costs money) | Full end-to-end behavior |

Mock the model for fast, free, deterministic CI tests.
Run integration tests sparingly — only before releases.

## The key testing insight

Nodes are just functions. Test them like functions:
```python
state = {"messages": [...], "route": "", "approved": False}
result = classify(state)
assert result["route"] == "answer"
```

No graph, no LangChain, no Claude needed.

## What's Next
Lesson 15: Subagents — Claude spawning and directing other Claude instances.
