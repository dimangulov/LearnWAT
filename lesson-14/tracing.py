"""
Lesson 14 — Debugging, Tracing, Monitoring, Testing

Four layers of observability for AI agents:

  1. stream()       — step-by-step node execution (LangGraph built-in)
  2. callbacks      — hooks into every LLM call (LangChain built-in)
  3. LangSmith      — cloud tracing dashboard (optional, free tier)
  4. testing        — deterministic tests for non-deterministic agents

Run this file to see all four in action on the Lesson 13 graph.
"""

import os
import json
from typing import Annotated, Literal, Any
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — stream(): see state after every node
# ─────────────────────────────────────────────────────────────────────────────
# graph.invoke()  → runs to completion, returns final state
# graph.stream()  → yields state after each node, then final state
#
# Use stream() when debugging — you see exactly which node ran and what changed.

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Callbacks: hooks into every LLM call
# ─────────────────────────────────────────────────────────────────────────────
# LangChain fires callbacks at key moments:
#   on_llm_start   — before sending to Claude
#   on_llm_end     — after receiving from Claude
#   on_tool_start  — before running a tool
#   on_tool_end    — after tool returns
#
# Subclass BaseCallbackHandler to intercept any of these.

class DebugCallbackHandler(BaseCallbackHandler):
    """Logs every LLM call and tool call with token counts."""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        print(f"\n  [callback] LLM call — {len(prompts)} prompt(s)")
        for p in prompts:
            print(f"    prompt: {str(p)[:80]}...")

    def on_llm_end(self, response: Any, **kwargs):
        # token usage is in response.llm_output
        usage = getattr(response, "llm_output", {}) or {}
        tokens = usage.get("token_usage", {})
        print(f"  [callback] LLM done — tokens: {tokens}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        print(f"  [callback] tool start: {serialized.get('name')} ← {input_str[:60]}")

    def on_tool_end(self, output: str, **kwargs):
        print(f"  [callback] tool end  → {str(output)[:60]}")

debug_handler = DebugCallbackHandler()

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — LangSmith: cloud tracing (optional)
# ─────────────────────────────────────────────────────────────────────────────
# LangSmith records every run — inputs, outputs, latency, tokens, cost.
# Free tier at smith.langchain.com
#
# To enable, set these env vars (no code changes needed):
#
#   export LANGCHAIN_TRACING_V2=true
#   export LANGCHAIN_API_KEY=ls__your_key
#   export LANGCHAIN_PROJECT=my-agent
#
# Every graph.invoke() / graph.stream() call then appears in the dashboard.
# You can replay runs, compare runs, and set up alerts.
#
# Check if it's active:

LANGSMITH_ACTIVE = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
print(f"LangSmith tracing: {'ON' if LANGSMITH_ACTIVE else 'OFF (set LANGCHAIN_TRACING_V2=true to enable)'}")

# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — Testing: deterministic tests for non-deterministic agents
# ─────────────────────────────────────────────────────────────────────────────
# Agents are hard to unit test — Claude's output varies.
# Three strategies:
#
#   a) Mock the LLM — replace Claude with a deterministic fake
#   b) Test the graph structure — assert nodes and edges exist
#   c) Test node functions in isolation — each node is just a function

# ─────────────────────────────────────────────────────────────────────────────
# Rebuild the Lesson 13 graph
# ─────────────────────────────────────────────────────────────────────────────

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    callbacks=[debug_handler]   # attach callbacks to the model
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    approved: bool

def classify(state: State) -> dict:
    last = state["messages"][-1].content
    response = model.invoke(
        f'Classify as "answer", "research", or "sensitive". Reply one word.\nRequest: {last}'
    )
    route = response.content.strip().lower()
    if route not in ("answer", "research", "sensitive"):
        route = "answer"
    return {"route": route}

def answer_node(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def search_node(state: State) -> dict:
    question = state["messages"][-1].content
    results = f"Search results for '{question}': [result1] [result2]"
    response = model.invoke(
        state["messages"] + [{"role": "user", "content": f"Use these results:\n{results}"}]
    )
    return {"messages": [response]}

def route_after_classify(state: State) -> Literal["answer", "research", "sensitive"]:
    return state["route"]

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("answer",   answer_node)
builder.add_node("search",   search_node)
builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_after_classify,
    {"answer": "answer", "research": "search", "sensitive": "answer"})
builder.add_edge("answer", END)
builder.add_edge("search", END)

graph = builder.compile(checkpointer=MemorySaver())

# ─────────────────────────────────────────────────────────────────────────────
# Demo 1: stream() — see every node as it runs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DEMO 1: stream() — step-by-step execution trace")
print("="*60)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("  SKIPPED — set ANTHROPIC_API_KEY to run this demo")
else:
    config = {"configurable": {"thread_id": "debug-1"}}
    question = "What is the capital of France?"

    for step in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config
    ):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        print(f"\n[node: {node_name}]")
        if "route" in node_output:
            print(f"  route = {node_output['route']}")
        if "messages" in node_output:
            for m in node_output["messages"]:
                content = getattr(m, "content", str(m))
                print(f"  message: {str(content)[:100]}")

# ─────────────────────────────────────────────────────────────────────────────
# Demo 2: test node functions in isolation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DEMO 2: testing node functions in isolation")
print("="*60)

from langchain_core.messages import HumanMessage

class FakeModel:
    """Drop-in replacement for ChatAnthropic in tests. No API calls, no cost."""
    def __init__(self, response: str):
        self._response = response

    def invoke(self, *args, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._response)

def make_classify(fake_model):
    """Return a classify function that uses the given model — enables injection."""
    def _classify(state: State) -> dict:
        last = state["messages"][-1].content
        response = fake_model.invoke(
            f'Classify as "answer", "research", or "sensitive". Reply one word.\nRequest: {last}'
        )
        route = response.content.strip().lower()
        if route not in ("answer", "research", "sensitive"):
            route = "answer"
        return {"route": route}
    return _classify

def test_classify_routes_correctly():
    """classify() should return 'answer' for a simple factual question."""
    _classify = make_classify(FakeModel("answer"))
    state = {"messages": [HumanMessage(content="What is 2+2?")], "route": "", "approved": False}
    result = _classify(state)
    assert result["route"] == "answer", f"Expected 'answer', got '{result['route']}'"
    print("  PASS: classify routes simple question to 'answer'")

def test_classify_rejects_unknown_routes():
    """classify() should fall back to 'answer' if model returns unexpected value."""
    _classify = make_classify(FakeModel("unknown_route"))   # bad model output
    state = {"messages": [HumanMessage(content="anything")], "route": "", "approved": False}
    result = _classify(state)
    assert result["route"] == "answer", "Should fall back to 'answer'"
    print("  PASS: classify falls back to 'answer' on unexpected model output")

def test_graph_has_required_nodes():
    """Graph structure test — no LLM calls needed."""
    nodes = list(graph.get_graph().nodes.keys())
    for required in ["classify", "answer", "search"]:
        assert required in nodes, f"Missing node: {required}"
    print(f"  PASS: graph contains required nodes: {[n for n in nodes if not n.startswith('__')]}")

test_classify_routes_correctly()
test_classify_rejects_unknown_routes()
test_graph_has_required_nodes()

print("\nAll tests passed.")
