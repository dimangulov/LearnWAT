"""
LangGraph visualization — two options:

1. ASCII diagram (always works, no dependencies)
2. PNG image (requires pygraphviz — hard to install on Windows)

We'll use the ASCII version.
"""

# Re-build the graph from graph_agent.py (same code, no REPL)
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    approved: bool

def classify(state): pass
def answer_node(state): pass
def search_node(state): pass
def human_review(state): pass
def act_node(state): pass
def cancel_node(state): pass
def route_after_classify(state) -> Literal["answer", "research", "sensitive"]:
    return state["route"]
def route_after_review(state) -> Literal["act", "cancel"]:
    return "act" if state.get("approved") else "cancel"

builder = StateGraph(State)
builder.add_node("classify",     classify)
builder.add_node("answer",       answer_node)
builder.add_node("search",       search_node)
builder.add_node("human_review", human_review)
builder.add_node("act",          act_node)
builder.add_node("cancel",       cancel_node)
builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_after_classify,
    {"answer": "answer", "research": "search", "sensitive": "human_review"})
builder.add_conditional_edges("human_review", route_after_review,
    {"act": "act", "cancel": "cancel"})
builder.add_edge("answer", END)
builder.add_edge("search", END)
builder.add_edge("act",    END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=MemorySaver())

# ── Option 1: ASCII ───────────────────────────────────────────────────────────

print("=== ASCII Graph ===\n")
graph.get_graph().print_ascii()

# ── Option 2: PNG (uncomment if pygraphviz is available) ──────────────────────

# png = graph.get_graph().draw_mermaid_png()
# with open("graph.png", "wb") as f:
#     f.write(png)
# print("Saved graph.png")

# ── Option 3: Mermaid source (paste into mermaid.live) ───────────────────────

print("\n=== Mermaid source (paste at mermaid.live) ===\n")
print(graph.get_graph().draw_mermaid())
