"""
Lesson 13 — LangGraph: Agents as Graphs

We build a research agent with three paths:
  - safe questions   → answer directly
  - research needed  → search tool → answer
  - sensitive action → human approval → act or cancel

Graph structure:
  [user input]
      │
      ▼
  [classify]  ─── "answer"    ──► [answer] ──► END
              ─── "research"  ──► [search] ──► [answer] ──► END
              ─── "sensitive" ──► [human_review] ──► (approved?) ──► [act] ──► END
                                                                  ──► [cancel] ──► END
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)

# ── State ─────────────────────────────────────────────────────────────────────
# Every node reads from and writes to this shared state.
# add_messages: appends new messages rather than overwriting.

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str          # which path the classifier chose
    approved: bool      # human approval decision

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [Wikipedia overview] [Academic paper] [News article]"

# ── Nodes — each is a function that receives state and returns updates ─────────

def classify(state: State) -> dict:
    """Decide which path to take based on the user's message."""
    last_message = state["messages"][-1].content
    response = model.invoke(
        f"""Classify this request into exactly one word:
- "answer"    if it's a general question you can answer directly
- "research"  if it needs web search
- "sensitive" if it involves deleting data, sending emails, or payments

Request: {last_message}

Reply with only one word."""
    )
    route = response.content.strip().lower()
    if route not in ("answer", "research", "sensitive"):
        route = "answer"   # fallback
    print(f"  [classify] → {route}")
    return {"route": route}

def answer_node(state: State) -> dict:
    """Answer directly from Claude's knowledge."""
    response = model.invoke(state["messages"])
    print(f"  [answer] {response.content[:80]}")
    return {"messages": [response]}

def search_node(state: State) -> dict:
    """Run a search then answer."""
    question = state["messages"][-1].content
    results = search.invoke({"query": question})
    response = model.invoke(
        state["messages"] + [{
            "role": "user",
            "content": f"Use these search results to answer:\n{results}"
        }]
    )
    print(f"  [search+answer] {response.content[:80]}")
    return {"messages": [response]}

def human_review(state: State) -> dict:
    """Pause and ask a human for approval."""
    print("\n  [human_review] ⚠ Sensitive action requested.")
    print(f"  Request: {state['messages'][-1].content}")
    decision = input("  Approve? (yes/no): ").strip().lower()
    return {"approved": decision == "yes"}

def act_node(state: State) -> dict:
    """Carry out the sensitive action (only reached if approved)."""
    response = model.invoke(
        state["messages"] + [{
            "role": "user",
            "content": "The action was approved. Confirm it has been carried out."
        }]
    )
    print(f"  [act] {response.content[:80]}")
    return {"messages": [response]}

def cancel_node(state: State) -> dict:
    """Tell the user the action was cancelled."""
    from langchain_core.messages import AIMessage
    msg = AIMessage(content="Action cancelled — not approved.")
    print("  [cancel]")
    return {"messages": [msg]}

# ── Routing functions — decide which edge to follow ───────────────────────────

def route_after_classify(state: State) -> Literal["answer", "research", "sensitive"]:
    return state["route"]

def route_after_review(state: State) -> Literal["act", "cancel"]:
    return "act" if state["approved"] else "cancel"

# ── Build the graph ───────────────────────────────────────────────────────────

builder = StateGraph(State)

# Add nodes
builder.add_node("classify",     classify)
builder.add_node("answer",       answer_node)
builder.add_node("search",       search_node)
builder.add_node("human_review", human_review)
builder.add_node("act",          act_node)
builder.add_node("cancel",       cancel_node)

# Entry point
builder.set_entry_point("classify")

# Conditional edges from classify
builder.add_conditional_edges("classify", route_after_classify, {
    "answer":    "answer",
    "research":  "search",
    "sensitive": "human_review",
})

# Conditional edge after human review
builder.add_conditional_edges("human_review", route_after_review, {
    "act":    "act",
    "cancel": "cancel",
})

# Terminal edges
builder.add_edge("answer", END)
builder.add_edge("search", END)
builder.add_edge("act",    END)
builder.add_edge("cancel", END)

# Compile with memory (state persists across turns)
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ── REPL ──────────────────────────────────────────────────────────────────────

CONFIG = {"configurable": {"thread_id": "user-1"}}

print("LangGraph agent. Type 'quit' to exit.\n")
print("Try:")
print("  'What is the capital of France?'  → answer path")
print("  'Search for Python asyncio'       → research path")
print("  'Delete my account'               → sensitive path\n")

while True:
    user_input = input("You: ").strip()
    if not user_input or user_input == "quit":
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=CONFIG
    )
    final = result["messages"][-1].content
    print(f"Claude: {final}\n")
