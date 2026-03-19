# Lesson 13 — LangGraph: Agents as Graphs

## Core concepts

| Concept | What it is |
|---|---|
| **State** | Shared dict that every node reads and writes |
| **Node** | A function: receives state, returns updates |
| **Edge** | A fixed connection between two nodes |
| **Conditional edge** | A function that decides which node to go to next |
| **Entry point** | The first node to run |
| **END** | Terminal node — graph stops here |

## The graph we built

```
[classify]
    ├── "answer"    → [answer]       → END
    ├── "research"  → [search]       → END
    └── "sensitive" → [human_review]
                          ├── approved  → [act]    → END
                          └── rejected  → [cancel] → END
```

## How this differs from our manual loop

| Manual loop (L03–L10) | LangGraph |
|---|---|
| One path: tool_use → end_turn | Multiple branching paths |
| Can't pause and wait | human_review node pauses execution |
| State lives in messages list | State is a typed dict, any shape |
| Loop logic in your code | Graph structure IS the logic |

## State is the key idea

Every node receives the full state and returns only what changed.
Nodes don't call each other — they only read/write state.
The graph engine decides what runs next based on edges.

## When to use LangGraph vs a simple loop

Use a loop when: one path, tools only, no branching needed.
Use LangGraph when: branching, human approval, complex multi-step workflows,
or you need to visualize/audit the execution path.

## What's Next
Lesson 14: Subagents — Claude spawning and directing other Claude instances.
