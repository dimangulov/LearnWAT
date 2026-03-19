# Lesson 07 — Multi-Agent: One Claude Orchestrating Others

## The pattern

```
User
 └── Orchestrator (Claude)
       ├── calls worker_A as a tool  →  Worker A (Claude)
       ├── calls worker_B as a tool  →  Worker B (Claude)
       └── assembles results into final answer
```

Workers are just functions that return strings.
The orchestrator sees them as tools — same API as Lessons 03–06.
Nothing new in the plumbing. Only the workers call Claude too.

## Why split into multiple agents?

| One agent | Multi-agent |
|---|---|
| One system prompt for everything | Each worker has a focused prompt |
| One model for everything | Use cheap model for workers, smart for orchestrator |
| Context grows with every subtask | Each worker has a small, clean context |
| Sequential by default | Workers can run in parallel (next lesson) |

## Model assignment strategy

| Role | Model choice | Why |
|---|---|---|
| Orchestrator | claude-sonnet / opus | Needs to plan and reason |
| Simple workers | claude-haiku | Fast, cheap, single-purpose |
| Structured output | claude-haiku | Schema does the heavy lifting |

## Trust between agents

Workers trust the orchestrator's inputs — they don't validate.
If the orchestrator is compromised (prompt injection), workers will execute bad inputs.
Always sanitize inputs at worker boundaries in production.

## What's Next
Lesson 08: Parallel workers — running all three workers at the same time with threads.
