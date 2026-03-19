# Lesson 01 — What is WAT?

## Core Idea
WAT = Workflow Automation with Tools
AI agent + toolbox → achieves goals autonomously

## The Three Building Blocks

### 1. The Model (Brain)
- Claude reasons and plans
- Decides which tools to call
- Interprets tool results
- Knows when the goal is achieved

### 2. Tools (Hands)
- Functions the model can invoke
- Defined with JSON Schema (name, description, parameters)
- Examples: read_file, search_web, run_code, send_email

### 3. The Agent Loop
```
while goal_not_achieved:
    think → pick tool → call tool → observe result → think again
```

## Key Vocabulary
- **Tool call** — when the model invokes a function
- **Tool result** — the function's return value fed back to the model
- **Agent** — model + tools + loop
- **Subagent** — a nested agent spawned to handle a subtask
- **Workflow** — a sequence or graph of agent tasks

## WAT vs. Traditional Code

| Traditional | WAT |
|-------------|-----|
| You write every step | Agent decides the steps |
| Logic is in your code | Logic is in the model |
| Brittle to variation | Handles variation naturally |
| Hard to extend | Add a tool, agent adapts |

## What's Next
Lesson 02: Writing your first tool and calling Claude with it.
