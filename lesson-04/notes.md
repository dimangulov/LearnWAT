# Lesson 04 — System Prompts: Giving Your Agent a Role

## What is a system prompt?

A string passed as `system=` in every API call.
Claude reads it before reading any user message.
It is the highest-authority instruction in the conversation.

## Priority order (highest → lowest)

1. System prompt
2. User message
3. Claude's defaults

If the system prompt says "never discuss X", that overrides any user request about X.

## What to put in a system prompt

| Category | Example |
|---|---|
| Role | "You are a customer support agent for Acme Corp" |
| Constraints | "Never reveal internal pricing. Never discuss competitors." |
| Output format | "Always respond in JSON. Never use markdown." |
| Tone | "Be formal and concise. No emoji." |
| Tool guidance | "Always look up live data before answering. Never guess." |

## What NOT to put in a system prompt

- Long examples → use few-shot messages instead
- Dynamic data (user name, current date) → inject into the user message
- Secret keys or passwords → they are NOT secret, treat as visible

## Key insight

The same tools + same loop + different system prompt = different agent.
This means you can build one codebase and deploy many specialized agents
by only changing the system prompt.

## What's Next
Lesson 05: Conversation memory — keeping context across multiple user turns.
