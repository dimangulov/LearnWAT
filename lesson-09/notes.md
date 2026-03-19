# Lesson 09 — Prompt Injection & Agent Security

## What is prompt injection?

Malicious instructions hidden in data the agent reads.
The agent can't tell the difference between your instructions and injected ones
— unless you build defences.

## Where it comes from

Any untrusted external data that enters the agent's context:
- Webpages, PDFs, emails, documents
- Database rows from user-submitted content
- API responses from third-party services
- Other agents' outputs (in multi-agent systems)

## Defence layers (use all of them)

| Layer | What it does | Limits |
|---|---|---|
| System prompt rules | Tells Claude to distrust tool output | Claude can still be fooled |
| Output sanitizer | Strips known injection patterns before Claude sees them | Can't catch everything |
| Minimal tool permissions | Agent can only read, not send email/delete files | Limits blast radius |
| Human-in-the-loop | Confirm before irreversible actions | Slows automation |

## The key mental model

```
Trusted:    system prompt (you wrote it)
Trusted:    user messages (your user sent them)
UNTRUSTED:  everything that comes back from tools
```

Always treat tool output as untrusted external data.
Never let it override your system prompt.

## The sanitizer is not enough alone

Pattern matching only catches known attacks.
A clever attacker uses synonyms, encoding, or indirect phrasing.
The system prompt + sanitizer + minimal permissions together raise the bar enough
to stop opportunistic attacks — targeted attacks need human review.

## What's Next
Lesson 10: Putting it all together — a real mini-agent with memory, tools, structured output, parallel workers, and security.
