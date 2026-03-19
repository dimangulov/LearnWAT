# Lesson 12 — RAG: Retrieval-Augmented Generation

## The 4-step pipeline

```
LOAD → CHUNK → EMBED+STORE → RETRIEVE+ANSWER
```

This runs once at startup (or when documents change), then RETRIEVE+ANSWER
runs on every user question.

## Why chunk?

You can't embed a 100-page document as one vector — the meaning gets averaged
into mush. A chunk about "timeout configuration" should be close to the
question "what is the default timeout", not diluted by 99 other pages.

Rule of thumb: chunks of 300–500 characters with 50–100 overlap.

## What is an embedding?

A vector (list of floats) where similar meaning = similar numbers.

"timeout error"   → [0.2, 0.8, 0.1, ...]
"request timeout" → [0.2, 0.7, 0.1, ...]  ← close
"install widget"  → [0.9, 0.1, 0.6, ...]  ← far

The vector store finds chunks whose vectors are closest to the question vector.

## FakeEmbeddings vs real embeddings

| | FakeEmbeddings | VoyageAI / OpenAI |
|---|---|---|
| Vectors | Random | Semantically meaningful |
| Cost | Free | Per token |
| Retrieval quality | Random (demo only) | High |
| Use for | Learning the pipeline | Production |

For production with Claude, use VoyageAI embeddings (Anthropic's recommended
embedding provider): pip install langchain-voyageai

## The system prompt matters in RAG

"Answer using ONLY the provided context" prevents Claude from using its
training knowledge to fill gaps. Without this, Claude might answer correctly
but from its own knowledge — and you won't know if your documents actually
covered the question.

## What's Next
Lesson 13: LangGraph — agents as graphs, branching, human-in-the-loop.
