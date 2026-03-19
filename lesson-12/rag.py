"""
Lesson 12 — RAG: Retrieval-Augmented Generation

4 steps:
  1. LOAD     — read documents into text chunks
  2. EMBED    — convert chunks to vectors (numbers representing meaning)
  3. STORE    — save vectors in a vector database (Chroma, local)
  4. RETRIEVE — on each question, find the most similar chunks → send to Claude

We use:
  - TextLoader       (document loader)
  - RecursiveCharacterTextSplitter  (chunker)
  - ChromaDB         (local vector store — no API key needed)
  - Claude           (to answer using retrieved chunks)

No LangChain agent loop here — RAG is a pipeline, not a loop.
"""

import os
import anthropic
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings  # no API key needed for demo

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
# TextLoader reads the file, wraps it in Document objects.
# doc.page_content = text, doc.metadata = {"source": "path/to/file"}

print("Step 1: Loading documents...")
loader = TextLoader("documents/manual.txt", encoding="utf-8")
docs = loader.load()
print(f"  Loaded {len(docs)} document(s)")

# ── STEP 2: CHUNK ─────────────────────────────────────────────────────────────
# We can't embed the whole document as one vector — meaning gets averaged out.
# Split into small overlapping chunks so each chunk has focused meaning.
#
# chunk_size=300    characters per chunk
# chunk_overlap=50  overlap between chunks (prevents cutting a sentence mid-thought)

print("\nStep 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"  {len(docs)} document → {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  chunk {i}: {chunk.page_content[:60].strip()}...")

# ── STEP 3: EMBED + STORE ─────────────────────────────────────────────────────
# Embedding = converting text to a vector (list of floats).
# Similar meaning → similar vectors → close together in vector space.
#
# FakeEmbeddings: random vectors — good enough to demonstrate retrieval mechanics.
# In production: use VoyageAIEmbeddings or OpenAIEmbeddings for real semantic search.
#
# Chroma: local vector DB, stores to disk, no server needed.

print("\nStep 3: Embedding and storing in Chroma...")
embedding = FakeEmbeddings(size=384)  # swap for real embeddings in production
vectorstore = Chroma.from_documents(
    chunks,
    embedding=embedding,
    persist_directory="./chroma_db"   # saved to disk
)
print(f"  Stored {len(chunks)} chunks in Chroma")

# ── STEP 4: RETRIEVE + ANSWER ─────────────────────────────────────────────────
# On each question:
#   a) embed the question into a vector
#   b) find the k most similar chunk vectors (cosine similarity)
#   c) send question + chunks to Claude

client = anthropic.Anthropic()

def answer(question: str, k: int = 3):
    print(f"\nQuestion: {question}")

    # Retrieve top-k relevant chunks
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(r.page_content for r in results)

    print(f"  Retrieved {len(results)} chunks:")
    for r in results:
        print(f"    · {r.page_content[:60].strip()}...")

    # Send to Claude with retrieved context
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system="""You are a helpful assistant that answers questions using only
the provided document context. If the answer is not in the context, say so.
Do not make up information.""",
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }]
    )
    print(f"  Answer: {response.content[0].text}")

# ── Test questions ────────────────────────────────────────────────────────────

answer("How do I install Widget Pro?")
answer("What is the default timeout?")
answer("How do I fix a rate limit error?")
answer("Does it support TypeScript?")   # not in the docs — Claude should say so
