"""
Lesson 12b — RAG simplified with LangChain chains

Compared to rag.py:
  - Steps 1-3 (load, chunk, store) are identical
  - Step 4 (retrieve + answer) collapses into a chain

The chain wires together:
  retriever | prompt | model | output parser
in a single pipeline you call with .invoke()
"""

import anthropic
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Steps 1-3: identical to rag.py ───────────────────────────────────────────

chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
).split_documents(
    TextLoader("documents/manual.txt", encoding="utf-8").load()
)

vectorstore = Chroma.from_documents(
    chunks,
    FakeEmbeddings(size=384),
    persist_directory="./chroma_db_simple"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Step 4: retrieve + answer as a chain ─────────────────────────────────────
# The | operator pipes output of one step into the next.
#
# RunnablePassthrough() — passes the question through unchanged
# retriever             — fetches relevant chunks for the question
# prompt                — formats question + chunks into a message
# model                 — sends to Claude
# StrOutputParser()     — extracts the text string from the response

prompt = ChatPromptTemplate.from_template("""
Answer using only the context below. If the answer isn't there, say so.

Context:
{context}

Question: {question}
""")

model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

# The chain — read left to right
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# ── Ask questions ─────────────────────────────────────────────────────────────

questions = [
    "How do I install Widget Pro?",
    "What is the default timeout?",
    "Does it support TypeScript?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}\n")
