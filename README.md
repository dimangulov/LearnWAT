# WAT Framework & AI Workflows — Tutoring Repo

Lesson-by-lesson course on building AI agents with the Anthropic SDK, LangChain, and LangGraph.

## Setup

### 1. Create the virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 5. Run a lesson or prototype

```bash
# Lesson example
.venv/Scripts/python lesson-03/multi_tool.py

# Insurance prototype (LangGraph version)
.venv/Scripts/python insurance-prototype/main_lg.py
```

## Lesson index

| Lesson | Topic |
|---|---|
| 01 | Agent vs Tool |
| 02 | First tool call |
| 03 | Multiple tools + loop |
| 04 | System prompt |
| 05 | Conversation memory |
| 06 | Structured output |
| 07 | Multi-agent |
| 08 | Async parallel |
| 09 | Prompt injection |
| 10 | Full agent |
| 11 | LangChain |
| 12 | RAG |
| 13 | LangGraph |
| 14 | Observability |

## Insurance prototype

`insurance-prototype/` — virtual insurance agency that produces a quote through a 4-stage agent pipeline.

| File | Description |
|---|---|
| `main.py` | Original — plain Anthropic SDK, sequential stages |
| `main_lg.py` | LangGraph refactor — Pydantic state, graph topology |
| `main_lg_v2.py` | Advanced — tool calling, validation errors, interrupt |
