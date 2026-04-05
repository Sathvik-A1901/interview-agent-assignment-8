# Deep Learning RAG Interview Prep Agent

A retrieval-augmented interview prep app focused on deep learning topics. Drop in your study notes, ingest them into a local vector store (ChromaDB), and chat with an agent that answers with **source-grounded citations**.

Built with **LangChain + LangGraph + ChromaDB** and a simple **Streamlit UI**.

---

## Demo

Screen recording: `docs/demo.mp4` (H.264 `.mp4` recommended). Raw URL for reference:  
https://raw.githubusercontent.com/Sathvik-A1901/interview-agent-assignment-8/main/docs/demo.mp4

<video src="https://raw.githubusercontent.com/Sathvik-A1901/interview-agent-assignment-8/main/docs/demo.mp4" controls playsinline width="100%">
  Your browser does not support the video tag.
</video>

**Alternative — YouTube (no large file in git):** replace `VIDEO_ID` and use a thumbnail link:

```markdown
[![Watch the demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)
```

---

## What you can do

- **Ingest** markdown-based study notes from `data/corpus/`
- **Search / retrieve** relevant chunks for a question
- **Chat** with citations back to the originating document/chunk
- **Avoid hallucinations** by responding with “no relevant context found” when retrieval is weak

---

## Quickstart (Windows / PowerShell)

From the repo root:

```powershell
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m streamlit run .\src\rag_agent\ui\app.py
```

If you don’t already have the virtualenv set up, see the install section below.

---

## Installation

This project is managed via `pyproject.toml`. You can use **uv** (recommended) or plain `pip`.

### Option A: uv (recommended)

1. Install uv (Windows PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Install deps + create `.venv`:

```powershell
uv sync
```

3. Run Streamlit:

```powershell
uv run streamlit run .\src\rag_agent\ui\app.py
```

### Option B: pip + venv

If you prefer pip, create a venv and install from `pyproject.toml` (editable install):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run .\src\rag_agent\ui\app.py
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and pick one LLM provider.

### Groq

```text
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Ollama (local)

```text
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### LM Studio (local, OpenAI-compatible server)

```text
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
```

Other useful knobs:
- `CORPUS_DIR`: where your study notes live (default `./data/corpus`)
- `CHROMA_DB_PATH`: local vector DB directory (default `./data/chroma_db`)
- `RETRIEVAL_K`, `SIMILARITY_THRESHOLD`: retrieval behavior

---

## Adding your own corpus

1. Put markdown files in `data/corpus/` (examples included):
   - `data/corpus/ann_intermediate.md`
   - `data/corpus/cnn_intermediate.md`
   - `data/corpus/rnn_intermediate.md`
2. Start the app and ingest from the UI.

Notes:
- `data/chroma_db/` is intentionally gitignored (local-only).
- PDFs under `data/corpus/*.pdf` are gitignored by default.

---

## Running tests

With uv:

```bash
uv run pytest tests/ -v
```

With the project venv:

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\ -v
```

---

## Project structure

```
deep-learning-rag-agent/
├── data/
│   ├── corpus/                 ← markdown study notes live here
│   └── chroma_db/              ← local DB (gitignored)
├── docs/
│   ├── demo.mp4                ← README demo video (add locally, then commit)
│   ├── architecture.md
│   └── rubric.md
├── examples/
│   ├── hello_retrieval.py
│   └── sample_chunk.json
├── src/
│   └── rag_agent/
│       ├── config.py           ← settings + factories (LLM/embeddings)
│       ├── corpus/
│       │   └── chunker.py      ← chunking logic
│       ├── vectorstore/
│       │   └── store.py        ← ChromaDB manager (ingest/query/dupes)
│       ├── agent/
│       │   ├── state.py        ← data models
│       │   ├── prompts.py      ← prompt templates
│       │   ├── nodes.py        ← LangGraph nodes
│       │   └── graph.py        ← graph wiring
│       └── ui/
│           └── app.py          ← Streamlit UI
└── tests/
    └── test_vectorstore.py
```

---

## Common issues

**Streamlit reruns and loses state**
- Session objects should be cached and/or stored in `st.session_state`.

**“No relevant context found” too often**
- Add more corpus content, increase `RETRIEVAL_K`, or lower `SIMILARITY_THRESHOLD`.

**Ollama connection refused**
- Start Ollama first: `ollama serve`
