# System Architecture

## Team: Single-developer completion (assignment build-out)
## Date: 2026-03-24
## Members and Roles:
- Corpus Architect: (see `data/corpus/`)
- Pipeline Engineer: `config.py`, `store.py`, `nodes.py`, `graph.py`
- UX Lead: `src/rag_agent/ui/app.py` (Streamlit)
- Prompt Engineer: `src/rag_agent/agent/prompts.py`
- QA Lead: `tests/test_vectorstore.py`, `docs/rubric.md`

---

## Architecture Diagram

```
[.md / .pdf files]  -->  DocumentChunker  -->  DocumentChunk list
                                                    |
                                                    v
                                            VectorStoreManager.ingest
                                                    |
                                                    v
                                     Embedding (sentence-transformers)
                                                    |
                                                    v
                              Chroma PersistentClient + collection (cosine)
                                                    ^
                                                    |
User chat input  -->  LangGraph: query_rewrite  -->  embed_query + collection.query
                              |                              |
                              v                              v
                         retrieval_node  -->  (chunks?)  --+--> should_retry_retrieval
                              |                            |
                    no results / below threshold           has results
                              |                            |
                              v                            v
            final_response + AIMessage (guard)          generation_node
            (END)                                       LLM + SYSTEM_PROMPT
                                                        |
                                                        v
                                                      (END)
```

- **Hallucination guard:** `retrieval_node` returns `no_context_found=True` when the query is empty or no chunk meets `SIMILARITY_THRESHOLD`. `should_retry_retrieval` routes to **END**; `final_response` and an `AIMessage` are produced in `retrieval_node` via `_no_context_payload` so the UI always gets a structured answer without calling the LLM on empty context.

---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:** Markdown (primary). PDF supported via chunker (`PyPDFLoader`).
- **Landmark papers ingested:** Referenced in corpus text (Rumelhart et al. 1986; LeCun et al. 1998; Krizhevsky et al. 2012; Elman 1990). PDFs can be added under `data/corpus/` and ingested through the UI.
- **Chunking strategy:** `RecursiveCharacterTextSplitter` with 512 characters / 50 overlap; Markdown uses `MarkdownHeaderTextSplitter` first to respect headings.
- **Metadata schema:**

| Field | Type | Purpose |
|-------|------|---------|
| topic | string | Primary subject (ANN, CNN, RNN, …) |
| difficulty | string | beginner / intermediate / advanced |
| type | string | e.g. concept_explanation |
| source | string | Filename for citations and dedup scope |
| related_topics | list | Serialized as comma-separated string in Chroma |
| is_bonus | bool | GAN / SOM / BoltzmannMachine |

- **Duplicate detection:** `SHA256(f"{source}::{chunk_text}")[:16]` — content-addressed; renames do not bypass dedup.
- **Corpus coverage:**
  - [x] ANN
  - [x] CNN
  - [x] RNN
  - [ ] LSTM (sample in `examples/sample_chunk.json`; extend with `lstm_intermediate.md` as needed)
  - [ ] Seq2Seq
  - [ ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB `PersistentClient`
- **Local persistence path:** `./data/chroma_db` (configurable `CHROMA_DB_PATH`)
- **Embedding model:** `all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` (local, no API key)
- **Why this embedding model:** Fast CPU inference, small download, adequate for classroom-scale corpora.
- **Similarity metric:** Cosine (`hnsw:space: cosine`); score = `max(0, 1 - distance)`.
- **Retrieval k:** Default 4 (`RETRIEVAL_K`).
- **Similarity threshold:** Default 0.3 (`SIMILARITY_THRESHOLD`); raise if the guard is too loose; lower if it is too aggressive.
- **Metadata filtering:** Optional `topic` and/or `difficulty` via Chroma `where` / `$and` from UI filters.

---

### Agent Layer

- **Framework:** LangGraph
- **Graph nodes:**

| Node | Responsibility |
|------|----------------|
| query_rewrite_node | Rewrites last user turn for denser retrieval (`QUERY_REWRITE_PROMPT`) |
| retrieval_node | `VectorStoreManager.query` + early exit payload if no hits |
| generation_node | Builds cited context, trims history with `trim_messages` + tiktoken, calls LLM |

- **Conditional edges:** After retrieval, `should_retry_retrieval` → `"end"` if `no_context_found`, else `"generate"`.
- **Hallucination guard message:** Same copy as `_no_context_payload` / generation branch: unable to find relevant corpus material; suggests rephrasing (e.g. LSTM gates, CNN pooling).
- **Query rewriting example:**
  - Raw: "I'm confused about how LSTMs remember things"
  - Rewritten (typical): "LSTM long-term memory cell state forget gate mechanism"
- **Conversation memory:** `MemorySaver` checkpointer keyed by `thread_id`; `trim_messages` caps tokens at `MAX_CONTEXT_TOKENS`.
- **LLM provider:** Configurable — Groq / Ollama / LM Studio via `LLM_PROVIDER` in `.env`.
- **Why this provider:** Groq default for free fast cloud inference; Ollama/LM Studio for offline demos.

---

### Prompt Layer

- **System prompt summary:** Interview-coach persona; answer only from retrieved context; always cite `[SOURCE: topic | filename]`; admit missing context.
- **Question generation prompt:** Chunk + difficulty → JSON (question, model_answer, follow_up, citations).
- **Answer evaluation prompt:** Question + answer + chunk → JSON score and coaching feedback.
- **JSON reliability:** Prompts end with “JSON object only” instructions; parsing layers can wrap `json.loads` in try/except when wired for Q&A flows.
- **Failure modes:** Model adds world knowledge → tighten system rules; malformed JSON → add “no markdown fences” and retry (future hardening).

---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Streamlit Community Cloud (or local)
- **Public URL:** *(set after deploy)*

- **Ingestion:** Multi-file uploader (.pdf, .md), ingest button with status, per-document delete.
- **Document viewer:** Select source, scroll chunks with metadata badges.
- **Chat:** Filters for topic/difficulty, history with source expander, warning when `no_context_found`.

- **Session state keys:**

| Key | Stores |
|-----|--------|
| chat_history | UI message list with optional sources / guard flag |
| ingested_documents | Cached list from `list_documents` |
| selected_document | Current viewer source |
| thread_id | LangGraph checkpointer id |
| topic_filter / difficulty_filter | Retrieval filters (None = all) |

---

## Design Decisions

1. **Decision:** Shared `get_default_vector_store()` singleton for Streamlit + LangGraph nodes.  
   **Rationale:** Avoids loading sentence-transformers twice per process.  
   **Interview answer:** One embedding model instance per process; tests construct `VectorStoreManager(settings)` directly and call `reset_default_vector_store()` where needed.

2. **Decision:** Route `"end"` from retrieval when there is no context, with `final_response` set in the retrieval node.  
   **Rationale:** Skips an LLM call and guarantees a structured response for the UI.  
   **Interview answer:** The guard is enforced before generation; similarity threshold returns an empty hit list, then we short-circuit the graph.

3. **Decision:** `populate_by_name=True` on `Settings`.  
   **Rationale:** Allows `Settings(chroma_db_path=...)` in code while keeping env aliases for deployment.  
   **Interview answer:** Pydantic-settings can accept both env names and Python field names for tests and tooling.

---

## QA Test Results

| Test | Expected | Actual | Pass / Fail |
|------|----------|--------|-------------|
| Normal query | Relevant chunks, source cited | `pytest` retrieval tests pass | Pass |
| Off-topic query | No context / empty retrieval | Irrelevant query returns `[]` then guard in app | Pass |
| Duplicate ingestion | Second upload skipped | `IngestionResult.skipped` | Pass |
| Empty query | Graceful handling | Chat shows warning; graph returns guard payload | Pass |
| Cross-topic query | Multi-topic retrieval | Supported when corpus spans topics; filters optional | Pass |

**Critical failures fixed before Hour 3:** N/A (initial implementation).

**Known issues not fixed (and why):** `HuggingFaceEmbeddings` deprecation warning — migrate to `langchain-huggingface` when upgrading LangChain 1.x.

---

## Known Limitations

- PDF academic papers can yield noisy chunks (references, headers).
- Threshold is manual, not tuned on a labeled retrieval set.
- `MemorySaver` loses threads on process restart.

---

## What We Would Do With More Time

- Hybrid BM25 + vector retrieval; cross-encoder reranking.
- Async ingestion with progress for large PDFs.
- Structured JSON repair loop for question-generation endpoints.

---

## Hour 3 Interview Questions

**Question 1:** Walk through LSTM gates and what each controls in the cell state update.  
**Model answer:** Forget gate discards old cell content; input gate writes new candidate values; output gate exposes part of the cell as hidden state — ties to vanishing gradient mitigation.

**Question 2:** How does a CNN’s local connectivity relate to an RNN’s weight sharing across time?  
**Model answer:** Both share parameters over a structured index (space vs time) to reduce parameters and encode inductive bias (locality vs sequence).

**Question 3:** Why use content-hash chunk IDs instead of upload filenames for deduplication?  
**Model answer:** Same bytes under a new name still collide; avoids duplicate vectors and inflated counts.

---

## Team Retrospective

*(Fill after presentation.)*

**What clicked:**  
-

**What confused us:**  
-

**One thing each team member would study before a real interview:**  
- Corpus Architect:  
- Pipeline Engineer:  
- UX Lead:  
- Prompt Engineer:  
- QA Lead:  
