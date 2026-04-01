"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager, get_default_vector_store


def _inject_app_styles() -> None:
    """Global typography, color tokens, and Streamlit component polish."""
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

  /* Do not use [class*="css"] — it matches Streamlit's hashed classes and breaks theme colours. */
  .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
  code, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

  .rag-hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 55%, #4f46e5 100%);
    padding: 1.35rem 1.5rem 1.25rem;
    border-radius: 14px;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 40px rgba(49, 46, 129, 0.35);
  }
  .rag-hero h1 {
    color: #f8fafc !important;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    font-size: 1.65rem;
  }
  .rag-hero p {
    color: #c7d2fe !important;
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.45;
  }

  /*
   * Sidebar: force light surface + dark text so widgets stay readable when the app
   * theme is dark (otherwise Streamlit keeps light-on-light labels).
   */
  [data-testid="stSidebar"] {
    color-scheme: light;
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
    border-right: 1px solid #c7d2fe !important;
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] li,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] h4,
  [data-testid="stSidebar"] h5,
  [data-testid="stSidebar"] h6,
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] [data-testid="stCaption"] {
    color: #0f172a !important;
  }
  [data-testid="stSidebar"] [data-testid="stMetricValue"],
  [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #0f172a !important;
  }
  [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
    color: #047857 !important;
  }
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] [data-testid="stCaption"] {
    color: #475569 !important;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] > div,
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
  }
  [data-testid="stSidebar"] .stAlert {
    color: #0f172a !important;
  }
  [data-testid="stSidebar"] button[kind="primary"],
  [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
    color: #ffffff !important;
  }
  [data-testid="stSidebar"] button[kind="secondary"] {
    color: #0f172a !important;
  }

  .rag-source-pill {
    display: inline-block;
    background: #eef2ff;
    color: #312e81;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    margin: 0.15rem 0.25rem 0.15rem 0;
    border: 1px solid #c7d2fe;
  }

  div[data-testid="stChatMessage"] { background: rgba(255,255,255,0.65); border-radius: 12px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _fields_from_final_response(fr: object) -> tuple[str, list[str], bool, float]:
    """
    LangGraph/checkpointers may return ``final_response`` as an ``AgentResponse``,
    a ``dict``, or a read-only ``Mapping``. Normalise for the chat UI.

    Returns
    -------
    content, sources, no_context_found, confidence
    """
    if fr is None:
        return "", [], True, 0.0
    if isinstance(fr, Mapping):
        answer = fr.get("answer")
        content = answer if isinstance(answer, str) else str(answer or "")
        sources_raw = fr.get("sources") or []
        sources = list(sources_raw) if isinstance(sources_raw, list) else []
        conf = fr.get("confidence")
        c = float(conf) if isinstance(conf, (int, float)) else 0.0
        return content, [str(s) for s in sources], bool(fr.get("no_context_found")), c
    answer = getattr(fr, "answer", "")
    content = answer if isinstance(answer, str) else str(answer)
    sources = list(getattr(fr, "sources", []) or [])
    conf = float(getattr(fr, "confidence", 0.0) or 0.0)
    return (
        content,
        [str(s) for s in sources],
        bool(getattr(fr, "no_context_found", False)),
        conf,
    )


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """Return the singleton VectorStoreManager."""
    return get_default_vector_store()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """Initialise ``st.session_state`` keys on first run."""
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "thread_id": "default-session",
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _new_conversation() -> None:
    """Reset chat and start a fresh LangGraph thread."""
    st.session_state.chat_history = []
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    st.sidebar.markdown("### Corpus")
    st.sidebar.caption("Upload `.md` or `.pdf` study files. Chunks are embedded locally.")

    uploaded = st.sidebar.file_uploader(
        "Add files",
        type=["pdf", "md"],
        accept_multiple_files=True,
        help="Files are chunked, embedded with sentence-transformers, and stored in ChromaDB.",
    )

    ingest = st.sidebar.button(
        "Ingest into vector store",
        disabled=not uploaded,
        type="primary",
        use_container_width=True,
    )

    if ingest:
        with st.sidebar.status("Processing…", expanded=True) as status:
            tmp = Path(tempfile.mkdtemp(prefix="rag_upload_"))
            try:
                paths: list[Path] = []
                for f in uploaded:
                    p = tmp / f.name
                    p.write_bytes(f.getbuffer())
                    paths.append(p)
                status.update(label="Chunking…", state="running")
                chunks = chunker.chunk_files(paths)
                status.update(label="Embedding & storing…", state="running")
                result = store.ingest(chunks)
                st.session_state.last_ingestion_result = result
                st.session_state.ingested_documents = store.list_documents()
                msg = (
                    f"Added **{result.ingested}** chunks · skipped **{result.skipped}** duplicates · "
                    f"**{len(result.errors)}** errors"
                )
                if result.errors:
                    status.update(label="Completed with errors", state="error")
                    for e in result.errors:
                        st.sidebar.error(e)
                elif result.ingested == 0 and result.skipped > 0:
                    status.update(label="All chunks already in store", state="complete")
                    st.sidebar.warning(msg)
                else:
                    status.update(label="Done", state="complete")
                    st.sidebar.success(msg)
            except Exception as exc:
                st.sidebar.exception(exc)
                status.update(label="Failed", state="error")

    docs = store.list_documents()
    if docs:
        st.sidebar.divider()
        st.sidebar.markdown("##### Indexed sources")
        for doc in docs:
            c1, c2 = st.sidebar.columns([5, 1])
            with c1:
                st.markdown(
                    f"<div style='font-size:0.85rem;color:#0f172a;'><b>{doc['source']}</b><br/>"
                    f"<span style='color:#475569;'>{doc.get('topic', '')} · "
                    f"{doc.get('chunk_count', 0)} chunks</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"del_{doc['source']}", help="Remove from index"):
                    store.delete_document(doc["source"])
                    st.session_state.ingested_documents = store.list_documents()
                    st.rerun()
    else:
        st.sidebar.info("No documents yet — upload to build the retrieval index.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    stats = store.get_collection_stats()
    st.sidebar.divider()
    st.sidebar.markdown("##### Index health")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.metric("Chunks", stats["total_chunks"])
    with c2:
        topics_n = len(stats.get("topics") or [])
        st.metric("Topics", topics_n)
    if stats["topics"]:
        st.sidebar.caption(", ".join(stats["topics"][:12]) + ("…" if topics_n > 12 else ""))
    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics (GAN / SOM / Boltzmann) detected")
    elif stats["total_chunks"] > 0:
        st.sidebar.caption("Optional: add bonus-topic materials for extra coverage.")


# ---------------------------------------------------------------------------
# Document viewer
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    with st.container(border=True):
        st.markdown("#### Document viewer")
        st.caption("Browse chunks and metadata stored in Chroma for each source file.")

        docs = store.list_documents()
        if not docs:
            st.markdown(
                "<div style='padding:2rem 1rem;text-align:center;color:#64748b;'>"
                "<p style='font-size:1.1rem;margin:0 0 0.5rem;'>No indexed documents yet</p>"
                "<p style='margin:0;font-size:0.9rem;'>Use the sidebar to ingest <code>.md</code> or "
                "<code>.pdf</code> files from <code>data/corpus/</code>.</p></div>",
                unsafe_allow_html=True,
            )
            return

        options = [d["source"] for d in docs]
        default = (
            st.session_state.selected_document
            if st.session_state.selected_document in options
            else options[0]
        )
        idx = options.index(default) if default in options else 0
        selected = st.selectbox("Source file", options=options, index=idx, label_visibility="collapsed")
        st.session_state.selected_document = selected

        chunks = store.get_document_chunks(selected)
        meta0 = chunks[0].metadata if chunks else None
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("Chunks", len(chunks))
        with t2:
            st.metric("Topic", meta0.topic if meta0 else "—")
        with t3:
            st.metric("Difficulty", meta0.difficulty if meta0 else "—")

        box = st.container(height=380)
        with box:
            for i, ch in enumerate(chunks, start=1):
                st.markdown(
                    f"**Chunk {i}** · `{ch.metadata.type}` · "
                    f"{ch.metadata.topic} / {ch.metadata.difficulty}"
                )
                st.markdown(ch.chunk_text)
                st.divider()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    with st.container(border=True):
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown("#### Interview chat")
            st.caption("Answers use retrieved chunks only. Off-topic questions trigger the guardrail.")
        with head_r:
            if st.button("New chat", use_container_width=True, help="Clear history & new thread"):
                _new_conversation()
                st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id}`")

        f1, f2 = st.columns(2)
        with f1:
            topic_options = [
                "All",
                "ANN",
                "CNN",
                "RNN",
                "LSTM",
                "Seq2Seq",
                "Autoencoder",
                "GAN",
                "SOM",
                "BoltzmannMachine",
            ]
            tsel = st.selectbox(
                "Topic",
                topic_options,
                index=topic_options.index(st.session_state.topic_filter)
                if st.session_state.topic_filter in topic_options
                else 0,
            )
            st.session_state.topic_filter = None if tsel == "All" else tsel
        with f2:
            diff_opts = ["All", "beginner", "intermediate", "advanced"]
            dsel = st.selectbox(
                "Difficulty",
                diff_opts,
                index=diff_opts.index(st.session_state.difficulty_filter)
                if st.session_state.difficulty_filter in diff_opts
                else 0,
            )
            st.session_state.difficulty_filter = None if dsel == "All" else dsel

        if not st.session_state.chat_history:
            st.info(
                "Ask anything covered in your corpus — e.g. *How does backpropagation work?* "
                "or *Compare pooling and strided convolutions.*"
            )

        chat_container = st.container(height=420)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    conf = message.get("confidence")
                    if conf is not None and message["role"] == "assistant" and not message.get(
                        "no_context_found"
                    ):
                        st.caption(f"Retrieval confidence (avg chunk score): {float(conf):.2f}")
                    if message.get("sources"):
                        st.markdown("**Sources**")
                        for src in message["sources"]:
                            st.markdown(
                                f'<span class="rag-source-pill">{src}</span>',
                                unsafe_allow_html=True,
                            )
                    if message.get("no_context_found"):
                        st.warning("No matching corpus content — answer is a guardrail response, not from your files.")

        if prompt := st.chat_input("Ask about deep learning in your corpus…"):
            if not prompt.strip():
                st.warning("Enter a non-empty question.")
            else:
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt.strip()}
                )
                with st.spinner("Retrieving & generating…"):
                    try:
                        result = graph.invoke(
                            {
                                "messages": [HumanMessage(content=prompt.strip())],
                                "topic_filter": st.session_state.topic_filter,
                                "difficulty_filter": st.session_state.difficulty_filter,
                            },
                            config={
                                "configurable": {"thread_id": st.session_state.thread_id}
                            },
                        )
                        fr = result.get("final_response")
                        if fr is None:
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": "No response produced.",
                                    "sources": [],
                                    "no_context_found": True,
                                    "confidence": None,
                                }
                            )
                        else:
                            content, sources, ncf, conf = _fields_from_final_response(fr)
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": content,
                                    "sources": sources,
                                    "no_context_found": ncf,
                                    "confidence": conf,
                                }
                            )
                    except Exception as exc:
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": f"**Error:** `{exc}`",
                                "sources": [],
                                "no_context_found": True,
                                "confidence": None,
                            }
                        )
                st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_app_styles()

    initialise_session_state()

    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    stats = store.get_collection_stats()

    st.markdown(
        f"""
<div class="rag-hero">
  <h1>{settings.app_title}</h1>
  <p>Grounded answers from your study corpus · LangGraph + Chroma · {stats["total_chunks"]} chunks indexed</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    vcol, ccol = st.columns([1, 1], gap="large")
    with vcol:
        render_document_viewer(store)
    with ccol:
        render_chat_interface(graph)

    st.divider()
    st.caption(
        "Deep Learning RAG Agent · Answers cite retrieved chunks. "
        "Configure LLM and paths in `.env`."
    )


if __name__ == "__main__":
    main()
