"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from loguru import logger

from rag_agent.agent.prompts import (
    QUERY_REWRITE_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, ChunkMetadata, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import get_default_vector_store


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    """Read graph state: dict, mappingproxy, UserDict, etc. — not always ``dict``."""
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _retrieved_chunks_from_state(state: Any) -> list[RetrievedChunk]:
    """Normalize chunks after checkpoint/serialization (dict or dataclass)."""
    raw = _state_get(state, "retrieved_chunks") or []
    out: list[RetrievedChunk] = []
    for item in raw:
        if isinstance(item, RetrievedChunk):
            out.append(item)
        elif isinstance(item, Mapping):
            meta = item.get("metadata") or {}
            out.append(
                RetrievedChunk(
                    chunk_id=str(item.get("chunk_id", "")),
                    chunk_text=str(item.get("chunk_text", "")),
                    metadata=ChunkMetadata.from_dict(meta)
                    if isinstance(meta, Mapping)
                    else meta,
                    score=float(item.get("score", 0.0)),
                )
            )
    return out


def _no_context_payload(rewritten_query: str) -> dict:
    """State updates when nothing relevant is retrieved (graph may END here)."""
    no_context_message = (
        "I was unable to find relevant information in the corpus for your query. "
        "This may mean the topic is not yet covered in the study material, or "
        "your query may need to be rephrased. Please try a more specific "
        "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
    )
    response = AgentResponse(
        answer=no_context_message,
        sources=[],
        confidence=0.0,
        no_context_found=True,
        rewritten_query=rewritten_query,
    )
    return {
        "retrieved_chunks": [],
        "no_context_found": True,
        "final_response": response,
        "messages": [AIMessage(content=no_context_message)],
    }


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    messages = _state_get(state, "messages") or []

    def _latest_human_text() -> str:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                c = m.content
                return c if isinstance(c, str) else str(c)
        return ""

    original_query = _latest_human_text().strip()
    if not original_query:
        return {"original_query": "", "rewritten_query": ""}

    llm = LLMFactory(get_settings()).create()
    prompt = QUERY_REWRITE_PROMPT.format(original_query=original_query)
    try:
        out = llm.invoke(prompt)
        text = getattr(out, "content", str(out))
        rewritten = (text if isinstance(text, str) else str(text)).strip()
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: {}", exc)
        rewritten = original_query

    if not rewritten:
        rewritten = original_query
    return {"original_query": original_query, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    store = get_default_vector_store()
    rewritten = _state_get(state, "rewritten_query") or ""
    q = rewritten.strip()
    if not q:
        return _no_context_payload(rewritten)

    chunks = store.query(
        query_text=q,
        topic_filter=_state_get(state, "topic_filter"),
        difficulty_filter=_state_get(state, "difficulty_filter"),
    )
    if not chunks:
        return _no_context_payload(rewritten)
    return {"retrieved_chunks": chunks, "no_context_found": False}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    settings = get_settings()
    llm = LLMFactory(settings).create()

    # ---- Hallucination Guard -----------------------------------------------
    if _state_get(state, "no_context_found"):
        return _no_context_payload(_state_get(state, "rewritten_query") or "")

    enc = tiktoken.get_encoding("cl100k_base")

    def _token_counter(msgs: list) -> int:
        total = 0
        for m in msgs:
            c = m.content if isinstance(m.content, str) else str(m.content)
            total += len(enc.encode(c))
        return total

    chunks_list = _retrieved_chunks_from_state(state)
    context_parts = []
    for ch in chunks_list:
        cite = f"[SOURCE: {ch.metadata.topic} | {ch.metadata.source}]"
        context_parts.append(f"{cite}\n{ch.chunk_text}")
    context_block = "\n\n---\n\n".join(context_parts)
    avg_conf = (
        sum(c.score for c in chunks_list) / len(chunks_list) if chunks_list else 0.0
    )
    sources = [c.to_citation() for c in chunks_list]

    history = list(_state_get(state, "messages") or [])
    trimmed = trim_messages(
        history,
        max_tokens=settings.max_context_tokens,
        strategy="last",
        token_counter=_token_counter,
        include_system=True,
        start_on="human",
        allow_partial=False,
    )

    llm = LLMFactory(settings).create()
    user_q = _state_get(state, "original_query") or ""
    prompt_messages = [
        SystemMessage(
            content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"Retrieved context from the corpus (cite using [SOURCE: topic | filename]):\n\n"
                f"{context_block}"
            )
        ),
        *trimmed,
        HumanMessage(content=user_q),
    ]
    ai = llm.invoke(prompt_messages)
    answer_text = ai.content if isinstance(ai.content, str) else str(ai.content)
    response = AgentResponse(
        answer=answer_text,
        sources=sources,
        confidence=avg_conf,
        no_context_found=False,
        rewritten_query=_state_get(state, "rewritten_query") or "",
    )
    return {
        "final_response": response,
        "messages": [AIMessage(content=answer_text)],
    }


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: decide whether to retry retrieval or generate.

    Called by the graph after retrieval_node. If no context was found,
    the graph routes back to query_rewrite_node for one retry with a
    broader query before triggering the hallucination guard.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — skip generation, return no_context response directly.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    if _state_get(state, "no_context_found"):
        return "end"
    return "generate"
