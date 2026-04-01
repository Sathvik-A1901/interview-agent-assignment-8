"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        db_path = Path(self._settings.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at {db_path}: {exc}"
            ) from exc
        count = self._collection.count()
        logger.info(
            "ChromaDB ready: collection={!r} path={} chunks={}",
            self._settings.chroma_collection_name,
            db_path.resolve(),
            count,
        )

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id])
        return bool(result.get("ids"))

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        pending: list[DocumentChunk] = []
        for chunk in chunks:
            if self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
                continue
            pending.append(chunk)

        batch_size = 100
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            texts = [c.chunk_text for c in batch]
            try:
                embeddings = self._embeddings.embed_documents(texts)
            except Exception as exc:
                for c in batch:
                    result.errors.append(f"{c.metadata.source}: embedding failed: {exc}")
                logger.exception("Batch embedding failed")
                continue
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata.to_dict() for c in batch]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            result.ingested += len(batch)
            for c in batch:
                if c.metadata.source not in result.document_ids:
                    result.document_ids.append(c.metadata.source)

        logger.info(
            "Ingestion complete: ingested={} skipped={} errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        where_filter = None
        if topic_filter and difficulty_filter:
            where_filter = {
                "$and": [
                    {"topic": topic_filter},
                    {"difficulty": difficulty_filter},
                ]
            }
        elif topic_filter:
            where_filter = {"topic": topic_filter}
        elif difficulty_filter:
            where_filter = {"difficulty": difficulty_filter}

        query_embedding = self._embeddings.embed_query(query_text)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        out: list[RetrievedChunk] = []
        ids_list = raw.get("ids") or []
        if not ids_list or not ids_list[0]:
            return out

        for idx, chunk_id in enumerate(ids_list[0]):
            dist = raw["distances"][0][idx]
            score = max(0.0, 1.0 - float(dist))
            if score < self._settings.similarity_threshold:
                continue
            doc = raw["documents"][0][idx]
            meta_raw = raw["metadatas"][0][idx]
            if meta_raw is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=doc or "",
                    metadata=ChunkMetadata.from_dict(meta_raw),
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        data = self._collection.get(include=["metadatas"])
        metas = data.get("metadatas") or []
        counts: defaultdict[str, int] = defaultdict(int)
        topic_for: dict[str, str] = {}
        for m in metas:
            if not m:
                continue
            src = str(m.get("source", "unknown"))
            counts[src] += 1
            topic_for.setdefault(src, str(m.get("topic", "")))
        return [
            {"source": s, "topic": topic_for.get(s, ""), "chunk_count": counts[s]}
            for s in sorted(counts.keys())
        ]

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        data = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        chunks: list[DocumentChunk] = []
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            text = docs[i] if i < len(docs) else ""
            if not meta:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    chunk_text=text or "",
                    metadata=ChunkMetadata.from_dict(meta),
                )
            )
        chunks.sort(key=lambda c: c.chunk_id)
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        data = self._collection.get(include=["metadatas"])
        metas = [m for m in (data.get("metadatas") or []) if m]
        topics = sorted({str(m.get("topic", "")) for m in metas if m.get("topic")})
        sources = sorted({str(m.get("source", "")) for m in metas if m.get("source")})
        bonus_topics = {"SOM", "BoltzmannMachine", "GAN"}
        bonus_present = any(str(m.get("topic", "")) in bonus_topics for m in metas)
        return {
            "total_chunks": self._collection.count(),
            "topics": topics,
            "sources": sources,
            "bonus_topics_present": bonus_present,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        existing = self._collection.get(where={"source": source})
        n = len(existing.get("ids") or [])
        if n:
            self._collection.delete(where={"source": source})
        return n


# Process-wide singleton so Streamlit and LangGraph nodes share one embedding model load.
_default_manager: VectorStoreManager | None = None


def get_default_vector_store() -> VectorStoreManager:
    """Return the shared VectorStoreManager (lazy). Tests should use VectorStoreManager(settings) directly."""
    global _default_manager
    if _default_manager is None:
        _default_manager = VectorStoreManager()
    return _default_manager


def reset_default_vector_store() -> None:
    """Clear the singleton (for tests)."""
    global _default_manager
    _default_manager = None
