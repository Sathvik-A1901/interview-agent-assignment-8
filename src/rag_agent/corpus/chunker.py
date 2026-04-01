"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            raw = self._chunk_pdf(file_path, chunk_size, chunk_overlap)
        elif suffix in (".md", ".markdown"):
            raw = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        base_meta = self._infer_metadata(file_path, metadata_overrides)
        chunks: list[DocumentChunk] = []
        for item in raw:
            text = item["text"].strip()
            if len(text) < 20:
                continue
            meta = ChunkMetadata(
                topic=base_meta.topic,
                difficulty=base_meta.difficulty,
                type=base_meta.type,
                source=base_meta.source,
                related_topics=list(base_meta.related_topics),
                is_bonus=base_meta.is_bonus,
            )
            cid = VectorStoreManager.generate_chunk_id(meta.source, text)
            chunks.append(DocumentChunk(chunk_id=cid, chunk_text=text, metadata=meta))
        logger.info("Chunked {} → {} chunks", file_path.name, len(chunks))
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        all_chunks: list[DocumentChunk] = []
        for fp in file_paths:
            all_chunks.extend(self.chunk_file(fp, metadata_overrides))
        return all_chunks

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = splitter.split_documents(docs)
        out: list[dict] = []
        for doc in splits:
            out.append({"text": doc.page_content, "page": doc.metadata.get("page", 0)})
        return out

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        headers = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        try:
            header_docs = md_splitter.split_text(text)
        except Exception:
            header_docs = [Document(page_content=text)]

        recursive = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_docs = recursive.split_documents(header_docs)
        return [{"text": d.page_content, "header": str(d.metadata)} for d in final_docs]

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        stem = file_path.stem.lower()
        topic = "General"
        difficulty = "intermediate"
        topic_map = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "som": "SOM",
            "gan": "GAN",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "boltzmann": "BoltzmannMachine",
            "boltzmannmachine": "BoltzmannMachine",
        }

        if "_" in stem:
            head, tail = stem.split("_", 1)
            topic = topic_map.get(head, head.upper())
            if tail.strip():
                difficulty = tail.strip().lower()
        else:
            topic = topic_map.get(stem, stem.upper() if stem else "General")

        is_bonus = topic in ("SOM", "GAN", "BoltzmannMachine")

        meta = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            related_topics=[],
            is_bonus=is_bonus,
        )
        if overrides:
            for key, val in overrides.items():
                if hasattr(meta, key):
                    setattr(meta, key, val)
        return meta
