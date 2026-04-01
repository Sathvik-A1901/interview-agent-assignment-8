"""
test_vectorstore.py
===================
Unit tests for VectorStoreManager.

Run with: uv run pytest tests/ -v
"""

from __future__ import annotations

import pytest

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager, reset_default_vector_store


@pytest.fixture
def isolated_settings(tmp_path) -> Settings:
    return Settings(
        chroma_db_path=str(tmp_path / "chroma_db"),
        chroma_collection_name="test_dl_corpus",
        retrieval_k=4,
        similarity_threshold=0.15,
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    yield
    reset_default_vector_store()


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    metadata = ChunkMetadata(
        topic="LSTM",
        difficulty="intermediate",
        type="concept_explanation",
        source="test_lstm.md",
        related_topics=["RNN", "vanishing_gradient"],
        is_bonus=False,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_lstm.md", "test content"),
        chunk_text=(
            "Long Short-Term Memory networks solve the vanishing gradient problem "
            "through gated mechanisms: the forget gate, input gate, and output gate. "
            "These gates control information flow through the cell state, allowing "
            "the network to maintain relevant information across long sequences."
        ),
        metadata=metadata,
    )


@pytest.fixture
def bonus_chunk() -> DocumentChunk:
    metadata = ChunkMetadata(
        topic="GAN",
        difficulty="advanced",
        type="architecture",
        source="test_gan.md",
        related_topics=["autoencoder", "generative_models"],
        is_bonus=True,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_gan.md", "gan content"),
        chunk_text=(
            "Generative Adversarial Networks consist of two competing neural networks: "
            "a generator that produces synthetic data and a discriminator that "
            "distinguishes real from generated samples. Training is a minimax game."
        ),
        metadata=metadata,
    )


class TestChunkIdGeneration:
    def test_same_content_produces_same_id(self) -> None:
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        assert id1 == id2

    def test_different_content_produces_different_id(self) -> None:
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "content one")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "content two")
        assert id1 != id2

    def test_different_source_produces_different_id(self) -> None:
        id1 = VectorStoreManager.generate_chunk_id("file_a.md", "same text")
        id2 = VectorStoreManager.generate_chunk_id("file_b.md", "same text")
        assert id1 != id2

    def test_id_is_16_characters(self) -> None:
        chunk_id = VectorStoreManager.generate_chunk_id("source.md", "text")
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


class TestDuplicateDetection:
    def test_new_chunk_is_not_duplicate(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        assert store.check_duplicate(sample_chunk.chunk_id) is False

    def test_ingested_chunk_is_duplicate(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        store.ingest([sample_chunk])
        assert store.check_duplicate(sample_chunk.chunk_id) is True

    def test_ingestion_skips_duplicate(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        r1 = store.ingest([sample_chunk])
        r2 = store.ingest([sample_chunk])
        assert r1.ingested == 1
        assert r2.skipped == 1


class TestRetrieval:
    def test_relevant_query_returns_results(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        store.ingest([sample_chunk])
        results = store.query("LSTM gate mechanism vanishing gradient")
        assert len(results) > 0

    def test_irrelevant_query_returns_empty(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        store.ingest([sample_chunk])
        results = store.query(
            "history of the roman empire ancient rome emperors",
            k=4,
        )
        assert results == []

    def test_topic_filter_restricts_results(
        self,
        isolated_settings,
        sample_chunk: DocumentChunk,
        bonus_chunk: DocumentChunk,
    ) -> None:
        store = VectorStoreManager(isolated_settings)
        store.ingest([sample_chunk, bonus_chunk])
        results = store.query(
            "neural network training architecture",
            topic_filter="LSTM",
        )
        assert results
        assert all(c.metadata.topic == "LSTM" for c in results)

    def test_results_sorted_by_score_descending(
        self, isolated_settings, sample_chunk: DocumentChunk
    ) -> None:
        meta2 = ChunkMetadata(
            topic="LSTM",
            difficulty="intermediate",
            type="concept_explanation",
            source="test_lstm2.md",
            related_topics=[],
            is_bonus=False,
        )
        text2 = (
            "Bidirectional LSTMs process sequences in both forward and backward "
            "directions and concatenate representations. They are common in NLP "
            "tagging tasks but add compute compared to unidirectional LSTMs."
        )
        chunk2 = DocumentChunk(
            chunk_id=VectorStoreManager.generate_chunk_id("test_lstm2.md", text2),
            chunk_text=text2,
            metadata=meta2,
        )
        store = VectorStoreManager(isolated_settings)
        store.ingest([chunk2, sample_chunk])
        results = store.query(
            "LSTM forget gate vanishing gradient cell state",
            k=4,
        )
        assert len(results) >= 2
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True)
