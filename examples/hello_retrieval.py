"""
Scratch script: ingest sample_chunk.json, query twice (second run skips duplicate).

Run from repo root:
  PYTHONPATH=src python examples/hello_retrieval.py
Windows PowerShell:
  $env:PYTHONPATH="src"; python examples/hello_retrieval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample_path = root / "examples" / "sample_chunk.json"
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    meta = ChunkMetadata.from_dict(data["metadata"])
    text = data["chunk_text"]
    cid = VectorStoreManager.generate_chunk_id(meta.source, text)
    chunk = DocumentChunk(chunk_id=cid, chunk_text=text, metadata=meta)

    store = VectorStoreManager()
    r1 = store.ingest([chunk])
    print("First ingest:", r1)
    r2 = store.ingest([chunk])
    print("Second ingest (expect skipped=1):", r2)

    hits = store.query("what is a neural network LSTM gates", k=3)
    print("Query hits:", len(hits))
    for h in hits:
        print(f"  score={h.score:.3f} {h.to_citation()}")


if __name__ == "__main__":
    main()
