# Judging Rubric (Course Reference)

Use this for Phase 1 risk assessment in `docs/checklist.md`.

| Risk category        | What breaks if ignored        | Mitigation (this repo)                          |
|---------------------|------------------------------|-------------------------------------------------|
| Retrieval quality | Wrong or empty answers       | Query rewrite, cosine threshold, chunk metadata |
| Hallucination       | Confident false answers      | Empty retrieval → guard, no LLM on off-topic    |
| Duplicate corpus    | Bloated DB, skewed metrics   | Content-hash chunk IDs + upsert semantics       |
| Ops / config        | Works on one laptop only     | `.env`, provider factories, persistent Chroma   |
