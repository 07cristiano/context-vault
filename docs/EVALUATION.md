# Retrieval Evaluation

## Question

Does hybrid retrieval preserve exact keyword matches while recovering a paraphrase that keyword retrieval misses?

## Method

The committed evaluation contains four short synthetic documents and five questions. Each question has exactly one expected source document.

The script indexes the corpus with the real `qwen3-embedding:0.6b` model in a temporary database, then compares:

1. SQLite FTS5/BM25 keyword ranking.
2. Exact cosine semantic ranking.
3. Reciprocal Rank Fusion of both rankings.

Metrics:

- **Hit Rate@3:** fraction of questions where the expected document appears in the top three.
- **Mean Reciprocal Rank (MRR):** average reciprocal position of the first expected document.

## Measured result

Run date: 2026-08-15.

| Method | Hit Rate@3 | MRR |
|---|---:|---:|
| Keyword | 0.800 | 0.800 |
| Semantic | 1.000 | 1.000 |
| Hybrid RRF | 1.000 | 1.000 |

Keyword retrieval returned no result for `How does the system protect confidential material?` because the relevant source describes a local privacy boundary without those exact words. Semantic and hybrid retrieval ranked `privacy.md` first.

All methods ranked the expected document first for the remaining four questions.

## Interpretation

This result supports a narrow claim: on this small synthetic corpus, semantic retrieval fixes one paraphrase failure, and RRF retains that benefit while also incorporating keyword ranks.

It does **not** demonstrate production search quality, statistical significance, or superiority on arbitrary corpora. A larger project should add more documents, adversarial near-matches, multiple relevant sources, and human relevance judgments.

## Reproduce

With Ollama and the embedding model available:

```bash
python scripts/evaluate_retrieval.py
```

The script uses a temporary database and does not modify the user's vault.
