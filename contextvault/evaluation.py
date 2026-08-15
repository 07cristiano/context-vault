"""Small, explicit retrieval metrics used by the local evaluation script."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    hit_rate_at_3: float
    mean_reciprocal_rank: float


def calculate_metrics(rankings: list[list[str]], expected: list[str]) -> RetrievalMetrics:
    if not rankings or len(rankings) != len(expected):
        raise ValueError("Rankings and expected documents must have equal non-zero length")

    hits = 0
    reciprocal_ranks = 0.0
    for ranked_documents, expected_document in zip(rankings, expected, strict=True):
        unique_documents = list(dict.fromkeys(ranked_documents))
        if expected_document in unique_documents[:3]:
            hits += 1
        if expected_document in unique_documents:
            reciprocal_ranks += 1.0 / (unique_documents.index(expected_document) + 1)

    count = len(expected)
    return RetrievalMetrics(
        hit_rate_at_3=hits / count,
        mean_reciprocal_rank=reciprocal_ranks / count,
    )
