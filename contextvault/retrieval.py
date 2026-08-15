"""Deterministic lexical, exact semantic, and Reciprocal Rank Fusion retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from contextvault.config import Settings
from contextvault.database import Database, LexicalMatch, StoredChunk
from contextvault.errors import DatabaseError, ModelResponseError
from contextvault.model_gateway import ModelGateway

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    modality: str
    content: str
    lexical_rank: int | None
    lexical_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    fused_score: float

    def qualifies_as_evidence(self, minimum_semantic_similarity: float) -> bool:
        return self.lexical_rank is not None or (
            self.semantic_score is not None and self.semantic_score >= minimum_semantic_similarity
        )


@dataclass(frozen=True, slots=True)
class RetrievalRankings:
    lexical: tuple[LexicalMatch, ...]
    semantic: tuple[tuple[StoredChunk, float], ...]
    hybrid: tuple[RetrievalHit, ...]


def build_fts_query(question: str) -> str:
    """Convert arbitrary user input into a safe OR query of Unicode words."""

    words = re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE)
    unique_words = list(
        dict.fromkeys(word for word in words if len(word) >= 2 and word not in STOPWORDS)
    )[:12]
    return " OR ".join(f'"{word}"' for word in unique_words)


def exact_cosine_scores(query_vector: np.ndarray, chunks: list[StoredChunk]) -> list[float]:
    if query_vector.ndim != 1:
        raise ModelResponseError("Query embedding must be one-dimensional")
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0 or not np.isfinite(query_norm):
        raise ModelResponseError("Query embedding has an invalid norm")
    if not chunks:
        return []

    vectors: list[np.ndarray] = []
    for chunk in chunks:
        vector = np.frombuffer(chunk.embedding, dtype="<f4")
        if vector.size != chunk.embedding_dim or vector.size != query_vector.size:
            raise DatabaseError(f"Chunk {chunk.id} has an incompatible embedding dimension")
        vectors.append(vector)

    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    denominators = norms * query_norm
    scores = np.divide(
        matrix @ query_vector,
        denominators,
        out=np.full(len(chunks), -1.0, dtype=np.float32),
        where=denominators > 0,
    )
    return [float(score) for score in scores]


def reciprocal_rank_fusion(
    lexical: list[LexicalMatch],
    semantic: list[tuple[StoredChunk, float]],
    *,
    rrf_k: int,
) -> list[RetrievalHit]:
    if rrf_k <= 0:
        raise ValueError("RRF k must be positive")

    candidates: dict[int, dict[str, object]] = {}
    for rank, match in enumerate(lexical, start=1):
        candidates[match.chunk.id] = {
            "chunk": match.chunk,
            "lexical_rank": rank,
            "lexical_score": match.score,
            "semantic_rank": None,
            "semantic_score": None,
            "fused_score": 1.0 / (rrf_k + rank),
        }

    for rank, (chunk, score) in enumerate(semantic, start=1):
        candidate = candidates.setdefault(
            chunk.id,
            {
                "chunk": chunk,
                "lexical_rank": None,
                "lexical_score": None,
                "semantic_rank": None,
                "semantic_score": None,
                "fused_score": 0.0,
            },
        )
        candidate["semantic_rank"] = rank
        candidate["semantic_score"] = score
        candidate["fused_score"] = float(candidate["fused_score"]) + 1.0 / (rrf_k + rank)

    hits = []
    for candidate in candidates.values():
        chunk = candidate["chunk"]
        if not isinstance(chunk, StoredChunk):
            raise TypeError("Invalid retrieval candidate")
        hits.append(
            RetrievalHit(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                modality=chunk.modality,
                content=chunk.content,
                lexical_rank=candidate["lexical_rank"],  # type: ignore[arg-type]
                lexical_score=candidate["lexical_score"],  # type: ignore[arg-type]
                semantic_rank=candidate["semantic_rank"],  # type: ignore[arg-type]
                semantic_score=candidate["semantic_score"],  # type: ignore[arg-type]
                fused_score=float(candidate["fused_score"]),
            )
        )
    return sorted(hits, key=lambda hit: (-hit.fused_score, hit.chunk_id))


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        model_gateway: ModelGateway,
    ) -> None:
        self.settings = settings
        self.database = database
        self.model_gateway = model_gateway

    def search(self, question: str) -> list[RetrievalHit]:
        return list(self.rankings(question).hybrid[: self.settings.retrieval_trace_limit])

    def rankings(self, question: str) -> RetrievalRankings:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty")

        lexical = self.database.lexical_search(
            build_fts_query(normalized_question), self.settings.lexical_candidates
        )
        chunks = self.database.all_chunks()
        if not chunks:
            return RetrievalRankings(lexical=tuple(lexical), semantic=(), hybrid=())

        query_embedding = self.model_gateway.embed([normalized_question])[0]
        semantic_scores = exact_cosine_scores(query_embedding, chunks)
        semantic = sorted(
            zip(chunks, semantic_scores, strict=True),
            key=lambda item: (-item[1], item[0].id),
        )[: self.settings.semantic_candidates]
        fused = reciprocal_rank_fusion(lexical, semantic, rrf_k=self.settings.rrf_k)
        return RetrievalRankings(
            lexical=tuple(lexical),
            semantic=tuple(semantic),
            hybrid=tuple(fused),
        )
