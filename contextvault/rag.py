"""Evidence selection, grounded generation, and application-owned citations."""

from __future__ import annotations

import time
from dataclasses import dataclass

from contextvault.config import Settings
from contextvault.errors import ModelResponseError
from contextvault.model_gateway import ModelGateway, PromptSource
from contextvault.retrieval import RetrievalHit, RetrievalService


@dataclass(frozen=True, slots=True)
class Evidence:
    label: str
    hit: RetrievalHit


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str
    sufficient: bool
    citations: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    trace: tuple[RetrievalHit, ...]
    retrieval_ms: float
    generation_ms: float


class RagService:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalService,
        model_gateway: ModelGateway,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.model_gateway = model_gateway

    def query(self, question: str) -> QueryResult:
        started = time.perf_counter()
        trace = self.retrieval.search(question)
        retrieval_ms = (time.perf_counter() - started) * 1000
        qualifying_hits = [
            hit for hit in trace if hit.qualifies_as_evidence(self.settings.min_semantic_similarity)
        ]
        selected_hits: list[RetrievalHit] = []
        if qualifying_hits:
            minimum_fused_score = (
                qualifying_hits[0].fused_score * self.settings.min_relative_evidence_score
            )
            selected_hits = [
                hit for hit in qualifying_hits if hit.fused_score >= minimum_fused_score
            ][: self.settings.max_evidence_sources]
        evidence = tuple(
            Evidence(label=f"S{index}", hit=hit) for index, hit in enumerate(selected_hits, start=1)
        )

        if not evidence:
            return QueryResult(
                answer="Insufficient evidence.",
                sufficient=False,
                citations=(),
                evidence=(),
                trace=tuple(trace),
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )

        generation_started = time.perf_counter()
        generated = self.model_gateway.generate_answer(
            question.strip(),
            [PromptSource(label=item.label, content=item.hit.content) for item in evidence],
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        known_labels = {item.label for item in evidence}
        if not set(generated.citations).issubset(known_labels):
            raise ModelResponseError("Generated answer contains an unknown citation")
        if generated.sufficient and not generated.citations:
            raise ModelResponseError("Grounded answers require at least one citation")

        return QueryResult(
            answer=generated.answer if generated.sufficient else "Insufficient evidence.",
            sufficient=generated.sufficient,
            citations=generated.citations if generated.sufficient else (),
            evidence=evidence,
            trace=tuple(trace),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
