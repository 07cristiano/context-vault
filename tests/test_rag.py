from dataclasses import replace

from contextvault.config import Settings
from contextvault.model_gateway import GeneratedAnswer
from contextvault.rag import RagService
from contextvault.retrieval import RetrievalHit


def hit(*, semantic_score: float, lexical_rank: int | None = None) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=1,
        document_id=2,
        filename="notes.txt",
        page_number=None,
        modality="text",
        content="ContextVault keeps documents on the user's computer.",
        lexical_rank=lexical_rank,
        lexical_score=-1.0 if lexical_rank else None,
        semantic_rank=1,
        semantic_score=semantic_score,
        fused_score=0.02,
    )


class FakeRetrieval:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def search(self, question: str) -> list[RetrievalHit]:
        return self.hits


class FakeGenerationGateway:
    def __init__(self) -> None:
        self.calls = 0
        self.sources: list[object] = []

    def generate_answer(self, question: str, sources: list[object]) -> GeneratedAnswer:
        self.calls += 1
        self.sources = sources
        return GeneratedAnswer(
            answer="Documents remain on the user's computer.",
            citations=("S1",),
            sufficient=True,
        )


def test_grounded_result_uses_selected_source_label(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    gateway = FakeGenerationGateway()
    service = RagService(settings, FakeRetrieval([hit(semantic_score=0.8)]), gateway)

    result = service.query("Where are documents stored?")

    assert result.sufficient is True
    assert result.citations == ("S1",)
    assert result.evidence[0].label == "S1"
    assert gateway.calls == 1


def test_weak_semantic_result_refuses_without_generation(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    gateway = FakeGenerationGateway()
    weak_hit = replace(hit(semantic_score=0.2), lexical_rank=None, lexical_score=None)
    service = RagService(settings, FakeRetrieval([weak_hit]), gateway)

    result = service.query("What is the launch date?")

    assert result.sufficient is False
    assert result.answer == "Insufficient evidence."
    assert result.citations == ()
    assert gateway.calls == 0


def test_only_strong_relative_hits_are_sent_to_generation(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    gateway = FakeGenerationGateway()
    hits = [
        replace(hit(semantic_score=0.90), chunk_id=1, fused_score=0.030),
        replace(hit(semantic_score=0.85), chunk_id=2, fused_score=0.027),
        replace(hit(semantic_score=0.80), chunk_id=3, fused_score=0.020),
    ]
    service = RagService(settings, FakeRetrieval(hits), gateway)

    service.query("What is a database?")

    assert [source.label for source in gateway.sources] == ["S1", "S2"]


def test_generation_evidence_is_capped_at_three_chunks(tmp_path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    gateway = FakeGenerationGateway()
    hits = [
        replace(hit(semantic_score=0.90), chunk_id=index, fused_score=0.030)
        for index in range(1, 5)
    ]
    service = RagService(settings, FakeRetrieval(hits), gateway)

    service.query("What is a database?")

    assert [source.label for source in gateway.sources] == ["S1", "S2", "S3"]
