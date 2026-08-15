from pathlib import Path

import numpy as np

from contextvault.config import Settings
from contextvault.database import Database, NewChunk
from contextvault.retrieval import (
    RetrievalService,
    build_fts_query,
    reciprocal_rank_fusion,
)


class QueryGateway:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.vector for _ in texts])


def add_chunk(
    database: Database,
    *,
    filename: str,
    sha256: str,
    content: str,
    vector: list[float],
) -> None:
    array = np.asarray(vector, dtype="<f4")
    database.add_document(
        original_filename=filename,
        stored_filename=f"stored-{filename}",
        media_type="text/plain",
        sha256=sha256,
        chunks=[
            NewChunk(
                position=0,
                page_number=None,
                modality="text",
                content=content,
                embedding=array.tobytes(),
                embedding_dim=len(vector),
            )
        ],
        max_documents=20,
        max_chunks=300,
    )


def make_retrieval(tmp_path: Path, vector: list[float]) -> RetrievalService:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "instance",
        embedding_dimension=3,
    )
    database = Database(settings.database_path)
    database.initialize()
    add_chunk(
        database,
        filename="python.txt",
        sha256="one",
        content="Python decorators wrap functions and preserve metadata.",
        vector=[1.0, 0.0, 0.0],
    )
    add_chunk(
        database,
        filename="fastapi.txt",
        sha256="two",
        content="FastAPI validates request bodies with Pydantic models.",
        vector=[0.8, 0.2, 0.0],
    )
    add_chunk(
        database,
        filename="bread.txt",
        sha256="three",
        content="Banana bread uses ripe fruit and flour.",
        vector=[0.0, 1.0, 0.0],
    )
    return RetrievalService(settings, database, QueryGateway(vector))


def test_fts_query_never_exposes_user_syntax() -> None:
    assert build_fts_query('" OR secret* (test)') == '"secret" OR "test"'


def test_semantic_retrieval_finds_paraphrase(tmp_path: Path) -> None:
    retrieval = make_retrieval(tmp_path, [1.0, 0.0, 0.0])

    hits = retrieval.search("How can I add behavior around a callable?")

    assert hits[0].filename == "python.txt"
    assert hits[0].semantic_rank == 1
    assert hits[0].semantic_score == 1.0


def test_hybrid_trace_preserves_both_rank_explanations(tmp_path: Path) -> None:
    retrieval = make_retrieval(tmp_path, [1.0, 0.0, 0.0])

    hits = retrieval.search("Pydantic validation")
    by_filename = {hit.filename: hit for hit in hits}

    assert by_filename["fastapi.txt"].lexical_rank == 1
    assert by_filename["python.txt"].semantic_rank == 1
    assert by_filename["fastapi.txt"].fused_score > 0


def test_rrf_is_stable_for_empty_inputs() -> None:
    assert reciprocal_rank_fusion([], [], rrf_k=60) == []
