from io import BytesIO
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

from contextvault.config import Settings
from contextvault.database import Database
from contextvault.errors import DuplicateDocumentError
from contextvault.ingestion import ExtractedSection, IngestionService, chunk_section
from contextvault.model_gateway import ModelStatus, VisualAnalysis


class FakeModelGateway:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.embedded_texts: list[str] = []

    def status(self) -> ModelStatus:
        return ModelStatus(True, True, True)

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded_texts.extend(texts)
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    def analyze_image(self, image_path: Path) -> VisualAnalysis:
        return VisualAnalysis(
            visible_text="Status: READY FOR REVIEW",
            description="A project dashboard",
        )


def make_service(tmp_path: Path) -> tuple[IngestionService, Database, FakeModelGateway]:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "instance",
        embedding_dimension=4,
        chunk_words=5,
        chunk_overlap_words=2,
    )
    settings.ensure_runtime_directories()
    database = Database(settings.database_path)
    database.initialize()
    gateway = FakeModelGateway()
    return IngestionService(settings, database, gateway), database, gateway


def test_chunking_is_deterministic_and_preserves_page() -> None:
    section = ExtractedSection(
        content="one two three four five six seven eight",
        page_number=3,
        modality="text",
    )

    chunks = chunk_section(section, chunk_words=5, overlap_words=2)

    assert [chunk.content for chunk in chunks] == [
        "one two three four five",
        "four five six seven eight",
    ]
    assert [chunk.page_number for chunk in chunks] == [3, 3]


def test_text_ingestion_is_searchable_and_duplicate_safe(tmp_path: Path) -> None:
    service, database, gateway = make_service(tmp_path)
    content = b"ContextVault combines keyword and semantic retrieval for privacy."

    document = service.ingest("notes.txt", BytesIO(content))

    assert document.filename == "notes.txt"
    assert document.chunk_count == 2
    assert len(gateway.embedded_texts) == 2
    with database.connection() as connection:
        match_count = connection.execute(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'privacy'"
        ).fetchone()[0]
        assert match_count == 1

    with pytest.raises(DuplicateDocumentError):
        service.ingest("copy.txt", BytesIO(content))


def test_image_ingestion_uses_visual_analysis_and_delete_cleans_file(tmp_path: Path) -> None:
    service, database, gateway = make_service(tmp_path)
    image_stream = BytesIO()
    Image.new("RGB", (100, 60), "white").save(image_stream, format="PNG")
    image_stream.seek(0)

    document = service.ingest("dashboard.png", image_stream)

    assert document.chunk_count == 1
    assert "READY FOR REVIEW" in gateway.embedded_texts[0]
    with database.connection() as connection:
        stored_filename = connection.execute(
            "SELECT stored_filename FROM documents WHERE id = ?", (document.id,)
        ).fetchone()[0]
    stored_path = service.settings.uploads_dir / stored_filename
    assert stored_path.is_file()

    service.delete(document.id)

    assert stored_path.exists() is False
    assert database.list_documents() == []


def test_pdf_ingestion_preserves_page_numbers(tmp_path: Path) -> None:
    service, database, _ = make_service(tmp_path)
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "First page discusses lexical keyword search.")
    second = pdf.new_page()
    second.insert_text((72, 72), "Second page discusses semantic vector search.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    document = service.ingest("retrieval.pdf", BytesIO(pdf_bytes))

    with database.connection() as connection:
        rows = connection.execute(
            "SELECT page_number, content FROM chunks WHERE document_id = ? ORDER BY position",
            (document.id,),
        ).fetchall()
    assert {row["page_number"] for row in rows} == {1, 2}
    page_one_text = " ".join(row["content"] for row in rows if row["page_number"] == 1)
    page_two_text = " ".join(row["content"] for row in rows if row["page_number"] == 2)
    assert "First" in page_one_text and "Second" not in page_one_text
    assert "Second" in page_two_text and "First" not in page_two_text
