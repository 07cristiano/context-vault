from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from contextvault.config import Settings
from contextvault.main import create_app
from contextvault.model_gateway import GeneratedAnswer, ModelStatus, VisualAnalysis


class ReadyModelGateway:
    def status(self) -> ModelStatus:
        return ModelStatus(
            reachable=True,
            embedding_ready=True,
            generation_ready=True,
            installed_models=("qwen3-embedding:0.6b", "qwen3.5:2b-q4_K_M"),
            detail="fake models ready",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors

    def analyze_image(self, image_path: Path) -> VisualAnalysis:
        return VisualAnalysis("visible text", "image description")

    def generate_answer(self, question: str, sources: list[object]) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Evidence stays inspectable.",
            citations=("S1",),
            sufficient=True,
        )


def test_status_endpoint_is_model_free(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())

    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["database"]["ready"] is True


def test_root_serves_local_frontend(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())

    with TestClient(app) as client:
        response = client.get("/")
        script = client.get("/static/app.js?v=4")

    assert response.status_code == 200
    assert "ContextVault" in response.text
    assert "/static/styles.css?v=4" in response.text
    assert "/static/app.js?v=4" in response.text
    assert "<svg" not in response.text
    assert script.status_code == 200
    assert 'excerpt.className = "trace-excerpt"' in script.text
    assert 'toggle.textContent = "Expand chunk"' in script.text


def test_document_upload_list_and_delete(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/documents",
            files={
                "file": ("notes.txt", b"Local retrieval keeps evidence inspectable.", "text/plain")
            },
        )
        listed = client.get("/api/documents")
        deleted = client.delete(f"/api/documents/{uploaded.json()['id']}")

    assert uploaded.status_code == 201
    assert uploaded.json()["filename"] == "notes.txt"
    assert len(listed.json()["documents"]) == 1
    assert deleted.status_code == 204


def test_query_returns_citations_and_retrieval_trace(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())

    with TestClient(app) as client:
        client.post(
            "/api/documents",
            files={"file": ("notes.txt", b"Evidence stays inspectable locally.", "text/plain")},
        )
        response = client.post("/api/query", json={"question": "Where is evidence?"})

    assert response.status_code == 200
    assert response.json()["sufficient"] is True
    assert response.json()["citations"] == ["S1"]
    assert response.json()["evidence"][0]["filename"] == "notes.txt"
    assert response.json()["retrieval_trace"]


def test_querying_an_empty_vault_returns_insufficient_evidence(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())

    with TestClient(app) as client:
        response = client.post("/api/query", json={"question": "What is in the vault?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Insufficient evidence."
    assert response.json()["sufficient"] is False
    assert response.json()["citations"] == []
    assert response.json()["evidence"] == []
    assert response.json()["retrieval_trace"] == []


def test_selected_evidence_contains_full_chunk_but_trace_stays_truncated(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "instance")
    app = create_app(settings=settings, model_gateway=ReadyModelGateway())
    long_text = ("A database stores related records in an organized way. " * 40).encode()

    with TestClient(app) as client:
        client.post(
            "/api/documents",
            files={"file": ("database.txt", long_text, "text/plain")},
        )
        response = client.post("/api/query", json={"question": "What is a database?"})

    assert response.status_code == 200
    assert len(response.json()["evidence"][0]["excerpt"]) > 600
    assert len(response.json()["retrieval_trace"][0]["excerpt"]) == 600
