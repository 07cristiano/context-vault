import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextvault.config import Settings
from contextvault.errors import ModelResponseError
from contextvault.ollama_service import OllamaService


class FakeClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.embed_arguments: dict[str, object] = {}

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            models=[
                SimpleNamespace(model="qwen3-embedding:0.6b"),
                SimpleNamespace(model="qwen3.5:2b-q4_K_M"),
            ]
        )

    def embed(self, **kwargs: object) -> SimpleNamespace:
        self.embed_arguments = kwargs
        return SimpleNamespace(embeddings=self.embeddings)


class FakeAnswerClient(FakeClient):
    def __init__(self, answer: dict[str, object]) -> None:
        super().__init__([])
        self.answer = answer
        self.chat_arguments: dict[str, object] = {}

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.chat_arguments = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.answer)))


def settings_for(tmp_path: Path, dimension: int = 3) -> Settings:
    return Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "instance",
        embedding_dimension=dimension,
    )


def test_status_and_embeddings_use_configured_models(tmp_path: Path) -> None:
    client = FakeClient([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    service = OllamaService(settings_for(tmp_path), client=client)

    status = service.status()
    vectors = service.embed(["first", "second"])

    assert status.reachable is True
    assert status.embedding_ready is True
    assert status.generation_ready is True
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 3)
    assert client.embed_arguments["keep_alive"] == 0


def test_invalid_embedding_shape_is_rejected(tmp_path: Path) -> None:
    service = OllamaService(settings_for(tmp_path), client=FakeClient([[1.0, 0.0]]))

    with pytest.raises(ModelResponseError, match="Expected embedding shape"):
        service.embed(["text"])


def test_generation_schema_requires_source_or_none(tmp_path: Path) -> None:
    client = FakeAnswerClient(
        {
            "answer": "Documents stay local.",
            "citations": ["S1"],
        }
    )
    service = OllamaService(settings_for(tmp_path), client=client)

    result = service.generate_answer(
        "Where are documents?",
        [SimpleNamespace(label="S1", content="Documents stay local.")],
    )

    citation_schema = client.chat_arguments["format"]["properties"]["citations"]
    assert citation_schema["minItems"] == 1
    assert citation_schema["items"]["enum"] == ["S1", "NONE"]
    assert "sufficient" not in client.chat_arguments["format"]["properties"]
    assert result.citations == ("S1",)
    assert result.sufficient is True


def test_none_sentinel_becomes_empty_application_citations(tmp_path: Path) -> None:
    client = FakeAnswerClient(
        {
            "answer": "Insufficient evidence.",
            "citations": ["NONE"],
        }
    )
    service = OllamaService(settings_for(tmp_path), client=client)

    result = service.generate_answer(
        "What is the revenue target?",
        [SimpleNamespace(label="S1", content="The source does not state revenue.")],
    )

    assert result.sufficient is False
    assert result.citations == ()


def test_generation_rejects_response_metadata_as_answer(tmp_path: Path) -> None:
    client = FakeAnswerClient(
        {
            "answer": "sufficient true",
            "citations": ["S1"],
        }
    )
    service = OllamaService(settings_for(tmp_path), client=client)

    with pytest.raises(ModelResponseError, match="response metadata"):
        service.generate_answer(
            "What is a database?",
            [SimpleNamespace(label="S1", content="A database stores organized data.")],
        )
