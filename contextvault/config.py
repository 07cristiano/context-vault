"""Application configuration with repository-local runtime paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from contextvault.errors import ConfigurationError


def _positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings for one local ContextVault process."""

    project_root: Path
    data_dir: Path
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    generation_model: str = "qwen3.5:2b-q4_K_M"
    embedding_dimension: int = 1024
    model_context_tokens: int = 4096
    max_upload_bytes: int = 15 * 1024 * 1024
    max_documents: int = 20
    max_chunks: int = 300
    max_chunks_per_document: int = 120
    embedding_batch_size: int = 16
    chunk_words: int = 180
    chunk_overlap_words: int = 30
    lexical_candidates: int = 10
    semantic_candidates: int = 10
    retrieval_trace_limit: int = 10
    max_evidence_sources: int = 3
    min_relative_evidence_score: float = 0.85
    rrf_k: int = 60
    min_semantic_similarity: float = 0.45

    @property
    def database_path(self) -> Path:
        return self.data_dir / "contextvault.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        project_root: Path | None = None,
    ) -> Settings:
        values = os.environ if environ is None else environ
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        raw_data_dir = values.get("CONTEXTVAULT_DATA_DIR")
        data_dir = Path(raw_data_dir) if raw_data_dir else root / "instance"
        if not data_dir.is_absolute():
            data_dir = root / data_dir

        settings = cls(
            project_root=root,
            data_dir=data_dir.resolve(),
            ollama_host=values.get("CONTEXTVAULT_OLLAMA_HOST", "http://127.0.0.1:11434"),
            embedding_model=values.get("CONTEXTVAULT_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            generation_model=values.get("CONTEXTVAULT_GENERATION_MODEL", "qwen3.5:2b-q4_K_M"),
            embedding_dimension=_positive_int(
                "CONTEXTVAULT_EMBEDDING_DIMENSION",
                values.get("CONTEXTVAULT_EMBEDDING_DIMENSION", "1024"),
            ),
            model_context_tokens=_positive_int(
                "CONTEXTVAULT_MODEL_CONTEXT_TOKENS",
                values.get("CONTEXTVAULT_MODEL_CONTEXT_TOKENS", "4096"),
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.data_dir.is_relative_to(self.project_root):
            raise ConfigurationError("CONTEXTVAULT_DATA_DIR must stay inside the repository")

        parsed_host = urlparse(self.ollama_host)
        if parsed_host.scheme != "http" or parsed_host.hostname not in {"127.0.0.1", "localhost"}:
            raise ConfigurationError("Ollama must use an HTTP localhost address")

        if not self.embedding_model.strip() or not self.generation_model.strip():
            raise ConfigurationError("Model identifiers cannot be empty")
        if self.chunk_overlap_words >= self.chunk_words:
            raise ConfigurationError("Chunk overlap must be smaller than chunk size")
        if not 0 < self.min_relative_evidence_score <= 1:
            raise ConfigurationError("Relative evidence score must be between zero and one")

    def ensure_runtime_directories(self) -> None:
        """Create only the repository-local directories owned by ContextVault."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
