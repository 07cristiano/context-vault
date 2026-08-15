from pathlib import Path

import pytest

from contextvault.config import Settings
from contextvault.errors import ConfigurationError


def test_default_runtime_paths_stay_inside_project(tmp_path: Path) -> None:
    settings = Settings.from_env({}, project_root=tmp_path)

    assert settings.data_dir == tmp_path / "instance"
    assert settings.database_path == tmp_path / "instance" / "contextvault.db"
    assert settings.uploads_dir == tmp_path / "instance" / "uploads"


def test_data_directory_cannot_escape_repository(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="inside the repository"):
        Settings.from_env(
            {"CONTEXTVAULT_DATA_DIR": str(tmp_path.parent / "outside")},
            project_root=tmp_path,
        )


def test_ollama_host_must_be_local(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="localhost"):
        Settings.from_env(
            {"CONTEXTVAULT_OLLAMA_HOST": "https://example.com"},
            project_root=tmp_path,
        )


def test_relative_evidence_score_must_be_a_ratio(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        data_dir=tmp_path / "instance",
        min_relative_evidence_score=0,
    )

    with pytest.raises(ConfigurationError, match="Relative evidence score"):
        settings.validate()
