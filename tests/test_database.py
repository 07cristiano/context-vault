from pathlib import Path

import numpy as np

from contextvault.database import SCHEMA_VERSION, Database


def test_schema_fts_and_delete_stay_consistent(tmp_path: Path) -> None:
    database = Database(tmp_path / "contextvault.db")
    database.initialize()

    vector = np.array([1.0, 0.0], dtype=np.float32).tobytes()
    with database.connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents(original_filename, stored_filename, media_type, sha256, status)
            VALUES (?, ?, ?, ?, 'indexed')
            """,
            ("notes.txt", "source.txt", "text/plain", "abc123"),
        )
        document_id = cursor.lastrowid
        chunk_cursor = connection.execute(
            """
            INSERT INTO chunks(
                document_id, position, page_number, modality, content, embedding, embedding_dim
            ) VALUES (?, 0, NULL, 'text', ?, ?, 2)
            """,
            (document_id, "hybrid retrieval combines two rankings", vector),
        )
        chunk_id = chunk_cursor.lastrowid
        connection.commit()

        match = connection.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'hybrid'"
        ).fetchone()
        assert match["rowid"] == chunk_id

        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.commit()
        assert connection.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'hybrid'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_database_health_reports_fts5(tmp_path: Path) -> None:
    database = Database(tmp_path / "contextvault.db")
    database.initialize()

    ready, detail = database.health()

    assert ready is True
    assert "FTS5 ready" in detail
