"""SQLite connection management and version-one schema."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from contextvault.errors import (
    CapacityError,
    DatabaseError,
    DocumentNotFoundError,
    DuplicateDocumentError,
)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('indexed', 'failed')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    error TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    page_number INTEGER,
    modality TEXT NOT NULL CHECK (modality IN ('text', 'image')),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
    image_path TEXT,
    UNIQUE(document_id, position)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


@dataclass(frozen=True, slots=True)
class NewChunk:
    position: int
    page_number: int | None
    modality: str
    content: str
    embedding: bytes
    embedding_dim: int
    image_path: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    id: int
    filename: str
    media_type: str
    chunk_count: int
    created_at: str


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: int
    document_id: int
    filename: str
    page_number: int | None
    modality: str
    content: str
    embedding: bytes
    embedding_dim: int


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    chunk: StoredChunk
    score: float


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(SCHEMA_SQL)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not initialize SQLite: {exc}") from exc

    def health(self) -> tuple[bool, str]:
        try:
            with self.connection() as connection:
                sqlite_version = connection.execute("SELECT sqlite_version()").fetchone()[0]
                connection.execute("SELECT count(*) FROM chunks_fts").fetchone()
                schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
        except sqlite3.Error as exc:
            return False, f"SQLite check failed: {exc}"

        if schema_version != SCHEMA_VERSION:
            return False, f"Unsupported schema version {schema_version}"
        return True, f"SQLite {sqlite_version}, schema {schema_version}, FTS5 ready"

    def contains_hash(self, sha256: str) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return row is not None

    def counts(self) -> tuple[int, int]:
        with self.connection() as connection:
            document_count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return document_count, chunk_count

    def add_document(
        self,
        *,
        original_filename: str,
        stored_filename: str,
        media_type: str,
        sha256: str,
        chunks: list[NewChunk],
        max_documents: int,
        max_chunks: int,
    ) -> DocumentSummary:
        if not chunks:
            raise ValueError("An indexed document requires at least one chunk")

        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                document_count = connection.execute("SELECT count(*) FROM documents").fetchone()[0]
                chunk_count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
                if document_count >= max_documents:
                    raise CapacityError(f"The vault is limited to {max_documents} documents")
                if chunk_count + len(chunks) > max_chunks:
                    raise CapacityError(f"The vault is limited to {max_chunks} chunks")

                cursor = connection.execute(
                    """
                    INSERT INTO documents(
                        original_filename, stored_filename, media_type, sha256, status
                    ) VALUES (?, ?, ?, ?, 'indexed')
                    """,
                    (original_filename, stored_filename, media_type, sha256),
                )
                document_id = int(cursor.lastrowid)
                connection.executemany(
                    """
                    INSERT INTO chunks(
                        document_id, position, page_number, modality, content,
                        embedding, embedding_dim, image_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            document_id,
                            chunk.position,
                            chunk.page_number,
                            chunk.modality,
                            chunk.content,
                            chunk.embedding,
                            chunk.embedding_dim,
                            chunk.image_path,
                        )
                        for chunk in chunks
                    ],
                )
                row = connection.execute(
                    "SELECT created_at FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                connection.commit()
        except sqlite3.IntegrityError as exc:
            if "documents.sha256" in str(exc):
                raise DuplicateDocumentError("This file is already indexed") from exc
            raise DatabaseError(f"Could not store document: {exc}") from exc
        except (CapacityError, ValueError):
            raise
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not store document: {exc}") from exc

        return DocumentSummary(
            id=document_id,
            filename=original_filename,
            media_type=media_type,
            chunk_count=len(chunks),
            created_at=row["created_at"],
        )

    def list_documents(self) -> list[DocumentSummary]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.id, d.original_filename, d.media_type, d.created_at,
                       count(c.id) AS chunk_count
                FROM documents AS d
                LEFT JOIN chunks AS c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()
        return [
            DocumentSummary(
                id=row["id"],
                filename=row["original_filename"],
                media_type=row["media_type"],
                chunk_count=row["chunk_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_document(self, document_id: int) -> str:
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT stored_filename FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if row is None:
                    raise DocumentNotFoundError(f"Document {document_id} was not found")
                connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                connection.commit()
        except DocumentNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not delete document: {exc}") from exc
        return str(row["stored_filename"])

    def lexical_search(self, fts_query: str, limit: int) -> list[LexicalMatch]:
        if not fts_query or limit <= 0:
            return []
        try:
            with self.connection() as connection:
                rows = connection.execute(
                    """
                    SELECT c.id, c.document_id, d.original_filename, c.page_number,
                           c.modality, c.content, c.embedding, c.embedding_dim,
                           bm25(chunks_fts) AS lexical_score
                    FROM chunks_fts
                    JOIN chunks AS c ON c.id = chunks_fts.rowid
                    JOIN documents AS d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY lexical_score ASC, c.id ASC
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Lexical search failed: {exc}") from exc

        return [
            LexicalMatch(
                chunk=StoredChunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    filename=row["original_filename"],
                    page_number=row["page_number"],
                    modality=row["modality"],
                    content=row["content"],
                    embedding=row["embedding"],
                    embedding_dim=row["embedding_dim"],
                ),
                score=float(row["lexical_score"]),
            )
            for row in rows
        ]

    def all_chunks(self) -> list[StoredChunk]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.document_id, d.original_filename, c.page_number,
                       c.modality, c.content, c.embedding, c.embedding_dim
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                ORDER BY c.id ASC
                """
            ).fetchall()
        return [
            StoredChunk(
                id=row["id"],
                document_id=row["document_id"],
                filename=row["original_filename"],
                page_number=row["page_number"],
                modality=row["modality"],
                content=row["content"],
                embedding=row["embedding"],
                embedding_dim=row["embedding_dim"],
            )
            for row in rows
        ]
