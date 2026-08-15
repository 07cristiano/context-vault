"""Safe, deterministic ingestion for the deliberately small document corpus."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path
from uuid import uuid4

import pymupdf
from PIL import Image, UnidentifiedImageError

from contextvault.config import Settings
from contextvault.database import Database, DocumentSummary, NewChunk
from contextvault.errors import (
    CapacityError,
    DuplicateDocumentError,
    ExtractionError,
    UploadValidationError,
)
from contextvault.model_gateway import ModelGateway

SUPPORTED_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True, slots=True)
class ExtractedSection:
    content: str
    page_number: int | None
    modality: str


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    paragraphs = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in text.split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def chunk_section(
    section: ExtractedSection,
    *,
    chunk_words: int,
    overlap_words: int,
) -> list[ExtractedSection]:
    words = section.content.split()
    if not words:
        return []

    chunks: list[ExtractedSection] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(
            ExtractedSection(
                content=" ".join(words[start:end]),
                page_number=section.page_number,
                modality=section.modality,
            )
        )
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        model_gateway: ModelGateway,
    ) -> None:
        self.settings = settings
        self.database = database
        self.model_gateway = model_gateway

    def ingest(self, filename: str, stream: BufferedIOBase) -> DocumentSummary:
        safe_name, suffix, media_type = self._validate_filename(filename)
        temp_path, sha256 = self._stage_upload(stream)
        final_path: Path | None = None
        try:
            if self.database.contains_hash(sha256):
                raise DuplicateDocumentError("This file is already indexed")

            sections = self._extract(temp_path, suffix)
            chunks = self._chunk_sections(sections)
            if len(chunks) > self.settings.max_chunks_per_document:
                raise CapacityError(
                    "This document produced too many chunks "
                    f"({len(chunks)} > {self.settings.max_chunks_per_document})"
                )

            document_count, chunk_count = self.database.counts()
            if document_count >= self.settings.max_documents:
                raise CapacityError(
                    f"The vault is limited to {self.settings.max_documents} documents"
                )
            if chunk_count + len(chunks) > self.settings.max_chunks:
                raise CapacityError(f"The vault is limited to {self.settings.max_chunks} chunks")

            embeddings = self.model_gateway.embed([chunk.content for chunk in chunks])
            stored_filename = f"{uuid4().hex}{suffix}"
            final_path = self.settings.uploads_dir / stored_filename
            os.replace(temp_path, final_path)

            new_chunks = [
                NewChunk(
                    position=position,
                    page_number=chunk.page_number,
                    modality=chunk.modality,
                    content=chunk.content,
                    embedding=embeddings[position].astype("<f4", copy=False).tobytes(),
                    embedding_dim=self.settings.embedding_dimension,
                    image_path=stored_filename if chunk.modality == "image" else None,
                )
                for position, chunk in enumerate(chunks)
            ]
            try:
                return self.database.add_document(
                    original_filename=safe_name,
                    stored_filename=stored_filename,
                    media_type=media_type,
                    sha256=sha256,
                    chunks=new_chunks,
                    max_documents=self.settings.max_documents,
                    max_chunks=self.settings.max_chunks,
                )
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
        finally:
            temp_path.unlink(missing_ok=True)

    def delete(self, document_id: int) -> None:
        stored_filename = self.database.delete_document(document_id)
        stored_path = (self.settings.uploads_dir / stored_filename).resolve()
        if stored_path.parent != self.settings.uploads_dir.resolve():
            raise RuntimeError("Stored document path escaped the uploads directory")
        stored_path.unlink(missing_ok=True)

    def _validate_filename(self, filename: str) -> tuple[str, str, str]:
        safe_name = Path(filename or "").name.strip()
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in SUPPORTED_TYPES:
            supported = ", ".join(sorted(SUPPORTED_TYPES))
            raise UploadValidationError(f"Supported file extensions: {supported}")
        return safe_name, suffix, SUPPORTED_TYPES[suffix]

    def _stage_upload(self, stream: BufferedIOBase) -> tuple[Path, str]:
        self.settings.ensure_runtime_directories()
        digest = hashlib.sha256()
        total_bytes = 0
        path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".upload-",
                suffix=".tmp",
                dir=self.settings.uploads_dir,
                delete=False,
            ) as temporary:
                path = Path(temporary.name)
                while block := stream.read(1024 * 1024):
                    total_bytes += len(block)
                    if total_bytes > self.settings.max_upload_bytes:
                        limit_mb = self.settings.max_upload_bytes // 1_048_576
                        raise UploadValidationError(f"Files must be no larger than {limit_mb} MB")
                    digest.update(block)
                    temporary.write(block)
            if total_bytes == 0:
                raise UploadValidationError("The uploaded file is empty")
        except Exception:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        if path is None:  # defensive; NamedTemporaryFile always assigns it
            raise RuntimeError("Could not create upload staging file")
        return path, digest.hexdigest()

    def _extract(self, path: Path, suffix: str) -> list[ExtractedSection]:
        if suffix == ".pdf":
            return self._extract_pdf(path)
        if suffix in {".txt", ".md"}:
            return self._extract_text(path)
        return [self._extract_image(path, suffix)]

    def _extract_pdf(self, path: Path) -> list[ExtractedSection]:
        try:
            with pymupdf.open(path) as document:
                if document.needs_pass:
                    raise UploadValidationError("Password-protected PDFs are not supported")
                sections = [
                    ExtractedSection(
                        content=normalize_text(page.get_text("text", sort=True)),
                        page_number=index + 1,
                        modality="text",
                    )
                    for index, page in enumerate(document)
                ]
        except UploadValidationError:
            raise
        except Exception as exc:
            raise UploadValidationError("The PDF could not be read") from exc

        sections = [section for section in sections if section.content]
        if not sections:
            raise ExtractionError(
                "No extractable PDF text was found; scanned PDFs are outside this project scope"
            )
        return sections

    def _extract_text(self, path: Path) -> list[ExtractedSection]:
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                raise UnicodeError
            content = normalize_text(raw.decode("utf-8-sig"))
        except UnicodeError as exc:
            raise UploadValidationError("Text files must use UTF-8 encoding") from exc
        if not content:
            raise ExtractionError("The text file contains no searchable content")
        return [ExtractedSection(content=content, page_number=None, modality="text")]

    def _extract_image(self, path: Path, suffix: str) -> ExtractedSection:
        expected_formats = {".png": {"PNG"}, ".jpg": {"JPEG"}, ".jpeg": {"JPEG"}}
        try:
            with Image.open(path) as image:
                actual_format = image.format
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise UploadValidationError("Images are limited to 20 megapixels")
                image.verify()
        except UploadValidationError:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise UploadValidationError("The image could not be read") from exc

        if actual_format not in expected_formats[suffix]:
            raise UploadValidationError("The image content does not match its file extension")
        analysis = self.model_gateway.analyze_image(path)
        parts = []
        if analysis.visible_text.strip():
            parts.append(f"Visible text:\n{normalize_text(analysis.visible_text)}")
        if analysis.description.strip():
            parts.append(f"Image description:\n{normalize_text(analysis.description)}")
        content = "\n\n".join(parts)
        if not content:
            raise ExtractionError("The image produced no searchable content")
        return ExtractedSection(content=content, page_number=None, modality="image")

    def _chunk_sections(self, sections: list[ExtractedSection]) -> list[ExtractedSection]:
        chunks: list[ExtractedSection] = []
        for section in sections:
            if section.modality == "image":
                chunks.append(section)
            else:
                chunks.extend(
                    chunk_section(
                        section,
                        chunk_words=self.settings.chunk_words,
                        overlap_words=self.settings.chunk_overlap_words,
                    )
                )
        if not chunks:
            raise ExtractionError("The file produced no searchable chunks")
        return chunks
