# ContextVault Architecture

## System boundary

ContextVault is one FastAPI process with one SQLite database and a static browser frontend. Ollama runs as a separate system service bound to localhost. The application never owns GPU drivers or model scheduling.

```text
Browser
  -> FastAPI API
     -> ingestion service -> PyMuPDF / Pillow -> Ollama vision
     -> retrieval service -> SQLite FTS5 + NumPy cosine similarity
     -> RAG service -> Ollama generation -> citation validator
     -> SQLite documents, chunks, embeddings, and source metadata
```

## Component responsibilities

### `main.py`

Creates the FastAPI application, initializes local paths and the database, mounts static assets, and registers API routes.

### `api.py`

Translates HTTP inputs into service calls and service errors into stable HTTP responses. It does not contain retrieval or model logic.

### `schemas.py`

Defines Pydantic request and response contracts. Internal storage rows are not returned directly.

### `database.py`

Owns the SQLite schema, transactions, vector serialization, FTS synchronization, document listing, and cascading deletion.

### `ingestion.py`

Validates files, extracts text, invokes visual analysis, chunks content, requests embeddings, and persists a complete document transactionally.

### `ollama_service.py`

Is the only application module allowed to call Ollama. It exposes narrow operations for health/model checks, embeddings, visual analysis, and grounded generation. Tests replace this boundary with a deterministic fake.

### `retrieval.py`

Runs lexical and semantic retrieval, combines rankings with Reciprocal Rank Fusion, and returns traceable evidence candidates.

### `rag.py`

Applies the evidence threshold, assigns stable source labels, builds the constrained prompt, validates returned citations, and constructs the final answer response.

## Data model

### Documents

- Stable integer ID.
- Original and safe stored filename.
- File type and content hash.
- Processing status, timestamp, and optional error.

### Chunks

- Stable integer ID and parent document ID.
- Page number when applicable.
- Chunk position and modality (`text` or `image`).
- Searchable content.
- Float32 embedding stored as a SQLite BLOB.
- Optional original-image path.

### FTS index

Stores the searchable chunk text and maps each row back to the canonical chunk ID. Database writes keep canonical and FTS records consistent in the same transaction.

## Ingestion invariants

- An indexed document has at least one non-empty chunk.
- Every chunk has exactly one embedding with the configured dimension.
- A PDF chunk never spans multiple pages.
- Failed processing does not leave partially searchable chunks.
- Duplicate content hashes are rejected before expensive model work.
- User-provided filenames are never used directly as storage paths.

## Query algorithm

1. Validate and normalize the question.
2. Retrieve lexical candidates from FTS5/BM25.
3. Generate one question embedding and calculate exact cosine similarities.
4. Convert both result sets to rank positions.
5. Fuse them with Reciprocal Rank Fusion.
6. Apply the evidence gate and choose at most three sources within 85% of the best fused score.
7. Assign labels `S1` through `S3`.
8. Generate a schema-constrained answer and list of source labels.
9. Validate every returned source label and render citations in application code.
10. Return the answer, evidence, timing, and complete retrieval trace.

## API contract

- `GET /api/status`: application, database, Ollama, and model readiness.
- `GET /api/documents`: indexed document summaries.
- `POST /api/documents`: one supported multipart upload.
- `DELETE /api/documents/{document_id}`: source and index removal.
- `POST /api/query`: question, answer/refusal, evidence, retrieval trace, and timings.

Ollama, filesystem, PDF, image, and SQLite calls are blocking. Corresponding FastAPI path operations use normal `def` functions so the event loop is not blocked directly.

## Offline boundary

Installing Ollama and running `ollama pull` require network access. During normal application use, ContextVault calls only its local database, local files, and the configured localhost Ollama endpoint. The frontend contains no CDN assets.

## Model memory lifecycle

The target GPU has 4 GB of VRAM. Embedding and generation requests use `keep_alive=0`, so Ollama unloads each model at the boundary instead of retaining both models simultaneously. This favors predictable memory use over minimum cold-start latency.

## Scaling boundary

Exact NumPy search is intentionally selected for the small corpus. A future design would move vectors to an approximate nearest-neighbor index only after profiling demonstrates the need. The application-level evidence and RAG contracts should remain unchanged across that replacement.
