# ContextVault Technical Walkthrough

This walkthrough follows a source and a question through ContextVault. It is organized by runtime boundaries so contributors can inspect, test, or replace one layer without treating the RAG pipeline as a black box.

## 1. Run the representative flow

1. Start Ollama and ContextVault.
2. Upload `sample_data/contextvault_demo.md`.
3. Ask `Who leads Project Atlas?`.
4. Open the evidence and retrieval trace.
5. Ask for a revenue target that is absent from the source and observe the refusal.
6. Upload `sample_data/contextvault_dashboard.png` and ask for the ORCHID-72 audit status.
7. Expand the image evidence to inspect the local vision transcription.

The two primary paths are:

```text
upload -> validate -> extract -> chunk -> embed -> store
question -> keyword search + vector search -> RRF -> evidence -> Qwen -> citation validation
```

## 2. Application and configuration boundaries

Start with:

1. `contextvault/main.py`
2. `contextvault/api.py`
3. `contextvault/schemas.py`
4. `contextvault/config.py`

`main.py` assembles one FastAPI application from explicit services. `api.py` translates domain errors into HTTP responses, while `schemas.py` defines the public contracts. `config.py` validates repository-local paths, model identifiers, corpus limits, and the localhost-only Ollama endpoint.

Important invariants:

- `create_app()` accepts a model gateway so automated tests do not need Ollama.
- Routes are synchronous because SQLite, PyMuPDF, filesystem access, and Ollama calls are blocking.
- Runtime writes stay under `instance/` by default.
- Ollama must use an HTTP localhost address.

## 3. Ingestion and storage

Read `contextvault/ingestion.py` and `contextvault/database.py` together.

The ingestion path:

1. Sanitizes the display filename and validates its extension.
2. Streams the upload into a randomly named staging file while calculating SHA-256.
3. Rejects duplicate content before an embedding request.
4. Extracts page-aware PDF text, UTF-8 text, or a local image transcription.
5. Creates deterministic overlapping word chunks.
6. Embeds the chunks through the model gateway.
7. Stores the document and every chunk in one SQLite transaction.

User-provided filenames never become storage paths. PDF chunks do not cross page boundaries. SQLite foreign keys and FTS5 triggers keep canonical chunks, lexical search rows, and deletion behavior consistent.

## 4. Retrieval

`contextvault/retrieval.py` implements two independent rankers.

### Keyword ranking

SQLite FTS5 finds exact terms and BM25 ranks them using term frequency and rarity. User questions are converted into quoted FTS terms so query syntax cannot be injected directly.

### Semantic ranking

The Qwen embedding model maps the question into the same 1024-dimensional space as stored chunks. Because the corpus is capped at 300 chunks, ContextVault loads the vectors into NumPy and calculates exact cosine similarity:

```text
cosine(q, d) = (q dot d) / (length(q) * length(d))
```

### Reciprocal Rank Fusion

BM25 and cosine scores have incompatible scales. RRF combines their rank positions instead:

```text
score = 1/(60 + keyword_rank) + 1/(60 + semantic_rank)
```

A chunk returned by both rankers receives both contributions. The fused score orders candidates; it is not a calibrated probability of relevance. Run `python scripts/evaluate_retrieval.py` to reproduce the included keyword, semantic, and hybrid comparison.

## 5. Evidence selection and grounded generation

Read `contextvault/rag.py`, `contextvault/model_gateway.py`, and `contextvault/ollama_service.py`.

Responsibilities remain separate:

- Retrieval ranks stored chunks.
- The RAG service selects at most three near-top candidates and assigns temporary labels such as `S1`.
- Qwen turns supplied evidence into a concise answer.
- Application code validates source labels and constructs the final response.

The model returns an `answer` plus supplied source IDs, or `NONE` when the requested fact is absent. ContextVault derives sufficiency from that choice, rejects unknown IDs and metadata-only answers, and renders citations from application data.

`docs/MODEL_EVALUATION.md` records why the response contract was reduced to these two fields and why the 2B quantized model replaced the smaller baseline.

## 6. Failure boundaries and tests

Representative tests cover:

- `test_database.py`: FTS synchronization and cascade deletion.
- `test_ingestion.py`: chunking, PDF pages, duplicates, and image ingestion.
- `test_retrieval.py`: safe FTS queries, semantic ranking, and RRF traces.
- `test_rag.py`: grounded answers, evidence thresholds, and refusal behavior.
- `test_ollama_service.py`: structured model responses and citation validation.
- `test_api.py`: complete HTTP flows through a deterministic fake gateway.

CI deliberately avoids model downloads and GPU requirements. Real model behavior and hardware measurements are kept in explicit local evaluations, while deterministic application behavior remains in pytest.

## 7. Extension points

The current boundaries allow focused changes without rewriting the entire application:

- Calibrate the evidence threshold against a larger labeled dataset.
- Add a reranker between RRF and evidence selection.
- Replace exact NumPy search when the corpus limit grows substantially.
- Move synchronous ingestion to a background job for larger files.
- Add scanned-PDF OCR as a separate extraction capability.

Any extension should preserve source provenance, deterministic refusal behavior, and the separation between retrieval, generation, and validation.
