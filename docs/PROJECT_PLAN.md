# ContextVault Project Plan

## Objective

Build a small, local-first multimodal retrieval and RAG application for searching a private document collection without cloud inference.

ContextVault must index text-based PDFs, TXT/Markdown files, and standalone screenshots or text-heavy images. It must retrieve evidence with lexical and semantic search, fuse the rankings, and produce a short Qwen-generated answer with valid source citations.

## Optimization rule

Every feature must improve at least one of these outcomes without seriously harming the others:

1. Reliable end-to-end local operation.
2. Inspectable retrieval and grounding decisions.
3. Predictable behavior on the target laptop hardware.
4. Reproducible setup, evaluation, and automated testing.

## Hard constraints

- Single-user localhost application.
- FastAPI backend and vanilla HTML/CSS/JavaScript frontend.
- Ollama is installed as a system service and is accessed only through localhost.
- All application source, environments, databases, uploads, and generated artifacts stay in this repository.
- Ollama's executable and model store are machine prerequisites and are never committed to Git.
- No cloud inference or external data transmission during normal use.
- Target hardware: 16 GB RAM and an optionally enabled RTX 3050 Ti with 4 GB VRAM.
- Current workspace interpreter: Python 3.14; dependency compatibility must pass before application implementation.
- Verified GPU: RTX 3050 Ti (4 GB VRAM), NVIDIA driver 595.84; Ollama uses it automatically.
- The application remains CPU-compatible, but the verified configuration uses the GPU.
- Target corpus: no more than 20 documents or approximately 300 chunks.

## Required user flows

### 1. Index a source

The user uploads a supported file, receives a clear processing result, and sees the indexed document in the vault.

### 2. Ask a question

The user enters a question and receives either a concise evidence-grounded answer or a clear insufficient-evidence response.

### 3. Inspect evidence

The user can see source excerpts, PDF page numbers or image labels, lexical rank, semantic similarity, fused rank, and which sources were supplied to the generator.

## Scope

### Required

- Text extraction from normal PDFs.
- TXT and Markdown ingestion.
- Standalone PNG/JPG/JPEG ingestion.
- Qwen-generated visible-text transcription and concise description for images.
- Deterministic word-based chunking that preserves PDF page boundaries.
- SQLite document/chunk metadata and FTS5 lexical index.
- Qwen embeddings stored locally as float32 vectors.
- Exact cosine similarity for the small corpus.
- Reciprocal Rank Fusion of lexical and semantic rankings.
- Top-evidence construction with stable source identifiers.
- Evidence-only Qwen answer generation.
- Citation validation and insufficient-evidence behavior.
- FastAPI status, document, upload, delete, and query endpoints.
- One responsive browser page.
- Model-free automated tests and CI.
- Small reproducible retrieval evaluation.
- README, architecture guide, technical walkthrough, and evaluation records.

### Explicitly excluded

- Scanned-PDF OCR or extraction of images embedded inside PDFs.
- Audio, video, native joint image-text embeddings, or neural reranking.
- Authentication, multiple users, chat memory, agents, or tool calling.
- LangChain, LlamaIndex, vector databases, ORM frameworks, background queues, WebSockets, or microservices.
- Docker, cloud APIs, public deployment, or fine-tuning.

## Planned models

- Embeddings: `qwen3-embedding:0.6b`.
- Vision and generation: `qwen3.5:2b-q4_K_M`.
- Rejected baseline: `qwen3.5:0.8b`, replaced after repeated focused-answer failures in live PDF tests.

Model identifiers must be configurable without code changes.

## Milestones and gates

### Gate 0: environment compatibility

Pass conditions:

- A repository-local virtual environment can be created.
- FastAPI, Uvicorn, Pydantic, NumPy, PyMuPDF, Pillow, Ollama's Python client, pytest, HTTPX, and Ruff install on the selected interpreter.
- SQLite FTS5 is available.
- A minimal test suite can run.

Failure response: select a project-local compatible Python runtime. Do not change the system interpreter.

### Gate 1: local-model feasibility

Pass conditions:

- Ollama is reachable through localhost.
- The embedding model returns a consistent vector dimension and finite values.
- The selected Qwen model transcribes a representative textual image sufficiently for retrieval.
- The selected model generates a short answer constrained to supplied evidence.
- No out-of-memory failure occurs.
- Warm response latency is acceptable for interactive use; actual measurements are recorded.

Failure response: benchmark the next candidate or use CPU fallback. Do not design the rest of the application around an unverified model assumption.

### Gate 2: deterministic retrieval core

Pass conditions:

- Text sources produce page-aware chunks.
- Database and FTS rows remain consistent.
- Semantic and lexical retrieval pass deterministic unit tests.
- Reciprocal Rank Fusion returns stable ordering.
- Deleting a document removes all related searchable data.

### Gate 3: grounded RAG

Pass conditions:

- Only retrieved evidence is placed in the generation prompt.
- Source labels map back to stored chunks.
- Unknown citations are rejected.
- Weak or absent evidence produces a refusal.

### Gate 4: user experience

Pass conditions:

- The three required user flows work from the browser.
- Model and processing failures are actionable rather than raw exceptions.
- The evidence trace is readable and useful during verification and debugging.

### Gate 5: repository quality

Pass conditions:

- Model-free tests pass in CI.
- Evaluation reports actual keyword-only, semantic-only, and hybrid results.
- Git ignores the virtual environment, uploaded files, database, caches, and model artifacts.
- Documentation matches the code and makes no unsupported originality or privacy claims.
- A clean clone can be set up by following the README.

## Risk register

| Risk | Early signal | Mitigation |
|---|---|---|
| Python 3.14 wheel incompatibility | Dependency installation fails | Use a project-local supported Python runtime |
| Ollama service unavailable | Status probe fails | Show an actionable health error and documented service-start command |
| GPU acceleration unavailable | Ollama reports CPU execution or high latency | Keep CPU compatibility and expose processor state in setup diagnostics |
| Local-model answer quality is weak | Focused answers fail despite relevant evidence | Isolate retrieval from generation, benchmark the 2B model, and simplify the response contract |
| Two models exceed GPU capacity | Eviction, swapping, or OOM | Use `keep_alive=0` at model boundaries so only one model remains resident |
| Long synchronous requests | Upload/query feels stalled | Keep small corpus/input limits and show explicit progress states; do not add a queue in v1 |
| Hallucinated citations | Model emits unknown source IDs | Validate citations and refuse unverified output |
| Retrieval threshold is arbitrary | False answers or excessive refusal | Calibrate against the included evaluation set and document limitations |
| CI cannot run models | Tests fail off-machine | Inject a deterministic fake model boundary |
| Public repository leaks data | Runtime files appear in Git status | Strong `.gitignore`, synthetic sample data, and final tracked-file audit |

## Verification matrix

| Claim | Verification |
|---|---|
| Runs locally | End-to-end browser test against localhost |
| Works after model setup without cloud inference | Review outbound integrations and run with network unavailable where practical |
| Supports PDFs and images | Integration tests plus sample-data demonstration |
| Hybrid retrieval is beneficial | Compare Hit Rate@3 and MRR for lexical, semantic, and hybrid retrieval |
| Answers are grounded | Citation validator tests and unanswerable evaluation questions |
| Repository is reproducible | Clean-environment setup verification and CI |

## Build order

1. Pass Gate 0 and Gate 1.
2. Create the application skeleton and model boundary.
3. Implement database and text ingestion.
4. Implement deterministic retrieval and pass Gate 2.
5. Implement grounding and pass Gate 3.
6. Add image ingestion through the already-verified model.
7. Build the browser interface and pass Gate 4.
8. Add evaluation, CI, documentation, and pass Gate 5.

Work must not move past a failed gate without recording the decision and adjustment.
