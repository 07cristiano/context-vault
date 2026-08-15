# ContextVault

ContextVault is a small, offline, evidence-first multimodal retrieval and RAG application. It indexes text PDFs, Markdown/TXT files, and text-heavy screenshots; combines keyword and semantic search; and produces short local answers with inspectable source citations.

It is intentionally scoped to one user, a small corpus, and laptop-class hardware. Instead of imitating a production-scale search platform, it keeps the retrieval, ranking, grounding, and model-lifecycle decisions visible and testable.

## What makes it interesting

- **Inspectable hybrid retrieval:** every ranked trace item exposes a chunk preview, keyword rank, semantic rank, cosine similarity, and Reciprocal Rank Fusion score.
- **Application-owned citations:** Qwen returns schema-constrained source IDs or `NONE`; application code validates the IDs and renders the citations.
- **Inspectable evidence:** selected chunks start collapsed but can be expanded to their complete stored text.
- **Multimodal without a second search system:** screenshots are locally transcribed and described, then embedded into the same text search space as PDFs.
- **Hardware-aware model lifecycle:** embedding and generation models are unloaded at operation boundaries so they do not compete for 4 GB of VRAM.
- **Measured rather than claimed:** the repository includes real-model feasibility measurements and a reproducible keyword-vs-semantic-vs-hybrid evaluation.

## Demonstrated result

The verified laptop configuration is an RTX 3050 Ti with 4 GB VRAM, NVIDIA driver 595.84, and Ollama 0.32.12.

- `qwen3-embedding:0.6b`: 1024-dimensional embeddings, 100% GPU.
- `qwen3.5:2b-q4_K_M`: 2.27B-parameter text-and-vision generation, measured at 1.6 GB runtime and 100% GPU with a 4096-token context.
- Live parsable-PDF check: correct database definition, citation `S1`, and page-5 provenance in approximately 7.0 seconds.
- Live screenshot checks: exact visible status with citation `S1`, plus a correct refusal for an absent budget; each query took approximately 6.8 seconds.
- Retrieval evaluation: hybrid Hit Rate@3 `1.000` and MRR `1.000` on the included five-question synthetic set.

These are local measurements, not general model-performance claims. See [model evaluation](docs/MODEL_EVALUATION.md) and [retrieval evaluation](docs/EVALUATION.md).

## Architecture

```text
Browser (vanilla HTML/CSS/JS)
              |
              v
        FastAPI routes
          /         \
         v           v
   Ingestion       Query/RAG
   - PyMuPDF       - SQLite FTS5 / BM25
   - Pillow        - exact NumPy cosine
   - Qwen vision   - Reciprocal Rank Fusion
         \          - evidence threshold
          v         - citation validation
        SQLite              |
   documents + chunks       v
   embeddings + FTS5   Ollama / Qwen
```

One FastAPI process owns application logic and one SQLite database. Ollama is a separate service bound to localhost. There is no vector database, orchestration framework, cloud API, frontend framework, or ORM.

## Technology stack

- Python 3.11–3.14
- FastAPI, Uvicorn, and Pydantic
- SQLite with FTS5/BM25
- NumPy exact cosine similarity
- PyMuPDF and Pillow
- Ollama
- `qwen3-embedding:0.6b`
- `qwen3.5:2b-q4_K_M`
- Vanilla HTML, CSS, and JavaScript
- pytest, Ruff, and GitHub Actions

## Setup

### 1. Prerequisites

Install Python and [Ollama](https://ollama.com/), then verify that Ollama is running:

```bash
ollama --version
curl http://127.0.0.1:11434/api/version
```

Pull the two local models once:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:2b-q4_K_M
```

The downloads require internet access. Normal application use does not.

### 2. Create the project environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Start ContextVault

```bash
uvicorn contextvault.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The application writes only to the Git-ignored `instance/` directory by default.

## Recommended demo

1. Upload [contextvault_demo.md](sample_data/contextvault_demo.md).
2. Ask: `Who is the Project Atlas project lead?`
3. Open the evidence and retrieval trace.
4. Ask: `What revenue target was assigned to Project Atlas?`
5. Show the explicit `Insufficient evidence.` response.
6. Upload [contextvault_dashboard.png](sample_data/contextvault_dashboard.png).
7. Ask: `What is the audit status of project ORCHID-72?`

The first request after a model has been unloaded can take longer because Ollama must load it into GPU memory.

## Retrieval algorithm

1. Normalize the question and build a safe FTS5 query from non-stopwords.
2. Rank keyword matches with SQLite BM25.
3. Embed the question with Qwen.
4. Calculate exact cosine similarity against every stored vector.
5. Combine the two rank lists with Reciprocal Rank Fusion:

   ```text
   RRF score(document) = sum(1 / (60 + rank_in_each_list))
   ```

6. Select at most three evidence chunks whose fused score is at least 85% of the best result.
7. Ask Qwen for a schema-constrained direct answer and source IDs.
8. Derive sufficiency from the citations and reject unknown, missing, or metadata-only answers.

Exact vector search is appropriate because the project is capped at 300 chunks. An approximate index would add complexity without solving a measured problem.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/status` | Check database, Ollama, and model readiness |
| `GET` | `/api/documents` | List indexed sources |
| `POST` | `/api/documents` | Validate, extract, chunk, embed, and index one file |
| `DELETE` | `/api/documents/{id}` | Remove a source and all searchable data |
| `POST` | `/api/query` | Retrieve evidence and return a grounded answer or refusal |

Interactive OpenAPI documentation is available at `/docs` while the app runs.

## Quality checks

Model-free checks used in CI:

```bash
ruff format --check .
ruff check .
pytest
```

Run the real embedding evaluation locally:

```bash
python scripts/evaluate_retrieval.py
```

CI does not download or execute AI models. Tests inject deterministic fake model responses, while real-model behavior belongs in explicit local evaluation.

## Scope and limitations

- Intended for at most 20 documents or approximately 300 chunks.
- Supports text PDFs, not scanned-PDF OCR or embedded PDF images.
- Screenshot quality is best for clean, text-heavy images.
- The 2B model has been evaluated primarily on focused evidence questions; compound-question completeness still requires broader evaluation.
- RRF is a rank-fusion score, not calibrated confidence. The relative evidence rule can supply an irrelevant near-ranked chunk, although only model-cited labels are rendered as citations.
- Retrieval thresholds were selected for the included small evaluation and should be recalibrated for a different corpus.
- No authentication or network hardening is included; keep the server bound to `127.0.0.1`.
- “Offline” means no cloud inference during normal use. Initial software and model downloads still require a network.

## Repository map

```text
contextvault/
  api.py              HTTP contracts and error translation
  config.py           validated local configuration
  database.py         SQLite schema, FTS5, and persistence
  ingestion.py        validation, extraction, chunking, indexing
  retrieval.py        BM25, cosine similarity, and RRF
  rag.py              evidence selection and grounding
  ollama_service.py   only module allowed to call Ollama
  static/             browser interface
evaluation/           synthetic corpus and question set
tests/                model-free unit and API tests
docs/                 architecture, decisions, evaluations, and technical walkthrough
```

## Developer documentation

Follow the [technical walkthrough](docs/TECHNICAL_WALKTHROUGH.md) for an end-to-end tour. The [architecture](docs/ARCHITECTURE.md), [design decisions](docs/DESIGN_DECISIONS.md), [model evaluation](docs/MODEL_EVALUATION.md), and [retrieval evaluation](docs/EVALUATION.md) record the system boundaries, tradeoffs, and measured behavior.

## License

[MIT](LICENSE)
