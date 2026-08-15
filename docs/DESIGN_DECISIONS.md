# ContextVault Design Decisions

## FastAPI instead of Flask

FastAPI provides typed Pydantic contracts, automatic input validation, OpenAPI output, and straightforward API testing. The application still uses synchronous route functions for blocking local work and does not introduce an async database stack.

## Ollama instead of direct Transformers

Ollama reduces model-loading, quantization, and GPU scheduling code. ContextVault treats the system-installed service as a separate localhost inference boundary. The application itself does not install or modify Ollama, drivers, services, users, or system configuration. This keeps the educational focus on ingestion, retrieval, grounding, evaluation, and API design.

## Caption-mediated visual retrieval

Screenshots are transcribed and described by a local vision-language model. The resulting text is embedded into the same search space as document chunks. Native multimodal embeddings are deferred because they add runtime and conceptual complexity without being required to demonstrate a complete multimodal user flow.

## SQLite and NumPy instead of a vector database

The target corpus is small enough for exact similarity search. SQLite gives durable metadata and FTS5, while NumPy makes the vector calculation visible and testable. This avoids introducing infrastructure that the project cannot justify at its scale.

## Hybrid retrieval with Reciprocal Rank Fusion

Keyword and semantic retrieval have complementary failure modes and incompatible raw score scales. Reciprocal Rank Fusion combines rank positions instead of attempting to calibrate unrelated score values.

## Evidence-first generation

Generation is downstream of retrieval and may not invent supporting evidence. The model returns a schema-constrained direct answer plus source IDs; `NONE` represents refusal, and application code derives sufficiency instead of asking the model for a redundant Boolean. ContextVault validates source IDs, rejects response-metadata phrases in the answer field, and renders citations itself. This contract replaced an earlier design in which small models sometimes wrote `sufficient true` as the answer. A fluent answer without valid evidence is considered a failed result.

## Sequential model residency

The verified embedding model uses about 2.4 GB at runtime and the final 2B Q4 generation model uses about 1.6 GB at a 4096-token context. Both ran fully on the GPU, but retaining them together would consume the 4 GB device's available memory. Requests therefore use `keep_alive=0` and accept model-switching latency in exchange for reliability.

## Model-free automated tests

Tests replace Ollama with deterministic responses. Real-model accuracy and latency belong in an explicit local benchmark, not in CI. This keeps the repository reproducible for contributors without the same hardware or models.
