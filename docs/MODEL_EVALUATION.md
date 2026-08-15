# Local Model Feasibility Record

## Environment

- Evaluation date: 2026-08-15.
- Ollama: 0.32.12, running as a system service on `127.0.0.1:11434`.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4096 MiB VRAM.
- NVIDIA driver: 595.84.
- Project generation context: 4096 tokens.

These measurements are specific to the development laptop. They are evidence that the selected models fit this machine, not universal performance guarantees.

## Embedding model

Model: `qwen3-embedding:0.6b`

- Disk size reported by `ollama list`: 639 MB.
- Embedding dimension: 1024.
- A two-input probe returned two finite, normalized vectors.
- Runtime reported by `ollama ps`: approximately 2.4 GB, using the GPU.
- First measured request: 23.438 seconds including cold model loading.

Result: pass.

## Generation-model decision

### Rejected baseline: `qwen3.5:0.8b`

The 0.8B model fit easily and passed a narrow synthetic screenshot test. It was rejected after repeated failures on ordinary, parsable PDF text:

- It answered a natural-language-processing question with only `yes`.
- It answered a Code-A-Thon rank question with a neighboring DSA-problem count.
- Under the first structured contract, it sometimes copied `sufficient true` into the answer field.

These failures were not caused by missing PDF text. Relevant passages were visible in the retrieval trace, which isolated the weakness to generation and the response contract.

### Selected model: `qwen3.5:2b-q4_K_M`

- Parameter count reported by the model metadata: approximately 2.27 billion.
- Quantization: `Q4_K_M`.
- Disk size reported by `ollama list`: approximately 1.9 GB.
- Runtime reported by `ollama ps`: approximately 1.6 GB at a 4096-token context, using the GPU.
- Measured cold probe: 4.584 seconds total, of which 4.508 seconds was model loading.

The 2B model is the smallest tested model that produced acceptable focused answers after the response contract was simplified. It leaves enough headroom on the 4 GB GPU because ContextVault sets `keep_alive=0` at model boundaries instead of keeping the embedding and generation models resident together.

Result: selected for the final project configuration.

## Generation contract experiment

The original schema asked the model for `answer`, `citations`, and a redundant `sufficient` Boolean. Both tested small models sometimes placed the Boolean metadata in the answer or refused despite supporting evidence.

Controlled probes established the following:

1. The retrieved passage itself contained the answer.
2. The 2B model answered correctly when given the same passage with a minimal prompt.
3. The final contract—`answer` plus `citations` only—also worked with the production prompt.
4. Application code can derive sufficiency more reliably: valid source IDs mean supported; `NONE` means refusal.

ContextVault now validates every returned source ID, rejects metadata-like answers such as `sufficient true`, and renders citations itself.

## Live end-to-end grounding checks

These checks used the complete FastAPI -> hybrid retrieval -> Ollama -> citation-validation path.

| Question | Observed result | Verification |
|---|---|---|
| `What is a database?` | Correct definition from the MySQL handbook | Citation `S1` mapped to the displayed page-5 passage |
| `What rank did he get in internal round of SIH?` | `Rank 7` | Citation `S1`; the expandable full chunk contains the Smart India Hackathon statement and rank |
| `Who leads Project Atlas?` | `Ananya Rao` | Citation `S1` mapped to the synthetic Markdown source |
| `Which delivery risk remains open for Project Atlas?` | Dependency security review is not complete | Citation `S1` mapped to the exact negative statement |
| Absent Project Atlas revenue target | `Insufficient evidence.` | Relevant source was retrieved, but the model returned no citation |
| ORCHID-72 screenshot audit status | `READY FOR REVIEW` | Exact visible text, image citation `S1`, 6.789 seconds total |
| Absent ORCHID-72 budget | `Insufficient evidence.` | The image was retrieved, but it contains no budget; no citation was returned |
| Unsupported favorite-color question | `Insufficient evidence.` | No evidence was sent to generation |

The rank-fusion score is useful for ordering results but is not a calibrated probability. Because Reciprocal Rank Fusion compresses nearby ranks, the relative evidence rule can still include a weak third passage. The interface exposes all supplied evidence so that this is inspectable rather than hidden.

## Screenshot-path result

The final `qwen3.5:2b-q4_K_M` configuration was tested with the repository's 1200x700 high-contrast [dashboard sample](../sample_data/contextvault_dashboard.png). It transcribed all three displayed fields correctly:

- Project owner: `Priya Nair`.
- Audit status: `READY FOR REVIEW`.
- Review date: `22 August 2026`.

The complete application then answered the supported status question with the exact text and image citation `S1`. When asked for a budget that was not visible, it returned `Insufficient evidence.` with no citation even though the image was retrieved.

Result: pass for the promised clean, text-heavy screenshot scope. This does not claim general OCR accuracy; difficult layouts, handwriting, and scanned PDFs remain outside the promised scope.

## Final resource decision

Use:

- `qwen3-embedding:0.6b` for dense retrieval.
- `qwen3.5:2b-q4_K_M` for screenshot transcription and grounded answer generation.

Both identifiers remain configurable through environment variables. The repository does not contain Ollama models, and it makes no system-level changes.
