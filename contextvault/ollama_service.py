"""All communication with the localhost Ollama service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ollama import Client
from pydantic import BaseModel, ConfigDict, ValidationError

from contextvault.config import Settings
from contextvault.errors import ModelResponseError, ModelUnavailableError
from contextvault.model_gateway import (
    GeneratedAnswer,
    ModelStatus,
    PromptSource,
    VisualAnalysis,
)


class _VisualPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_text: str
    description: str


class _AnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str]


class OllamaService:
    """Validated model operations; no other module should call Ollama directly."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client or Client(host=settings.ollama_host)

    def status(self) -> ModelStatus:
        try:
            response = self.client.list()
            installed = tuple(sorted(model.model for model in response.models if model.model))
        except Exception as exc:  # external service boundary
            return ModelStatus(
                reachable=False,
                embedding_ready=False,
                generation_ready=False,
                detail=f"Ollama is unreachable: {exc}",
            )

        return ModelStatus(
            reachable=True,
            embedding_ready=self.settings.embedding_model in installed,
            generation_ready=self.settings.generation_model in installed,
            installed_models=installed,
            detail="Ollama is reachable",
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding input must contain non-empty text")

        batches: list[np.ndarray] = []
        batch_size = self.settings.embedding_batch_size
        try:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                is_last = start + batch_size >= len(texts)
                response = self.client.embed(
                    model=self.settings.embedding_model,
                    input=batch,
                    truncate=False,
                    keep_alive=0 if is_last else "2m",
                )
                vectors = np.asarray(response.embeddings, dtype=np.float32)
                expected_shape = (len(batch), self.settings.embedding_dimension)
                if vectors.shape != expected_shape:
                    raise ModelResponseError(
                        f"Expected embedding shape {expected_shape}, received {vectors.shape}"
                    )
                if not np.isfinite(vectors).all():
                    raise ModelResponseError("Embedding response contains non-finite values")
                batches.append(vectors)
        except ModelResponseError:
            raise
        except Exception as exc:  # external service boundary
            raise ModelUnavailableError(f"Embedding request failed: {exc}") from exc

        return np.concatenate(batches, axis=0)

    def analyze_image(self, image_path: Path) -> VisualAnalysis:
        if not image_path.is_file():
            raise ValueError("Image path does not exist")

        try:
            response = self.client.chat(
                model=self.settings.generation_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Transcribe all visible text exactly, then give one short factual "
                            "description. Do not infer facts that are not visible."
                        ),
                        "images": [str(image_path)],
                    }
                ],
                format=_VisualPayload.model_json_schema(),
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": self.settings.model_context_tokens,
                    "num_predict": 500,
                },
                keep_alive=0,
            )
            payload = _VisualPayload.model_validate_json(response.message.content)
        except ValidationError as exc:
            raise ModelResponseError("Image analysis did not match the required schema") from exc
        except ModelResponseError:
            raise
        except Exception as exc:  # external service boundary
            raise ModelUnavailableError(f"Image analysis failed: {exc}") from exc

        visible_text = payload.visible_text.strip()
        description = payload.description.strip()
        if not visible_text and not description:
            raise ModelResponseError("Image analysis returned no searchable content")
        return VisualAnalysis(visible_text=visible_text, description=description)

    def generate_answer(self, question: str, sources: list[PromptSource]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not sources:
            raise ValueError("At least one evidence source is required")

        labels = [source.label for source in sources]
        if len(labels) != len(set(labels)):
            raise ValueError("Evidence labels must be unique")

        schema = _AnswerPayload.model_json_schema()
        schema["properties"]["citations"]["items"] = {
            "type": "string",
            "enum": [*labels, "NONE"],
        }
        schema["properties"]["citations"]["minItems"] = 1
        schema["properties"]["citations"]["maxItems"] = len(labels)
        schema["properties"]["answer"]["description"] = (
            "Only the direct answer to the question, never a field name or decision phrase"
        )
        schema["properties"]["citations"]["description"] = (
            "Supporting source IDs, or only NONE when the evidence does not answer the question"
        )
        evidence = "\n\n".join(f"[{source.label}]\n{source.content.strip()}" for source in sources)
        prompt = (
            "Answer the question using only the evidence below. Put only the direct response "
            "to the question in the answer field. Never put JSON field names, booleans, or "
            "decision phrases in the answer. Cite only sources that directly support the "
            "answer. If the requested information is not explicitly present, use the exact "
            "answer 'Insufficient evidence.' and citations ['NONE']. Do not combine NONE with "
            "a source ID.\n\n"
            f"Evidence:\n{evidence}\n\nQuestion: {question.strip()}"
        )

        try:
            response = self.client.chat(
                model=self.settings.generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the fact requested by the question from the supplied "
                            "evidence. Do not discuss whether the evidence is sufficient and "
                            "never add outside facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                format=schema,
                think=False,
                options={
                    "temperature": 0,
                    "num_ctx": self.settings.model_context_tokens,
                    "num_predict": 300,
                },
                keep_alive=0,
            )
            payload = _AnswerPayload.model_validate_json(response.message.content)
        except ValidationError as exc:
            raise ModelResponseError("Generated answer did not match the required schema") from exc
        except ModelResponseError:
            raise
        except Exception as exc:  # external service boundary
            raise ModelUnavailableError(f"Answer generation failed: {exc}") from exc

        raw_citations = tuple(dict.fromkeys(payload.citations))
        unknown = set(raw_citations) - {*labels, "NONE"}
        if unknown:
            raise ModelResponseError("Generated answer contains an unknown citation")
        if "NONE" in raw_citations and raw_citations != ("NONE",):
            raise ModelResponseError("NONE cannot be combined with a source citation")
        sufficient = raw_citations != ("NONE",)
        citations = raw_citations if sufficient else ()

        answer = payload.answer.strip()
        if not answer:
            raise ModelResponseError("Generated answer is empty")
        metadata_answer = " ".join(answer.casefold().split()).strip(".:")
        if metadata_answer in {"sufficient true", "sufficient false", "true", "false"}:
            raise ModelResponseError(
                "Generated answer contains response metadata instead of an answer"
            )
        return GeneratedAnswer(
            answer=answer if sufficient else "Insufficient evidence.",
            citations=citations,
            sufficient=sufficient,
        )
