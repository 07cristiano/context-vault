"""Narrow, replaceable boundary between application logic and local models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelStatus:
    reachable: bool
    embedding_ready: bool
    generation_ready: bool
    installed_models: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True, slots=True)
class VisualAnalysis:
    visible_text: str
    description: str


@dataclass(frozen=True, slots=True)
class PromptSource:
    label: str
    content: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    citations: tuple[str, ...]
    sufficient: bool


class ModelGateway(Protocol):
    def status(self) -> ModelStatus: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def analyze_image(self, image_path: Path) -> VisualAnalysis: ...

    def generate_answer(self, question: str, sources: list[PromptSource]) -> GeneratedAnswer: ...
