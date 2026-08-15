"""Run the local five-question retrieval evaluation with real Qwen embeddings."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from contextvault.config import Settings
from contextvault.database import Database
from contextvault.evaluation import calculate_metrics
from contextvault.ingestion import IngestionService
from contextvault.ollama_service import OllamaService
from contextvault.retrieval import RetrievalService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def filenames(items: list[object]) -> list[str]:
    return [item.chunk.filename if hasattr(item, "chunk") else item[0].filename for item in items]


def main() -> None:
    cases = json.loads((PROJECT_ROOT / "evaluation" / "cases.json").read_text())
    corpus_dir = PROJECT_ROOT / "evaluation" / "corpus"

    with tempfile.TemporaryDirectory(prefix="contextvault-evaluation-") as temporary:
        temporary_root = Path(temporary)
        settings = Settings(
            project_root=temporary_root,
            data_dir=temporary_root / "instance",
        )
        settings.ensure_runtime_directories()
        database = Database(settings.database_path)
        database.initialize()
        models = OllamaService(settings)
        ingestion = IngestionService(settings, database, models)
        retrieval = RetrievalService(settings, database, models)

        for source_path in sorted(corpus_dir.glob("*.md")):
            with source_path.open("rb") as source:
                ingestion.ingest(source_path.name, source)

        methods: dict[str, list[list[str]]] = {
            "keyword": [],
            "semantic": [],
            "hybrid": [],
        }
        expected = []
        rows = []
        for case in cases:
            rankings = retrieval.rankings(case["question"])
            ranked = {
                "keyword": filenames(list(rankings.lexical)),
                "semantic": filenames(list(rankings.semantic)),
                "hybrid": [hit.filename for hit in rankings.hybrid],
            }
            for method, documents in ranked.items():
                methods[method].append(documents)
            expected.append(case["expected_document"])
            rows.append(
                {
                    "question": case["question"],
                    "expected": case["expected_document"],
                    "top_keyword": ranked["keyword"][0] if ranked["keyword"] else "—",
                    "top_semantic": ranked["semantic"][0] if ranked["semantic"] else "—",
                    "top_hybrid": ranked["hybrid"][0] if ranked["hybrid"] else "—",
                }
            )

        print("\nContextVault retrieval evaluation\n")
        print(f"{'Method':<12} {'Hit Rate@3':>12} {'MRR':>8}")
        for method, rankings in methods.items():
            metrics = calculate_metrics(rankings, expected)
            print(
                f"{method:<12} {metrics.hit_rate_at_3:>12.3f} {metrics.mean_reciprocal_rank:>8.3f}"
            )
        print("\nPer-question top result")
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
