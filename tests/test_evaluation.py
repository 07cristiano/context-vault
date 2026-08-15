import pytest

from contextvault.evaluation import calculate_metrics


def test_retrieval_metrics_deduplicate_document_ranks() -> None:
    metrics = calculate_metrics(
        rankings=[
            ["wrong.md", "expected.md", "expected.md"],
            ["second.md", "other.md"],
        ],
        expected=["expected.md", "second.md"],
    )

    assert metrics.hit_rate_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == pytest.approx(0.75)


def test_retrieval_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_metrics([[]], [])
