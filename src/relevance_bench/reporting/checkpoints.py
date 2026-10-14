"""Coverage summaries and report preparation from saved checkpoints."""

import json
from pathlib import Path

from datasets import Dataset

from ..checkpoints import (
    read_jsonl,
    reference_pairs,
    successful_judgments,
    validate_judgments,
)
from ..data_handlers.datasets import DatasetSpec


def coverage_report(qrels, rows: list[dict], failures=()) -> dict:
    """Derive coverage from successful pairs; logs only explain missing pairs."""
    expected = reference_pairs(qrels)
    validate_judgments(rows)
    completed = {(row["query_id"], row["passage_id"]) for row in rows}
    if not completed <= expected.keys():
        raise ValueError("Checkpoint contains unexpected pairs")
    if not completed:
        raise ValueError(
            "No judgments to report; checkpoint has no successful judgments"
        )
    missing = expected.keys() - completed
    diagnostics = {}
    for failure in failures:
        pair = (failure.get("query_id"), failure.get("passage_id"))
        if pair in missing:
            diagnostics.setdefault(pair, []).append(failure)
    missing_pairs = [
        {
            "query_id": pair[0],
            "passage_id": pair[1],
            "reference_relevance": relevance,
            "failures": diagnostics.get(pair, []),
        }
        for pair, relevance in expected.items()
        if pair in missing
    ]
    refusal_count = sum(
        any(
            answer.get("name") == "relevance" and answer.get("type") == "refusal"
            for failure in item["failures"]
            for answer in (failure.get("details") or {}).get("answers", [])
        )
        for item in missing_pairs
    )
    return {
        "status": "partial" if missing else "complete",
        "expected_judgments": len(expected),
        "completed_judgments": len(completed),
        "missing_judgments": len(missing),
        "judgment_fraction": len(completed) / len(expected),
        "expected_queries": len({pair[0] for pair in expected}),
        "evaluated_queries": len({pair[0] for pair in completed}),
        "missing_pairs": missing_pairs,
        "refusal_count": refusal_count,
    }


def report_checkpoint(
    output: Path,
    qrels,
    dataset: DatasetSpec,
    method: str,
    *,
    allow_partial: bool = False,
) -> Dataset:
    rows = read_jsonl(output / "judgments.jsonl")
    validate_judgments(rows)
    metadata_path = output / "experiment.json"
    if not metadata_path.exists():
        raise ValueError("Judgment checkpoint has no experiment metadata")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("dataset") != dataset.repository
        or metadata.get("dataset_name", dataset.name) != dataset.name
        or metadata.get("method", method) != method
    ):
        raise ValueError("Checkpoint dataset or method differs")
    completed = {(row["query_id"], row["passage_id"]): row for row in rows}
    return successful_judgments(completed, qrels, allow_partial=allow_partial)
