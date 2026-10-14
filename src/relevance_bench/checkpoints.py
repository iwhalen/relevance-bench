"""Durable judgments and experiment compatibility, including legacy caches."""

import hashlib
import json
from pathlib import Path

from datasets import Dataset

from .data_handlers.datasets import DatasetSpec
from .rubric import CRITERIA, INSTRUCTIONS, truncation_config


def save_json(path: Path, value) -> None:
    """Replace JSON atomically, rejecting nonstandard NaN values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_judgments(rows: list[dict]) -> None:
    pairs = set()
    for row in rows:
        pair = (row["query_id"], row["passage_id"])
        if pair in pairs:
            raise ValueError(f"Duplicate judgment: {pair}")
        pairs.add(pair)
        for field in ("human_relevance", "predicted_relevance"):
            if type(row[field]) is not int or row[field] not in range(4):
                raise ValueError(f"Invalid {field}: {pair}")


def judgment_dataset(rows: list[dict]) -> Dataset:
    """Preserve optional fields in older checkpoints."""
    columns = dict.fromkeys(key for row in rows for key in row)
    return Dataset.from_dict({key: [row.get(key) for row in rows] for key in columns})


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_checkpoint(
    output: Path,
    *,
    dataset: DatasetSpec,
    method: str,
    model: str,
    configuration: dict,
    inputs: list,
    pairs: list,
    concurrency: int,
) -> dict:
    legacy_payload = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "criteria": CRITERIA,
        "inputs": inputs,
        **configuration,
    }
    signature = digest(
        {
            **legacy_payload,
            "dataset": dataset.repository,
            "dataset_name": dataset.name,
            "method": method,
            "signature_version": 2,
            "truncation": truncation_config(),
        }
    )
    cached = read_jsonl(output / "judgments.jsonl")
    validate_judgments(cached)
    completed = {(r["query_id"], r["passage_id"]): r for r in cached}
    if not completed.keys() <= set(pairs):
        raise ValueError("Checkpoint contains unexpected pairs")
    metadata_path = output / "experiment.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        compatible = previous.get("signature") == signature
        if (
            dataset.name == "trec-dl-2023"
            and method in {"jev", "openai"}
            and "signature_version" not in previous
            and previous.get("dataset") == dataset.repository
            and truncation_config() == {"character_budget": 120_000, "margin": 2_000}
        ):
            compatible |= previous.get("signature") == digest(legacy_payload)
        if not compatible and cached:
            raise ValueError("Cached experiment differs; use a new --data-dir")
    elif (output / "judgments.jsonl").exists():
        raise ValueError("Judgment checkpoint has no experiment metadata")
    save_json(
        metadata_path,
        {
            "signature": signature,
            "signature_version": 2,
            "dataset": dataset.repository,
            "dataset_name": dataset.name,
            "method": method,
            "model": model,
            "instructions": INSTRUCTIONS,
            "criteria": CRITERIA,
            "concurrency": concurrency,
            **configuration,
            "truncation": truncation_config(),
        },
    )
    return completed


def reference_pairs(qrels) -> dict:
    """Index reference labels in their original order, rejecting duplicate pairs."""
    expected = {}
    for row in qrels:
        pair = (row["query_id"], row["passage_id"])
        if pair in expected:
            raise ValueError("Duplicate reference query-passage pairs")
        expected[pair] = int(row["relevance"])
    return expected


def successful_judgments(completed: dict, qrels, *, allow_partial: bool) -> Dataset:
    """Select successful pairs in reference order and refresh their human labels."""
    expected = reference_pairs(qrels)
    if not completed.keys() <= expected.keys():
        raise ValueError("Checkpoint contains unexpected pairs")
    if not completed:
        raise ValueError(
            "No judgments to report; checkpoint has no successful judgments"
        )
    if not allow_partial and completed.keys() != expected.keys():
        raise ValueError(
            "Report requires a complete checkpoint matching the reference qrels; "
            "use --allow-partial to report successful judgments"
        )
    return judgment_dataset(
        [
            {**completed[pair], "human_relevance": relevance}
            for pair, relevance in expected.items()
            if pair in completed
        ]
    )
