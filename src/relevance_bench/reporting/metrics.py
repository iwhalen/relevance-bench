"""Calculation of pooled judgment agreement and recorded usage costs."""

import numpy as np

from ..checkpoints import validate_judgments


def agreement_report(rows: list[dict]) -> dict:
    validate_judgments(rows)
    counts = np.zeros((4, 4), dtype=int)
    for row in rows:
        counts[row["human_relevance"], row["predicted_relevance"]] += 1
    total = int(counts.sum())
    if total == 0:
        raise ValueError("No judgments to report")
    observed = float(np.trace(counts) / total)
    expected = float(counts.sum(axis=0) @ counts.sum(axis=1) / total**2)
    return {
        "labels": [0, 1, 2, 3],
        "rows": "human",
        "columns": "method",
        "counts": counts.tolist(),
        "count": total,
        "agree_count": int(np.trace(counts)),
        "exact_agreement": observed,
        "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "undefined_reason": "No variation in either assessor"
        if expected == 1
        else None,
        "human_support": counts.sum(axis=1).tolist(),
        "method_support": counts.sum(axis=0).tolist(),
    }


def usage_report(rows: list[dict], *, model: str, token_costs: dict | None) -> dict:
    """Report recorded successful-call usage; missing counts remain unknown."""
    recorded = [
        row["usage"]
        for row in rows
        if row.get("usage") is not None
        and row["usage"].get("input_tokens") is not None
        and row["usage"].get("output_tokens") is not None
    ]
    input_tokens = sum(usage["input_tokens"] for usage in recorded)
    output_tokens = sum(usage["output_tokens"] for usage in recorded)
    input_cost = (
        input_tokens * token_costs["input"] / 1_000_000 if token_costs else None
    )
    output_cost = (
        output_tokens * token_costs["output"] / 1_000_000 if token_costs else None
    )
    return {
        "cost": {
            "input": input_cost,
            "output": output_cost,
            "total": input_cost + output_cost if token_costs else None,
            "estimated": True,
            "model": model,
            "undefined_reason": None
            if token_costs
            else f"No token prices configured for {model}",
            "token_costs_per_million": token_costs,
            "complete": token_costs is not None and len(recorded) == len(rows),
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "judgments_with_usage": len(recorded),
            "judgments_without_usage": len(rows) - len(recorded),
        },
    }
