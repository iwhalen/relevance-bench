"""Generate, save, plot, and display experiment reports."""

import hashlib
import json
import math
import re
from pathlib import Path

import ir_measures
import numpy as np
import typer
from datasets import Dataset
from scipy.stats import kendalltau
from tqdm import tqdm

from ..checkpoints import save_json
from ..data_handlers.datasets import DatasetSpec
from ..data_handlers.trec_runs import load_run
from ..providers import token_prices
from .bootstrap import (
    DEFAULT_BOOTSTRAP,
    BootstrapConfig,
    bootstrap_report,
    interval_annotation,
)
from .checkpoints import coverage_report
from .metrics import agreement_report, usage_report


def format_optional(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def performance_report(
    judgments: Dataset,
    method: str,
    *,
    run_paths: list[Path],
    output_dir: Path,
    dataset: DatasetSpec,
    model: str,
    token_costs: dict | None,
    coverage: dict | None = None,
    bootstrap: BootstrapConfig = DEFAULT_BOOTSTRAP,
) -> dict:
    """Compare paired grades and system rankings on identical qrels support."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = judgments.to_list()
    agreement = agreement_report(rows)
    if coverage is None:
        coverage = coverage_report(
            [{**row, "relevance": row["human_relevance"]} for row in rows], rows
        )
    plot_annotation = (
        "\nPartial report — judgment coverage: "
        f"{coverage['completed_judgments']:,}/{coverage['expected_judgments']:,} "
        f"({coverage['judgment_fraction']:.3%})"
        if coverage["status"] == "partial"
        else ""
    )
    if len(run_paths) < 2:
        raise ValueError(
            "At least two passage runs are required for system rank correlation"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    measures = [ir_measures.nDCG @ 10, ir_measures.nDCG @ 100, ir_measures.AP(rel=2)]
    references = {
        source: [
            ir_measures.Qrel(r["query_id"], r["passage_id"], r[field]) for r in rows
        ]
        for source, field in (
            ("human", "human_relevance"),
            (method, "predicted_relevance"),
        )
    }
    scores = {}
    manifest = []
    for path in tqdm(run_paths, desc="Evaluate runs"):
        path = Path(path)
        run = load_run(path, dataset)
        if path.name in scores:
            raise ValueError(f"Duplicate run filename: {path.name}")
        scores[path.name] = {}
        for source, qrels in references.items():
            per_query = {
                qid: {str(m): 0.0 for m in measures}
                for qid in {r["query_id"] for r in rows}
            }
            filtered = {qid: docs for qid, docs in run.items() if qid in per_query}
            for result in ir_measures.iter_calc(measures, qrels, filtered):
                per_query[result.query_id][str(result.measure)] = float(result.value)
            scores[path.name][source] = {
                "aggregate": {
                    str(m): sum(v[str(m)] for v in per_query.values()) / len(per_query)
                    for m in measures
                },
                "per_query": per_query,
            }
        manifest.append(
            {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    intervals = bootstrap_report(rows, scores, method, bootstrap)
    if intervals is not None:
        agreement["bootstrap"] = intervals[1]
    with plt.style.context("dark_background"):
        counts = np.array(agreement["counts"])
        for normalized in (False, True):
            values = counts.astype(float)
            if normalized:
                values = np.divide(
                    values,
                    counts.sum(axis=1, keepdims=True),
                    out=np.zeros_like(values),
                    where=counts.sum(axis=1, keepdims=True) != 0,
                )
            fig, ax = plt.subplots()
            image = ax.imshow(values, cmap="Blues")
            for i in range(4):
                for j in range(4):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.1%}" if normalized else str(counts[i, j]),
                        ha="center",
                        va="center",
                        color="orange",
                    )
            ax.set(
                xticks=range(4),
                yticks=range(4),
                xlabel=f"{method} grade",
                ylabel="Human grade",
                title=f"Assessment agreement (n={len(rows):,}, κ={format_optional(agreement['cohen_kappa'])})"
                + interval_annotation(agreement.get("bootstrap"))
                + plot_annotation,
            )
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(
                output
                / (
                    "agreement_normalized.png" if normalized else "agreement_counts.png"
                ),
                dpi=180,
            )
            plt.close(fig)

    correlations = {}
    with plt.style.context("dark_background"):
        for measure in measures:
            key = str(measure)
            human = [v["human"]["aggregate"][key] for v in scores.values()]
            predicted = [v[method]["aggregate"][key] for v in scores.values()]
            tau, pvalue = kendalltau(human, predicted, variant="b")
            correlations[key] = {
                "tau_b": float(tau) if math.isfinite(tau) else None,
                "p_value": float(pvalue) if math.isfinite(pvalue) else None,
                "system_count": len(scores),
                "undefined_reason": "Constant system scores"
                if not math.isfinite(tau)
                else None,
            }
            if intervals is not None:
                correlations[key]["bootstrap"] = intervals[2][key]
            fig, ax = plt.subplots()
            ax.scatter(human, predicted)
            low, high = min(human + predicted), max(human + predicted)
            ax.plot([low, high], [low, high], "--", alpha=0.5)
            ax.set(
                xlabel=f"Human {key}",
                ylabel=f"{method} {key}",
                title=f"System score agreement: τ-b={format_optional(correlations[key]['tau_b'])} (n={len(scores)})"
                + interval_annotation(correlations[key].get("bootstrap"))
                + plot_annotation,
            )
            fig.tight_layout()
            fig.savefig(
                output / f"system_scores_{re.sub(r'[^a-zA-Z0-9]', '_', key)}.png",
                dpi=180,
            )
            plt.close(fig)
    report = {
        "method": method,
        "coverage": coverage,
        **usage_report(rows, model=model, token_costs=token_costs),
        "dataset": dataset.repository,
        "judgment_count": len(rows),
        "query_count": len({r["query_id"] for r in rows}),
        "agreement": agreement,
        "system_rank_correlations": correlations,
        "runs": manifest,
        "evaluation": {
            "ndcg_gain": "linear (trec_eval)",
            "binary_relevance_threshold": 2,
            "unjudged": "nonrelevant; rankings are not condensed",
            "reference_qrels": dataset.reference_qrels,
        },
    }
    if intervals is not None:
        report["bootstrap"] = intervals[0]
    save_json(output / "run_scores.json", scores)
    save_json(output / "results.json", report)
    return report


def print_report(
    judgments,
    *,
    qrels,
    method: str,
    dataset,
    paths,
    data_dir: Path,
    bootstrap: BootstrapConfig = DEFAULT_BOOTSTRAP,
):
    checkpoint = data_dir / method
    failure_path = checkpoint / "failures.json"
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else []
    coverage = coverage_report(qrels, judgments.to_list(), failures)
    output = checkpoint / "partial" if coverage["status"] == "partial" else checkpoint
    metadata = json.loads((checkpoint / "experiment.json").read_text())
    model = metadata["model"]
    result = performance_report(
        judgments,
        method,
        run_paths=paths,
        output_dir=output,
        dataset=dataset,
        model=model,
        token_costs=token_prices(method, model),
        coverage=coverage,
        bootstrap=bootstrap,
    )
    typer.echo(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "bootstrap",
                    "coverage",
                    "usage",
                    "cost",
                    "agreement",
                    "system_rank_correlations",
                )
                if key in result
            },
            indent=2,
            allow_nan=False,
        )
    )
    typer.echo(f"Reports saved in {output}")
