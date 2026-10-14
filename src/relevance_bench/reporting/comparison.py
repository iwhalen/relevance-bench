"""Compare local reports without loading datasets, retrieval runs, or providers."""

import json
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    folder: str
    method: str
    model: str
    completed: int
    expected: int
    partial: bool
    tau: float | None
    kappa: float | None
    tau_ci: str
    kappa_ci: str
    cost: float | None
    cost_complete: bool
    latency: float | None
    notes: list[str] = field(default_factory=list)
    tau_confidence: float | None = None
    kappa_confidence: float | None = None

    @property
    def coverage(self) -> str:
        counts = f"{self.completed}/{self.expected}"
        return f"partial ({counts})" if self.partial else counts


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error


def _number(value, name: str, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (nonnegative and value < 0)
    ):
        raise ValueError(
            f"{name} must be a finite number" + (" >= 0" if nonnegative else "")
        )
    return value


def _count(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _interval(metric: dict, name: str, notes: list[str]) -> tuple[str, float | None]:
    interval = metric.get("bootstrap")
    if interval is None:
        notes.append(f"{name} CI: no saved bootstrap interval.")
        return "N/A", None
    confidence = _number(interval["confidence"], f"{name} CI confidence")
    if confidence is None or not 0 < confidence < 1:
        raise ValueError(f"{name} CI confidence must be between 0 and 1")
    lower = _number(interval["lower"], f"{name} CI lower")
    upper = _number(interval["upper"], f"{name} CI upper")
    level = f"{100 * confidence:.6g}%"
    if (lower is None) != (upper is None):
        raise ValueError(f"{name} CI must have both bounds or neither")
    if lower is None:
        reason = interval.get("unavailable_reason") or "saved interval is unavailable"
        notes.append(f"{name} CI ({level}): {reason}.")
        return "N/A", confidence
    if lower > upper:
        raise ValueError(f"{name} CI lower bound exceeds upper bound")
    return f"[{lower:.4f}, {upper:.4f}]", confidence


def _latency(path: Path, count: int, notes: list[str]) -> float | None:
    if not path.exists():
        notes.append("P50 latency: judgments.jsonl is missing.")
        return None
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("expected judgment objects")
        pairs = [(row["query_id"], row["passage_id"]) for row in rows]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate judgments")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if len(rows) != count:
        raise ValueError(
            f"{path}: checkpoint/report judgment-count mismatch "
            f"({len(rows)} vs {count}); regenerate the report before comparing"
        )
    durations = [row.get("latency_seconds") for row in rows]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in durations
    ):
        notes.append(
            "P50 latency: missing or invalid latency_seconds in successful judgments."
        )
        return None
    return statistics.median(durations)


def _read_experiment(folder: Path, path: Path) -> tuple[Experiment, tuple]:
    try:
        report = _json(path)
        method = _text(report["method"], "method")
        dataset = _text(report["dataset"], "dataset")
        count = _count(report["judgment_count"], "judgment_count")
        coverage = report["coverage"]
        completed = _count(coverage["completed_judgments"], "completed_judgments")
        expected = _count(coverage["expected_judgments"], "expected_judgments")
        partial = completed < expected
        if (
            completed != count
            or completed > expected
            or coverage["status"] != ("partial" if partial else "complete")
            or partial != (path.parent != folder)
        ):
            raise ValueError(
                "inconsistent report coverage or complete/partial report location"
            )
        evaluation = report["evaluation"]
        if not isinstance(evaluation, dict):
            raise TypeError("evaluation must be an object")
        for key in (
            "ndcg_gain",
            "binary_relevance_threshold",
            "unjudged",
            "reference_qrels",
        ):
            if key not in evaluation or evaluation[key] is None:
                raise ValueError(f"evaluation is missing {key}")
        runs = report["runs"]
        if not isinstance(runs, list) or len(runs) < 2:
            raise ValueError(
                "runs must contain at least two retrieval-run manifest entries"
            )
        manifest = sorted(
            (_text(run["file"], "run file"), _text(run["sha256"], "run sha256"))
            for run in runs
        )
        if len({name for name, _ in manifest}) != len(manifest):
            raise ValueError("duplicate retrieval-run filenames")
        agreement = report["agreement"]
        correlation = report["system_rank_correlations"]["nDCG@10"]
        kappa = _number(agreement["cohen_kappa"], "cohen_kappa")
        tau = _number(correlation["tau_b"], "nDCG@10 tau_b")
        notes = []
        for name, value, metric in (
            ("nDCG@10 tau-b", tau, correlation),
            ("Kappa", kappa, agreement),
        ):
            if value is None:
                notes.append(
                    f"{name}: {metric.get('undefined_reason') or 'saved estimate is unavailable'}."
                )
        tau_ci, tau_confidence = _interval(correlation, "nDCG@10 tau-b", notes)
        kappa_ci, kappa_confidence = _interval(agreement, "Kappa", notes)
        cost_report = report.get("cost")
        if cost_report is None:
            cost_report = {}
        if not isinstance(cost_report, dict):
            raise TypeError("cost must be an object")
        cost = _number(cost_report.get("total"), "cost total", nonnegative=True)
        cost_complete = cost_report.get("complete", False)
        if type(cost_complete) is not bool:
            raise ValueError("cost complete must be boolean")
        if cost is None:
            notes.append(
                f"Estimated cost: {cost_report.get('undefined_reason') or 'no saved total cost'}."
            )
        if not cost_complete:
            notes.append(
                "Cost estimate is incomplete: pricing or successful-judgment usage is missing; excluded from cost plots."
            )
        model = cost_report.get("model")
        if model is None and (folder / "experiment.json").exists():
            model = _json(folder / "experiment.json").get("model")
        if model is None:
            model = "N/A"
            notes.append("Model: no model saved in the report or experiment metadata.")
        model = _text(model, "model")
        latency = _latency(folder / "judgments.jsonl", count, notes)
        experiment = Experiment(
            folder.name,
            method,
            model,
            completed,
            expected,
            partial,
            tau,
            kappa,
            tau_ci,
            kappa_ci,
            cost,
            cost_complete,
            latency,
            notes,
            tau_confidence,
            kappa_confidence,
        )
        return experiment, (dataset, evaluation, manifest)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Invalid report {path}: {error}") from error


def discover_experiments(
    data_dir: Path,
    *,
    output_dir: Path | None = None,
    message: Callable[[str], None] = lambda text: print(text, file=sys.stderr),
) -> list[Experiment]:
    """Load immediate experiment children and verify identical evaluation inputs."""
    if not data_dir.is_dir():
        raise ValueError(
            f"Data directory does not exist or is not a directory: {data_dir}"
        )
    experiments = []
    baseline = None
    baseline_folder = None
    for folder in sorted(data_dir.iterdir(), key=lambda p: p.name):
        if not folder.is_dir() or (
            output_dir is not None and folder.resolve() == output_dir.resolve()
        ):
            continue
        path = folder / "results.json"
        if not path.exists():
            path = folder / "partial" / "results.json"
        if not path.exists():
            message(
                f"Skipping {folder.name}: no results.json or partial/results.json; no report will be generated."
            )
            continue
        experiment, compatibility = _read_experiment(folder, path)
        if baseline is None:
            baseline, baseline_folder = compatibility, folder.name
        else:
            for label, current, reference in zip(
                ("datasets", "evaluation settings", "retrieval-run manifests"),
                compatibility,
                baseline,
            ):
                if current != reference:
                    raise ValueError(
                        f"Incompatible {label}: {folder.name} and {baseline_folder}"
                    )
        experiments.append(experiment)
    if not experiments:
        raise ValueError(f"No usable reports in {data_dir}")
    return experiments


def pareto_indices(points: list[tuple[float, float]]) -> list[int]:
    """Return all nondominated indices, minimizing x and maximizing y; retain ties."""
    return [
        i
        for i, (x, y) in enumerate(points)
        if not any(
            other_x <= x and other_y >= y and (other_x < x or other_y > y)
            for other_x, other_y in points
        )
    ]


def _plot(
    experiments: list[Experiment], output: Path, y_metric: str, x_metric: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = []
    exclusions = []
    for experiment in experiments:
        reasons = []
        x, y = getattr(experiment, x_metric), getattr(experiment, y_metric)
        if x is None:
            reasons.append(f"{x_metric} unavailable")
        if y is None:
            reasons.append(f"{y_metric} unavailable")
        if x_metric == "cost" and not experiment.cost_complete:
            reasons.append("incomplete cost estimate")
        if reasons:
            exclusions.append(f"{experiment.folder}: {', '.join(reasons)}")
        else:
            eligible.append((experiment, x, y))
    front = pareto_indices([(x, y) for _, x, y in eligible])
    ylabel = "nDCG@10 Kendall tau-b" if y_metric == "tau" else "Pooled Cohen’s kappa"
    xlabel = "Estimated total cost (USD)" if x_metric == "cost" else "p50 latency (s)"
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, (experiment, x, y) in enumerate(eligible):
            ax.scatter(
                [x],
                [y],
                marker="^" if experiment.partial else "o",
                s=70,
                color="orange" if i in front else "steelblue",
                zorder=3,
            )
            label = f"{experiment.folder} / {experiment.method} / {experiment.model}\n{experiment.coverage}"
            # Offset tied labels so each experiment remains visible.
            tied_before = sum(px == x and py == y for _, px, py in eligible[:i])
            ax.annotate(
                label,
                (x, y),
                xytext=(6, 8 + 28 * tied_before),
                textcoords="offset points",
                fontsize=8,
            )
        coordinates = sorted({(eligible[i][1], eligible[i][2]) for i in front})
        if coordinates:
            ax.plot(*zip(*coordinates), color="orange", alpha=0.7, zorder=2)
        else:
            ax.text(
                0.5,
                0.5,
                "No eligible experiments\nRequired values are unavailable or cost estimates are incomplete.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set(
            xlabel=xlabel + " (minimize)",
            ylabel=ylabel + " (maximize)",
            title=f"{ylabel} vs {xlabel}",
        )
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.grid(alpha=0.2)
        fig.text(
            0.5,
            0.02,
            "Orange: nondominated • Circle: complete • Triangle: partial\nDiffering judgment coverage affects comparability.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(output, dpi=180)
        plt.close(fig)
    return exclusions


def _cell(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _format(value: float | None, places: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{places}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(map(_cell, row)) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def compare_reports(data_dir: Path, output_dir: Path | None = None) -> str:
    """Save all comparison artifacts and return the exact Markdown to print."""
    output_dir = output_dir if output_dir is not None else data_dir / "comparison"
    experiments = discover_experiments(data_dir, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agreement_rows, efficiency_rows = [], []
    ci_headers = []
    confidence_notes = []
    for metric, label in (("tau", "Tau"), ("kappa", "Kappa")):
        levels = {
            getattr(experiment, f"{metric}_confidence") for experiment in experiments
        } - {None}
        heading = f"{label} CI"
        if len(levels) == 1:
            heading += f" ({100 * next(iter(levels)):.6g}%)"
        elif len(levels) > 1:
            heading += " (varies)"
            for experiment in experiments:
                confidence = getattr(experiment, f"{metric}_confidence")
                if confidence is not None:
                    confidence_notes.append(
                        f"- {_cell(experiment.folder)}: {label} CI confidence is {100 * confidence:.6g}%."
                    )
        ci_headers.append(heading)
    for experiment in experiments:
        identity = [
            experiment.method,
            experiment.model,
            experiment.coverage,
        ]
        agreement_rows.append(
            identity
            + [
                _format(experiment.tau),
                experiment.tau_ci,
                _format(experiment.kappa),
                experiment.kappa_ci,
            ]
        )
        cost = _format(experiment.cost, 6) + (
            " (incomplete)" if not experiment.cost_complete else ""
        )
        efficiency_rows.append(identity + [cost, _format(experiment.latency)])
    sections = [
        "# Saved experiment comparison",
        "## Agreement",
        _table(
            [
                "Provider",
                "Model",
                "Judgment coverage",
                "nDCG@10 tau-b",
                ci_headers[0],
                "Pooled Cohen’s kappa",
                ci_headers[1],
            ],
            agreement_rows,
        ),
        "## Efficiency",
        _table(
            [
                "Provider",
                "Model",
                "Judgment coverage",
                "Estimated total cost (USD)",
                "P50 latency (seconds)",
            ],
            efficiency_rows,
        ),
        "## Interpretation",
        (
            "Point estimates, confidence bounds and levels, and costs are copied from saved reports. "
            "Differing judgment coverage affects comparability: partial reports evaluate only successful judgments, "
            "and their costs cover that subset. Even equal counts can cover different pairs. "
            "Recorded costs exclude failed attempts; incomplete cost estimates are marked and excluded from cost plots. "
            "P50 latency is the median of successful judgments’ latency_seconds, including retries within "
            "each recorded duration. Provider time_taken is never substituted. N/A means unavailable."
        ),
    ]
    for experiment in experiments:
        for note in experiment.notes:
            sections.append(f"- {_cell(experiment.folder)}: {_cell(note)}")
    sections.extend(confidence_notes)
    sections.extend(
        [
            "## Pareto plots",
            (
                "Linear axes minimize cost or latency and maximize agreement, using unrounded point estimates only. "
                "Orange points are nondominated; lines connect unique frontier coordinates in ascending x order. "
                "Tied experiments are retained. Circles indicate complete reports; triangles indicate partial reports. "
                "Coverage is annotated for every experiment."
            ),
        ]
    )
    for y_metric in ("tau", "kappa"):
        for x_metric in ("cost", "latency"):
            filename = f"pareto_{y_metric}_{x_metric}.png"
            exclusions = _plot(experiments, output_dir / filename, y_metric, x_metric)
            sections.append(f"![{y_metric} vs {x_metric}]({filename})")
            sections.append(
                "Exclusions: "
                + (
                    "; ".join(_cell(item) for item in exclusions) + "."
                    if exclusions
                    else "none."
                )
            )
    markdown = "\n\n".join(sections) + "\n"
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    return markdown
