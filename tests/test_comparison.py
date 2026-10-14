import json
import socket
from copy import deepcopy
from dataclasses import replace

import matplotlib.figure
import pytest
from typer.testing import CliRunner

from relevance_bench import __main__ as cli
from relevance_bench.data_handlers.datasets import DatasetSpec
from relevance_bench.reporting import reports
from relevance_bench.reporting.comparison import (
    _plot,
    compare_reports,
    discover_experiments,
    pareto_indices,
)


@pytest.fixture
def saved_report(tmp_path):
    def write(folder="experiment", *, partial=False, report=None, durations=(1.0, 3.0)):
        count = len(durations)
        result = (
            report
            if report is not None
            else {
                "method": "custom-method",
                "dataset": "local/reference",
                "judgment_count": count,
                "coverage": {
                    "status": "partial" if partial else "complete",
                    "completed_judgments": count,
                    "expected_judgments": count + 1 if partial else count,
                },
                "agreement": {
                    "cohen_kappa": -0.23456789,
                    "bootstrap": {
                        "confidence": 0.9,
                        "lower": -0.456789,
                        "upper": 0.123456,
                    },
                },
                "system_rank_correlations": {
                    "nDCG@10": {
                        "tau_b": 0.12345678,
                        "bootstrap": {
                            "confidence": 0.8,
                            "lower": -0.123456,
                            "upper": 0.456789,
                        },
                    }
                },
                "evaluation": {
                    "ndcg_gain": "linear (trec_eval)",
                    "binary_relevance_threshold": 2,
                    "unjudged": "nonrelevant; rankings are not condensed",
                    "reference_qrels": "test reference",
                },
                "runs": [
                    {"file": "a.run", "sha256": "a" * 64},
                    {"file": "b.run", "sha256": "b" * 64},
                ],
                "cost": {
                    "total": 0.123456789,
                    "complete": True,
                    "model": "saved-model",
                },
            }
        )
        directory = tmp_path / folder
        output = directory / "partial" if partial else directory
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(result))
        (directory / "judgments.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "query_id": "q",
                        "passage_id": f"p{i}",
                        "latency_seconds": duration,
                        "time_taken": 999,
                    }
                )
                + "\n"
                for i, duration in enumerate(durations)
            )
        )
        return result

    return write


def test_discovery_precedence_partial_duplicates_order_and_skips(
    saved_report, tmp_path, capsys
):
    saved_report("z-duplicate")
    saved_report("a-complete")
    # A malformed stale partial report must not override a complete report.
    stale = tmp_path / "a-complete" / "partial"
    stale.mkdir()
    (stale / "results.json").write_text("not json")
    saved_report("m-partial", partial=True)
    checkpoint = tmp_path / "unfinished"
    checkpoint.mkdir()
    (checkpoint / "judgments.jsonl").write_text("")
    # Discovery does not descend into grandchildren.
    saved_report("nested/hidden")
    rows = discover_experiments(tmp_path)
    assert [row.folder for row in rows] == ["a-complete", "m-partial", "z-duplicate"]
    assert [row.method for row in rows] == ["custom-method"] * 3
    assert rows[1].coverage == "partial (2/3)"
    assert rows[0].coverage == "2/2"
    assert "Skipping unfinished" in capsys.readouterr().err


@pytest.mark.parametrize("missing", [False, True])
def test_no_usable_reports(tmp_path, missing):
    path = tmp_path / "missing" if missing else tmp_path
    with pytest.raises(ValueError, match="Data directory|No usable reports"):
        discover_experiments(path)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("dataset", "other/reference", "datasets"),
        (
            "evaluation",
            {
                "ndcg_gain": "exp",
                "binary_relevance_threshold": 2,
                "unjudged": "zero",
                "reference_qrels": "test",
            },
            "evaluation settings",
        ),
        (
            "runs",
            [
                {"file": "a.run", "sha256": "changed"},
                {"file": "b.run", "sha256": "b" * 64},
            ],
            "retrieval-run manifests",
        ),
    ],
)
def test_incompatible_inputs(saved_report, tmp_path, field, value, match):
    report = saved_report("a")
    changed = deepcopy(report)
    changed[field] = value
    saved_report("b", report=changed)
    with pytest.raises(ValueError, match=f"Incompatible {match}"):
        discover_experiments(tmp_path)
    assert not (tmp_path / "comparison").exists()


def test_manifest_order_is_irrelevant(saved_report, tmp_path):
    report = saved_report("a")
    report["runs"].reverse()
    saved_report("b", report=report)
    assert len(discover_experiments(tmp_path)) == 2


@pytest.mark.parametrize(
    "invalid",
    [
        "json",
        "object",
        "missing_metric",
        "nan",
        "string",
        "coverage",
        "evaluation",
        "duplicate_runs",
        "negative_cost",
        "interval",
        "confidence",
        "cost_complete",
    ],
)
def test_malformed_reports_fail(saved_report, tmp_path, invalid):
    report = saved_report()
    if invalid == "missing_metric":
        del report["agreement"]["cohen_kappa"]
    elif invalid == "nan":
        report["agreement"]["cohen_kappa"] = float("nan")
    elif invalid == "string":
        report["system_rank_correlations"]["nDCG@10"]["tau_b"] = "0.5"
    elif invalid == "coverage":
        report["coverage"]["completed_judgments"] = 1
    elif invalid == "evaluation":
        report["evaluation"] = {}
    elif invalid == "duplicate_runs":
        report["runs"][1] = report["runs"][0]
    elif invalid == "negative_cost":
        report["cost"]["total"] = -1
    elif invalid == "interval":
        report["agreement"]["bootstrap"]["upper"] = None
    elif invalid == "confidence":
        report["agreement"]["bootstrap"]["confidence"] = 1
    elif invalid == "cost_complete":
        report["cost"]["complete"] = "yes"
    text = (
        "{"
        if invalid == "json"
        else "[]"
        if invalid == "object"
        else json.dumps(report)
    )
    (tmp_path / "experiment" / "results.json").write_text(text)
    with pytest.raises(ValueError, match="Invalid report.*results.json"):
        discover_experiments(tmp_path)


@pytest.mark.parametrize(
    "durations",
    [(None, 1), (-1, 1), (True, 1), ("1", 1), (float("nan"), 1), (float("inf"), 1)],
)
def test_invalid_durations_are_unavailable(saved_report, tmp_path, durations):
    saved_report(durations=durations)
    row = discover_experiments(tmp_path)[0]
    assert row.latency is None
    assert any("missing or invalid latency_seconds" in note for note in row.notes)


@pytest.mark.parametrize(
    "durations, expected",
    [((100.0, 1.0, 3.0), 3.0), ((100.0, 1.0, 5.0, 3.0), 4.0), ((0.0,), 0.0)],
)
def test_latency_p50(saved_report, tmp_path, durations, expected):
    saved_report(durations=durations)
    assert discover_experiments(tmp_path)[0].latency == expected


def test_latency_zero_missing_file_and_missing_field(saved_report, tmp_path):
    saved_report(durations=(0, 0))
    assert discover_experiments(tmp_path)[0].latency == 0
    checkpoint = tmp_path / "experiment" / "judgments.jsonl"
    checkpoint.write_text(
        ' {"query_id":"q", "passage_id":"p1", "time_taken":12}\n'
        '{"query_id":"q", "passage_id":"p2", "latency_seconds":2}\n'
    )
    assert discover_experiments(tmp_path)[0].latency is None
    checkpoint.unlink()
    row = discover_experiments(tmp_path)[0]
    assert row.latency is None
    assert any("judgments.jsonl is missing" in note for note in row.notes)


@pytest.mark.parametrize(
    "content,match",
    [
        ("", "judgment-count mismatch.*regenerate"),
        ('{"query_id":"q","passage_id":"p"}\n', "judgment-count mismatch.*regenerate"),
        ("{", "Cannot read"),
        ("[]\n", "expected judgment objects"),
        ('{"query_id":"q","passage_id":"p"}\n' * 2, "duplicate judgments"),
    ],
)
def test_stale_and_malformed_checkpoints(saved_report, tmp_path, content, match):
    saved_report()
    (tmp_path / "experiment" / "judgments.jsonl").write_text(content)
    with pytest.raises(ValueError, match=match):
        discover_experiments(tmp_path)


@pytest.mark.parametrize(
    "points,expected",
    [
        ([], []),
        ([(0, -1)], [0]),
        ([(1, -0.5), (1, -0.5)], [0, 1]),
        ([(1, -0.5), (2, -0.5), (1, -0.6)], [0]),
        ([(1, -0.5), (2, 0), (3, -0.1), (0.5, -0.7)], [0, 1, 3]),
        ([(1, 0.123441), (1, 0.123449)], [1]),
    ],
)
def test_pareto_dominance(points, expected):
    assert pareto_indices(points) == expected


def test_cli_offline_values_and_all_artifacts(saved_report, tmp_path, monkeypatch):
    saved_report("b-complete")
    saved_report("a-partial", partial=True)
    original = {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        pytest.fail("compare must use saved local files only")

    for name in (
        "resolve_runs",
        "download_runs",
        "open_provider",
        "provider_config",
        "report_checkpoint",
        "print_report",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(DatasetSpec, "load", forbidden)
    monkeypatch.setattr(reports, "performance_report", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    result = CliRunner().invoke(cli.app, ["compare", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    output = tmp_path / "comparison"
    assert result.stdout == (output / "comparison.md").read_text()
    assert "Comparison saved in" in result.stderr
    assert "Comparison saved in" not in result.stdout
    assert (
        "| custom-method | saved-model | partial (2/3) | **0.1235** | [-0.1235, 0.4568] | **-0.2346** | [-0.4568, 0.1235] |"
        in result.stdout
    )
    assert (
        "| custom-method | saved-model | 2/2 | **0.123457** | **2.0000** |"
        in result.stdout
    )
    assert result.stdout.count("| Provider | Model | Judgment coverage |") == 2
    assert "| P50 latency (seconds) |" in result.stdout
    assert "P50 latency is the median" in result.stdout
    assert "| Tau CI (80%) |" in result.stdout
    assert "| Kappa CI (90%) |" in result.stdout
    assert "Experiment / method" not in result.stdout
    assert "Recorded costs exclude failed attempts" in result.stdout
    assert "Differing judgment coverage affects comparability" in result.stdout
    expected = {"comparison.md"}
    for y in ("tau", "kappa"):
        for x in ("cost", "latency"):
            name = f"pareto_{y}_{x}.png"
            expected.add(name)
            assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            assert f"]({name})" in result.stdout
    assert {p.name for p in output.iterdir()} == expected
    assert all(path.read_bytes() == content for path, content in original.items())


def test_best_values_use_unrounded_estimates_and_retain_ties(saved_report, tmp_path):
    for folder, tau, kappa, cost, latency in (
        ("a", 0.123441, -0.1, 0, 2),
        ("b", 0.123449, -0.2, 1, 0),
        ("c", 0.123449, -0.2, 0, 0),
    ):
        report = saved_report(folder)
        report["method"] = folder
        report["system_rank_correlations"]["nDCG@10"]["tau_b"] = tau
        report["agreement"]["cohen_kappa"] = kappa
        report["cost"]["total"] = cost
        saved_report(folder, report=report, durations=(latency, latency))
    markdown = compare_reports(tmp_path)
    assert "| a | saved-model | 2/2 | 0.1234 | [-0.1235, 0.4568] | **-0.1000** |" in markdown
    for folder in ("b", "c"):
        assert f"| {folder} | saved-model | 2/2 | **0.1234** | [-0.1235, 0.4568] | -0.2000 |" in markdown
    assert "| a | saved-model | 2/2 | **0.000000** | 2.0000 |" in markdown
    assert "| b | saved-model | 2/2 | 1.000000 | **0.0000** |" in markdown
    assert "| c | saved-model | 2/2 | **0.000000** | **0.0000** |" in markdown


def test_unavailable_and_incomplete_values(saved_report, tmp_path):
    report = saved_report()
    report["agreement"]["cohen_kappa"] = None
    report["agreement"]["undefined_reason"] = "No variation"
    report["agreement"]["bootstrap"] = {
        "confidence": 0.9,
        "lower": None,
        "upper": None,
        "unavailable_reason": "Too few queries",
    }
    del report["system_rank_correlations"]["nDCG@10"]["bootstrap"]
    report["cost"]["complete"] = False
    saved_report(report=report, durations=(None, 3))
    unknown = deepcopy(report)
    unknown["cost"] = {
        "total": None,
        "complete": False,
        "undefined_reason": "No token prices",
    }
    saved_report("unknown", report=unknown)
    markdown = compare_reports(tmp_path)
    assert "**0.123457** (incomplete)" in markdown
    assert "N/A (incomplete)" in markdown
    assert "**N/A**" not in markdown
    assert "| Kappa CI (90%) |" in markdown
    assert "N/A (90%)" not in markdown
    assert "Too few queries" in markdown
    assert "No variation" in markdown
    assert "no saved bootstrap interval" in markdown
    assert "No token prices" in markdown
    assert "P50 latency: missing or invalid" in markdown
    assert "incomplete cost estimate" in markdown
    assert "Model: no model saved" in markdown


def test_model_fallback_and_escaping(saved_report, tmp_path):
    report = saved_report("name|with-markup")
    del report["cost"]["model"]
    report["method"] = "provider|with-markup"
    saved_report("name|with-markup", report=report)
    (tmp_path / "name|with-markup" / "experiment.json").write_text(
        '{"model": "<model>"}'
    )
    markdown = compare_reports(tmp_path)
    assert "provider&#124;with-markup" in markdown
    assert "&lt;model&gt;" in markdown


@pytest.mark.parametrize("eligible", [False, True])
@pytest.mark.parametrize("x_metric", ["cost", "latency"])
def test_plot_zero_or_one_eligible_point(
    saved_report, tmp_path, monkeypatch, eligible, x_metric
):
    saved_report(partial=True)
    experiment = discover_experiments(tmp_path)[0]
    if not eligible:
        experiment = replace(experiment, cost_complete=False, latency=None)
    captured = []
    original = matplotlib.figure.Figure.savefig

    def capture(fig, *args, **kwargs):
        ax = fig.axes[0]
        captured.append(
            (
                len(ax.collections),
                [t.get_text() for t in ax.texts],
                ax.get_xscale(),
                ax.get_yscale(),
                ax.get_xlabel(),
            )
        )
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    exclusions = _plot([experiment], tmp_path / "plot.png", "kappa", x_metric)
    count, texts, xscale, yscale, xlabel = captured[0]
    assert count == int(eligible)
    assert xscale == yscale == "linear"
    assert (
        ("partial (2/3)" in texts[0])
        if eligible
        else ("No eligible experiments" in texts[0])
    )
    assert bool(exclusions) != eligible
    assert xlabel == (
        "Estimated total cost (USD) (minimize)"
        if x_metric == "cost"
        else "P50 latency (s) (minimize)"
    )


def test_cli_custom_output_and_clear_failure(saved_report, tmp_path):
    saved_report()
    output = tmp_path / "custom-output"
    result = CliRunner().invoke(
        cli.app, ["compare", "--data-dir", str(tmp_path), "--output-dir", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert (output / "comparison.md").exists()
    (tmp_path / "experiment" / "results.json").write_text("{")
    result = CliRunner().invoke(
        cli.app, ["compare", "--data-dir", str(tmp_path), "--output-dir", str(output)]
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Invalid report" in result.stderr
