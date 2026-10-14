import asyncio
import json
import math
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from datasets import Dataset
from typer.testing import CliRunner

from relevance_bench import __main__ as cli
from relevance_bench.checkpoints import (
    judgment_dataset,
    prepare_checkpoint,
    read_jsonl,
    save_json,
)
from relevance_bench.data_handlers.datasets import DEFAULT_DATASET, DatasetSpec
from relevance_bench.errors import ProviderResponseError
from relevance_bench.providers import PROVIDERS, provider_config, token_prices
from relevance_bench.providers.base import Judgment
from relevance_bench.reporting.checkpoints import coverage_report, report_checkpoint
from relevance_bench.reporting.reports import performance_report

METHODS = list(PROVIDERS)
ARTIFACTS = {
    "agreement_counts.png",
    "agreement_normalized.png",
    "system_scores_nDCG_10.png",
    "system_scores_nDCG_100.png",
    "system_scores_AP_rel_2_.png",
    "results.json",
    "run_scores.json",
}


def write_checkpoint(output, tables, method, rows):
    config = provider_config(method)
    queries = {r["query_id"]: r["text"] for r in tables["queries"]}
    passages = {r["passage_id"]: r["text"] for r in tables["passages"]}
    pairs = [(r["query_id"], r["passage_id"]) for r in tables["qrels"]]
    prepare_checkpoint(
        output,
        dataset=DEFAULT_DATASET,
        method=method,
        model=config.model_name,
        configuration=PROVIDERS[method].provider_type.configuration(config),
        inputs=[(qid, pid, queries[qid], passages[pid]) for qid, pid in pairs],
        pairs=pairs,
        concurrency=1,
    )
    (output / "judgments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def judgment(row, predicted=2):
    return {
        "query_id": row["query_id"],
        "passage_id": row["passage_id"],
        "human_relevance": row["relevance"],
        "predicted_relevance": predicted,
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def refusal(row, *, name="relevance"):
    return {
        "query_id": row["query_id"],
        "passage_id": row["passage_id"],
        "error": "refused",
        "type": "ProviderResponseError",
        "details": {"answers": [{"name": name, "type": "refusal"}]},
    }


def snapshot_inputs(output):
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in output.iterdir()
        if p.name in {"judgments.jsonl", "experiment.json", "failures.json"}
    }


def read_result(output):
    return json.loads((output / "results.json").read_text())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("allow_partial", [False, True])
def test_partial_cli_run_report_and_resume(
    method, allow_partial, monkeypatch, tables, run_paths, tmp_path
):
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    failure = ProviderResponseError(
        "refused", details=refusal(tables["qrels"][0])["details"]
    )
    calls = AsyncMock(side_effect=[failure, Judgment(1, 100, 0, 0)])

    @asynccontextmanager
    async def provider(*args):
        yield SimpleNamespace(adjudge=calls)

    monkeypatch.setattr(cli, "open_provider", provider)
    args = ["--method", method, "--data-dir", str(tmp_path), "--local-runs"]
    flags = ["--allow-partial"] if allow_partial else []
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", *args, *flags])
    assert result.exit_code == (0 if allow_partial else 1), result.output
    if not allow_partial:
        assert "--allow-partial" in str(result.exception)
    assert calls.await_count == 2
    output = tmp_path / method
    assert len(read_jsonl(output / "judgments.jsonl")) == 1
    assert len(json.loads((output / "failures.json").read_text())) == 1
    assert not (output / "results.json").exists()
    run_result = read_result(output / "partial") if allow_partial else None

    def no_provider(*args, **kwargs):
        pytest.fail("Reporting must not initialize or configure a provider")

    monkeypatch.setattr(cli, "open_provider", no_provider)
    monkeypatch.setattr(cli, "provider_config", no_provider)
    result = runner.invoke(cli.app, ["report", *args])
    assert result.exit_code == 1
    assert "--allow-partial" in str(result.exception)
    before = snapshot_inputs(output)
    result = runner.invoke(cli.app, ["report", *args, "--allow-partial"])
    assert result.exit_code == 0, result.output
    assert before == snapshot_inputs(output)
    partial = read_result(output / "partial")
    if allow_partial:
        assert partial == run_result
    assert partial["coverage"] == {
        "status": "partial",
        "expected_judgments": 2,
        "completed_judgments": 1,
        "missing_judgments": 1,
        "judgment_fraction": 0.5,
        "expected_queries": 1,
        "evaluated_queries": 1,
        "missing_pairs": [
            {
                "query_id": "q1",
                "passage_id": "msmarco_passage_1",
                "reference_relevance": 3,
                "failures": json.loads((output / "failures.json").read_text()),
            }
        ],
        "refusal_count": 1,
    }
    terminal, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    assert terminal["coverage"] == partial["coverage"]
    assert {p.name for p in (output / "partial").iterdir()} == ARTIFACTS
    assert partial["cost"]["complete"]  # Independent of judgment coverage.

    monkeypatch.setattr(cli, "open_provider", provider)
    monkeypatch.setattr(cli, "provider_config", provider_config)
    calls.reset_mock(side_effect=True)
    calls.side_effect = failure
    saved = (output / "judgments.jsonl").read_bytes()
    result = runner.invoke(cli.app, ["run", *args, "--allow-partial"])
    assert result.exit_code == 0, result.output
    assert calls.await_count == 1  # Existing successes survive all new failures.
    assert (output / "judgments.jsonl").read_bytes() == saved
    assert read_result(output / "partial") == partial

    calls.reset_mock(side_effect=True)
    calls.return_value = Judgment(3, 100, 0, 0)
    result = runner.invoke(cli.app, ["run", *args, "--allow-partial"])
    assert result.exit_code == 0, result.output
    assert calls.await_count == 1
    assert len(read_jsonl(output / "judgments.jsonl")) == 2
    assert json.loads((output / "failures.json").read_text()) == []
    assert read_result(output)["coverage"]["status"] == "complete"


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("allow_partial", [False, True])
def test_complete_cli_reports_unchanged(
    method, allow_partial, monkeypatch, tables, run_paths, tmp_path, fake_providers
):
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    args = ["--method", method, "--data-dir", str(tmp_path), "--local-runs"]
    flags = ["--allow-partial"] if allow_partial else []
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", *args, *flags])
    assert result.exit_code == 0, result.output
    output = tmp_path / method
    original = read_result(output)
    original_scores = (output / "run_scores.json").read_bytes()
    assert original["coverage"]["status"] == "complete"
    assert original["coverage"]["judgment_fraction"] == 1
    assert not (output / "partial").exists()
    save_json(
        output / "failures.json",
        [
            refusal(tables["qrels"][0]),
            refusal({"query_id": "unrelated", "passage_id": "unrelated"}),
        ],
    )
    before = snapshot_inputs(output)
    for report_flags in ([], ["--allow-partial"]):
        result = runner.invoke(cli.app, ["report", *args, *report_flags])
        assert result.exit_code == 0, result.output
        assert read_result(output) == original
        assert (output / "run_scores.json").read_bytes() == original_scores
        assert snapshot_inputs(output) == before
    assert len(fake_providers) == 1


@pytest.mark.parametrize("method", METHODS)
def test_all_failed_partial_run_fails(method, monkeypatch, tables, run_paths, tmp_path):
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    calls = AsyncMock(side_effect=ValueError("cannot judge"))

    @asynccontextmanager
    async def provider(*args):
        yield SimpleNamespace(adjudge=calls)

    monkeypatch.setattr(cli, "open_provider", provider)
    args = [
        "--method",
        method,
        "--data-dir",
        str(tmp_path),
        "--local-runs",
        "--allow-partial",
    ]
    result = CliRunner().invoke(cli.app, ["run", *args])
    assert result.exit_code == 1
    assert "no successful judgments" in str(result.exception)
    assert calls.await_count == 2
    output = tmp_path / method
    assert len(json.loads((output / "failures.json").read_text())) == 2
    assert read_jsonl(output / "judgments.jsonl") == []
    assert not (output / "partial").exists()
    result = CliRunner().invoke(cli.app, ["report", *args])
    assert result.exit_code == 1
    assert "no successful judgments" in str(result.exception)


@pytest.mark.parametrize("allow_partial", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate",
        "human_grade",
        "predicted_grade",
        "unexpected",
        "empty",
        "reference_duplicate",
        "dataset",
        "dataset_name",
        "method",
        "metadata_missing",
    ],
)
def test_report_validation_stays_strict(allow_partial, invalid, tmp_path, tables):
    rows = [judgment(row) for row in tables["qrels"]]
    write_checkpoint(tmp_path, tables, "jev", rows)
    qrels = tables["qrels"]
    messages = {
        "duplicate": "Duplicate judgment",
        "human_grade": "Invalid human_relevance",
        "predicted_grade": "Invalid predicted_relevance",
        "unexpected": "unexpected pairs",
        "empty": "no successful judgments",
        "reference_duplicate": "Duplicate reference",
        "dataset": "dataset or method differs",
        "dataset_name": "dataset or method differs",
        "method": "dataset or method differs",
        "metadata_missing": "no experiment metadata",
    }
    if invalid == "duplicate":
        rows.append(rows[0])
    elif invalid in {"human_grade", "predicted_grade"}:
        rows[0][
            "human_relevance" if invalid == "human_grade" else "predicted_relevance"
        ] = 4
    elif invalid == "unexpected":
        rows[0]["passage_id"] = "msmarco_passage_999"
    elif invalid == "empty":
        rows = []
    elif invalid == "reference_duplicate":
        qrels.append(qrels[0])
    elif invalid == "metadata_missing":
        (tmp_path / "experiment.json").unlink()
    else:
        metadata = json.loads((tmp_path / "experiment.json").read_text())
        metadata[invalid] = "different"
        save_json(tmp_path / "experiment.json", metadata)
    (tmp_path / "judgments.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    with pytest.raises(ValueError, match=messages[invalid]):
        report_checkpoint(
            tmp_path, qrels, DEFAULT_DATASET, "jev", allow_partial=allow_partial
        )


@pytest.mark.parametrize(
    "invalid", ["duplicate", "grade", "unexpected", "reference_duplicate", "signature"]
)
def test_partial_run_validation_stays_strict(invalid, monkeypatch, tmp_path, tables):
    rows = [judgment(tables["qrels"][0])]
    write_checkpoint(tmp_path, tables, "jev", rows)
    if invalid == "duplicate":
        rows.append(rows[0])
    elif invalid == "grade":
        rows[0]["predicted_relevance"] = -1
    elif invalid == "unexpected":
        rows[0]["passage_id"] = "msmarco_passage_999"
    elif invalid == "reference_duplicate":
        tables["qrels"].append(tables["qrels"][0])
    else:
        metadata = json.loads((tmp_path / "experiment.json").read_text())
        metadata["signature"] = "incompatible"
        save_json(tmp_path / "experiment.json", metadata)
    (tmp_path / "judgments.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    monkeypatch.setattr(
        cli,
        "open_provider",
        lambda *a: pytest.fail("Invalid checkpoints must fail before judging"),
    )
    with pytest.raises(ValueError):
        asyncio.run(
            cli.judge_async(
                **tables,
                output_dir=tmp_path,
                config=provider_config("jev"),
                concurrency=1,
                method="jev",
                allow_partial=True,
            )
        )


def test_reference_order_and_refreshed_labels(monkeypatch, tmp_path, tables):
    rows = [judgment(r) for r in reversed(tables["qrels"])]
    write_checkpoint(tmp_path, tables, "jev", rows)
    for row in tables["qrels"]:
        row["relevance"] = 1
    monkeypatch.setattr(
        cli,
        "open_provider",
        lambda *a: pytest.fail("Complete checkpoints skip provider initialization"),
    )
    generated = asyncio.run(
        cli.judge_async(
            **tables,
            output_dir=tmp_path,
            config=provider_config("jev"),
            concurrency=1,
            method="jev",
            allow_partial=True,
        )
    )
    reported = report_checkpoint(
        tmp_path, tables["qrels"], DEFAULT_DATASET, "jev", allow_partial=True
    )
    assert isinstance(generated, Dataset) and isinstance(reported, Dataset)
    assert generated.to_list() == reported.to_list()
    assert generated["passage_id"] == [r["passage_id"] for r in tables["qrels"]]
    assert generated["human_relevance"] == [1, 1]
    assert read_jsonl(tmp_path / "judgments.jsonl") == rows


def test_coverage_refusals_and_stale_diagnostics():
    qrels = [
        {"query_id": f"q{i}", "passage_id": f"p{i}", "relevance": i % 4}
        for i in range(7)
    ]
    rows = [judgment(qrels[0])]
    other_error = {
        "query_id": "q3",
        "passage_id": "p3",
        "type": "ValueError",
        "error": "refused",
    }
    failures = [
        refusal(qrels[0]),  # Completed pair.
        refusal({"query_id": "other", "passage_id": "other"}),
        refusal(qrels[1]),
        refusal(qrels[1]),  # Count pairs, not duplicate logs.
        refusal(qrels[2], name="another_question"),
        other_error,
        {**refusal(qrels[4]), "details": None},
        {
            **refusal(qrels[5]),
            "details": {"answers": [{"name": "relevance", "type": "score"}]},
        },
    ]
    result = coverage_report(qrels, rows, failures)
    assert result["refusal_count"] == 1
    assert result["missing_judgments"] == 6
    assert result["expected_queries"] == 7 and result["evaluated_queries"] == 1
    assert len(result["missing_pairs"][0]["failures"]) == 2
    assert result["missing_pairs"][2]["failures"] == [other_error]
    assert result["missing_pairs"][-1]["failures"] == []
    without_logs = coverage_report(qrels, rows)
    assert without_logs["refusal_count"] == 0
    assert without_logs["judgment_fraction"] == 1 / 7


def test_partial_metrics_match_manual_subset_and_plot_labels(
    monkeypatch, tmp_path, tables, run_paths
):
    import matplotlib.figure

    qrels = [
        {"query_id": "q1", "passage_id": "msmarco_passage_1", "relevance": 3},
        {"query_id": "q1", "passage_id": "msmarco_passage_2", "relevance": 2},
        {"query_id": "q1", "passage_id": "msmarco_passage_3", "relevance": 0},
        {"query_id": "q2", "passage_id": "msmarco_passage_4", "relevance": 3},
        {"query_id": "q3", "passage_id": "msmarco_passage_5", "relevance": 3},
    ]
    tables["qrels"] = qrels
    tables["queries"] = [{"query_id": f"q{i}", "text": "query"} for i in range(1, 4)]
    tables["passages"] = [
        {"passage_id": f"msmarco_passage_{i}", "text": "passage"} for i in range(1, 6)
    ]
    successful = [
        judgment(qrels[i], predicted) for i, predicted in [(1, 3), (2, 1), (3, 2)]
    ]
    output = tmp_path / "jev"
    # Disk order and human labels deliberately differ from reference qrels.
    write_checkpoint(
        output,
        tables,
        "jev",
        [{**row, "human_relevance": 0} for row in reversed(successful)],
    )
    run_paths[0].write_text(
        "q1 Q0 msmarco_passage_1 1 3 good\n"
        "q1 Q0 msmarco_passage_2 2 2 good\n"
        "q1 Q0 msmarco_passage_3 3 1 good\n"
        "q2 Q0 msmarco_passage_4 1 1 good\n"
        "q3 Q0 msmarco_passage_5 1 1 good\n"
    )
    run_paths[1].write_text(
        "q1 Q0 msmarco_passage_3 1 3 bad\n"
        "q1 Q0 msmarco_passage_1 2 2 bad\n"
        "q1 Q0 msmarco_passage_2 3 1 bad\n"
        "q2 Q0 msmarco_passage_4 1 1 bad\n"
        "q3 Q0 msmarco_passage_5 1 1 bad\n"
    )
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    monkeypatch.setattr(
        cli, "open_provider", lambda *a: pytest.fail("Report must not judge")
    )
    titles = []
    savefig = matplotlib.figure.Figure.savefig

    def capture_plot(figure, *args, **kwargs):
        titles.append(figure.axes[0].get_title())
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_plot)
    before = snapshot_inputs(output)  # No failures.json is required.
    result = CliRunner().invoke(
        cli.app,
        [
            "report",
            "--method",
            "jev",
            "--data-dir",
            str(tmp_path),
            "--local-runs",
            "--allow-partial",
        ],
    )
    assert result.exit_code == 0, result.output
    assert snapshot_inputs(output) == before
    assert not (output / "failures.json").exists()
    assert len(titles) == 5
    assert all(
        "Partial report" in title and "3/5 (60.000%)" in title for title in titles
    )
    partial = read_result(output / "partial")
    assert partial["coverage"]["expected_queries"] == 3
    assert partial["coverage"]["evaluated_queries"] == 2
    assert partial["bootstrap"]["query_count"] == 2  # Omit the wholly missing q3.
    assert partial["bootstrap"]["completed_replicates"] == 10000
    assert all("95% bootstrap CI" in title for title in titles)
    assert partial["coverage"]["refusal_count"] == 0
    assert [
        pair["reference_relevance"] for pair in partial["coverage"]["missing_pairs"]
    ] == [3, 3]
    scores = json.loads((output / "partial" / "run_scores.json").read_text())
    assert scores["good.run"]["human"]["aggregate"]["AP(rel=2)"] == 0.75
    assert scores["bad.run"]["human"]["aggregate"]["AP(rel=2)"] == pytest.approx(2 / 3)
    assert scores["good.run"]["human"]["aggregate"]["nDCG@10"] == pytest.approx(
        (1 / math.log2(3) + 1) / 2
    )
    assert scores["bad.run"]["human"]["aggregate"]["nDCG@10"] == 0.75
    assert all(
        set(source["per_query"]) == {"q1", "q2"}
        for run in scores.values()
        for source in run.values()
    )
    assert partial["agreement"]["counts"] == [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ]
    manual_output = tmp_path / "manual"
    manual = performance_report(
        judgment_dataset(successful),
        "jev",
        run_paths=run_paths,
        output_dir=manual_output,
        dataset=DEFAULT_DATASET,
        model="jev-latest",
        token_costs=token_prices("jev", "jev-latest"),
    )
    assert json.loads((manual_output / "run_scores.json").read_text()) == scores
    for key in ("agreement", "system_rank_correlations", "usage", "cost"):
        assert partial[key] == manual[key]
    assert all("Partial report" not in title for title in titles[5:])


def test_partial_mode_does_not_hide_initialization_or_reporting_errors(
    monkeypatch, tmp_path, tables, run_paths
):
    from relevance_bench.reporting import reports

    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    output = tmp_path / "jev"
    write_checkpoint(output, tables, "jev", [judgment(tables["qrels"][0])])

    @asynccontextmanager
    async def provider(*args):
        raise RuntimeError("provider initialization failed")
        yield

    monkeypatch.setattr(cli, "open_provider", provider)
    args = [
        "--method",
        "jev",
        "--data-dir",
        str(tmp_path),
        "--local-runs",
        "--allow-partial",
    ]
    result = CliRunner().invoke(cli.app, ["run", *args])
    assert result.exit_code == 1 and "initialization failed" in str(result.exception)

    def fail_report(*args, **kwargs):
        raise RuntimeError("report generation failed")

    monkeypatch.setattr(reports, "performance_report", fail_report)
    result = CliRunner().invoke(cli.app, ["report", *args])
    assert result.exit_code == 1 and "report generation failed" in str(result.exception)
    result = CliRunner().invoke(cli.app, ["run", *args, "--reasoning-effort", "low"])
    assert result.exit_code == 2
