import gzip
import json
from dataclasses import replace
from io import BytesIO

import pytest
from typer.testing import CliRunner

from relevance_bench import __main__ as cli
from relevance_bench.checkpoints import judgment_dataset
from relevance_bench.data_handlers import trec_runs
from relevance_bench.data_handlers.datasets import (
    DATASETS,
    DEFAULT_DATASET,
    DatasetSpec,
)
from relevance_bench.providers import token_prices
from relevance_bench.reporting.metrics import agreement_report, usage_report
from relevance_bench.reporting.reports import performance_report


def test_agreement_and_retrieval_metrics(tmp_path, tables, run_paths):
    rows = [
        {
            "query_id": r["query_id"],
            "passage_id": r["passage_id"],
            "human_relevance": r["relevance"],
            "predicted_relevance": r["relevance"],
        }
        for r in tables["qrels"]
    ]
    report = performance_report(
        judgment_dataset(rows),
        "jev",
        run_paths=run_paths,
        output_dir=tmp_path / "jev",
        dataset=DEFAULT_DATASET,
        model="jev-latest",
        token_costs=token_prices("jev", "jev-latest"),
    )
    assert (
        report["agreement"]["exact_agreement"]
        == report["agreement"]["cohen_kappa"]
        == 1
    )
    assert all(c["tau_b"] == 1 for c in report["system_rank_correlations"].values())
    scores = json.loads((tmp_path / "jev" / "run_scores.json").read_text())
    assert scores["good.run"]["human"]["aggregate"]["AP(rel=2)"] == 1
    assert scores["bad.run"]["human"]["aggregate"]["AP(rel=2)"] == 0.5
    assert scores["bad.run"]["human"]["aggregate"]["nDCG@10"] == pytest.approx(
        0.6309297536
    )


def test_constant_scores_generate_all_artifacts(tmp_path, tables, run_paths):
    rows = [
        {
            "query_id": r["query_id"],
            "passage_id": r["passage_id"],
            "human_relevance": 0,
            "predicted_relevance": 0,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }
        for r in tables["qrels"]
    ]
    report = performance_report(
        judgment_dataset(rows),
        "openai-decisions",
        run_paths=run_paths,
        output_dir=tmp_path / "openai-decisions",
        dataset=DEFAULT_DATASET,
        model="gpt-6-luna",
        token_costs=token_prices("openai-decisions", "gpt-6-luna"),
    )
    assert report["agreement"]["cohen_kappa"] is None
    assert all(c["tau_b"] is None for c in report["system_rank_correlations"].values())
    expected = {
        "agreement_counts.png",
        "agreement_normalized.png",
        "results.json",
        "run_scores.json",
        "system_scores_nDCG_10.png",
        "system_scores_nDCG_100.png",
        "system_scores_AP_rel_2_.png",
    }
    assert {p.name for p in (tmp_path / "openai-decisions").iterdir()} == expected
    assert "NaN" not in (tmp_path / "openai-decisions" / "results.json").read_text()


def test_method_prices_and_missing_usage():
    rows = [{"usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}]
    responses = usage_report(
        rows, model="gpt-6-luna", token_costs=token_prices("openai", "gpt-6-luna")
    )
    decisions = usage_report(
        rows,
        model="gpt-6-luna",
        token_costs=token_prices("openai-decisions", "gpt-6-luna"),
    )
    assert responses["cost"]["total"] == 0.6
    assert decisions["cost"]["total"] == 0.1 and decisions["cost"]["output"] == 0
    partial = usage_report(
        [*rows, {"usage": {"input_tokens": None, "output_tokens": None}}],
        model="gpt-6-luna",
        token_costs=token_prices("openai-decisions", "gpt-6-luna"),
    )
    assert (
        partial["usage"]["judgments_without_usage"] == 1
        and not partial["cost"]["complete"]
    )
    unknown = usage_report(rows, model="unknown", token_costs=None)
    assert unknown["cost"]["total"] is None and not unknown["cost"]["complete"]
    with pytest.raises(ValueError, match="No judgments"):
        agreement_report([])


def test_compressed_run_and_invalid_runs(tmp_path, run_paths):
    compressed = tmp_path / "run.txt.gz"
    with gzip.open(compressed, "wt") as output:
        output.write(run_paths[0].read_text())
    assert trec_runs.load_run(compressed) == trec_runs.load_run(run_paths[0])
    invalid = tmp_path / "invalid.run"
    for content in (
        "",
        "not six fields",
        "q1 Q0 invalid 0 1 x\n",
        "q1 Q0 msmarco_passage_1 -1 1 x\n",
        "q1 Q0 msmarco_passage_1 0 nan x\n",
        "q1 Q0 msmarco_passage_1 0 1 x\n" * 2,
    ):
        invalid.write_text(content)
        with pytest.raises(ValueError):
            trec_runs.load_run(invalid)


@pytest.mark.parametrize("compressed", [False, True])
def test_real_trec_dl_2023_passage_ids(tmp_path, compressed):
    path = tmp_path / (
        "agg-cocondenser.run.gz" if compressed else "agg-cocondenser.run"
    )
    content = (
        "2000138 Q0 msmarco_passage_59_841186336 1 48.404667 agg-cocondenser\n"
        "2000138 Q0 msmarco_passage_04_207262207 2 48.402252 agg-cocondenser\n"
    )
    if compressed:
        with gzip.open(path, "wt") as output:
            output.write(content)
    else:
        path.write_text(content)
    assert trec_runs.load_run(path) == {
        "2000138": {
            "msmarco_passage_59_841186336": 48.404667,
            "msmarco_passage_04_207262207": 48.402252,
        },
    }


@pytest.mark.parametrize(
    "passage_id",
    [
        "msmarco_passage_",
        "msmarco_passage_59_",
        "msmarco_passage_59_abc",
        "msmarco_passage_59_841186336_extra",
        "msmarco_passage_59_1_2",
    ],
)
def test_malformed_passage_ids_remain_invalid(passage_id):
    assert not DEFAULT_DATASET.valid_passage_id(passage_id)


def test_download_command_and_cache(monkeypatch, tmp_path, run_paths):
    dataset = replace(DEFAULT_DATASET, run_names=("first", "second"))
    monkeypatch.setitem(DATASETS, "trec-dl-2023", dataset)
    monkeypatch.setattr(
        DatasetSpec, "load", lambda *a: pytest.fail("Downloads must not load tables")
    )
    requests = []

    def download(url, timeout):
        requests.append(url)
        return BytesIO(run_paths[0].read_bytes())

    monkeypatch.setattr(trec_runs, "urlopen", download)
    cache = tmp_path / "downloaded"
    args = ["download-runs", "--data-dir", str(cache)]
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert len(requests) == 2
    assert CliRunner().invoke(cli.app, args).exit_code == 0
    assert len(requests) == 2
    manifest = json.loads((cache / "runs" / "manifest.json").read_text())
    assert len(manifest) == 2 and all(len(entry["sha256"]) == 64 for entry in manifest)


def test_invalid_download_is_not_cached(monkeypatch, tmp_path):
    dataset = replace(DEFAULT_DATASET, run_names=("invalid",))
    monkeypatch.setattr(trec_runs, "urlopen", lambda *a, **k: BytesIO(b"invalid"))
    with pytest.raises(RuntimeError, match="Cannot download"):
        trec_runs.download_runs(tmp_path, dataset)
    assert not list(tmp_path.glob("*.run*"))
