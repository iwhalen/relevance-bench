import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from relevance_bench import __main__ as cli
from relevance_bench.checkpoints import (
    digest,
    prepare_checkpoint,
    read_jsonl,
    save_json,
)
from relevance_bench.data_handlers.datasets import DEFAULT_DATASET, DatasetSpec
from relevance_bench.errors import ContextLimitError
from relevance_bench.providers import PROVIDERS, provider_config
from relevance_bench.providers.base import Judgment
from relevance_bench.reporting.checkpoints import report_checkpoint
from relevance_bench.rubric import CRITERIA, INSTRUCTIONS


@pytest.mark.parametrize("method", ["jev", "openai", "openai-decisions"])
def test_cli_run_report_and_resume(
    method, monkeypatch, tables, run_paths, tmp_path, fake_providers
):
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    runner = CliRunner()
    args = ["--method", method, "--data-dir", str(tmp_path), "--local-runs"]
    result = runner.invoke(cli.app, ["run", *args])
    assert result.exit_code == 0, result.output
    assert fake_providers[0][1].await_count == 2
    rows = read_jsonl(tmp_path / method / "judgments.jsonl")
    assert len(rows) == 2 and all(r["predicted_relevance"] == 1 for r in rows)
    assert all(r["latency_seconds"] >= r["time_taken"] for r in rows)
    assert (tmp_path / method / "results.json").exists()
    tables["qrels"][0]["relevance"] = 2
    result = runner.invoke(cli.app, ["run", *args])
    assert result.exit_code == 0, result.output
    assert len(fake_providers) == 1

    def no_client(*args):
        pytest.fail("Reporting must not initialize a client")

    monkeypatch.setattr(cli, "open_provider", no_client)
    result = runner.invoke(cli.app, ["report", *args])
    assert result.exit_code == 0, result.output
    scores = json.loads((tmp_path / method / "run_scores.json").read_text())
    assert scores["good.run"]["human"]["aggregate"]["nDCG@10"] == 1


@pytest.mark.parametrize("method", ["jev", "openai"])
def test_legacy_signature_and_changed_configuration(method, tmp_path, tables):
    config = provider_config(method)
    configuration = PROVIDERS[method].provider_type.configuration(config)
    inputs = [("q1", p["passage_id"], "query", p["text"]) for p in tables["passages"]]
    payload = {
        "model": config.model_name,
        "instructions": INSTRUCTIONS,
        "criteria": CRITERIA,
        "inputs": inputs,
        **configuration,
    }
    signature = digest(payload)
    if method == "jev":
        # Calculated from the original main.py, before the migration.
        assert (
            signature
            == "5408cb31415e1ccde1db33b05675e57a806ba2a9bcb85419248c1b909389a141"
        )
    save_json(
        tmp_path / "experiment.json",
        {
            "signature": signature,
            "dataset": DEFAULT_DATASET.repository,
            "model": config.model_name,
            **configuration,
        },
    )
    row = {
        "query_id": "q1",
        "passage_id": "msmarco_passage_1",
        "human_relevance": 3,
        "predicted_relevance": 2,
    }
    (tmp_path / "judgments.jsonl").write_text(json.dumps(row) + "\n")
    kwargs = dict(
        dataset=DEFAULT_DATASET,
        method=method,
        model=config.model_name,
        configuration=configuration,
        inputs=inputs,
        pairs=[(r["query_id"], r["passage_id"]) for r in tables["qrels"]],
        concurrency=16,
    )
    assert len(prepare_checkpoint(tmp_path, **kwargs)) == 1
    assert (
        json.loads((tmp_path / "experiment.json").read_text())["signature_version"] == 2
    )
    assert len(prepare_checkpoint(tmp_path, **{**kwargs, "concurrency": 1})) == 1
    for change in (
        {"model": "different"},
        {"method": "openai-decisions"},
        {"dataset": replace(DEFAULT_DATASET, repository="another/dataset")},
        {"inputs": [("q1", "msmarco_passage_1", "changed", "passage")]},
    ):
        with pytest.raises(ValueError, match="Cached experiment differs"):
            prepare_checkpoint(tmp_path, **{**kwargs, **change})


def test_truncation_policy_is_in_signature(monkeypatch, tmp_path):
    from relevance_bench import rubric

    row = {
        "query_id": "q1",
        "passage_id": "msmarco_passage_1",
        "human_relevance": 3,
        "predicted_relevance": 2,
    }
    kwargs = dict(
        dataset=DEFAULT_DATASET,
        method="jev",
        model="jev-latest",
        configuration=PROVIDERS["jev"].provider_type.configuration(
            provider_config("jev")
        ),
        inputs=[("q1", "msmarco_passage_1", "q", "p")],
        pairs=[("q1", "msmarco_passage_1")],
        concurrency=1,
    )
    prepare_checkpoint(tmp_path, **kwargs)
    (tmp_path / "judgments.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(rubric, "REQUEST_CHAR_BUDGET", 10)
    with pytest.raises(ValueError, match="Cached experiment differs"):
        prepare_checkpoint(tmp_path, **kwargs)


def test_failure_resume_and_context_reduction(monkeypatch, tmp_path, tables):
    calls = []
    fail_second = [True]

    async def adjudge(query, passage):
        calls.append(passage)
        if fail_second[0] and passage == "short":
            raise ValueError("refused")
        if len(passage) > 4:
            raise ContextLimitError("too long")
        return Judgment(2, None, None, 0, model="jev-latest")

    @asynccontextmanager
    async def provider(method, config):
        yield SimpleNamespace(adjudge=adjudge)

    monkeypatch.setattr(cli, "open_provider", provider)
    kwargs = dict(
        output_dir=tmp_path, config=provider_config("jev"), concurrency=1, method="jev"
    )
    with pytest.raises(RuntimeError, match="1 judgments failed"):
        asyncio.run(
            cli.judge_async(
                tables["queries"], tables["passages"], tables["qrels"], **kwargs
            )
        )
    saved = read_jsonl(tmp_path / "judgments.jsonl")
    assert len(saved) == 1 and saved[0]["length_retries"] == 2
    assert (
        saved[0]["passage_original_chars"] == 12 and saved[0]["passage_used_chars"] == 3
    )
    assert saved[0]["passage_truncated"] is True
    assert (
        json.loads((tmp_path / "failures.json").read_text())[0]["type"] == "ValueError"
    )
    calls.clear()
    fail_second[0] = False
    result = asyncio.run(
        cli.judge_async(
            tables["queries"], tables["passages"], tables["qrels"], **kwargs
        )
    )
    assert len(result) == 2 and calls == ["short", "sh"]
    assert json.loads((tmp_path / "failures.json").read_text()) == []


def test_decisions_refusal_details_saved_and_resume(
    tmp_path, tables, fake_providers, responses
):
    response = responses["openai-decisions"]
    score = response.answers[0]
    response.answers = [SimpleNamespace(name="relevance", type="refusal")]
    kwargs = dict(
        output_dir=tmp_path,
        config=provider_config("openai-decisions"),
        concurrency=1,
        method="openai-decisions",
    )
    with pytest.raises(RuntimeError, match="2 judgments failed"):
        asyncio.run(cli.judge_async(**tables, **kwargs))
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert len(failures) == 2
    for failure, row in zip(failures, tables["qrels"], strict=True):
        assert failure == {
            "query_id": row["query_id"],
            "passage_id": row["passage_id"],
            "error": "OpenAI decision refused relevance question",
            "type": "ProviderResponseError",
            "details": {
                "model": "gpt-6-luna",
                "request_id": "decisions-request",
                "answers": [{"name": "relevance", "type": "refusal"}],
            },
        }
    assert read_jsonl(tmp_path / "judgments.jsonl") == []
    assert fake_providers[0][1].await_count == 2
    response.answers = [score]
    judgments = asyncio.run(cli.judge_async(**tables, **kwargs))
    assert len(judgments) == 2
    assert json.loads((tmp_path / "failures.json").read_text()) == []
    asyncio.run(cli.judge_async(**tables, **kwargs))
    assert len(fake_providers) == 2


def test_context_failure_at_one_character(monkeypatch, tmp_path, tables):
    calls = AsyncMock(side_effect=ContextLimitError("too long"))

    @asynccontextmanager
    async def provider(method, config):
        yield SimpleNamespace(adjudge=calls)

    monkeypatch.setattr(cli, "open_provider", provider)
    with pytest.raises(RuntimeError, match="2 judgments failed"):
        asyncio.run(
            cli.judge_async(
                tables["queries"],
                tables["passages"],
                tables["qrels"],
                output_dir=tmp_path,
                config=provider_config("jev"),
                concurrency=1,
                method="jev",
            )
        )
    assert calls.await_count == 7


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--dataset", "unknown"],
        ["run", "--method", "unknown"],
        ["run", "--concurrency", "0"],
        ["run", "--method", "openai", "--openai-rpm", "1"],
        ["run", "--reasoning-effort", "low"],
        ["run", "--method", "openai-decisions", "--openai-tpm", "1"],
        ["run", "--method", "openai-decisions", "--reasoning-effort", "none"],
        ["run", "--method", "openai-decisions", "--model", "unknown"],
        ["run", "--method", "openai", "--reasoning-effort", "unknown"],
    ],
)
def test_cli_validation_before_network(arguments, monkeypatch):
    monkeypatch.setattr(
        cli,
        "resolve_runs",
        lambda *a, **k: pytest.fail("Validation must precede network"),
    )
    assert CliRunner().invoke(cli.app, arguments).exit_code == 2


def test_incomplete_and_duplicate_report(tmp_path, tables):
    save_json(
        tmp_path / "experiment.json",
        {"dataset": DEFAULT_DATASET.repository, "method": "jev"},
    )
    row = {
        "query_id": "q1",
        "passage_id": "msmarco_passage_1",
        "human_relevance": 3,
        "predicted_relevance": 2,
    }
    path = tmp_path / "judgments.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="complete checkpoint"):
        report_checkpoint(tmp_path, tables["qrels"], DEFAULT_DATASET, "jev")
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate judgment"):
        report_checkpoint(tmp_path, tables["qrels"], DEFAULT_DATASET, "jev")
