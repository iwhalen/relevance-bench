import itertools
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.stats import kendalltau
from threadpoolctl import threadpool_info
from typer.testing import CliRunner

from relevance_bench import __main__ as cli
from relevance_bench.checkpoints import judgment_dataset
from relevance_bench.data_handlers.datasets import DEFAULT_DATASET, DatasetSpec
from relevance_bench.reporting import bootstrap as boot
from relevance_bench.reporting.bootstrap import BootstrapConfig
from relevance_bench.reporting.metrics import agreement_report
from relevance_bench.reporting.reports import performance_report


@pytest.fixture
def example():
    # Unequal judgment counts, a constant query, and two different retrieval measures.
    grades = [[(0, 0), (2, 1), (3, 3)], [(0, 0)], [(1, 2), (2, 2)]]
    rows = [
        {
            "query_id": f"q{i}",
            "passage_id": f"p{i}-{j}",
            "human_relevance": h,
            "predicted_relevance": p,
        }
        for i, group in enumerate(grades)
        for j, (h, p) in enumerate(group)
    ]
    matrices = {
        "a": (
            np.array([[0, 0.5, 1], [0, 0, 0], [1, 0.5, 0]]),
            np.array([[1, 0.5, 0], [0, 0, 0], [0.5, 0.5, 1]]),
        ),
        "b": (
            np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]]),
            np.array([[0, 1, 1], [0, 0, 0], [1, 1, 0]]),
        ),
    }
    scores = {
        f"s{s}": {
            source: {
                "aggregate": {
                    key: float(pair[side][:, s].mean())
                    for key, pair in matrices.items()
                },
                "per_query": {
                    f"q{q}": {
                        key: float(pair[side][q, s]) for key, pair in matrices.items()
                    }
                    for q in range(3)
                },
            }
            for side, source in enumerate(("human", "jev"))
        }
        for s in range(3)
    }
    return rows, scores, matrices


@pytest.mark.parametrize(
    "human,predicted",
    [
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([1, 1, 2, 3], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 1, 2, 3]),
        ([1, 1, 2, 3], [1, 1, 3, 2]),
        ([1, 1, 2, 2], [1, 2, 1, 2]),
        ([1, 1, 1, 1], [1, 2, 3, 4]),
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([0, 1], [1, 0]),
    ],
)
def test_tau_matches_scipy(human, predicted):
    expected = kendalltau(human, predicted, variant="b").statistic
    actual = boot.tau_b(np.array([human]), np.array([predicted]))[0]
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_tau_blocks_large_run_collection(monkeypatch):
    rng = np.random.default_rng(42)
    human = rng.integers(0, 10, size=(7, 120))
    predicted = rng.integers(0, 10, size=(7, 120))
    expected = [
        kendalltau(h, p).statistic for h, p in zip(human, predicted, strict=True)
    ]
    np.testing.assert_allclose(boot.tau_b(human, predicted), expected)
    monkeypatch.setattr(boot, "PAIR_BLOCK_SIZE", 13)
    np.testing.assert_allclose(boot.tau_b(human, predicted), expected)
    with pytest.raises(ValueError, match="matching"):
        boot.tau_b(human, predicted[:1])


def repeated_reference(rows, matrices, query_indices):
    repeated = [
        {**row, "passage_id": f"copy{copy}-{row['passage_id']}"}
        for copy, q in enumerate(query_indices)
        for row in rows
        if row["query_id"] == f"q{q}"
    ]
    kappa = agreement_report(repeated)["cohen_kappa"]
    return (
        np.nan if kappa is None else kappa,
        {
            key: kendalltau(
                h[query_indices].mean(axis=0), p[query_indices].mean(axis=0)
            ).statistic
            for key, (h, p) in matrices.items()
        },
    )


def test_batch_matches_explicitly_repeated_queries(example):
    rows, scores, matrices = example
    indices = [list(q) for q in itertools.product(range(3), repeat=3)]
    weights = np.array([np.bincount(q, minlength=3) for q in indices])
    counts, prepared = boot.query_matrices(rows, scores, "jev")
    kappa, taus = boot.resample_batch(weights, counts, prepared)
    for i, selected in enumerate(indices):
        expected_kappa, expected_taus = repeated_reference(rows, matrices, selected)
        np.testing.assert_allclose(kappa[i], expected_kappa, equal_nan=True)
        for key in matrices:
            np.testing.assert_allclose(taus[key][i], expected_taus[key], equal_nan=True)
    # This also distinguishes pooling judgments from averaging per-query kappas.
    query_kappas = boot.kappa_from_counts(counts)
    assert kappa[5] != pytest.approx(np.nanmean(query_kappas[[0, 1, 2]]))


def test_report_percentiles_match_slow_reference(example):
    rows, scores, matrices = example
    config = BootstrapConfig(resamples=533, confidence=0.8, seed=42, workers=1)
    kappas = []
    taus = {key: [] for key in matrices}
    for index, size in enumerate((512, 21)):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index]))
        for weights in rng.multinomial(3, [1 / 3] * 3, size=size):
            selected = np.repeat(np.arange(3), weights).tolist()
            kappa, correlations = repeated_reference(rows, matrices, selected)
            kappas.append(kappa)
            for key, value in correlations.items():
                taus[key].append(value)
    metadata, kappa, correlations = boot.bootstrap_report(rows, scores, "jev", config)
    assert metadata["completed_replicates"] == 533
    assert metadata["configuration"] == {
        "resamples": 533,
        "confidence": 0.8,
        "seed": 42,
        "workers": 1,
    }
    assert metadata["resampling_unit"] == "query"
    assert metadata["method"] == "percentile" and metadata["excludes_undefined"]
    assert kappa == boot.percentile_interval(np.array(kappas), 0.8)
    for key, values in taus.items():
        expected = boot.percentile_interval(np.array(values), 0.8)
        actual = correlations[key].copy()
        for bound in ("lower", "upper"):
            assert actual.pop(bound) == pytest.approx(expected.pop(bound), abs=1e-14)
        assert actual == expected
        assert correlations[key]["undefined_replicates"] > 0


def test_repeatability_across_workers_and_order(example):
    rows, scores, _ = example
    config = BootstrapConfig(resamples=2053, seed=37, workers=1)
    serial = boot.bootstrap_report(rows, scores, "jev", config)
    assert boot.bootstrap_report(rows, scores, "jev", config) == serial
    for workers in (0, 4):
        result = boot.bootstrap_report(
            rows, scores, "jev", replace(config, workers=workers)
        )
        assert result[1:] == serial[1:]
    reordered = {
        system: {
            source: {
                **data,
                "per_query": dict(reversed(list(data["per_query"].items()))),
            }
            for source, data in reversed(list(sources.items()))
        }
        for system, sources in reversed(list(scores.items()))
    }
    assert boot.bootstrap_report(rows[::-1], reordered, "jev", config) == serial
    assert (
        boot.bootstrap_report(rows, scores, "jev", replace(config, seed=38))[1:]
        != serial[1:]
    )


def test_percentile_finite_filter_and_insufficient_outcomes():
    interval = boot.percentile_interval(
        np.array([0, 1, 2, 3, np.nan, np.inf, -np.inf]), 0.5
    )
    assert interval == {
        "confidence": 0.5,
        "lower": 0.75,
        "upper": 2.25,
        "valid_replicates": 4,
        "undefined_replicates": 3,
        "unavailable_reason": None,
    }
    for values in ([], [np.nan, np.inf], [1, np.nan]):
        result = boot.percentile_interval(np.array(values), 0.95)
        assert result["lower"] is result["upper"] is None
        assert (
            result["unavailable_reason"] == "Fewer than two finite bootstrap replicates"
        )
        json.dumps(result, allow_nan=False)


def test_all_undefined_and_one_replicate(example):
    rows, scores, _ = example
    for row in rows:
        row["human_relevance"] = row["predicted_relevance"] = 0
    for run in scores.values():
        for source in run.values():
            for values in source["per_query"].values():
                values.update(dict.fromkeys(values, 0))
    result = boot.bootstrap_report(rows, scores, "jev", BootstrapConfig(resamples=17))
    assert result[0]["completed_replicates"] == 17
    for interval in [result[1], *result[2].values()]:
        assert interval["valid_replicates"] == 0
        assert interval["undefined_replicates"] == 17
        assert interval["lower"] is interval["upper"] is None
        assert interval["unavailable_reason"]
    json.dumps(result, allow_nan=False)
    result = boot.bootstrap_report(rows, scores, "jev", BootstrapConfig(resamples=1))
    assert result[1]["unavailable_reason"]


def test_single_query_and_disabled_skip_resampling(monkeypatch, example):
    rows, scores, _ = example
    monkeypatch.setattr(
        boot, "resample_batch", lambda *a: pytest.fail("Must skip draws")
    )
    result = boot.bootstrap_report(rows[:3], scores, "jev", BootstrapConfig())
    assert result[0]["query_count"] == 1
    assert result[0]["completed_replicates"] == result[0]["workers_used"] == 0
    for interval in [result[1], *result[2].values()]:
        assert interval["unavailable_reason"] == "Fewer than two evaluated queries"
        assert interval["valid_replicates"] == interval["undefined_replicates"] == 0
    monkeypatch.setattr(
        boot, "query_matrices", lambda *a: pytest.fail("Disabled must skip preparation")
    )
    assert (
        boot.bootstrap_report(rows, scores, "jev", BootstrapConfig(resamples=0)) is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resamples": -1},
        {"resamples": 1.5},
        {"resamples": True},
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": -0.1},
        {"confidence": np.nan},
        {"confidence": np.inf},
        {"confidence": "0.95"},
        {"confidence": True},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": False},
        {"workers": -1},
        {"workers": 1.5},
        {"workers": True},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="Bootstrap"):
        BootstrapConfig(**kwargs)


def test_immutable_defaults():
    config = BootstrapConfig()
    assert (config.resamples, config.confidence, config.seed, config.workers) == (
        10000,
        0.95,
        0,
        0,
    )
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


def test_worker_bounds(monkeypatch):
    monkeypatch.setattr(boot.os, "process_cpu_count", lambda: 16)
    config = BootstrapConfig()
    assert boot.worker_count(config, 0) == 0
    assert boot.worker_count(config, 3) == 1
    assert boot.worker_count(config, 4) == 4
    assert boot.worker_count(replace(config, workers=8), 3) == 3
    monkeypatch.setattr(boot.os, "process_cpu_count", lambda: 2)
    assert boot.worker_count(config, 5) == 2
    monkeypatch.setattr(boot.os, "process_cpu_count", lambda: None)
    assert boot.worker_count(config, 5) == 1


def test_blas_threads_limited_and_restored(monkeypatch, example):
    before = threadpool_info()
    original = boot.resample_batch

    def checked(*args):
        assert all(
            pool["num_threads"] == 1
            for pool in threadpool_info()
            if pool["user_api"] == "blas"
        )
        return original(*args)

    monkeypatch.setattr(boot, "resample_batch", checked)
    boot.bootstrap_report(
        *example[:2], "jev", BootstrapConfig(resamples=1025, workers=3)
    )
    assert threadpool_info() == before


@pytest.mark.parametrize("command", ["run", "report"])
@pytest.mark.parametrize(
    "flag,value",
    [
        ("--bootstrap-resamples", "-1"),
        ("--bootstrap-confidence", "0"),
        ("--bootstrap-confidence", "1"),
        ("--bootstrap-confidence", "nan"),
        ("--bootstrap-confidence", "inf"),
        ("--bootstrap-seed", "-1"),
        ("--bootstrap-workers", "-1"),
    ],
)
def test_cli_validation_before_network(command, flag, value, monkeypatch):
    def fail(*a, **k):
        pytest.fail("Bootstrap validation must precede network or judging")

    monkeypatch.setattr(cli, "resolve_runs", fail)
    monkeypatch.setattr(cli, "provider_config", fail)
    monkeypatch.setattr(DatasetSpec, "load", fail)
    assert CliRunner().invoke(cli.app, [command, flag, value]).exit_code == 2


def test_cli_propagation_json_disabled_and_checkpoint_compatibility(
    monkeypatch,
    tmp_path,
    tables,
    run_paths,
    fake_providers,
):
    monkeypatch.setattr(DatasetSpec, "load", lambda self, table: tables[table])
    import matplotlib.figure

    titles = []
    original = matplotlib.figure.Figure.savefig

    def capture(figure, *args, **kwargs):
        titles.append(figure.axes[0].get_title())
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    args = ["--data-dir", str(tmp_path), "--local-runs"]
    flags = [
        "--bootstrap-resamples",
        "513",
        "--bootstrap-confidence",
        "0.8",
        "--bootstrap-seed",
        "27",
        "--bootstrap-workers",
        "2",
    ]
    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", *args, *flags])
    assert result.exit_code == 0, result.output
    output = tmp_path / "jev"
    initial = json.loads((output / "results.json").read_text())
    terminal, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    assert terminal["bootstrap"] == initial["bootstrap"]
    assert initial["bootstrap"]["configuration"] == {
        "resamples": 513,
        "confidence": 0.8,
        "seed": 27,
        "workers": 2,
    }
    inputs = {
        p.name: p.read_bytes()
        for p in output.iterdir()
        if p.name
        in {"judgments.jsonl", "experiment.json", "failures.json", "run_scores.json"}
    }
    result = runner.invoke(cli.app, ["report", *args, *flags])
    assert result.exit_code == 0, result.output
    assert json.loads((output / "results.json").read_text()) == initial
    # Bootstrap settings can change while resuming the exact same checkpoint.
    for command in ("run", "report"):
        result = runner.invoke(cli.app, [command, *args, "--bootstrap-resamples", "0"])
        assert result.exit_code == 0, result.output
        disabled = json.loads((output / "results.json").read_text())
        terminal, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
        assert "bootstrap" not in disabled and "bootstrap" not in terminal
        assert "bootstrap" not in disabled["agreement"]
        assert all(
            "bootstrap" not in c for c in disabled["system_rank_correlations"].values()
        )
        for key, content in inputs.items():
            assert (output / key).read_bytes() == content
    assert len(fake_providers) == 1
    assert all(
        "80% bootstrap CI unavailable: Fewer than two evaluated queries" in title
        for title in titles[:10]
    )
    assert all("bootstrap" not in title for title in titles[10:])


def test_enabled_disabled_points_scores_artifacts_and_titles(
    monkeypatch, tmp_path, tables, run_paths
):
    import matplotlib.figure

    rows = [
        {
            "query_id": qid,
            "passage_id": row["passage_id"],
            "human_relevance": row["relevance"],
            "predicted_relevance": row["relevance"],
        }
        for qid in ("q1", "q2")
        for row in tables["qrels"]
    ]
    titles = []
    original = matplotlib.figure.Figure.savefig

    def capture(figure, *args, **kwargs):
        titles.append(figure.axes[0].get_title())
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    kwargs = {
        "judgments": judgment_dataset(rows),
        "method": "jev",
        "run_paths": run_paths,
        "dataset": DEFAULT_DATASET,
        "model": "jev-latest",
        "token_costs": None,
    }
    enabled = performance_report(**kwargs, output_dir=tmp_path / "enabled")
    disabled = performance_report(
        **kwargs,
        output_dir=tmp_path / "disabled",
        bootstrap=BootstrapConfig(resamples=0),
    )
    assert len(titles) == 10
    assert all("95% bootstrap CI" in title for title in titles[:5])
    assert all("bootstrap" not in title for title in titles[5:])
    assert {p.name for p in (tmp_path / "enabled").iterdir()} == {
        p.name for p in (tmp_path / "disabled").iterdir()
    }
    assert (tmp_path / "enabled" / "run_scores.json").read_bytes() == (
        tmp_path / "disabled" / "run_scores.json"
    ).read_bytes()
    assert enabled.pop("bootstrap")["completed_replicates"] == 10000
    kappa = enabled["agreement"].pop("bootstrap")
    assert kappa["lower"] == kappa["upper"] == 1
    for correlation in enabled["system_rank_correlations"].values():
        interval = correlation.pop("bootstrap")
        assert interval["lower"] == interval["upper"] == 1
        assert interval["valid_replicates"] + interval["undefined_replicates"] == 10000
    assert enabled == disabled
    assert "NaN" not in (tmp_path / "enabled" / "results.json").read_text()
    assert "Infinity" not in (tmp_path / "enabled" / "results.json").read_text()
