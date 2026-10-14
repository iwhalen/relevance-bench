"""Generate pseudo relevance judgments and compare them with human assessments.

Use run to judge passages and evaluate retrieval systems, report to evaluate an
existing checkpoint, compare to compare saved reports locally, or download-runs
to cache the reference retrieval runs.
Invoke as relevance-bench or python -m relevance_bench.
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Annotated

import typer
from tenacity import AsyncRetrying, retry_if_exception_type
from tqdm import tqdm

from .checkpoints import (
    prepare_checkpoint,
    save_json,
    successful_judgments,
    validate_judgments,
)
from .data_handlers.datasets import DATASETS, DEFAULT_DATASET, DatasetName
from .data_handlers.trec_runs import download_runs, resolve_runs
from .errors import ContextLimitError, ProviderResponseError, RetryableError
from .providers import PROVIDERS, Method, open_provider, provider_config
from .providers.openai import REASONING_EFFORTS
from .reporting.bootstrap import BootstrapConfig
from .reporting.checkpoints import report_checkpoint
from .reporting.reports import print_report
from .rubric import truncate_passage

app = typer.Typer(help=__doc__, no_args_is_help=True)


def retry_wait(state) -> float:
    error = state.outcome.exception()
    delay = min(60, 2 ** (state.attempt_number - 1)) + random.uniform(0, 0.5)
    return max(delay, error.retry_after or 0)


def retry_stop(state) -> bool:
    error = state.outcome.exception()
    return state.attempt_number >= (9 if error.rate_limited else 5)


async def adjudge_with_retry(provider, query: str, passage: str):
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(RetryableError),
        stop=retry_stop,
        wait=retry_wait,
        reraise=True,
    ):
        with attempt:
            return await provider.adjudge(query, passage)


async def judge_async(
    queries,
    passages,
    qrels,
    *,
    output_dir: Path,
    config,
    concurrency: int,
    method: str,
    dataset=DEFAULT_DATASET,
    allow_partial: bool = False,
):
    """Judge independently with bounded workers and durable per-pair checkpoints."""
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    output_dir = Path(output_dir)
    query_text = {row["query_id"]: row["text"] for row in queries}
    passage_text = {row["passage_id"]: row["text"] for row in passages}
    pairs = [(row["query_id"], row["passage_id"]) for row in qrels]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate reference query-passage pairs")
    for qid, pid in pairs:
        if qid not in query_text or pid not in passage_text:
            raise ValueError(f"Missing query or passage: {qid}, {pid}")
    completed = prepare_checkpoint(
        output_dir,
        dataset=dataset,
        method=method,
        model=config.model_name,
        configuration=PROVIDERS[method].provider_type.configuration(config),
        inputs=[(qid, pid, query_text[qid], passage_text[pid]) for qid, pid in pairs],
        pairs=pairs,
        concurrency=concurrency,
    )
    pending = iter(
        row for row in qrels if (row["query_id"], row["passage_id"]) not in completed
    )
    failures = []
    with tqdm(
        total=len(pairs), initial=len(completed), desc=f"{method} judgments"
    ) as progress:
        if len(completed) < len(pairs):
            async with open_provider(method, config) as provider:
                with (output_dir / "judgments.jsonl").open(
                    "a", encoding="utf-8"
                ) as checkpoint:

                    async def worker():
                        for row in pending:
                            qid, pid = row["query_id"], row["passage_id"]
                            start = time.perf_counter()
                            try:
                                passage = truncate_passage(
                                    query_text[qid], passage_text[pid]
                                )
                                length_retries = 0
                                while True:
                                    try:
                                        answer = await adjudge_with_retry(
                                            provider, query_text[qid], passage
                                        )
                                        break
                                    except ContextLimitError:
                                        if len(passage) <= 1:
                                            raise
                                        passage = passage[: len(passage) // 2]
                                        length_retries += 1
                                result = {
                                    "query_id": qid,
                                    "passage_id": pid,
                                    "human_relevance": int(row["relevance"]),
                                    "passage_original_chars": len(passage_text[pid]),
                                    "passage_used_chars": len(passage),
                                    "passage_truncated": len(passage)
                                    < len(passage_text[pid]),
                                    "length_retries": length_retries,
                                    **answer.checkpoint_fields(),
                                    "latency_seconds": time.perf_counter() - start,
                                }
                                validate_judgments([result])
                                checkpoint.write(
                                    json.dumps(result, allow_nan=False) + "\n"
                                )
                                checkpoint.flush()
                                completed[(qid, pid)] = result
                            except Exception as error:
                                failure = {
                                    "query_id": qid,
                                    "passage_id": pid,
                                    "error": str(error),
                                    "type": type(error).__name__,
                                }
                                if isinstance(error, ProviderResponseError):
                                    failure["details"] = error.details
                                failures.append(failure)
                            progress.update(1)

                    await asyncio.gather(*(worker() for _ in range(concurrency)))
    save_json(output_dir / "failures.json", failures)
    if failures and not allow_partial:
        raise RuntimeError(
            f"{len(failures)} judgments failed; rerun to resume. See failures.json; "
            "use --allow-partial to report successful judgments"
        )
    return successful_judgments(completed, qrels, allow_partial=allow_partial)


@app.command()
def run(
    dataset: Annotated[
        DatasetName,
        typer.Option(
            help="Dataset supplying queries, passages, and human relevance labels."
        ),
    ] = DatasetName.trec_dl_2023,
    method: Annotated[
        Method,
        typer.Option(
            help="Judgment provider: jev, OpenAI Responses (openai), OpenAI Decisions, or OpenRouter Decisions Alpha."
        ),
    ] = Method.jev,
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "Override the provider's model. Defaults to "
                f"{PROVIDERS['jev'].default_model} for jev and "
                f"{PROVIDERS['openai'].default_model} for OpenAI methods, and "
                f"{PROVIDERS['openrouter'].default_model} for OpenRouter."
            )
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(min=1, help="Maximum number of simultaneous judgment requests."),
    ] = 16,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            help="Reasoning effort for --method openai only: "
            + ", ".join(REASONING_EFFORTS)
            + ". Defaults to none."
        ),
    ] = None,
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Base directory for runs/ and <method>/ checkpoints and reports, relative to the working directory."
        ),
    ] = Path("data"),
    local_runs: Annotated[
        bool,
        typer.Option(
            help="Use at least two cached .gz, .txt, .trec, or .run files in <data-dir>/runs/ instead of downloading runs."
        ),
    ] = False,
    allow_partial: Annotated[
        bool,
        typer.Option(
            help="Report successful judgments despite individual failures; requires at least one success and writes incomplete reports to <data-dir>/<method>/partial/."
        ),
    ] = False,
    bootstrap_resamples: Annotated[
        int,
        typer.Option(
            min=0, help="Query bootstrap replicates; zero disables intervals."
        ),
    ] = 10_000,
    bootstrap_confidence: Annotated[
        float,
        typer.Option(help="Bootstrap confidence level, strictly between 0 and 1."),
    ] = 0.95,
    bootstrap_seed: Annotated[
        int,
        typer.Option(min=0, help="Seed for reproducible query bootstrap draws."),
    ] = 0,
    bootstrap_workers: Annotated[
        int,
        typer.Option(
            min=0, help="Bootstrap threads: 0 automatic (up to four), 1 serial."
        ),
    ] = 0,
):
    """Judge passages (resuming checkpoints), then generate reports.

    Assumes API keys already set in environment.

    Successful judgments saved to <data-dir>/<method>/judgments.jsonl.
    Rerunning with same arguments will resume an execution.

    Performance report will also be generated at the end of the run.

    Example: relevance-bench run --method openai --concurrency 8
    """
    try:
        bootstrap = BootstrapConfig(
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
            workers=bootstrap_workers,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    selected = DATASETS[dataset.value]
    name = method.value
    try:
        config = provider_config(
            name,
            model,
            reasoning_effort=reasoning_effort,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    paths = resolve_runs(data_dir, selected, local=local_runs)
    qrels = selected.load("qrels")
    judgments = asyncio.run(
        judge_async(
            selected.load("queries"),
            selected.load("passages"),
            qrels,
            output_dir=data_dir / name,
            config=config,
            concurrency=concurrency,
            method=name,
            dataset=selected,
            allow_partial=allow_partial,
        )
    )
    print_report(
        judgments,
        qrels=qrels,
        method=name,
        dataset=selected,
        paths=paths,
        data_dir=data_dir,
        bootstrap=bootstrap,
    )


@app.command()
def report(
    dataset: Annotated[
        DatasetName,
        typer.Option(
            help="Reference dataset matching the checkpoint's human relevance labels."
        ),
    ] = DatasetName.trec_dl_2023,
    method: Annotated[
        Method,
        typer.Option(help="Method identifying the checkpoint in <data-dir>/<method>/."),
    ] = Method.jev,
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Base directory containing runs/ and <method>/ checkpoints; reports are written here, relative to the working directory."
        ),
    ] = Path("data"),
    local_runs: Annotated[
        bool,
        typer.Option(
            help="Use at least two cached .gz, .txt, .trec, or .run files in <data-dir>/runs/ instead of downloading runs."
        ),
    ] = False,
    allow_partial: Annotated[
        bool,
        typer.Option(
            help="Report an incomplete checkpoint using successful judgments only; requires at least one success and writes to <data-dir>/<method>/partial/."
        ),
    ] = False,
    bootstrap_resamples: Annotated[
        int,
        typer.Option(
            min=0, help="Query bootstrap replicates; zero disables intervals."
        ),
    ] = 10_000,
    bootstrap_confidence: Annotated[
        float,
        typer.Option(help="Bootstrap confidence level, strictly between 0 and 1."),
    ] = 0.95,
    bootstrap_seed: Annotated[
        int,
        typer.Option(min=0, help="Seed for reproducible query bootstrap draws."),
    ] = 0,
    bootstrap_workers: Annotated[
        int,
        typer.Option(
            min=0, help="Bootstrap threads: 0 automatic (up to four), 1 serial."
        ),
    ] = 0,
):
    """Report a checkpoint without judgment API calls or credentials.

    Load saved judgments from <data-dir>/<method>/ and reference labels from
    the hosted dataset.

    JSON and plots go beside the checkpoint, or in partial/ when judgments
    are missing. Complete reports use the checkpoint directory even when
    --allow-partial is set.

    Example: relevance-bench report --method jev --local-runs
    """
    try:
        bootstrap = BootstrapConfig(
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
            workers=bootstrap_workers,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    selected = DATASETS[dataset.value]
    paths = resolve_runs(data_dir, selected, local=local_runs)
    qrels = selected.load("qrels")
    judgments = report_checkpoint(
        data_dir / method.value,
        qrels,
        selected,
        method.value,
        allow_partial=allow_partial,
    )
    print_report(
        judgments,
        qrels=qrels,
        method=method.value,
        dataset=selected,
        paths=paths,
        data_dir=data_dir,
        bootstrap=bootstrap,
    )


@app.command()
def compare(
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing experiment folders with saved reports."
        ),
    ] = Path("data"),
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to <data-dir>/comparison."),
    ] = None,
):
    """Compare saved reports and checkpoints using local files only.

    Print and save two Markdown tables plus four Pareto PNGs. No credentials,
    hosted datasets, providers, or retrieval-run downloads are needed.
    """
    from .reporting.comparison import compare_reports

    try:
        markdown = compare_reports(data_dir, output_dir)
    except (ValueError, OSError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    typer.echo(markdown, nl=False)
    typer.echo(
        f"Comparison saved in {output_dir if output_dir is not None else data_dir / 'comparison'}",
        err=True,
    )


@app.command("download-runs")
def cache_runs(
    dataset: Annotated[
        DatasetName,
        typer.Option(help="Dataset whose reference retrieval runs should be cached."),
    ] = DatasetName.trec_dl_2023,
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Base directory for the runs/ cache, relative to the working directory."
        ),
    ] = Path("data"),
):
    """Cache and validate retrieval runs without loading dataset tables.

    Download missing reference runs into <data-dir>/runs/.

    Example: relevance-bench download-runs --data-dir data
    """
    paths = download_runs(data_dir / "runs", DATASETS[dataset.value])
    typer.echo(f"Cached {len(paths)} runs in {data_dir / 'runs'}")


def main():
    """Dispatch CLI commands and exit with status 1 on validation or run errors."""
    try:
        app()
    except (ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
