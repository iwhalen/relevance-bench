# `relevance-bench`

This is a small benchmark of the TREC DL 2023 judged passages.

Judges are given `(query, passage)` pairs and expected to output a judgment in `{0, 1, 2, 3}`.

## Quick start

First, clone the repository.

```bash
git clone https://github.com/iwhalen/relevance-bench.git
```

Then install [uv](https://docs.astral.sh/uv/).

Set `TYPESAFE_API_KEY` in `.env` for Jev, or `OPENAI_API_KEY` for OpenAI methods.

Then run:

```bash
uv run --env-file .env relevance-bench run
```

By default, this will output results for Jev to `data/jev`.

Methods can be changed with the `--method` flag. Available choices are:

- `jev`: generated judgments with Jev through the TypeSafe API.
- `openai`: generate judgments using structured outputs and the Responses API.
- `openai-decisions`: generate judgments with the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

## CLI docs

See here for more detailed CLI documentation.

The CLI is a Typer application, so using the `--help` flag will always be a good way to get the best information on command.

### `run`

### `report`

### `compare`

### `download-runs`

Compare all saved experiments in a data directory without credentials or network access:

```bash
uv run relevance-bench compare --data-dir data
uv run relevance-bench compare --data-dir data --output-dir comparisons/latest
```

The command discovers experiment folders immediately inside `--data-dir` (default
`data`), including custom folder names and multiple experiments using the same
method. It reads `results.json`, falling back to `partial/results.json`, and skips
folders without reports with a message on stderr. It never generates reports or
downloads datasets or retrieval runs. Reports must share the same dataset,
evaluation settings, and retrieval-run manifest.

Two Markdown tables are printed to stdout and saved as `comparison.md` in
`<data-dir>/comparison` unless `--output-dir` is supplied. Progress and errors go
to stderr. The same folder contains four PNGs, linked from the Markdown:
`pareto_tau_cost.png`, `pareto_tau_latency.png`, `pareto_kappa_cost.png`, and
`pareto_kappa_latency.png`. These compare saved nDCG@10 Kendall tau-b and pooled
Cohen’s kappa against cost and p50 latency. Confidence levels and intervals
are copied from the reports. Pareto frontiers use unrounded point estimates,
minimize cost/latency, maximize agreement, and retain tied experiments.

P50 latency is the median of successful judgments’ `latency_seconds`
in the experiment’s `judgments.jsonl`, including retries within each recorded
duration. Provider `time_taken` is never substituted. Missing or invalid durations
make latency `N/A`; a checkpoint/report judgment-count mismatch requires
regenerating the report. Unavailable values have explanatory notes and are
excluded from affected plots. A plot with no eligible points explains why it is
empty.

Partial rows show completed/expected judgment counts and use triangle markers.
Differing coverage affects comparability, even when experiments have equal
counts but judged different pairs. Partial reports’ costs cover only successful
judgments, and all recorded costs exclude failed attempts. Estimates with missing
usage or pricing are marked incomplete and excluded from cost plots.


## Development

```bash
uv run pytest
```

All reporting code lives in `src/relevance_bench/reporting/`: `reports.py` generates
and displays reports and plots, `comparison.py` compares saved experiments,
`metrics.py` calculates agreement and usage, `bootstrap.py` computes confidence
intervals, and `checkpoints.py` prepares saved checkpoints and summarizes coverage.


## Build the TREC DL 2023 judged passage dataset

> [!Note]
> There's no need to build this dataset unless you want your own copy.
>  
> The package pulls the [hosted version](https://huggingface.co/datasets/iwhalen/trec-dl-2023-judged) rather than rebuilding it.

Authenticate with `hf auth login` or set `HF_TOKEN` in `.env`, then run:

```bash
uv run --env-file .env scripts/make-trec-dl-2023-judged.py owner/repo
```

## Verifying the judged dataset

Run:

```bash
uv run scripts/summarize-trec-dl-2023-judged.py
```

This should output the following information:

```
Number of queries: 82
Number of unique passages: 21,873
Number of judgments: 22,327
Average query length (words): 6.84
Minimum query length (words): 2
Maximum query length (words): 15
Average judged passages per query: 272.28

Relevance grade | Judgments | Average judged passages per query
              0 |    13,866 | 169.10
              1 |     4,372 | 53.32
              2 |     2,259 | 27.55
              3 |     1,830 | 22.32
```

Which can be verified as correct using the original TREC DL 2023 paper: https://arxiv.org/pdf/2507.08890
