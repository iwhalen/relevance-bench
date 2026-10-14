# `relevance-bench`

This is a small benchmark of the TREC DL 2023 judged passages.

Judges are given `(query, passage)` pairs and expected to output a judgment in `{0, 1, 2, 3}`.

## Quick start

First, clone the repository.

```bash
git clone https://github.com/iwhalen/relevance-bench.git
cd relevance-bench
```

Then install [uv](https://docs.astral.sh/uv/).

Set `TYPESAFE_API_KEY` in `.env` for Jev, or `OPENAI_API_KEY` for OpenAI methods.

Then run:

```bash
uv run --env-file .env relevance-bench run
```

By default, this will output results for Jev to `data/jev`.

Methods can be changed with the `--method` flag. Available choices are:

- `jev`: generate judgments with Jev through the TypeSafe API.
- `openai`: generate judgments using structured outputs and the Responses API.
- `openai-decisions`: generate judgments with the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

## CLI docs

Use `--help` on any command to see its options.

### `run`

Generate judgments and reports:

```bash
uv run --env-file .env relevance-bench run --method openai
```

Results are saved to `data/<method>`. Rerun with the same arguments to resume an interrupted run.

### `report`

Regenerate reports from saved judgments. No API key is needed.

```bash
uv run relevance-bench report --method openai
```

Use `--allow-partial` to report an incomplete run. These reports are saved to `data/<method>/partial`.

### `compare`

Compare saved reports without credentials or network access:

```bash
uv run relevance-bench compare --data-dir data
```

Reports must use the same dataset, evaluation settings, and retrieval runs. Tables and plots are saved to `data/comparison`. Use `--output-dir` to change this.

### `download-runs`

Cache reference retrieval runs in `data/runs`:

```bash
uv run relevance-bench download-runs
```

The `run` and `report` commands download these automatically. Use `--local-runs` to use cached runs.

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