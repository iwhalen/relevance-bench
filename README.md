# `relevance-bench`

This is a small benchmark of the TREC DL 2023 judged passages.

Judges are given `(query, passage)` pairs and expected to output a judgment in `{0, 1, 2, 3}`.

## Results

### Agreement

| Provider | Model | Judgment coverage | nDCG@10 tau-b | Tau CI (95%) | Pooled Cohen’s kappa | Kappa CI (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| jev | jev-latest | 22327/22327 | 0.9352 | [0.8545, 0.9453] | 0.1907 | [0.1539, 0.2279] |
| openrouter | microsoft/microsoft-decision-1 | partial (22256/22327) | 0.9429 | [0.8780, 0.9529] | **0.2696** | [0.2275, 0.3106] |
| openai | gpt-6-luna | 22327/22327 | 0.9328 | [0.8655, 0.9496] | 0.2391 | [0.2000, 0.2767] |
| openai-decisions | gpt-6-luna | partial (22324/22327) | 0.9294 | [0.8622, 0.9462] | 0.1775 | [0.1427, 0.2124] |
| openrouter | perplexity/pplx-decider-v1.1-27b | partial (22320/22327) | **0.9453** | [0.8780, 0.9554] | 0.2419 | [0.2015, 0.2825] |

### Efficiency

| Provider | Model | Judgment coverage | Estimated total cost (USD) | P50 latency (seconds) |
| --- | --- | --- | --- | --- |
| jev | jev-latest | 22327/22327 | 0.678263 | **0.1357** |
| openrouter | microsoft/microsoft-decision-1 | partial (22256/22327) | 0.338348 | 0.2578 |
| openai | gpt-6-luna | 22327/22327 | 1.328348 | 1.7142 |
| openai-decisions | gpt-6-luna | partial (22324/22327) | 1.236627 | 0.1578 |
| openrouter | perplexity/pplx-decider-v1.1-27b | partial (22320/22327) | **0.246585** | 0.2602 |


## Quick start

First, clone the repository.

```bash
git clone https://github.com/iwhalen/relevance-bench.git
cd relevance-bench
```

Then install [uv](https://docs.astral.sh/uv/).

Set `TYPESAFE_API_KEY` in `.env` for Jev, `OPENAI_API_KEY` for OpenAI methods, or `OPENROUTER_API_KEY` for OpenRouter.

Then run:

```bash
uv run --env-file .env relevance-bench run
```

By default, this will output results for Jev to `data/jev`.

Methods can be changed with the `--method` flag. Available choices are:

- `jev`: generate judgments with Jev through the TypeSafe API.
- `openai`: generate judgments using structured outputs and the Responses API.
- `openai-decisions`: generate judgments with the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).
- `openrouter`: generate judgments with the [OpenRouter Decisions Alpha API](https://openrouter.ai/docs/client-sdks/python/sdks/decisions/README).

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
