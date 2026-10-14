"""Query bootstrap confidence intervals for saved experiment reports."""

import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from threadpoolctl import threadpool_limits

BATCH_SIZE = 512
PAIR_BLOCK_SIZE = 4096


@dataclass(frozen=True)
class BootstrapConfig:
    """Reporting settings; zero resamples disables all bootstrap output."""

    resamples: int = 10_000
    confidence: float = 0.95
    seed: int = 0
    workers: int = 0

    def __post_init__(self):
        for name in ("resamples", "seed", "workers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Bootstrap {name} must be a nonnegative integer")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 < self.confidence < 1
        ):
            raise ValueError(
                "Bootstrap confidence must be finite and strictly between 0 and 1"
            )


DEFAULT_BOOTSTRAP = BootstrapConfig()


def kappa_from_counts(counts: np.ndarray) -> np.ndarray:
    """Calculate pooled, judgment-weighted kappa for a batch of 4-by-4 counts."""
    counts = np.asarray(counts, dtype=float).reshape(-1, 4, 4)
    total = counts.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.trace(counts, axis1=1, axis2=2) / total
        expected = (counts.sum(axis=1) * counts.sum(axis=2)).sum(axis=1) / total**2
        return np.divide(
            observed - expected,
            1 - expected,
            out=np.full(len(counts), np.nan),
            where=(total > 0) & (expected < 1),
        )


def _pair_blocks(system_count: int, block_size: int):
    """Enumerate unordered system pairs without allocating an O(systems**2) grid."""
    left = np.empty(block_size, dtype=int)
    right = np.empty(block_size, dtype=int)
    used = 0
    for i in range(system_count - 1):
        j = i + 1
        while j < system_count:
            size = min(block_size - used, system_count - j)
            left[used : used + size] = i
            right[used : used + size] = np.arange(j, j + size)
            used += size
            j += size
            if used == block_size:
                yield left, right
                used = 0
    if used:
        yield left[:used], right[:used]


def tau_b(human: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Tie-aware tau-b for batch-by-system means, following SciPy's definition.

    The numerator counts concordant minus discordant pairs. The denominator
    uses the non-tied pairs on each side, excluding pairs tied on both sides.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    """
    human = np.asarray(human, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if human.ndim != 2 or human.shape != predicted.shape:
        raise ValueError("Tau inputs must be matching batch-by-system matrices")
    numerator = np.zeros(len(human), dtype=float)
    human_pairs = np.zeros(len(human), dtype=float)
    predicted_pairs = np.zeros(len(human), dtype=float)
    for left, right in _pair_blocks(human.shape[1], PAIR_BLOCK_SIZE):
        h = np.sign(human[:, left] - human[:, right])
        p = np.sign(predicted[:, left] - predicted[:, right])
        numerator += (h * p).sum(axis=1)
        human_pairs += np.count_nonzero(h, axis=1)
        predicted_pairs += np.count_nonzero(p, axis=1)
    denominator = np.sqrt(human_pairs * predicted_pairs)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(human), np.nan),
        where=denominator > 0,
    )


def query_matrices(rows: list[dict], scores: dict, method: str):
    """Canonically order evaluated queries and fixed systems; reuse saved scores."""
    queries = sorted({row["query_id"] for row in rows})
    systems = sorted(scores)
    indices = {qid: i for i, qid in enumerate(queries)}
    counts = np.zeros((len(queries), 16), dtype=float)
    for row in rows:
        cell = 4 * row["human_relevance"] + row["predicted_relevance"]
        counts[indices[row["query_id"]], cell] += 1
    measures = sorted(scores[systems[0]]["human"]["aggregate"]) if systems else []
    matrices = {
        measure: tuple(
            np.array(
                [
                    [
                        scores[system][source]["per_query"][qid][measure]
                        for system in systems
                    ]
                    for qid in queries
                ],
                dtype=float,
            ).reshape(len(queries), len(systems))
            for source in ("human", method)
        )
        for measure in measures
    }
    return counts, matrices


def resample_batch(multiplicities: np.ndarray, counts: np.ndarray, matrices: dict):
    """Apply the same query draw to pooled agreement and every retrieval measure."""
    weights = np.asarray(multiplicities, dtype=float)
    kappa = kappa_from_counts(weights @ counts)
    query_count = counts.shape[0]
    correlations = {
        measure: tau_b(weights @ human / query_count, weights @ predicted / query_count)
        for measure, (human, predicted) in matrices.items()
    }
    return kappa, correlations


def percentile_interval(values: np.ndarray, confidence: float, *, reason=None) -> dict:
    """Use finite outcomes only; individual replicate values are never reported."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    lower = upper = None
    if reason is None and len(finite) < 2:
        reason = "Fewer than two finite bootstrap replicates"
    if reason is None:
        tail = (1 - confidence) / 2
        lower, upper = map(
            float, np.quantile(finite, [tail, 1 - tail], method="linear")
        )
    return {
        "confidence": confidence,
        "lower": lower,
        "upper": upper,
        "valid_replicates": len(finite),
        "undefined_replicates": len(values) - len(finite),
        "unavailable_reason": reason,
    }


def worker_count(config: BootstrapConfig, batches: int) -> int:
    """Bound automatic concurrency to four CPUs and require four batches."""
    available = os.process_cpu_count() or 1
    requested = config.workers or (min(4, available) if batches >= 4 else 1)
    return min(requested, batches) if batches else 0


def bootstrap_report(
    rows: list[dict], scores: dict, method: str, config: BootstrapConfig
):
    """Return metadata, a kappa interval, and tau intervals (or None if disabled).

    Each batch has an independent seed stream and is collected in index order.
    Only worker-count batches are in flight; BLAS uses one thread for the entire
    computation to avoid nested pools. Undefined outcomes are excluded from CIs.
    """
    if config.resamples == 0:
        return None
    counts, matrices = query_matrices(rows, scores, method)
    query_count = len(counts)
    batches = math.ceil(config.resamples / BATCH_SIZE) if query_count >= 2 else 0
    workers = worker_count(config, batches)
    metadata = {
        "configuration": asdict(config),
        "resampling_unit": "query",
        "method": "percentile",
        "percentile_interpolation": "linear",
        "excludes_undefined": True,
        "query_count": query_count,
        "batch_size": BATCH_SIZE,
        "workers_used": workers,
        "completed_replicates": config.resamples if batches else 0,
    }
    if query_count < 2:
        interval = percentile_interval(
            np.array([]), config.confidence, reason="Fewer than two evaluated queries"
        )
        return metadata, interval, {key: interval.copy() for key in matrices}

    def compute(index):
        size = min(BATCH_SIZE, config.resamples - index * BATCH_SIZE)
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, index]))
        weights = rng.multinomial(
            query_count, np.full(query_count, 1 / query_count), size=size
        )
        return resample_batch(weights, counts, matrices)

    kappa_parts = []
    tau_parts = {key: [] for key in matrices}

    def collect(result):
        kappa, correlations = result
        kappa_parts.append(kappa)
        for key, values in correlations.items():
            tau_parts[key].append(values)

    with threadpool_limits(limits=1, user_api="blas"):
        if workers == 1:
            for index in range(batches):
                collect(compute(index))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = deque(
                    executor.submit(compute, index) for index in range(workers)
                )
                next_index = workers
                while pending:
                    collect(pending.popleft().result())
                    if next_index < batches:
                        pending.append(executor.submit(compute, next_index))
                        next_index += 1
    return (
        metadata,
        percentile_interval(np.concatenate(kappa_parts), config.confidence),
        {
            key: percentile_interval(np.concatenate(parts), config.confidence)
            for key, parts in tau_parts.items()
        },
    )


def interval_annotation(interval: dict | None) -> str:
    """Append a separate plot-title line while preserving coverage annotations."""
    if interval is None:
        return ""
    label = f"{100 * interval['confidence']:.6g}% bootstrap CI"
    if interval["unavailable_reason"]:
        return f"\n{label} unavailable: {interval['unavailable_reason']}"
    return f"\n{label}: [{interval['lower']:.4f}, {interval['upper']:.4f}]"
