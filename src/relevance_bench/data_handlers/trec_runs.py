"""TREC run handlers."""

import gzip
import hashlib
import math
from pathlib import Path
from urllib.request import urlopen

from tqdm import tqdm

from ..checkpoints import save_json
from .datasets import DEFAULT_DATASET, DatasetSpec


def load_run(path: Path, dataset: DatasetSpec = DEFAULT_DATASET) -> dict:
    """Accept zero- or one-based ranks; trec_eval orders by score, not rank."""
    opener = gzip.open if ".gz" in path.suffixes else open
    run = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{number}: expected six TREC columns")
            qid, _, pid, rank, score, _ = fields
            if not dataset.valid_passage_id(pid) or int(rank) < 0:
                raise ValueError(f"{path}:{number}: invalid passage or rank")
            score = float(score)
            documents = run.setdefault(qid, {})
            if not math.isfinite(score) or pid in documents:
                raise ValueError(f"{path}:{number}: duplicate passage or invalid score")
            documents[pid] = score
    if not run:
        raise ValueError(f"Empty run: {path}")
    return run


def download_runs(runs_dir: Path, dataset: DatasetSpec = DEFAULT_DATASET) -> list[Path]:
    """Cache passage runs, validating before atomic replacement."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in tqdm(dataset.run_names, desc="GitHub runs"):
        path = runs_dir / f"{name}.run"
        url = dataset.run_base + name
        if not path.exists():
            temporary = path.with_suffix(".run.partial")
            try:
                with urlopen(url, timeout=120) as response:
                    with temporary.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                load_run(temporary, dataset)
                temporary.replace(path)
            except Exception as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Cannot download {url}. Populate {runs_dir} and use --local-runs."
                ) from error
        load_run(path, dataset)
        manifest.append(
            {
                "file": path.name,
                "url": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    save_json(runs_dir / "manifest.json", manifest)
    return [runs_dir / entry["file"] for entry in manifest]


def resolve_runs(
    data_dir: Path,
    dataset: DatasetSpec,
    *,
    local: bool = False,
) -> list[Path]:
    runs_dir = data_dir / "runs"
    paths = (
        sorted(
            p
            for p in runs_dir.glob("*")
            if p.suffix in {".gz", ".txt", ".trec", ".run"}
        )
        if local
        else download_runs(runs_dir, dataset)
    )
    if len(paths) < 2:
        raise ValueError("At least two passage runs are required")
    for path in paths:
        load_run(path, dataset)
    return paths
