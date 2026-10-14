"""Dataset adapters: tables, retrieval runs, and evaluation provenance."""

import re
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repository: str
    run_base: str
    run_names: tuple[str, ...]
    passage_id_pattern: str
    reference_qrels: str
    split: str = "test"

    def load(self, table: str):
        from datasets import load_dataset

        return load_dataset(self.repository, table, split=self.split)

    def valid_passage_id(self, passage_id: str) -> bool:
        return re.fullmatch(self.passage_id_pattern, passage_id) is not None


DATASETS = {
    "trec-dl-2023": DatasetSpec(
        name="trec-dl-2023",
        repository="iwhalen/trec-dl-2023-judged",
        run_base=(
            "https://raw.githubusercontent.com/rahmanidashti/"
            "SyntheticTestCollections/main/dl-2023-runs/"
        ),
        run_names=(
            "agg-cocondenser",
            "bm25_splades",
            "cip_run_1",
            "cip_run_2",
            "cip_run_3",
            "cip_run_4",
            "cip_run_5",
            "cip_run_6",
            "cip_run_7",
            "naverloo-frgpt4",
            "naverloo-rgpt4",
            "naverloo_bm25_RR",
            "naverloo_bm25_splades_RR",
            "naverloo_fs",
            "naverloo_fs_RR",
            "naverloo_fs_RR_duo",
            "slim-pp-0shot-uw",
            "splade_pp_ensemble_distil",
            "splade_pp_self_distil",
            "uogtr_b_grf_e",
            "uogtr_b_grf_e_gb",
            "uogtr_be",
            "uogtr_be_gb",
            "uogtr_dph",
            "uogtr_dph_bo1",
            "uogtr_qr_be",
            "uogtr_qr_be_gb",
            "uogtr_s",
            "uogtr_se",
            "uogtr_se_gb",
            "uot-yahoo_LLMs-blender",
            "uot-yahoo_rankgpt35",
            "uot-yahoo_rankgpt4",
            "WatS-Augmented-BM25",
            "WatS-LLM-Rerank",
        ),
        passage_id_pattern=r"msmarco_passage_\d+(?:_\d+)?",
        reference_qrels="NIST withDupes; includes propagated duplicate labels",
    ),
}
DatasetName = Enum(
    "DatasetName", {key.replace("-", "_"): key for key in DATASETS}, type=str
)
DEFAULT_DATASET = DATASETS["trec-dl-2023"]
