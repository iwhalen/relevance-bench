"""Shared relevance rubric and conservative passage truncation."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

REQUEST_CHAR_BUDGET = 120_000
TRUNCATION_MARGIN = 2_000
INSTRUCTIONS = (
    "Rate the passage's relevance to the query using the criteria. "
    "Use only information in the passage."
)
CRITERIA = [
    "Irrelevant: The passage has nothing to do with the query.",
    "Related: The passage seems related to the query but does not answer it.",
    (
        "Highly relevant: The passage has some answer for the query, but the answer "
        "may be a bit unclear, or hidden amongst extraneous information."
    ),
    "Perfectly relevant: The passage is dedicated to the query and contains the exact answer.",
]


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: Literal[0, 1, 2, 3]


def openai_instructions() -> str:
    return (
        INSTRUCTIONS
        + "\n"
        + "\n".join(
            f"[{grade}] {criterion}" for grade, criterion in enumerate(CRITERIA)
        )
    )


def truncate_passage(query: str, passage: str) -> str:
    """Use the same conservative initial passage prefix for every provider."""
    overhead = len(
        json.dumps(
            {
                "query": query,
                "passage": "",
                "instructions": INSTRUCTIONS,
                "criteria": CRITERIA,
            },
            ensure_ascii=False,
        )
    )
    return passage[: max(1, REQUEST_CHAR_BUDGET - overhead - TRUNCATION_MARGIN)]


def truncation_config() -> dict:
    return {"character_budget": REQUEST_CHAR_BUDGET, "margin": TRUNCATION_MARGIN}
