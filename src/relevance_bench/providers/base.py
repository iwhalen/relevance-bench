"""Typed contract for a single provider request attempt."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Judgment:
    """Represents a single pseudo-relevance judgment."""

    judgment: Literal[0, 1, 2, 3]
    input_tokens: int | None
    output_tokens: int | None
    time_taken: float

    reasoning: str | None = None
    confidence: float | None = None
    probabilities: dict[int, float] | None = None
    score: float | None = None
    model: str | None = None
    request_id: str | None = None

    def __post_init__(self):
        if type(self.judgment) is not int or self.judgment not in range(4):
            raise ValueError("Invalid relevance grade")
        for tokens in (self.input_tokens, self.output_tokens):
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise ValueError("Invalid token usage")
        if not math.isfinite(self.time_taken) or self.time_taken < 0:
            raise ValueError("Invalid request duration")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Invalid confidence")
        if self.score is not None and (
            not math.isfinite(self.score) or not 0 <= self.score <= 3
        ):
            raise ValueError("Invalid relevance score")
        if self.probabilities is not None:
            most_probable_grade(self.probabilities)

    def checkpoint_fields(self) -> dict:
        fields = {
            "predicted_relevance": self.judgment,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "model": self.model,
            "request_id": self.request_id,
            "time_taken": self.time_taken,
        }
        for key in ("reasoning", "confidence", "score"):
            if (value := getattr(self, key)) is not None:
                fields[key] = value
        if self.probabilities is not None:
            fields["probabilities"] = {
                str(grade): probability
                for grade, probability in self.probabilities.items()
            }
        return fields


@dataclass(frozen=True)
class ProviderConfig:
    """Prices are estimates in dollars per million tokens; None means unknown."""

    model_name: str
    input_token_cost: float | None = None
    output_token_cost: float | None = None


class BaseProvider(ABC):
    """Base provider class, defines the async judgment interface."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def adjudge(self, query: str, text: str) -> Judgment:
        """Judge the given text for relevance against the query.

        Args:
            query: the information need.
            text: the given document / passage / information.

        Returns:
            Judgment, the TREC-style judgment of the (query, text) pair.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def configuration(config: ProviderConfig) -> dict:
        """Describe judgment-affecting settings for checkpoint compatibility."""
        raise NotImplementedError


def most_probable_grade(probabilities: dict[int, float]) -> int:
    if set(probabilities) != set(range(4)) or any(
        type(g) is not int for g in probabilities
    ):
        raise ValueError("Expected probabilities for grades 0 through 3")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Invalid relevance probabilities")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Relevance probabilities must sum to one")
    return max(range(4), key=probabilities.get)


def usage_fields(response) -> dict:
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens if usage is not None else None,
        "output_tokens": usage.output_tokens if usage is not None else None,
    }
