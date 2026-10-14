"""OpenRouter Decisions Alpha scoring; retries belong to the caller."""

import json
import math
import time
from collections.abc import Mapping

import httpx
from openrouter import OpenRouter
from openrouter.errors import NoResponseError, OpenRouterError, ResponseValidationError

from ..errors import (
    ContextLimitError,
    ProviderResponseError,
    RetryableError,
    retry_hint,
)
from ..rubric import CRITERIA, INSTRUCTIONS, truncation_config
from .base import (
    BaseProvider,
    Judgment,
    ProviderConfig,
    most_probable_grade,
    usage_fields,
)

PROBABILITY_SUM_TOLERANCE = 0.02


def _json_safe(value):
    """Keep response diagnostics serializable, including malformed numeric values."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _body(body: str):
    try:
        return json.loads(body)
    except (ValueError, TypeError):
        return body


def _probabilities(raw) -> dict[int, float]:
    if not isinstance(raw, dict) or set(raw) != {"0", "1", "2", "3"}:
        raise ValueError("Expected probabilities keyed by 0 through 3")
    if any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
        for p in raw.values()
    ):
        raise ValueError("Invalid relevance probabilities")
    total = sum(raw.values())
    # Four probabilities rounded to two decimals can differ from one by 0.02.
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE + 1e-12:
        raise ValueError("Relevance probabilities must sum to one within 0.02")
    return {int(grade): probability / total for grade, probability in raw.items()}


class OpenRouterDecisionsProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, client: OpenRouter):
        super().__init__(config)
        self.client = client

    @staticmethod
    def configuration(config: ProviderConfig) -> dict:
        return {
            "api": "openrouter_alpha_decisions",
            "input_format": "query_passage_state_object_v1",
            "question_type": "score",
            "decision": "argmax_probability_lower_grade_on_tie",
            "probability_normalization": {
                "policy": "divide_by_sum",
                "sum_absolute_tolerance": PROBABILITY_SUM_TOLERANCE,
                "floating_point_tolerance": 1e-12,
            },
            "truncation": truncation_config(),
        }

    async def adjudge(self, query: str, text: str) -> Judgment:
        start = time.perf_counter()
        try:
            response = await self.client.alpha.decisions.create_async(
                model=self.config.model_name,
                state={"query": query, "passage": text},
                questions={
                    "relevance": {
                        "type": "score",
                        "instructions": INSTRUCTIONS,
                        "criteria": CRITERIA,
                    }
                },
            )
        except (httpx.TransportError, NoResponseError) as error:
            raise RetryableError(str(error)) from error
        except ResponseValidationError as error:
            raise ProviderResponseError(
                "OpenRouter response validation failed",
                details={
                    "model": self.config.model_name,
                    "status_code": error.status_code,
                    "response": _json_safe(_body(error.body)),
                },
            ) from error
        except OpenRouterError as error:
            message = f"{error}: {error.body}"
            lowered = message.lower()
            if error.status_code == 413 or any(
                marker in lowered
                for marker in (
                    "context_length_exceeded",
                    "context_limit_exceeded",
                    "max_tokens_exceeded",
                    "maximum context length",
                    "context window exceeded",
                )
            ):
                raise ContextLimitError(message) from error
            quota_error = any(
                marker in lowered
                for marker in (
                    "insufficient_quota",
                    "quota_exceeded",
                    "quota exceeded",
                    "billing",
                    "insufficient credits",
                    "insufficient balance",
                    "payment required",
                )
            )
            if not quota_error and (
                error.status_code in (408, 409, 429) or error.status_code >= 500
            ):
                raise RetryableError(
                    message,
                    rate_limited=error.status_code == 429,
                    retry_after=retry_hint(error.headers, message),
                ) from error
            raise

        try:
            if (
                not isinstance(response.answers, dict)
                or "relevance" not in response.answers
            ):
                raise ValueError("OpenRouter decision missing relevance answer")
            answer = response.answers["relevance"]
            if answer.type != "score":
                raise ValueError(
                    f"OpenRouter decision expected relevance score, got {answer.type!r}"
                )
            probabilities = _probabilities(answer.probabilities)
            if type(answer.score) not in (int, float):
                raise ValueError("Invalid relevance score")
            confidence = getattr(answer, "confidence", None)
            if confidence is not None and type(confidence) not in (int, float):
                raise ValueError("Invalid confidence")
            return Judgment(
                judgment=most_probable_grade(probabilities),
                **usage_fields(response),
                time_taken=time.perf_counter() - start,
                probabilities=probabilities,
                score=answer.score,
                confidence=confidence,
                model=response.model,
                request_id=getattr(response, "id", None),
            )
        except (ValueError, TypeError, AttributeError) as error:
            raise ProviderResponseError(
                str(error),
                details={
                    "model": getattr(response, "model", self.config.model_name),
                    "request_id": getattr(response, "id", None),
                    "answers": _json_safe(getattr(response, "answers", None)),
                },
            ) from error
