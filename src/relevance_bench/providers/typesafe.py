"""Jev relevance scoring through the TypeSafe client."""

import time

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Score,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeBadRequestError,
)

from ..errors import ContextLimitError, RetryableError, retry_hint
from ..rubric import CRITERIA, INSTRUCTIONS
from .base import (
    BaseProvider,
    Judgment,
    ProviderConfig,
    most_probable_grade,
    usage_fields,
)


class TypeSafeProvider(BaseProvider):
    def __init__(self, config: ProviderConfig, client: AsyncTypeSafeClient):
        super().__init__(config)
        self.client = client

    @staticmethod
    def configuration(config: ProviderConfig) -> dict:
        return {"decision": "argmax_probability_lower_grade_on_tie"}

    async def adjudge(self, query: str, text: str) -> Judgment:
        start = time.perf_counter()
        try:
            response = await self.client.system_one(
                state={"query": query, "passage": text},
                questions={
                    "relevance": Score(instructions=INSTRUCTIONS, criteria=CRITERIA)
                },
            )
        except TypeSafeAPIConnectionError as error:
            raise RetryableError(str(error)) from error
        except TypeSafeAPIError as error:
            if isinstance(
                error, TypeSafeBadRequestError
            ) and "max_tokens_exceeded" in str(error):
                raise ContextLimitError(str(error)) from error
            body = error.body if isinstance(error.body, dict) else {}
            nested_error = body.get("error", {})
            code = body.get("code")
            if isinstance(nested_error, dict):
                code = nested_error.get("code", code)
            quota_error = code in {"insufficient_quota", "quota_exceeded"} or any(
                marker in str(error).lower()
                for marker in (
                    "insufficient_quota",
                    "quota_exceeded",
                    "billing",
                )
            )
            if (
                (error.status == 429 and not quota_error)
                or error.status in (408, 409)
                or error.status >= 500
            ):
                raise RetryableError(
                    str(error),
                    rate_limited=error.status == 429,
                    retry_after=retry_hint(error.headers, str(error)),
                ) from error
            raise
        answer = response.scores["relevance"]
        probabilities = dict(answer.probabilities)
        return Judgment(
            judgment=most_probable_grade(probabilities),
            **usage_fields(response),
            time_taken=time.perf_counter() - start,
            score=float(answer.score),
            confidence=float(answer.confidence),
            probabilities=probabilities,
            request_id=response.request_id,
            model=response.model,
        )
