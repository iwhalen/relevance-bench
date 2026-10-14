"""OpenAI Responses and Decisions adapters; retries belong to the caller."""

import json
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, BadRequestError

from ..errors import (
    ContextLimitError,
    ProviderResponseError,
    RetryableError,
    retry_hint,
)
from ..rubric import (
    CRITERIA,
    INSTRUCTIONS,
    RelevanceJudgment,
    openai_instructions,
    truncation_config,
)
from .base import (
    BaseProvider,
    Judgment,
    ProviderConfig,
    most_probable_grade,
    usage_fields,
)

REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class OpenAIResponsesConfig(ProviderConfig):
    reasoning_effort: str = "none"


class OpenAIProvider(BaseProvider):
    """Base class for common OpenAI functionality."""

    def __init__(self, config: ProviderConfig, client: AsyncOpenAI):
        super().__init__(config)
        self.client = client

    async def _request(self, call, **payload):
        try:
            return await call(**payload)
        except APIConnectionError as error:
            raise RetryableError(str(error)) from error
        except APIStatusError as error:
            if (
                isinstance(error, BadRequestError)
                and error.code == "context_length_exceeded"
            ):
                raise ContextLimitError(str(error)) from error
            if (
                (error.status_code == 429 and error.code == "rate_limit_exceeded")
                or error.status_code in (408, 409)
                or error.status_code >= 500
            ):
                raise RetryableError(
                    str(error),
                    rate_limited=error.status_code == 429,
                    retry_after=retry_hint(error.response.headers, str(error)),
                ) from error
            raise


class OpenAIResponsesProvider(OpenAIProvider):
    """Generates outcomes with the responses API."""

    @staticmethod
    def configuration(config: OpenAIResponsesConfig) -> dict:
        return {
            "decision": "structured_grade",
            "reasoning_effort": config.reasoning_effort,
            "prompt": openai_instructions(),
            "input_format": "query_passage_json_v1",
            "output_schema": RelevanceJudgment.model_json_schema(),
            "truncation": truncation_config(),
        }

    async def adjudge(self, query: str, text: str) -> Judgment:
        start = time.perf_counter()
        response = await self._request(
            self.client.responses.parse,
            model=self.config.model_name,
            reasoning={"effort": self.config.reasoning_effort},
            instructions=openai_instructions(),
            input=json.dumps({"query": query, "passage": text}, ensure_ascii=False),
            text_format=RelevanceJudgment,
            store=False,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ValueError(
                "OpenAI judgment incomplete, refused, or missing structured output"
            )
        return Judgment(
            judgment=response.output_parsed.grade,
            **usage_fields(response),
            time_taken=time.perf_counter() - start,
            model=response.model,
            request_id=response._request_id,
        )


class OpenAIDecisionsProvider(OpenAIProvider):
    """Generates outcomes with the decisions API."""

    @staticmethod
    def configuration(config: ProviderConfig) -> dict:
        return {
            "decision": "argmax_probability_lower_grade_on_tie",
            "input_format": "query_passage_json_v1",
            "question_type": "score",
            "truncation": truncation_config(),
        }

    async def adjudge(self, query: str, text: str) -> Judgment:
        start = time.perf_counter()
        response = await self._request(
            self.client.decisions.create,
            model=self.config.model_name,
            input=json.dumps({"query": query, "passage": text}, ensure_ascii=False),
            questions=[
                {
                    "name": "relevance",
                    "type": "score",
                    "instructions": INSTRUCTIONS,
                    "levels": [
                        {
                            "label": criterion.split(":", 1)[0],
                            "description": criterion.split(":", 1)[1].strip(),
                        }
                        for criterion in CRITERIA
                    ],
                }
            ],
        )
        try:
            answers = [a for a in response.answers if a.name == "relevance"]
            if not answers:
                raise ValueError("OpenAI decision missing relevance answer")
            if len(answers) != 1:
                raise ValueError("OpenAI decision returned duplicate relevance answers")
            answer = answers[0]
            if answer.type == "refusal":
                raise ValueError("OpenAI decision refused relevance question")
            if answer.type != "score":
                raise ValueError(
                    f"OpenAI decision expected relevance score, got {answer.type!r}"
                )
            probabilities = {p.value: p.probability for p in answer.probabilities}
            if len(answer.probabilities) != 4:
                raise ValueError("Expected four unique relevance probabilities")
            grade = most_probable_grade(probabilities)
        except ValueError as error:
            raise ProviderResponseError(
                str(error),
                details={
                    "model": response.model,
                    "request_id": getattr(response, "_request_id", None),
                    "answers": [
                        {"name": a.name, "type": a.type} for a in response.answers
                    ],
                },
            ) from error
        return Judgment(
            judgment=grade,
            **usage_fields(response),
            time_taken=time.perf_counter() - start,
            model=response.model,
            request_id=getattr(response, "_request_id", None),
            score=answer.score,
            confidence=answer.confidence,
            probabilities=probabilities,
        )
