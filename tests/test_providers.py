import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from conftest import make_provider
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeBadRequestError,
    TypeSafeRateLimitError,
)

from relevance_bench import __main__ as cli
from relevance_bench.errors import (
    ContextLimitError,
    ProviderResponseError,
    RetryableError,
    retry_hint,
)
from relevance_bench.providers import PROVIDERS, provider_config, token_prices
from relevance_bench.providers.base import most_probable_grade
from relevance_bench.providers.openai import OpenAIDecisionsProvider
from relevance_bench.rubric import CRITERIA, INSTRUCTIONS, RelevanceJudgment


@pytest.mark.parametrize("method", list(PROVIDERS))
def test_provider_payload_and_result(method, responses):
    provider, request = make_provider(
        method, provider_config(method), responses[method]
    )
    judgment = asyncio.run(provider.adjudge("query", "passage"))
    assert judgment.judgment == 1
    assert judgment.input_tokens == 100 and judgment.output_tokens == 0
    assert judgment.time_taken >= 0 and judgment.request_id is not None
    payload = request.call_args.kwargs
    if method == "jev":
        assert payload["state"] == {"query": "query", "passage": "passage"}
        assert payload["questions"]["relevance"].criteria == CRITERIA
    elif method == "openrouter":
        assert payload == {
            "model": "microsoft/microsoft-decision-1",
            "state": {"query": "query", "passage": "passage"},
            "questions": {
                "relevance": {
                    "type": "score",
                    "instructions": INSTRUCTIONS,
                    "criteria": CRITERIA,
                }
            },
        }
        assert judgment.model == payload["model"]
        assert judgment.request_id == "openrouter-request"
        assert judgment.score == 1.5 and judgment.confidence == 0.8
        assert judgment.checkpoint_fields()["probabilities"]["1"] == 0.4
    else:
        assert payload["model"] == "gpt-6-luna"
        assert payload["input"] == '{"query": "query", "passage": "passage"}'
        if method == "openai":
            assert payload["store"] is False
            assert payload["text_format"] is RelevanceJudgment
            assert payload["reasoning"] == {"effort": "none"}
        else:
            question = payload["questions"][0]
            assert question["name"] == "relevance" and question["type"] == "score"
            assert question["instructions"] == INSTRUCTIONS
            assert [p["label"] for p in question["levels"]] == [
                c.split(":", 1)[0] for c in CRITERIA
            ]
            assert judgment.score == 1.5 and judgment.confidence == 0.8
            assert judgment.checkpoint_fields()["probabilities"]["1"] == 0.4


@pytest.mark.parametrize("method", list(PROVIDERS))
def test_missing_usage_is_unknown(method, responses):
    responses[method].usage = None
    provider, _ = make_provider(method, provider_config(method), responses[method])
    result = asyncio.run(provider.adjudge("q", "p"))
    assert result.input_tokens is None and result.output_tokens is None


@pytest.mark.parametrize("method", ["openai", "openai-decisions"])
def test_refusal_is_terminal(method, responses):
    if method == "openai":
        responses[method].output_parsed = None
    else:
        responses[method].answers = [SimpleNamespace(name="relevance", type="refusal")]
    provider, request = make_provider(
        method, provider_config(method), responses[method]
    )
    with pytest.raises(ValueError, match="refused"):
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert request.await_count == 1


@pytest.mark.parametrize(
    "answers, message",
    [
        ([("relevance", "refusal")], "refused relevance question"),
        ([], "missing relevance answer"),
        ([("other", "score")], "missing relevance answer"),
        ([(None, "refusal")], "missing relevance answer"),
        (
            [("relevance", "score"), ("relevance", "score")],
            "duplicate relevance answers",
        ),
        ([("relevance", "predicate")], "expected relevance score, got 'predicate'"),
    ],
)
@pytest.mark.parametrize("request_id", ["decisions-request", None])
def test_decisions_failure_diagnostics(answers, message, request_id, responses):
    response = responses["openai-decisions"]
    response.answers = [SimpleNamespace(name=name, type=kind) for name, kind in answers]
    if request_id is None:
        del response._request_id
    provider, request = make_provider(
        "openai-decisions", provider_config("openai-decisions"), response
    )
    with pytest.raises(ProviderResponseError, match=message) as caught:
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert caught.value.details == {
        "model": "gpt-6-luna",
        "request_id": request_id,
        "answers": [{"name": name, "type": kind} for name, kind in answers],
    }
    assert request.await_count == 1


@pytest.mark.parametrize(
    "probabilities",
    [
        {0: 0.5, 1: 0.5},
        {0: -0.1, 1: 0.5, 2: 0.5, 3: 0.1},
        {0: float("nan"), 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 0, 3: 0},
    ],
)
def test_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        most_probable_grade(probabilities)


def test_decisions_maps_values_not_array_order(responses):
    responses["openai-decisions"].answers[0].probabilities.reverse()
    provider, _ = make_provider(
        "openai-decisions",
        provider_config("openai-decisions"),
        responses["openai-decisions"],
    )
    assert asyncio.run(provider.adjudge("q", "p")).judgment == 1


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("connection", RetryableError),
        ("server", RetryableError),
        ("rate_limit", RetryableError),
        ("context", ContextLimitError),
        ("quota", RateLimitError),
    ],
)
def test_openai_error_translation(kind, expected, responses):
    request = httpx2.Request("POST", "https://api.openai.com/v1/decisions")
    if kind == "connection":
        error = APIConnectionError(request=request)
    else:
        status = 400 if kind == "context" else 500 if kind == "server" else 429
        response = httpx2.Response(
            status, request=request, headers={"retry-after-ms": "1500"}
        )
        code = {
            "rate_limit": "rate_limit_exceeded",
            "quota": "insufficient_quota",
            "context": "context_length_exceeded",
            "server": "server_error",
        }[kind]
        cls = (
            BadRequestError
            if status == 400
            else InternalServerError
            if status == 500
            else RateLimitError
        )
        error = cls("failure", response=response, body={"code": code})
    provider, call = make_provider(
        "openai-decisions",
        provider_config("openai-decisions"),
        responses["openai-decisions"],
    )
    call.side_effect = error
    with pytest.raises(expected) as caught:
        asyncio.run(provider.adjudge("q", "p"))
    if expected == RetryableError:
        assert caught.value.rate_limited is (kind == "rate_limit")
    if expected in (RetryableError, ContextLimitError):
        assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("context", ContextLimitError),
        ("connection", RetryableError),
        ("rate_limit", RetryableError),
        ("quota", TypeSafeRateLimitError),
    ],
)
def test_jev_error_translation(kind, expected, responses):
    if kind == "connection":
        error = TypeSafeAPIConnectionError("connection")
    elif kind == "context":
        error = TypeSafeBadRequestError(
            400, {"message": "max_tokens_exceeded"}, httpx2.Headers()
        )
    else:
        code = "insufficient_quota" if kind == "quota" else "rate_limit_exceeded"
        error = TypeSafeRateLimitError(
            429, {"message": code}, httpx2.Headers({"retry-after": "2"})
        )
    provider, call = make_provider("jev", provider_config("jev"), responses["jev"])
    call.side_effect = error
    with pytest.raises(expected):
        asyncio.run(provider.adjudge("q", "p"))


@pytest.mark.parametrize("rate_limited, attempts", [(False, 5), (True, 9)])
def test_retry_limits(monkeypatch, rate_limited, attempts):
    monkeypatch.setattr(cli, "retry_wait", lambda state: 0)
    provider = SimpleNamespace(
        adjudge=AsyncMock(
            side_effect=RetryableError("retry", rate_limited=rate_limited)
        ),
    )
    with pytest.raises(RetryableError):
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert provider.adjudge.await_count == attempts


def test_retry_success(monkeypatch, responses):
    monkeypatch.setattr(cli, "retry_wait", lambda state: 0)
    provider, request = make_provider(
        "openai", provider_config("openai"), responses["openai"]
    )
    request.side_effect = [
        APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com")),
        responses["openai"],
    ]
    assert asyncio.run(cli.adjudge_with_retry(provider, "q", "p")).judgment == 1
    assert request.await_count == 2


def test_server_hints_and_backoff():
    assert retry_hint({"retry-after-ms": "1500"}, "") == 1.5
    assert retry_hint({"retry-after": "2"}, "try again in 3000ms") == 3
    assert retry_hint({"retry-after": "nan"}, "") is None
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 28 < retry_hint({"retry-after": future}, "") <= 30
    state = SimpleNamespace(
        attempt_number=3,
        outcome=SimpleNamespace(exception=lambda: RetryableError("e", retry_after=90)),
    )
    assert cli.retry_wait(state) == 90


@pytest.mark.parametrize("method", ["openai", "openai-decisions"])
def test_tenacity_honors_rate_limit_delay(monkeypatch, responses, method):
    sleep = AsyncMock()
    retrying = cli.AsyncRetrying
    monkeypatch.setattr(
        cli, "AsyncRetrying", lambda **kwargs: retrying(sleep=sleep, **kwargs)
    )
    monkeypatch.setattr(cli.random, "uniform", lambda *args: 0)
    provider, request = make_provider(
        method, provider_config(method), responses[method]
    )
    request.side_effect = [
        RateLimitError(
            "Try again later",
            response=httpx2.Response(
                429,
                request=httpx2.Request("POST", "https://api.openai.com"),
                headers={"retry-after": "7"},
            ),
            body={"code": "rate_limit_exceeded"},
        ),
        responses[method],
    ]
    assert asyncio.run(cli.adjudge_with_retry(provider, "q", "p")).judgment == 1
    assert request.await_count == 2
    sleep.assert_awaited_once_with(7)


def test_prices_and_sdk_retries_disabled(monkeypatch):
    from relevance_bench import providers

    captured = []
    monkeypatch.setattr(
        providers, "AsyncOpenAI", lambda **kwargs: captured.append(kwargs)
    )
    monkeypatch.setattr(
        providers, "AsyncTypeSafeClient", lambda **kwargs: captured.append(kwargs)
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        providers, "OpenRouter", lambda **kwargs: captured.append(kwargs)
    )
    for method in PROVIDERS:
        PROVIDERS[method].client_factory(provider_config(method))
    assert captured[0]["retry"].max_retries == 0
    assert captured[1]["max_retries"] == captured[2]["max_retries"] == 0
    assert captured[3] == {"api_key": "test-key", "retry_config": None}
    assert token_prices("openrouter", "microsoft/microsoft-decision-1") == {
        "input": 0.042,
        "output": 0.0,
    }
    assert token_prices("openrouter", "unknown") is None
    assert provider_config("openrouter", "unknown").model_name == "unknown"
    assert token_prices("openai", "gpt-6-luna")["output"] == 0.5
    assert token_prices("openai-decisions", "gpt-6-luna")["output"] == 0
    assert token_prices("openai", "unknown") is None
    with pytest.raises(ValueError, match="supports only"):
        provider_config("openai-decisions", "unknown")


def test_decisions_through_real_sdk_without_network():
    captured = []

    def transport(request):
        import json

        captured.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"x-request-id": "sdk-request"},
            json={
                "model": "gpt-6-luna",
                "answers": [
                    {
                        "name": "relevance",
                        "type": "score",
                        "score": 2.0,
                        "confidence": 0.9,
                        "probabilities": [
                            {
                                "value": grade,
                                "label": str(grade),
                                "probability": float(grade == 2),
                            }
                            for grade in range(4)
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "total_tokens": 10,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    async def scenario():
        async with AsyncOpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(transport)),
        ) as client:
            provider = OpenAIDecisionsProvider(
                provider_config("openai-decisions"), client
            )
            judgment = await provider.adjudge("query", "passage")
            assert judgment.judgment == 2 and judgment.request_id == "sdk-request"
            assert judgment.input_tokens == 10 and judgment.output_tokens == 0

    asyncio.run(scenario())
    assert captured[0]["questions"][0]["type"] == "score"


def test_nested_jev_quota_error_is_terminal(responses):
    provider, call = make_provider("jev", provider_config("jev"), responses["jev"])
    call.side_effect = TypeSafeRateLimitError(
        429,
        {"error": {"code": "insufficient_quota", "message": "No remaining credits"}},
        httpx2.Headers(),
    )
    with pytest.raises(TypeSafeRateLimitError):
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert call.await_count == 1
