import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from conftest import make_provider
from openrouter import OpenRouter
from openrouter.errors import NoResponseError, OpenRouterError, ResponseValidationError

from relevance_bench import __main__ as cli
from relevance_bench.errors import (
    ContextLimitError,
    ProviderResponseError,
    RetryableError,
)
from relevance_bench.providers import open_provider, provider_config
from relevance_bench.providers.base import most_probable_grade
from relevance_bench.providers.openrouter import OpenRouterDecisionsProvider
from relevance_bench.rubric import CRITERIA, INSTRUCTIONS


def provider_for(responses):
    return make_provider(
        "openrouter", provider_config("openrouter"), responses["openrouter"]
    )


@pytest.mark.parametrize(
    "raw, grade",
    [
        ({"3": 0.1, "2": 0.4, "1": 0.4, "0": 0.1}, 1),
        ({"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}, 0),
        ({"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.29}, 2),
        ({"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.31}, 2),
        ({"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.28}, 2),
        ({"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.32}, 2),
    ],
)
def test_rounded_distributions_and_ties(raw, grade, responses):
    responses["openrouter"].answers["relevance"].probabilities = raw
    provider, _ = provider_for(responses)
    result = asyncio.run(provider.adjudge("q", "p"))
    assert result.judgment == grade
    assert most_probable_grade(result.probabilities) == grade
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.probabilities == {
        int(k): pytest.approx(v / sum(raw.values())) for k, v in raw.items()
    }
    if sum(raw.values()) != pytest.approx(1):
        with pytest.raises(ValueError, match="sum to one"):
            most_probable_grade({int(k): v for k, v in raw.items()})


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"0": 1},
        {0: 0.1, 1: 0.4, 2: 0.4, 3: 0.1},
        {"00": 0.1, "1": 0.4, "2": 0.4, "3": 0.1},
        {"0": 0.1, "1": 0.4, "2": 0.4, "3": 0.1, "4": 0},
        {"0": 0, "1": 0, "2": 0, "3": 0},
        *[
            {"0": p, "1": 0.4, "2": 0.4, "3": 0.1}
            for p in (
                -0.1,
                1.1,
                float("nan"),
                float("inf"),
                "0.1",
                True,
                None,
                0.07,
                0.13,
            )
        ],
    ],
)
def test_invalid_probabilities_are_terminal_and_json_safe(raw, responses):
    responses["openrouter"].answers["relevance"].probabilities = raw
    provider, request = provider_for(responses)
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert request.await_count == 1
    assert caught.value.details["request_id"] == "openrouter-request"
    json.dumps(caught.value.details, allow_nan=False)


@pytest.mark.parametrize(
    "kind", ["missing", "other", "choice", "refusal", "no_type", "no_probabilities"]
)
def test_missing_or_unexpected_answer(kind, responses):
    response = responses["openrouter"]
    if kind == "missing":
        response.answers = {}
    elif kind == "other":
        response.answers = {"other": response.answers["relevance"]}
    elif kind == "no_type":
        del response.answers["relevance"].type
    elif kind == "no_probabilities":
        del response.answers["relevance"].probabilities
    else:
        response.answers["relevance"].type = kind
    provider, request = provider_for(responses)
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    assert request.await_count == 1
    json.dumps(caught.value.details, allow_nan=False)


@pytest.mark.parametrize(
    "field, value",
    [
        ("score", -1),
        ("score", 4),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", None),
        ("score", "2"),
        ("score", True),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", "0.5"),
        ("confidence", True),
    ],
)
def test_invalid_score_and_confidence(field, value, responses):
    setattr(responses["openrouter"].answers["relevance"], field, value)
    provider, request = provider_for(responses)
    with pytest.raises(ProviderResponseError) as caught:
        asyncio.run(cli.adjudge_with_retry(provider, "q", "p"))
    json.dumps(caught.value.details, allow_nan=False)
    assert request.await_count == 1


def test_optional_metadata(responses):
    response = responses["openrouter"]
    del response.answers["relevance"].confidence
    del response.id
    response.usage = None
    provider, _ = provider_for(responses)
    result = asyncio.run(provider.adjudge("q", "p"))
    assert (
        result.confidence
        is result.request_id
        is result.input_tokens
        is result.output_tokens
        is None
    )


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (408, "timeout", RetryableError),
        (409, "conflict", RetryableError),
        (429, "rate limit", RetryableError),
        (500, "server", RetryableError),
        (502, "gateway", RetryableError),
        (503, "unavailable", RetryableError),
        (524, "timeout", RetryableError),
        (529, "overloaded", RetryableError),
        (413, "too large", ContextLimitError),
        (400, {"error": {"code": "context_length_exceeded"}}, ContextLimitError),
        (
            400,
            {"error": {"metadata": {"raw": "maximum context length is 2048"}}},
            ContextLimitError,
        ),
        (400, "max_tokens_exceeded", ContextLimitError),
        (400, "invalid request", OpenRouterError),
        (422, "validation", OpenRouterError),
        (401, "auth", OpenRouterError),
        (402, "billing", OpenRouterError),
        (403, "forbidden", OpenRouterError),
        (404, "model missing", OpenRouterError),
        (429, {"error": {"code": "insufficient_quota"}}, OpenRouterError),
        (429, {"error": {"code": "quota_exceeded"}}, OpenRouterError),
        (429, "Insufficient credits", OpenRouterError),
        (429, "billing limit", OpenRouterError),
        (500, "billing problem", OpenRouterError),
    ],
)
def test_http_error_translation(status, body, expected, responses):
    provider, call = provider_for(responses)
    error = OpenRouterError(
        "failure", httpx.Response(status, json=body, headers={"retry-after-ms": "1500"})
    )
    call.side_effect = error
    with pytest.raises(expected) as caught:
        asyncio.run(provider.adjudge("q", "p"))
    assert call.await_count == 1
    if expected == OpenRouterError:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error
    if expected == RetryableError:
        assert caught.value.retry_after == 1.5
        assert caught.value.rate_limited is (status == 429)


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("connection"), httpx.ReadTimeout("timeout"), NoResponseError()],
)
def test_transport_error_translation(error, responses):
    provider, call = provider_for(responses)
    call.side_effect = error
    with pytest.raises(RetryableError) as caught:
        asyncio.run(provider.adjudge("q", "p"))
    assert caught.value.__cause__ is error


@pytest.fixture
def sdk_response():
    return {
        "id": "sdk-request",
        "model": "microsoft/microsoft-decision-1",
        "answers": {
            "relevance": {
                "type": "score",
                "score": 2.1,
                "confidence": 0.9,
                "probabilities": {"3": 0.31, "0": 0.1, "2": 0.4, "1": 0.2},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


@pytest.mark.parametrize(
    "kind",
    ["success", "server", "quota", "context", "schema", "unknown", "malformed_json"],
)
def test_real_sdk_serialization_parsing_and_no_retries(kind, sdk_response, monkeypatch):
    from relevance_bench import providers

    captured = []

    def transport(request):
        captured.append(request)
        if kind in ("server", "quota", "context"):
            status = {"server": 503, "quota": 429, "context": 413}[kind]
            return httpx.Response(
                status,
                json={
                    "error": {
                        "code": status,
                        "message": "insufficient_quota"
                        if kind == "quota"
                        else "failure",
                    }
                },
                headers={"retry-after": "7"},
            )
        if kind == "schema":
            sdk_response["usage"]["input_tokens"] = {"bad": "usage"}
        elif kind == "unknown":
            sdk_response["answers"]["relevance"]["type"] = "future-answer"
        elif kind == "malformed_json":
            return httpx.Response(
                200, headers={"content-type": "application/json"}, content="{"
            )
        return httpx.Response(200, json=sdk_response)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
            monkeypatch.setattr(
                providers,
                "OpenRouter",
                lambda **kwargs: OpenRouter(**kwargs, async_client=http),
            )
            async with open_provider(
                "openrouter", provider_config("openrouter")
            ) as provider:
                if kind == "success":
                    result = await provider.adjudge("query", "passage")
                    assert (
                        result.judgment == 2
                        and result.score == 2.1
                        and result.confidence == 0.9
                    )
                    assert (
                        result.request_id == "sdk-request"
                        and result.model == sdk_response["model"]
                    )
                    assert result.input_tokens == 10 and result.output_tokens == 0
                    assert result.probabilities[3] == pytest.approx(0.31 / 1.01)
                else:
                    expected = {
                        "server": RetryableError,
                        "quota": OpenRouterError,
                        "context": ContextLimitError,
                    }.get(kind, ProviderResponseError)
                    with pytest.raises(expected) as caught:
                        await provider.adjudge("query", "passage")
                    if expected == ProviderResponseError:
                        json.dumps(caught.value.details, allow_nan=False)
                        if kind in ("schema", "malformed_json"):
                            assert isinstance(
                                caught.value.__cause__, ResponseValidationError
                            )
                    if kind == "server":
                        assert caught.value.retry_after == 7

    asyncio.run(scenario())
    assert len(captured) == 1
    request = captured[0]
    assert request.url.path == "/api/alpha/decisions" and request.method == "POST"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "microsoft/microsoft-decision-1",
        "state": {"query": "query", "passage": "passage"},
        "questions": {
            "relevance": {
                "type": "score",
                "criteria": CRITERIA,
                "instructions": INSTRUCTIONS,
            }
        },
    }


def test_shared_retry_honors_openrouter_hint(monkeypatch, responses):
    sleep = AsyncMock()
    retrying = cli.AsyncRetrying
    monkeypatch.setattr(
        cli, "AsyncRetrying", lambda **kwargs: retrying(sleep=sleep, **kwargs)
    )
    provider, request = provider_for(responses)
    request.side_effect = [
        OpenRouterError(
            "rate limit",
            httpx.Response(
                429, json={"error": "rate limit"}, headers={"retry-after": "7"}
            ),
        ),
        responses["openrouter"],
    ]
    assert asyncio.run(cli.adjudge_with_retry(provider, "q", "p")).judgment == 1
    assert request.await_count == 2
    sleep.assert_awaited_once_with(7)


def test_credentials_only_required_when_constructing_client():
    config = provider_config("openrouter")
    assert (
        OpenRouterDecisionsProvider.configuration(config)["api"]
        == "openrouter_alpha_decisions"
    )

    async def scenario():
        with pytest.raises(KeyError, match="OPENROUTER_API_KEY"):
            async with open_provider("openrouter", config):
                pytest.fail("Missing key must prevent construction")

    asyncio.run(scenario())
