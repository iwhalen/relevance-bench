from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relevance_bench import __main__ as cli
from relevance_bench.providers.openai import (
    OpenAIDecisionsProvider,
    OpenAIResponsesProvider,
)
from relevance_bench.providers.openrouter import OpenRouterDecisionsProvider
from relevance_bench.providers.typesafe import TypeSafeProvider
from relevance_bench.rubric import RelevanceJudgment


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tables():
    return {
        "queries": [{"query_id": "q1", "text": "query"}],
        "passages": [
            {"passage_id": "msmarco_passage_1", "text": "long passage"},
            {"passage_id": "msmarco_passage_2", "text": "short"},
        ],
        "qrels": [
            {"query_id": "q1", "passage_id": "msmarco_passage_1", "relevance": 3},
            {"query_id": "q1", "passage_id": "msmarco_passage_2", "relevance": 0},
        ],
    }


@pytest.fixture
def run_paths(tmp_path):
    directory = tmp_path / "runs"
    directory.mkdir()
    paths = [directory / "good.run", directory / "bad.run"]
    paths[0].write_text(
        "q1 Q0 msmarco_passage_1 0 2 good\nq1 Q0 msmarco_passage_2 1 1 good\n"
    )
    paths[1].write_text(
        "q1 Q0 msmarco_passage_2 1 2 bad\nq1 Q0 msmarco_passage_1 2 1 bad\n"
    )
    return paths


@pytest.fixture
def responses():
    usage = SimpleNamespace(input_tokens=100, output_tokens=0)
    probabilities = {0: 0.1, 1: 0.4, 2: 0.4, 3: 0.1}
    score = SimpleNamespace(score=1.5, confidence=0.8, probabilities=probabilities)
    decision_score = SimpleNamespace(
        name="relevance",
        type="score",
        score=1.5,
        confidence=0.8,
        probabilities=[
            SimpleNamespace(value=k, probability=v) for k, v in probabilities.items()
        ],
    )
    return {
        "jev": SimpleNamespace(
            scores={"relevance": score},
            usage=usage,
            model="jev-latest",
            request_id="jev-request",
        ),
        "openai": SimpleNamespace(
            status="completed",
            output_parsed=RelevanceJudgment(grade=1),
            usage=usage,
            model="gpt-6-luna",
            _request_id="responses-request",
        ),
        "openai-decisions": SimpleNamespace(
            answers=[decision_score],
            usage=usage,
            model="gpt-6-luna",
            _request_id="decisions-request",
        ),
        "openrouter": SimpleNamespace(
            answers={
                "relevance": SimpleNamespace(
                    type="score",
                    score=1.5,
                    confidence=0.8,
                    probabilities={str(k): v for k, v in probabilities.items()},
                )
            },
            usage=usage,
            model="microsoft/microsoft-decision-1",
            id="openrouter-request",
        ),
    }


def make_provider(method, config, response):
    request = AsyncMock(return_value=response)
    if method == "jev":
        provider = TypeSafeProvider(config, SimpleNamespace(system_one=request))
    elif method == "openai":
        provider = OpenAIResponsesProvider(
            config, SimpleNamespace(responses=SimpleNamespace(parse=request))
        )
    elif method == "openrouter":
        provider = OpenRouterDecisionsProvider(
            config,
            SimpleNamespace(
                alpha=SimpleNamespace(decisions=SimpleNamespace(create_async=request))
            ),
        )
    else:
        provider = OpenAIDecisionsProvider(
            config, SimpleNamespace(decisions=SimpleNamespace(create=request))
        )
    return provider, request


@pytest.fixture
def fake_providers(monkeypatch, responses):
    made = []

    @asynccontextmanager
    async def factory(method, config):
        provider, request = make_provider(method, config, responses[method])
        made.append((provider, request))
        yield provider

    monkeypatch.setattr(cli, "open_provider", factory)
    return made
