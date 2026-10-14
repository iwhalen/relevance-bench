"""Provider registry, pricing estimates, and managed client construction."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any

from openai import AsyncOpenAI
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .base import BaseProvider, ProviderConfig
from .openai import (
    REASONING_EFFORTS,
    OpenAIDecisionsProvider,
    OpenAIResponsesConfig,
    OpenAIResponsesProvider,
)
from .typesafe import TypeSafeProvider


@dataclass(frozen=True)
class ProviderSpec:
    provider_type: type[BaseProvider]
    config_type: type[ProviderConfig]
    default_model: str
    client_factory: Callable[[ProviderConfig], Any]
    prices: dict[str, tuple[float, float]]
    allowed_models: tuple[str, ...] | None = None
    model_aliases: dict[str, str] = field(default_factory=dict)


PROVIDERS = {
    "jev": ProviderSpec(
        TypeSafeProvider,
        ProviderConfig,
        "jev-latest",
        lambda config: AsyncTypeSafeClient(
            model=config.model_name,
            retry=RetryPolicy(max_retries=0),
        ),
        {"*": (0.042, 0.0)},
        model_aliases={"jev": "jev-latest"},
    ),
    "openai": ProviderSpec(
        OpenAIResponsesProvider,
        OpenAIResponsesConfig,
        "gpt-6-luna",
        lambda config: AsyncOpenAI(max_retries=0),
        {"gpt-6-luna": (0.1, 0.5)},
    ),
    "openai-decisions": ProviderSpec(
        OpenAIDecisionsProvider,
        ProviderConfig,
        "gpt-6-luna",
        lambda config: AsyncOpenAI(max_retries=0),
        {"gpt-6-luna": (0.1, 0.0)},
        ("gpt-6-luna",),
    ),
}
Method = Enum("Method", {key: key for key in PROVIDERS}, type=str)


def token_prices(method: str, model: str) -> dict | None:
    prices = PROVIDERS[method].prices
    rates = prices.get(model, prices.get("*"))
    return {"input": rates[0], "output": rates[1]} if rates is not None else None


def provider_config(
    method: str, model: str | None = None, **settings
) -> ProviderConfig:
    spec = PROVIDERS[method]
    model = model or spec.default_model
    model = spec.model_aliases.get(model, model)
    if spec.allowed_models is not None and model not in spec.allowed_models:
        raise ValueError(f"{method} supports only {', '.join(spec.allowed_models)}")
    settings = {key: value for key, value in settings.items() if value is not None}
    allowed_settings = {item.name for item in fields(spec.config_type)}
    for key, value in settings.items():
        option = "--" + key.replace("_", "-")
        if key not in allowed_settings:
            raise ValueError(f"{option} is not supported with --method {method}")
        if key == "reasoning_effort" and value not in REASONING_EFFORTS:
            raise ValueError(f"{option} must be one of {', '.join(REASONING_EFFORTS)}")
    rates = token_prices(method, model)
    return spec.config_type(
        model_name=model,
        input_token_cost=rates["input"] if rates else None,
        output_token_cost=rates["output"] if rates else None,
        **settings,
    )


@asynccontextmanager
async def open_provider(method: str, config: ProviderConfig):
    spec = PROVIDERS[method]
    async with spec.client_factory(config) as client:
        yield spec.provider_type(config, client)
