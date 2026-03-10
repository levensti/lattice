from __future__ import annotations

import os

from pydantic import BaseModel


class LatticeConfig(BaseModel):
    service_name: str = "lattice"
    otel_endpoint: str | None = None
    otel_enabled: bool = True
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    judge_api_key: str | None = None
    judge_api_base: str = "https://api.openai.com/v1"
    judge_max_concurrency: int = 5
    judge_timeout: float = 60.0


_config: LatticeConfig | None = None


def configure(**kwargs) -> LatticeConfig:
    """Initialize lattice with the given settings.

    When *judge_model* is provided without an explicit *judge_provider*,
    the provider, API base URL, and API key environment variable are
    inferred automatically from the model name:

    - ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*`` → OpenAI (``OPENAI_API_KEY``)
    - ``claude-*`` → Anthropic (``ANTHROPIC_API_KEY``)
    - Names containing ``/`` or unrecognised → OpenRouter (``OPENROUTER_API_KEY``)
    """
    global _config

    model = kwargs.get("judge_model", "gpt-4o")

    if "judge_provider" not in kwargs:
        from .judge.providers import _route_model

        provider_name, api_base, env_var = _route_model(model)
        kwargs.setdefault("judge_provider", provider_name)
        kwargs.setdefault("judge_api_base", api_base)
        kwargs.setdefault("judge_api_key", os.environ.get(env_var))
    else:
        kwargs.setdefault("judge_api_key", os.environ.get("OPENAI_API_KEY"))

    _config = LatticeConfig(**kwargs)
    if _config.otel_enabled and _config.otel_endpoint:
        from .otel import setup_tracer

        setup_tracer(_config.service_name, _config.otel_endpoint)
    return _config


def get_config() -> LatticeConfig:
    """Return the current config, creating a default one if needed."""
    global _config
    if _config is None:
        _config = LatticeConfig(
            judge_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _config
