from __future__ import annotations

import os

from pydantic import BaseModel


class AgentTraceConfig(BaseModel):
    service_name: str = "agent-trace"
    otel_endpoint: str | None = None
    otel_enabled: bool = True
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    judge_api_key: str | None = None
    judge_api_base: str = "https://api.openai.com/v1"
    judge_max_concurrency: int = 5
    judge_max_retries: int = 3
    judge_retry_base_delay: float = 1.0
    judge_retry_max_delay: float = 30.0


_config: AgentTraceConfig | None = None


def configure(**kwargs) -> AgentTraceConfig:
    """Initialize agent-trace with the given settings."""
    global _config
    kwargs.setdefault("judge_api_key", os.environ.get("OPENAI_API_KEY"))
    _config = AgentTraceConfig(**kwargs)
    if _config.otel_enabled:
        from .otel import setup_tracer

        setup_tracer(_config.service_name, _config.otel_endpoint)
    return _config


def get_config() -> AgentTraceConfig:
    """Return the current config, creating a default one if needed."""
    global _config
    if _config is None:
        _config = AgentTraceConfig(
            judge_api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return _config
