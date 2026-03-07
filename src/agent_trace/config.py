from __future__ import annotations

import os

from pydantic import BaseModel


class JudgeConfig(BaseModel):
    """Inference settings for the LLM judge.

    All fields are optional.  When ``None`` a value is inherited from the
    next layer up (per-step → per-score-call → global ``configure()``).
    """

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None


def _merge_judge_configs(*configs: JudgeConfig | None) -> JudgeConfig:
    """Merge configs with earlier entries taking priority (first non-None wins)."""
    merged: dict = {}
    for cfg in configs:
        if cfg is None:
            continue
        for field_name in JudgeConfig.model_fields:
            if field_name not in merged:
                val = getattr(cfg, field_name)
                if val is not None:
                    merged[field_name] = val
    return JudgeConfig(**merged)


class AgentTraceConfig(BaseModel):
    service_name: str = "agent-trace"
    otel_endpoint: str | None = None
    otel_enabled: bool = True
    judge_provider: str = "openai"
    judge_model: str = "gpt-4o"
    judge_api_key: str | None = None
    judge_api_base: str = "https://api.openai.com/v1"
    judge_max_concurrency: int = 5
    judge_temperature: float = 0.1
    judge_top_k: int | None = None
    judge_top_p: float | None = None
    judge_max_tokens: int | None = None

    def to_judge_config(self) -> JudgeConfig:
        """Convert the global judge settings to a ``JudgeConfig``."""
        return JudgeConfig(
            provider=self.judge_provider,
            model=self.judge_model,
            api_key=self.judge_api_key,
            api_base=self.judge_api_base,
            temperature=self.judge_temperature,
            top_k=self.judge_top_k,
            top_p=self.judge_top_p,
            max_tokens=self.judge_max_tokens,
        )


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
