from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger("lattice")

_DEFAULT_TIMEOUT = 60.0


@runtime_checkable
class JudgeProvider(Protocol):
    """Interface for LLM providers used by the judge scorer."""

    def judge(self, system_prompt: str, user_prompt: str) -> str: ...
    async def ajudge(self, system_prompt: str, user_prompt: str) -> str: ...


def _extract_openai_content(resp: httpx.Response) -> str:
    """Extract the assistant message from an OpenAI-style response."""
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected response from {resp.url}: {resp.text[:500]}"
        ) from exc


def _extract_anthropic_content(resp: httpx.Response) -> str:
    """Extract the text from an Anthropic Messages response."""
    try:
        return resp.json()["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected response from {resp.url}: {resp.text[:500]}"
        ) from exc


class OpenAIJudgeProvider:
    """Calls any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_openai_content(resp)

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_openai_content(resp)


class AnthropicJudgeProvider:
    """Calls the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.anthropic.com/v1"
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 512,
            "temperature": 0.1,
        }

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_anthropic_content(resp)

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_anthropic_content(resp)


def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    api_base: str = "https://api.openai.com/v1",
    timeout: float = _DEFAULT_TIMEOUT,
) -> OpenAIJudgeProvider | AnthropicJudgeProvider:
    """Factory that returns the right provider instance."""
    if provider_name == "openai":
        return OpenAIJudgeProvider(api_key, model, api_base, timeout=timeout)
    if provider_name == "anthropic":
        return AnthropicJudgeProvider(api_key, model, timeout=timeout)
    raise ValueError(
        f"Unknown judge provider '{provider_name}'. Use 'openai' or 'anthropic'."
    )


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_ANTHROPIC_PREFIXES = ("claude-",)
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"


def _route_model(model: str) -> tuple[str, str, str]:
    """Return (provider_type, api_base, env_var_name) for a model name.

    Models containing ``/`` (e.g. ``openai/gpt-4o``, ``google/gemini-2.0-flash``)
    are routed through OpenRouter. Bare model names are matched by prefix to
    their native API.
    """
    if "/" in model:
        return "openai", _OPENROUTER_BASE, _OPENROUTER_ENV_VAR

    if any(model.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai", "https://api.openai.com/v1", "OPENAI_API_KEY"

    if any(model.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"

    return "openai", _OPENROUTER_BASE, _OPENROUTER_ENV_VAR


def resolve_provider(
    model: str,
    api_key: str | None = None,
) -> OpenAIJudgeProvider | AnthropicJudgeProvider:
    """Create a provider by auto-detecting the backend from the model name.

    Routing rules:
        - ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*`` → OpenAI
        - ``claude-*`` → Anthropic
        - Names containing ``/`` → OpenRouter
        - Anything else → OpenRouter

    API keys are read from standard environment variables when *api_key*
    is not provided (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, or
    ``OPENROUTER_API_KEY``).
    """
    provider_name, api_base, env_var = _route_model(model)

    if api_key is None:
        api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(
            f"No API key for model '{model}'. "
            f"Set the {env_var} environment variable or pass api_key=."
        )

    return create_provider(provider_name, api_key, model, api_base)
