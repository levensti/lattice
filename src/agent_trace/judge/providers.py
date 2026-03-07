from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class JudgeProvider(Protocol):
    """Interface for LLM providers used by the judge scorer."""

    def judge(self, system_prompt: str, user_prompt: str) -> str: ...
    async def ajudge(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIJudgeProvider:
    """Calls any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        *,
        temperature: float = 0.1,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class AnthropicJudgeProvider:
    """Calls the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        *,
        temperature: float = 0.1,
        top_k: int | None = None,
        top_p: float | None = None,
        max_tokens: int = 512,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.anthropic.com/v1"
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict:
        payload: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]


def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    api_base: str = "https://api.openai.com/v1",
    *,
    temperature: float = 0.1,
    top_k: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> OpenAIJudgeProvider | AnthropicJudgeProvider:
    """Factory that returns the right provider instance."""
    if provider_name == "openai":
        return OpenAIJudgeProvider(
            api_key, model, api_base,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    if provider_name == "anthropic":
        return AnthropicJudgeProvider(
            api_key, model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens or 512,
        )
    raise ValueError(
        f"Unknown judge provider '{provider_name}'. Use 'openai' or 'anthropic'."
    )
