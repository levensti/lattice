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
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")

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
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.anthropic.com/v1"

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
) -> OpenAIJudgeProvider | AnthropicJudgeProvider:
    """Factory that returns the right provider instance."""
    if provider_name == "openai":
        return OpenAIJudgeProvider(api_key, model, api_base)
    if provider_name == "anthropic":
        return AnthropicJudgeProvider(api_key, model)
    raise ValueError(
        f"Unknown judge provider '{provider_name}'. Use 'openai' or 'anthropic'."
    )
