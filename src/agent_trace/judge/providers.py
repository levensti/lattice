from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import httpx


@runtime_checkable
class JudgeProvider(Protocol):
    """Interface for LLM providers used by the judge scorer."""

    def judge(self, system_prompt: str, user_prompt: str) -> str: ...
    async def ajudge(self, system_prompt: str, user_prompt: str) -> str: ...
    def close(self) -> None: ...
    async def aclose(self) -> None: ...


class BaseJudgeProvider(ABC):
    """Shared HTTP-client lifecycle and request logic for judge providers.

    Subclasses define the API-specific details: headers, payload shape, URL
    path, and how to extract the text from the JSON response.
    """

    api_base: str

    def __init__(self) -> None:
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # -- alias kept for backward compatibility with tests that use _client --

    @property
    def _client(self) -> httpx.Client | None:
        return self._sync_client

    @_client.setter
    def _client(self, value: httpx.Client | None) -> None:
        self._sync_client = value

    # -- lazy client accessors --

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=60.0)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=60.0)
        return self._async_client

    # -- abstract hooks for subclasses --

    @abstractmethod
    def _headers(self) -> dict[str, str]: ...

    @abstractmethod
    def _payload(self, system_prompt: str, user_prompt: str) -> dict: ...

    @abstractmethod
    def _url(self) -> str: ...

    @abstractmethod
    def _extract_text(self, body: Any) -> str: ...

    # -- judge / ajudge --

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_sync_client()
        resp = client.post(
            self._url(),
            headers=self._headers(),
            json=self._payload(system_prompt, user_prompt),
        )
        resp.raise_for_status()
        return self._extract_text(resp.json())

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_async_client()
        resp = await client.post(
            self._url(),
            headers=self._headers(),
            json=self._payload(system_prompt, user_prompt),
        )
        resp.raise_for_status()
        return self._extract_text(resp.json())

    # -- resource management --

    def close(self) -> None:
        """Close underlying HTTP clients and release connection pools.

        Only the synchronous client can be properly closed here.  The async
        client requires ``await aclose()`` or an ``async with`` block.
        """
        try:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None
        finally:
            # Cannot await in a sync method, so just discard the reference.
            self._async_client = None

    async def aclose(self) -> None:
        """Async close underlying HTTP clients and release connection pools."""
        try:
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None
        finally:
            if self._async_client is not None:
                await self._async_client.aclose()
                self._async_client = None

    def __enter__(self) -> BaseJudgeProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> BaseJudgeProvider:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


class OpenAIJudgeProvider(BaseJudgeProvider):
    """Calls any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
    ):
        super().__init__()
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

    def _url(self) -> str:
        return f"{self.api_base}/chat/completions"

    def _extract_text(self, body: Any) -> str:
        return body["choices"][0]["message"]["content"]


class AnthropicJudgeProvider(BaseJudgeProvider):
    """Calls the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        super().__init__()
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

    def _url(self) -> str:
        return f"{self.api_base}/messages"

    def _extract_text(self, body: Any) -> str:
        return body["content"][0]["text"]


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
