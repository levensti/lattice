"""InferenceProvider ABC and concrete implementations for judge scoring."""

from __future__ import annotations

import logging
import os
from abc import ABC
from enum import Enum

import httpx

logger = logging.getLogger("lattice")

_DEFAULT_TIMEOUT = 60.0


class ApiType(Enum):
    """The wire protocol used to communicate with the LLM.

    Each :class:`InferenceProvider` has an ``api_type`` that determines
    which internal method is called when ``judge()`` / ``ajudge()`` runs.
    """

    CHAT_COMPLETIONS = "chat_completions"
    """OpenAI Chat Completions (``/v1/chat/completions``)."""

    RESPONSES = "responses"
    """OpenAI Responses (``/v1/responses``)."""

    MESSAGES = "messages"
    """Anthropic Messages (``/v1/messages``)."""


class InferenceProvider(ABC):
    """Abstract base class for LLM providers used by the judge scorer.

    Each provider sets ``api_type`` to declare which wire protocol it
    uses. The public ``judge()`` / ``ajudge()`` methods dispatch to the
    matching internal method automatically.

    **Built-in providers** set a default ``api_type`` and implement the
    relevant methods. For example, :class:`OpenAIProvider` defaults to
    ``ApiType.CHAT_COMPLETIONS`` but also supports ``ApiType.RESPONSES``.

    **Custom providers** subclass ``InferenceProvider``, set ``api_type``,
    and implement the matching pair of methods::

        class MyProvider(InferenceProvider):
            api_type = ApiType.CHAT_COMPLETIONS

            def _chat_completions(self, system_prompt, user_prompt):
                return my_llm_call(system_prompt, user_prompt)

            async def _achat_completions(self, system_prompt, user_prompt):
                return await my_async_llm_call(system_prompt, user_prompt)

        lattice.score_trace(session, provider=MyProvider())
    """

    api_type: ApiType

    def __new__(cls, *args, **kwargs):
        if cls is InferenceProvider:
            raise TypeError(
                "InferenceProvider cannot be instantiated directly — "
                "use OpenAIProvider, AnthropicProvider, OpenRouterProvider, "
                "or subclass InferenceProvider."
            )
        return super().__new__(cls)

    # ── Public dispatch ──────────────────────────────────────────────

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        """Send a judge request synchronously and return the response text.

        Dispatches to the internal method matching ``self.api_type``.
        """
        match self.api_type:
            case ApiType.CHAT_COMPLETIONS:
                return self._chat_completions(system_prompt, user_prompt)
            case ApiType.RESPONSES:
                return self._responses(system_prompt, user_prompt)
            case ApiType.MESSAGES:
                return self._messages(system_prompt, user_prompt)

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        """Send a judge request asynchronously and return the response text.

        Dispatches to the async internal method matching ``self.api_type``.
        """
        match self.api_type:
            case ApiType.CHAT_COMPLETIONS:
                return await self._achat_completions(system_prompt, user_prompt)
            case ApiType.RESPONSES:
                return await self._aresponses(system_prompt, user_prompt)
            case ApiType.MESSAGES:
                return await self._amessages(system_prompt, user_prompt)

    # ── Per-API-type methods (override the ones your provider supports) ──

    def _chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.CHAT_COMPLETIONS"
        )

    async def _achat_completions(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.CHAT_COMPLETIONS"
        )

    def _responses(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.RESPONSES"
        )

    async def _aresponses(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.RESPONSES"
        )

    def _messages(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.MESSAGES"
        )

    async def _amessages(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support ApiType.MESSAGES"
        )


# ── Response extractors ──────────────────────────────────────────────


def _extract_chat_completions_content(resp: httpx.Response) -> str:
    """Extract the assistant message from a Chat Completions response."""
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected response from {resp.url}: {resp.text[:500]}"
        ) from exc


def _extract_responses_content(resp: httpx.Response) -> str:
    """Extract the text from an OpenAI Responses response."""
    try:
        for block in resp.json()["output"]:
            if block.get("type") == "message":
                for content in block["content"]:
                    if content.get("type") == "output_text":
                        return content["text"]
        raise ValueError("No output_text block found in response")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected response from {resp.url}: {resp.text[:500]}"
        ) from exc


def _extract_messages_content(resp: httpx.Response) -> str:
    """Extract the text from an Anthropic Messages response."""
    try:
        return resp.json()["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected response from {resp.url}: {resp.text[:500]}"
        ) from exc


# ── Concrete providers ───────────────────────────────────────────────


class OpenAIProvider(InferenceProvider):
    """Calls the OpenAI API (or any OpenAI-compatible endpoint).

    Supports both the Chat Completions and Responses wire protocols.

    Args:
        model: Model name (e.g. ``"gpt-4o"``).
        api_key: API key. Falls back to ``OPENAI_API_KEY`` env var.
        api_base: Base URL for the API. Defaults to OpenAI's endpoint.
            Override for Fireworks, Sail, or any OpenAI-compatible service.
        api_type: Wire protocol. ``CHAT_COMPLETIONS`` (default) or
            ``RESPONSES``.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
    """

    _SUPPORTED = frozenset({ApiType.CHAT_COMPLETIONS, ApiType.RESPONSES})

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        api_type: ApiType = ApiType.CHAT_COMPLETIONS,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = 0.1,
        top_p: float | None = None,
    ):
        if api_type not in self._SUPPORTED:
            raise ValueError(
                f"OpenAIProvider supports {', '.join(t.value for t in self._SUPPORTED)}, "
                f"not {api_type.value}"
            )
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key for OpenAIProvider. "
                "Set OPENAI_API_KEY or pass api_key=."
            )
        self.api_type = api_type
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Chat Completions ─────────────────────────────────────────────

    def _cc_payload(self, system_prompt: str, user_prompt: str) -> dict:
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
        return payload

    def _chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._cc_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_chat_completions_content(resp)

    async def _achat_completions(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._cc_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_chat_completions_content(resp)

    # ── Responses ────────────────────────────────────────────────────

    def _responses_payload(self, system_prompt: str, user_prompt: str) -> dict:
        payload: dict = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def _responses(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/responses",
                headers=self._headers(),
                json=self._responses_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_responses_content(resp)

    async def _aresponses(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/responses",
                headers=self._headers(),
                json=self._responses_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_responses_content(resp)


class AnthropicProvider(InferenceProvider):
    """Calls the Anthropic Messages API.

    Args:
        model: Model name (e.g. ``"claude-sonnet-4-20250514"``).
        api_key: API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = 0.1,
        top_p: float | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key for AnthropicProvider. "
                "Set ANTHROPIC_API_KEY or pass api_key=."
            )
        self.api_type = ApiType.MESSAGES
        self.model = model
        self.api_key = api_key
        self.api_base = "https://api.anthropic.com/v1"
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _messages_payload(self, system_prompt: str, user_prompt: str) -> dict:
        payload: dict = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 512,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload

    def _messages(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._messages_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_messages_content(resp)

    async def _amessages(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/messages",
                headers=self._headers(),
                json=self._messages_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_messages_content(resp)


class OpenRouterProvider(InferenceProvider):
    """Routes requests through OpenRouter's unified API.

    OpenRouter provides access to many models (Gemini, Llama, Mistral, etc.)
    through a single endpoint with its own API key. Uses the Chat Completions
    wire protocol.

    Args:
        model: Model name as listed on OpenRouter (e.g.
            ``"google/gemini-2.0-flash"``, ``"meta-llama/llama-3-70b"``).
        api_key: API key. Falls back to ``OPENROUTER_API_KEY`` env var.
        timeout: Request timeout in seconds.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = 0.1,
        top_p: float | None = None,
    ):
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key for OpenRouterProvider. "
                "Set OPENROUTER_API_KEY or pass api_key=."
            )
        self.api_type = ApiType.CHAT_COMPLETIONS
        self.model = model
        self.api_key = api_key
        self.api_base = "https://openrouter.ai/api/v1"
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _cc_payload(self, system_prompt: str, user_prompt: str) -> dict:
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
        return payload

    def _chat_completions(self, system_prompt: str, user_prompt: str) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._cc_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_chat_completions_content(resp)

    async def _achat_completions(self, system_prompt: str, user_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers(),
                json=self._cc_payload(system_prompt, user_prompt),
            )
            resp.raise_for_status()
            return _extract_chat_completions_content(resp)
