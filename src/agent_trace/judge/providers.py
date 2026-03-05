from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, Protocol, TypeVar, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception represents a transient failure worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


def _sleep_duration(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute sleep time using exponential backoff with full jitter."""
    exp_delay = base_delay * (2 ** attempt)
    capped = min(exp_delay, max_delay)
    return random.uniform(0, capped)


def _retry_sync(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """Execute *fn* with retries on transient HTTP / connection errors."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = _sleep_duration(attempt, base_delay, max_delay)
            logger.warning(
                "Retry %d/%d after %.2fs – %s: %s",
                attempt + 1,
                max_retries,
                delay,
                type(exc).__name__,
                exc,
            )
            time.sleep(delay)
    # Unreachable, but keeps type-checkers happy.
    raise RuntimeError("unreachable")  # pragma: no cover


async def _retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """Async version of :func:`_retry_sync`."""
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = _sleep_duration(attempt, base_delay, max_delay)
            logger.warning(
                "Retry %d/%d after %.2fs – %s: %s",
                attempt + 1,
                max_retries,
                delay,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


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
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

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
        def _call() -> str:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        return _retry_sync(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async def _call() -> str:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        return await _retry_async(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )


class AnthropicJudgeProvider:
    """Calls the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://api.anthropic.com/v1"
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

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
        def _call() -> str:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{self.api_base}/messages",
                    headers=self._headers(),
                    json=self._payload(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]

        return _retry_sync(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    async def ajudge(self, system_prompt: str, user_prompt: str) -> str:
        async def _call() -> str:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_base}/messages",
                    headers=self._headers(),
                    json=self._payload(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]

        return await _retry_async(
            _call,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )


def create_provider(
    provider_name: str,
    api_key: str,
    model: str,
    api_base: str = "https://api.openai.com/v1",
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> OpenAIJudgeProvider | AnthropicJudgeProvider:
    """Factory that returns the right provider instance."""
    if provider_name == "openai":
        return OpenAIJudgeProvider(
            api_key,
            model,
            api_base,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
    if provider_name == "anthropic":
        return AnthropicJudgeProvider(
            api_key,
            model,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
    raise ValueError(
        f"Unknown judge provider '{provider_name}'. Use 'openai' or 'anthropic'."
    )
