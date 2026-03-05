"""Tests for retry logic in judge providers."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from agent_trace.judge.providers import (
    AnthropicJudgeProvider,
    OpenAIJudgeProvider,
    _is_retryable,
    _retry_async,
    _retry_sync,
    _sleep_duration,
    create_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Build an ``httpx.HTTPStatusError`` with the given status code."""
    response = httpx.Response(status_code, request=httpx.Request("POST", "https://x"))
    return httpx.HTTPStatusError(
        f"{status_code} error", request=response.request, response=response,
    )


class _Counter:
    """Callable that fails *n* times with *exc* then returns *value*."""

    def __init__(self, failures: list[Exception], value: str = "ok"):
        self._failures = list(failures)
        self._value = value
        self.call_count = 0

    def __call__(self) -> str:
        self.call_count += 1
        if self._failures:
            raise self._failures.pop(0)
        return self._value


class _AsyncCounter:
    """Async version of :class:`_Counter`."""

    def __init__(self, failures: list[Exception], value: str = "ok"):
        self._failures = list(failures)
        self._value = value
        self.call_count = 0

    async def __call__(self) -> str:
        self.call_count += 1
        if self._failures:
            raise self._failures.pop(0)
        return self._value


# ---------------------------------------------------------------------------
# _is_retryable
# ---------------------------------------------------------------------------

def test_retryable_429():
    assert _is_retryable(_make_http_status_error(429)) is True


def test_retryable_500():
    assert _is_retryable(_make_http_status_error(500)) is True


def test_retryable_502():
    assert _is_retryable(_make_http_status_error(502)) is True


def test_retryable_503():
    assert _is_retryable(_make_http_status_error(503)) is True


def test_retryable_504():
    assert _is_retryable(_make_http_status_error(504)) is True


def test_not_retryable_400():
    assert _is_retryable(_make_http_status_error(400)) is False


def test_not_retryable_401():
    assert _is_retryable(_make_http_status_error(401)) is False


def test_not_retryable_403():
    assert _is_retryable(_make_http_status_error(403)) is False


def test_not_retryable_404():
    assert _is_retryable(_make_http_status_error(404)) is False


def test_retryable_timeout():
    assert _is_retryable(httpx.ReadTimeout("timed out")) is True


def test_retryable_connect_error():
    assert _is_retryable(httpx.ConnectError("connection refused")) is True


def test_not_retryable_generic_exception():
    assert _is_retryable(ValueError("bad value")) is False


def test_not_retryable_runtime_error():
    assert _is_retryable(RuntimeError("oops")) is False


# ---------------------------------------------------------------------------
# _sleep_duration
# ---------------------------------------------------------------------------

def test_sleep_duration_attempt_zero():
    # attempt 0: exp_delay = base * 2^0 = base; jitter in [0, base]
    for _ in range(50):
        d = _sleep_duration(0, base_delay=1.0, max_delay=30.0)
        assert 0 <= d <= 1.0


def test_sleep_duration_attempt_one():
    # attempt 1: exp_delay = 1.0 * 2^1 = 2.0; jitter in [0, 2.0]
    for _ in range(50):
        d = _sleep_duration(1, base_delay=1.0, max_delay=30.0)
        assert 0 <= d <= 2.0


def test_sleep_duration_capped_by_max_delay():
    # attempt 10: exp_delay = 1.0 * 2^10 = 1024, capped to 5.0
    for _ in range(50):
        d = _sleep_duration(10, base_delay=1.0, max_delay=5.0)
        assert 0 <= d <= 5.0


def test_sleep_duration_zero_base():
    d = _sleep_duration(5, base_delay=0.0, max_delay=10.0)
    assert d == 0.0


# ---------------------------------------------------------------------------
# _retry_sync
# ---------------------------------------------------------------------------

@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_success_first_try(mock_sleep):
    counter = _Counter(failures=[], value="hello")
    result = _retry_sync(counter, max_retries=3, base_delay=1.0, max_delay=30.0)
    assert result == "hello"
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_succeeds_after_transient_failures(mock_sleep):
    failures = [
        _make_http_status_error(429),
        _make_http_status_error(500),
    ]
    counter = _Counter(failures=failures, value="recovered")
    result = _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert result == "recovered"
    assert counter.call_count == 3
    assert mock_sleep.call_count == 2


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_exhausts_retries(mock_sleep):
    failures = [_make_http_status_error(429)] * 4  # more than max_retries
    counter = _Counter(failures=failures, value="never")
    with pytest.raises(httpx.HTTPStatusError):
        _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert counter.call_count == 4  # initial + 3 retries
    assert mock_sleep.call_count == 3


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_non_retryable_raises_immediately(mock_sleep):
    counter = _Counter(failures=[_make_http_status_error(401)], value="never")
    with pytest.raises(httpx.HTTPStatusError):
        _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_timeout_is_retryable(mock_sleep):
    failures = [httpx.ReadTimeout("timeout")]
    counter = _Counter(failures=failures, value="ok")
    result = _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert result == "ok"
    assert counter.call_count == 2
    assert mock_sleep.call_count == 1


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_connect_error_is_retryable(mock_sleep):
    failures = [httpx.ConnectError("refused")]
    counter = _Counter(failures=failures, value="ok")
    result = _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert result == "ok"
    assert counter.call_count == 2
    assert mock_sleep.call_count == 1


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_zero_retries_no_retry(mock_sleep):
    counter = _Counter(failures=[_make_http_status_error(429)], value="never")
    with pytest.raises(httpx.HTTPStatusError):
        _retry_sync(counter, max_retries=0, base_delay=0.01, max_delay=1.0)
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent_trace.judge.providers.time.sleep")
def test_retry_sync_generic_exception_not_retried(mock_sleep):
    counter = _Counter(failures=[ValueError("bad")], value="never")
    with pytest.raises(ValueError, match="bad"):
        _retry_sync(counter, max_retries=3, base_delay=0.01, max_delay=1.0)
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# _retry_async
# ---------------------------------------------------------------------------

@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_success_first_try(mock_sleep):
    counter = _AsyncCounter(failures=[], value="hello")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=1.0, max_delay=30.0)

    result = asyncio.run(_run())
    assert result == "hello"
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_succeeds_after_transient_failures(mock_sleep):
    failures = [
        _make_http_status_error(429),
        _make_http_status_error(503),
    ]
    counter = _AsyncCounter(failures=failures, value="recovered")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=0.01, max_delay=1.0)

    result = asyncio.run(_run())
    assert result == "recovered"
    assert counter.call_count == 3
    assert mock_sleep.call_count == 2


@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_exhausts_retries(mock_sleep):
    failures = [_make_http_status_error(500)] * 4
    counter = _AsyncCounter(failures=failures, value="never")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=0.01, max_delay=1.0)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_run())
    assert counter.call_count == 4
    assert mock_sleep.call_count == 3


@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_non_retryable_raises_immediately(mock_sleep):
    counter = _AsyncCounter(failures=[_make_http_status_error(403)], value="never")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=0.01, max_delay=1.0)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_run())
    assert counter.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_timeout_is_retryable(mock_sleep):
    failures = [httpx.ReadTimeout("timeout")]
    counter = _AsyncCounter(failures=failures, value="ok")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=0.01, max_delay=1.0)

    result = asyncio.run(_run())
    assert result == "ok"
    assert counter.call_count == 2
    assert mock_sleep.call_count == 1


@patch("agent_trace.judge.providers.asyncio.sleep")
def test_retry_async_connect_error_is_retryable(mock_sleep):
    failures = [httpx.ConnectError("refused")]
    counter = _AsyncCounter(failures=failures, value="ok")

    async def _run():
        return await _retry_async(counter, max_retries=3, base_delay=0.01, max_delay=1.0)

    result = asyncio.run(_run())
    assert result == "ok"
    assert counter.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# Factory and config
# ---------------------------------------------------------------------------

def test_create_provider_openai_retry_config():
    p = create_provider(
        "openai", "key", "gpt-4o",
        max_retries=5, retry_base_delay=2.0, retry_max_delay=60.0,
    )
    assert isinstance(p, OpenAIJudgeProvider)
    assert p.max_retries == 5
    assert p.retry_base_delay == 2.0
    assert p.retry_max_delay == 60.0


def test_create_provider_anthropic_retry_config():
    p = create_provider(
        "anthropic", "key", "claude-sonnet-4-20250514",
        max_retries=7, retry_base_delay=0.5, retry_max_delay=10.0,
    )
    assert isinstance(p, AnthropicJudgeProvider)
    assert p.max_retries == 7
    assert p.retry_base_delay == 0.5
    assert p.retry_max_delay == 10.0


def test_create_provider_default_retry_config():
    p = create_provider("openai", "key", "gpt-4o")
    assert p.max_retries == 3
    assert p.retry_base_delay == 1.0
    assert p.retry_max_delay == 30.0


def test_config_retry_defaults():
    from agent_trace.config import AgentTraceConfig

    cfg = AgentTraceConfig()
    assert cfg.judge_max_retries == 3
    assert cfg.judge_retry_base_delay == 1.0
    assert cfg.judge_retry_max_delay == 30.0


def test_config_retry_custom():
    from agent_trace.config import AgentTraceConfig

    cfg = AgentTraceConfig(
        judge_max_retries=10,
        judge_retry_base_delay=0.5,
        judge_retry_max_delay=120.0,
    )
    assert cfg.judge_max_retries == 10
    assert cfg.judge_retry_base_delay == 0.5
    assert cfg.judge_retry_max_delay == 120.0
