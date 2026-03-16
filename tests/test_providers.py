"""Tests for InferenceProvider ABC, ApiType dispatch, and concrete providers."""

import pytest

from lattice.context import ActionRecord, GroupRecord, TraceSession
from lattice.judge.providers import (
    AnthropicProvider,
    ApiType,
    InferenceProvider,
    OpenAIProvider,
    OpenRouterProvider,
)


# ── ABC enforcement ──────────────────────────────────────────────────


def test_cannot_instantiate_bare_inference_provider():
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        InferenceProvider()


def test_subclass_must_set_api_type():
    class NoApiType(InferenceProvider):
        pass

    prov = NoApiType()
    with pytest.raises(AttributeError):
        prov.judge("sys", "user")


def test_subclass_dispatches_to_matching_method():
    class Custom(InferenceProvider):
        api_type = ApiType.CHAT_COMPLETIONS

        def _chat_completions(self, system_prompt, user_prompt):
            return f"cc:{system_prompt}:{user_prompt}"

        async def _achat_completions(self, system_prompt, user_prompt):
            return f"acc:{system_prompt}:{user_prompt}"

    prov = Custom()
    assert isinstance(prov, InferenceProvider)
    assert prov.judge("sys", "usr") == "cc:sys:usr"


def test_subclass_unsupported_api_type_raises():
    class OnlyMessages(InferenceProvider):
        api_type = ApiType.MESSAGES

    prov = OnlyMessages()
    # _messages not overridden → NotImplementedError
    with pytest.raises(NotImplementedError, match="MESSAGES"):
        prov.judge("sys", "user")


# ── ApiType enum ─────────────────────────────────────────────────────


def test_api_type_values():
    assert ApiType.CHAT_COMPLETIONS.value == "chat_completions"
    assert ApiType.RESPONSES.value == "responses"
    assert ApiType.MESSAGES.value == "messages"


# ── OpenAIProvider ───────────────────────────────────────────────────


def test_openai_provider_reads_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    prov = OpenAIProvider("gpt-4o")
    assert prov.api_key == "test-key"
    assert prov.model == "gpt-4o"


def test_openai_provider_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    prov = OpenAIProvider("gpt-4o", api_key="explicit-key")
    assert prov.api_key == "explicit-key"


def test_openai_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        OpenAIProvider("gpt-4o")


def test_openai_provider_custom_api_base(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("my-model", api_base="https://api.fireworks.ai/v1")
    assert prov.api_base == "https://api.fireworks.ai/v1"


def test_openai_provider_default_api_type(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("gpt-4o")
    assert prov.api_type == ApiType.CHAT_COMPLETIONS


def test_openai_provider_responses_api_type(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("gpt-4o", api_type=ApiType.RESPONSES)
    assert prov.api_type == ApiType.RESPONSES


def test_openai_provider_rejects_messages_api_type(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(ValueError, match="not messages"):
        OpenAIProvider("gpt-4o", api_type=ApiType.MESSAGES)


def test_openai_provider_cc_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("gpt-4o", temperature=0.5, top_p=0.9)
    payload = prov._cc_payload("sys", "user")
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9
    assert payload["messages"][0]["role"] == "system"


def test_openai_provider_cc_top_p_omitted_when_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("gpt-4o", temperature=0.1)
    payload = prov._cc_payload("sys", "user")
    assert "top_p" not in payload


def test_openai_provider_responses_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prov = OpenAIProvider("gpt-4o", api_type=ApiType.RESPONSES, temperature=0.3)
    payload = prov._responses_payload("sys", "user")
    assert payload["instructions"] == "sys"
    assert payload["input"] == "user"
    assert payload["temperature"] == 0.3


# ── AnthropicProvider ────────────────────────────────────────────────


def test_anthropic_provider_reads_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    prov = AnthropicProvider("claude-sonnet-4-20250514")
    assert prov.api_key == "test-key"


def test_anthropic_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        AnthropicProvider()


def test_anthropic_provider_api_type_is_messages(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    prov = AnthropicProvider()
    assert prov.api_type == ApiType.MESSAGES


def test_anthropic_provider_messages_payload(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    prov = AnthropicProvider(temperature=0.3, top_p=0.8)
    payload = prov._messages_payload("sys", "user")
    assert payload["system"] == "sys"
    assert payload["messages"][0]["content"] == "user"
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8


def test_anthropic_provider_top_p_omitted_when_none(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    prov = AnthropicProvider(temperature=0.1)
    payload = prov._messages_payload("sys", "user")
    assert "top_p" not in payload


# ── OpenRouterProvider ───────────────────────────────────────────────


def test_openrouter_provider_reads_env_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    prov = OpenRouterProvider("google/gemini-2.0-flash")
    assert prov.api_key == "or-key"
    assert "openrouter" in prov.api_base


def test_openrouter_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        OpenRouterProvider("google/gemini-2.0-flash")


def test_openrouter_provider_is_inference_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    prov = OpenRouterProvider("google/gemini-2.0-flash")
    assert isinstance(prov, InferenceProvider)
    assert not isinstance(prov, OpenAIProvider)


def test_openrouter_provider_api_type_is_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    prov = OpenRouterProvider("google/gemini-2.0-flash")
    assert prov.api_type == ApiType.CHAT_COMPLETIONS


# ── TraceSession.to_dict ──────────────────────────────────────────────


def test_to_dict_excludes_private_counter():
    session = TraceSession(goal="test goal")
    session.next_index()
    session.next_index()
    d = session.to_dict()
    assert "_action_counter" not in d
    assert d["goal"] == "test goal"
    assert d["trace_id"]  # should have a trace ID


def test_to_dict_includes_actions():
    session = TraceSession(goal="g")
    session.add_action(ActionRecord(
        span_id="s1", name="step1", description="", goal="g",
        input_data="in", output_data="out", action_index=0, latency_ms=10.0,
    ))
    d = session.to_dict()
    assert len(d["actions"]) == 1
    assert d["actions"][0]["name"] == "step1"


def test_to_dict_includes_groups():
    session = TraceSession(goal="g")
    session.add_group(GroupRecord(
        group_id="g1", group_type="loop", name="react",
    ))
    d = session.to_dict()
    assert len(d["groups"]) == 1
    assert d["groups"][0]["group_type"] == "loop"
