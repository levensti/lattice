"""Tests for provider routing and session serialization."""

from lattice.context import ActionRecord, GroupRecord, TraceSession
from lattice.judge.providers import _route_model


# ── Model routing ─────────────────────────────────────────────────────


def test_route_openai_models():
    for prefix in ("gpt-4o", "o1-mini", "o3-preview", "o4-mini"):
        prov, base, env = _route_model(prefix)
        assert prov == "openai"
        assert "openai.com" in base
        assert env == "OPENAI_API_KEY"


def test_route_anthropic_models():
    prov, base, env = _route_model("claude-sonnet-4-20250514")
    assert prov == "anthropic"
    assert "anthropic.com" in base
    assert env == "ANTHROPIC_API_KEY"


def test_route_openrouter_slash():
    prov, base, env = _route_model("google/gemini-2.0-flash")
    assert prov == "openai"
    assert "openrouter" in base
    assert env == "OPENROUTER_API_KEY"


def test_route_unknown_falls_to_openrouter():
    prov, base, env = _route_model("some-unknown-model")
    assert "openrouter" in base
    assert env == "OPENROUTER_API_KEY"


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
