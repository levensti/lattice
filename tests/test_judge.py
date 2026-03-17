import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.context import ActionRecord, JudgeConfig, JudgeResult, TraceSession
from lattice.judge.prompt_builder import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    build_session_judge_prompt,
)
from lattice.judge.providers import AnthropicProvider, OpenAIProvider
from lattice.judge.scorer import _parse_judge_response, BackgroundScorer, score_session, score_trace

_RUBRIC = "Score 1-5. Respond: {\"reasoning\": \"...\", \"score\": <1-5>, \"explanation\": \"...\"}"


def _make_action(**overrides):
    defaults = dict(
        span_id="s1", name="researcher", description="Searches for info",
        goal="Must cite sources", input_data="What is Python?",
        output_data="Python is a programming language.",
        action_index=0, latency_ms=10.0,
    )
    defaults.update(overrides)
    return ActionRecord(**defaults)


def _fake_provider(response: str = '{"score": 4, "explanation": "Good"}'):
    prov = MagicMock()
    prov.judge.return_value = response
    prov.ajudge = AsyncMock(return_value=response)
    return prov


# ── Prompt builder ────────────────────────────────────────────────────


def test_build_judge_prompt_contains_fields():
    prompt = build_judge_prompt(
        name="researcher",
        description="Searches for information",
        goal="Must cite sources",
        input_data="What is Python?",
        output_data="Python is a programming language.",
    )
    assert "researcher" in prompt
    assert "Searches for information" in prompt
    assert "Must cite sources" in prompt
    assert "What is Python?" in prompt
    assert "Python is a programming language." in prompt


def test_build_judge_prompt_default_goal():
    prompt = build_judge_prompt(
        name="x", description="", goal="", input_data="in", output_data="out",
    )
    assert "quality" in prompt.lower()


# ── Response parsing ──────────────────────────────────────────────────


def test_parse_valid_json():
    score, explanation = _parse_judge_response(
        '{"score": 4, "explanation": "Good but missing sources"}'
    )
    assert score == 4.0
    assert "missing sources" in explanation


def test_parse_json_with_reasoning_field():
    score, explanation = _parse_judge_response(
        '{"reasoning": "Step by step...", "score": 4, "explanation": "Good"}'
    )
    assert score == 4.0
    assert "Good" in explanation


def test_parse_json_in_code_block():
    score, explanation = _parse_judge_response(
        '```json\n{"score": 3, "explanation": "Acceptable"}\n```'
    )
    assert score == 3.0
    assert "Acceptable" in explanation


def test_parse_fallback_score_slash():
    score, _ = _parse_judge_response("I'd rate this 2/5 because it lacks detail.")
    assert score == 2.0


def test_parse_failure():
    score, explanation = _parse_judge_response("This is gibberish with no score.")
    assert score == 0.0
    assert "Could not parse" in explanation


# ── Global judge ──────────────────────────────────────────────────────


def test_score_trace_custom_system_prompt():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_trace(session, provider=prov, system_prompt="Be a strict auditor.")

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Be a strict auditor."


def test_score_trace_default_system_prompt():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_trace(session, provider=prov)

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == JUDGE_SYSTEM_PROMPT


def test_score_trace_custom_action_prompt_builder():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    def custom_builder(*, name, description, goal, input_data, output_data):
        return f"CUSTOM: {name} | {goal}"

    prov = _fake_provider()
    score_trace(session, provider=prov, action_prompt_builder=custom_builder)

    user_arg = prov.judge.call_args[0][1]
    assert user_arg == "CUSTOM: researcher | Must cite sources"


# ── Session scoring ───────────────────────────────────────────────────


def test_score_session_custom_session_prompt_builder():
    session = TraceSession(goal="Summarize the article")
    session.workflow_name = "summarizer"
    session.add_action(_make_action())

    def custom_builder(*, goal, final_output, workflow_name):
        return f"CUSTOM SESSION: {workflow_name} | {goal}"

    prov = _fake_provider()
    score_session(session, provider=prov, session_prompt_builder=custom_builder)

    user_arg = prov.judge.call_args[0][1]
    assert user_arg == "CUSTOM SESSION: summarizer | Summarize the article"


def test_score_session_custom_system_prompt():
    session = TraceSession(goal="test goal")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_session(session, provider=prov, system_prompt="Judge harshly.")

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Judge harshly."


def test_score_session_defaults():
    session = TraceSession(goal="test goal")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_session(session, provider=prov)

    system_arg = prov.judge.call_args[0][0]
    user_arg = prov.judge.call_args[0][1]
    assert system_arg == JUDGE_SYSTEM_PROMPT
    expected = build_session_judge_prompt(
        goal="test goal",
        final_output="Python is a programming language.",
        workflow_name="",
    )
    assert user_arg == expected


# ── Errored steps are skipped ─────────────────────────────────────────


def test_score_trace_skips_errored_steps():
    session = TraceSession(goal="test")
    session.add_action(_make_action(name="ok"))
    session.add_action(_make_action(name="failed", error="boom"))

    prov = _fake_provider()
    score_trace(session, provider=prov)

    assert prov.judge.call_count == 1


# ── BackgroundScorer ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_background_scorer_scores_submitted_session():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    prov = _fake_provider()
    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    scorer.submit(session)
    await scorer.drain()
    await scorer.cancel()

    assert prov.ajudge.call_count == 1
    assert session.actions[0].score == 4.0


@pytest.mark.asyncio
async def test_background_scorer_submit_is_nonblocking():
    session = TraceSession(goal="test")
    session.add_action(_make_action())
    judged = []

    async def slow_judge(system, prompt):
        await asyncio.sleep(0.05)
        judged.append(True)
        return '{"score": 3, "explanation": "ok"}'

    prov = MagicMock()
    prov.ajudge = slow_judge

    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    scorer.submit(session)
    assert judged == []
    await scorer.drain()
    assert judged == [True]
    await scorer.cancel()


@pytest.mark.asyncio
async def test_background_scorer_multiple_sessions():
    sessions = [TraceSession(goal="test") for _ in range(3)]
    for s in sessions:
        s.add_action(_make_action())

    prov = _fake_provider()
    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    for s in sessions:
        scorer.submit(s)
    await scorer.drain()
    await scorer.cancel()

    assert prov.ajudge.call_count == 3
    for s in sessions:
        assert s.actions[0].score == 4.0


@pytest.mark.asyncio
async def test_background_scorer_skips_errored_steps():
    session = TraceSession(goal="test")
    session.add_action(_make_action(name="ok"))
    session.add_action(_make_action(name="failed", error="boom"))

    prov = _fake_provider()
    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    scorer.submit(session)
    await scorer.drain()
    await scorer.cancel()

    assert prov.ajudge.call_count == 1


@pytest.mark.asyncio
async def test_background_scorer_submit_before_start_raises():
    scorer = BackgroundScorer(provider=_fake_provider())
    with pytest.raises(RuntimeError, match="not started"):
        scorer.submit(TraceSession(goal="test"))


@pytest.mark.asyncio
async def test_background_scorer_cancel_stops_immediately():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    async def never_returns(system, prompt):
        await asyncio.sleep(999)
        return '{"score": 5, "explanation": "done"}'

    prov = MagicMock()
    prov.ajudge = never_returns

    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    scorer.submit(session)
    await scorer.cancel()

    assert scorer._worker_task is None
    assert session.actions[0].score is None


@pytest.mark.asyncio
async def test_background_scorer_context_manager_does_not_block():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    async def never_returns(system, prompt):
        await asyncio.sleep(999)
        return '{"score": 5, "explanation": "done"}'

    prov = MagicMock()
    prov.ajudge = never_returns

    async with BackgroundScorer(provider=prov) as scorer:
        scorer.submit(session)


@pytest.mark.asyncio
async def test_background_scorer_worker_error_does_not_crash_worker():
    good_session = TraceSession(goal="test")
    good_session.add_action(_make_action())
    call_count = 0

    async def flaky_judge(system, prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient failure")
        return '{"score": 5, "explanation": "great"}'

    bad_session = TraceSession(goal="bad")
    bad_session.add_action(_make_action(name="first"))

    prov = MagicMock()
    prov.ajudge = flaky_judge

    scorer = BackgroundScorer(provider=prov)
    await scorer.start()
    scorer.submit(bad_session)
    scorer.submit(good_session)
    await scorer.drain()
    await scorer.cancel()

    assert good_session.actions[0].score == 5.0


# ── JudgeConfig / per-action judge ────────────────────────────────────


def test_judge_config_requires_system_prompt_and_provider():
    with pytest.raises(TypeError):
        JudgeConfig()  # missing both

    with pytest.raises(TypeError):
        JudgeConfig(system_prompt="...")  # missing provider

    with pytest.raises(TypeError):
        JudgeConfig(provider=_fake_provider())  # missing system_prompt


def test_judge_config_repr():
    jc = JudgeConfig(system_prompt="Score strictly.", provider=_fake_provider())
    assert "system_prompt=..." in repr(jc)
    assert "provider=" in repr(jc)


def test_per_action_judge_config_overrides_global():
    """A JudgeConfig on an action uses that config's provider and system_prompt."""
    per_action_prov = _fake_provider('{"score": 5, "explanation": "Perfect"}')
    global_prov = _fake_provider('{"score": 1, "explanation": "Bad"}')

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(system_prompt="Strict rubric.", provider=per_action_prov)
    session.add_action(action)

    score_trace(session, provider=global_prov)

    global_prov.judge.assert_not_called()
    per_action_prov.judge.assert_called_once()
    system_arg = per_action_prov.judge.call_args[0][0]
    assert system_arg == "Strict rubric."
    assert session.actions[0].score == 5.0


def test_per_action_judge_system_prompt_used():
    """JudgeConfig.system_prompt replaces the global system_prompt."""
    prov = _fake_provider()
    jc = JudgeConfig(system_prompt="Per-action rubric.", provider=prov)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = jc
    session.add_action(action)

    score_trace(session, provider=MagicMock(), system_prompt="Global prompt.")

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Per-action rubric."


def test_per_action_judge_custom_prompt_builder():
    """JudgeConfig.action_prompt_builder overrides global action_prompt_builder."""
    prov = _fake_provider()

    def custom_builder(*, name, description, goal, input_data, output_data):
        return f"PER-ACTION: {name}"

    jc = JudgeConfig(system_prompt=_RUBRIC, provider=prov, action_prompt_builder=custom_builder)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = jc
    session.add_action(action)

    score_trace(session, provider=MagicMock())

    user_arg = prov.judge.call_args[0][1]
    assert user_arg == "PER-ACTION: researcher"


@pytest.mark.asyncio
async def test_per_action_judge_async():
    from lattice.judge.scorer import async_score_trace

    per_action_prov = _fake_provider('{"reasoning": "...", "score": 5, "explanation": "great"}')
    global_prov = _fake_provider()

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(system_prompt=_RUBRIC, provider=per_action_prov)
    session.add_action(action)

    await async_score_trace(session, provider=global_prov)

    assert per_action_prov.ajudge.call_count == 1
    assert session.actions[0].score == 5.0


def test_judge_stripped_from_serialization():
    """judge field must not appear in to_dict() output (non-serializable)."""
    prov = _fake_provider()
    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(system_prompt=_RUBRIC, provider=prov)
    session.add_action(action)

    d = session.to_dict()
    assert "judge" not in d["actions"][0]
    assert "judge_result" in d["actions"][0]
    # judge_config summary should be populated from the live JudgeConfig
    cfg = d["actions"][0]["judge_config"]
    assert cfg is not None
    assert cfg["system_prompt"] == _RUBRIC
    assert cfg["provider"] == type(prov).__name__
