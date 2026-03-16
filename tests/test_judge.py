import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.context import ActionRecord, JudgeConfig, JudgeResult, TraceSession
from lattice.judge.prompt_builder import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
    build_session_judge_prompt,
)
from lattice.judge.providers import AnthropicJudgeProvider, OpenAIJudgeProvider
from lattice.judge.scorer import _parse_judge_response, BackgroundScorer, score_session, score_trace


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


def test_build_judge_prompt_default_criteria():
    prompt = build_judge_prompt(
        name="x", description="", goal="", input_data="in", output_data="out",
    )
    assert "quality" in prompt.lower()


def test_build_judge_prompt_with_criteria():
    prompt = build_judge_prompt(
        name="researcher",
        description="",
        goal="Research accurately",
        input_data="query",
        output_data="answer",
        criteria={
            "factual_accuracy": "Are facts correct?",
            "citation_quality": "Are sources cited?",
        },
    )
    assert "factual_accuracy" in prompt
    assert "Are facts correct?" in prompt
    assert "citation_quality" in prompt
    assert "Are sources cited?" in prompt
    # response format should include per-criterion JSON shape
    assert '"factual_accuracy"' in prompt
    assert '"citation_quality"' in prompt


def test_build_judge_prompt_with_reference():
    prompt = build_judge_prompt(
        name="qa",
        description="",
        goal="Answer correctly",
        input_data="question",
        output_data="answer",
        reference="The correct answer is 42.",
    )
    assert "Reference answer" in prompt
    assert "The correct answer is 42." in prompt


def test_build_judge_prompt_with_criteria_and_reference():
    prompt = build_judge_prompt(
        name="qa",
        description="",
        goal="Answer correctly",
        input_data="question",
        output_data="answer",
        criteria={"correctness": "Does it match?"},
        reference="Ground truth here.",
    )
    assert "correctness" in prompt
    assert "Ground truth here." in prompt


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
    score, _ = _parse_judge_response(
        "I'd rate this 2/5 because it lacks detail."
    )
    assert score == 2.0


def test_parse_failure():
    score, explanation = _parse_judge_response(
        "This is gibberish with no score."
    )
    assert score == 0.0
    assert "Could not parse" in explanation


# ── Custom system prompt ──────────────────────────────────────────────


def test_score_trace_custom_system_prompt():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_trace(session, provider=prov, system_prompt="Be a strict auditor.")

    prov.judge.assert_called_once()
    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Be a strict auditor."


def test_score_trace_default_system_prompt():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    prov = _fake_provider()
    score_trace(session, provider=prov)

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == JUDGE_SYSTEM_PROMPT


# ── Custom step prompt builder ────────────────────────────────────────


def test_score_trace_custom_action_prompt_builder():
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    def custom_builder(*, name, description, goal, input_data, output_data):
        return f"CUSTOM: {name} | {goal}"

    prov = _fake_provider()
    score_trace(session, provider=prov, action_prompt_builder=custom_builder)

    user_arg = prov.judge.call_args[0][1]
    assert user_arg == "CUSTOM: researcher | Must cite sources"


# ── Custom session prompt builder ─────────────────────────────────────


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
    """submit() returns immediately without awaiting the judge."""
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

    scorer.submit(session)   # must not block
    assert judged == []      # judge hasn't run yet

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
    """cancel() kills the worker without waiting for pending sessions."""
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

    # cancel() should return quickly even though the judge is "stuck"
    await scorer.cancel()
    assert scorer._worker_task is None
    # Session was NOT scored — that's fine, it's in SQLite for later
    assert session.actions[0].score is None


@pytest.mark.asyncio
async def test_background_scorer_context_manager_does_not_block():
    """__aexit__ calls cancel(), not drain()."""
    session = TraceSession(goal="test")
    session.add_action(_make_action())

    async def never_returns(system, prompt):
        await asyncio.sleep(999)
        return '{"score": 5, "explanation": "done"}'

    prov = MagicMock()
    prov.ajudge = never_returns

    async with BackgroundScorer(provider=prov) as scorer:
        scorer.submit(session)
    # If __aexit__ called drain(), this test would hang forever


@pytest.mark.asyncio
async def test_background_scorer_worker_error_does_not_crash_worker():
    """A bad session should log an error but leave the worker alive."""
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
    scorer.submit(bad_session)    # will raise inside worker
    scorer.submit(good_session)   # should still be scored
    await scorer.drain()
    await scorer.cancel()

    assert good_session.actions[0].score == 5.0


# ── JudgeConfig / per-action judge ────────────────────────────────────


def test_judge_config_repr_with_model():
    jc = JudgeConfig(model="gpt-4o", temperature=0.0)
    assert "gpt-4o" in repr(jc)
    assert "temperature=0.0" in repr(jc)


def test_judge_config_repr_with_provider():
    prov = MagicMock()
    jc = JudgeConfig(provider=prov)
    assert "provider=" in repr(jc)


def test_judge_config_repr_with_criteria():
    jc = JudgeConfig(model="gpt-4o", criteria={"accuracy": "Is it right?"})
    assert "accuracy" in repr(jc)


def test_judge_config_repr_with_reference():
    jc = JudgeConfig(model="gpt-4o", reference="The answer is 42.")
    assert "reference=..." in repr(jc)


def test_per_action_judge_config_used():
    """A JudgeConfig on an action uses that config's provider, not the global one."""
    per_action_prov = _fake_provider('{"score": 5, "explanation": "Perfect"}')
    global_prov = _fake_provider('{"score": 1, "explanation": "Bad"}')

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(provider=per_action_prov)
    session.add_action(action)

    score_trace(session, provider=global_prov)

    global_prov.judge.assert_not_called()
    per_action_prov.judge.assert_called_once()
    assert session.actions[0].score == 5.0


def test_per_action_judge_custom_system_prompt():
    """JudgeConfig.system_prompt overrides the global system_prompt for that action."""
    prov = _fake_provider()
    jc = JudgeConfig(provider=prov, system_prompt="Be extra strict.")

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = jc
    session.add_action(action)

    score_trace(session, provider=MagicMock())

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Be extra strict."


def test_per_action_judge_falls_back_to_global_system_prompt():
    """When JudgeConfig.system_prompt is None, fall back to global system_prompt."""
    prov = _fake_provider()
    jc = JudgeConfig(provider=prov)  # no system_prompt

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = jc
    session.add_action(action)

    score_trace(session, provider=MagicMock(), system_prompt="Global prompt.")

    system_arg = prov.judge.call_args[0][0]
    assert system_arg == "Global prompt."


def test_per_action_judge_custom_prompt_builder():
    """JudgeConfig.action_prompt_builder overrides global action_prompt_builder."""
    prov = _fake_provider()

    def custom_builder(*, name, description, goal, input_data, output_data):
        return f"PER-ACTION: {name}"

    jc = JudgeConfig(provider=prov, action_prompt_builder=custom_builder)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = jc
    session.add_action(action)

    score_trace(session, provider=MagicMock())

    user_arg = prov.judge.call_args[0][1]
    assert user_arg == "PER-ACTION: researcher"


def test_criteria_single_call_multiple_scores():
    """criteria= triggers one LLM call and produces per-criterion JudgeResults."""
    response = (
        '{"reasoning": "step by step...", '
        '"criteria": {'
        '  "citation_quality": {"score": 4, "explanation": "Mostly cited"},'
        '  "factual_accuracy": {"score": 2, "explanation": "Facts wrong"}'
        '}, "score": 3.0, "explanation": "Mixed"}'
    )
    prov = _fake_provider(response)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        criteria={
            "citation_quality": "Are sources cited properly?",
            "factual_accuracy": "Are facts correct?",
        },
    )
    session.add_action(action)

    score_trace(session, provider=None)

    # Only one LLM call was made (not two)
    assert prov.judge.call_count == 1
    assert session.actions[0].score == pytest.approx(3.0)
    assert len(session.actions[0].judge_results) == 2
    scores = {r.name: r.score for r in session.actions[0].judge_results}
    assert scores["citation_quality"] == 4.0
    assert scores["factual_accuracy"] == 2.0


def test_criteria_prompt_includes_rubric():
    """criteria= injects criterion names and descriptions into the prompt."""
    prov = _fake_provider(
        '{"reasoning": "...", "criteria": {"accuracy": {"score": 4, "explanation": "ok"}}, "score": 4, "explanation": "ok"}'
    )

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        criteria={"accuracy": "Is the output factually correct?"},
    )
    session.add_action(action)

    score_trace(session, provider=None)

    user_prompt = prov.judge.call_args[0][1]
    assert "accuracy" in user_prompt
    assert "Is the output factually correct?" in user_prompt


def test_reference_injected_into_prompt():
    """reference= is included in the judge prompt."""
    prov = _fake_provider()

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        reference="Expected: Python was created by Guido van Rossum.",
    )
    session.add_action(action)

    score_trace(session, provider=None)

    user_prompt = prov.judge.call_args[0][1]
    assert "Python was created by Guido van Rossum" in user_prompt


def test_criteria_score_explanation_includes_axis_names():
    """With criteria, score_explanation shows each criterion name."""
    response = (
        '{"reasoning": "...", "criteria": {'
        '"citation_quality": {"score": 4, "explanation": "Good citations"},'
        '"factual_accuracy": {"score": 2, "explanation": "Facts off"}'
        '}, "score": 3.0, "explanation": "Mixed"}'
    )
    prov = _fake_provider(response)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        criteria={
            "citation_quality": "Are sources cited?",
            "factual_accuracy": "Are facts correct?",
        },
    )
    session.add_action(action)

    score_trace(session, provider=None)

    explanation = session.actions[0].score_explanation
    assert "citation_quality" in explanation
    assert "factual_accuracy" in explanation


@pytest.mark.asyncio
async def test_criteria_async_single_call():
    """async_score_trace with criteria makes one call and returns per-criterion results."""
    from lattice.judge.scorer import async_score_trace

    response = (
        '{"reasoning": "...", "criteria": {'
        '"clarity": {"score": 3, "explanation": "ok"},'
        '"accuracy": {"score": 5, "explanation": "great"}'
        '}, "score": 4.0, "explanation": "Overall good"}'
    )
    prov = _fake_provider(response)

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        criteria={
            "clarity": "Is the output clear?",
            "accuracy": "Is the output accurate?",
        },
    )
    session.add_action(action)

    await async_score_trace(session, provider=None)

    assert prov.ajudge.call_count == 1
    assert session.actions[0].score == pytest.approx(4.0)
    names = {r.name for r in session.actions[0].judge_results}
    assert names == {"clarity", "accuracy"}


def test_no_global_provider_raises_for_action_without_judge():
    """If no global provider and an action has no judge, score_trace raises."""
    import os
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        session = TraceSession(goal="test")
        session.add_action(_make_action())  # no judge

        with pytest.raises(ValueError, match="No judge provider"):
            score_trace(session)
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_judge_result_fields_populated():
    """JudgeResult has score, explanation, name, and model."""
    prov = _fake_provider('{"score": 3, "explanation": "Acceptable"}')
    prov.model = "gpt-4o"

    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(
        provider=prov,
        criteria={"factual_accuracy": "Are facts correct?"},
    )
    session.add_action(action)

    score_trace(session, provider=MagicMock())

    # With criteria, judge_results is populated from the criteria parse
    # The fallback path fires here since the mock returns simple JSON
    result = session.actions[0].judge_results[0]
    assert isinstance(result, JudgeResult)
    assert result.name == "factual_accuracy"
    assert result.model == "gpt-4o"


def test_judge_stripped_from_serialization():
    """judge field must not appear in to_dict() output (non-serializable)."""
    prov = _fake_provider()
    session = TraceSession(goal="test")
    action = _make_action()
    action.judge = JudgeConfig(provider=prov)
    session.add_action(action)

    d = session.to_dict()
    assert "judge" not in d["actions"][0]
    assert "judge_results" in d["actions"][0]


def test_openai_provider_temperature_and_top_p():
    """temperature and top_p are included in the OpenAI request payload."""
    prov = OpenAIJudgeProvider("key", "gpt-4o", temperature=0.5, top_p=0.9)
    payload = prov._payload("sys", "user")
    assert payload["temperature"] == 0.5
    assert payload["top_p"] == 0.9


def test_openai_provider_top_p_omitted_when_none():
    prov = OpenAIJudgeProvider("key", "gpt-4o", temperature=0.1)
    payload = prov._payload("sys", "user")
    assert "top_p" not in payload


def test_anthropic_provider_temperature_and_top_p():
    """temperature and top_p are included in the Anthropic request payload."""
    prov = AnthropicJudgeProvider("key", temperature=0.3, top_p=0.8)
    payload = prov._payload("sys", "user")
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8


def test_anthropic_provider_top_p_omitted_when_none():
    prov = AnthropicJudgeProvider("key", temperature=0.1)
    payload = prov._payload("sys", "user")
    assert "top_p" not in payload
