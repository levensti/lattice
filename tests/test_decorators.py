import asyncio
import threading
import time

import pytest

from lattice import action, trace_session, JudgeConfig
from lattice.decorators import instrument, trace_action


# ── @action decorator tests ──────────────────────────────────────────────


def test_sync_step_recorded():
    @action(name="greeter", goal="Must greet by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session(goal="test") as session:
        result = greet("Alice")

    assert result == "Hello, Alice!"
    assert len(session.actions) == 1
    s = session.actions[0]
    assert s.name == "greeter"
    assert '"Alice"' in s.input_data
    assert "Hello, Alice!" in s.output_data
    assert s.latency_ms >= 0
    assert s.error is None


def test_async_step_recorded():
    @action(name="async_greeter", goal="Must greet")
    async def greet(name: str) -> str:
        return f"Hi, {name}!"

    async def _run():
        with trace_session(goal="test") as session:
            result = await greet("Bob")
        return result, session

    result, session = asyncio.run(_run())
    assert result == "Hi, Bob!"
    assert len(session.actions) == 1
    assert session.actions[0].name == "async_greeter"


def test_step_with_tags():
    @action(name="search", goal="Return results", tags=["io", "external"])
    def search(query: str) -> list:
        return [{"title": "result"}]

    with trace_session(goal="test") as session:
        search("python")

    assert session.actions[0].tags == ["io", "external"]


def test_step_nested_spans():
    @action(name="inner", goal="inner criteria")
    def inner_fn(x: int) -> int:
        return x * 2

    @action(name="outer", goal="outer criteria")
    def outer_fn(x: int) -> int:
        return inner_fn(x) + 1

    with trace_session(goal="test") as session:
        result = outer_fn(5)

    assert result == 11
    assert len(session.actions) == 2
    inner_action = next(s for s in session.actions if s.name == "inner")
    outer_action = next(s for s in session.actions if s.name == "outer")
    assert inner_action.parent_span_id == outer_action.span_id


def test_step_error_captured():
    @action(name="failer", goal="Should not fail")
    def failing() -> str:
        raise ValueError("something went wrong")

    with trace_session(goal="test") as session:
        with pytest.raises(ValueError, match="something went wrong"):
            failing()

    assert len(session.actions) == 1
    assert session.actions[0].error == "something went wrong"


def test_step_no_session_passthrough():
    @action(name="standalone", goal="criteria")
    def standalone(x: int) -> int:
        return x + 1

    assert standalone(10) == 11


def test_step_ordering():
    @action(name="step_a", goal="a")
    def step_a() -> str:
        return "a"

    @action(name="step_b", goal="b")
    def step_b() -> str:
        return "b"

    @action(name="step_c", goal="c")
    def step_c() -> str:
        return "c"

    with trace_session(goal="test") as session:
        step_a()
        step_b()
        step_c()

    assert [s.name for s in session.actions] == ["step_a", "step_b", "step_c"]
    assert [s.action_index for s in session.actions] == [0, 1, 2]


# ── optional name ─────────────────────────────────────────────────────


def test_step_name_inferred_from_function():
    @action(goal="must work")
    def my_function() -> str:
        return "ok"

    with trace_session(goal="test") as session:
        my_function()

    assert session.actions[0].name == "my_function"


def test_action_bare_decorator_works_without_goal():
    """Bare @action (no parens) should work for tracing without scoring."""
    @action
    def bare() -> str:
        return "bare"

    with trace_session(goal="test") as session:
        assert bare() == "bare"

    assert session.actions[0].name == "bare"


def test_action_empty_parens_works_without_goal():
    """@action() should work for tracing without scoring."""
    @action()
    def empty_parens() -> str:
        return "ok"

    with trace_session(goal="test") as session:
        assert empty_parens() == "ok"

    assert session.actions[0].name == "empty_parens"


def test_action_requires_goal_when_judge_set():
    """@action with judge= but no goal should raise."""
    from unittest.mock import MagicMock
    prov = MagicMock()

    with pytest.raises(TypeError, match="requires a goal"):
        @action(judge=JudgeConfig(system_prompt="Rate it.", provider=prov))
        def judged() -> str:
            return "ok"


def test_step_explicit_name_overrides():
    @action(name="explicit", goal="test")
    def inferred() -> str:
        return "ok"

    with trace_session(goal="test") as session:
        inferred()

    assert session.actions[0].name == "explicit"


# ── self/cls exclusion ────────────────────────────────────────────────


def test_self_excluded_from_input():
    class MyAgent:
        @action(goal="process data")
        def process(self, data: str) -> str:
            return f"processed {data}"

    agent = MyAgent()
    with trace_session(goal="test") as session:
        agent.process("hello")

    assert "self" not in session.actions[0].input_data
    assert "hello" in session.actions[0].input_data


def test_cls_excluded_from_input():
    class MyService:
        @classmethod
        @action(goal="class method")
        def create(cls, name: str) -> str:
            return f"created {name}"

    with trace_session(goal="test") as session:
        MyService.create("test")

    assert "cls" not in session.actions[0].input_data
    assert "test" in session.actions[0].input_data


# ── trace_action context manager ───────────────────────────────────────


def test_trace_step_records():
    with trace_session(goal="test") as session:
        with trace_action("manual_step", goal="do something") as ts:
            result = 2 + 2
            ts.set_output(result)

    assert len(session.actions) == 1
    s = session.actions[0]
    assert s.name == "manual_step"
    assert s.goal == "do something"
    assert "4" in s.output_data
    assert s.error is None


def test_trace_step_captures_error():
    with trace_session(goal="test") as session:
        with pytest.raises(RuntimeError):
            with trace_action("failing_step", goal="should fail"):
                raise RuntimeError("boom")

    assert len(session.actions) == 1
    assert session.actions[0].error == "boom"


def test_trace_step_with_input():
    with trace_session(goal="test") as session:
        with trace_action("search", goal="find results", input_data={"query": "test"}) as ts:
            ts.set_output(["result1"])

    s = session.actions[0]
    assert "test" in s.input_data
    assert "result1" in s.output_data


def test_trace_action_scores_inline():
    """trace_action with judge= should score in background, just like @action."""
    from unittest.mock import MagicMock

    prov = MagicMock()
    prov.judge.return_value = '{"score": 3.5, "explanation": "decent"}'

    with trace_session(goal="test", persist=False) as session:
        with trace_action(
            "external_call",
            goal="get results",
            judge=JudgeConfig(system_prompt="Rate it.", provider=prov),
        ) as ts:
            ts.set_output("some results")

    # Let background thread finish
    time.sleep(0.2)
    assert session.actions[0].score == 3.5


def test_trace_step_nested_under_decorator():
    @action(goal="parent step")
    def parent() -> str:
        with trace_action("child_block", goal="inner work") as ts:
            ts.set_output("inner result")
        return "done"

    with trace_session(goal="test") as session:
        parent()

    assert len(session.actions) == 2
    child = next(s for s in session.actions if s.name == "child_block")
    parent_action = next(s for s in session.actions if s.name == "parent")
    assert child.parent_span_id == parent_action.span_id


# ── instrument() ─────────────────────────────────────────────────────


def test_instrument_wraps_function():
    def original(x: int) -> int:
        return x * 3

    traced = instrument(original, goal="triple the input")

    with trace_session(goal="test") as session:
        result = traced(7)

    assert result == 21
    assert len(session.actions) == 1
    assert session.actions[0].name == "original"
    assert session.actions[0].goal == "triple the input"


def test_instrument_with_custom_name():
    def original() -> str:
        return "hi"

    traced = instrument(original, name="custom", goal="greet")

    with trace_session(goal="test") as session:
        traced()

    assert session.actions[0].name == "custom"


def test_instrument_works_without_goal():
    """instrument() without goal should work for tracing without scoring."""
    def original() -> str:
        return "ok"

    traced = instrument(original)

    with trace_session(goal="test") as session:
        assert traced() == "ok"

    assert session.actions[0].name == "original"


def test_instrument_requires_goal_when_judge_set():
    from unittest.mock import MagicMock
    prov = MagicMock()

    def original() -> str:
        return "ok"

    with pytest.raises(TypeError, match="requires a goal"):
        instrument(original, judge=JudgeConfig(system_prompt="Rate it.", provider=prov))


def test_instrument_does_not_modify_original():
    call_count = 0

    def original() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    traced = instrument(original, goal="return ok")

    original()
    assert call_count == 1

    with trace_session(goal="test") as session:
        traced()

    assert call_count == 2
    assert len(session.actions) == 1


def test_instrument_bound_method():
    class Agent:
        def search(self, query: str) -> list:
            return [f"result for {query}"]

    agent = Agent()
    agent.search = instrument(agent.search, goal="find results")

    with trace_session(goal="test") as session:
        result = agent.search("test")

    assert result == ["result for test"]
    assert session.actions[0].name == "search"
    assert "self" not in session.actions[0].input_data


# ── auto-transitions ─────────────────────────────────────────────────


def test_auto_transitions_from_call_graph():
    @action(goal="orchestrate")
    def parent_fn() -> str:
        child_fn()
        return "done"

    @action(goal="do child work")
    def child_fn() -> str:
        return "child"

    with trace_session(goal="test") as session:
        parent_fn()

    auto_transitions = [t for t in session.transitions if t.auto]
    assert len(auto_transitions) == 1
    assert auto_transitions[0].to_name == "child_fn"


def test_manual_transition_preferred_over_auto():
    from lattice import trace_transition

    @action(goal="route to target")
    def router() -> str:
        trace_transition(to="target", reason="custom reason")
        target()
        return "done"

    @action(goal="handle request")
    def target() -> str:
        return "hi"

    with trace_session(goal="test") as session:
        router()

    transitions = session.transitions
    to_target = [t for t in transitions if t.to_name == "target"]
    manual = [t for t in to_target if not t.auto]
    assert len(manual) == 1
    assert manual[0].reason == "custom reason"


# ── non-blocking inline scoring ─────────────────────────────────────


def test_inline_scoring_does_not_block_caller():
    """@action with judge= should return immediately; scoring runs in background."""
    from unittest.mock import MagicMock

    scoring_started = threading.Event()
    scoring_gate = threading.Event()

    def slow_judge(system, prompt):
        scoring_started.set()
        scoring_gate.wait(timeout=5)
        return '{"score": 4, "explanation": "Good"}'

    prov = MagicMock()
    prov.judge = slow_judge

    @action(
        goal="greet",
        judge=JudgeConfig(system_prompt="Rate it.", provider=prov),
    )
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session(goal="test", persist=False) as session:
        t0 = time.perf_counter()
        result = greet("Alice")
        elapsed_ms = (time.perf_counter() - t0) * 1000

    # The call returned without waiting for the judge
    assert result == "Hello, Alice!"
    assert elapsed_ms < 500  # would be >5s if blocking

    # Score is None immediately (judge hasn't finished)
    assert session.actions[0].score is None

    # Let the judge finish and verify score is written back
    scoring_started.wait(timeout=5)
    scoring_gate.set()
    time.sleep(0.1)  # brief wait for thread to write back
    assert session.actions[0].score == 4.0


def test_inline_scoring_failure_does_not_crash():
    """If the judge raises, the action still completes and score stays None."""
    from unittest.mock import MagicMock

    prov = MagicMock()
    prov.judge.side_effect = RuntimeError("judge down")

    @action(
        goal="greet",
        judge=JudgeConfig(system_prompt="Rate it.", provider=prov),
    )
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session(goal="test", persist=False) as session:
        result = greet("Bob")

    assert result == "Hello, Bob!"
    time.sleep(0.2)  # let the thread finish
    assert session.actions[0].score is None
