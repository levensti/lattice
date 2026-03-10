import asyncio

import pytest

from lattice import step, trace_session
from lattice.decorators import instrument, trace_step


# ── @step decorator tests ──────────────────────────────────────────────


def test_sync_step_recorded():
    @step(name="greeter", goal="Must greet by name")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    with trace_session() as session:
        result = greet("Alice")

    assert result == "Hello, Alice!"
    assert len(session.steps) == 1
    s = session.steps[0]
    assert s.name == "greeter"
    assert '"Alice"' in s.input_data
    assert "Hello, Alice!" in s.output_data
    assert s.latency_ms >= 0
    assert s.error is None


def test_async_step_recorded():
    @step(name="async_greeter", goal="Must greet")
    async def greet(name: str) -> str:
        return f"Hi, {name}!"

    async def _run():
        with trace_session() as session:
            result = await greet("Bob")
        return result, session

    result, session = asyncio.run(_run())
    assert result == "Hi, Bob!"
    assert len(session.steps) == 1
    assert session.steps[0].name == "async_greeter"


def test_step_with_tags():
    @step(name="search", goal="Return results", tags=["io", "external"])
    def search(query: str) -> list:
        return [{"title": "result"}]

    with trace_session() as session:
        search("python")

    assert session.steps[0].tags == ["io", "external"]


def test_step_nested_spans():
    @step(name="inner", goal="inner criteria")
    def inner_fn(x: int) -> int:
        return x * 2

    @step(name="outer", goal="outer criteria")
    def outer_fn(x: int) -> int:
        return inner_fn(x) + 1

    with trace_session() as session:
        result = outer_fn(5)

    assert result == 11
    assert len(session.steps) == 2
    inner_step = next(s for s in session.steps if s.name == "inner")
    outer_step = next(s for s in session.steps if s.name == "outer")
    assert inner_step.parent_span_id == outer_step.span_id


def test_step_error_captured():
    @step(name="failer", goal="Should not fail")
    def failing() -> str:
        raise ValueError("something went wrong")

    with trace_session() as session:
        with pytest.raises(ValueError, match="something went wrong"):
            failing()

    assert len(session.steps) == 1
    assert session.steps[0].error == "something went wrong"


def test_step_no_session_passthrough():
    @step(name="standalone", goal="criteria")
    def standalone(x: int) -> int:
        return x + 1

    assert standalone(10) == 11


def test_step_ordering():
    @step(name="step_a", goal="a")
    def step_a() -> str:
        return "a"

    @step(name="step_b", goal="b")
    def step_b() -> str:
        return "b"

    @step(name="step_c", goal="c")
    def step_c() -> str:
        return "c"

    with trace_session() as session:
        step_a()
        step_b()
        step_c()

    assert [s.name for s in session.steps] == ["step_a", "step_b", "step_c"]
    assert [s.step_index for s in session.steps] == [0, 1, 2]


# ── optional name ─────────────────────────────────────────────────────


def test_step_name_inferred_from_function():
    @step(goal="must work")
    def my_function() -> str:
        return "ok"

    with trace_session() as session:
        my_function()

    assert session.steps[0].name == "my_function"


def test_step_bare_decorator():
    @step
    def bare() -> str:
        return "bare"

    with trace_session() as session:
        result = bare()

    assert result == "bare"
    assert session.steps[0].name == "bare"


def test_step_bare_parens():
    @step()
    def empty_parens() -> str:
        return "ok"

    with trace_session() as session:
        empty_parens()

    assert session.steps[0].name == "empty_parens"


def test_step_positional_name():
    @step("custom_name")
    def original() -> str:
        return "ok"

    with trace_session() as session:
        original()

    assert session.steps[0].name == "custom_name"


def test_step_explicit_name_overrides():
    @step(name="explicit", goal="test")
    def inferred() -> str:
        return "ok"

    with trace_session() as session:
        inferred()

    assert session.steps[0].name == "explicit"


# ── self/cls exclusion ────────────────────────────────────────────────


def test_self_excluded_from_input():
    class MyAgent:
        @step(goal="process data")
        def process(self, data: str) -> str:
            return f"processed {data}"

    agent = MyAgent()
    with trace_session() as session:
        agent.process("hello")

    assert "self" not in session.steps[0].input_data
    assert "hello" in session.steps[0].input_data


def test_cls_excluded_from_input():
    class MyService:
        @classmethod
        @step(goal="class method")
        def create(cls, name: str) -> str:
            return f"created {name}"

    with trace_session() as session:
        MyService.create("test")

    assert "cls" not in session.steps[0].input_data
    assert "test" in session.steps[0].input_data


# ── trace_step context manager ────────────────────────────────────────


def test_trace_step_records():
    with trace_session() as session:
        with trace_step("manual_step", goal="do something") as ts:
            result = 2 + 2
            ts.set_output(result)

    assert len(session.steps) == 1
    s = session.steps[0]
    assert s.name == "manual_step"
    assert s.goal == "do something"
    assert "4" in s.output_data
    assert s.error is None


def test_trace_step_captures_error():
    with trace_session() as session:
        with pytest.raises(RuntimeError):
            with trace_step("failing_step", goal="should fail"):
                raise RuntimeError("boom")

    assert len(session.steps) == 1
    assert session.steps[0].error == "boom"


def test_trace_step_with_input():
    with trace_session() as session:
        with trace_step("search", input_data={"query": "test"}) as ts:
            ts.set_output(["result1"])

    s = session.steps[0]
    assert "test" in s.input_data
    assert "result1" in s.output_data


def test_trace_step_nested_under_decorator():
    @step(goal="parent step")
    def parent() -> str:
        with trace_step("child_block", goal="inner work") as ts:
            ts.set_output("inner result")
        return "done"

    with trace_session() as session:
        parent()

    assert len(session.steps) == 2
    child = next(s for s in session.steps if s.name == "child_block")
    parent_step = next(s for s in session.steps if s.name == "parent")
    assert child.parent_span_id == parent_step.span_id


# ── instrument() ─────────────────────────────────────────────────────


def test_instrument_wraps_function():
    def original(x: int) -> int:
        return x * 3

    traced = instrument(original, goal="triple the input")

    with trace_session() as session:
        result = traced(7)

    assert result == 21
    assert len(session.steps) == 1
    assert session.steps[0].name == "original"
    assert session.steps[0].goal == "triple the input"


def test_instrument_with_custom_name():
    def original() -> str:
        return "hi"

    traced = instrument(original, name="custom", goal="greet")

    with trace_session() as session:
        traced()

    assert session.steps[0].name == "custom"


def test_instrument_does_not_modify_original():
    call_count = 0

    def original() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    traced = instrument(original)

    original()
    assert call_count == 1

    with trace_session() as session:
        traced()

    assert call_count == 2
    assert len(session.steps) == 1


def test_instrument_bound_method():
    class Agent:
        def search(self, query: str) -> list:
            return [f"result for {query}"]

    agent = Agent()
    agent.search = instrument(agent.search, goal="find results")

    with trace_session() as session:
        result = agent.search("test")

    assert result == ["result for test"]
    assert session.steps[0].name == "search"
    assert "self" not in session.steps[0].input_data


# ── auto-transitions ─────────────────────────────────────────────────


def test_auto_transitions_from_call_graph():
    @step(goal="")
    def parent_fn() -> str:
        child_fn()
        return "done"

    @step(goal="")
    def child_fn() -> str:
        return "child"

    with trace_session() as session:
        parent_fn()

    auto_transitions = [t for t in session.transitions if t.auto]
    assert len(auto_transitions) == 1
    assert auto_transitions[0].to_name == "child_fn"


def test_manual_transition_preferred_over_auto():
    from lattice import trace_transition

    @step(goal="")
    def router() -> str:
        trace_transition(to="target", reason="custom reason")
        target()
        return "done"

    @step(goal="")
    def target() -> str:
        return "hi"

    with trace_session() as session:
        router()

    transitions = session.transitions
    to_target = [t for t in transitions if t.to_name == "target"]
    manual = [t for t in to_target if not t.auto]
    assert len(manual) == 1
    assert manual[0].reason == "custom reason"
