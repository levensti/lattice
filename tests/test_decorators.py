import asyncio

import pytest

from agent_trace import step, trace_session


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
